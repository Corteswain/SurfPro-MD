#!/usr/bin/env python
# coding: utf-8
"""
surrogate_pca_models_parallel.py

CPU-parallelised version of surrogate_pca_models.py.

Two levels of parallelism:
  1. All N_FOLDS folds within a test split run concurrently (joblib threads).
  2. Optuna trials within each fold run concurrently (study.optimize n_jobs).

XGBoost runs on CPU with nthread=XGB_THREADS.  Total thread budget:
  N_FOLD_JOBS × N_OPTUNA_JOBS × XGB_THREADS  ≈  SLURM_CPUS_PER_TASK

Usage:
  python surrogate_pca_models_parallel.py           # 30 Optuna trials (full)
  python surrogate_pca_models_parallel.py --quick   # 10 trials (fast test)
"""

import re
import sys
import os
import pickle
import warnings
import argparse
import datetime
import threading

import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIG
# =============================================================================
DATA_CSV = "../data/SurfPro-MD.csv"
TARGETS  = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

N_FOLDS            = 5
N_TEST_SPLITS      = 5
MIN_TEST_FRACTION  = 0.07
MAX_TEST_FRACTION  = 0.11
MAX_REJECT_TRIES   = 5
MIN_VAL_FRACTION   = 0.12
MAX_VAL_FRACTION   = 0.18
RANDOM_STATE       = 3
NN_ISOLATION_THRESHOLD = 50.0
COMPRESSION_THRESHOLD  = 0.80

# =============================================================================
# PARALLELISM BUDGET  (derived from SLURM allocation)
# =============================================================================
N_CPUS        = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
N_FOLD_JOBS   = min(N_FOLDS, N_CPUS)           # parallel folds per test split
N_OPTUNA_JOBS = max(1, N_CPUS // N_FOLD_JOBS)  # parallel Optuna trials per fold
XGB_THREADS   = 1                               # XGBoost threads per model

# =============================================================================
# ARGUMENT PARSING + LOGGING
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="10 Optuna trials instead of 30")
parser.add_argument("--global-only", action="store_true",
                    help="Skip two-stage PCA run (models_reduced.pkl) and only train global PCA model")
args     = parser.parse_args()
N_TRIALS = 10 if args.quick else 30

_ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_fh = open(f"surrogate_pca_models_{_ts}.log", "w")
_print_lock = threading.Lock()

class _Tee:
    def __init__(self, r, f): self._r, self._f = r, f
    def write(self, t): self._r.write(t); self._f.write(t); self._f.flush()
    def flush(self): self._r.flush(); self._f.flush()
    def isatty(self): return False

sys.stdout = _Tee(sys.stdout, _log_fh)

print(f"Logging to surrogate_pca_models_{_ts}.log\n")
print(f"Parallelism: {N_CPUS} CPUs → "
      f"{N_FOLD_JOBS} parallel folds × "
      f"{N_OPTUNA_JOBS} parallel Optuna trials × "
      f"{XGB_THREADS} XGBoost thread(s) = "
      f"{N_FOLD_JOBS * N_OPTUNA_JOBS * XGB_THREADS} total threads\n")

# =============================================================================
# FEATURE GROUP RULES
# =============================================================================
GROUP_RULES = {
    "Chi_n":      lambda n: bool(re.fullmatch(r"Chi\d+n", n)),
    "Chi_v":      lambda n: bool(re.fullmatch(r"Chi\d+v", n)),
    "Chi":        lambda n: bool(re.fullmatch(r"Chi\d+",  n)),
    "Kappa":      lambda n: (bool(re.fullmatch(r"Kappa\d+",  n)) or
                             bool(re.fullmatch(r"KappaM\d+", n)) or
                             n in ("Phi", "HallKierAlpha")),
    "PEOE_VSA":   lambda n: bool(re.fullmatch(r"PEOE_VSA\d+",  n)),
    "SMR_VSA":    lambda n: bool(re.fullmatch(r"SMR_VSA\d+",   n)),
    "SlogP_VSA":  lambda n: bool(re.fullmatch(r"SlogP_VSA\d+", n)),
    "EState_VSA": lambda n: bool(re.fullmatch(r"EState_VSA\d+",n)),
    "VSA_EState": lambda n: bool(re.fullmatch(r"VSA_EState\d+",n)),
    "MQN":        lambda n: bool(re.fullmatch(r"MQN\d+",       n)),
    "BCUT2D":     lambda n: n.startswith("BCUT2D_"),
    "MolWt":      lambda n: n in ("MolWt", "HeavyAtomMolWt", "ExactMolWt"),
    "RingCounts": lambda n: n in (
                      "RingCount", "NumAromaticRings", "NumAliphaticRings",
                      "NumSaturatedRings", "NumAromaticHeterocycles",
                      "NumAromaticCarbocycles", "NumSaturatedHeterocycles",
                      "NumSaturatedCarbocycles", "NumAliphaticHeterocycles",
                      "NumAliphaticCarbocycles"),
}

def assign_group(col):
    bare = col.replace("rdkit-", "")
    for label, rule in GROUP_RULES.items():
        if rule(bare): return label
    return None

# =============================================================================
# TRAINING HELPERS
# =============================================================================
def _sample_test_clusters(clusters, sizes, mn, mx, rng):
    available, selected, n, streak = set(clusters), set(), 0, 0
    while available:
        c = rng.choice(sorted(available)); available.remove(c)
        new_n = n + sizes[c]
        if new_n < mn:
            selected.add(c); n = new_n; streak = 0; continue
        if mn <= new_n <= mx:
            selected.add(c); break
        streak += 1
        if streak >= MAX_REJECT_TRIES:
            with _print_lock: print("  WARN: could not fill set")
            break
    return selected


def create_cluster_splits(df_input, target_col):
    df_t  = df_input.dropna(subset=[target_col, "cluster"]).copy()
    df_t["cluster"] = df_t["cluster"].astype(int)
    total = len(df_t)
    cs    = df_t.groupby("cluster").size().to_dict()
    clust = np.array(list(cs.keys()))

    for tid in range(N_TEST_SPLITS):
        rng  = np.random.RandomState(RANDOM_STATE + tid)
        tc   = _sample_test_clusters(clust, cs,
                                     int(total * MIN_TEST_FRACTION),
                                     int(total * MAX_TEST_FRACTION), rng)
        colt = f"{target_col}_test_split_{tid}"
        df_t[colt] = "train_val"
        df_t.loc[df_t["cluster"].isin(tc), colt] = "test"

        rem  = df_t[df_t[colt] != "test"]
        rc   = np.array(rem["cluster"].unique())
        rs   = rem.groupby("cluster").size().to_dict()
        rn   = len(rem)
        for fid in range(N_FOLDS):
            rng_f = np.random.RandomState(RANDOM_STATE + tid * 100 + fid)
            vc    = _sample_test_clusters(rc, rs,
                                          int(rn * MIN_VAL_FRACTION),
                                          int(rn * MAX_VAL_FRACTION), rng_f)
            colf  = f"{target_col}_split_t{tid}_f{fid}"
            df_t[colf] = "unused"
            df_t.loc[df_t["cluster"].isin(set(rc) - set(vc)), colf] = "train"
            df_t.loc[df_t["cluster"].isin(vc),  colf] = "val"
            df_t.loc[df_t[colt] == "test",      colf] = "test"
    return df_t


def filter_outliers(df, target):
    n0 = len(df)
    if target == "surface_tension_avg":
        df = df[df["surface_tension_avg"].notnull() & (df["surface_tension_avg"] >= 250)]
    elif target == "viscosity":
        df = df[df["viscosity"].notnull() & (df["viscosity"] <= 0.003)]
    elif target == "AW_ST_CMC":
        df = df[df["AW_ST_CMC"].notnull() & (df["AW_ST_CMC"] <= 52)]
    elif target == "Gamma_max":
        df = df[df["Gamma_max"].notnull() & (df["Gamma_max"] <= 6)]
    elif target == "Area_min":
        df = df[df["Area_min"].notnull() & (df["Area_min"] <= 4.2)]
    elif target == "pC20":
        df = df[df["pC20"].notnull() & (df["pC20"] >= 1.8)]
    elif target == "D_MOL":
        df = df[df["D_MOL"].notnull() & (df["D_MOL"] <= 0.8)]
    n1 = len(df)
    with _print_lock:
        print(f"  filter_outliers: {n0} → {n1} (-{n0-n1})")
    return df


def filter_nn_isolated(df, feature_cols, threshold=NN_ISOLATION_THRESHOLD):
    X = df[feature_cols].to_numpy().astype(float)
    means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = means[j]
    Xs    = StandardScaler().fit_transform(X)
    dists = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(Xs).kneighbors(Xs)[0][:, 1]
    keep  = dists <= threshold
    n_rm  = int((~keep).sum())
    with _print_lock:
        print(f"  NN isolation filter: {len(df)} → {len(df)-n_rm} (-{n_rm})")
    return df.loc[df.index[keep]]


def optimise(df, feature_cols, target_col, split, fold_id=None):
    Xtr = df.loc[split == "train", feature_cols].reset_index(drop=True)
    Xva = df.loc[split == "val",   feature_cols].reset_index(drop=True)
    ytr = df.loc[split == "train", target_col].reset_index(drop=True)
    yva = df.loc[split == "val",   target_col].reset_index(drop=True)

    sc     = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xva_sc = sc.transform(Xva)

    def objective(trial):
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "max_depth":     trial.suggest_int("max_depth", 3, 10),
             "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
             "random_state": 72, "verbosity": 0,
             "nthread": XGB_THREADS}
        m = xgb.XGBRegressor(**p); m.fit(Xtr_sc, ytr)
        return r2_score(yva, m.predict(Xva_sc))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_OPTUNA_JOBS)

    best = xgb.XGBRegressor(**study.best_params, random_state=72, nthread=XGB_THREADS)
    best.fit(Xtr_sc, ytr)
    tag = f"fold {fold_id}" if fold_id is not None else ""
    with _print_lock:
        print(f"    {tag} R²(val)={r2_score(yva, best.predict(Xva_sc)):.3f}  "
              f"{study.best_params}")
    return {"model": best.get_booster(), "scaler": sc,
            "features": feature_cols, "best_params": study.best_params}


def ensemble_predict(fold_models, X):
    return np.mean([m["model"].inplace_predict(m["scaler"].transform(X))
                    for m in fold_models], axis=0)


def print_summary(results, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for t in TARGETS:
        if t not in results: continue
        runs = {k: v for k, v in results[t].items() if isinstance(k, int)}
        r2s  = [r["metrics"]["r2"]       for r in runs.values()]
        rhos = [r["metrics"]["spearman"] for r in runs.values()]
        print(f"  {t:<22} R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}  "
              f"ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}")


# =============================================================================
# PCA PREPROCESSING
# =============================================================================
def _impute_and_scale(df_t, rdkit_cols):
    X    = df_t[rdkit_cols].to_numpy().astype(float)
    means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = means[j]
    sc = StandardScaler()
    return sc.fit_transform(X), sc


def build_two_stage_features(df_t, rdkit_cols):
    X_sc, raw_sc = _impute_and_scale(df_t, rdkit_cols)

    groups, ungrouped_idx = {}, []
    for ci, col in enumerate(rdkit_cols):
        g = assign_group(col)
        if g: groups.setdefault(g, []).append(ci)
        else: ungrouped_idx.append(ci)

    kept_cols, pooled_idx = [], list(ungrouped_idx)
    group_pcas = {}

    for label, idxs in groups.items():
        pca_g    = PCA(n_components=1, random_state=RANDOM_STATE)
        pc1      = pca_g.fit_transform(X_sc[:, idxs]).ravel()
        var_expl = float(pca_g.explained_variance_ratio_[0])
        if var_expl >= COMPRESSION_THRESHOLD:
            col_name = f"grp_{label}"
            df_t = df_t.copy(); df_t[col_name] = pc1
            kept_cols.append(col_name)
            group_pcas[label] = {"pca": pca_g, "col_indices": idxs}
        else:
            pooled_idx.extend(idxs)

    pool_sc    = StandardScaler()
    X_pool_sc  = pool_sc.fit_transform(X_sc[:, pooled_idx])
    pool_pca   = PCA(random_state=RANDOM_STATE)
    X_pool_pca = pool_pca.fit_transform(X_pool_sc)
    n_pool     = int(np.searchsorted(np.cumsum(pool_pca.explained_variance_ratio_), 0.95)) + 1

    pool_cols = [f"pool_{i}" for i in range(n_pool)]
    df_t = df_t.copy()
    for i, col in enumerate(pool_cols):
        df_t[col] = X_pool_pca[:, i]

    feature_schema = kept_cols + pool_cols
    pca_transform  = {
        "type":             "two_stage",
        "raw_feature_cols": rdkit_cols,
        "raw_scaler":       raw_sc,
        "group_pcas":       group_pcas,
        "pool_col_indices": pooled_idx,
        "pool_scaler":      pool_sc,
        "pool_pca":         pool_pca,
        "n_pool_pcs":       n_pool,
        "feature_schema":   feature_schema,
    }
    print(f"  Two-stage PCA: {len(rdkit_cols)} raw → "
          f"{len(kept_cols)} group PC1s + {n_pool} pool PCs = {len(feature_schema)} features")
    return df_t, feature_schema, pca_transform


def build_global_pca_features(df_t, rdkit_cols):
    X_sc, raw_sc = _impute_and_scale(df_t, rdkit_cols)
    global_pca   = PCA(random_state=RANDOM_STATE)
    X_pca        = global_pca.fit_transform(X_sc)
    n_pcs        = int(np.searchsorted(np.cumsum(global_pca.explained_variance_ratio_), 0.95)) + 1

    pc_cols = [f"pc_{i}" for i in range(n_pcs)]
    df_t = df_t.copy()
    for i, col in enumerate(pc_cols):
        df_t[col] = X_pca[:, i]

    pca_transform = {
        "type":             "global",
        "raw_feature_cols": rdkit_cols,
        "raw_scaler":       raw_sc,
        "global_pca":       global_pca,
        "n_pcs":            n_pcs,
        "feature_schema":   pc_cols,
    }
    print(f"  Global PCA: {len(rdkit_cols)} raw → {n_pcs} PCs (95% var)")
    return df_t, pc_cols, pca_transform


# =============================================================================
# TRAINING LOOP  — folds parallelised within each test split
# =============================================================================
def run_training(df_base, rdkit_cols, build_features_fn, out_file, label):
    print(f"\n{'#'*60}\n# {label}\n# Output: {out_file}\n{'#'*60}")
    results = {}

    for target in TARGETS:
        print(f"\nWorking on {target}")

        df_t = create_cluster_splits(df_base, target)
        df_t = filter_outliers(df_t, target)
        df_t, feature_schema, pca_transform = build_features_fn(df_t, rdkit_cols)
        df_t = filter_nn_isolated(df_t, feature_schema)

        results[target] = {"_pca_transform": pca_transform}

        for tid in range(N_TEST_SPLITS):
            print(f"\n  ### TEST SPLIT {tid}  "
                  f"(running {N_FOLD_JOBS} folds × {N_OPTUNA_JOBS} Optuna trials in parallel)")

            # All N_FOLDS folds run concurrently via threads.
            # XGBoost releases the GIL so threading gives real CPU parallelism.
            fold_models = Parallel(n_jobs=N_FOLD_JOBS, prefer="threads")(
                delayed(optimise)(
                    df_t, feature_schema, target,
                    df_t[f"{target}_split_t{tid}_f{fid}"],
                    fold_id=fid,
                )
                for fid in range(N_FOLDS)
            )

            test_idx = df_t.index[df_t[f"{target}_split_t{tid}_f0"] == "test"].to_numpy()
            X_test   = df_t.loc[test_idx, feature_schema].to_numpy()
            y_test   = df_t.loc[test_idx, target].to_numpy()
            y_pred   = ensemble_predict(fold_models, X_test)

            r2   = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            rho, _ = spearmanr(y_test, y_pred)
            pcc, _ = pearsonr(y_test, y_pred)
            print(f"  Ensemble R² = {r2:.3f}")

            results[target][tid] = {
                "fold_models":    fold_models,
                "test_idx":       test_idx,
                "feature_schema": feature_schema,
                "y_test":         y_test.tolist(),
                "y_pred":         y_pred.tolist(),
                "metrics": {"r2": r2, "rmse": rmse, "spearman": rho, "pearson": pcc},
            }

        with open(out_file, "wb") as fh:
            pickle.dump(results, fh)
        print(f"  Checkpoint saved → {out_file}")

    print_summary(results, label)
    return results


# =============================================================================
# 1. LOAD DATA + BUTINA CLUSTERING
# =============================================================================
print("Loading data ...")
df        = pd.read_csv(DATA_CSV)
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid  = df[df["mol"].notnull()].copy()
print(f"  {len(df_valid)} valid molecules")

fps   = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in df_valid["mol"]]
dists = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - s for s in sims])
clusters_butina = Butina.ClusterData(dists, len(fps), distThresh=0.6, isDistData=True)
cid = np.zeros(len(fps), dtype=int)
for i, cl in enumerate(clusters_butina):
    for k in cl: cid[k] = i
df_valid["cluster"] = cid
df["cluster"] = np.nan
df.loc[df_valid.index, "cluster"] = cid
print(f"  {len(clusters_butina)} Butina clusters")

rdkit_cols = sorted([c for c in df_valid.columns if c.startswith("rdkit-")])
print(f"  {len(rdkit_cols)} RDKit features")

# =============================================================================
# 2. TRAIN  models_reduced.pkl  (two-stage PCA, per-target)
# =============================================================================
if not args.global_only:
    run_training(
        df_base           = df,
        rdkit_cols        = rdkit_cols,
        build_features_fn = build_two_stage_features,
        out_file          = "models_reduced.pkl",
        label             = "Two-stage PCA (group PC1s + pool 95%) — per-target fit",
    )
else:
    print("\nSkipping two-stage PCA run (--global-only flag set).")

# =============================================================================
# 3. TRAIN  models_pca.pkl  (global PCA, per-target)
# =============================================================================
run_training(
    df_base           = df,
    rdkit_cols        = rdkit_cols,
    build_features_fn = build_global_pca_features,
    out_file          = "models_pca.pkl",
    label             = "Global PCA (95% variance) — per-target fit",
)

print("\nAll done.")
