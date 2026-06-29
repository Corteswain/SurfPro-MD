#!/usr/bin/env python3
"""
XGBoost surrogate models trained on Morgan fingerprint features.

Mirrors the architecture of surrogate_pca_models_parallel.py:
  - same Butina clustering (radius=2, 1024 bits, Tc=0.6) for train/test splits
  - same 5 test splits × 5 CV folds per property
  - same Optuna hyperparameter search (30 trials, parallelised across folds)
  - same NN-isolation outlier filter in feature space

Two operating modes:
  raw  (--pca not set): StandardScaler → XGBoost  on raw fp bit vector
  pca  (--pca):         StandardScaler → PCA(95%) → StandardScaler → XGBoost

Stored pkl structure per property:
  results[target]["_morgan_config"]  = {radius, nbits, pca}
  results[target]["_pca_transform"]  = {...}          # PCA mode only
  results[target][test_id] = {
      fold_models, test_idx, feature_schema,
      y_test, y_pred, metrics
  }

Usage:
    python surrogate_morgan.py --radius 2 --nbits 1024
    python surrogate_morgan.py --radius 2 --nbits 1024 --pca
"""

import sys, os, pickle, datetime, argparse, threading
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import xgboost as xgb
import optuna

RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONSTANTS  (identical to existing surrogate models)
# =============================================================================
TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]
N_TEST_SPLITS      = 5
N_FOLDS            = 5
MIN_TEST_FRACTION  = 0.07
MAX_TEST_FRACTION  = 0.11
MAX_REJECT_TRIES   = 5
MIN_VAL_FRACTION   = 0.12
MAX_VAL_FRACTION   = 0.18
RANDOM_STATE       = 3
NN_ISOLATION_THRESHOLD = 50.0

DATA_CSV = ("/proj/berzelius-2026-62/users/x_ribec"
            "/surfactant-surrogates/SurfPro-MD/data/SurfPro-MD.csv")
OUT_DIR  = ("/proj/berzelius-2026-62/users/x_ribec"
            "/surfactant-surrogates/SurfPro-MD/morgan-models")

N_CPUS        = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
N_FOLD_JOBS   = min(N_FOLDS, N_CPUS)
N_OPTUNA_JOBS = max(1, N_CPUS // N_FOLD_JOBS)
XGB_THREADS   = 1

_print_lock = threading.Lock()

# =============================================================================
# ARGUMENT PARSING + LOGGING
# =============================================================================
parser = argparse.ArgumentParser(description="Morgan fingerprint surrogate models")
parser.add_argument("--radius",   type=int, required=True, help="Morgan radius")
parser.add_argument("--nbits",    type=int, required=True, help="Fingerprint size")
parser.add_argument("--pca",      action="store_true",     help="Apply global PCA (95%% var)")
parser.add_argument("--n-trials", type=int, default=30,    help="Optuna trials per fold")
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
_mode     = "pca" if args.pca else "raw"
_ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = os.path.join(OUT_DIR, f"morgan_{_mode}_r{args.radius}_n{args.nbits}_{_ts}.log")
_log_fh   = open(_log_path, "w")

class _Tee:
    def __init__(self, r, f): self._r, self._f = r, f
    def write(self, t): self._r.write(t); self._f.write(t); self._f.flush()
    def flush(self): self._r.flush(); self._f.flush()
    def isatty(self): return False

sys.stdout = _Tee(sys.stdout, _log_fh)
print(f"=== Logging to {_log_path} ===\n")
print(f"radius={args.radius}  nbits={args.nbits}  pca={args.pca}  n_trials={args.n_trials}")
print(f"CPUs={N_CPUS}  fold_jobs={N_FOLD_JOBS}  optuna_jobs={N_OPTUNA_JOBS}")

# =============================================================================
# PIPELINE HELPERS  (identical logic to surrogate_pca_models_parallel.py)
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
            with _print_lock: print("  WARN: could not fill cluster set")
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
        rem = df_t[df_t[colt] != "test"]
        rc  = np.array(rem["cluster"].unique())
        rs  = rem.groupby("cluster").size().to_dict()
        rn  = len(rem)
        for fid in range(N_FOLDS):
            rng_f = np.random.RandomState(RANDOM_STATE + tid * 100 + fid)
            vc    = _sample_test_clusters(rc, rs,
                                          int(rn * MIN_VAL_FRACTION),
                                          int(rn * MAX_VAL_FRACTION), rng_f)
            colf  = f"{target_col}_split_t{tid}_f{fid}"
            df_t[colf] = "unused"
            df_t.loc[df_t["cluster"].isin(set(rc) - set(vc)), colf] = "train"
            df_t.loc[df_t["cluster"].isin(vc),                colf] = "val"
            df_t.loc[df_t[colt] == "test",                    colf] = "test"
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
    with _print_lock:
        print(f"  filter_outliers({target}): {n0} → {len(df)} (-{n0 - len(df)})")
    return df


def filter_nn_isolated(df, feature_cols, threshold=NN_ISOLATION_THRESHOLD):
    X     = df[feature_cols].to_numpy().astype(float)
    Xs    = StandardScaler().fit_transform(X)
    dists = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(Xs).kneighbors(Xs)[0][:, 1]
    keep  = dists <= threshold
    n_rm  = int((~keep).sum())
    with _print_lock:
        print(f"  NN isolation (thr={threshold}): {len(df)} → {len(df) - n_rm} (-{n_rm})")
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
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
            "random_state": 72, "verbosity": 0, "nthread": XGB_THREADS,
        }
        m = xgb.XGBRegressor(**p)
        m.fit(Xtr_sc, ytr)
        return r2_score(yva, m.predict(Xva_sc))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=N_OPTUNA_JOBS)

    best = xgb.XGBRegressor(**study.best_params, random_state=72, nthread=XGB_THREADS)
    best.fit(Xtr_sc, ytr)
    tag = f"fold {fold_id}" if fold_id is not None else ""
    with _print_lock:
        print(f"    {tag} R²(val)={r2_score(yva, best.predict(Xva_sc)):.3f}"
              f"  {study.best_params}")
    return {"model": best.get_booster(), "scaler": sc,
            "features": list(feature_cols), "best_params": study.best_params}


def ensemble_predict(fold_models, X):
    return np.mean([m["model"].inplace_predict(m["scaler"].transform(X))
                    for m in fold_models], axis=0)


def print_summary(results):
    label = (f"Morgan r={args.radius} n={args.nbits}"
             + (" + PCA(95%)" if args.pca else " (raw)"))
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for t in TARGETS:
        if t not in results:
            continue
        runs = {k: v for k, v in results[t].items() if isinstance(k, int)}
        r2s  = [r["metrics"]["r2"]       for r in runs.values()]
        rhos = [r["metrics"]["spearman"] for r in runs.values()]
        print(f"  {t:<22} R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}"
              f"  ρ={np.mean(rhos):.3f}±{np.std(rhos):.3f}")


# =============================================================================
# DATA LOADING + BUTINA CLUSTERING
# (clustering always uses radius=2, 1024-bit fps for split reproducibility)
# =============================================================================
print(f"\nLoading {DATA_CSV} …")
df       = pd.read_csv(DATA_CSV)
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid  = df[df["mol"].notnull()].copy()
print(f"  {len(df_valid)} valid molecules")

print("Butina clustering (radius=2, 1024 bits, Tc-threshold=0.6) …")
fps_clust = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
             for m in df_valid["mol"]]
dists_clust = []
for i in range(1, len(fps_clust)):
    sims = DataStructs.BulkTanimotoSimilarity(fps_clust[i], fps_clust[:i])
    dists_clust.extend([1 - s for s in sims])
clusters_butina = Butina.ClusterData(
    dists_clust, len(fps_clust), distThresh=0.6, isDistData=True)
cid = np.zeros(len(fps_clust), dtype=int)
for i, cl in enumerate(clusters_butina):
    for k in cl:
        cid[k] = i
df_valid["cluster"] = cid
df["cluster"] = np.nan
df.loc[df_valid.index, "cluster"] = cid
print(f"  {len(clusters_butina)} clusters")

# =============================================================================
# MORGAN FINGERPRINTS FOR MODEL FEATURES
# =============================================================================
print(f"\nComputing Morgan fingerprints (radius={args.radius}, nbits={args.nbits}) …")
fp_cols   = [f"fp_{i}" for i in range(args.nbits)]
fp_matrix = np.zeros((len(df_valid), args.nbits), dtype=np.uint8)
for idx, mol in enumerate(df_valid["mol"]):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=args.nbits)
    DataStructs.ConvertToNumpyArray(fp, fp_matrix[idx])

fp_df = pd.DataFrame(fp_matrix.astype(np.float32),
                     index=df_valid.index, columns=fp_cols)
df = df.join(fp_df)
print(f"  Fingerprint matrix: {fp_matrix.shape}  density="
      f"{fp_matrix.mean():.4f}")

# =============================================================================
# TRAINING LOOP
# =============================================================================
morgan_config = {"radius": args.radius, "nbits": args.nbits, "pca": args.pca}
out_name      = (f"models_morgan_{'pca_' if args.pca else ''}"
                 f"r{args.radius}_n{args.nbits}.pkl")
out_path      = os.path.join(OUT_DIR, out_name)

print(f"\n{'#'*60}")
print(f"# Training → {out_name}")
print(f"{'#'*60}")

results = {}

for target in TARGETS:
    print(f"\n{'='*50}\nTarget: {target}")

    df_t = create_cluster_splits(df, target)
    df_t = filter_outliers(df_t, target)

    if args.pca:
        # Fit PCA on all filtered molecules (consistent with existing PCA approach)
        X_fp     = df_t[fp_cols].to_numpy().astype(float)
        raw_sc   = StandardScaler()
        X_fp_sc  = raw_sc.fit_transform(X_fp)
        gpca     = PCA(random_state=RANDOM_STATE)
        X_pc_all = gpca.fit_transform(X_fp_sc)
        n_pcs    = int(np.searchsorted(
                       np.cumsum(gpca.explained_variance_ratio_), 0.95)) + 1
        pc_cols  = [f"pc_{i}" for i in range(n_pcs)]
        df_t     = df_t.copy()
        for i, col in enumerate(pc_cols):
            df_t[col] = X_pc_all[:, i]

        pca_transform = {
            "type":             "global",
            "raw_feature_cols": fp_cols,
            "raw_scaler":       raw_sc,
            "global_pca":       gpca,
            "n_pcs":            n_pcs,
            "feature_schema":   pc_cols,
        }
        feature_schema = pc_cols
        print(f"  PCA: {args.nbits} bits → {n_pcs} PCs (95% variance)")
    else:
        feature_schema = fp_cols
        pca_transform  = None

    df_t = filter_nn_isolated(df_t, feature_schema)

    results[target] = {"_morgan_config": morgan_config}
    if pca_transform is not None:
        results[target]["_pca_transform"] = pca_transform

    for tid in range(N_TEST_SPLITS):
        print(f"\n  ### TEST SPLIT {tid}"
              f"  ({N_FOLD_JOBS} folds × {N_OPTUNA_JOBS} Optuna trials in parallel)")

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

        r2     = r2_score(y_test, y_pred)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        rho, _ = spearmanr(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)
        print(f"  Ensemble R² = {r2:.3f}  ρ = {rho:.3f}")

        results[target][tid] = {
            "fold_models":    fold_models,
            "test_idx":       test_idx,
            "feature_schema": feature_schema,
            "y_test":         y_test.tolist(),
            "y_pred":         y_pred.tolist(),
            "metrics":        {"r2": r2, "rmse": rmse, "spearman": rho, "pearson": pcc},
        }

    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"  Checkpoint → {out_path}")

print_summary(results)
print("\nAll done.")
