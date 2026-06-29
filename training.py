#!/usr/bin/env python3
"""
training.py — Unified XGBoost surrogate model training for SurfPro-MD.

Covers every feature / PCA combination used in the ablation study.
All training details (splits, Optuna search space, outlier filters, NN
isolation, parallelism scheme) are identical to the original scripts.

Reproducing existing model files
---------------------------------
  # models.pkl  (raw RDKit, 217 descriptors)
  python training.py --features rdkit

  # models_pca.pkl  (RDKit + global PCA, 95 % variance)
  python training.py --features rdkit --pca

  # models_reduced.pkl  (RDKit + two-stage PCA, 95 % variance)
  python training.py --features rdkit --pca --pca-type two-stage

  # models_morgan_r2_n1024.pkl  (Morgan fp, radius 2, 1024 bits)
  python training.py --features morgan --radius 2 --nbits 1024

  # models_morgan_pca_r2_n1024.pkl  (same + global PCA)
  python training.py --features morgan --radius 2 --nbits 1024 --pca

Flags
-----
  --features    {rdkit,morgan}        Feature representation      [rdkit]
  --pca                               Apply PCA preprocessing     [off]
  --pca-variance FLOAT                Cumulative variance kept    [0.95]
  --pca-type    {global,two-stage}    PCA architecture            [global]
  --radius      INT                   Morgan radius               [2]
  --nbits       INT                   Morgan bit-vector length    [1024]
  --n-trials    INT                   Optuna trials per fold      [30]
  --out-dir     PATH                  Output directory            [cwd]
  --output      PATH                  Override output pkl path    [auto]
  --data        PATH                  SurfPro-MD.csv location     [see DEFAULT_DATA]
"""

import sys, os, re, pickle, datetime, argparse, threading
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors
import xgboost as xgb
import optuna

RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONSTANTS  — match original scripts exactly
# =============================================================================
DEFAULT_DATA = ("/proj/berzelius-2026-62/users/x_ribec"
                "/surfactant-surrogates/SurfPro-MD/data/SurfPro-MD.csv")

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

# Two-stage PCA: a feature group is summarised by its PC1 only when that
# component explains at least this fraction of within-group variance.
# Groups below the threshold have their members sent to the residual pool.
COMPRESSION_THRESHOLD = 0.80

# RDKit descriptor group rules (used for two-stage PCA only)
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

# Thread budget (resolved after arg parsing)
N_CPUS = N_FOLD_JOBS = N_OPTUNA_JOBS = XGB_THREADS = None
_print_lock = threading.Lock()

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features",     choices=["rdkit", "morgan"], default="rdkit",
                   help="Feature representation (default: rdkit)")
    p.add_argument("--pca",          action="store_true",
                   help="Apply PCA preprocessing")
    p.add_argument("--pca-variance", type=float, default=0.95, metavar="FLOAT",
                   help="Cumulative variance threshold for PCA (default: 0.95)")
    p.add_argument("--pca-type",     choices=["global", "two-stage"], default="global",
                   help="PCA architecture; two-stage only valid with --features rdkit "
                        "(default: global)")
    p.add_argument("--radius",       type=int, default=2, metavar="INT",
                   help="Morgan fingerprint radius (default: 2)")
    p.add_argument("--nbits",        type=int, default=1024, metavar="INT",
                   help="Morgan fingerprint bit-vector length (default: 1024)")
    p.add_argument("--n-trials",     type=int, default=30, metavar="INT",
                   help="Optuna trials per fold (default: 30)")
    p.add_argument("--out-dir",      type=str, default=".", metavar="PATH",
                   help="Output directory (default: current working directory)")
    p.add_argument("--output",       type=str, default=None, metavar="PATH",
                   help="Override the auto-generated output pkl path")
    p.add_argument("--data",         type=str, default=DEFAULT_DATA, metavar="PATH",
                   help=f"Path to SurfPro-MD.csv (default: {DEFAULT_DATA})")
    return p


def auto_output_name(args):
    parts = [args.features]
    if args.features == "morgan":
        parts += [f"r{args.radius}", f"n{args.nbits}"]
    if args.pca:
        pca_tag = "pca2s" if args.pca_type == "two-stage" else "pca"
        pct = int(round(args.pca_variance * 100))
        parts.append(f"{pca_tag}{pct}")
    return "models_" + "_".join(parts) + ".pkl"


# =============================================================================
# PIPELINE HELPERS
# =============================================================================
def _sample_clusters(clusters, sizes, mn, mx, rng):
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
        rng = np.random.RandomState(RANDOM_STATE + tid)
        tc  = _sample_clusters(clust, cs,
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
            vc    = _sample_clusters(rc, rs,
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
    X  = df[feature_cols].to_numpy().astype(float)
    # Impute NaN with column means before scaling (RDKit descriptors may have NaN)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_means[j]
    Xs    = StandardScaler().fit_transform(X)
    dists = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(Xs).kneighbors(Xs)[0][:, 1]
    keep  = dists <= threshold
    n_rm  = int((~keep).sum())
    with _print_lock:
        print(f"  NN isolation (thr={threshold}): {len(df)} → {len(df) - n_rm} (-{n_rm})")
    return df.loc[df.index[keep]]


def optimise(df, feature_cols, target_col, split, n_trials, fold_id=None):
    Xtr = df.loc[split == "train", feature_cols].reset_index(drop=True)
    Xva = df.loc[split == "val",   feature_cols].reset_index(drop=True)
    ytr = df.loc[split == "train", target_col].reset_index(drop=True)
    yva = df.loc[split == "val",   target_col].reset_index(drop=True)

    sc     = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xva_sc = sc.transform(Xva)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
            "random_state": 72, "verbosity": 0, "nthread": XGB_THREADS,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr_sc, ytr)
        return r2_score(yva, m.predict(Xva_sc))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=N_OPTUNA_JOBS)

    best = xgb.XGBRegressor(**study.best_params, random_state=72, nthread=XGB_THREADS)
    best.fit(Xtr_sc, ytr)
    tag = f"fold {fold_id}" if fold_id is not None else ""
    with _print_lock:
        print(f"    {tag}  R²(val)={r2_score(yva, best.predict(Xva_sc)):.3f}"
              f"  {study.best_params}")
    return {"model": best.get_booster(), "scaler": sc,
            "features": list(feature_cols), "best_params": study.best_params}


def ensemble_predict(fold_models, X):
    return np.mean(
        [m["model"].inplace_predict(m["scaler"].transform(X)) for m in fold_models],
        axis=0,
    )


# =============================================================================
# FEATURE BUILDING
# =============================================================================
def build_rdkit_features(df_valid):
    """Return (df_with_features, rdkit_col_names).
    If rdkit-* columns are already in df, reuse them; otherwise compute."""
    rdkit_cols = sorted([c for c in df_valid.columns if c.startswith("rdkit-")])
    if rdkit_cols:
        print(f"  Using {len(rdkit_cols)} pre-computed RDKit columns from CSV.")
        return df_valid, rdkit_cols

    print("  Computing RDKit descriptors …")
    desc_names = [n for n, _ in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rows = []
    for mol in df_valid["mol"]:
        try:
            vals = calc.CalcDescriptors(mol)
        except Exception:
            vals = [np.nan] * len(desc_names)
        rows.append(vals)
    rdkit_df   = pd.DataFrame(rows, index=df_valid.index,
                               columns=[f"rdkit-{n}" for n in desc_names])
    rdkit_cols = sorted(rdkit_df.columns.tolist())
    df_out = df_valid.join(rdkit_df)
    print(f"  Computed {len(rdkit_cols)} RDKit descriptors.")
    return df_out, rdkit_cols


def build_morgan_features(df_valid, radius, nbits):
    """Return (df_with_features, fp_col_names)."""
    print(f"  Computing Morgan fingerprints (radius={radius}, nbits={nbits}) …")
    fp_cols   = [f"fp_{i}" for i in range(nbits)]
    fp_matrix = np.zeros((len(df_valid), nbits), dtype=np.float32)
    for idx, mol in enumerate(df_valid["mol"]):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        DataStructs.ConvertToNumpyArray(fp, fp_matrix[idx])
    fp_df  = pd.DataFrame(fp_matrix, index=df_valid.index, columns=fp_cols)
    df_out = df_valid.join(fp_df)
    density = float(fp_matrix.mean())
    print(f"  Fingerprint matrix: {fp_matrix.shape}  density={density:.4f}")
    return df_out, fp_cols


# =============================================================================
# PCA TRANSFORMS  (fitted per target on the filtered molecule set)
# =============================================================================
def _impute_and_scale(X, existing_scaler=None):
    """Impute NaN with column means, then StandardScale.
    Returns (X_scaled, fitted_scaler)."""
    X = X.copy().astype(float)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_means[j]
    if existing_scaler is None:
        sc = StandardScaler()
        return sc.fit_transform(X), sc
    return existing_scaler.transform(X), existing_scaler


def apply_global_pca(df_t, raw_cols, pca_variance):
    """Fit global PCA on all filtered molecules; add pc_* columns to df_t."""
    X_sc, raw_sc = _impute_and_scale(df_t[raw_cols].to_numpy())
    gpca         = PCA(random_state=RANDOM_STATE)
    X_pca        = gpca.fit_transform(X_sc)
    n_pcs        = int(np.searchsorted(
                       np.cumsum(gpca.explained_variance_ratio_), pca_variance)) + 1
    pc_cols      = [f"pc_{i}" for i in range(n_pcs)]
    df_t         = df_t.copy()
    for i, col in enumerate(pc_cols):
        df_t[col] = X_pca[:, i]
    transform = {
        "type":             "global",
        "raw_feature_cols": list(raw_cols),
        "raw_scaler":       raw_sc,
        "global_pca":       gpca,
        "n_pcs":            n_pcs,
        "feature_schema":   pc_cols,
    }
    print(f"  Global PCA: {len(raw_cols)} features → {n_pcs} PCs "
          f"({pca_variance*100:.0f}% var)")
    return df_t, pc_cols, transform


def apply_two_stage_pca(df_t, rdkit_cols, pca_variance):
    """Two-stage PCA (RDKit only): compress homogeneous feature groups to
    their PC1 when that component clears COMPRESSION_THRESHOLD; the rest
    go to a shared residual pool that is further compressed by PCA."""
    X_sc, raw_sc = _impute_and_scale(df_t[rdkit_cols].to_numpy())

    def _bare(col):
        return col.replace("rdkit-", "")

    groups, ungrouped_idx = {}, []
    for ci, col in enumerate(rdkit_cols):
        bare = _bare(col)
        matched = False
        for label, rule in GROUP_RULES.items():
            if rule(bare):
                groups.setdefault(label, []).append(ci)
                matched = True
                break
        if not matched:
            ungrouped_idx.append(ci)

    kept_cols, pooled_idx = [], list(ungrouped_idx)
    group_pcas = {}
    df_t = df_t.copy()

    for label, idxs in groups.items():
        pca_g    = PCA(n_components=1, random_state=RANDOM_STATE)
        pc1      = pca_g.fit_transform(X_sc[:, idxs]).ravel()
        var_expl = float(pca_g.explained_variance_ratio_[0])
        if var_expl >= COMPRESSION_THRESHOLD:
            col_name = f"grp_{label}"
            df_t[col_name] = pc1
            kept_cols.append(col_name)
            group_pcas[label] = {"pca": pca_g, "col_indices": idxs}
        else:
            pooled_idx.extend(idxs)

    pool_sc      = StandardScaler()
    X_pool_sc    = pool_sc.fit_transform(X_sc[:, pooled_idx])
    pool_pca     = PCA(random_state=RANDOM_STATE)
    X_pool_pca   = pool_pca.fit_transform(X_pool_sc)
    n_pool       = int(np.searchsorted(
                       np.cumsum(pool_pca.explained_variance_ratio_), pca_variance)) + 1
    pool_cols    = [f"pool_{i}" for i in range(n_pool)]
    for i, col in enumerate(pool_cols):
        df_t[col] = X_pool_pca[:, i]

    feature_schema = kept_cols + pool_cols
    transform = {
        "type":             "two_stage",
        "raw_feature_cols": list(rdkit_cols),
        "raw_scaler":       raw_sc,
        "group_pcas":       group_pcas,
        "pool_col_indices": pooled_idx,
        "pool_scaler":      pool_sc,
        "pool_pca":         pool_pca,
        "n_pool_pcs":       n_pool,
        "feature_schema":   feature_schema,
    }
    print(f"  Two-stage PCA: {len(rdkit_cols)} raw → "
          f"{len(kept_cols)} group PC1s + {n_pool} pool PCs "
          f"= {len(feature_schema)} features")
    return df_t, feature_schema, transform


# =============================================================================
# MAIN
# =============================================================================
def main():
    global N_CPUS, N_FOLD_JOBS, N_OPTUNA_JOBS, XGB_THREADS

    args = build_parser().parse_args()

    # Validate argument combinations
    if args.pca_type == "two-stage" and args.features == "morgan":
        build_parser().error(
            "--pca-type two-stage is only valid with --features rdkit")
    if not args.pca and args.pca_type != "global":
        build_parser().error("--pca-type has no effect without --pca")

    # Thread budget
    N_CPUS        = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    N_FOLD_JOBS   = min(N_FOLDS, N_CPUS)
    N_OPTUNA_JOBS = max(1, N_CPUS // N_FOLD_JOBS)
    XGB_THREADS   = 1

    # Output path
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = (args.output if args.output
                else os.path.join(args.out_dir, auto_output_name(args)))

    # Logging
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_stem = os.path.splitext(os.path.basename(out_path))[0]
    log_path = os.path.join(args.out_dir, f"{log_stem}_{ts}.log")
    log_fh   = open(log_path, "w")

    class _Tee:
        def __init__(self, r, f): self._r, self._f = r, f
        def write(self, t): self._r.write(t); self._f.write(t); self._f.flush()
        def flush(self): self._r.flush(); self._f.flush()
        def isatty(self): return False

    sys.stdout = _Tee(sys.stdout, log_fh)

    print(f"=== {os.path.basename(out_path)} ===")
    print(f"features={args.features}  pca={args.pca}  "
          f"pca_variance={args.pca_variance}  pca_type={args.pca_type}")
    if args.features == "morgan":
        print(f"radius={args.radius}  nbits={args.nbits}")
    print(f"n_trials={args.n_trials}  out={out_path}")
    print(f"CPUs={N_CPUS}  fold_jobs={N_FOLD_JOBS}  "
          f"optuna_jobs={N_OPTUNA_JOBS}  xgb_threads={XGB_THREADS}")
    print(f"Logging → {log_path}\n")

    # ── Load data + Butina clustering ────────────────────────────────────────
    print(f"Loading {args.data} …")
    df        = pd.read_csv(args.data)
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
    clusters_b = Butina.ClusterData(
        dists_clust, len(fps_clust), distThresh=0.6, isDistData=True)
    cid = np.zeros(len(fps_clust), dtype=int)
    for i, cl in enumerate(clusters_b):
        for k in cl:
            cid[k] = i
    df_valid["cluster"] = cid
    df["cluster"] = np.nan
    df.loc[df_valid.index, "cluster"] = cid
    print(f"  {len(clusters_b)} clusters\n")

    # ── Build raw feature columns ────────────────────────────────────────────
    if args.features == "rdkit":
        df, raw_cols = build_rdkit_features(df)
    else:
        df, raw_cols = build_morgan_features(df_valid, args.radius, args.nbits)
        # join back onto full df so indices stay consistent
        df = df.join(df[raw_cols]) if raw_cols[0] in df.columns else df

    # ── Training loop ────────────────────────────────────────────────────────
    config = {
        "features":     args.features,
        "pca":          args.pca,
        "pca_variance": args.pca_variance,
        "pca_type":     args.pca_type if args.pca else None,
        "radius":       args.radius if args.features == "morgan" else None,
        "nbits":        args.nbits  if args.features == "morgan" else None,
        "n_trials":     args.n_trials,
    }
    results = {"_config": config}

    for target in TARGETS:
        print(f"\n{'='*50}\nTarget: {target}")

        df_t = create_cluster_splits(df, target)
        df_t = filter_outliers(df_t, target)

        # Apply PCA if requested (fitted on all filtered molecules)
        if args.pca:
            if args.pca_type == "two-stage":
                df_t, feature_schema, pca_transform = apply_two_stage_pca(
                    df_t, raw_cols, args.pca_variance)
            else:
                df_t, feature_schema, pca_transform = apply_global_pca(
                    df_t, raw_cols, args.pca_variance)
        else:
            feature_schema = raw_cols
            pca_transform  = None

        df_t = filter_nn_isolated(df_t, feature_schema)

        results[target] = {}
        if pca_transform is not None:
            results[target]["_pca_transform"] = pca_transform

        for tid in range(N_TEST_SPLITS):
            print(f"\n  ### TEST SPLIT {tid}  "
                  f"({N_FOLD_JOBS} folds × {N_OPTUNA_JOBS} Optuna in parallel)")

            fold_models = Parallel(n_jobs=N_FOLD_JOBS, prefer="threads")(
                delayed(optimise)(
                    df_t, feature_schema, target,
                    df_t[f"{target}_split_t{tid}_f{fid}"],
                    n_trials=args.n_trials,
                    fold_id=fid,
                )
                for fid in range(N_FOLDS)
            )

            test_idx = df_t.index[
                df_t[f"{target}_split_t{tid}_f0"] == "test"].to_numpy()
            X_test   = df_t.loc[test_idx, feature_schema].to_numpy()
            y_test   = df_t.loc[test_idx, target].to_numpy()
            y_pred   = ensemble_predict(fold_models, X_test)

            r2     = r2_score(y_test, y_pred)
            rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            rho, _ = spearmanr(y_test, y_pred)
            pcc, _ = pearsonr(y_test, y_pred)
            print(f"  Ensemble R²={r2:.3f}  ρ={rho:.3f}")

            results[target][tid] = {
                "fold_models":    fold_models,
                "test_idx":       test_idx,
                "feature_schema": list(feature_schema),
                "y_test":         y_test.tolist(),
                "y_pred":         y_pred.tolist(),
                "metrics":        {
                    "r2": r2, "rmse": rmse, "spearman": rho, "pearson": pcc},
            }

        with open(out_path, "wb") as fh:
            pickle.dump(results, fh)
        print(f"  Checkpoint → {out_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY: {os.path.basename(out_path)}")
    print(f"{'='*60}")
    for t in TARGETS:
        if t not in results:
            continue
        runs = {k: v for k, v in results[t].items() if isinstance(k, int)}
        r2s  = [r["metrics"]["r2"]       for r in runs.values()]
        rhos = [r["metrics"]["spearman"] for r in runs.values()]
        print(f"  {t:<22}  R²={np.mean(r2s):+.3f}±{np.std(r2s):.3f}"
              f"  ρ={np.mean(rhos):+.3f}±{np.std(rhos):.3f}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
