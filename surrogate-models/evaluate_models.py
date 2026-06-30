#!/usr/bin/env python3
"""
Evaluate all 3 surrogate model sets on their held-out test sets.

models.pkl was trained with XGBoost 1.x; the current environment has 2.1.4.
The version change broke inplace_predict for the old pickled Boosters (base_score
is no longer correctly reconstructed, so predictions are offset by ~2-3 units).
For models.pkl we therefore use the stored y_test / y_pred (ensemble) values,
which were computed correctly at training time.

For models_pca.pkl and models_reduced.pkl (trained with XGBoost 2.1.4) we can
reconstruct individual fold predictions and compute both per-fold and
per-ensemble metrics.

Table A: mean ± std computed across all 25 (test_split × fold) models.
         For models.pkl only 5 ensemble values are available → same as Table B.
Table B: fold predictions are first averaged within each test split (reproducing
         the ensemble), then mean ± std is taken across the 5 per-split metrics.
"""

import pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]
N_TEST_SPLITS = 5
N_FOLDS       = 5
BASE = "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models"
DATA_CSV = "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/data/SurfPro-MD.csv"

print("Loading CSV …", flush=True)
df = pd.read_csv(DATA_CSV)
print(f"  {len(df)} rows", flush=True)


# ---------------------------------------------------------------------------
# PCA feature reconstruction
# ---------------------------------------------------------------------------
def get_pca_features(test_idx, pca_transform):
    raw_cols   = pca_transform["raw_feature_cols"]
    raw_scaler = pca_transform["raw_scaler"]
    X = df.loc[test_idx, raw_cols].to_numpy().astype(float)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = raw_scaler.mean_[j]
    X_sc = raw_scaler.transform(X)

    pca_type = pca_transform["type"]
    if pca_type == "global":
        return pca_transform["global_pca"].transform(X_sc)[:, :pca_transform["n_pcs"]]

    if pca_type == "two_stage":
        grp_cols     = {}
        for label, info in pca_transform["group_pcas"].items():
            grp_cols[f"grp_{label}"] = info["pca"].transform(
                X_sc[:, info["col_indices"]])[:, 0]
        pool_idx = pca_transform["pool_col_indices"]
        X_pool   = pca_transform["pool_pca"].transform(
            pca_transform["pool_scaler"].transform(X_sc[:, pool_idx])
        )[:, :pca_transform["n_pool_pcs"]]

        parts = []
        for name in pca_transform["feature_schema"]:
            if name in grp_cols:
                parts.append(grp_cols[name].reshape(-1, 1))
            elif name.startswith("pool_"):
                parts.append(X_pool[:, int(name.split("_")[1]):int(name.split("_")[1])+1])
        return np.concatenate(parts, axis=1)

    raise ValueError(f"Unknown PCA type: {pca_type}")


def fold_prediction_pca(pkl_data, target, test_id, fold_id):
    run        = pkl_data[target][test_id]
    fold_model = run["fold_models"][fold_id]
    test_idx   = run["test_idx"]
    y_test     = np.array(run["y_test"])
    pca_tf     = pkl_data[target]["_pca_transform"]
    X          = get_pca_features(test_idx, pca_tf)
    X_sc       = fold_model["scaler"].transform(X)
    y_pred     = fold_model["model"].inplace_predict(X_sc)
    return y_test, y_pred


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
HDR = f"{'Property':<22}  {'R² mean±std':>22}  {'Spearman ρ mean±std':>22}"
SEP = "-" * len(HDR)

def print_table(title, r2_dict, rho_dict):
    print(f"\n{title}")
    print(HDR); print(SEP)
    for t in TARGETS:
        if t not in r2_dict or not r2_dict[t]:
            continue
        r2  = np.array(r2_dict[t])
        rho = np.array(rho_dict[t])
        print(f"{t:<22}  {np.mean(r2):6.3f} ± {np.std(r2):.3f}          "
              f"{np.mean(rho):6.3f} ± {np.std(rho):.3f}")


# ---------------------------------------------------------------------------
# models.pkl  — stored ensemble predictions (XGBoost 1.x incompatible)
# ---------------------------------------------------------------------------
print(f"\n{'#'*70}")
print("# Raw RDKit (all 217 features) — models.pkl")
print("# NOTE: trained with XGBoost 1.x; individual fold predictions cannot")
print("#       be reconstructed with the current XGBoost 2.1.4 environment.")
print("#       Both tables use the 5 per-split ensemble predictions stored at")
print("#       training time.")
print(f"{'#'*70}")

with open(f"{BASE}/models.pkl", "rb") as f:
    d_raw = pickle.load(f)

ens_r2_raw  = {t: [] for t in TARGETS}
ens_rho_raw = {t: [] for t in TARGETS}

for target in TARGETS:
    if target not in d_raw:
        continue
    print(f"  {target}", end="", flush=True)
    for tid in range(N_TEST_SPLITS):
        if tid not in d_raw[target]:
            continue
        run    = d_raw[target][tid]
        y_test = np.array(run["y_test"])
        y_pred = np.array(run["y_pred"])          # stored ensemble
        ens_r2_raw[target].append(r2_score(y_test, y_pred))
        ens_rho_raw[target].append(spearmanr(y_test, y_pred)[0])
        print(".", end="", flush=True)
    print()

print_table("Table A — std across all 25 fold models  (= Table B, see note above)",
            ens_r2_raw, ens_rho_raw)
print_table("Table B — std across 5 per-split ensembles",
            ens_r2_raw, ens_rho_raw)


# ---------------------------------------------------------------------------
# models_pca.pkl  and  models_reduced.pkl  — individual fold predictions
# ---------------------------------------------------------------------------
for pkl_name, label in [
    ("models_pca.pkl",     "Global PCA (95% variance) — models_pca.pkl"),
    ("models_reduced.pkl", "Two-stage PCA (group PC1s + pool 95%) — models_reduced.pkl"),
]:
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"{'#'*70}")

    with open(f"{BASE}/{pkl_name}", "rb") as f:
        pkl_data = pickle.load(f)

    fold_r2s  = {t: [] for t in TARGETS}
    fold_rhos = {t: [] for t in TARGETS}
    ens_r2s   = {t: [] for t in TARGETS}
    ens_rhos  = {t: [] for t in TARGETS}

    for target in TARGETS:
        if target not in pkl_data:
            continue
        print(f"  {target}", end="", flush=True)

        for tid in range(N_TEST_SPLITS):
            if tid not in pkl_data[target]:
                continue
            y_test_ref = np.array(pkl_data[target][tid]["y_test"])
            fold_preds = []

            for fid in range(N_FOLDS):
                y_test, y_pred = fold_prediction_pca(pkl_data, target, tid, fid)
                fold_r2s[target].append(r2_score(y_test, y_pred))
                fold_rhos[target].append(spearmanr(y_test, y_pred)[0])
                fold_preds.append(y_pred)
                print(".", end="", flush=True)

            y_ens = np.mean(fold_preds, axis=0)
            ens_r2s[target].append(r2_score(y_test_ref, y_ens))
            ens_rhos[target].append(spearmanr(y_test_ref, y_ens)[0])

        print()

    print_table("Table A — std across all 25 individual fold models",
                fold_r2s, fold_rhos)
    print_table("Table B — std across 5 per-split ensembles",
                ens_r2s, ens_rhos)

print("\nDone.")
