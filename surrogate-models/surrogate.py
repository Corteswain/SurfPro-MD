#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# IMPORTS
# =============================================================================
import sys
import os
import re
import glob
import json
import pickle
import random
import string
import itertools
import datetime
import warnings
import argparse
from collections import Counter, defaultdict
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from scipy.stats import pearsonr, rankdata, sem, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity, ConvertToNumpyArray
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors

from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBRegressor
import joblib
import optuna
import shap

# =============================================================================
# GLOBAL PLOTTING STYLE
# =============================================================================
sns.set_theme(
    context="notebook",
    style="whitegrid",
    palette="viridis",
    font_scale=1.2,
)

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
    "image.cmap": "viridis",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.borderpad": 0.1,
    "legend.labelspacing": 0.2,
    "savefig.dpi": 600,
})

print("Global plotting style configured.")

# =============================================================================
# CONSTANTS
# =============================================================================
# Canonical training target order
TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

# Short display names (LaTeX)
target_labels = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          r"$\gamma_{\mathrm{CMC}}$",
    "Gamma_max":          r"$\Gamma_{\max}$",
    "pC20":               "pC$_{20}$",
    "Pi_CMC":             r"$\Pi_{\mathrm{CMC}}$",
    "Area_min":           r"A$_{\min}$",
    "viscosity":          r"$\eta$",
    "D_MOL":              r"D$_{\mathrm{MOL}}$",
    "D_SOL":              r"D$_{\mathrm{SOL}}$",
    "surface_tension_avg": r"$\gamma$",
}

# Axis labels with units
unit_labels = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          r"$\gamma_{\mathrm{CMC}}$ [mN m$^{-1}$]",
    "Gamma_max":          r"$\Gamma_{\max}$ [$\mu$mol m$^{-2}$]",
    "pC20":               "pC$_{20}$",
    "Pi_CMC":             r"$\Pi_{\mathrm{CMC}}$ [mN m$^{-1}$]",
    "Area_min":           r"Area$_{\min}$ [nm$^2$]",
    "viscosity":          r"$\eta$ [mPa·s]",
    "D_MOL":              r"D$_\mathrm{MOL}$ [m$^2$s$^{-1}$]",
    "D_SOL":              r"D$_\mathrm{SOL}$ [m$^2$s$^{-1}$]",
    "surface_tension_avg": r"$\gamma$ [mN m$^{-1}$]",
}

# Training hyper-parameters
N_FOLDS            = 5
N_TEST_SPLITS      = 5
MIN_TEST_FRACTION  = 0.07
MAX_TEST_FRACTION  = 0.11
MAX_REJECT_TRIES   = 5
MIN_VAL_FRACTION   = 0.12
MAX_VAL_FRACTION   = 0.18
RANDOM_STATE       = 3

# Number of equilibration steps skipped when reading .xvg files
EQ_STEPS = 100

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description="SurfPro-MD surrogate model pipeline")
parser.add_argument(
    "--train", action="store_true", default=False,
    help="Run the full training loop and overwrite models.pkl. "
         "Without this flag the script loads existing models from models.pkl.",
)
parser.add_argument(
    "--model", default="models.pkl",
    help="Path to the models pickle file (default: models.pkl).",
)
args = parser.parse_args()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_xvg(fname, quantity):
    """Parse a GROMACS .xvg file and return (mean, std) for the named quantity."""
    if fname[-3:] != "xvg":
        print("The provided file is not in .xvg format.")
        sys.exit(0)
    with open(fname, "r") as fid:
        ll = fid.readlines()
    M = len(ll[-1].split())
    labels = [""] * M
    labels[0] = "Time (ps)"
    ALL = []
    icol = -1
    i = 1
    for l in ll:
        if l.startswith("@ s"):
            label = " ".join(l.split()[3:])[1:-1]
            labels[i] = label
            if label.strip() == quantity.strip():
                icol = i
            i += 1
        if l[0] in ["#", "@"]:
            continue
        s = l.split()
        ALL.append([float(x) for x in s])
    if icol == -1:
        print("QUANTITY HAS NOT BEEN COMPUTED IN " + str(fname))
        return 0
    ALL = np.asarray(ALL).transpose()
    average = np.average(ALL[icol, EQ_STEPS:])
    std = np.std(ALL[icol, EQ_STEPS:])
    return [average, std]


def get_colors(n):
    """Return n evenly-spaced viridis colors."""
    cmap = cm.get_cmap("viridis")
    return [cmap(i / (n - 1)) for i in range(n)]


def sample_test_clusters(clusters, cluster_sizes, min_size, max_size, rng,
                         max_rejects=5):
    available_clusters = set(clusters)
    test_clusters = set()
    test_count = 0
    reject_streak = 0

    while available_clusters:
        c = rng.choice(list(available_clusters))
        available_clusters.remove(c)
        new_size = test_count + cluster_sizes[c]

        if new_size < min_size:
            test_clusters.add(c)
            test_count = new_size
            reject_streak = 0
            continue

        if min_size <= new_size <= max_size:
            test_clusters.add(c)
            test_count = new_size
            break

        reject_streak += 1
        if reject_streak >= max_rejects:
            print("COULD NOT CREATE SET")
            break

    return test_clusters, test_count


def create_cluster_splits(df_input, target_col):
    df_t = df_input.dropna(subset=[target_col, "cluster"]).copy()
    df_t["cluster"] = df_t["cluster"].astype(int)

    total_mols = len(df_t)
    min_test_size = int(total_mols * MIN_TEST_FRACTION)
    max_test_size = int(total_mols * MAX_TEST_FRACTION)

    cluster_sizes_all = df_t.groupby("cluster").size().to_dict()
    clusters = np.array(list(cluster_sizes_all.keys()))

    for test_split_id in range(N_TEST_SPLITS):
        rng = np.random.RandomState(RANDOM_STATE + test_split_id)

        test_clusters, _ = sample_test_clusters(
            clusters=clusters,
            cluster_sizes=cluster_sizes_all,
            min_size=min_test_size,
            max_size=max_test_size,
            rng=rng,
            max_rejects=MAX_REJECT_TRIES,
        )

        col_test = f"{target_col}_test_split_{test_split_id}"
        df_t[col_test] = "train_val"
        df_t.loc[df_t["cluster"].isin(test_clusters), col_test] = "test"

        remaining_df = df_t[df_t[col_test] != "test"]
        remaining_clusters = np.array(remaining_df["cluster"].unique())
        remaining_sizes = remaining_df.groupby("cluster").size().to_dict()

        remaining_mols = len(remaining_df)
        min_val_size = int(remaining_mols * MIN_VAL_FRACTION)
        max_val_size = int(remaining_mols * MAX_VAL_FRACTION)

        for fold_id in range(N_FOLDS):
            rng_fold = np.random.RandomState(
                RANDOM_STATE + test_split_id * 100 + fold_id
            )

            val_clusters, _ = sample_test_clusters(
                clusters=remaining_clusters,
                cluster_sizes=remaining_sizes,
                min_size=min_val_size,
                max_size=max_val_size,
                rng=rng_fold,
                max_rejects=MAX_REJECT_TRIES,
            )

            train_clusters = set(remaining_clusters) - set(val_clusters)
            col = f"{target_col}_split_t{test_split_id}_f{fold_id}"

            df_t[col] = "unused"
            df_t.loc[df_t["cluster"].isin(train_clusters), col] = "train"
            df_t.loc[df_t["cluster"].isin(val_clusters), col]   = "val"
            df_t.loc[df_t[col_test] == "test", col]             = "test"

    return df_t


def filter_outliers(df, target):
    n_before = len(df)

    if target == "surface_tension_avg":
        df = df[(df["surface_tension_avg"].notnull()) & (df["surface_tension_avg"] >= 250)]
    elif target == "viscosity":
        df = df[(df["viscosity"].notnull()) & (df["viscosity"] <= 0.003)]
    elif target == "AW_ST_CMC":
        df = df[(df["AW_ST_CMC"].notnull()) & (df["AW_ST_CMC"] <= 52)]
    elif target == "Gamma_max":
        df = df[(df["Gamma_max"].notnull()) & (df["Gamma_max"] <= 6)]
    elif target == "Area_min":
        df = df[(df["Area_min"].notnull()) & (df["Area_min"] <= 4.2)]
    elif target == "pC20":
        df = df[(df["pC20"].notnull()) & (df["pC20"] >= 1.8)]
    elif target == "D_MOL":
        df = df[(df["D_MOL"].notnull()) & (df["D_MOL"] <= 0.8)]

    n_after = len(df)
    print(f"Filtering Target: {target}")
    print(f"  Rows before: {n_before}  |  after: {n_after}  |  removed: {n_before - n_after} "
          f"({(n_before - n_after) / n_before:.2%})")
    return df


def build_df_target(df, target):
    """Convenience wrapper: cluster-split + outlier filter for one target."""
    df_target = create_cluster_splits(df, target)
    df_target = filter_outliers(df_target, target)
    return df_target


def optimise(df, feature_cols, target_col, split, outname,
             draw_plots=False, n_trials=30):
    """Optuna-optimised XGBoost fit.  Returns a dict with model, scaler, features."""

    X_train = df.loc[split == "train", feature_cols].reset_index(drop=True)
    X_val   = df.loc[split == "val",   feature_cols].reset_index(drop=True)

    y_train = df.loc[split == "train", target_col].reset_index(drop=True)
    y_val   = df.loc[split == "val",   target_col].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # Baseline (untuned)
    baseline_model = xgb.XGBRegressor(random_state=72)
    baseline_model.fit(X_train_scaled, y_train)
    baseline_preds = baseline_model.predict(X_val_scaled)
    baseline_r2 = r2_score(y_val, baseline_preds)

    # Optuna optimisation
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
            "random_state":  72,
            "verbosity":     0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train)
        return r2_score(y_val, model.predict(X_val_scaled))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("  Best params:", study.best_params)
    print(f"  Best R² (val): {study.best_value:.3f}")

    best_model = xgb.XGBRegressor(**study.best_params, random_state=72)
    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_val_scaled)
    r2 = r2_score(y_val, y_pred)

    print(f"  original R² = {baseline_r2:.3f}  |  optimal R² = {r2:.3f}  |  ΔR² = {r2 - baseline_r2:.3f}")

    return {
        "model":       best_model.get_booster(),
        "scaler":      scaler,
        "features":    feature_cols,
        "best_params": study.best_params,
    }


def ensemble_predict(fold_models, X):
    """Average predictions from a list of fold model dicts over a feature matrix X."""
    preds = []
    for m in fold_models:
        X_scaled = m["scaler"].transform(X)
        preds.append(m["model"].inplace_predict(X_scaled))
    return np.mean(preds, axis=0)


# =============================================================================
# 1. LOAD DATA AND COMPUTE CLUSTERS
# =============================================================================
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

fname = "../data/SurfPro-MD.csv"
df = pd.read_csv(fname)

# --- Commented-out block: original data preprocessing pipeline ---
# (run once to build SurfPro-MD.csv; kept here for reference)
#
# df = df.dropna(subset=["SMILES"]).copy()
# print(df.columns)
#
# def canonicalize_smiles(smiles, keep_largest=True):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None
#         if keep_largest and '.' in smiles:
#             frags = Chem.GetMolFrags(mol, asMols=True)
#             mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
#         return Chem.MolToSmiles(mol, canonical=True)
#     except Exception:
#         return None
#
# tqdm.pandas(desc="Canonicalizing SMILES")
# df["SMILES_canonical"] = df["SMILES"].progress_apply(canonicalize_smiles)
# df = df.dropna(subset=["SMILES_canonical"]).reset_index(drop=True)
# n_total = len(df)
# n_unique = df["SMILES_canonical"].nunique()
# print(f"Found {n_total - n_unique} duplicates based on canonical SMILES.")
# df = df.drop_duplicates(subset="SMILES_canonical", keep="first").reset_index(drop=True)
# print(f"After cleaning: {len(df)} unique, valid molecules")
#
# descriptor_names = [desc_name for desc_name, _ in Descriptors._descList]
# calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
#
# def compute_rdkit_descriptors(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return {f"rdkit-{name}": np.nan for name in descriptor_names}
#     try:
#         values = calculator.CalcDescriptors(mol)
#         return {f"rdkit-{name}": val for name, val in zip(descriptor_names, values)}
#     except Exception:
#         return {f"rdkit-{name}": np.nan for name in descriptor_names}
#
# tqdm.pandas(desc="Descriptors")
# rdkit_descs = df["SMILES_canonical"].progress_apply(compute_rdkit_descriptors)
# rdkit_df = pd.DataFrame.from_records(rdkit_descs)
# assert len(rdkit_df) == len(df), "Descriptor DataFrame length mismatch!"
# df_final = pd.concat([df, rdkit_df], axis=1)
# df_final["SMILES"] = df_final["SMILES_canonical"]
# df_final.reset_index(drop=True, inplace=True)
# df = df_final

print(df.columns)

# --- Butina clustering on Morgan fingerprints ---
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid = df[df["mol"].notnull()].copy()

def get_fingerprint(mol, radius=2, n_bits=1024):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

fps = [get_fingerprint(m) for m in df_valid["mol"]]
df_valid["fingerprint"] = fps
nfps = len(fps)
print(f"Generated {nfps} valid fingerprints.")

dists = []
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])

dist_thresh = 0.6
clusters_butina = Butina.ClusterData(dists, nfps, distThresh=dist_thresh, isDistData=True)

cluster_id = np.zeros(nfps, dtype=int)
for i, cluster in enumerate(clusters_butina):
    for idx in cluster:
        cluster_id[idx] = i

df_valid["cluster"] = cluster_id
df_valid["cluster"] = df_valid["cluster"].astype("Int64")
print(f"Found {len(clusters_butina)} clusters.")

# t-SNE embedding for visualisation
fps_np = np.zeros((nfps, 1024), dtype=int)
for i, fp in enumerate(fps):
    DataStructs.ConvertToNumpyArray(fp, fps_np[i])

tsne = TSNE(n_components=2, perplexity=20, random_state=3, metric="cosine")
tsne_result = tsne.fit_transform(fps_np)

df_valid["tsne_1"] = tsne_result[:, 0]
df_valid["tsne_2"] = tsne_result[:, 1]

df["cluster"] = np.nan
df.loc[df_valid.index, "cluster"] = df_valid["cluster"].values
df.loc[df_valid.index, "tsne_1"]  = df_valid["tsne_1"].values
df.loc[df_valid.index, "tsne_2"]  = df_valid["tsne_2"].values

print(df.columns)

# =============================================================================
# 2. FIGURE — DATA OVERVIEW (histograms + correlation matrix)
# =============================================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 9-panel histogram grid + correlation matrix
TARGETS_HIST = [
    "pCMC", "D_MOL", "D_SOL",
    "surface_tension_avg", "viscosity", "AW_ST_CMC",
    "Gamma_max", "Area_min", "pC20",
]

fig = plt.figure(figsize=(11, 5))
gs = fig.add_gridspec(
    nrows=1, ncols=4,
    width_ratios=[1, 0.2, 1.0, 0.05],
    wspace=0.1,
)
gs_left = gs[0, 0].subgridspec(3, 3, wspace=0.25, hspace=0.5)
ax_corr = fig.add_subplot(gs[0, 2])
ax_cbar = fig.add_subplot(gs[0, 3])

colors_hist = cm.viridis(np.linspace(0.15, 0.85, len(TARGETS_HIST)))

for i, target in enumerate(TARGETS_HIST):
    ax = fig.add_subplot(gs_left[i // 3, i % 3])
    values = df[target].dropna().values
    ax.hist(values, bins=25, density=True, alpha=0.75, color=colors_hist[0])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.25)
    ax.set_xlabel("")
    ax.text(0.05, 1.05, unit_labels[target], transform=ax.transAxes,
            ha="left", va="top", fontsize=14)

# Correlation matrix.
# TARGETS_CORR and labels must be in the same order.
TARGETS_CORR = [
    "pCMC", "AW_ST_CMC", "Gamma_max",
    "pC20", "Pi_CMC", "Area_min",
    "viscosity", "D_MOL", "D_SOL", "surface_tension_avg",
]
corr_labels = [
    "pCMC",
    r"$\gamma_{\mathrm{CMC}}$",
    r"$\Gamma_{\max}$",
    "pC$_{20}$",
    r"$\Pi_{\mathrm{CMC}}$",
    r"A$_{\min}$",
    r"$\eta$",
    r"D$_{\mathrm{MOL}}$",
    r"D$_{\mathrm{SOL}}$",
    r"$\gamma$",
]

corr = df[TARGETS_CORR].corr()

ax_corr.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
ax_corr.set_xticks(np.arange(len(TARGETS_CORR)))
ax_corr.set_yticks(np.arange(len(TARGETS_CORR)))
ax_corr.set_xticklabels(corr_labels, rotation=45, ha="right", fontsize=14)
ax_corr.set_yticklabels(corr_labels, fontsize=14)

for i in range(len(TARGETS_CORR)):
    for j in range(len(TARGETS_CORR)):
        color = "white" if corr.values[i, j] <= -0.46 else "black"
        ax_corr.text(j, i, f"{corr.values[i, j]:.1f}",
                     ha="center", va="center", fontsize=11, color=color)

im = ax_corr.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("Pearson correlation", fontsize=11)
cbar.ax.tick_params(labelsize=11)
ax_corr.grid(False)
ax_corr.set_aspect("equal")

fig.text(0.1,  0.85, "a", fontsize=14, fontweight="bold")
fig.text(0.50, 0.85, "b", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("figure_data.pdf", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 3. TRAINING LOOP  (skipped unless --train is passed)
# =============================================================================
if args.train:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = {}

    for target in TARGETS:
        print(f"\nWorking on {target}")

        df_target = create_cluster_splits(df, target)
        df_target = filter_outliers(df_target, target)

        rdkit_cols = sorted([c for c in df_target.columns if c.startswith("rdkit-")])
        FEATURE_SCHEMA = rdkit_cols

        results[target] = {}

        for test_id in range(N_TEST_SPLITS):
            print(f"\n  ### TEST SPLIT {test_id}")

            fold_models = []
            for fold in range(N_FOLDS):
                split_col = f"{target}_split_t{test_id}_f{fold}"
                split = df_target[split_col]
                model = optimise(
                    df=df_target,
                    feature_cols=FEATURE_SCHEMA,
                    target_col=target,
                    split=split,
                    outname=target,
                    draw_plots=False,
                )
                fold_models.append(model)

            # Test set (defined by fold 0; all folds share the same test split)
            split_col0 = f"{target}_split_t{test_id}_f0"
            test_idx = df_target.index[df_target[split_col0] == "test"].to_numpy()
            X_test   = df_target.loc[test_idx, FEATURE_SCHEMA].to_numpy()
            y_test   = df_target.loc[test_idx, target].to_numpy()

            ensemble_preds = ensemble_predict(fold_models, X_test)
            print(f"  Ensemble R² = {r2_score(y_test, ensemble_preds):.3f}")

            r2   = r2_score(y_test, ensemble_preds)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
            rho, _ = spearmanr(y_test, ensemble_preds)
            pcc, _ = pearsonr(y_test, ensemble_preds)

            results[target][test_id] = {
                "fold_models":    fold_models,
                "test_idx":       test_idx,
                "split_column":   split_col0,
                "feature_schema": FEATURE_SCHEMA,
                "y_test":         y_test.tolist(),
                "y_pred":         ensemble_preds.tolist(),
                "metrics": {
                    "r2":       r2,
                    "rmse":     rmse,
                    "spearman": rho,
                    "pearson":  pcc,
                },
            }

            with open(args.model, "wb") as f:
                pickle.dump(results, f)

else:
    print(f"Loading models from {args.model} (pass --train to retrain)")
    with open(args.model, "rb") as f:
        results = pickle.load(f)

# =============================================================================
# 4. METRICS SUMMARY BAR CHART
# =============================================================================
metrics_summary = {}

for target, test_runs in results.items():
    r2_vals   = [run["metrics"]["r2"]       for run in test_runs.values()]
    rho_vals  = [run["metrics"]["spearman"] for run in test_runs.values()]
    rmse_vals = [run["metrics"]["rmse"]     for run in test_runs.values()]
    pcc_vals  = [run["metrics"]["pearson"]  for run in test_runs.values()]

    metrics_summary[target] = {
        "r2_mean":   np.mean(r2_vals),   "r2_std":   np.std(r2_vals),
        "rho_mean":  np.mean(rho_vals),  "rho_std":  np.std(rho_vals),
        "rmse_mean": np.mean(rmse_vals), "rmse_std": np.std(rmse_vals),
        "pcc_mean":  np.mean(pcc_vals),  "pcc_std":  np.std(pcc_vals),
    }

x = np.arange(len(TARGETS))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width / 2,
        [metrics_summary[t]["r2_mean"]  for t in TARGETS], width,
        yerr=[metrics_summary[t]["r2_std"]  for t in TARGETS], label="R²")
plt.bar(x + width / 2,
        [metrics_summary[t]["rho_mean"] for t in TARGETS], width,
        yerr=[metrics_summary[t]["rho_std"] for t in TARGETS], label="Spearman ρ")
plt.xticks(x, TARGETS, rotation=45, ha="right")
plt.ylim([-1, 1])
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# 5. PERMUTATION IMPORTANCE
# =============================================================================
with open(args.model, "rb") as f:
    results = pickle.load(f)

for target in TARGETS:
    N_REPEATS = 20

    df_target = create_cluster_splits(df, target)
    df_target = filter_outliers(df_target, target)

    all_importances = []

    for test_id in range(N_TEST_SPLITS):
        run = results[target][test_id]
        fold_models = run["fold_models"]

        for fold, model_dict in enumerate(fold_models):
            print(f"Computing importance: target={target}, test_id={test_id}, fold={fold}")

            split_col = f"{target}_split_t{test_id}_f{fold}"
            test_mask = df_target[split_col] == "test"

            X_test = model_dict["scaler"].transform(
                df_target.loc[test_mask, model_dict["features"]].to_numpy()
            )
            y_test = df_target.loc[test_mask, target].to_numpy()

            perm = permutation_importance(
                model_dict["model"], X_test, y_test,
                scoring="r2", n_repeats=N_REPEATS,
                random_state=42, n_jobs=-1,
            )
            all_importances.append(perm.importances_mean)

    all_importances = np.asarray(all_importances)
    print("Importance matrix shape:", all_importances.shape)

    mean_imp = all_importances.mean(axis=0)
    ci95     = 1.96 * sem(all_importances, axis=0)

    feature_names = fold_models[0]["features"]
    order = np.argsort(mean_imp)[::-1]
    mean_sorted = mean_imp[order]
    ci95_sorted = ci95[order]
    names_sorted = np.array(feature_names)[order]

    plt.figure(figsize=(12, 6))
    plt.errorbar(np.arange(len(mean_sorted)), mean_sorted, yerr=ci95_sorted,
                 fmt="-", linewidth=1.5, capsize=2)
    plt.xlabel("Feature rank")
    plt.ylabel("Permutation importance (ΔR²)")
    plt.title(f"{target} feature importance\nMean ± 95% CI across models")
    plt.tight_layout()
    plt.show()

    print("\nTop 30 features:\n")
    for rank, (name, imp, err) in enumerate(
        zip(names_sorted[:30], mean_sorted[:30], ci95_sorted[:30]), start=1
    ):
        print(f"{rank:2d}  {name:40s} {imp:10.5f} ± {err:.5f}")

# =============================================================================
# 6. SHAP IMPORTANCE
# =============================================================================
with open(args.model, "rb") as f:
    results = pickle.load(f)

shap_results = {}

for target in TARGETS:
    print(f"\nComputing SHAP for {target}")

    df_target = create_cluster_splits(df, target)
    df_target = filter_outliers(df_target, target)

    feature_cols = sorted([c for c in df_target.columns if c.startswith("rdkit-")])
    all_shap = []

    for test_id in range(N_TEST_SPLITS):
        run = results[target][test_id]
        fold_models = run["fold_models"]

        test_mask = df_target[run["split_column"]] == "test"
        X_np = df_target.loc[test_mask, feature_cols].to_numpy()

        fold_shap_values = []
        for model_dict in fold_models:
            X_scaled = model_dict["scaler"].transform(X_np)
            explainer = shap.Explainer(
                model_dict["model"].inplace_predict,
                shap.maskers.Independent(X_scaled),
            )
            fold_shap_values.append(explainer(X_scaled))

        all_shap.append(np.mean(np.array(fold_shap_values), axis=0))

    shap_results[target] = np.concatenate(all_shap, axis=0)

for target in TARGETS:
    shap_vals = shap_results[target]
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    order = np.argsort(mean_abs_shap)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(
        np.array(feature_cols)[order][:20][::-1],
        mean_abs_shap[order][:20][::-1],
    )
    plt.title(f"{target} - SHAP feature importance")
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. FIGURE — MD VALIDATION
#    Data for this figure is loaded into df_md (surface tension) and df_md_visc
#    (viscosity) to avoid overwriting the main training dataframe df.
# =============================================================================
plt.rcParams.update({"font.size": 14})

fig = plt.figure(figsize=(11, 5))
gsAB = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.2)
ax_tension = fig.add_subplot(gsAB[0, 0])
ax_visc    = fig.add_subplot(gsAB[0, 1])

axins = inset_axes(
    ax_tension, width="100%", height="100%",
    bbox_to_anchor=(0.5, 0.3, 0.4, 0.4),
    bbox_transform=ax_tension.transAxes,
    loc="lower left",
)
axins.set_xlim(-0.02, 0.2)
axins.set_ylim(0.6, 1.1)
axins.tick_params(labelsize=8)

# --- Panel A: Relative Surface Tension ---
all_nmol_st = [150, 300, 450, 1500, 2500]
all_mol_st  = ["methanol", "ethanol", "propanol"]
colors_st   = get_colors(len(all_mol_st))

for icolor, mol in enumerate(all_mol_st):
    df_ref = pd.read_csv(f"/home/ricbec/PFAS-project/reference/{mol}_tension.csv")

    replica_data = {}
    for nmol in all_nmol_st:
        runs = []
        for replica in range(19):
            fname = (f"/home/ricbec/PFAS-project/data/tension-alcohols/"
                     f"{mol}_{nmol}/ST_{str(replica).zfill(2)}/potential.xvg")
            avg_single, _ = read_xvg(fname, "#Surf*SurfTen")
            runs.append(avg_single / 20)
        replica_data[nmol] = np.array(runs)

    gamma0_replicas = replica_data[all_nmol_st[0]]

    rows = []
    for nmol in all_nmol_st:
        rel = replica_data[nmol] / gamma0_replicas
        rows.append({
            "c":                nmol / 15000,
            "rel_tension_mean": np.mean(rel),
            "rel_tension_std":  np.std(rel),
        })

    df_md = pd.DataFrame(rows).sort_values("c")

    df_ref = df_ref.sort_values("mol_fraction")
    df_ref["rel_tension"] = df_ref["surface_tension"] / df_ref["surface_tension"].iloc[0]

    ax_tension.errorbar(
        df_md["c"], df_md["rel_tension_mean"], yerr=df_md["rel_tension_std"],
        color=colors_st[icolor], marker="o", ms=0.1, lw=1.5, capsize=3,
        label=f"{mol} comp.",
    )
    ax_tension.plot(
        df_ref["mol_fraction"], df_ref["rel_tension"],
        ls="--", marker="o", ms=4, color=colors_st[icolor], lw=1.5,
        label=f"{mol} exp.",
    )
    axins.errorbar(
        df_md["c"], df_md["rel_tension_mean"], yerr=df_md["rel_tension_std"],
        color=colors_st[icolor], lw=1.2,
    )
    axins.plot(
        df_ref["mol_fraction"], df_ref["rel_tension"],
        ls="--", color=colors_st[icolor], lw=1.2,
    )

ax_tension.set_xlabel("Mol Fraction")
ax_tension.set_ylabel("Relative surface tension")
ax_tension.set_ylim([0.3, 1.2])

style_handles = [
    Line2D([0], [0], color="gray", lw=1.5, ls="-",  label="comp."),
    Line2D([0], [0], color="gray", lw=1.5, ls="--", label="exp."),
]
mol_handles_st = [
    Line2D([0], [0], color=colors_st[i], lw=2, label=mol)
    for i, mol in enumerate(all_mol_st)
]
ax_tension.legend(handles=style_handles + mol_handles_st, frameon=False)

# --- Panel B: Relative Viscosity ---
all_nmol_visc = [10, 30, 50, 100, 150, 200]
all_mol_visc  = ["formic", "acetic", "propionic", "valeric"]
colors_visc   = get_colors(len(all_mol_visc))

df_ref_visc = pd.read_csv("/home/ricbec/PFAS-project/reference/acid_reference.csv")
df_ref_visc["c"] = df_ref_visc["c*100"] / 100

# Baseline (pure solvent) viscosity per acid
base_data = pd.DataFrame()
for mol in all_mol_visc:
    all_runs = []
    for replica in range(10):
        fname = (f"/home/ricbec/PFAS-project/data/viscosities/"
                 f"{mol}_acid_0/ST_{str(replica).zfill(2)}/visco.xvg")
        avg_single, _ = read_xvg(fname, "1/Viscosity")
        all_runs.append(1 / avg_single)
    base_data = pd.concat([base_data, pd.DataFrame([{
        "acid": mol, "viscosity": np.mean(all_runs),
    }])], ignore_index=True)

# Concentration-dependent viscosity
df_md_visc = pd.DataFrame()
for mol in all_mol_visc:
    for nmol in all_nmol_visc:
        all_runs = []
        all_c    = []
        for replica in range(10):
            fname = (f"/home/ricbec/PFAS-project/data/viscosities/"
                     f"{mol}_acid_{nmol}/ST_{str(replica).zfill(2)}/visco.xvg")
            avg_single, _ = read_xvg(fname, "1/Viscosity")
            all_runs.append(1 / avg_single)

            with open(f"/home/ricbec/PFAS-project/data/viscosities/"
                      f"{mol}_acid_{nmol}/ST_{str(replica).zfill(2)}/production.gro") as fgro:
                boxsize = float(fgro.readlines()[-1].split()[0])
            all_c.append(nmol / (boxsize ** 3) * 10 / 6.023)

        visco      = np.mean(all_runs)
        std        = np.std(all_runs)
        base_visco = base_data.loc[base_data["acid"] == mol, "viscosity"].values[0]
        rel_visco  = 1 + (visco / base_visco - 1) * 72 / 50

        df_md_visc = pd.concat([df_md_visc, pd.DataFrame([{
            "c":            np.mean(all_c),
            "acid":         mol,
            "rel_viscosity": rel_visco,
            "error":        std / base_visco,
        }])], ignore_index=True)

for icolor, mol in enumerate(all_mol_visc):
    subset     = df_md_visc[df_md_visc["acid"] == mol].sort_values("c")
    subset_ref = df_ref_visc[df_ref_visc["acid"] == mol].sort_values("c")
    rel_visco_ref = subset_ref["viscosity"] / subset_ref["viscosity"].iloc[0]

    ax_visc.errorbar(
        subset["c"], subset["rel_viscosity"], yerr=subset["error"],
        color=colors_visc[icolor], marker="o", ms=0.1, lw=1.5, capsize=3,
        label=f"{mol} comp.",
    )
    ax_visc.plot(
        subset_ref["c"], rel_visco_ref,
        ls="--", marker="o", ms=4, color=colors_visc[icolor], lw=1.5,
        label=f"{mol} exp.",
    )

ax_visc.yaxis.tick_left()
ax_visc.set_xlabel("Concentration [mol/L]")
ax_visc.set_ylabel("Relative viscosity")
ax_visc.set_ylim([1, 1.3499])
ax_tension.set_ylim([0.3, 1.1])
ax_visc.tick_params(axis="y", pad=2)
ax_tension.tick_params(axis="y", pad=2)

mol_handles_visc = [
    Line2D([0], [0], color=colors_visc[i], lw=2, label=mol)
    for i, mol in enumerate(all_mol_visc)
]
ax_visc.legend(handles=style_handles + mol_handles_visc, frameon=False)

ax_tension.spines["left"].set_visible(True)
ax_visc.spines["left"].set_visible(True)

fig.text(0.05, 0.85, "a", fontsize=14, fontweight="bold")
fig.text(0.50, 0.85, "b", fontsize=14, fontweight="bold")

plt.savefig("MD_validation.pdf")
plt.show()

# =============================================================================
# 8. FIGURE — FULL RESULTS (ML performance + correlation plots)
# =============================================================================
plt.rcParams.update({"font.size": 14})

fig = plt.figure(figsize=(11, 12))
gs = gridspec.GridSpec(
    nrows=4, ncols=3,
    height_ratios=[1, 1, 1, 1],
    hspace=0.35, wspace=0.15,
)
ax_metrics = fig.add_subplot(gs[0, :])

# --- Metrics bar chart ---
targets_sorted = sorted(
    metrics_summary.keys(), key=lambda t: -metrics_summary[t]["r2_mean"]
)
x = np.arange(len(targets_sorted))
width = 0.35

ax_metrics.bar(
    x - width / 2,
    [metrics_summary[t]["r2_mean"]  for t in targets_sorted], width,
    yerr=[metrics_summary[t]["r2_std"]  for t in targets_sorted], label="R²",
)
ax_metrics.bar(
    x + width / 2,
    [metrics_summary[t]["rho_mean"] for t in targets_sorted], width,
    yerr=[metrics_summary[t]["rho_std"] for t in targets_sorted], label="Spearman ρ",
)
ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(
    [target_labels[t] for t in targets_sorted], rotation=45, ha="right"
)
ax_metrics.set_ylabel("Score", labelpad=-10)
ax_metrics.set_ylim([-0.25, 1])
ax_metrics.legend(frameon=False)
ax_metrics.spines["top"].set_visible(False)

# --- Correlation scatter plots (one per target, in training order) ---
alphabet = list(string.ascii_lowercase)
cmap_scatter = cm.get_cmap("viridis", N_TEST_SPLITS)

for i, target in enumerate(results.keys()):
    ax = fig.add_subplot(gs[1 + (i // 3), i % 3])
    ax.set_aspect("equal", adjustable="box")

    vmin, vmax = np.inf, -np.inf

    for test_id, run in results[target].items():
        y_true = np.array(run["y_test"])
        y_pred = np.array(run["y_pred"])

        ax.scatter(y_true, y_pred, s=10, alpha=0.7,
                   color=cmap_scatter(test_id), label=f"test {test_id}")

        vmin = min(vmin, y_true.min(), y_pred.min())
        vmax = max(vmax, y_true.max(), y_pred.max())

    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)

    m = metrics_summary[target]
    ax.text(
        0.05, 0.95,
        f"$R^2$ = {m['r2_mean']:.2f} ± {m['r2_std']:.2f}\n"
        f"$\\rho$ = {m['rho_mean']:.2f} ± {m['rho_std']:.2f}",
        transform=ax.transAxes, va="top", fontsize=14,
    )

    ax.set_ylabel(unit_labels[target])
    ax.set_xlabel(unit_labels[target])

    if i % 3 == 0:
        ax.set_ylabel(f"Predicted\n{unit_labels[target]}", fontsize=14)
    if i >= 6:
        ax.set_xlabel(f"{unit_labels[target]}\nTrue", fontsize=14)

    if target == "viscosity":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: x * 1000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x * 1000))

    ax.text(-0.15, 1.04, alphabet[i + 1], transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

fig.text(0.01, 0.99, "a", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("full_results.pdf", dpi=300, bbox_inches="tight")
plt.show()
