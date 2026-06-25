#!/usr/bin/env python
# coding: utf-8
"""
nn-analysis.py

For each target property, compute the distance to the k-th nearest neighbour
for every molecule in the filtered dataset (k = 1, 2, 5, 10).  Results are
cached to a pickle file so the expensive computation only runs once.

One 3×3 figure is produced per k value, saved as nn_distances_k{k}.png.
"""

import os
import pickle
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_CSV   = "../data/SurfPro-MD.csv"
MODELS_PKL = "models.pkl"
CACHE_FILE = "nn_distances_cache.pkl"
K_VALUES   = [1, 2, 5, 10]

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

target_labels = {
    "pCMC":                "pCMC",
    "AW_ST_CMC":           r"$\gamma_{\mathrm{CMC}}$",
    "Gamma_max":           r"$\Gamma_{\max}$",
    "pC20":                "pC$_{20}$",
    "Area_min":            r"A$_{\min}$",
    "viscosity":           r"$\eta$",
    "D_MOL":               r"D$_{\mathrm{MOL}}$",
    "D_SOL":               r"D$_{\mathrm{SOL}}$",
    "surface_tension_avg": r"$\gamma$",
}

# =============================================================================
# OUTLIER FILTER  (must match surrogate.py exactly)
# =============================================================================
def filter_outliers(df, target):
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
    return df

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================
print("Loading data ...")
df = pd.read_csv(DATA_CSV)
print(f"  {len(df)} rows loaded.")

print("Loading models ...")
with open(MODELS_PKL, "rb") as f:
    results = pickle.load(f)

# =============================================================================
# DISTANCE COMPUTATION  (or load from cache)
# =============================================================================
if os.path.exists(CACHE_FILE):
    print(f"Loading cached distances from {CACHE_FILE} ...")
    with open(CACHE_FILE, "rb") as f:
        nn_distances = pickle.load(f)
else:
    print("Computing k-NN distances ...")
    nn_distances = {}   # {target: {k: np.array shape (N,)}}
    max_k = max(K_VALUES)

    for target in TARGETS:
        # Scaler and feature schema from test_id=0, fold=0
        model_dict   = results[target][0]["fold_models"][0]
        scaler       = model_dict["scaler"]
        feature_cols = model_dict["features"]

        # All molecules with valid, in-range target values
        df_target = df.dropna(subset=[target]).copy()
        df_target = filter_outliers(df_target, target)
        n = len(df_target)
        print(f"  {target}: {n} molecules", end=" ... ", flush=True)

        X = df_target[feature_cols].to_numpy()

        # Impute NaN features with the scaler's column means (same as predict.py)
        nan_mask = np.isnan(X)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = scaler.mean_[j]

        X_scaled = scaler.transform(X)

        # Query k+1 neighbours: column 0 is the molecule itself (distance = 0),
        # so column k gives the true k-th nearest neighbour.
        k_query = min(max_k + 1, n)
        nn = NearestNeighbors(n_neighbors=k_query, metric="euclidean", n_jobs=-1)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)

        nn_distances[target] = {}
        for k in K_VALUES:
            if k < k_query:
                nn_distances[target][k] = distances[:, k]
            else:
                nn_distances[target][k] = np.full(n, np.nan)

        print("done")

    print(f"Saving cache to {CACHE_FILE} ...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(nn_distances, f)

# =============================================================================
# FIGURES — one per k value
# =============================================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
alphabet = list(string.ascii_lowercase)

for k in K_VALUES:
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for i, target in enumerate(TARGETS):
        ax    = axes[i]
        dists = nn_distances[target][k]
        dists = dists[~np.isnan(dists)]

        ax.hist(dists, bins=40, color="steelblue", alpha=0.8, edgecolor="none")

        mn  = dists.min()
        mx  = dists.max()
        avg = dists.mean()
        std = dists.std()

        stats_text = (
            f"min  = {mn:.2f}\n"
            f"max  = {mx:.2f}\n"
            f"mean = {avg:.2f}\n"
            f"std  = {std:.2f}"
        )
        ax.text(0.97, 0.97, stats_text,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none"))

        ax.set_title(target_labels[target], fontsize=13)
        ax.set_xlabel(f"Distance to {k}-NN (scaled feature space)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.text(-0.12, 1.04, alphabet[i], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")

    fig.suptitle(f"k = {k} nearest-neighbour distances within training dataset",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fname = f"nn_distances_k{k}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

print("Done.")
