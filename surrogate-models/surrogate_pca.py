#!/usr/bin/env python
# coding: utf-8
"""
surrogate_pca.py

Lightweight PCA analysis of the RDKit descriptor space.
Produces two diagnostic figures to guide the choice of n_components
before training:

  pca_variance.png  — scree plot + cumulative explained variance
  pca_scatter.png   — PC1 vs PC2 coloured by each of the 9 target properties
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import string

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_CSV = "../data/SurfPro-MD.csv"

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
# LOAD DATA AND BUILD FEATURE MATRIX
# =============================================================================
print("Loading data ...")
df = pd.read_csv(DATA_CSV)
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid  = df[df["mol"].notnull()].copy()
print(f"  {len(df_valid)} valid molecules")

rdkit_cols = sorted([c for c in df_valid.columns if c.startswith("rdkit-")])
print(f"  {len(rdkit_cols)} RDKit features")

X = df_valid[rdkit_cols].to_numpy().astype(float)
col_means = np.nanmean(X, axis=0)
for j in range(X.shape[1]):
    X[np.isnan(X[:, j]), j] = col_means[j]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# FIT PCA
# =============================================================================
print("Fitting PCA ...")
pca      = PCA(random_state=3)
X_pca    = pca.fit_transform(X_scaled)
n_total  = pca.n_components_
cum_var  = np.cumsum(pca.explained_variance_ratio_)

for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_needed = int(np.searchsorted(cum_var, threshold)) + 1
    print(f"  {threshold:.0%} variance explained by {n_needed} components")

# =============================================================================
# FIGURE 1 — scree plot + cumulative variance
# =============================================================================
plt.rcParams.update({
    "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.spines.top": False, "axes.spines.right": False,
})

N_SHOW = 60   # components shown in scree plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Scree plot
ax1.bar(np.arange(1, N_SHOW + 1),
        pca.explained_variance_ratio_[:N_SHOW] * 100,
        color="steelblue", alpha=0.8)
ax1.set_xlabel("Principal component", fontsize=12)
ax1.set_ylabel("Explained variance [%]", fontsize=12)
ax1.set_title("Scree plot (first 60 components)", fontsize=12)

# Cumulative variance
ax2.plot(np.arange(1, n_total + 1), cum_var * 100, color="steelblue", lw=2)
for threshold, color in [(0.80, "#e07b39"), (0.90, "#c0392b"),
                         (0.95, "#8e44ad"), (0.99, "#2c3e50")]:
    n = int(np.searchsorted(cum_var, threshold)) + 1
    ax2.axhline(threshold * 100, color=color, lw=1, ls="--", alpha=0.8)
    ax2.axvline(n, color=color, lw=1, ls="--", alpha=0.8)
    ax2.text(n + 1, threshold * 100 - 1.5, f"{threshold:.0%} → {n} PCs",
             color=color, fontsize=10, va="top")

ax2.set_xlabel("Number of components", fontsize=12)
ax2.set_ylabel("Cumulative explained variance [%]", fontsize=12)
ax2.set_title("Cumulative explained variance", fontsize=12)
ax2.set_xlim(0, n_total)
ax2.set_ylim(0, 101)

fig.tight_layout()
fig.savefig("pca_variance.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved pca_variance.png")

# =============================================================================
# FIGURE 2 — PC1 vs PC2 coloured by each target property
# =============================================================================
alphabet = list(string.ascii_lowercase)
cmap     = cm.get_cmap("viridis")

fig2, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]

for i, target in enumerate(TARGETS):
    ax     = axes[i]
    values = df_valid[target].values.astype(float)
    valid  = np.isfinite(values)

    # grey background for molecules without this target measured
    ax.scatter(pc1[~valid], pc2[~valid], s=4, color="lightgray",
               linewidths=0, alpha=0.4, zorder=1, rasterized=True)

    sc = ax.scatter(pc1[valid], pc2[valid], c=values[valid],
                    s=8, cmap="viridis", alpha=0.7,
                    linewidths=0, zorder=2, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(target_labels[target], fontsize=12)
    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.text(-0.15, 1.04, alphabet[i], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

fig2.suptitle("PC1 vs PC2 — coloured by target property", fontsize=13, y=1.01)
fig2.tight_layout()
fig2.savefig("pca_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("Saved pca_scatter.png")
