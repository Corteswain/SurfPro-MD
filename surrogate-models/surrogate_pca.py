#!/usr/bin/env python
# coding: utf-8
"""
surrogate_pca.py  —  two-stage dimensionality reduction

Stage 1 — group PC1 (interpretable features)
    Groups whose first principal component explains >= COMPRESSION_THRESHOLD
    of the group's variance are replaced by that PC1.  These become
    interpretable aggregated features.

Stage 2 — global PCA on the remainder
    Features that don't compress well by a single PC (poorly-compressed
    groups + all ungrouped individual features) are pooled and compressed
    together by a standard PCA.

Outputs
    pca_groups.png    — per-group bar chart with threshold line;
                        bars coloured by whether the group is kept (stage 1)
                        or sent to the pool (stage 2)
    pca_variance.png  — scree + cumulative variance of the stage-2 PCA
    pca_scatter.png   — first two stage-2 PCs coloured by target property
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_CSV             = "../data/SurfPro-MD.csv"
COMPRESSION_THRESHOLD = 0.80   # min within-group PC1 variance to "keep" a group

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
# GROUP DEFINITIONS
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


def assign_group(feature_name):
    bare = feature_name.replace("rdkit-", "")
    for label, rule in GROUP_RULES.items():
        if rule(bare):
            return label
    return None


# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data ...")
df        = pd.read_csv(DATA_CSV)
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid  = df[df["mol"].notnull()].copy()
print(f"  {len(df_valid)} valid molecules")

rdkit_cols = sorted([c for c in df_valid.columns if c.startswith("rdkit-")])
print(f"  {len(rdkit_cols)} RDKit features")

X_raw     = df_valid[rdkit_cols].to_numpy().astype(float)
col_means = np.nanmean(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    X_raw[np.isnan(X_raw[:, j]), j] = col_means[j]

scaler_all = StandardScaler()
X_scaled   = scaler_all.fit_transform(X_raw)

# =============================================================================
# ASSIGN FEATURES TO GROUPS
# =============================================================================
groups        = {}   # {label: [column indices in rdkit_cols]}
ungrouped_idx = []

for ci, col in enumerate(rdkit_cols):
    g = assign_group(col)
    if g:
        groups.setdefault(g, []).append(ci)
    else:
        ungrouped_idx.append(ci)

# =============================================================================
# STAGE 1 — fit per-group PCA; classify as "kept" or "pooled"
# =============================================================================
print(f"\nGrouping (threshold = {COMPRESSION_THRESHOLD:.0%}) ...")

group_info    = {}   # {label: {n, var, kept, top}}
kept_pcs      = []   # arrays added to stage-1 output
kept_labels   = []
pooled_idx    = list(ungrouped_idx)   # column indices fed to stage-2 PCA

for label, idxs in groups.items():
    Xg       = X_scaled[:, idxs]
    pca_g    = PCA(n_components=1, random_state=3)
    pc1      = pca_g.fit_transform(Xg).ravel()
    var_expl = float(pca_g.explained_variance_ratio_[0])

    feat_names = [rdkit_cols[i].replace("rdkit-", "") for i in idxs]
    load_order = np.argsort(np.abs(pca_g.components_[0]))[::-1]
    top_loads  = [(feat_names[k],
                   float(pca_g.components_[0][k])) for k in load_order[:3]]

    kept = var_expl >= COMPRESSION_THRESHOLD
    group_info[label] = {"n": len(idxs), "var": var_expl,
                         "kept": kept, "top": top_loads}

    if kept:
        kept_pcs.append(pc1)
        kept_labels.append(f"PC1_{label}")
        tag = "KEPT  "
    else:
        pooled_idx.extend(idxs)   # send all member features to the pool
        tag = "pooled"

    print(f"  {tag}  {label:12s}  {len(idxs):3d} feat  "
          f"PC1 = {var_expl:.1%}  "
          f"top: {', '.join(n for n, _ in top_loads)}")

n_kept_groups = len(kept_pcs)
n_pool        = len(pooled_idx)
print(f"\n  Stage 1: {n_kept_groups} group PC1s")
print(f"  Stage 2 pool: {n_pool} features "
      f"({len(groups) - n_kept_groups} poorly-compressed groups "
      f"+ {len(ungrouped_idx)} ungrouped)")

# =============================================================================
# STAGE 2 — PCA on the pooled features
# =============================================================================
print("\nFitting PCA on pooled features ...")
X_pool    = X_scaled[:, pooled_idx]
scaler_p  = StandardScaler()
X_pool_sc = scaler_p.fit_transform(X_pool)

pca      = PCA(random_state=3)
X_pca    = pca.fit_transform(X_pool_sc)
cum_var  = np.cumsum(pca.explained_variance_ratio_)

for threshold in [0.80, 0.90, 0.95, 0.99]:
    n = int(np.searchsorted(cum_var, threshold)) + 1
    print(f"  {threshold:.0%} variance → {n} pool PCs "
          f"(+ {n_kept_groups} group PC1s = {n + n_kept_groups} total features)")

# =============================================================================
# FIGURE 1 — per-group summary
# =============================================================================
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.spines.top": False, "axes.spines.right": False,
})

C_KEPT   = "#2e7d32"   # green  — stage-1 groups
C_POOLED = "#c62828"   # red    — sent to stage-2 pool

labels_g  = list(group_info.keys())
n_feats   = [group_info[g]["n"]        for g in labels_g]
var_expls = [group_info[g]["var"] * 100 for g in labels_g]
colors_v  = [C_KEPT if group_info[g]["kept"] else C_POOLED for g in labels_g]

fig1, (ax_n, ax_v) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
x = np.arange(len(labels_g))

ax_n.bar(x, n_feats, color=colors_v, alpha=0.8)
ax_n.set_ylabel("Features in group", fontsize=11)
ax_n.set_title(
    f"Feature groups — green = stage-1 PC1 kept (≥{COMPRESSION_THRESHOLD:.0%}), "
    f"red = pooled for stage-2 PCA",
    fontsize=11,
)

ax_v.bar(x, var_expls, color=colors_v, alpha=0.8)
ax_v.axhline(COMPRESSION_THRESHOLD * 100, color="0.3", lw=1.2, ls="--")
ax_v.text(len(labels_g) - 0.5, COMPRESSION_THRESHOLD * 100 + 1,
          f"{COMPRESSION_THRESHOLD:.0%} threshold",
          ha="right", va="bottom", fontsize=10, color="0.3")
ax_v.set_ylabel("Within-group PC1 variance [%]", fontsize=11)
ax_v.set_xticks(x)
ax_v.set_xticklabels(labels_g, rotation=30, ha="right", fontsize=11)
ax_v.set_ylim(0, 110)

fig1.tight_layout()
fig1.savefig("pca_groups.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("Saved pca_groups.png")

# =============================================================================
# FIGURE 2 — scree + cumulative variance (stage-2 pool PCA)
# =============================================================================
N_SHOW = min(60, pca.n_components_)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.bar(np.arange(1, N_SHOW + 1),
        pca.explained_variance_ratio_[:N_SHOW] * 100,
        color="steelblue", alpha=0.8)
ax1.set_xlabel("Pool PC")
ax1.set_ylabel("Explained variance [%]")
ax1.set_title(
    f"Scree plot — stage-2 PCA on {n_pool} pooled features\n"
    f"(+ {n_kept_groups} group PC1s not shown here)"
)

ax2.plot(np.arange(1, pca.n_components_ + 1), cum_var * 100,
         color="steelblue", lw=2)
for thr, color in [(0.80, "#e07b39"), (0.90, "#c0392b"),
                   (0.95, "#8e44ad"), (0.99, "#2c3e50")]:
    n = int(np.searchsorted(cum_var, thr)) + 1
    ax2.axhline(thr * 100, color=color, lw=1, ls="--", alpha=0.8)
    ax2.axvline(n, color=color, lw=1, ls="--", alpha=0.8)
    ax2.text(n + 0.5, thr * 100 - 1.5,
             f"{thr:.0%} → {n} pool PCs", color=color, fontsize=10, va="top")

ax2.set_xlabel("Number of pool PCs")
ax2.set_ylabel("Cumulative variance of pool [%]")
ax2.set_title("Cumulative variance — stage-2 pool PCA")
ax2.set_xlim(0, pca.n_components_)
ax2.set_ylim(0, 101)

fig2.tight_layout()
fig2.savefig("pca_variance.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("Saved pca_variance.png")

# =============================================================================
# FIGURE 3 — first two pool PCs coloured by target
# =============================================================================
alphabet = list(string.ascii_lowercase)
pc1_vals = X_pca[:, 0]
pc2_vals = X_pca[:, 1]

fig3, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

for i, target in enumerate(TARGETS):
    ax     = axes[i]
    values = df_valid[target].values.astype(float)
    valid  = np.isfinite(values)

    ax.scatter(pc1_vals[~valid], pc2_vals[~valid], s=4, color="lightgray",
               linewidths=0, alpha=0.4, zorder=1, rasterized=True)
    sc = ax.scatter(pc1_vals[valid], pc2_vals[valid], c=values[valid],
                    s=8, cmap="viridis", alpha=0.7,
                    linewidths=0, zorder=2, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(target_labels[target], fontsize=12)
    ax.set_xlabel(
        f"Pool PC1  ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(
        f"Pool PC2  ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.text(-0.15, 1.04, alphabet[i], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

fig3.suptitle(
    f"Stage-2 pool PCA — PC1 vs PC2\n"
    f"({n_kept_groups} group PC1s extracted beforehand; "
    f"{n_pool} features fed into pool)",
    fontsize=12, y=1.01,
)
fig3.tight_layout()
fig3.savefig("pca_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig3)
print("Saved pca_scatter.png")
