#!/usr/bin/env python
# coding: utf-8
"""
feature_plots.py

Figure 1 — 3×3 grid: scatter plot of the top RDKit feature (by mean XGBoost
            gain importance across all 25 models) vs the target property for
            each of the 9 targets.

Figure 2 — top 3 features for viscosity shown side-by-side.
"""
import pickle
import string
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
DATA_CSV   = "../data/SurfPro-MD.csv"
MODELS_PKL = "models.pkl"

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]
target_labels = {
    "pCMC":                "pCMC",
    "AW_ST_CMC":           r"$\gamma_{\mathrm{CMC}}$  [mN m$^{-1}$]",
    "Gamma_max":           r"$\Gamma_{\max}$  [$\mu$mol m$^{-2}$]",
    "pC20":                "pC$_{20}$",
    "Area_min":            r"A$_{\min}$  [nm$^2$]",
    "viscosity":           r"$\eta$  [mPa·s]",
    "D_MOL":               r"D$_{\mathrm{MOL}}$  [m$^2$s$^{-1}$]",
    "D_SOL":               r"D$_{\mathrm{SOL}}$  [m$^2$s$^{-1}$]",
    "surface_tension_avg": r"$\gamma$  [mN m$^{-1}$]",
}
DISPLAY_SCALE = {"viscosity": 1e3}

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
    return df.dropna(subset=[target])

# =============================================================================
# LOAD
# =============================================================================
print("Loading data and models ...")
df = pd.read_csv(DATA_CSV)
with open(MODELS_PKL, "rb") as f:
    results = pickle.load(f)

# =============================================================================
# COMPUTE AVERAGE GAIN IMPORTANCE PER TARGET
# =============================================================================
def avg_gain_importance(results, target):
    """
    Average XGBoost gain importance across all 25 fold models.
    Returns (feature_names list, importance array).
    Feature names come from the stored 'features' key; the Booster uses f0/f1/…
    so we map f{i} -> feature_names[i].
    """
    feature_names = results[target][0]["fold_models"][0]["features"]
    n = len(feature_names)
    total = np.zeros(n)
    count = 0
    for tid in range(5):
        for fold_model in results[target][tid]["fold_models"]:
            booster = fold_model["model"]
            scores  = booster.get_score(importance_type="gain")
            for key, val in scores.items():
                total[int(key[1:])] += val   # 'f7' -> 7
            count += 1
    return feature_names, total / count


print("Computing feature importances ...")
importances = {}
for target in TARGETS:
    feat_names, imp = avg_gain_importance(results, target)
    order = np.argsort(imp)[::-1]
    importances[target] = {
        "names": np.array(feat_names)[order],
        "imp":   imp[order],
    }
    top3 = ", ".join(importances[target]["names"][:3])
    cum50 = np.searchsorted(np.cumsum(imp[order]) / imp.sum(), 0.5) + 1
    print(f"  {target}: top feature = {importances[target]['names'][0]} "
          f"| features to reach 50% gain = {cum50}")

# =============================================================================
# FIGURE 1 — 3×3 grid: top feature vs property
# =============================================================================
plt.rcParams.update({
    "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.spines.top": False, "axes.spines.right": False,
})
alphabet = list(string.ascii_lowercase)
cmap     = plt.get_cmap("viridis")

fig1, axes = plt.subplots(3, 3, figsize=(12, 11))
axes = axes.flatten()

for i, target in enumerate(TARGETS):
    ax      = axes[i]
    scale   = DISPLAY_SCALE.get(target, 1.0)
    top_feat = importances[target]["names"][0]
    top_disp = top_feat.replace("rdkit-", "")

    df_t = filter_outliers(df.copy(), target)
    x    = df_t[top_feat].values
    y    = df_t[target].values * scale

    # drop rows where either is NaN
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]

    r, _  = pearsonr(x, y)
    ax.scatter(x, y, s=8, alpha=0.5, color=cmap(0.65), linewidths=0)
    ax.text(0.97, 0.97, f"r = {r:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10)

    ax.set_title(target_labels[target], fontsize=12)
    ax.set_xlabel(top_disp, fontsize=10)
    ax.set_ylabel(target_labels[target], fontsize=10)
    ax.text(-0.15, 1.04, alphabet[i], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

fig1.suptitle("Top RDKit feature vs target property (Pearson r)", fontsize=13, y=1.01)
fig1.tight_layout()
fig1.savefig("top_feature_correlation.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("Saved top_feature_correlation.png")

# =============================================================================
# FIGURE 2 — top 3 features for viscosity
# =============================================================================
target  = "viscosity"
scale   = DISPLAY_SCALE[target]
df_visc = filter_outliers(df.copy(), target)
y_visc  = df_visc[target].values * scale

top6_names = importances[target]["names"][:6]
top6_imp   = importances[target]["imp"][:6]
total_imp  = importances[target]["imp"].sum()
cum_pct    = np.cumsum(top6_imp) / total_imp * 100

fig2, axes2 = plt.subplots(2, 3, figsize=(13, 9))
axes2 = axes2.flatten()

for j, feat in enumerate(top6_names):
    ax    = axes2[j]
    disp  = feat.replace("rdkit-", "")
    x     = df_visc[feat].values
    valid = np.isfinite(x) & np.isfinite(y_visc)
    xv, yv = x[valid], y_visc[valid]
    r, _   = pearsonr(xv, yv)

    ax.scatter(xv, yv, s=12, alpha=0.6, color=cmap(0.65), linewidths=0)
    ax.text(0.97, 0.97, f"r = {r:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11)
    ax.set_xlabel(disp, fontsize=12)
    ax.set_ylabel(r"$\eta$  [mPa·s]", fontsize=12)
    ax.set_title(
        f"{disp}\n"
        f"gain: {top6_imp[j]/total_imp*100:.1f}%  (cumul. {cum_pct[j]:.1f}%)",
        fontsize=10,
    )
    ax.text(-0.15, 1.04, alphabet[j], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

fig2.suptitle("Viscosity: top 6 features by XGBoost gain importance",
              fontsize=13, y=1.02)
fig2.tight_layout()
fig2.savefig("viscosity_top6_features.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("Saved viscosity_top6_features.png")

# =============================================================================
# FIGURE 3 — correlation matrix of top 6 viscosity features
# =============================================================================
disp_names = [f.replace("rdkit-", "") for f in top6_names]

# collect valid rows (no NaN in any of the 6 features)
feat_data = df_visc[top6_names].dropna()
corr = feat_data.corr(method="pearson")

fig3, ax3 = plt.subplots(figsize=(6, 5))
im = ax3.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

ax3.set_xticks(range(6))
ax3.set_yticks(range(6))
ax3.set_xticklabels(disp_names, rotation=45, ha="right", fontsize=11)
ax3.set_yticklabels(disp_names, fontsize=11)

for row in range(6):
    for col in range(6):
        val = corr.values[row, col]
        color = "white" if abs(val) > 0.7 else "black"
        ax3.text(col, row, f"{val:.2f}", ha="center", va="center",
                 fontsize=10, color=color)

cbar = fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", fontsize=11)

ax3.set_title("Feature correlation matrix — viscosity top 6", fontsize=12)
fig3.tight_layout()
fig3.savefig("viscosity_feature_corr_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig3)
print("Saved viscosity_feature_corr_matrix.png")
