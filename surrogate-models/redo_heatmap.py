#!/usr/bin/env python3
"""Regenerate shap_crossover_heatmap.png with readable y-axis labels."""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CACHE_FILE = "shap_results_cache.pkl"
TOP_K = 20
N_HEATMAP = 20

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]
TARGET_LABELS = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          r"$\gamma_{\rm CMC}$",
    "Gamma_max":          r"$\Gamma_{\max}$",
    "pC20":               r"pC$_{20}$",
    "Area_min":           r"A$_{\min}$",
    "viscosity":          r"$\eta$",
    "D_MOL":              r"D$_{\rm MOL}$",
    "D_SOL":              r"D$_{\rm SOL}$",
    "surface_tension_avg": r"$\gamma$",
}

sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
plt.rcParams.update({"savefig.dpi": 200})

# ── Load cached SHAP results ──────────────────────────────────────────────────
print(f"Loading {CACHE_FILE}...")
with open(CACHE_FILE, "rb") as fh:
    mean_abs_shap = pickle.load(fh)

# ── Rebuild matrices ──────────────────────────────────────────────────────────
shap_matrix      = pd.DataFrame(mean_abs_shap)
shap_matrix_norm = shap_matrix.div(shap_matrix.sum(axis=0), axis=1)

top_sets = {t: set(mean_abs_shap[t].nlargest(TOP_K).index) for t in TARGETS}
all_top  = set().union(*top_sets.values())
feature_counts = pd.Series(
    {f: sum(f in s for s in top_sets.values()) for f in all_top}
).sort_values(ascending=False)
multi_target = feature_counts[feature_counts >= 2]

# ── Build heatmap data ────────────────────────────────────────────────────────
n_show     = min(N_HEATMAP, len(multi_target))
top_shared = multi_target.head(n_show).index.tolist()

heat_data = shap_matrix_norm.loc[top_shared].copy()
heat_data = heat_data.div(heat_data.max(axis=0), axis=1)

# Embed target-count in label: "BCUT2D_MWLOW  ×8"
row_labels = [
    f"{f.replace('rdkit-', '')}  ×{int(feature_counts[f])}"
    for f in top_shared
]
col_labels = [TARGET_LABELS[t] for t in TARGETS]

# ── Figure ────────────────────────────────────────────────────────────────────
ROW_HEIGHT_IN = 0.46          # inches per feature row — enough to avoid clipping
fig_h = n_show * ROW_HEIGHT_IN + 2.5   # +2.5 for title, x labels, cbar
fig_w = 10

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

sns.heatmap(
    heat_data,
    ax=ax,
    cmap="viridis",
    xticklabels=col_labels,
    yticklabels=row_labels,
    linewidths=0.5,
    linecolor="#e8e8e8",
    cbar_kws={
        "label": "Relative importance\n(within-target, max-normalised)",
        "shrink": 0.45,
        "pad": 0.02,
    },
    vmin=0,
    vmax=1,
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, va="center")

ax.set_title(
    f"SHAP feature importance — features shared across ≥2 targets\n"
    f"(top-{TOP_K} per target;  {n_show} features shown;  ×N = number of targets)",
    fontsize=12,
    pad=10,
)

plt.tight_layout()
plt.savefig("shap_crossover_heatmap.png", bbox_inches="tight")
plt.close()
print(f"Saved: shap_crossover_heatmap.png  ({fig_w}×{fig_h:.1f} in)")
