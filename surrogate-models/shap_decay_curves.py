#!/usr/bin/env python3
"""
shap_decay_curves.py — Plot ranked SHAP importance decay curves per target.

X-axis: feature rank (sorted by importance, highest first)
Y-axis: mean |SHAP| value

Two figures are produced:
  shap_decay_grid.png     — 3×3 grid, one panel per target (raw values, log y)
  shap_decay_overlay.png  — all 9 targets overlaid on one axes (normalised 0-1)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

CACHE_FILE = "shap_results_cache.pkl"
TOP_K      = 20   # shaded region showing "top-K" cut-off

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

sns.set_theme(context="notebook", style="whitegrid", font_scale=1.1)
plt.rcParams.update({"savefig.dpi": 200})

# ── Load cache ────────────────────────────────────────────────────────────────
print(f"Loading {CACHE_FILE}...")
with open(CACHE_FILE, "rb") as fh:
    mean_abs_shap = pickle.load(fh)

n_features = len(next(iter(mean_abs_shap.values())))
colors_9 = cm.tab10(np.linspace(0, 0.9, len(TARGETS)))

# ── Figure 1: 3×3 grid, raw log-scale ────────────────────────────────────────
fig1, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
axes = axes.flatten()

for i, target in enumerate(TARGETS):
    ax = axes[i]
    vals = mean_abs_shap[target].sort_values(ascending=False).values
    ranks = np.arange(1, n_features + 1)

    # Shaded region for top-K
    ax.axvspan(0.5, TOP_K + 0.5, color="#e3f2fd", zorder=0, label=f"top {TOP_K}")

    ax.plot(ranks, vals, color=colors_9[i], linewidth=1.5, zorder=2)
    ax.scatter(ranks[:TOP_K], vals[:TOP_K], s=18, color=colors_9[i],
               zorder=3, linewidths=0)

    ax.set_yscale("log")
    ax.set_xlim(0, n_features + 1)
    ax.set_title(TARGET_LABELS[target], fontsize=13)
    ax.set_xlabel("Feature rank", fontsize=10)
    ax.set_ylabel("Mean |SHAP|", fontsize=10)

    # Mark rank at which 50% and 80% of total importance is captured
    cumsum = np.cumsum(vals)
    total  = cumsum[-1]
    r50 = int(np.searchsorted(cumsum, 0.50 * total)) + 1
    r80 = int(np.searchsorted(cumsum, 0.80 * total)) + 1
    ax.axvline(r50, color="gray", lw=1, ls="--", alpha=0.7)
    ax.axvline(r80, color="gray", lw=1, ls=":",  alpha=0.7)
    ax.text(r50 + 1, ax.get_ylim()[1] if ax.get_yscale() == "linear" else vals[0],
            f"50%\n(r={r50})", va="top", fontsize=7, color="gray")
    ax.text(r80 + 1, vals[r80 - 1] if r80 <= len(vals) else vals[-1],
            f"80%\n(r={r80})", va="top", fontsize=7, color="gray")

    sns.despine(ax=ax)

# Shared legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_items = [
    Patch(color="#e3f2fd", label=f"Top-{TOP_K} features"),
    Line2D([0], [0], color="gray", lw=1, ls="--", label="50% cumulative importance"),
    Line2D([0], [0], color="gray", lw=1, ls=":",  label="80% cumulative importance"),
]
fig1.legend(handles=legend_items, loc="lower center", ncol=3,
            fontsize=10, bbox_to_anchor=(0.5, -0.025), framealpha=0.9)

fig1.suptitle(
    "SHAP feature importance decay per target\n"
    "(features sorted by importance; log y-scale)",
    fontsize=14, y=1.01,
)
plt.tight_layout()
plt.savefig("shap_decay_grid.png", bbox_inches="tight")
plt.close()
print("Saved: shap_decay_grid.png")

# ── Figure 2: overlay, normalised 0-1 ────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5.5))

ranks = np.arange(1, n_features + 1)
ax2.axvspan(0.5, TOP_K + 0.5, color="#e3f2fd", zorder=0, label=f"Top-{TOP_K} region")

for i, target in enumerate(TARGETS):
    vals = mean_abs_shap[target].sort_values(ascending=False).values
    vals_norm = vals / vals[0]   # normalise to max = 1
    ax2.plot(ranks, vals_norm, color=colors_9[i],
             linewidth=1.6, label=TARGET_LABELS[target], zorder=2)

ax2.set_xlim(0, n_features + 1)
ax2.set_ylim(bottom=0)
ax2.set_xlabel("Feature rank", fontsize=12)
ax2.set_ylabel("Normalised mean |SHAP|  (max = 1)", fontsize=12)
ax2.set_title(
    "SHAP importance decay — all targets overlaid\n"
    "(each target normalised to its own maximum)",
    fontsize=12,
)
ax2.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)
sns.despine(ax=ax2)
plt.tight_layout()
plt.savefig("shap_decay_overlay.png", bbox_inches="tight")
plt.close()
print("Saved: shap_decay_overlay.png")

# ── Print cumulative-importance rank table ────────────────────────────────────
print(f"\n{'Target':22s}  {'r(50%)':>7}  {'r(80%)':>7}  {'r(90%)':>7}  n_features")
print("-" * 62)
for target in TARGETS:
    vals   = mean_abs_shap[target].sort_values(ascending=False).values
    cumsum = np.cumsum(vals)
    total  = cumsum[-1]
    r50 = int(np.searchsorted(cumsum, 0.50 * total)) + 1
    r80 = int(np.searchsorted(cumsum, 0.80 * total)) + 1
    r90 = int(np.searchsorted(cumsum, 0.90 * total)) + 1
    print(f"  {target:20s}  {r50:7d}  {r80:7d}  {r90:7d}  {len(vals)}")
