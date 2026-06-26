#!/usr/bin/env python
# coding: utf-8
"""
model_comparison.py

Single bar chart comparing Butina-clustered models (models.pkl) vs
k-means-clustered models (models_fs.pkl).

4 bars per target property:
  Butina R²     |  Butina Spearman ρ  |  k-means R²  |  k-means Spearman ρ
Error bars show std across the 5 outer test splits.
Targets sorted by Butina R² (descending).
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODELS_BUTINA = "models.pkl"
MODELS_KMEANS = "models_fs.pkl"

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

# Colours: two shades each for Butina (purple family) and k-means (orange family)
C_B_R2  = "#3d2b8c"   # Butina R²       — dark purple
C_B_RHO = "#4a86c8"   # Butina ρ        — steel blue
C_K_R2  = "#b5390a"   # k-means R²      — dark orange-red
C_K_RHO = "#e8a020"   # k-means ρ       — amber

with open(MODELS_BUTINA, "rb") as f:
    res_b = pickle.load(f)
with open(MODELS_KMEANS, "rb") as f:
    res_k = pickle.load(f)

# Collect metrics
metrics = {}
for t in TARGETS:
    r2_b  = np.array([res_b[t][s]["metrics"]["r2"]       for s in range(5)])
    rho_b = np.array([res_b[t][s]["metrics"]["spearman"] for s in range(5)])
    r2_k  = np.array([res_k[t][s]["metrics"]["r2"]       for s in range(5)])
    rho_k = np.array([res_k[t][s]["metrics"]["spearman"] for s in range(5)])
    metrics[t] = dict(
        r2_b_mean=r2_b.mean(),   r2_b_std=r2_b.std(),
        rho_b_mean=rho_b.mean(), rho_b_std=rho_b.std(),
        r2_k_mean=r2_k.mean(),   r2_k_std=r2_k.std(),
        rho_k_mean=rho_k.mean(), rho_k_std=rho_k.std(),
    )

# Sort by Butina R² descending
targets_sorted = sorted(TARGETS, key=lambda t: -metrics[t]["r2_b_mean"])

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 13, "axes.grid": True,
    "grid.alpha": 0.35, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.bottom": True,
})

fig, ax = plt.subplots(figsize=(12, 5))

x     = np.arange(len(targets_sorted))
w     = 0.18
offs  = [-0.27, -0.09, 0.09, 0.27]

bar_specs = [
    ("r2_b",  C_B_R2,  "$R^2$  (Butina)"),
    ("rho_b", C_B_RHO, r"$\rho$  (Butina)"),
    ("r2_k",  C_K_R2,  "$R^2$  (k-means)"),
    ("rho_k", C_K_RHO, r"$\rho$  (k-means)"),
]

for (key, color, label), off in zip(bar_specs, offs):
    means = np.array([metrics[t][f"{key}_mean"] for t in targets_sorted])
    stds  = np.array([metrics[t][f"{key}_std"]  for t in targets_sorted])
    ax.bar(x + off, means, width=w, color=color, label=label, zorder=3)
    ax.errorbar(x + off, means, yerr=stds,
                fmt="none", color="k", capsize=3, lw=1.2, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels([target_labels[t] for t in targets_sorted], fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_ylim(-0.35, 1.05)
ax.set_axisbelow(True)
ax.xaxis.grid(False)
ax.axhline(0, color="0.4", lw=0.8, zorder=2)

# Group separators between targets
for xi in x[:-1] + 0.5:
    ax.axvline(xi, color="0.85", lw=0.8, zorder=1)

# Legend — two rows: R² pair / ρ pair
handles = [
    mpatches.Patch(color=C_B_R2,  label="$R^2$  (Butina)"),
    mpatches.Patch(color=C_K_R2,  label="$R^2$  (k-means)"),
    mpatches.Patch(color=C_B_RHO, label=r"$\rho$  (Butina)"),
    mpatches.Patch(color=C_K_RHO, label=r"$\rho$  (k-means)"),
]
ax.legend(handles=handles, ncol=2, fontsize=11,
          loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved model_comparison.png")
