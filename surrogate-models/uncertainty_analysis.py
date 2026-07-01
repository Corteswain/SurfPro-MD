#!/usr/bin/env python3
"""
uncertainty_analysis.py — compare three uncertainty-estimation strategies.

Generates three publication-quality 3×3 figures (one panel per target):

  unc_error_vs_nndist.pdf          — |error| vs nearest-neighbour distance to train
  unc_error_vs_ens_std.pdf         — |error| vs ensemble prediction std
  unc_conformal_calibration.pdf    — conformal prediction calibration curves
                                     (nominal coverage vs empirical coverage)

Conformal prediction is implemented as split/cross-conformal regression:
  - For each (target, test_split): pool the residuals |ŷ_val − y_val| from
    all 5 validation folds as calibration scores.
  - For nominal coverage (1−α), the guaranteed interval half-width is
      q = quantile(cal_scores, (1−α) · (1 + 1/n_cal))
  - The calibration curve plots this nominal vs empirical coverage.

Usage
-----
  python uncertainty_analysis.py
  python uncertainty_analysis.py --model models.pkl --data ../data/SurfPro-MD.csv
"""

import argparse
import os
import pickle
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models.pkl")
parser.add_argument("--data",  default="../data/SurfPro-MD.csv")
args = parser.parse_args()

# ── Typography (matches plot_for_paper.py) ────────────────────────────────────

FONT_BASE  = 9
FONT_LABEL = 10
FONT_PANEL = 11
FONT_ANNOT = 8.5
DPI        = 300

matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          FONT_BASE,
    "axes.titlesize":     FONT_LABEL,
    "axes.labelsize":     FONT_LABEL,
    "xtick.labelsize":    FONT_BASE,
    "ytick.labelsize":    FONT_BASE,
    "legend.fontsize":    FONT_BASE,
    "lines.linewidth":    1.5,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "legend.frameon":     True,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.8",
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── Constants (must match surrogate.py) ──────────────────────────────────────

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

TARGET_LABELS = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          r"$\gamma_{\mathrm{CMC}}$",
    "Gamma_max":          r"$\Gamma_{\max}$",
    "pC20":               r"pC$_{20}$",
    "Area_min":           r"A$_{\min}$",
    "viscosity":          r"$\eta$",
    "D_MOL":              r"D$_{\mathrm{MOL}}$",
    "D_SOL":              r"D$_{\mathrm{SOL}}$",
    "surface_tension_avg": r"$\gamma$",
}

UNIT_LABELS = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          r"$\gamma_{\mathrm{CMC}}$ [mN m$^{-1}$]",
    "Gamma_max":          r"$\Gamma_{\max}$ [$\mu$mol m$^{-2}$]",
    "pC20":               r"pC$_{20}$",
    "Area_min":           r"A$_{\min}$ [nm$^2$]",
    "viscosity":          r"$\eta$ [mPa$\cdot$s]",
    "D_MOL":              r"D$_\mathrm{MOL}$ [$\times\!10^{-9}$ m$^2$s$^{-1}$]",
    "D_SOL":              r"D$_\mathrm{SOL}$ [$\times\!10^{-9}$ m$^2$s$^{-1}$]",
    "surface_tension_avg": r"$\gamma$ [mN m$^{-1}$]",
}

_DISPLAY_SCALE    = {"viscosity": 1e3, "surface_tension_avg": 0.1}
N_FOLDS           = 5
N_TEST_SPLITS     = 5
MIN_TEST_FRACTION = 0.07
MAX_TEST_FRACTION = 0.11
MAX_REJECT_TRIES  = 5
MIN_VAL_FRACTION  = 0.12
MAX_VAL_FRACTION  = 0.18
RANDOM_STATE      = 3
NN_ISOLATION_THRESHOLD = 50.0
NN_DIST_MAX       = 40

ALPHA_LEVELS = np.linspace(0.02, 0.98, 60)   # for calibration curves

ALPHABET = list(string.ascii_lowercase)
cmap_splits = cm.get_cmap("viridis", N_TEST_SPLITS)

# ── Data-prep helpers (exact copies from surrogate.py) ────────────────────────

def _sample_test_clusters(clusters, cluster_sizes, min_size, max_size, rng,
                           max_rejects=5):
    available = set(clusters)
    test_clusters, test_count, streak = set(), 0, 0
    while available:
        c = rng.choice(list(available))
        available.remove(c)
        new_size = test_count + cluster_sizes[c]
        if new_size < min_size:
            test_clusters.add(c); test_count = new_size; streak = 0; continue
        if min_size <= new_size <= max_size:
            test_clusters.add(c); test_count = new_size; break
        streak += 1
        if streak >= max_rejects:
            break
    return test_clusters, test_count


def create_cluster_splits(df_in, target_col):
    df_t = df_in.dropna(subset=[target_col, "cluster"]).copy()
    df_t["cluster"] = df_t["cluster"].astype(int)
    total = len(df_t)
    cs_all = df_t.groupby("cluster").size().to_dict()
    clusters = np.array(list(cs_all.keys()))
    for ts in range(N_TEST_SPLITS):
        rng = np.random.RandomState(RANDOM_STATE + ts)
        tc, _ = _sample_test_clusters(
            clusters, cs_all,
            int(total * MIN_TEST_FRACTION), int(total * MAX_TEST_FRACTION),
            rng, MAX_REJECT_TRIES,
        )
        col_test = f"{target_col}_test_split_{ts}"
        df_t[col_test] = "train_val"
        df_t.loc[df_t["cluster"].isin(tc), col_test] = "test"
        rem = df_t[df_t[col_test] != "test"]
        rc = np.array(rem["cluster"].unique())
        rs = rem.groupby("cluster").size().to_dict()
        rm = len(rem)
        for fi in range(N_FOLDS):
            rng_f = np.random.RandomState(RANDOM_STATE + ts * 100 + fi)
            vc, _ = _sample_test_clusters(
                rc, rs,
                int(rm * MIN_VAL_FRACTION), int(rm * MAX_VAL_FRACTION),
                rng_f, MAX_REJECT_TRIES,
            )
            col = f"{target_col}_split_t{ts}_f{fi}"
            df_t[col] = "unused"
            df_t.loc[df_t["cluster"].isin(set(rc) - set(vc)), col] = "train"
            df_t.loc[df_t["cluster"].isin(vc),               col] = "val"
            df_t.loc[df_t[col_test] == "test",               col] = "test"
    return df_t


def filter_outliers(df, target):
    rules = {
        "surface_tension_avg": lambda d: d["surface_tension_avg"] >= 250,
        "viscosity":           lambda d: d["viscosity"] <= 0.003,
        "AW_ST_CMC":           lambda d: d["AW_ST_CMC"] <= 52,
        "Gamma_max":           lambda d: d["Gamma_max"] <= 6,
        "Area_min":            lambda d: d["Area_min"] <= 4.2,
        "pC20":                lambda d: d["pC20"] >= 1.8,
        "D_MOL":               lambda d: d["D_MOL"] <= 0.8,
    }
    if target in rules:
        return df[df[target].notnull() & rules[target](df)]
    return df


def filter_nn_isolated(df, target, feature_cols,
                        threshold=NN_ISOLATION_THRESHOLD):
    X = df[feature_cols].to_numpy().astype(float)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]
    X_sc = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    nn.fit(X_sc)
    d = nn.kneighbors(X_sc)[0][:, 1]
    return df.loc[df.index[d <= threshold]]


# ── Load data & cluster ───────────────────────────────────────────────────────

print(f"Loading {args.data} …")
df = pd.read_csv(args.data)
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid  = df[df["mol"].notnull()].copy()

fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in df_valid["mol"]]
nfps = len(fps)
dists = []
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend(1 - s for s in sims)

clusters_butina = Butina.ClusterData(dists, nfps, distThresh=0.6, isDistData=True)
cid = np.zeros(nfps, dtype=int)
for ci, cl in enumerate(clusters_butina):
    for idx in cl:
        cid[idx] = ci
df_valid["cluster"] = cid.astype(int)
df["cluster"] = np.nan
df.loc[df_valid.index, "cluster"] = df_valid["cluster"].values
print(f"  {len(clusters_butina)} clusters, {nfps} molecules.")

# ── Load models ───────────────────────────────────────────────────────────────

print(f"\nLoading {args.model} …")
with open(args.model, "rb") as fh:
    results = pickle.load(fh)

# ── Compute diagnostics + conformal calibration ───────────────────────────────

print("\nComputing per-target diagnostics and conformal calibration …")

# Storage: per-target, per-test-split
data = {}   # data[target] = list of dicts, one per test_split

for target in TARGETS:
    print(f"  {target}", flush=True)
    df_t         = create_cluster_splits(df, target)
    df_t         = filter_outliers(df_t, target)
    feature_cols = results[target][0]["fold_models"][0]["features"]
    df_t         = filter_nn_isolated(df_t, target, feature_cols)
    scale        = _DISPLAY_SCALE.get(target, 1.0)

    splits = []
    for ts in range(N_TEST_SPLITS):
        run         = results[target][ts]
        fold_models = run["fold_models"]
        split_col   = run["split_column"]   # e.g. "viscosity_split_t0_f0"

        # ── Test set ─────────────────────────────────────────────────────────
        test_mask  = df_t[split_col] == "test"
        train_mask = ~test_mask
        X_test  = df_t.loc[test_mask,  feature_cols].to_numpy()
        X_train = df_t.loc[train_mask, feature_cols].to_numpy()
        y_test  = df_t.loc[test_mask,  target].to_numpy() * scale

        fold_preds = np.array([
            m["model"].inplace_predict(m["scaler"].transform(X_test)) * scale
            for m in fold_models
        ])
        y_pred = fold_preds.mean(axis=0)
        y_std  = fold_preds.std(axis=0)
        errors = np.abs(y_pred - y_test)

        # ── NN distance (test → train, in scaled feature space) ──────────────
        scaler     = fold_models[0]["scaler"]
        X_test_sc  = scaler.transform(X_test)
        X_train_sc = scaler.transform(X_train)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X_train_sc)
        nn_dist = nn.kneighbors(X_test_sc)[0][:, 0]

        # ── Conformal calibration scores (val folds) ──────────────────────────
        # Each fold model i was trained excluding its own val set.
        # col name: f"{target}_split_t{ts}_f{fi}"
        cal_errors = []
        for fi, fm in enumerate(fold_models):
            col_fi = f"{target}_split_t{ts}_f{fi}"
            if col_fi not in df_t.columns:
                continue
            val_mask = df_t[col_fi] == "val"
            if val_mask.sum() == 0:
                continue
            X_val = df_t.loc[val_mask, feature_cols].to_numpy()
            y_val = df_t.loc[val_mask, target].to_numpy() * scale

            y_val_pred = fm["model"].inplace_predict(
                fm["scaler"].transform(X_val)) * scale
            cal_errors.extend(np.abs(y_val_pred - y_val))

        cal_errors = np.array(cal_errors)
        n_cal      = len(cal_errors)

        nominal_coverages = 1.0 - ALPHA_LEVELS
        conf_empirical = []

        for alpha in ALPHA_LEVELS:
            level  = np.minimum(1.0, (1 - alpha) * (1 + 1 / n_cal))
            q_conf = np.quantile(cal_errors, level)
            conf_empirical.append(np.mean(errors <= q_conf))

        splits.append({
            "errors":         errors,
            "y_std":          y_std,
            "nn_dist":        nn_dist,
            "nominal":        nominal_coverages,
            "conf_empirical": np.array(conf_empirical),
        })

    data[target] = splits

print("  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: shared legend below a 3×3 figure
# ─────────────────────────────────────────────────────────────────────────────

def _add_split_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap_splits(tid), markersize=5,
               label=f"Split {tid + 1}")
        for tid in range(N_TEST_SPLITS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=N_TEST_SPLITS,
               fontsize=FONT_BASE, frameon=True,
               bbox_to_anchor=(0.5, -0.012),
               handletextpad=0.3, columnspacing=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — |error| vs NN distance
# ─────────────────────────────────────────────────────────────────────────────

print("\nPlotting unc_error_vs_nndist.pdf …")

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i, target in enumerate(TARGETS):
    ax     = axes.flatten()[i]
    splits = data[target]

    all_nn_dist = np.concatenate([s["nn_dist"] for s in splits])
    all_errors  = np.concatenate([s["errors"]  for s in splits])
    all_tids    = np.concatenate([[ts] * len(s["errors"]) for ts, s in enumerate(splits)])

    keep   = all_nn_dist <= NN_DIST_MAX
    n_excl = (~keep).sum()
    rho, _ = spearmanr(all_nn_dist[keep], all_errors[keep])

    for tid in range(N_TEST_SPLITS):
        mask = keep & (all_tids == tid)
        ax.scatter(all_nn_dist[mask], all_errors[mask],
                   s=8, alpha=0.50, color=cmap_splits(tid), linewidths=0,
                   rasterized=True)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(TARGET_LABELS[target], fontsize=FONT_LABEL, pad=3)
    ax.set_xlabel("NN distance (feature space)", fontsize=FONT_BASE)
    ax.set_ylabel(f"|error| [{UNIT_LABELS[target].split('[')[-1].rstrip(']') if '[' in UNIT_LABELS[target] else UNIT_LABELS[target]}]",
                  fontsize=FONT_BASE)
    ax.text(0.97, 0.97, fr"$\rho$ = {rho:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FONT_ANNOT)
    if n_excl:
        ax.text(0.97, 0.88, f"({n_excl} excl.)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="0.5")
    ax.text(-0.16, 1.05, ALPHABET[i], transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

fig.tight_layout(rect=[0, 0.03, 1, 1])
_add_split_legend(fig)
plt.savefig("unc_error_vs_nndist.pdf")
plt.close()
print("  Saved: unc_error_vs_nndist.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — |error| vs ensemble std
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting unc_error_vs_ens_std.pdf …")

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i, target in enumerate(TARGETS):
    ax     = axes.flatten()[i]
    splits = data[target]

    all_std    = np.concatenate([s["y_std"]  for s in splits])
    all_errors = np.concatenate([s["errors"] for s in splits])
    all_tids   = np.concatenate([[ts] * len(s["errors"]) for ts, s in enumerate(splits)])

    rho, _ = spearmanr(all_std, all_errors)
    lim    = max(all_std.max(), all_errors.max()) * 1.05

    for tid in range(N_TEST_SPLITS):
        mask = all_tids == tid
        ax.scatter(all_std[mask], all_errors[mask],
                   s=8, alpha=0.50, color=cmap_splits(tid), linewidths=0,
                   rasterized=True)

    ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(TARGET_LABELS[target], fontsize=FONT_LABEL, pad=3)
    unit_str = UNIT_LABELS[target].split('[')[-1].rstrip(']') if '[' in UNIT_LABELS[target] else ""
    unit_fmt = f" [{unit_str}]" if unit_str else ""
    ax.set_xlabel(f"Ensemble std{unit_fmt}", fontsize=FONT_BASE)
    ax.set_ylabel(f"|error|{unit_fmt}",      fontsize=FONT_BASE)
    ax.text(0.97, 0.03, fr"$\rho$ = {rho:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_ANNOT)
    ax.text(-0.16, 1.05, ALPHABET[i], transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

fig.tight_layout(rect=[0, 0.03, 1, 1])
_add_split_legend(fig)
plt.savefig("unc_error_vs_ens_std.pdf")
plt.close()
print("  Saved: unc_error_vs_ens_std.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Conformal calibration curves
# ─────────────────────────────────────────────────────────────────────────────
#
# Each panel: nominal coverage (x) vs empirical coverage (y) for one target.
# A well-calibrated method lies on the diagonal; under-coverage (curve below
# diagonal) reflects distributional shift between val folds and test clusters.

CONFORMAL_COLOR = cm.viridis(0.15)

print("Plotting unc_conformal_calibration.pdf …")

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i, target in enumerate(TARGETS):
    ax        = axes.flatten()[i]
    splits    = data[target]
    conf_mean = np.mean([s["conf_empirical"] for s in splits], axis=0)
    nominal   = splits[0]["nominal"]

    ax.plot(nominal, conf_mean, color=CONFORMAL_COLOR, lw=2.0, zorder=4)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k-", lw=0.8, alpha=0.5, zorder=5)

    # Reference lines at 90 %
    ax.axvline(0.9, color="0.6", lw=0.6, ls="--")
    ax.axhline(0.9, color="0.6", lw=0.6, ls="--")

    # Annotate empirical coverage at nominal = 90%
    idx90 = np.argmin(np.abs(nominal - 0.90))
    cov90 = conf_mean[idx90]
    ax.annotate(f"{cov90:.2f}", xy=(0.90, cov90), fontsize=7,
                color=CONFORMAL_COLOR,
                xytext=(0.76, cov90 - 0.08),
                arrowprops=dict(arrowstyle="-", color=CONFORMAL_COLOR, lw=0.7))

    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal coverage", fontsize=FONT_BASE)
    ax.set_ylabel("Empirical coverage", fontsize=FONT_BASE)
    ax.set_title(TARGET_LABELS[target], fontsize=FONT_LABEL, pad=3)
    ax.text(-0.16, 1.05, ALPHABET[i], transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight="bold", va="bottom")

legend_handles = [
    Line2D([0], [0], color=CONFORMAL_COLOR, lw=2.0, label="Conformal"),
    Line2D([0], [0], color="k",             lw=0.8, label="Perfect calibration"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=FONT_BASE, frameon=True,
           bbox_to_anchor=(0.5, -0.01),
           handletextpad=0.5, columnspacing=1.2)

fig.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("unc_conformal_calibration.pdf")
plt.close()
print("  Saved: unc_conformal_calibration.pdf")

print("\nAll done.")
