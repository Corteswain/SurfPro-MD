#!/usr/bin/env python3
"""
shap_overlap_analysis.py — Cross-target SHAP feature importance analysis
for the SurfPro-MD surrogate models.

Computes population-level mean |SHAP| for each of the 9 predicted targets
using the held-out test sets from all 5 outer splits × 5 CV folds.
Identifies features that rank as highly important for multiple targets and
produces five publication-quality figures:

  shap_feature_count_dist.png  — how many features appear in top-K for N targets
  shap_crossover_heatmap.png   — heat-map of shared features across all targets
  shap_per_target_ranked.png   — per-target top-15 bars, colour-coded by sharing
  shap_presence_matrix.png     — binary presence grid (feature × target)
  shap_universal_features.png  — grouped bar chart for the most universal features

Results are cached in shap_results_cache.pkl so reruns are fast.
"""

import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import sem

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import shap

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ─── Constants (must match surrogate.py) ─────────────────────────────────────

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

N_FOLDS            = 5
N_TEST_SPLITS      = 5
MIN_TEST_FRACTION  = 0.07
MAX_TEST_FRACTION  = 0.11
MAX_REJECT_TRIES   = 5
MIN_VAL_FRACTION   = 0.12
MAX_VAL_FRACTION   = 0.18
RANDOM_STATE       = 3
NN_ISOLATION_THRESHOLD = 50.0

TOP_K = 20         # top-K per target for overlap counting
N_HEATMAP = 40     # max features to show in heatmap
N_PRESENCE = 30    # max features in presence matrix
N_PER_TARGET = 15  # top features shown per-target bar chart

MODELS_FILE = "models.pkl"
DATA_FILE   = "../data/SurfPro-MD.csv"
CACHE_FILE  = "shap_results_cache.pkl"

# ─── Plotting style ───────────────────────────────────────────────────────────

sns.set_theme(context="notebook", style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 200,
    "figure.dpi": 100,
})

# ─── Data helpers (mirror of surrogate.py) ───────────────────────────────────

def _sample_test_clusters(clusters, cluster_sizes, min_size, max_size, rng,
                           max_rejects=5):
    available = set(clusters)
    test_clusters = set()
    test_count = 0
    streak = 0
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
    min_test = int(total * MIN_TEST_FRACTION)
    max_test = int(total * MAX_TEST_FRACTION)
    cs_all   = df_t.groupby("cluster").size().to_dict()
    clusters = np.array(list(cs_all.keys()))
    for ts in range(N_TEST_SPLITS):
        rng = np.random.RandomState(RANDOM_STATE + ts)
        tc, _ = _sample_test_clusters(clusters, cs_all, min_test, max_test, rng, MAX_REJECT_TRIES)
        col_test = f"{target_col}_test_split_{ts}"
        df_t[col_test] = "train_val"
        df_t.loc[df_t["cluster"].isin(tc), col_test] = "test"
        rem = df_t[df_t[col_test] != "test"]
        rc  = np.array(rem["cluster"].unique())
        rs  = rem.groupby("cluster").size().to_dict()
        rm  = len(rem)
        min_val = int(rm * MIN_VAL_FRACTION)
        max_val = int(rm * MAX_VAL_FRACTION)
        for fi in range(N_FOLDS):
            rng_f = np.random.RandomState(RANDOM_STATE + ts * 100 + fi)
            vc, _ = _sample_test_clusters(rc, rs, min_val, max_val, rng_f, MAX_REJECT_TRIES)
            tr_c  = set(rc) - set(vc)
            col   = f"{target_col}_split_t{ts}_f{fi}"
            df_t[col] = "unused"
            df_t.loc[df_t["cluster"].isin(tr_c), col] = "train"
            df_t.loc[df_t["cluster"].isin(vc),   col] = "val"
            df_t.loc[df_t[col_test] == "test",   col] = "test"
    return df_t


def filter_outliers(df, target):
    filters = {
        "surface_tension_avg": lambda d: d["surface_tension_avg"] >= 250,
        "viscosity":           lambda d: d["viscosity"] <= 0.003,
        "AW_ST_CMC":           lambda d: d["AW_ST_CMC"] <= 52,
        "Gamma_max":           lambda d: d["Gamma_max"] <= 6,
        "Area_min":            lambda d: d["Area_min"] <= 4.2,
        "pC20":                lambda d: d["pC20"] >= 1.8,
        "D_MOL":               lambda d: d["D_MOL"] <= 0.8,
    }
    if target in filters:
        mask = df[target].notnull() & filters[target](df)
        return df[mask]
    return df


def filter_nn_isolated(df, target, feature_cols, threshold=NN_ISOLATION_THRESHOLD):
    X = df[feature_cols].to_numpy().astype(float)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    nn.fit(X_sc)
    d = nn.kneighbors(X_sc)[0][:, 1]
    return df.loc[df.index[d <= threshold]]


# ─── SHAP helper (XGBoost 2.x / SHAP version fix) ────────────────────────────

def make_tree_explainer(booster):
    """Wrap save_raw to strip bracket-notation from base_score before SHAP reads it."""
    _orig = booster.save_raw

    def _patched(raw_format="ubj"):
        raw = _orig(raw_format=raw_format)
        if raw_format == "ubj":
            def _unwrap(m):
                val   = float(m.group(1))
                fixed = f"{val:.10g}".encode()
                pad   = len(m.group(0)) - len(fixed)
                return fixed + b" " * pad if pad >= 0 else fixed[:len(m.group(0))]
            raw = re.sub(rb"\[(-?[\d.]+(?:[Ee][+-]?\d+)?)\]", _unwrap, raw)
        return raw

    booster.save_raw = _patched
    try:
        explainer = shap.TreeExplainer(booster)
    finally:
        booster.save_raw = _orig
    return explainer


# ─── 1. Load data ─────────────────────────────────────────────────────────────

print("Loading dataset and computing Butina clusters...")
df = pd.read_csv(DATA_FILE)
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

# ─── 2. Load models ───────────────────────────────────────────────────────────

print(f"\nLoading models from {MODELS_FILE}...")
with open(MODELS_FILE, "rb") as fh:
    results = pickle.load(fh)

# ─── 3. Compute SHAP per target (or load cache) ───────────────────────────────

if os.path.exists(CACHE_FILE):
    print(f"\nLoading cached SHAP results from {CACHE_FILE}...")
    with open(CACHE_FILE, "rb") as fh:
        mean_abs_shap = pickle.load(fh)
    print("  Cache loaded.")
else:
    print("\nComputing population-level SHAP values (test sets, all splits & folds)...")
    mean_abs_shap = {}   # target → pd.Series(feature → mean |SHAP|)

    for target in TARGETS:
        print(f"\n  [{target}]")

        df_t = create_cluster_splits(df, target)
        df_t = filter_outliers(df_t, target)
        # Use feature list from stored model to guarantee alignment
        feature_cols = results[target][0]["feature_schema"]
        df_t = filter_nn_isolated(df_t, target, feature_cols)

        pool_shap = []   # list of (n_test_molecules, n_features) arrays

        for ts in range(N_TEST_SPLITS):
            run         = results[target][ts]
            split_col   = run["split_column"]
            fold_models = run["fold_models"]

            test_mask = df_t[split_col] == "test"
            X_test = df_t.loc[test_mask, feature_cols].to_numpy().astype(float)

            # Impute NaN features with column means
            col_means = np.nanmean(X_test, axis=0)
            for j in range(X_test.shape[1]):
                X_test[np.isnan(X_test[:, j]), j] = col_means[j]

            for fi, fm in enumerate(fold_models):
                X_sc = fm["scaler"].transform(X_test)
                expl = make_tree_explainer(fm["model"])
                sv   = expl.shap_values(X_sc)  # (n_test, n_features)
                pool_shap.append(np.abs(sv))
                print(f"    split {ts}, fold {fi}: {X_sc.shape[0]} test molecules", flush=True)

        stacked = np.vstack(pool_shap)  # (total_rows, n_features)
        mean_abs_shap[target] = pd.Series(stacked.mean(axis=0), index=feature_cols)
        print(f"  → pooled {stacked.shape[0]} rows, {stacked.shape[1]} features")

    print(f"\nSaving SHAP cache to {CACHE_FILE}...")
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(mean_abs_shap, fh)

# ─── 4. Build cross-target importance matrix ──────────────────────────────────

feature_cols = results[TARGETS[0]][0]["feature_schema"]

# Full matrix: (n_features, n_targets), normalised per target (fraction of sum)
shap_matrix      = pd.DataFrame(mean_abs_shap)           # (n_features, n_targets)
shap_matrix_norm = shap_matrix.div(shap_matrix.sum(axis=0), axis=1)

# Top-K sets per target
top_sets = {t: set(mean_abs_shap[t].nlargest(TOP_K).index) for t in TARGETS}

# Count how many targets each feature is in the top-K for
all_top = set().union(*top_sets.values())
feature_counts = pd.Series(
    {f: sum(f in s for s in top_sets.values()) for f in all_top}
).sort_values(ascending=False)

multi_target   = feature_counts[feature_counts >= 2]
shared_strong  = set(feature_counts[feature_counts >= 3].index)   # ≥3 targets
shared_mild    = set(feature_counts[feature_counts == 2].index)   # exactly 2 targets
specific       = set(feature_counts[feature_counts == 1].index)   # target-specific

print(f"\nFeature overlap summary (top-{TOP_K} per target):")
print(f"  Features appearing in ≥3 targets : {len(shared_strong)}")
print(f"  Features appearing in exactly 2  : {len(shared_mild)}")
print(f"  Features appearing in only 1     : {len(specific)}")
print(f"  Total unique features across all : {len(all_top)}")

print(f"\nTop shared features (≥2 targets):")
print(f"{'Feature':45s} {'N':>3}  Targets")
print("-" * 80)
for feat, cnt in multi_target.head(40).items():
    which = [TARGET_LABELS[t] for t in TARGETS if feat in top_sets[t]]
    fname = feat.replace("rdkit-", "")
    print(f"  {fname:43s} {cnt:3d}  {', '.join(which)}")

# ─── Figure 1: count distribution ────────────────────────────────────────────

print("\nGenerating Figure 1: feature count distribution...")

cnt_dist = feature_counts.value_counts().sort_index()
bar_colors = cm.viridis(np.linspace(0.15, 0.85, len(cnt_dist)))

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
bars = ax1.bar(cnt_dist.index, cnt_dist.values, color=bar_colors,
               edgecolor="white", linewidth=0.8)
for x, y in zip(cnt_dist.index, cnt_dist.values):
    ax1.text(x, y + 0.4, str(y), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_xlabel(f"Number of targets a feature ranks in top-{TOP_K}", fontsize=12)
ax1.set_ylabel("Number of features", fontsize=12)
ax1.set_title(
    f"Cross-target feature sharing\n(top-{TOP_K} SHAP importance per target, {len(TARGETS)} targets)",
    fontsize=12,
)
ax1.set_xticks(cnt_dist.index)
ax1.set_xticklabels([str(i) for i in cnt_dist.index])
sns.despine(ax=ax1, left=False)
plt.tight_layout()
plt.savefig("shap_feature_count_dist.png", bbox_inches="tight")
plt.close()
print("  Saved: shap_feature_count_dist.png")

# ─── Figure 2: heatmap of shared features ────────────────────────────────────

print("Generating Figure 2: cross-target heatmap...")

n_show = min(N_HEATMAP, len(multi_target))
top_shared = multi_target.head(n_show).index.tolist()

# Normalise per target so max within each target = 1
heat_data = shap_matrix_norm.loc[top_shared].copy()
heat_data  = heat_data.div(heat_data.max(axis=0), axis=1)

row_labels = [f.replace("rdkit-", "") for f in top_shared]
col_labels = [TARGET_LABELS[t] for t in TARGETS]

fig_h = max(7, n_show * 0.28)
fig2, ax2 = plt.subplots(figsize=(10, fig_h))
sns.heatmap(
    heat_data,
    ax=ax2,
    cmap="viridis",
    xticklabels=col_labels,
    yticklabels=row_labels,
    linewidths=0.4,
    linecolor="#e0e0e0",
    cbar_kws={"label": "Relative importance\n(within-target, max-normalised)", "shrink": 0.55},
    vmin=0, vmax=1,
)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=11)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=8.5)

# Annotate the count of targets each feature is shared in
for row_i, feat in enumerate(top_shared):
    cnt = int(feature_counts[feat])
    ax2.text(-0.55, row_i + 0.5, f"×{cnt}", ha="right", va="center",
             fontsize=7.5, color="#555555", transform=ax2.transData)

ax2.set_title(
    f"SHAP feature importance — features shared across ≥2 targets\n"
    f"(top-{TOP_K} per target; {n_show} features shown)",
    fontsize=12, pad=12,
)
plt.tight_layout()
plt.savefig("shap_crossover_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved: shap_crossover_heatmap.png")

# ─── Figure 3: per-target top-N bar charts, colour-coded by sharing ──────────

print("Generating Figure 3: per-target ranked features...")

# Colours: blue = shared ≥3, teal = shared 2, grey = specific
COLOUR_STRONG   = "#1565c0"   # dark blue  — shared ≥3 targets
COLOUR_MILD     = "#4fc3f7"   # light blue — shared in 2 targets
COLOUR_SPECIFIC = "#b0bec5"   # grey       — target-specific

fig3, axes3 = plt.subplots(3, 3, figsize=(17, 15))
axes3 = axes3.flatten()

for i, target in enumerate(TARGETS):
    ax = axes3[i]
    top_f   = mean_abs_shap[target].nlargest(N_PER_TARGET)
    names   = [f.replace("rdkit-", "") for f in top_f.index]
    vals    = top_f.values
    y_pos   = np.arange(N_PER_TARGET)

    colors = []
    for feat in top_f.index:
        cnt = feature_counts.get(feat, 0)
        if cnt >= 3:
            colors.append(COLOUR_STRONG)
        elif cnt == 2:
            colors.append(COLOUR_MILD)
        else:
            colors.append(COLOUR_SPECIFIC)

    ax.barh(y_pos[::-1], vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_title(TARGET_LABELS[target], fontsize=12)
    ax.set_xlabel("Mean |SHAP|", fontsize=9)
    sns.despine(ax=ax)

legend_patches = [
    mpatches.Patch(color=COLOUR_STRONG,   label=f"Shared ≥3 targets (top-{TOP_K})"),
    mpatches.Patch(color=COLOUR_MILD,     label=f"Shared 2 targets  (top-{TOP_K})"),
    mpatches.Patch(color=COLOUR_SPECIFIC, label="Target-specific"),
]
fig3.legend(handles=legend_patches, loc="lower center", ncol=3,
            fontsize=10, bbox_to_anchor=(0.5, -0.025),
            framealpha=0.9, edgecolor="lightgray")
fig3.suptitle(
    f"Top-{N_PER_TARGET} SHAP features per target  (colour = cross-target sharing)",
    fontsize=14, y=1.005,
)
plt.tight_layout()
plt.savefig("shap_per_target_ranked.png", bbox_inches="tight")
plt.close()
print("  Saved: shap_per_target_ranked.png")

# ─── Figure 4: binary presence matrix ────────────────────────────────────────

print("Generating Figure 4: presence matrix...")

n_bin = min(N_PRESENCE, len(multi_target))
top_bin   = multi_target.head(n_bin).index.tolist()
bin_names = [f.replace("rdkit-", "") for f in top_bin]
col_labels = [TARGET_LABELS[t] for t in TARGETS]

binary = np.array([[1 if feat in top_sets[t] else 0 for t in TARGETS]
                    for feat in top_bin])

fig4_h = max(6, n_bin * 0.32)
fig4, ax4 = plt.subplots(figsize=(9, fig4_h))

cmap_bin = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "bin", ["#f5f5f5", "#1565c0"]
)
ax4.imshow(binary, cmap=cmap_bin, aspect="auto", vmin=0, vmax=1)

ax4.set_xticks(np.arange(len(TARGETS)))
ax4.set_yticks(np.arange(n_bin))
ax4.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=11)
ax4.set_yticklabels(bin_names, fontsize=8.5)

# White grid lines
for x in np.arange(-0.5, len(TARGETS), 1):
    ax4.axvline(x, color="white", lw=1.8)
for y in np.arange(-0.5, n_bin, 1):
    ax4.axhline(y, color="white", lw=0.9)

# Mark "present" cells with a dot for clarity
for ri in range(n_bin):
    for ci in range(len(TARGETS)):
        if binary[ri, ci]:
            ax4.text(ci, ri, "●", ha="center", va="center",
                     fontsize=9, color="white", alpha=0.85)

# Count annotation on the right
for ri in range(n_bin):
    cnt = int(binary[ri].sum())
    ax4.text(len(TARGETS) - 0.5 + 0.6, ri, str(cnt),
             ha="left", va="center", fontsize=8.5, color="#333333")
ax4.text(len(TARGETS) - 0.5 + 0.6, -0.8, "#", ha="left", va="center",
         fontsize=8.5, color="#555555", style="italic")

ax4.set_title(
    f"Feature presence in top-{TOP_K} SHAP importance per target\n"
    f"({n_bin} shared features shown, ordered by target count)",
    fontsize=12, pad=10,
)
plt.tight_layout()
plt.savefig("shap_presence_matrix.png", bbox_inches="tight")
plt.close()
print("  Saved: shap_presence_matrix.png")

# ─── Figure 5: universal features — grouped bar chart ────────────────────────

print("Generating Figure 5: universal features grouped bar chart...")

n_univ   = min(20, len(multi_target))
top_univ = multi_target.head(n_univ).index.tolist()
univ_names = [f.replace("rdkit-", "") for f in top_univ]

target_colors = cm.tab10(np.linspace(0, 0.9, len(TARGETS)))

fig5, ax5 = plt.subplots(figsize=(11, 8))

n_t = len(TARGETS)
bar_h = 0.7 / n_t
y_positions = np.arange(n_univ)

for j, target in enumerate(TARGETS):
    vals = shap_matrix_norm.loc[top_univ, target].values
    y_off = y_positions + (j - n_t / 2 + 0.5) * bar_h
    ax5.barh(y_off, vals, height=bar_h * 0.92,
             color=target_colors[j], label=TARGET_LABELS[target], alpha=0.88)

ax5.set_yticks(y_positions)
ax5.set_yticklabels(univ_names, fontsize=9)
ax5.set_xlabel("Normalised SHAP importance (fraction of per-target total)", fontsize=11)
ax5.set_title(
    f"Most cross-target important features\n"
    f"(top-{n_univ} features by number of targets in top-{TOP_K})",
    fontsize=12,
)
ax5.legend(loc="lower right", fontsize=9, ncol=2,
           framealpha=0.9, edgecolor="lightgray")
sns.despine(ax=ax5)
plt.tight_layout()
plt.savefig("shap_universal_features.png", bbox_inches="tight")
plt.close()
print("  Saved: shap_universal_features.png")

# ─── 5. Print ranked summary table ───────────────────────────────────────────

print("\n" + "=" * 85)
print(f"CROSS-TARGET FEATURE IMPORTANCE SUMMARY  (top-{TOP_K} per target)")
print("=" * 85)
print(f"{'Feature':44s} {'N':>4}  {'Targets'}")
print("-" * 85)
for feat, cnt in feature_counts.items():
    which = [TARGET_LABELS.get(t, t) for t in TARGETS if feat in top_sets[t]]
    fname = feat.replace("rdkit-", "")
    print(f"  {fname:42s} {cnt:4d}  {', '.join(which)}")
print("=" * 85)

# ─── Per-target top-5 table ───────────────────────────────────────────────────

print(f"\nPER-TARGET TOP-5 FEATURES")
print("-" * 70)
for target in TARGETS:
    top5 = mean_abs_shap[target].nlargest(5)
    print(f"\n  {TARGET_LABELS[target]} ({target})")
    for rank, (feat, val) in enumerate(top5.items(), 1):
        cnt  = int(feature_counts.get(feat, 0))
        flag = f"[shared {cnt}/9]" if cnt > 1 else ""
        print(f"    {rank}.  {feat.replace('rdkit-',''):38s}  {val:.5f}  {flag}")

print("\n=== Analysis complete. Five figures saved to surrogate-models/ ===")
