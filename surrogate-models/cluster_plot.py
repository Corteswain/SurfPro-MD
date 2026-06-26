#!/usr/bin/env python
# coding: utf-8
"""
cluster_plot.py

Visualise the cluster-based train/test splitting in 2-D (first two principal
components of standardised RDKit descriptors).

Left panel  — Butina clustering on Morgan fingerprints (Tanimoto distance)
Right panel — k-means clustering on standardised RDKit descriptors

Colour  = the lowest-index test split the molecule appears in (across all 9
          targets).  Grey = never assigned to any test set.

Marker  = number of distinct test-split IDs the molecule is assigned to
          across all 9 targets x 5 outer test splits:
            ·  (tiny dot)  0  — never in any test set
            o  (circle)    1  — appears in exactly one test split
            s  (square)    2  — appears in two different test splits
            ^  (triangle)  3
            D  (diamond)   4
            *  (star)      5
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION  (must match surrogate.py / surrogate_feature_space.py)
# =============================================================================
DATA_CSV          = "../data/SurfPro-MD.csv"
N_CLUSTERS_KM     = 90
RANDOM_STATE      = 3
N_TEST_SPLITS     = 5
MIN_TEST_FRACTION = 0.07
MAX_TEST_FRACTION = 0.11
MAX_REJECT_TRIES  = 5

TARGETS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]

MARKERS       = [".", "o", "s", "^", "D", "*"]
MARKER_SIZES  = [ 18,  30,  40,  40,  40,  55]   # one per marker level
MARKER_LABELS = [
    "Never in test",
    "1 test split", "2 test splits", "3 test splits",
    "4 test splits", "5 test splits",
]
SPLIT_COLORS  = cm.get_cmap("tab10")(np.linspace(0, 0.5, N_TEST_SPLITS))
MULTI_COLOR   = "#333333"   # dark grey for molecules in >1 test split

# =============================================================================
# HELPER — cluster-based split assignment
# =============================================================================
def _sample_test_clusters(clusters, cluster_sizes, min_size, max_size,
                           rng, max_rejects):
    available     = set(clusters)
    test_clusters = set()
    test_count    = 0
    reject_streak = 0
    while available:
        c = rng.choice(sorted(available))   # sorted for reproducibility
        available.remove(c)
        new_size = test_count + cluster_sizes[c]
        if new_size < min_size:
            test_clusters.add(c)
            test_count = new_size
            reject_streak = 0
            continue
        if min_size <= new_size <= max_size:
            test_clusters.add(c)
            break
        reject_streak += 1
        if reject_streak >= max_rejects:
            break
    return test_clusters


def get_test_assignments(df_with_clusters):
    """
    Returns a dict {df_row_index: set_of_test_split_ids} aggregated over all
    TARGETS.  A molecule can appear in up to N_TEST_SPLITS distinct splits if
    different targets assigned its cluster to different test rounds.
    """
    mol_to_splits = {idx: set() for idx in df_with_clusters.index}

    for target in TARGETS:
        df_t = df_with_clusters.dropna(subset=[target, "cluster"]).copy()
        df_t["cluster"] = df_t["cluster"].astype(int)

        total            = len(df_t)
        min_test         = int(total * MIN_TEST_FRACTION)
        max_test         = int(total * MAX_TEST_FRACTION)
        cluster_sizes    = df_t.groupby("cluster").size().to_dict()
        cluster_keys     = np.array(list(cluster_sizes.keys()))

        for split_id in range(N_TEST_SPLITS):
            rng           = np.random.RandomState(RANDOM_STATE + split_id)
            test_clusters = _sample_test_clusters(
                cluster_keys, cluster_sizes,
                min_test, max_test, rng, MAX_REJECT_TRIES,
            )
            for idx in df_t.index[df_t["cluster"].isin(test_clusters)]:
                mol_to_splits[idx].add(split_id)

    return mol_to_splits

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data ...")
df       = pd.read_csv(DATA_CSV)
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df_valid  = df[df["mol"].notnull()].copy()
n_valid   = len(df_valid)
print(f"  {n_valid} valid molecules, {len(df) - n_valid} skipped")

# =============================================================================
# RDKit DESCRIPTOR MATRIX  (shared for k-means and PCA)
# =============================================================================
rdkit_cols = sorted([c for c in df_valid.columns if c.startswith("rdkit-")])
X_raw      = df_valid[rdkit_cols].to_numpy().astype(float)
col_means  = np.nanmean(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    X_raw[np.isnan(X_raw[:, j]), j] = col_means[j]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# =============================================================================
# 2-D PCA PROJECTION  (shared axis for both panels)
# =============================================================================
print("PCA projection ...")
pca   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
var1, var2 = pca.explained_variance_ratio_
print(f"  PC1 {var1:.2%}  PC2 {var2:.2%}")

# =============================================================================
# CLUSTERING 1 — Butina / Morgan fingerprints
# =============================================================================
print("Butina clustering (Morgan FP, Tanimoto) ...")
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
       for m in df_valid["mol"]]
dists_tanimoto = []
for i in range(1, n_valid):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists_tanimoto.extend([1 - s for s in sims])

clusters_butina  = Butina.ClusterData(
    dists_tanimoto, n_valid, distThresh=0.6, isDistData=True)
cid_butina = np.zeros(n_valid, dtype=int)
for i, cl in enumerate(clusters_butina):
    for k in cl:
        cid_butina[k] = i
print(f"  {len(clusters_butina)} clusters")

df_butina = df.copy()
df_butina["cluster"] = np.nan
df_butina.loc[df_valid.index, "cluster"] = cid_butina

# =============================================================================
# CLUSTERING 2 — k-means / RDKit descriptors
# =============================================================================
print(f"k-means clustering (k={N_CLUSTERS_KM}) ...")
km       = KMeans(n_clusters=N_CLUSTERS_KM, random_state=RANDOM_STATE, n_init=10)
cid_km   = km.fit_predict(X_scaled)
print("  Done")

df_km = df.copy()
df_km["cluster"] = np.nan
df_km.loc[df_valid.index, "cluster"] = cid_km

# =============================================================================
# TEST-SET ASSIGNMENTS
# =============================================================================
print("Deriving test-set assignments ...")
splits_butina = get_test_assignments(df_butina)
splits_km     = get_test_assignments(df_km)
print("  Done")

# count summaries
for label, sp in [("Butina", splits_butina), ("k-means", splits_km)]:
    counts = np.array([len(sp[i]) for i in df_valid.index])
    print(f"  {label}:  never-in-test={( counts==0).sum()}  "
          + "  ".join(f"{n}x={(counts==n).sum()}" for n in range(1, 6)))

# =============================================================================
# FIGURE
# =============================================================================
plt.rcParams.update({
    "font.size": 12, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))


def draw_panel(ax, splits, title):
    valid_indices = list(df_valid.index)

    # Precompute per-molecule properties
    n_sets      = np.array([len(splits[i]) for i in valid_indices])
    first_split = np.array([
        min(splits[i]) if splits[i] else -1
        for i in valid_indices
    ])

    # Background: molecules not in any test set
    bg = n_sets == 0
    ax.scatter(X_pca[bg, 0], X_pca[bg, 1],
               color="lightgray", alpha=0.7,
               s=MARKER_SIZES[0], marker="o", linewidths=0, zorder=1, rasterized=True)

    # n=1: colour by which test split (unambiguous)
    for sid in range(N_TEST_SPLITS):
        mask = (n_sets == 1) & (first_split == sid)
        if not mask.any():
            continue
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=SPLIT_COLORS[sid],
            s=MARKER_SIZES[1], marker=MARKERS[1],
            linewidths=0, alpha=0.85, zorder=3, rasterized=True,
        )

    # n≥2: dark colour — colour is ambiguous so we use a neutral shade;
    # shape encodes the count
    for n in range(2, N_TEST_SPLITS + 1):
        mask = n_sets == n
        if not mask.any():
            continue
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=MULTI_COLOR,
            s=MARKER_SIZES[n], marker=MARKERS[n],
            linewidths=0.4, edgecolors="k",
            alpha=0.9, zorder=4 + n, rasterized=True,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1  ({var1:.1%} var.)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var2:.1%} var.)", fontsize=11)


draw_panel(axes[0], splits_butina,
           "Butina / Morgan fingerprints")
draw_panel(axes[1], splits_km,
           f"k-means (k={N_CLUSTERS_KM}) / RDKit descriptors")

# ── Legends ──────────────────────────────────────────────────────────────
colour_handles = (
    [Line2D([0], [0], marker="o", linestyle="none",
            markerfacecolor=SPLIT_COLORS[i], markeredgecolor="none",
            markersize=8, label=f"Test split {i}")
     for i in range(N_TEST_SPLITS)]
    + [Line2D([0], [0], marker="o", linestyle="none",
              markerfacecolor=MULTI_COLOR, markeredgecolor="none",
              markersize=8, label="Multiple splits")]
    + [Line2D([0], [0], marker=".", linestyle="none",
              markerfacecolor="lightgray", markeredgecolor="none",
              markersize=8, label="Never in test")]
)

shape_handles = [
    Line2D([0], [0], marker=MARKERS[n], linestyle="none",
           markerfacecolor="0.55",
           markeredgecolor="k" if n > 1 else "none",
           markersize=9 if n < 5 else 11,
           label=MARKER_LABELS[n])
    for n in range(len(MARKERS))
]

axes[1].legend(
    handles=colour_handles + [Line2D([], [], linestyle="none")] + shape_handles,
    loc="upper right", fontsize=9, framealpha=0.9,
    title="Colour (circles only)\nShape = # test splits",
    title_fontsize=9,
)

fig.suptitle(
    "Cluster-based train/test splitting — 2-D PCA of RDKit descriptors",
    fontsize=13, y=1.01,
)
fig.tight_layout()
fig.savefig("cluster_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved cluster_comparison.png")
