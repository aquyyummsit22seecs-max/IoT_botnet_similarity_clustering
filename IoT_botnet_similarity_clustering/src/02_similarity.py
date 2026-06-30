"""
02_similarity.py — Step 2 of 6: Pairwise Behavioral Similarity
Computes cosine and Jaccard similarity matrices and produces heatmap figures.
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from log_utils import setup_script_logging
from viz_utils import set_academic_style, save_figure

os.makedirs("output/similarity", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
setup_script_logging("02_similarity.py")

INPUT  = "output/features/behavioral_features.csv"
LABELS = "output/features/ground_truth_labels.csv"

if not os.path.exists(INPUT):
    print(f"[ERROR] {INPUT} not found. Run 01_feature_extraction.py first.")
    sys.exit(1)

set_academic_style()

FAMILY_COLORS = {
    "mirai": "#E74C3C", "hajime": "#3498DB", "gafgyt": "#2ECC71",
    "bashlite": "#2ECC71", "unlabeled": "#95A5A6", "singleton": "#F39C12",
    "tsunami": "#1ABC9C",
}
FAMILY_ORDER = ["mirai","hajime","gafgyt","unlabeled","singleton","tsunami","bashlite"]

df        = pd.read_csv(INPUT)
ids       = df["sample_id"].tolist()
feat_cols = [c for c in df.columns if c != "sample_id"]
X_bin     = df[feat_cols].values.astype(bool)
n         = len(ids)

family_map = {}
if os.path.exists(LABELS):
    df_lbl = pd.read_csv(LABELS).set_index("sample_id")
    family_map = {sid: df_lbl["family"].get(sid, "unlabeled") for sid in ids}
else:
    family_map = {sid: "unlabeled" for sid in ids}

print(f"[INFO] {n} samples × {len(feat_cols)} features")
print(f"[INFO] Computing {n}×{n} pairwise similarity matrices...")


def save_matrix_csv(matrix, row_ids, col_ids, path):
    pd.DataFrame(matrix, index=row_ids, columns=col_ids).to_csv(path, index_label="sample_id")
    print(f"[DONE] {path}")


def save_subset_matrix_csv(matrix, sub_idx, path):
    sub_ids = [ids[i] for i in sub_idx]
    save_matrix_csv(matrix[np.ix_(sub_idx, sub_idx)], sub_ids, sub_ids, path)


# ── TF-IDF for cosine ──
docs = []
for _, row in df.iterrows():
    present = [f for f in feat_cols if row[f] == 1]
    docs.append(" ".join(present) if present else "__empty__")

tfidf   = TfidfVectorizer(min_df=1, sublinear_tf=True, norm="l2")
X_tfidf = tfidf.fit_transform(docs)
vocab_size = len(tfidf.vocabulary_)
n_dropped  = len(feat_cols) - vocab_size
print(f"[INFO] TF-IDF vocabulary: {vocab_size} / {len(feat_cols)} features "
      f"(min_df=1 — 0 features dropped by frequency threshold)")
if n_dropped > 0:
    dropped_feats = sorted(set(feat_cols) - set(tfidf.get_feature_names_out()))
    print(f"[WARN] Dropped {n_dropped} features not meeting min_df threshold: {dropped_feats}")

# ── Cosine similarity ──
print("[INFO] Computing cosine similarity...")
cos_sim = np.clip(cosine_similarity(X_tfidf), 0.0, 1.0)
save_matrix_csv(cos_sim, ids, ids, "output/similarity/cosine_similarity_matrix.csv")
print(f"  Shape: {cos_sim.shape}")

# ── Jaccard similarity ──
print("[INFO] Computing Jaccard similarity...")
zero_rows = (X_bin.sum(axis=1) == 0).sum()
if zero_rows:
    print(f"  [WARN] {zero_rows} all-zero rows — Jaccard undefined for these")
jac_sim = np.clip(1.0 - pairwise_distances(X_bin, metric="jaccard"), 0.0, 1.0)
save_matrix_csv(jac_sim, ids, ids, "output/similarity/jaccard_similarity_matrix.csv")
print(f"  Shape: {jac_sim.shape}")

# ── Statistics ──
mask     = ~np.eye(n, dtype=bool)
cos_vals = cos_sim[mask]
jac_vals = jac_sim[mask]

df_stats = pd.DataFrame({
    "metric":       ["Cosine Similarity", "Jaccard Similarity"],
    "mean":         [cos_vals.mean(),       jac_vals.mean()],
    "median":       [np.median(cos_vals),   np.median(jac_vals)],
    "std":          [cos_vals.std(),        jac_vals.std()],
    "min":          [cos_vals.min(),        jac_vals.min()],
    "max":          [cos_vals.max(),        jac_vals.max()],
    "pct_high_0.5": [(cos_vals > 0.5).mean()*100, (jac_vals > 0.5).mean()*100],
    "pct_low_0.1":  [(cos_vals < 0.1).mean()*100, (jac_vals < 0.1).mean()*100],
}).round(4)
df_stats.to_csv("output/similarity/similarity_stats.csv", index=False)

cos_dist = np.clip(1.0 - cos_sim, 0.0, 1.0)
cos_dist = (cos_dist + cos_dist.T) / 2.0
np.fill_diagonal(cos_dist, 0.0)
assert np.allclose(cos_dist, cos_dist.T, atol=1e-8), \
    "[ASSERT] cos_dist is not symmetric — check similarity computation"

with open("output/similarity/similarity_matrices.pkl", "wb") as f:
    pickle.dump({
        "ids": ids, "X_tfidf": X_tfidf, "X_bin": X_bin,
        "cos_sim": cos_sim, "jac_sim": jac_sim, "cos_dist": cos_dist,
        "feat_cols": feat_cols, "family_map": family_map,
    }, f)


# ── Heatmap helper ──
def build_subsample(ids, family_map, max_n=300):
    order = sorted(range(len(ids)), key=lambda i: (
        FAMILY_ORDER.index(family_map.get(ids[i], "unlabeled"))
        if family_map.get(ids[i], "unlabeled") in FAMILY_ORDER else 99,
        ids[i]
    ))
    step = max(1, len(order) // max_n)
    return order[::step][:max_n]


def plot_heatmap_annotated(matrix, sub_idx, title, cmap, outpath, metric_name):
    sub_ids = [ids[i] for i in sub_idx]
    sub_mat = matrix[np.ix_(sub_idx, sub_idx)]
    n_sub   = len(sub_idx)

    present_families = sorted(set(family_map.get(s,"unlabeled") for s in sub_ids),
                               key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)
    fam_int  = {f: i for i, f in enumerate(present_families)}
    strip    = np.array([fam_int.get(family_map.get(s,"unlabeled"), 0) for s in sub_ids])
    cmap_fam = mcolors.ListedColormap([FAMILY_COLORS.get(f,"#7F8C8D") for f in present_families])

    fig = plt.figure(figsize=(12, 11))
    gs  = plt.GridSpec(3, 2, width_ratios=[0.05, 1], height_ratios=[0.05, 1, 0.18],
                       hspace=0.04, wspace=0.04, left=0.08, right=0.92, top=0.90, bottom=0.06)

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, :])

    ax_top.imshow(strip[np.newaxis, :], cmap=cmap_fam, vmin=0, vmax=len(present_families)-1, aspect="auto")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_xlabel("Family (x-axis)", fontsize=8, labelpad=2)

    ax_left.imshow(strip[:, np.newaxis], cmap=cmap_fam, vmin=0, vmax=len(present_families)-1, aspect="auto")
    ax_left.set_xticks([]); ax_left.set_yticks([])
    ax_left.set_ylabel("Family (y-axis)", fontsize=8, labelpad=2)

    im = ax_main.imshow(sub_mat, cmap=cmap, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    ax_main.set_xticks([]); ax_main.set_yticks([])
    ax_main.set_xlabel(f"Samples  (n={n_sub}, sorted by family label)", fontsize=10)
    ax_main.set_ylabel(f"Samples  (n={n_sub}, sorted by family label)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.040, pad=0.015)
    cbar.set_label(f"{metric_name}  [0 = dissimilar  |  1 = identical]", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    legend_patches = [mpatches.Patch(color=FAMILY_COLORS.get(f,"#7F8C8D"), label=f.title())
                      for f in present_families]
    ax_main.legend(handles=legend_patches, title="Botnet Family", title_fontsize=9, fontsize=9,
                   loc="upper left", bbox_to_anchor=(1.12, 1.0), framealpha=0.95, edgecolor="#BBBBBB")

    off_vals = sub_mat[~np.eye(n_sub, dtype=bool)]
    ax_info.axis("off")
    stat_text = (
        f"Off-diagonal statistics (n×n = {n_sub}×{n_sub} = {n_sub*n_sub:,} pairs):   "
        f"Mean = {off_vals.mean():.4f}   Median = {np.median(off_vals):.4f}   "
        f"Std = {off_vals.std():.4f}   Min = {off_vals.min():.4f}   "
        f"Max = {off_vals.max():.4f}   Pairs > 0.5: {(off_vals>0.5).mean()*100:.1f}%"
    )
    ax_info.text(0.5, 0.55, stat_text, transform=ax_info.transAxes, ha="center", va="center",
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#EBF5FB", ec="#2E86C1", alpha=0.9))
    ax_info.text(0.5, 0.08,
                 "Block structure along diagonal = high within-family similarity. "
                 "Off-diagonal brightness = cross-family behavioral overlap.",
                 transform=ax_info.transAxes, ha="center", va="center",
                 fontsize=8.5, color="#555555", style="italic")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")


sub_idx = build_subsample(ids, family_map, max_n=min(n, 300))
sub_ids = [ids[i] for i in sub_idx]
pd.DataFrame({
    "subset_order": list(range(1, len(sub_ids) + 1)),
    "sample_id": sub_ids,
    "family": [family_map.get(sid, "unlabeled") for sid in sub_ids],
}).to_csv("output/similarity/heatmap_subset_samples.csv", index=False)
print("[DONE] output/similarity/heatmap_subset_samples.csv")

save_subset_matrix_csv(cos_sim, sub_idx, "output/similarity/cosine_similarity_subset_matrix.csv")
save_subset_matrix_csv(jac_sim, sub_idx, "output/similarity/jaccard_similarity_subset_matrix.csv")

plot_heatmap_annotated(
    cos_sim, sub_idx,
    "Cosine Similarity Heatmap — IoT Botnet Behavioral Profiles\n"
    "(TF-IDF weighted vectors, sorted by family label)",
    "viridis", "output/similarity/fig_cosine_heatmap.png", "Cosine Similarity"
)
plot_heatmap_annotated(
    jac_sim, sub_idx,
    "Jaccard Similarity Heatmap — IoT Botnet Behavioral Profiles\n"
    "(Binary feature set overlap, sorted by family label)",
    "magma", "output/similarity/fig_jaccard_heatmap.png", "Jaccard Similarity"
)


# ── Side-by-side comparison ──
fig, axes = plt.subplots(1, 2, figsize=(20, 8.6))
fig.patch.set_facecolor("#FAFAFA")
for ax, mat, cmap, title, metric, desc in [
    (axes[0], cos_sim, "viridis", "Cosine Similarity",  "Cosine",
     "TF-IDF weighted — rewards discriminative, rare behaviors"),
    (axes[1], jac_sim, "magma",   "Jaccard Similarity", "Jaccard",
     "Binary set overlap — treats all features equally"),
]:
    sub  = mat[np.ix_(sub_idx, sub_idx)]
    im   = ax.imshow(sub, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(f"{metric} Score [0 – 1]", fontsize=10)
    ax.set_title(f"{title}\n{desc}", fontweight="bold", fontsize=11, pad=18)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"n={len(sub_idx)} samples (sorted by family)", fontsize=9)
    off = sub[~np.eye(len(sub_idx), dtype=bool)]
    ax.text(0.5, -0.07,
            f"Mean={off.mean():.3f}   Median={np.median(off):.3f}   "
            f"Std={off.std():.3f}   Pairs>0.5: {(off>0.5).mean()*100:.1f}%",
            transform=ax.transAxes, ha="center", fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#BBBBBB"))

legend_patches_cmp = [mpatches.Patch(color=FAMILY_COLORS.get(f,"#7F8C8D"), label=f.title())
                      for f in set(family_map.values())]
fig.legend(handles=legend_patches_cmp, title="Family label (sort order)", title_fontsize=9, fontsize=9,
           loc="lower center", bbox_to_anchor=(0.5, 0.02),
           ncol=max(3, min(6, len(set(family_map.values())))), framealpha=0.95)
fig.suptitle(
    "Behavioral Similarity Comparison — Cosine vs Jaccard\n"
    f"IoT Botnet Dataset  |  N={n} samples  |  "
    f"Representative subsample n={len(sub_idx)}, sorted by family",
    fontsize=13, fontweight="bold", y=0.985
)
fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.18)
plt.savefig("output/similarity/fig_similarity_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("[DONE] output/similarity/fig_similarity_comparison.png")

print("[INFO] Skipped fig_similarity_stats.png (non-core, repetitive details).")
print(f"\n[INFO] Similarity Statistics (off-diagonal):")
print(df_stats.to_string(index=False))
print(f"\n[DONE] All similarity outputs → output/similarity/")
