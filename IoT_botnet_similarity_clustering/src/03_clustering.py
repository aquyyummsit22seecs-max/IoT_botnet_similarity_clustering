"""03_clustering_updated.py — Step 3 of 6: Multi-Matrix Two-Algorithm Clustering
Runs Hierarchical and Spectral clustering on Cosine and Jaccard similarity matrices.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from log_utils import setup_script_logging
from viz_utils import set_academic_style, wrap_label


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_NAME = "03_clustering.py"
SIMILARITY_PKL = Path("output/similarity/similarity_matrices.pkl")
OUT_DIR = Path("output/clustering")
LOG_DIR = Path("output/logs")
RANDOM_STATE = 42
K = 4

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(SCRIPT_NAME)
set_academic_style()


# Academic, print-safe color palette.
CLUSTER_COLORS = [
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
    "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC", "#2F4B7C",
]
FAMILY_COLORS = {
    "mirai": "#4E79A7",
    "hajime": "#F28E2B",
    "gafgyt": "#59A14F",
    "bashlite": "#8CD17D",
    "tsunami": "#76B7B2",
    "dofloo": "#B07AA1",
    "lightaidra": "#86BCB6",
    "singleton": "#EDC948",
    "unlabeled": "#BAB0AC",
}
FAMILY_MARKERS = {
    "mirai": "^",
    "hajime": "s",
    "gafgyt": "D",
    "bashlite": "P",
    "tsunami": "X",
    "dofloo": "v",
    "lightaidra": "<",
    "singleton": "*",
    "unlabeled": "o",
}


# Muted IEEE-style palette for publication figures.
IEEE_BLUE = "#1F4E79"
IEEE_TEAL = "#5B9BD5"
IEEE_GRAY = "#7F7F7F"
IEEE_DARK_GRAY = "#404040"
IEEE_ORANGE = "#C55A11"
IEEE_GREEN = "#548235"
IEEE_LIGHT_GRAY = "#D9E2F3"
RUN_COLORS = {
    "Hierarchical Cosine": IEEE_BLUE,
    "Hierarchical Jaccard": IEEE_TEAL,
    "Spectral Cosine": IEEE_GRAY,
    "Spectral Jaccard": IEEE_ORANGE,
}
MATRIX_COLORS = {"Cosine": IEEE_BLUE, "Jaccard": IEEE_TEAL}


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def load_similarity_data(path: Path) -> dict:
    if not path.exists():
        fail(f"{path} not found. Run 02_similarity.py first.")
    with path.open("rb") as f:
        return pickle.load(f)


def clean_similarity(sim: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=float)
    sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0)
    sim = np.clip(sim, 0.0, 1.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim


def make_distance(sim: np.ndarray) -> np.ndarray:
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist


def cluster_order(labels: np.ndarray) -> list[int]:
    return sorted(int(x) for x in set(labels))


def count_clusters(labels: np.ndarray) -> int:
    return len(set(np.asarray(labels)))


def fit_hierarchical(dist: np.ndarray, k: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=k,
            affinity="precomputed",
            linkage="average",
        )
    return model.fit_predict(dist)


def fit_spectral(sim: np.ndarray, k: int) -> np.ndarray:
    return SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=20,
        random_state=RANDOM_STATE,
    ).fit_predict(sim)


def compute_silhouette(labels: np.ndarray, dist: np.ndarray) -> float:
    labels = np.asarray(labels)
    if count_clusters(labels) < 2:
        return np.nan
    return float(silhouette_score(dist, labels, metric="precomputed"))


def cluster_profile(
    labels: np.ndarray,
    families_per_sample: list[str],
    all_families: list[str],
) -> pd.DataFrame:
    rows = []
    labels = np.asarray(labels)

    for cid in cluster_order(labels):
        idx = np.where(labels == cid)[0]
        fams = [families_per_sample[i] for i in idx]
        counts = pd.Series(fams).value_counts()
        total = int(len(idx))
        dominant_family = counts.index[0] if len(counts) else "N/A"
        dominant_count = int(counts.iloc[0]) if len(counts) else 0

        row = {
            "cluster": int(cid),
            "cluster_name": f"Cluster {cid}",
            "total": total,
            "dominant_family": dominant_family,
            "dominant_count": dominant_count,
            "purity_pct": round((dominant_count / total * 100) if total else 0.0, 1),
        }
        for fam in all_families:
            row[fam] = int(counts.get(fam, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def save_cluster_profile(
    labels: np.ndarray,
    families_per_sample: list[str],
    all_families: list[str],
    outpath: Path,
) -> pd.DataFrame:
    profile = cluster_profile(labels, families_per_sample, all_families)
    profile.to_csv(outpath, index=False)
    print(f"[DONE] {outpath}")
    return profile


def plot_pca_clusters(
    labels: np.ndarray,
    x_pca: np.ndarray,
    var1: float,
    var2: float,
    families_per_sample: list[str],
    all_families: list[str],
    title: str,
    outpath: Path,
    subtitle: str,
    silhouette: float,
) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    fig.patch.set_facecolor("#FAFAFA")

    labels = np.asarray(labels)
    for cid in cluster_order(labels):
        mask = labels == cid
        ax.scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            s=34,
            alpha=0.78,
            marker="o",
            edgecolors="white",
            linewidths=0.4,
            label=f"C{cid} (n={int(mask.sum())})",
        )

    # Family markers are drawn as outlines to show true-family distribution
    # without replacing the cluster color encoding.
    for fam in all_families:
        idx = [i for i, f in enumerate(families_per_sample) if f == fam]
        if not idx:
            continue
        ax.scatter(
            x_pca[idx, 0],
            x_pca[idx, 1],
            marker=FAMILY_MARKERS.get(fam, "o"),
            s=78,
            facecolors="none",
            edgecolors=FAMILY_COLORS.get(fam, "#7F8C8D"),
            linewidths=1.15,
            alpha=0.9,
        )

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(alpha=0.25)

    info = [
        f"Samples: {len(labels)}",
        f"Clusters: {count_clusters(labels)}",
        f"Silhouette: {silhouette:.4f}" if np.isfinite(silhouette) else "Silhouette: N/A",
        subtitle,
    ]
    ax.text(
        1.01,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BDBDBD", alpha=0.95),
    )

    ax.legend(
        fontsize=7.2,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.9,
        title="Predicted cluster",
        title_fontsize=8.0,
    )

    fig.subplots_adjust(left=0.07, right=0.78, top=0.90, bottom=0.12)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")


def plot_cluster_sizes(
    labels: np.ndarray,
    families_per_sample: list[str],
    all_families: list[str],
    title: str,
    outpath: Path,
) -> None:
    profile = cluster_profile(labels, families_per_sample, all_families)
    x = np.arange(len(profile))

    fig, ax = plt.subplots(figsize=(13.8, 6.6))
    bottom = np.zeros(len(profile))

    for fam in all_families:
        vals = profile[fam].values if fam in profile else np.zeros(len(profile))
        if vals.sum() == 0:
            continue
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.65,
            color=FAMILY_COLORS.get(fam, "#7F8C8D"),
            edgecolor="white",
            linewidth=0.8,
            label=fam.title(),
        )
        bottom += vals

    max_total = max(float(profile["total"].max()), 1.0)
    for xi, total in zip(x, profile["total"]):
        ax.text(
            xi,
            total + max(max_total * 0.03, 0.5),
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(v, 12) for v in profile["cluster_name"]], fontsize=8.8)
    ax.set_ylabel("Samples")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.28)

    ax.legend(
        title="True family",
        fontsize=7.8,
        title_fontsize=8.0,
        ncol=min(4, max(1, len(all_families))),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.9,
    )

    fig.subplots_adjust(left=0.07, right=0.82, top=0.88, bottom=0.16)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")



def _draw_cluster_size_panel(
    ax: plt.Axes,
    labels: np.ndarray,
    families_per_sample: list[str],
    all_families: list[str],
    title: str,
) -> None:
    """Draw one stacked cluster size panel on a supplied axis."""
    profile = cluster_profile(labels, families_per_sample, all_families)
    x = np.arange(len(profile))
    bottom = np.zeros(len(profile))

    for fam in all_families:
        vals = profile[fam].values if fam in profile else np.zeros(len(profile))
        if vals.sum() == 0:
            continue
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.68,
            color=FAMILY_COLORS.get(fam, "#7F8C8D"),
            edgecolor="white",
            linewidth=0.65,
            label=fam.title(),
        )
        bottom += vals

    max_total = max(float(profile["total"].max()), 1.0)
    for xi, total in zip(x, profile["total"]):
        ax.text(
            xi,
            total + max(max_total * 0.025, 0.5),
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=IEEE_DARK_GRAY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(v, 10) for v in profile["cluster_name"]], fontsize=7.4)
    ax.set_title(title, fontweight="bold", fontsize=9.5)
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.24, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max_total * 1.18)


def plot_cluster_size_comparison_grid(
    results: dict[tuple[str, str], np.ndarray],
    families_per_sample: list[str],
    all_families: list[str],
    outpath: Path,
) -> None:
    """Save one 2 by 2 figure containing all cluster size outputs."""
    plot_order = [
        ("hierarchical", "cosine", "Hierarchical + Cosine"),
        ("hierarchical", "jaccard", "Hierarchical + Jaccard"),
        ("spectral", "cosine", "Spectral + Cosine"),
        ("spectral", "jaccard", "Spectral + Jaccard"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.2), sharey=False)
    legend_handles = None
    legend_labels = None

    for ax, (algo, matrix_name, title) in zip(axes.ravel(), plot_order):
        _draw_cluster_size_panel(
            ax=ax,
            labels=results[(algo, matrix_name)],
            families_per_sample=families_per_sample,
            all_families=all_families,
            title=title,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle(
        "Side by side cluster size comparison across algorithms and similarity matrices",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="True family",
            loc="lower center",
            ncol=min(5, max(1, len(legend_labels))),
            fontsize=8.0,
            title_fontsize=8.2,
            frameon=False,
        )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.12, hspace=0.36, wspace=0.20)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")


def plot_cluster_size_total_comparison(
    results: dict[tuple[str, str], np.ndarray],
    outpath: Path,
) -> None:
    """Save grouped bars comparing cluster totals for all four runs."""
    run_specs = [
        ("hierarchical", "cosine", "Hierarchical Cosine"),
        ("hierarchical", "jaccard", "Hierarchical Jaccard"),
        ("spectral", "cosine", "Spectral Cosine"),
        ("spectral", "jaccard", "Spectral Jaccard"),
    ]
    cluster_ids = sorted({int(cid) for labels in results.values() for cid in set(labels)})
    x = np.arange(len(cluster_ids))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13.8, 6.8))
    for i, (algo, matrix_name, run_name) in enumerate(run_specs):
        labels = np.asarray(results[(algo, matrix_name)])
        counts = pd.Series(labels).value_counts().to_dict()
        values = [int(counts.get(cid, 0)) for cid in cluster_ids]
        offset = (i - (len(run_specs) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=run_name,
            color=RUN_COLORS.get(run_name, IEEE_GRAY),
            edgecolor=IEEE_DARK_GRAY,
            linewidth=0.45,
        )
        for bar, value in zip(bars, values):
            if value == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(max(values) * 0.015, 0.5),
                str(value),
                ha="center",
                va="bottom",
                fontsize=7.0,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {cid}" for cid in cluster_ids], fontsize=8.8)
    ax.set_ylabel("Samples")
    ax.set_title("Cluster size totals compared side by side", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=8.2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.24)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")

def plot_algorithm_comparison(
    results: dict[tuple[str, str], np.ndarray],
    matrices: dict,
    comparison_df: pd.DataFrame,
    x_pca: np.ndarray,
    var1: float,
    var2: float,
    outpath: Path,
) -> None:
    plot_order = [
        ("hierarchical", "cosine"),
        ("spectral", "cosine"),
        ("hierarchical", "jaccard"),
        ("spectral", "jaccard"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0))
    for ax, (algo, matrix_name) in zip(axes.ravel(), plot_order):
        labels = results[(algo, matrix_name)]
        row = comparison_df.loc[comparison_df["label_column"] == f"{algo}_{matrix_name}"].iloc[0]
        sil = row["silhouette"]

        for cid in cluster_order(labels):
            mask = labels == cid
            ax.scatter(
                x_pca[mask, 0],
                x_pca[mask, 1],
                c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                s=22,
                alpha=0.78,
                linewidths=0.25,
                edgecolors="white",
            )

        algo_name = "Hierarchical" if algo == "hierarchical" else "Spectral"
        ax.set_title(
            f"{algo_name} + {matrices[matrix_name]['display']}\nSilhouette={sil:.4f}",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Clustering comparison by algorithm and similarity matrix",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98, hspace=0.35, wspace=0.25)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")


def plot_metric_bar(
    comparison_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    df = comparison_df.copy()
    df["run"] = df["algorithm"] + " " + df["matrix"]
    df["run_label"] = df["algorithm"] + "\n" + df["matrix"]
    x = np.arange(len(df))

    metric_values = df[metric].astype(float)
    fig_width = 11.8 if metric == "silhouette" else 10.8
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    if metric == "silhouette":
        colors = [RUN_COLORS.get(run, IEEE_GRAY) for run in df["run"]]
        bar_width = 0.48
        upper_pad = 0.075
    else:
        colors = [MATRIX_COLORS.get(matrix, IEEE_GRAY) for matrix in df["matrix"]]
        bar_width = 0.52
        upper_pad = max(metric_values.max() * 0.22, 1.0)

    bars = ax.bar(
        x,
        metric_values,
        width=bar_width,
        color=colors,
        edgecolor=IEEE_DARK_GRAY,
        linewidth=0.65,
    )

    for bar, value in zip(bars, metric_values):
        label = f"{value:.4f}" if metric == "silhouette" else str(int(value))
        offset = 0.012 if metric == "silhouette" else max(metric_values.max() * 0.035, 0.20)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=IEEE_DARK_GRAY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(v, 14) for v in df["run_label"]], fontsize=8.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric == "silhouette":
        ax.set_ylim(0, min(1.0, max(metric_values.max() + upper_pad, 0.60)))
        ax.axhline(0.50, color=IEEE_GREEN, linestyle="--", linewidth=0.9, alpha=0.65)
        ax.axhline(0.25, color=IEEE_ORANGE, linestyle="--", linewidth=0.9, alpha=0.65)
        handles = [plt.Rectangle((0, 0), 1, 1, color=RUN_COLORS.get(run, IEEE_GRAY)) for run in df["run"]]
        ax.legend(
            handles,
            df["run"],
            title="Clustering run",
            fontsize=7.6,
            title_fontsize=8.0,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.30)
    else:
        ax.set_ylim(0, metric_values.max() + upper_pad)
        handles = [plt.Rectangle((0, 0), 1, 1, color=MATRIX_COLORS[m]) for m in sorted(df["matrix"].unique()) if m in MATRIX_COLORS]
        labels = [m for m in sorted(df["matrix"].unique()) if m in MATRIX_COLORS]
        if handles:
            ax.legend(handles, labels, title="Similarity matrix", fontsize=8.0, title_fontsize=8.0, loc="upper right", frameon=False)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.20)

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")


def remove_obsolete_pca_outputs() -> None:
    # Any file matching fig_*_pca.png or the old flat fig_hierarchical/spectral_cluster_sizes.png
    # will be caught even if future naming changes occur.
    import glob as _glob
    patterns = [
        str(OUT_DIR / "fig_*_pca.png"),
        str(OUT_DIR / "fig_hierarchical_cluster_sizes.png"),
        str(OUT_DIR / "fig_spectral_cluster_sizes.png"),
    ]
    removed = 0
    for pattern in patterns:
        for path_str in _glob.glob(pattern):
            path = Path(path_str)
            if path.exists():
                path.unlink()
                print(f"[CLEAN] Removed obsolete output: {path}")
                removed += 1
    if removed:
        print(f"[CLEAN] Removed {removed} obsolete file(s) from {OUT_DIR}")
    else:
        print("[CLEAN] No obsolete PCA/stale outputs found.")

def main() -> None:
    data = load_similarity_data(SIMILARITY_PKL)

    ids = list(data["ids"])
    x_tfidf = data["X_tfidf"]
    x_dense = x_tfidf.toarray()
    x_bin = np.asarray(data.get("X_bin", x_dense > 0), dtype=float)
    family_map = data.get("family_map", {})

    n_samples = len(ids)
    if n_samples < K:
        fail(f"Need at least K={K} samples for clustering, found {n_samples}.")

    cos_sim = clean_similarity(data.get("cos_sim", 1.0 - data["cos_dist"]))
    jac_sim = clean_similarity(data["jac_sim"])

    matrices = {
        "cosine": {
            "display": "Cosine",
            "sim": cos_sim,
            "dist": make_distance(cos_sim),
            "features": x_dense,
            "note": "TF-IDF weighted behavioral similarity",
        },
        "jaccard": {
            "display": "Jaccard",
            "sim": jac_sim,
            "dist": make_distance(jac_sim),
            "features": x_bin,
            "note": "Binary behavioral feature overlap",
        },
    }

    x_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x_dense)
    var1, var2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(x_dense).explained_variance_ratio_ * 100

    families_per_sample = [family_map.get(sid, "unlabeled") for sid in ids]
    all_families = sorted(set(families_per_sample))

    print(f"[INFO] Clustering {n_samples} samples with K={K}")
    print("[INFO] Algorithms enabled: Hierarchical, Spectral")
    print("[INFO] Matrices enabled: Cosine, Jaccard")

    all_results: dict[tuple[str, str], np.ndarray] = {}
    comparison_rows: list[dict] = []
    label_df = pd.DataFrame({"sample_id": ids})

    algorithm_specs = [
        {
            "slug": "hierarchical",
            "name": "Hierarchical",
            "runner": lambda sim, dist: fit_hierarchical(dist, K),
            "input_kind": "distance",
            "params": f"average linkage, k={K}",
        },
        {
            "slug": "spectral",
            "name": "Spectral",
            "runner": lambda sim, dist: fit_spectral(sim, K),
            "input_kind": "similarity affinity",
            "params": f"k={K}, assign_labels=kmeans, n_init=20",
        },
    ]

    for matrix_name, matrix_spec in matrices.items():
        display = matrix_spec["display"]
        sim = matrix_spec["sim"]
        dist = matrix_spec["dist"]

        print(f"\n[INFO] Matrix: {display} similarity")
        for algo in algorithm_specs:
            slug = algo["slug"]
            name = algo["name"]
            label_col = f"{slug}_{matrix_name}"

            print(f"[INFO] Running {name} Clustering")
            labels = algo["runner"](sim, dist)
            silhouette = compute_silhouette(labels, dist)

            all_results[(slug, matrix_name)] = labels
            label_df[label_col] = labels

            comparison_rows.append(
                {
                    "algorithm": name,
                    "matrix": display,
                    "label_column": label_col,
                    "n_clusters": count_clusters(labels),
                    "n_noise": 0,
                    "silhouette": round(silhouette, 4) if np.isfinite(silhouette) else np.nan,
                    "input": f"{display} {algo['input_kind']} matrix",
                    "params": algo["params"],
                }
            )

            profile_path = OUT_DIR / f"profile_{slug}_{matrix_name}.csv"
            save_cluster_profile(labels, families_per_sample, all_families, profile_path)

            plot_cluster_sizes(
                labels,
                families_per_sample,
                all_families,
                f"{name} Clustering, {display} Matrix",
                OUT_DIR / f"fig_{slug}_{matrix_name}_cluster_sizes.png",
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "algorithm_comparison.csv", index=False)

    # Clean aliases for later pipeline stages. These select the best matrix for
    # each algorithm by silhouette, while preserving per-matrix columns.
    for algo_slug in ["hierarchical", "spectral"]:
        subset = comparison_df[comparison_df["label_column"].str.startswith(f"{algo_slug}_")].copy()
        best_row = subset.sort_values("silhouette", ascending=False, na_position="last").iloc[0]
        label_df[algo_slug] = label_df[str(best_row["label_column"])]

    label_df.to_csv(OUT_DIR / "cluster_labels.csv", index=False)
    print(f"[DONE] {OUT_DIR / 'cluster_labels.csv'}")
    print(f"[DONE] {OUT_DIR / 'algorithm_comparison.csv'}")

    matrix_summary = []
    for matrix_name, matrix_spec in matrices.items():
        sub = comparison_df[comparison_df["matrix"] == matrix_spec["display"]].copy()
        matrix_summary.append(
            {
                "matrix": matrix_spec["display"],
                "mean_silhouette": round(float(sub["silhouette"].dropna().mean()), 4),
                "best_algorithm": sub.sort_values("silhouette", ascending=False).iloc[0]["algorithm"],
                "best_silhouette": round(float(sub["silhouette"].dropna().max()), 4),
                "mean_clusters": round(float(sub["n_clusters"].mean()), 2),
                "total_noise": int(sub["n_noise"].sum()),
                "interpretation": matrix_spec["note"],
            }
        )

    matrix_df = pd.DataFrame(matrix_summary)
    matrix_df["rank_by_mean_silhouette"] = (
        matrix_df["mean_silhouette"].rank(ascending=False, method="dense").astype(int)
    )
    matrix_df.to_csv(OUT_DIR / "matrix_comparison_summary.csv", index=False)
    print(f"[DONE] {OUT_DIR / 'matrix_comparison_summary.csv'}")

    plot_algorithm_comparison(
        all_results,
        matrices,
        comparison_df,
        x_pca,
        var1,
        var2,
        OUT_DIR / "fig_algorithm_comparison.png",
    )
    plot_metric_bar(
        comparison_df,
        "silhouette",
        "Silhouette score",
        "Silhouette comparison across clustering runs",
        OUT_DIR / "fig_silhouette_comparison.png",
    )
    plot_metric_bar(
        comparison_df,
        "n_clusters",
        "Number of clusters",
        "Cluster-count comparison across clustering runs",
        OUT_DIR / "fig_cluster_count_comparison.png",
    )
    plot_cluster_size_comparison_grid(
        all_results,
        families_per_sample,
        all_families,
        OUT_DIR / "fig_all_algorithm_cluster_sizes_side_by_side.png",
    )
    plot_cluster_size_total_comparison(
        all_results,
        OUT_DIR / "fig_cluster_size_totals_side_by_side.png",
    )

    remove_obsolete_pca_outputs()

    best_hier_row = (
        comparison_df[comparison_df["algorithm"] == "Hierarchical"]
        .sort_values("silhouette", ascending=False, na_position="last")
        .iloc[0]
    )
    best_spectral_row = (
        comparison_df[comparison_df["algorithm"] == "Spectral"]
        .sort_values("silhouette", ascending=False, na_position="last")
        .iloc[0]
    )

    with (OUT_DIR / "clustering_results.pkl").open("wb") as f:
        pickle.dump(
            {
                "ids": ids,
                "X_tfidf": x_tfidf,
                "X_dense": x_dense,
                "X_bin": x_bin,
                "family_map": family_map,
                "K": K,
                "matrices": matrices,
                "labels": {f"{algo}_{matrix}": labels for (algo, matrix), labels in all_results.items()},
                "algorithm_comparison": comparison_df,
                "matrix_comparison_summary": matrix_df,
                "best_hierarchical_column": str(best_hier_row["label_column"]),
                "best_spectral_column": str(best_spectral_row["label_column"]),
                "hier_labels": label_df["hierarchical"].values,
                "spectral_labels": label_df["spectral"].values,
                "cos_sim": cos_sim,
                "cos_dist": matrices["cosine"]["dist"],
                "jac_sim": jac_sim,
                "jac_dist": matrices["jaccard"]["dist"],
                "X_pca": x_pca,
                "var_explained": (var1, var2),
            },
            f,
        )

    print(f"[DONE] {OUT_DIR / 'clustering_results.pkl'}")
    print("\n[INFO] Matrix comparison by mean silhouette:")
    print(matrix_df.to_string(index=False))
    print("\n[DONE] Clustering complete. Only Hierarchical and Spectral clustering were executed.")


if __name__ == "__main__":
    main()
