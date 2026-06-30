"""04_validation_updated.py, Step 4 of 6: Validation for multi-matrix clustering.

Validates all clustering runs created by 03_clustering_updated_side_by_side.py:
    hierarchical_cosine
    hierarchical_jaccard
    spectral_cosine
    spectral_jaccard

Outputs are saved under output/validation.
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
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from log_utils import setup_script_logging
from viz_utils import set_academic_style, wrap_label

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_NAME = "04_validation.py"
CLUSTERING_PKL = Path("output/clustering/clustering_results.pkl")
OUT_DIR = Path("output/validation")
LOG_DIR = Path("output/logs")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(SCRIPT_NAME)
set_academic_style()

IEEE_BLUE = "#1F4E79"
IEEE_TEAL = "#5B9BD5"
IEEE_GRAY = "#7F7F7F"
IEEE_DARK_GRAY = "#404040"
IEEE_ORANGE = "#C55A11"
IEEE_GREEN = "#548235"
RUN_COLORS = {
    "Hierarchical Cosine": IEEE_BLUE,
    "Hierarchical Jaccard": IEEE_TEAL,
    "Spectral Cosine": IEEE_GRAY,
    "Spectral Jaccard": IEEE_ORANGE,
}

RUN_ORDER = [
    ("hierarchical_cosine", "Hierarchical", "Cosine", "Hierarchical Cosine"),
    ("hierarchical_jaccard", "Hierarchical", "Jaccard", "Hierarchical Jaccard"),
    ("spectral_cosine", "Spectral", "Cosine", "Spectral Cosine"),
    ("spectral_jaccard", "Spectral", "Jaccard", "Spectral Jaccard"),
]


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def load_clustering_data(path: Path) -> dict:
    if not path.exists():
        fail(f"{path} not found. Run 03_clustering.py first.")
    with path.open("rb") as f:
        return pickle.load(f)


def count_clusters(labels: np.ndarray) -> int:
    return int(len(set(np.asarray(labels))))


def get_distance_matrix(data: dict, matrix_name: str) -> np.ndarray:
    if matrix_name.lower() == "cosine":
        return np.asarray(data["cos_dist"], dtype=float)
    if matrix_name.lower() == "jaccard":
        return np.asarray(data["jac_dist"], dtype=float)
    fail(f"Unsupported matrix name: {matrix_name}")


def get_feature_matrix(data: dict, matrix_name: str) -> np.ndarray:
    if matrix_name.lower() == "cosine":
        return np.asarray(data.get("X_dense", data["X_tfidf"].toarray()), dtype=float)
    if matrix_name.lower() == "jaccard":
        return np.asarray(data.get("X_bin", data.get("X_dense")), dtype=float)
    fail(f"Unsupported matrix name: {matrix_name}")


def build_true_family_labels(ids: list[str], family_map: dict) -> np.ndarray | None:
    if not family_map:
        return None
    labels = np.asarray([family_map.get(sid, "unlabeled") for sid in ids])
    if len(set(labels)) < 2:
        return None
    return labels


def safe_internal_metrics(features: np.ndarray, dist: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n_clusters = count_clusters(labels)
    if n_clusters < 2 or n_clusters >= len(labels):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    metrics = {}
    try:
        metrics["silhouette"] = float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        metrics["silhouette"] = np.nan

    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(features, labels))
    except Exception:
        metrics["davies_bouldin"] = np.nan

    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(features, labels))
    except Exception:
        metrics["calinski_harabasz"] = np.nan

    return metrics


def safe_external_metrics(true_labels: np.ndarray | None, pred_labels: np.ndarray) -> dict:
    if true_labels is None:
        return {"ari": np.nan, "nmi": np.nan}
    try:
        ari = float(adjusted_rand_score(true_labels, pred_labels))
    except Exception:
        ari = np.nan
    try:
        nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
    except Exception:
        nmi = np.nan
    return {"ari": ari, "nmi": nmi}


def validate_runs(data: dict) -> pd.DataFrame:
    ids = list(data["ids"])
    labels_dict = data.get("labels", {})
    true_labels = build_true_family_labels(ids, data.get("family_map", {}))

    rows = []
    for label_key, algorithm, matrix, run_name in RUN_ORDER:
        if label_key not in labels_dict:
            print(f"[WARN] Missing labels for {label_key}. Skipping.")
            continue

        pred_labels = np.asarray(labels_dict[label_key])
        dist = get_distance_matrix(data, matrix)
        features = get_feature_matrix(data, matrix)
        internal = safe_internal_metrics(features, dist, pred_labels)
        external = safe_external_metrics(true_labels, pred_labels)

        rows.append(
            {
                "run": run_name,
                "label_column": label_key,
                "algorithm": algorithm,
                "matrix": matrix,
                "n_samples": int(len(pred_labels)),
                "n_clusters": count_clusters(pred_labels),
                "silhouette": round(internal["silhouette"], 4) if np.isfinite(internal["silhouette"]) else np.nan,
                "davies_bouldin": round(internal["davies_bouldin"], 4) if np.isfinite(internal["davies_bouldin"]) else np.nan,
                "calinski_harabasz": round(internal["calinski_harabasz"], 4) if np.isfinite(internal["calinski_harabasz"]) else np.nan,
                "ari": round(external["ari"], 4) if np.isfinite(external["ari"]) else np.nan,
                "nmi": round(external["nmi"], 4) if np.isfinite(external["nmi"]) else np.nan,
            }
        )

    if not rows:
        fail("No clustering labels were found for validation.")
    return pd.DataFrame(rows)


def add_bar_labels(ax: plt.Axes, bars, values: list[float], metric: str) -> None:
    clean_values = [v for v in values if np.isfinite(v)]
    if not clean_values:
        return
    max_val = max(clean_values)
    offset = max(abs(max_val) * 0.025, 0.01)
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        text = f"{value:.4f}" if metric != "n_clusters" else str(int(value))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            text,
            ha="center",
            va="bottom",
            fontsize=7.2,
            rotation=90 if len(text) > 6 else 0,
            color=IEEE_DARK_GRAY,
        )


def plot_metric_side_by_side(validation_df: pd.DataFrame, metric: str, ylabel: str, title: str, outpath: Path) -> None:
    df = validation_df.copy()
    values = df[metric].astype(float).tolist()
    x = np.arange(len(df))
    colors = [RUN_COLORS.get(run, IEEE_GRAY) for run in df["run"]]

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    bars = ax.bar(x, values, width=0.52, color=colors, edgecolor=IEEE_DARK_GRAY, linewidth=0.6)
    add_bar_labels(ax, bars, values, metric)

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(v, 16) for v in df["run"]], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    clean_values = [v for v in values if np.isfinite(v)]
    if clean_values:
        if metric in {"silhouette", "ari", "nmi"}:
            ax.set_ylim(min(0.0, min(clean_values) - 0.05), min(1.0, max(clean_values) + 0.12))
        else:
            ax.set_ylim(0, max(clean_values) * 1.25)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.22)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")


def plot_validation_dashboard(validation_df: pd.DataFrame, outpath: Path) -> None:
    metrics = [
        ("silhouette", "Silhouette"),
        ("davies_bouldin", "Davies Bouldin"),
        ("calinski_harabasz", "Calinski Harabasz"),
        ("ari", "ARI"),
        ("nmi", "NMI"),
        ("n_clusters", "Clusters"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16.2, 13.5))
    x = np.arange(len(validation_df))
    colors = [RUN_COLORS.get(run, IEEE_GRAY) for run in validation_df["run"]]

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = validation_df[metric].astype(float).tolist()
        bars = ax.bar(x, values, width=0.55, color=colors, edgecolor=IEEE_DARK_GRAY, linewidth=0.55)
        add_bar_labels(ax, bars, values, metric)
        ax.set_xticks(x)
        ax.set_xticklabels([wrap_label(v, 13) for v in validation_df["run"]], fontsize=7.0)
        ax.set_title(title, fontweight="bold", fontsize=10.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        clean_values = [v for v in values if np.isfinite(v)]
        if clean_values:
            if metric in {"silhouette", "ari", "nmi"}:
                ax.set_ylim(min(0.0, min(clean_values) - 0.05), min(1.0, max(clean_values) + 0.12))
            else:
                ax.set_ylim(0, max(clean_values) * 1.25)

    fig.suptitle("Side by side validation comparison across clustering runs", fontsize=13.5, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.06, hspace=0.50, wspace=0.22)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {outpath}")


def save_best_run_summary(validation_df: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    df = validation_df.copy()
    df["silhouette_rank"] = df["silhouette"].rank(ascending=False, method="dense")
    df["davies_bouldin_rank"] = df["davies_bouldin"].rank(ascending=True, method="dense")
    df["calinski_harabasz_rank"] = df["calinski_harabasz"].rank(ascending=False, method="dense")
    if df["ari"].notna().any():
        df["ari_rank"] = df["ari"].rank(ascending=False, method="dense")
    else:
        df["ari_rank"] = np.nan
    if df["nmi"].notna().any():
        df["nmi_rank"] = df["nmi"].rank(ascending=False, method="dense")
    else:
        df["nmi_rank"] = np.nan

    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    df["mean_rank"] = df[rank_cols].mean(axis=1, skipna=True).round(2)
    df = df.sort_values(["mean_rank", "silhouette_rank"], ascending=True)
    df.to_csv(outpath, index=False)
    print(f"[DONE] {outpath}")
    return df


def main() -> None:
    data = load_clustering_data(CLUSTERING_PKL)
    validation_df = validate_runs(data)
    validation_df.to_csv(OUT_DIR / "validation_metrics_all_runs.csv", index=False)
    print(f"[DONE] {OUT_DIR / 'validation_metrics_all_runs.csv'}")

    save_best_run_summary(validation_df, OUT_DIR / "validation_ranked_summary.csv")

    plot_validation_dashboard(validation_df, OUT_DIR / "fig_validation_metrics_side_by_side.png")
    plot_metric_side_by_side(
        validation_df,
        "silhouette",
        "Silhouette score",
        "Silhouette validation comparison",
        OUT_DIR / "fig_validation_silhouette_side_by_side.png",
    )
    plot_metric_side_by_side(
        validation_df,
        "davies_bouldin",
        "Davies Bouldin index, lower is better",
        "Davies Bouldin validation comparison",
        OUT_DIR / "fig_validation_davies_bouldin_side_by_side.png",
    )
    plot_metric_side_by_side(
        validation_df,
        "calinski_harabasz",
        "Calinski Harabasz score, higher is better",
        "Calinski Harabasz validation comparison",
        OUT_DIR / "fig_validation_calinski_harabasz_side_by_side.png",
    )
    plot_metric_side_by_side(
        validation_df,
        "ari",
        "Adjusted Rand Index",
        "External validation comparison, ARI",
        OUT_DIR / "fig_validation_ari_side_by_side.png",
    )
    plot_metric_side_by_side(
        validation_df,
        "nmi",
        "Normalized Mutual Information",
        "External validation comparison, NMI",
        OUT_DIR / "fig_validation_nmi_side_by_side.png",
    )

    print("\n[INFO] Validation comparison:")
    print(validation_df.to_string(index=False))
    print("\n[DONE] Validation complete for all clustering runs.")


if __name__ == "__main__":
    main()
