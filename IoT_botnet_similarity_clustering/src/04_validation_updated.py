"""04_validation_updated.py — Step 4 of 6: Cluster Quality Validation
Internal and external validation for Hierarchical + Spectral clustering over Cosine and Jaccard matrices.
"""

import os
import sys
import pickle
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)

from log_utils import setup_script_logging
from viz_utils import set_academic_style, style_table, table_set_widths, wrap_label

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = "output/validation"
PKL = "output/clustering/clustering_results.pkl"
LBL = "output/features/ground_truth_labels.csv"
ALGORITHMS = ["hierarchical", "spectral"]
KNOWN_FAMILIES = {"mirai", "gafgyt", "hajime", "tsunami", "bashlite", "dofloo", "lightaidra"}

GREEN = "#548235"
ORANGE = "#C55A11"
RED = "#A23B3B"
GRAY = "#7F7F7F"
LIGHT_GREEN = "#E8F5E9"
LIGHT_ORANGE = "#FFF3E0"
LIGHT_RED = "#FFEBEE"
LIGHT_GRAY = "#F5F5F5"

IEEE_BLUE = "#1F4E79"
IEEE_TEAL = "#5B9BD5"
IEEE_GRAY = "#7F7F7F"
IEEE_DARK_GRAY = "#404040"
IEEE_ORANGE = "#C55A11"
IEEE_GREEN = "#548235"
IEEE_RED = "#A23B3B"
METRIC_COLORS = {"ARI": IEEE_BLUE, "NMI": IEEE_TEAL, "Purity": IEEE_GRAY}
RUN_COLORS = {
    "Hierarchical Cosine": IEEE_BLUE,
    "Hierarchical Jaccard": IEEE_TEAL,
    "Spectral Cosine": IEEE_GRAY,
    "Spectral Jaccard": IEEE_ORANGE,
}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
setup_script_logging("04_validation.py")
set_academic_style()


def require_file(path: str, message: str) -> None:
    if not os.path.exists(path):
        print(f"[ERROR] {message}")
        sys.exit(1)


def title_algo(algo: str) -> str:
    return {"hierarchical": "Hierarchical", "spectral": "Spectral"}.get(algo, algo.title())


def cluster_count(labels: np.ndarray) -> int:
    labels = np.asarray(labels)
    return int(len(set(labels)))


def clean_distance(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
    dist = np.clip(dist, 0.0, 1.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist


def score_band(value, metric: str) -> str:
    if pd.isna(value):
        return "na"

    metric = metric.lower()
    if metric in {"silhouette", "mean_sil", "mean_silhouette"}:
        if value >= 0.50:
            return "good"
        if value >= 0.25:
            return "moderate"
        return "poor"

    if metric in {
        "ari", "nmi", "ami", "fmi", "purity", "homogeneity",
        "completeness", "v_measure", "dominant_family_percent",
    }:
        if metric == "dominant_family_percent":
            if value >= 80:
                return "good"
            if value >= 60:
                return "moderate"
            return "poor"
        if value >= 0.80:
            return "good"
        if value >= 0.60:
            return "moderate"
        return "poor"

    if metric in {"davies_bouldin", "db"}:
        if value <= 0.60:
            return "good"
        if value <= 1.20:
            return "moderate"
        return "poor"

    return "na"


def score_color(value, metric: str) -> str:
    return {"good": GREEN, "moderate": ORANGE, "poor": RED}.get(score_band(value, metric), GRAY)


def score_fill(value, metric: str) -> str:
    return {"good": LIGHT_GREEN, "moderate": LIGHT_ORANGE, "poor": LIGHT_RED}.get(score_band(value, metric), LIGHT_GRAY)


def relative_color(values, value) -> str:
    vals = pd.Series(values).dropna().astype(float)
    if pd.isna(value) or vals.empty:
        return GRAY
    q1 = vals.quantile(0.33)
    q2 = vals.quantile(0.66)
    if value >= q2:
        return GREEN
    if value >= q1:
        return ORANGE
    return RED


def relative_fill(values, value) -> str:
    color = relative_color(values, value)
    if color == GREEN:
        return LIGHT_GREEN
    if color == ORANGE:
        return LIGHT_ORANGE
    if color == RED:
        return LIGHT_RED
    return LIGHT_GRAY


def color_metric_column(tbl, rows, col_idx: int, metric_key: str, relative: bool = False) -> None:
    vals = [row.get(metric_key, np.nan) for row in rows]
    for ridx, row in enumerate(rows, start=1):
        value = row.get(metric_key, np.nan)
        cell = tbl[(ridx, col_idx)]
        if relative:
            cell.set_facecolor(relative_fill(vals, value))
            cell.get_text().set_color(relative_color(vals, value))
        else:
            cell.set_facecolor(score_fill(value, metric_key))
            cell.get_text().set_color(score_color(value, metric_key))
        cell.get_text().set_fontweight("bold")


def safe_internal_metrics(labels: np.ndarray, dist: np.ndarray, features: np.ndarray):
    """Compute internal cluster quality metrics.

    Metric space note (V-01):
      - Silhouette score: computed with metric='precomputed' using the exact
        cosine or Jaccard distance matrix — fully consistent with the clustering
        geometry.
      - Davies-Bouldin index: computed via davies_bouldin_score(features, labels)
        which uses Euclidean distance internally.  For cosine runs 'features' is
        the TF-IDF dense matrix; for Jaccard runs it is the binary feature matrix.
        Both are Euclidean approximations of the true cluster geometry and should
        be interpreted as supporting indicators only.  Disclose in thesis methods.
      - Calinski-Harabasz index: same Euclidean-space caveat as DB above.
    """
    labels = np.asarray(labels)
    clusters = cluster_count(labels)

    if clusters < 2:
        return np.nan, np.nan, np.nan, "Fewer than two clusters"

    try:
        sil = float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        sil = np.nan

    try:
        db = float(davies_bouldin_score(features, labels))
    except Exception:
        db = np.nan

    try:
        ch = float(calinski_harabasz_score(features, labels))
    except Exception:
        ch = np.nan

    return sil, db, ch, "All samples included"


def cleanup_obsolete_per_cluster_outputs() -> None:
    obsolete = [
        "per_cluster_silhouette_all_algorithms.csv",
        "per_cluster_silhouette_hierarchical.csv",
        "per_cluster_silhouette_spectral.csv",
        "per_cluster_silhouette.csv",
        "fig_per_cluster_silhouette_hierarchical.png",
        "fig_per_cluster_silhouette_spectral.png",
    ]
    for name in obsolete:
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"[CLEAN] Removed obsolete output: {path}")


def clustering_purity(y_true, y_pred) -> float:
    tab = pd.crosstab(pd.Series(y_pred, name="cluster"), pd.Series(y_true, name="family"))
    if tab.empty:
        return np.nan
    return float(tab.max(axis=1).sum() / tab.values.sum())


def purity_level(value) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 80:
        return "High"
    if value >= 60:
        return "Moderate"
    return "Low"


def family_count_text(counts: pd.Series) -> str:
    parts = [f"{fam}:{int(count)}" for fam, count in counts.items() if int(count) > 0]
    return "; ".join(parts) if parts else "N/A"


def wrap(value, width=42) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False, break_on_hyphens=False))


def make_agglomerative(k: int, dist: np.ndarray) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(dist)


def make_spectral(k: int, sim: np.ndarray) -> np.ndarray:
    return SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=20,
        random_state=42,
    ).fit_predict(sim)


require_file(PKL, f"{PKL} not found. Run 03_clustering.py first.")

with open(PKL, "rb") as f:
    cr = pickle.load(f)

ids = list(cr["ids"])
X_dense = np.asarray(cr["X_dense"], dtype=float)
X_bin = np.asarray(cr.get("X_bin", X_dense > 0), dtype=float)
K = int(cr.get("K", 4))
n = len(ids)

labels_dict = cr.get("labels", {})
if not labels_dict:
    labels_dict = {}
    if "hier_labels" in cr:
        labels_dict["hierarchical_cosine"] = np.asarray(cr["hier_labels"])
    if "spectral_labels" in cr:
        labels_dict["spectral_cosine"] = np.asarray(cr["spectral_labels"])

matrices_raw = cr.get("matrices", {})
if matrices_raw:
    MATRICES = {
        "cosine": {
            "display": "Cosine",
            "dist": clean_distance(matrices_raw["cosine"]["dist"]),
            "sim": np.asarray(matrices_raw["cosine"].get("sim", 1.0 - matrices_raw["cosine"]["dist"]), dtype=float),
            "features": np.asarray(matrices_raw["cosine"].get("features", X_dense), dtype=float),
        },
        "jaccard": {
            "display": "Jaccard",
            "dist": clean_distance(matrices_raw["jaccard"]["dist"]),
            "sim": np.asarray(matrices_raw["jaccard"].get("sim", 1.0 - matrices_raw["jaccard"]["dist"]), dtype=float),
            "features": np.asarray(matrices_raw["jaccard"].get("features", X_bin), dtype=float),
        },
    }
else:
    MATRICES = {
        "cosine": {
            "display": "Cosine",
            "dist": clean_distance(cr["cos_dist"]),
            "sim": np.asarray(cr.get("cos_sim", 1.0 - cr["cos_dist"]), dtype=float),
            "features": X_dense,
        }
    }
    if "jac_dist" in cr:
        MATRICES["jaccard"] = {
            "display": "Jaccard",
            "dist": clean_distance(cr["jac_dist"]),
            "sim": np.asarray(cr.get("jac_sim", 1.0 - cr["jac_dist"]), dtype=float),
            "features": X_bin,
        }

print(f"[INFO] Validating {n} samples")
print("[INFO] Expected clustering runs: hierarchical_cosine, hierarchical_jaccard, spectral_cosine, spectral_jaccard")

internal_rows = []

for matrix_name, mspec in MATRICES.items():
    for algo in ALGORITHMS:
        key = f"{algo}_{matrix_name}"
        if key not in labels_dict:
            print(f"[WARN] {key} not found in clustering_results.pkl. Skipped.")
            continue

        labels = np.asarray(labels_dict[key], dtype=int)
        sil, db, ch, note = safe_internal_metrics(labels, mspec["dist"], mspec["features"])

        internal_rows.append({
            "algorithm": title_algo(algo),
            "matrix": mspec["display"],
            "label_column": key,
            "n_clusters": cluster_count(labels),
            "n_noise": 0,
            "silhouette": round(sil, 4) if np.isfinite(sil) else np.nan,
            "davies_bouldin": round(db, 4) if np.isfinite(db) else np.nan,
            "calinski_harabasz": round(ch, 2) if np.isfinite(ch) else np.nan,
            "metric_basis": f"Silhouette uses {mspec['display']} precomputed distance; DB and CH use matching feature space",
            "note": note,
        })


internal_df = pd.DataFrame(internal_rows)

internal_df.to_csv(f"{OUT_DIR}/internal_validation_summary.csv", index=False)
cleanup_obsolete_per_cluster_outputs()

print("[INFO] Internal validation summary:")
if not internal_df.empty:
    print(internal_df[[
        "algorithm", "matrix", "n_clusters", "silhouette",
        "davies_bouldin", "calinski_harabasz"
    ]].to_string(index=False))
else:
    print("No internal validation rows produced.")


external_rows = []
composition_frames = []

if os.path.exists(LBL):
    label_df = pd.read_csv(LBL).set_index("sample_id")

    for matrix_name, mspec in MATRICES.items():
        for algo in ALGORITHMS:
            key = f"{algo}_{matrix_name}"
            if key not in labels_dict:
                continue

            labels = np.asarray(labels_dict[key], dtype=int)
            y_true, y_pred = [], []

            for i, sid in enumerate(ids):
                fam = str(label_df["family"].get(sid, "unlabeled")).lower()
                if fam in KNOWN_FAMILIES:
                    y_true.append(fam)
                    y_pred.append(int(labels[i]))

            if len(y_true) >= 6:
                ari = adjusted_rand_score(y_true, y_pred)
                nmi = normalized_mutual_info_score(y_true, y_pred)
                ami = adjusted_mutual_info_score(y_true, y_pred)
                fmi = fowlkes_mallows_score(y_true, y_pred)
                purity = clustering_purity(y_true, y_pred)
                hom, comp, vme = homogeneity_completeness_v_measure(y_true, y_pred)
                note = "Known family subset only"

                tab = pd.crosstab(pd.Series(y_true, name="true_family"), pd.Series(y_pred, name="cluster"))
                comp_rows = []
                for cid in sorted(tab.columns):
                    col = tab[cid].sort_values(ascending=False)
                    total = int(col.sum())
                    dominant = str(col.idxmax()) if total else "N/A"
                    dominant_n = int(col.max()) if total else 0
                    dominant_pct = round(dominant_n / total * 100, 2) if total else np.nan
                    comp_rows.append({
                        "algorithm": title_algo(algo),
                        "matrix": mspec["display"],
                        "label_column": key,
                        "cluster": int(cid),
                        "total_labeled_samples": total,
                        "dominant_family": dominant,
                        "dominant_family_count": dominant_n,
                        "dominant_family_percent": dominant_pct,
                        "families_present": int((col > 0).sum()) if total else 0,
                        "family_counts": family_count_text(col),
                        "purity_level": purity_level(dominant_pct),
                    })
                composition_frames.append(pd.DataFrame(comp_rows))
            else:
                ari = nmi = ami = fmi = purity = hom = comp = vme = np.nan
                note = "Insufficient labeled samples"

            external_rows.append({
                "algorithm": title_algo(algo),
                "matrix": mspec["display"],
                "label_column": key,
                "n_labeled": len(y_true),
                "ARI": round(float(ari), 4) if pd.notna(ari) else np.nan,
                "NMI": round(float(nmi), 4) if pd.notna(nmi) else np.nan,
                "AMI": round(float(ami), 4) if pd.notna(ami) else np.nan,
                "FMI": round(float(fmi), 4) if pd.notna(fmi) else np.nan,
                "Purity": round(float(purity), 4) if pd.notna(purity) else np.nan,
                "Homogeneity": round(float(hom), 4) if pd.notna(hom) else np.nan,
                "Completeness": round(float(comp), 4) if pd.notna(comp) else np.nan,
                "V_measure": round(float(vme), 4) if pd.notna(vme) else np.nan,
                "note": note,
            })
else:
    print(f"[WARN] {LBL} not found. External validation skipped.")

external_df = pd.DataFrame(external_rows)
external_df.to_csv(f"{OUT_DIR}/external_validation_summary.csv", index=False)

if composition_frames:
    comp_all = pd.concat(composition_frames, ignore_index=True)
else:
    comp_all = pd.DataFrame(columns=[
        "algorithm", "matrix", "label_column", "cluster", "total_labeled_samples",
        "dominant_family", "dominant_family_count", "dominant_family_percent",
        "families_present", "family_counts", "purity_level",
    ])

comp_all.to_csv(f"{OUT_DIR}/cluster_composition_all_algorithms.csv", index=False)

for algo in ALGORITHMS:
    comp_all[comp_all["algorithm"] == title_algo(algo)].to_csv(
        f"{OUT_DIR}/cluster_composition_{algo}.csv", index=False
    )

for matrix_display in sorted(comp_all["matrix"].dropna().unique()):
    comp_all[comp_all["matrix"] == matrix_display].to_csv(
        f"{OUT_DIR}/cluster_composition_all_algo_with_{matrix_display.lower()}.csv", index=False
    )


summary_rows = []
for matrix_display in internal_df["matrix"].dropna().unique():
    int_sub = internal_df[internal_df["matrix"] == matrix_display].copy()
    ext_sub = external_df[external_df["matrix"] == matrix_display].copy() if not external_df.empty else pd.DataFrame()

    valid_sil = int_sub["silhouette"].dropna()
    valid_db = int_sub["davies_bouldin"].dropna()
    valid_ch = int_sub["calinski_harabasz"].dropna()

    row = {
        "matrix": matrix_display,
        "mean_silhouette": round(float(valid_sil.mean()), 4) if not valid_sil.empty else np.nan,
        "best_internal_algorithm": int_sub.sort_values("silhouette", ascending=False).iloc[0]["algorithm"] if not int_sub.empty else "N/A",
        "best_internal_silhouette": round(float(valid_sil.max()), 4) if not valid_sil.empty else np.nan,
        "mean_davies_bouldin": round(float(valid_db.mean()), 4) if not valid_db.empty else np.nan,
        "mean_calinski_harabasz": round(float(valid_ch.mean()), 2) if not valid_ch.empty else np.nan,
    }

    if not ext_sub.empty and ext_sub["ARI"].notna().any():
        valid_ari = ext_sub["ARI"].dropna()
        valid_nmi = ext_sub["NMI"].dropna()
        valid_purity = ext_sub["Purity"].dropna()
        row.update({
            "mean_ARI": round(float(valid_ari.mean()), 4) if not valid_ari.empty else np.nan,
            "mean_NMI": round(float(valid_nmi.mean()), 4) if not valid_nmi.empty else np.nan,
            "mean_Purity": round(float(valid_purity.mean()), 4) if not valid_purity.empty else np.nan,
            "best_external_algorithm": ext_sub.sort_values("ARI", ascending=False).iloc[0]["algorithm"],
            "best_external_ARI": round(float(valid_ari.max()), 4) if not valid_ari.empty else np.nan,
        })
    else:
        row.update({
            "mean_ARI": np.nan,
            "mean_NMI": np.nan,
            "mean_Purity": np.nan,
            "best_external_algorithm": "N/A",
            "best_external_ARI": np.nan,
        })

    summary_rows.append(row)

matrix_comparison = pd.DataFrame(summary_rows)
if not matrix_comparison.empty and matrix_comparison["mean_silhouette"].notna().any():
    matrix_comparison["rank_internal_silhouette"] = matrix_comparison["mean_silhouette"].rank(
        ascending=False, method="dense"
    ).astype(int)
else:
    matrix_comparison["rank_internal_silhouette"] = np.nan

if not matrix_comparison.empty and matrix_comparison["mean_ARI"].notna().any():
    matrix_comparison["rank_external_ARI"] = matrix_comparison["mean_ARI"].rank(
        ascending=False, method="dense"
    )
else:
    matrix_comparison["rank_external_ARI"] = np.nan

matrix_comparison.to_csv(f"{OUT_DIR}/matrix_comparison_summary.csv", index=False)


k_rows = []
for matrix_name, mspec in MATRICES.items():
    for algo in ALGORITHMS:
        for k_test in range(2, 11):
            try:
                if algo == "hierarchical":
                    labels = make_agglomerative(k_test, mspec["dist"])
                else:
                    labels = make_spectral(k_test, mspec["sim"])
                sil, db, ch, _ = safe_internal_metrics(labels, mspec["dist"], mspec["features"])
            except Exception:
                sil = db = ch = np.nan

            k_rows.append({
                "algorithm": title_algo(algo),
                "matrix": mspec["display"],
                "k": k_test,
                "silhouette": round(sil, 4) if np.isfinite(sil) else np.nan,
                "davies_bouldin": round(db, 4) if np.isfinite(db) else np.nan,
                "calinski_harabasz": round(ch, 2) if np.isfinite(ch) else np.nan,
                "selected_k": k_test == K,
            })

k_df = pd.DataFrame(k_rows)
k_df.to_csv(f"{OUT_DIR}/k_sweep.csv", index=False)


def plot_internal_summary() -> None:
    fig, (ax_tbl, ax) = plt.subplots(
        1, 2, figsize=(17.8, 7.2), gridspec_kw={"width_ratios": [1.35, 1.05]}
    )

    ax_tbl.axis("off")
    row_dicts = []
    rows = []

    for _, r in internal_df.iterrows():
        row = r.to_dict()
        row_dicts.append(row)
        rows.append([
            r["algorithm"],
            r["matrix"],
            str(int(r["n_clusters"])),
            f"{r['silhouette']:.4f}" if pd.notna(r["silhouette"]) else "N/A",
            f"{r['davies_bouldin']:.4f}" if pd.notna(r["davies_bouldin"]) else "N/A",
            f"{r['calinski_harabasz']:.2f}" if pd.notna(r["calinski_harabasz"]) else "N/A",
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Algorithm", "Matrix", "Clusters", "Sil", "DB", "CH"],
        cellLoc="center",
        loc="center",
        bbox=[0.03, 0.08, 0.94, 0.82],
    )
    style_table(tbl, fontsize=8.2, header_fontsize=8.0, alt1="#F8F9FA", alt2="#EEF4FB", row_height=0.12)
    table_set_widths(tbl, [0.20, 0.15, 0.12, 0.14, 0.14, 0.16])
    color_metric_column(tbl, row_dicts, 3, "silhouette")
    color_metric_column(tbl, row_dicts, 4, "davies_bouldin")
    color_metric_column(tbl, row_dicts, 5, "calinski_harabasz", relative=True)
    ax_tbl.set_title("Internal validation summary", fontweight="bold", pad=12)

    plot_df = internal_df.dropna(subset=["silhouette"]).copy()
    plot_df["run"] = plot_df["algorithm"] + " " + plot_df["matrix"]
    plot_df["run_label"] = plot_df["algorithm"] + "\n" + plot_df["matrix"]
    x = np.arange(len(plot_df))
    colors = [RUN_COLORS.get(v, IEEE_GRAY) for v in plot_df["run"]]

    bars = ax.bar(x, plot_df["silhouette"], width=0.50, color=colors, edgecolor=IEEE_DARK_GRAY, linewidth=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(v, 14) for v in plot_df["run_label"]], fontsize=8.4)
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette comparison", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.50, color=IEEE_GREEN, linestyle="--", linewidth=0.9, alpha=0.65)
    ax.axhline(0.25, color=IEEE_ORANGE, linestyle="--", linewidth=0.9, alpha=0.65)
    ax.set_ylim(0, min(1.0, max(float(plot_df["silhouette"].max()) + 0.08, 0.60)))

    for bar, val in zip(bars, plot_df["silhouette"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=IEEE_DARK_GRAY,
            fontweight="bold",
        )

    handles = [plt.Rectangle((0, 0), 1, 1, color=RUN_COLORS.get(v, IEEE_GRAY)) for v in plot_df["run"]]
    ax.legend(
        handles,
        plot_df["run"],
        title="Clustering run",
        fontsize=7.5,
        title_fontsize=8.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
    )

    fig.suptitle("Internal validation across cosine and Jaccard matrices", fontsize=12.8, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.28, left=0.04, right=0.98, wspace=0.28)
    plt.savefig(f"{OUT_DIR}/fig_internal_validation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {OUT_DIR}/fig_internal_validation.png")


def plot_external_summary() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 7.0))

    if external_df.empty or not external_df["ARI"].notna().any():
        ax.axis("off")
        ax.text(0.5, 0.55, "External validation unavailable", ha="center", va="center",
                fontsize=13, fontweight="bold")
    else:
        plot_df = external_df.dropna(subset=["ARI"]).copy()
        plot_df["run"] = plot_df["algorithm"] + "\n" + plot_df["matrix"]
        x = np.arange(len(plot_df))
        width = 0.22

        bar_specs = [
            ("ARI", x - width),
            ("NMI", x),
            ("Purity", x + width),
        ]
        for metric, xpos in bar_specs:
            bars = ax.bar(
                xpos,
                plot_df[metric],
                width=width,
                label=metric,
                color=METRIC_COLORS.get(metric, IEEE_GRAY),
                edgecolor=IEEE_DARK_GRAY,
                linewidth=0.55,
            )
            for bar, value in zip(bars, plot_df[metric]):
                if pd.notna(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.018,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.4,
                        color=IEEE_DARK_GRAY,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([wrap_label(v, 14) for v in plot_df["run"]], fontsize=8.2)
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("Score")
        ax.set_title("External validation against known botnet family labels", fontweight="bold", pad=12)
        ax.grid(axis="y", alpha=0.24, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.80, color=IEEE_GREEN, linestyle="--", linewidth=0.9, alpha=0.65)
        ax.axhline(0.60, color=IEEE_ORANGE, linestyle="--", linewidth=0.9, alpha=0.65)
        ax.legend(fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

    fig.subplots_adjust(top=0.88, bottom=0.26, left=0.07, right=0.98)
    plt.savefig(f"{OUT_DIR}/fig_external_validation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {OUT_DIR}/fig_external_validation.png")

def plot_matrix_comparison() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.axis("off")

    rows, row_dicts = [], []
    for _, r in matrix_comparison.iterrows():
        row_dicts.append(r.to_dict())
        best_internal = (
            f"{r['best_internal_algorithm']} ({r['best_internal_silhouette']:.4f})"
            if pd.notna(r["best_internal_silhouette"]) else "N/A"
        )
        best_external = (
            f"{r['best_external_algorithm']} ({r['best_external_ARI']:.4f})"
            if pd.notna(r["best_external_ARI"]) else "N/A"
        )
        rows.append([
            r["matrix"],
            f"{r['mean_silhouette']:.4f}" if pd.notna(r["mean_silhouette"]) else "N/A",
            best_internal,
            f"{r['mean_ARI']:.4f}" if pd.notna(r["mean_ARI"]) else "N/A",
            best_external,
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=["Matrix", "Mean Sil", "Best internal", "Mean ARI", "Best external"],
        cellLoc="center",
        loc="center",
        bbox=[0.03, 0.22, 0.94, 0.58],
    )
    style_table(tbl, fontsize=8.5, header_fontsize=8.3, alt1="#F8F9FA", alt2="#EEF4FB", row_height=0.17)
    table_set_widths(tbl, [0.16, 0.17, 0.27, 0.16, 0.24])
    color_metric_column(tbl, row_dicts, 1, "mean_silhouette")
    color_metric_column(tbl, row_dicts, 3, "mean_ARI")

    winner = "N/A"
    if not matrix_comparison.empty and matrix_comparison["mean_silhouette"].notna().any():
        winner = matrix_comparison.sort_values("mean_silhouette", ascending=False).iloc[0]["matrix"]

    ax.set_title(f"Matrix comparison summary, internal winner: {winner}", fontweight="bold", fontsize=12, pad=12)
    ax.text(
        0.5, 0.09,
        "Internal comparison uses mean silhouette across Hierarchical and Spectral. External comparison uses known family labels when available.",
        ha="center", va="center", fontsize=8.5, color="#555555", transform=ax.transAxes,
    )
    plt.savefig(f"{OUT_DIR}/fig_matrix_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {OUT_DIR}/fig_matrix_comparison.png")


def plot_k_sweep() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    line_styles = {"Hierarchical": "-", "Spectral": "--"}
    markers = {"Cosine": "o", "Jaccard": "s"}

    for (algo, matrix_display), sub in k_df.groupby(["algorithm", "matrix"]):
        run_name = f"{algo} {matrix_display}"
        ax.plot(
            sub["k"],
            sub["silhouette"],
            marker=markers.get(matrix_display, "o"),
            linestyle=line_styles.get(algo, "-"),
            linewidth=1.8,
            markersize=5.2,
            color=RUN_COLORS.get(run_name, IEEE_GRAY),
            label=run_name,
        )

    ax.axvline(K, linestyle=":", linewidth=1.3, color=IEEE_DARK_GRAY, label=f"selected k={K}")
    ax.axhline(0.50, color=IEEE_GREEN, linestyle="--", linewidth=0.9, alpha=0.55)
    ax.axhline(0.25, color=IEEE_ORANGE, linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K sweep for Hierarchical and Spectral clustering", fontweight="bold", pad=12)
    ax.set_xticks(sorted(k_df["k"].unique()))
    ax.grid(alpha=0.26, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.26, left=0.08, right=0.98)
    plt.savefig(f"{OUT_DIR}/fig_k_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {OUT_DIR}/fig_k_sweep.png")

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.2))
    for ax, algo in zip(axes, ["Hierarchical", "Spectral"]):
        ax.axis("off")
        sub_df = k_df[k_df["algorithm"] == algo].copy()
        rows, row_dicts = [], []
        for _, r in sub_df.iterrows():
            row_dicts.append(r.to_dict())
            rows.append([
                r["matrix"],
                str(int(r["k"])),
                f"{r['silhouette']:.4f}" if pd.notna(r["silhouette"]) else "N/A",
                f"{r['davies_bouldin']:.4f}" if pd.notna(r["davies_bouldin"]) else "N/A",
                f"{r['calinski_harabasz']:.2f}" if pd.notna(r["calinski_harabasz"]) else "N/A",
                "Selected" if bool(r["selected_k"]) else "",
            ])

        tbl = ax.table(
            cellText=rows,
            colLabels=["Matrix", "k", "Sil", "DB", "CH", "Selection"],
            cellLoc="center",
            loc="center",
            bbox=[0.02, 0.06, 0.96, 0.86],
        )
        style_table(tbl, fontsize=7.0, header_fontsize=7.0, alt1="#F8F9FA", alt2="#EEF4FB", row_height=0.045)
        table_set_widths(tbl, [0.18, 0.08, 0.16, 0.16, 0.18, 0.18])
        color_metric_column(tbl, row_dicts, 2, "silhouette")
        color_metric_column(tbl, row_dicts, 3, "davies_bouldin")
        color_metric_column(tbl, row_dicts, 4, "calinski_harabasz", relative=True)
        ax.set_title(f"{algo} k sweep", fontweight="bold", pad=10)

    fig.suptitle("K sweep tables by clustering algorithm", fontsize=12.6, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.05, left=0.03, right=0.98, wspace=0.08)
    # Only the canonical 'fig_k_sweep_table.png' is kept, matching the runner's
    # EXPECTED_OUTPUTS declaration and avoiding undeclared artefact proliferation.
    plt.savefig(f"{OUT_DIR}/fig_k_sweep_table.png", dpi=300, bbox_inches="tight")
    print(f"[DONE] {OUT_DIR}/fig_k_sweep_table.png")
    plt.close()


def build_composition_table(sub: pd.DataFrame, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(16.0, max(5.5, 2.8 + 0.28 * len(sub))))
    ax.axis("off")

    if sub.empty:
        ax.text(0.5, 0.55, title + " unavailable", ha="center", va="center", fontsize=12, fontweight="bold")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[DONE] {out_path}")
        return

    rows, row_dicts, levels = [], [], []
    for _, r in sub.iterrows():
        d = r.to_dict()
        row_dicts.append(d)
        levels.append(r.get("purity_level", "N/A"))
        rows.append([
            r.get("algorithm", ""),
            r.get("matrix", ""),
            str(int(r["cluster"])),
            str(int(r["total_labeled_samples"])),
            str(r["dominant_family"]),
            f"{r['dominant_family_percent']:.1f}%" if pd.notna(r["dominant_family_percent"]) else "N/A",
            wrap(r["family_counts"], 30),
            r["purity_level"],
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=["Algorithm", "Matrix", "Cluster", "Labeled n", "Dominant family", "Dominant %", "Family counts", "Purity"],
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.08, 0.96, 0.84],
    )
    style_table(tbl, fontsize=7.5, header_fontsize=7.4, alt1="#F8F9FA", alt2="#EEF4FB",
                row_height=0.062 if len(sub) > 12 else 0.075)
    table_set_widths(tbl, [0.15, 0.12, 0.08, 0.10, 0.16, 0.12, 0.19, 0.08])
    color_metric_column(tbl, row_dicts, 5, "dominant_family_percent")

    for ridx, level in enumerate(levels, start=1):
        cell = tbl[(ridx, 7)]
        if level == "High":
            cell.set_facecolor(LIGHT_GREEN)
            cell.get_text().set_color(GREEN)
        elif level == "Moderate":
            cell.set_facecolor(LIGHT_ORANGE)
            cell.get_text().set_color(ORANGE)
        elif level == "Low":
            cell.set_facecolor(LIGHT_RED)
            cell.get_text().set_color(RED)
        else:
            cell.set_facecolor(LIGHT_GRAY)
            cell.get_text().set_color(GRAY)
        cell.get_text().set_fontweight("bold")

    ax.set_title(title, fontweight="bold", pad=10)
    ax.text(0.02, 0.02, "Purity color scale: High = green, Moderate = orange, Low = red.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, color="#555555")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {out_path}")


def plot_cluster_composition() -> None:
    for algo in ALGORITHMS:
        sub = comp_all[comp_all["algorithm"] == title_algo(algo)].copy()
        build_composition_table(sub, f"Cluster composition, {title_algo(algo)}",
                                f"{OUT_DIR}/fig_cluster_composition_{algo}.png")

    for matrix_display in ["Cosine", "Jaccard"]:
        sub = comp_all[comp_all["matrix"] == matrix_display].copy()
        build_composition_table(sub, f"Cluster composition across algorithms, {matrix_display} matrix",
                                f"{OUT_DIR}/fig_cluster_composition_all_algo_with_{matrix_display.lower()}.png")

    build_composition_table(comp_all.copy(), "Cluster composition, all algorithm and matrix runs",
                            f"{OUT_DIR}/fig_cluster_composition_all_algorithms.png")


plot_internal_summary()
plot_external_summary()
plot_matrix_comparison()
plot_k_sweep()
plot_cluster_composition()

winner_internal = "N/A"
if not matrix_comparison.empty and matrix_comparison["mean_silhouette"].notna().any():
    winner_internal = matrix_comparison.sort_values("mean_silhouette", ascending=False).iloc[0]["matrix"]

winner_external = "N/A"
if not matrix_comparison.empty and matrix_comparison["mean_ARI"].notna().any():
    winner_external = matrix_comparison.sort_values("mean_ARI", ascending=False).iloc[0]["matrix"]

report_lines = [
    "Cluster Validation Report",
    "=========================",
    f"Samples: {n}",
    f"Selected fixed k for hierarchical and spectral clustering: {K}",
    "",
    "Validated runs:",
]

for _, r in internal_df.iterrows():
    report_lines.append(
        f"  {r['algorithm']} with {r['matrix']}: "
        f"clusters={r['n_clusters']}, silhouette={r['silhouette']}, "
        f"davies_bouldin={r['davies_bouldin']}, calinski_harabasz={r['calinski_harabasz']}"
    )

report_lines.extend([
    "",
    "Matrix comparison:",
    matrix_comparison.to_string(index=False) if not matrix_comparison.empty else "No matrix comparison available",
    "",
    f"Best matrix by internal mean silhouette: {winner_internal}",
    f"Best matrix by external mean ARI, if labels are available: {winner_external}",
    "",
    "Interpretation guidance:",
    "  Prefer the matrix and algorithm combination with higher silhouette and lower Davies-Bouldin score for internal structure.",
    "  If reliable family labels are available, use ARI, NMI, AMI, FMI, purity, homogeneity, completeness, and V-measure as external confirmation.",
    "  K sweep is reported for hierarchical clustering because it directly tests the fixed-k assumption used in both selected algorithms.",
])

with open(f"{OUT_DIR}/validation_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"[DONE] {OUT_DIR}/validation_report.txt")
print("\n[INFO] Matrix comparison summary:")
print(matrix_comparison.to_string(index=False) if not matrix_comparison.empty else "No matrix comparison available")
print("\n[DONE] Validation complete for Hierarchical and Spectral clustering over Cosine and Jaccard matrices.")
