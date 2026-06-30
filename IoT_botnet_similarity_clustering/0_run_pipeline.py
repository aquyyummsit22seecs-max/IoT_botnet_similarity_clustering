"""
0_run_pipeline.py — IoT Botnet Clustering Pipeline Orchestrator
Runs all six pipeline steps sequentially, validates expected outputs, and writes a consolidated log.
"""

import os, sys, time, datetime, subprocess

for d in ["output/features", "output/similarity", "output/clustering",
          "output/validation", "output/ttp_mitre", "output/intelligence", "output/logs"]:
    os.makedirs(d, exist_ok=True)

if sys.version_info < (3, 8):
    print(f"[ERROR] Python 3.8+ required. Found: {sys.version}")
    sys.exit(1)

SCRIPTS = [
    "src/01_feature_extraction.py",
    "src/02_similarity.py",
    "src/03_clustering.py",
    "src/04_validation_updated.py",
    "src/05_ttp_mitre.py",
    "src/06_end_report.py",
]

for s in SCRIPTS:
    if not os.path.exists(s):
        print(f"[ERROR] Required pipeline script not found: {s}")
        sys.exit(1)

EXPECTED_OUTPUTS = {
    "src/01_feature_extraction.py": [
        "output/features/behavioral_features.csv",
        "output/features/ground_truth_labels.csv",
        "output/features/feature_summary.csv",
        "output/features/fig_feature_stats.png",
        "output/features/fig_label_distribution.png",
        "output/features/fig_feature_coverage.png",
        "output/features/fig_feature_category_tables.png",
    ],
    "src/02_similarity.py": [
        "output/similarity/cosine_similarity_matrix.csv",
        "output/similarity/jaccard_similarity_matrix.csv",
        "output/similarity/cosine_similarity_subset_matrix.csv",
        "output/similarity/jaccard_similarity_subset_matrix.csv",
        "output/similarity/heatmap_subset_samples.csv",
        "output/similarity/similarity_stats.csv",
        "output/similarity/similarity_matrices.pkl",
        "output/similarity/fig_cosine_heatmap.png",
        "output/similarity/fig_jaccard_heatmap.png",
        "output/similarity/fig_similarity_comparison.png",
    ],
    "src/03_clustering.py": [
        "output/clustering/cluster_labels.csv",
        "output/clustering/algorithm_comparison.csv",
        "output/clustering/matrix_comparison_summary.csv",
        "output/clustering/clustering_results.pkl",
        "output/clustering/fig_hierarchical_cosine_cluster_sizes.png",
        "output/clustering/fig_hierarchical_jaccard_cluster_sizes.png",
        "output/clustering/fig_spectral_cosine_cluster_sizes.png",
        "output/clustering/fig_spectral_jaccard_cluster_sizes.png",
        "output/clustering/fig_algorithm_comparison.png",
        "output/clustering/fig_silhouette_comparison.png",
        "output/clustering/fig_cluster_count_comparison.png",
    ],
    "src/04_validation_updated.py": [
        "output/validation/internal_validation_summary.csv",
        "output/validation/external_validation_summary.csv",
        "output/validation/matrix_comparison_summary.csv",
        "output/validation/k_sweep.csv",
        "output/validation/per_cluster_silhouette.csv",
        "output/validation/per_cluster_silhouette_all_algorithms.csv",
        "output/validation/cluster_composition_all_algorithms.csv",
        "output/validation/cluster_composition_all_algo_with_cosine.csv",
        "output/validation/cluster_composition_all_algo_with_jaccard.csv",
        "output/validation/validation_report.txt",
        "output/validation/fig_k_sweep.png",
        "output/validation/fig_k_sweep_table.png",
        "output/validation/fig_internal_validation.png",
        "output/validation/fig_external_validation.png",
        "output/validation/fig_matrix_comparison.png",
        "output/validation/fig_cluster_composition_hierarchical.png",
        "output/validation/fig_cluster_composition_spectral.png",
        "output/validation/fig_cluster_composition_all_algo_with_cosine.png",
        "output/validation/fig_cluster_composition_all_algo_with_jaccard.png",
    ],
    "src/05_ttp_mitre.py": [
        "output/ttp_mitre/sample_ttps.csv",
        "output/ttp_mitre/ttp_frequency.csv",
        "output/ttp_mitre/mitigations.csv",
        "output/ttp_mitre/cluster_ttp_profile.csv",
        "output/ttp_mitre/strategic_cluster_profile.csv",
        "output/ttp_mitre/mitre_pipeline_sync_review.txt",
        "output/ttp_mitre/fig_top_ttps.png",
        "output/ttp_mitre/fig_cluster_ttp_heatmap.png",
        "output/ttp_mitre/fig_top_mitigations.png",
        "output/ttp_mitre/fig_tactic_distribution.png",
        "output/ttp_mitre/fig_strategic_clusters.png",
    ],
    "src/06_end_report.py": [
        "output/intelligence/final_cluster_intelligence_report.txt",
        "output/intelligence/final_cluster_intelligence_summary.csv",
    ],
}

pipeline_log = "output/logs/pipeline.log"
pipeline_start = datetime.datetime.now()

def log_line(line):
    ts_line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}"
    print(ts_line)
    with open(pipeline_log, "a", encoding="utf-8") as f:
        f.write(ts_line.rstrip() + "\n")

open(pipeline_log, "w").close()

print("\n" + "=" * 65)
print("  IoT Botnet Clustering Pipeline")
print("  Similarity Techniques + MITRE ATT&CK Mapping")
print("=" * 65 + "\n")
log_line(f"[PIPELINE START] started={pipeline_start.strftime('%Y-%m-%d %H:%M:%S')} | python={sys.version.split()[0]}")

env = os.environ.copy()
env["PIPELINE_START_TS"] = pipeline_start.strftime("%Y-%m-%d %H:%M:%S")
env["PIPELINE_LOG_PATH"] = pipeline_log

t0_total = time.time()
for script in SCRIPTS:
    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line(f"[STEP START] {script} | started={start_str}")
    t0 = time.time()
    r = subprocess.run([sys.executable, script], env=env)
    dt = time.time() - t0
    end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if r.returncode != 0:
        log_line(f"[STEP END]   {script} | ended={end_str} | elapsed={dt:.1f}s | status=FAILED")
        log_line("Pipeline halted. Fix the error above and rerun.")
        sys.exit(1)

    log_line(f"[STEP END]   {script} | ended={end_str} | elapsed={dt:.1f}s | status=SUCCESS")
    missing = [p for p in EXPECTED_OUTPUTS.get(script, []) if not os.path.exists(p)]
    if missing:
        for m in missing:
            log_line(f"  [WARN] Missing expected output: {m}")
    else:
        log_line(f"  [OK] All {len(EXPECTED_OUTPUTS.get(script, []))} expected outputs produced")

total = time.time() - t0_total
pipeline_end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_line(f"[PIPELINE END] ended={pipeline_end_str} | elapsed={total:.1f}s | status=SUCCESS")
print(f"\n{'=' * 65}")
print(f"  Pipeline complete in {total:.1f}s")
print(f"  Log: {pipeline_log}")
print(f"{'=' * 65}\n")
