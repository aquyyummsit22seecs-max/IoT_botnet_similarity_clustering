"""06_intelligence_report.py — Step 6 of 6: Final Intelligence Report Generator
Synthesises all upstream pipeline outputs into a narrative report and structured CSV summary.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from log_utils import setup_script_logging
except Exception:  # pragma: no cover
    setup_script_logging = None  # type: ignore[assignment]

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_NAME = "06_intelligence_report.py"
OUT_INTEL   = Path("output/intelligence")
OUT_LOGS    = Path("output/logs")
OUT_INTEL.mkdir(parents=True, exist_ok=True)
OUT_LOGS.mkdir(parents=True, exist_ok=True)

if setup_script_logging:
    setup_script_logging(SCRIPT_NAME)

# ── Human-readable feature label map ─────────────────────────────────────────
READABLE: dict[str, str] = {
    "net_cnc":            "C2 Communication",
    "net_cnc_tcp":        "TCP C2 Channel",
    "net_cnc_udp":        "UDP C2 Channel",
    "net_scanning":       "Port Scanning",
    "net_scan_telnet":    "Telnet Scan",
    "net_scan_tr069":     "TR069 Router Exploit Scan",
    "net_telnet_brute":   "Telnet Brute Force Login",
    "net_syn_scan":       "SYN Scan or Flood Signal",
    "net_ddos":           "DDoS Traffic",
    "net_dns":            "DNS Activity",
    "net_p2p":            "Peer-to-Peer C2",
    "net_blacklisted_ip": "Blacklisted IP Contact",
    "net_http_exploit":   "HTTP Exploit Attempt",     # now correctly populated (Patch F-01)
    "beh_watchdog":       "Watchdog Persistence",
    "beh_proc_net":       "Network Connection Enumeration",
    "beh_proc_fd_enum":   "Process FD Enumeration",
    "beh_proc_masquerade":"Process Masquerading",
    "beh_stage2_drop":    "Stage-2 Payload Drop",
    "beh_recon":          "System Reconnaissance",
    "beh_antivm":         "Anti-VM Behaviour",
    "beh_antidbg":        "Anti-Debug Behaviour",
    "beh_persistence":    "Persistence Behaviour",
    "beh_file_removal":   "File Deletion / Cleanup",
    "beh_proc_inject":    "Process Injection",
    "beh_kernel_module":  "Kernel Module Activity",
    "binary_packed":      "Packed Binary",
    "high_entropy":       "High-Entropy Binary",
    "binary_stripped":    "Stripped Binary",
    "suspicious_strings": "Suspicious Strings",
    "arch_mips":          "MIPS Architecture",
    "arch_arm":           "ARM Architecture",
    "arch_x86":           "x86 Architecture",
    "sys_exec":           "Process Execution",
    "sys_fork":           "Process Forking",
    "sys_clone":          "Process Clone",
    "sys_kill":           "Process Termination",
    "sys_prctl":          "Process Control",
    "sys_daemonize":      "Daemonisation",
    "sys_socket":         "Socket Creation",
    "sys_connect":        "Outbound Connect",
    "sys_bind":           "Port Binding",
    "sys_listen":         "Socket Listen",
    "sys_send":           "Data Send",
    "sys_sendto":         "UDP Send",
    "sys_recv":           "Data Receive",
    "sys_recvfrom":       "UDP Receive",
    "sys_recvmsg":        "Message Receive",
    "sys_select":         "I/O Multiplexing",
    "sys_open":           "File Open",
    "sys_read":           "File Read",
    "sys_write":          "File Write",
    "sys_dir_enum":       "Directory Enumeration",
    "sys_mmap":           "Memory Mapping",
    "sys_mprotect":       "Memory Protection Change",
    "sys_sleep":          "Sleep / Timing",
    "env_dynamic_ran":    "Dynamic Analysis Confirmed",
    "prog_cnc_string":    "C2 String in Program Log",
}

# Ordered preference for selecting the cluster column from cluster_labels.csv
CLUSTER_COLUMN_CANDIDATES = [
    "hierarchical",
    "spectral",
    "hierarchical_cosine",
    "hierarchical_jaccard",
    "spectral_cosine",
    "spectral_jaccard",
]


# ── CSV helpers ───────────────────────────────────────────────────────────────

def safe_csv(path: str | Path) -> Optional[pd.DataFrame]:
    """Load a CSV quietly; return None if absent or unreadable (optional inputs)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def require_csv(path: str | Path, description: str) -> pd.DataFrame:
    """Load a required CSV; print an actionable error and exit if missing.

    PATCH I-01: previously the first required read had no guard, producing a
    raw FileNotFoundError traceback.  Now produces a clear [ERROR] message
    directing the user to the correct prerequisite step.
    """
    df = safe_csv(path)
    if df is None:
        print(f"[ERROR] Missing required {description}: {path}")
        print(f"[ERROR] Ensure the upstream pipeline step completed successfully "
              f"before running {SCRIPT_NAME}.")
        sys.exit(1)
    return df


def clean_id_columns(*dfs: Optional[pd.DataFrame]) -> None:
    """Coerce sample_id to str in-place for consistent join keys."""
    for df in dfs:
        if df is not None and "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str)


# ── Text helpers ──────────────────────────────────────────────────────────────

def wrap_text(
    value: object,
    width: int = 88,
    max_lines: Optional[int] = None,
) -> str:
    text = "" if pd.isna(value) else str(value)
    if not text.strip():
        return "N/A"
    parts: list[str] = []
    for part in text.split("; "):
        parts.extend(
            textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False)
            or [part]
        )
    if max_lines is not None and len(parts) > max_lines:
        parts = parts[:max_lines]
        parts[-1] = parts[-1].rstrip(" .,;") + " ..."
    return "\n".join(parts)


# ── Cluster-column selection ──────────────────────────────────────────────────

def choose_cluster_column(
    df_labels: pd.DataFrame,
    df_selected: Optional[pd.DataFrame],
) -> tuple[str, str]:
    """Return (column_name, selection_reason) for the best available cluster run."""
    # Honour Step 5's explicit selection if present.
    if (
        df_selected is not None
        and not df_selected.empty
        and "selected_cluster_column" in df_selected.columns
    ):
        col = str(df_selected.iloc[0]["selected_cluster_column"])
        if col in df_labels.columns:
            return col, "selected by Step 5 (highest silhouette)"

    # Fall back through ordered preference list.
    for col in CLUSTER_COLUMN_CANDIDATES:
        if col in df_labels.columns:
            return col, "auto-selected from available clustering outputs"

    # Last resort: first non-sample_id column.
    candidates = [c for c in df_labels.columns if c != "sample_id"]
    if not candidates:
        print("[ERROR] cluster_labels.csv contains no cluster label columns.")
        sys.exit(1)
    return candidates[0], "fallback to first available column"


# ── Validation text helpers ───────────────────────────────────────────────────

def selected_validation_text(
    df_internal: Optional[pd.DataFrame],
    cluster_col: str,
) -> str:
    if df_internal is None or df_internal.empty:
        return "Validation metrics unavailable"
    if "label_column" not in df_internal.columns:
        return "Validation summary does not contain label_column"
    row = df_internal[df_internal["label_column"].astype(str) == str(cluster_col)]
    if row.empty:
        return "Validation row not found for selected cluster column"
    r = row.iloc[0]
    return (
        f"Algorithm={r.get('algorithm', 'N/A')}, "
        f"Matrix={r.get('matrix', 'N/A')}, "
        f"Silhouette={r.get('silhouette', 'N/A')}, "
        f"Davies-Bouldin={r.get('davies_bouldin', 'N/A')}, "
        f"Calinski-Harabasz={r.get('calinski_harabasz', 'N/A')}"
    )


def selected_overall_silhouette(
    df_internal: Optional[pd.DataFrame],
    cluster_col: str,
) -> str:
    if df_internal is None or df_internal.empty or "label_column" not in df_internal.columns:
        return "N/A"
    row = df_internal[df_internal["label_column"].astype(str) == str(cluster_col)]
    if row.empty or "silhouette" not in row.columns:
        return "N/A"
    val = pd.to_numeric(row.iloc[0]["silhouette"], errors="coerce")
    return f"{float(val):.4f}" if pd.notna(val) else "N/A"


# ── Per-cluster data helpers ──────────────────────────────────────────────────

def top_family_labels(
    df_lbl_gt: Optional[pd.DataFrame],
    sample_ids: list[str],
) -> str:
    if df_lbl_gt is None or df_lbl_gt.empty or "family" not in df_lbl_gt.columns:
        return "No labeled samples"
    mask = df_lbl_gt["sample_id"].astype(str).isin(sample_ids)
    fam_counts = df_lbl_gt[mask]["family"].value_counts()
    if fam_counts.empty:
        return "No labeled samples"
    return "; ".join(f"{fam}: {count}" for fam, count in fam_counts.items())


def get_strategic_value(
    df_strat: Optional[pd.DataFrame],
    cluster_id: int,
    column: str,
    default: str,
) -> str:
    if df_strat is None or df_strat.empty or "cluster" not in df_strat.columns:
        return default
    rows = df_strat[df_strat["cluster"].astype(str) == str(cluster_id)]
    if rows.empty or column not in rows.columns:
        return default
    value = rows.iloc[0].get(column, default)
    if pd.isna(value) or str(value).strip() == "":
        return default
    return str(value)


def top_ttp_text(
    df_cl_ttp: Optional[pd.DataFrame],
    cluster_id: int,
    limit: int = 5,
) -> str:
    if df_cl_ttp is None or df_cl_ttp.empty or "cluster" not in df_cl_ttp.columns:
        return "No TTPs extracted"
    ct = df_cl_ttp[df_cl_ttp["cluster"].astype(str) == str(cluster_id)].copy()
    if ct.empty:
        return "No TTPs extracted"
    sort_cols = [c for c in ["prevalence", "sample_count"] if c in ct.columns]
    if sort_cols:
        ct = ct.sort_values(sort_cols, ascending=False)
    ct = ct.head(limit)
    parts = []
    for _, r in ct.iterrows():
        tid  = r.get("ttp_id", "TTP")
        name = r.get("official_name", r.get("technique_name", "Unknown technique"))
        if "prevalence" in ct.columns and pd.notna(r.get("prevalence")):
            parts.append(f"{tid} {name} ({float(r['prevalence']) * 100:.0f}%)")
        else:
            parts.append(f"{tid} {name}")
    return "; ".join(parts)


def cluster_feature_profile(
    df_feat: pd.DataFrame,
    feat_cols: list[str],
    sample_ids: list[str],
    threshold: float = 0.05,
) -> pd.Series:
    subset = df_feat[df_feat["sample_id"].astype(str).isin(sample_ids)]
    if subset.empty:
        return pd.Series(dtype=float)
    values = subset[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).mean()
    return values[values >= threshold].sort_values(ascending=False).head(10)


# ── Cluster summary table ─────────────────────────────────────────────────────

def build_cluster_summary(
    df_feat: pd.DataFrame,
    df_labels: pd.DataFrame,
    df_lbl_gt: Optional[pd.DataFrame],
    df_cl_ttp: Optional[pd.DataFrame],
    df_strat: Optional[pd.DataFrame],
    cluster_col: str,
) -> pd.DataFrame:
    feat_cols = [c for c in df_feat.columns if c != "sample_id"]
    work = (
        df_labels[["sample_id", cluster_col]]
        .rename(columns={cluster_col: "cluster"})
        .copy()
    )
    work["sample_id"] = work["sample_id"].astype(str)
    work["cluster"]   = work["cluster"].astype(int)
    total_n = len(df_feat)

    rows = []
    for cid in sorted(work["cluster"].dropna().unique()):
        cid = int(cid)
        sample_ids = work[work["cluster"] == cid]["sample_id"].astype(str).tolist()
        top_features = cluster_feature_profile(df_feat, feat_cols, sample_ids)
        feature_text = "; ".join(
            f"{READABLE.get(feat, feat)} ({float(score) * 100:.1f}%)"
            for feat, score in top_features.items()
        ) or "No dominant behavioural indicators above threshold"

        rows.append({
            "cluster":                   cid,
            "cluster_size":              len(sample_ids),
            "pct_of_dataset":            round(len(sample_ids) / max(total_n, 1) * 100, 2),
            "risk_level":                get_strategic_value(df_strat, cid, "risk_level",        "MEDIUM"),
            "archetype":                 get_strategic_value(df_strat, cid, "archetype",          "Unknown"),
            "dominant_tactic":           get_strategic_value(df_strat, cid, "dominant_tactic",    "Unknown"),
            "family_labels":             top_family_labels(df_lbl_gt, sample_ids),
            "top_behavioral_indicators": feature_text,
            "top_attack_techniques":     top_ttp_text(df_cl_ttp, cid),
            "recommended_mitigations":   get_strategic_value(df_strat, cid, "recommended_mitigations",
                                                              "No mitigations available"),
            "description":               get_strategic_value(df_strat, cid, "description",
                                                              "No strategic profile available"),
        })
    return pd.DataFrame(rows)


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(
    df_feat:     pd.DataFrame,
    df_labels:   pd.DataFrame,
    df_internal: Optional[pd.DataFrame],
    df_lbl_gt:   Optional[pd.DataFrame],
    df_cl_ttp:   Optional[pd.DataFrame],
    df_mits:     Optional[pd.DataFrame],
    df_strat:    Optional[pd.DataFrame],
    df_selected: Optional[pd.DataFrame],
) -> None:
    cluster_col, cluster_reason = choose_cluster_column(df_labels, df_selected)
    feat_cols = [c for c in df_feat.columns if c != "sample_id"]
    clean_id_columns(df_feat, df_labels, df_lbl_gt)

    summary_df = build_cluster_summary(
        df_feat, df_labels, df_lbl_gt, df_cl_ttp, df_strat, cluster_col
    )

    # ── Write CSV summary ─────────────────────────────────────────────────
    summary_csv = OUT_INTEL / "final_cluster_intelligence_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # ── Build text report ─────────────────────────────────────────────────
    total_samples = len(df_feat)
    n_clusters    = summary_df["cluster"].nunique() if not summary_df.empty else 0
    overall_sil   = selected_overall_silhouette(df_internal, cluster_col)

    SEP  = "=" * 76
    SEP2 = "-" * 76

    lines = [
        SEP,
        "IoT BOTNET FINAL CLUSTER INTELLIGENCE REPORT",
        "Research: IoT Botnet Clustering using Similarity Measures and MITRE ATT&CK",
        SEP,
        f"Total samples analysed  : {total_samples}",
        f"Behavioural features    : {len(feat_cols)}",
        f"Clusters identified     : {n_clusters}",
        f"Selected cluster run    : {cluster_col} ({cluster_col.replace('_', ' ').title()})",
        f"Selection source        : {cluster_reason}",
        f"Validation summary      : {selected_validation_text(df_internal, cluster_col)}",
        f"Selected run silhouette : {overall_sil}",
        "",
        "Note: Step 6 generates reports only.  Visual evidence (heatmaps,",
        "dendrograms, ATT&CK kill-chain figures) is produced by Steps 1–5.",
        "",
    ]

    for _, r in summary_df.iterrows():
        indent = "  "
        lines.extend([
            "",
            SEP2,
            (
                f"CLUSTER {int(r['cluster'])}  |  "
                f"{int(r['cluster_size'])} samples  |  "
                f"{float(r['pct_of_dataset']):.1f}% of dataset"
            ),
            f"Risk level      : {r['risk_level']}",
            f"Archetype       : {r['archetype']}",
            f"Description     : {r['description']}",
            f"Dominant tactic : {r['dominant_tactic']}",
            f"Family labels   : {r['family_labels']}",
            "",
            "Top behavioural indicators:",
            indent + wrap_text(r["top_behavioral_indicators"], 90, 5).replace(
                "\n", "\n" + indent
            ),
            "",
            "Top ATT&CK techniques:",
            indent + wrap_text(r["top_attack_techniques"], 90, 5).replace(
                "\n", "\n" + indent
            ),
            "",
            "Recommended mitigations:",
            indent + wrap_text(r["recommended_mitigations"], 90, 5).replace(
                "\n", "\n" + indent
            ),
        ])

    if df_mits is not None and not df_mits.empty:
        lines.extend(["", SEP2, "TOP OVERALL MITIGATIONS:"])
        for _, r in df_mits.head(10).iterrows():
            mit    = r.get("mitigation",     "Unknown mitigation")
            freq   = r.get("frequency",      "N/A")
            samps  = r.get("unique_samples", "N/A")
            tactic = r.get("dominant_tactic","N/A")
            lines.append(
                f"  {mit} | "
                f"frequency={freq}, samples={samps}, dominant_tactic={tactic}"
            )

    lines.extend(["", SEP, "END OF REPORT", SEP, ""])

    # ── Write and echo text report ────────────────────────────────────────
    report_path = OUT_INTEL / "final_cluster_intelligence_report.txt"
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"[DONE] {report_path}")
    print(f"[DONE] {summary_csv}")
    print("[DONE] Step 6 complete — no plots generated (visual evidence from Steps 1–5).")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # cleanly if a file is absent, instead of raising a raw traceback.
    df_feat   = require_csv("output/features/behavioral_features.csv",
                            "behavioural feature table (run Step 1 first)")
    df_labels = require_csv("output/clustering/cluster_labels.csv",
                            "cluster label table (run Step 3 first)")

    # Optional upstream inputs — absence is tolerated gracefully.
    df_lbl_gt   = safe_csv("output/features/ground_truth_labels.csv")
    df_internal = safe_csv("output/validation/internal_validation_summary.csv")
    df_ttps     = safe_csv("output/ttp_mitre/sample_ttps.csv")
    df_cl_ttp   = safe_csv("output/ttp_mitre/cluster_ttp_profile.csv")
    df_mits     = safe_csv("output/ttp_mitre/mitigations.csv")
    df_strat    = safe_csv("output/ttp_mitre/strategic_cluster_profile.csv")
    df_selected = safe_csv("output/ttp_mitre/selected_clustering_run_summary.csv")

    clean_id_columns(df_feat, df_labels, df_lbl_gt, df_ttps, df_cl_ttp)

    # Column presence guards on required frames.
    if "sample_id" not in df_feat.columns:
        print("[ERROR] behavioral_features.csv must contain a 'sample_id' column.")
        sys.exit(1)
    if "sample_id" not in df_labels.columns:
        print("[ERROR] cluster_labels.csv must contain a 'sample_id' column.")
        sys.exit(1)

    print("[INFO] Step 6 running in report-only mode.")
    write_report(
        df_feat, df_labels, df_internal,
        df_lbl_gt, df_cl_ttp, df_mits, df_strat, df_selected,
    )


if __name__ == "__main__":
    main()
