"""05_ttp_mitre_updated.py — Step 5 of 6: TTP Extraction and MITRE ATT&CK Mapping
Maps IoT botnet behavioral features to MITRE ATT&CK techniques, aggregates by cluster and family.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    from log_utils import setup_script_logging
except Exception:  # pragma: no cover
    setup_script_logging = None

try:
    from viz_utils import style_table, table_set_widths
except Exception:  # pragma: no cover
    def style_table(table, fontsize: float = 8.0, header_fontsize: float = 8.0,
                    alt1: str = "#FFFFFF", alt2: str = "#F8F9FA") -> None:
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#FFFFFF")
            if row == 0:
                cell.set_facecolor("#231F20")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_fontsize(header_fontsize)
            else:
                cell.set_facecolor(alt1 if row % 2 else alt2)
                cell.get_text().set_wrap(True)

    def table_set_widths(table, widths: Sequence[float]) -> None:
        for (row, col), cell in table.get_celld().items():
            if col < len(widths):
                cell.set_width(widths[col])


@dataclass(frozen=True)
class Config:
    feat_file: Path = Path("output/features/behavioral_features.csv")
    cluster_file: Path = Path("output/clustering/cluster_labels.csv")
    validation_file: Path = Path("output/validation/internal_validation_summary.csv")
    label_file: Path = Path("output/features/ground_truth_labels.csv")
    attack_file: Path = Path("../dataset/mitre/enterprise-attack.json")
    out_dir: Path = Path("output/ttp_mitre")
    cluster_col: str = "auto"
    top_n: int = 15
    top_heatmap_n: int = 16
    min_feature_prevalence: float = 0.50
    figure_dpi: int = 300


MITRE_BRAND = {
    "attack_orange": "#C64227",
    "attack_black": "#231F20",
    "attack_gray": "#5F6369",
    "light_gray": "#F2F3F4",
    "paper": "#FFFFFF",
}

ATTACK_TACTIC_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

TACTIC_TO_PHASE = {tactic: idx + 1 for idx, tactic in enumerate(ATTACK_TACTIC_ORDER)}

TACTIC_COLORS = {
    "Reconnaissance": "#6BAED6",
    "Resource Development": "#9E9E9E",
    "Initial Access": "#2CA25F",
    "Execution": "#E34A33",
    "Persistence": "#756BB1",
    "Privilege Escalation": "#9E9AC8",
    "Defense Evasion": "#FDAE6B",
    "Credential Access": "#E6550D",
    "Discovery": "#3182BD",
    "Lateral Movement": "#31A354",
    "Collection": "#BDB76B",
    "Command and Control": "#08519C",
    "Exfiltration": "#74C476",
    "Impact": "#CB181D",
    "Unknown": "#7F8C8D",
}

RISK_COLORS = {
    "CRITICAL": "#CB181D",
    "HIGH": "#E6550D",
    "MEDIUM": "#FDD049",
    "LOW": "#31A354",
}

TACTIC_RISK = {
    "Impact": 5,
    "Credential Access": 4,
    "Command and Control": 4,
    "Lateral Movement": 4,
    "Initial Access": 3,
    "Execution": 3,
    "Persistence": 3,
    "Privilege Escalation": 3,
    "Defense Evasion": 2,
    "Discovery": 2,
    "Collection": 2,
    "Resource Development": 1,
    "Reconnaissance": 1,
}

KNOWN_FAMILIES = {
    "mirai", "gafgyt", "hajime", "tsunami", "bashlite", "dofloo", "lightaidra"
}

FEATURE_LABELS = {
    "net_cnc": "C2 communication",
    "net_cnc_tcp": "TCP C2 channel",
    "net_cnc_udp": "UDP C2 channel",
    "net_scanning": "Port or service scanning",
    "net_scan_telnet": "Telnet scan",
    "net_scan_tr069": "TR-069 router exploit scan",
    "net_telnet_brute": "Telnet brute force login",
    "net_syn_scan": "SYN scan or flood signal",
    "net_ddos": "DDoS traffic",
    "net_dns": "DNS activity",
    "net_p2p": "Peer to peer C2",
    "net_blacklisted_ip": "Blacklisted IP contact",
    "net_http_exploit": "HTTP exploit attempt",
    "beh_watchdog": "Watchdog persistence",
    "beh_proc_net": "Network connection enumeration",
    "beh_proc_fd_enum": "Process file descriptor enumeration",
    "beh_proc_masquerade": "Process masquerading",
    "beh_stage2_drop": "Second stage payload drop",
    "beh_recon": "System reconnaissance",
    "beh_antivm": "Anti VM behavior",
    "beh_antidbg": "Anti debug behavior",
    "beh_persistence": "Persistence behavior",
    "beh_file_removal": "File deletion or cleanup",
    "beh_proc_inject": "Process injection behavior",
    "beh_kernel_module": "Kernel module activity",
    "binary_packed": "Packed binary",
    "high_entropy": "High entropy binary",
    "binary_stripped": "Stripped binary",
    "suspicious_strings": "Suspicious embedded strings",
    "sys_exec": "Process execution syscall",
    "sys_daemonize": "Daemonization syscall",
    "sys_prctl": "Process control syscall",
    "sys_dir_enum": "Directory enumeration syscall",
    "sys_kill": "Process termination syscall",
    "sys_mmap": "Memory mapping syscall",
    "sys_mprotect": "Memory protection syscall",
    "sys_sleep": "Sleep or timing syscall",
    "prog_cnc_string": "C2 string in program log",
}

CLUSTER_COLUMN_CANDIDATES = [
    "hierarchical",
    "spectral",
    "hierarchical_cosine",
    "hierarchical_jaccard",
    "spectral_cosine",
    "spectral_jaccard",
]

SUPPORTED_CLUSTERING_RUNS = {
    "hierarchical_cosine",
    "hierarchical_jaccard",
    "spectral_cosine",
    "spectral_jaccard",
    "hierarchical",
    "spectral",
}

DISPLAY_TO_ALGO = {
    "Hierarchical": "hierarchical",
    "Spectral": "spectral",
}

DISPLAY_TO_MATRIX = {
    "Cosine": "cosine",
    "Jaccard": "jaccard",
}

# Manual feature token to ATT&CK technique mapping.
# High confidence entries generate TTP records directly.
# Supporting entries are retained as evidence but do not create TTP records by themselves.
# This avoids overclaiming from generic syscall and file/network I/O tokens.
TOKEN_TTP: Dict[str, Tuple[str, str, str, str]] = {
    # Static or binary evidence.
    "binary_packed": ("T1027.002", "Defense Evasion", "Software Packing", "HIGH"),
    "binary_stripped": ("T1027.008", "Defense Evasion", "Stripped Payloads", "MEDIUM"),
    "high_entropy": ("T1027", "Defense Evasion", "Obfuscated Files or Information", "MEDIUM"),
    "suspicious_strings": ("T1027", "Defense Evasion", "Obfuscated Files or Information", "MEDIUM"),

    # Process execution. Use Native API for syscall level execution evidence instead of overclaiming Unix Shell.
    "sys_exec": ("T1106", "Execution", "Native API", "MEDIUM"),

    # Persistence.
    # IoT Linux malware daemonises via fork+setsid+double-fork, NOT via systemd.
    # T1543.002 specifically implies systemd interaction which is overclaiming.
    # T1543 (Create or Modify System Process) is accurate and defensible.
    "sys_daemonize": ("T1543", "Persistence", "Create or Modify System Process", "MEDIUM"),
    "beh_watchdog": ("T1053", "Persistence", "Scheduled Task/Job", "MEDIUM"),
    "beh_persistence": ("T1546", "Persistence", "Event Triggered Execution", "HIGH"),
    "beh_kernel_module": ("T1547.006", "Persistence", "Kernel Modules and Extensions", "HIGH"),

    # Defense evasion.
    "sys_prctl": ("T1036.005", "Defense Evasion", "Match Legitimate Resource Name or Location", "MEDIUM"),
    "beh_proc_masquerade": ("T1036.005", "Defense Evasion", "Match Legitimate Resource Name or Location", "HIGH"),
    "beh_antivm": ("T1497.001", "Defense Evasion", "System Checks", "HIGH"),
    "beh_antidbg": ("T1622", "Defense Evasion", "Debugger Evasion", "HIGH"),
    "beh_file_removal": ("T1070.004", "Defense Evasion", "File Deletion", "HIGH"),
    "beh_proc_inject": ("T1055", "Defense Evasion", "Process Injection", "HIGH"),

    # Discovery and reconnaissance inside the compromised host/network.
    "sys_dir_enum": ("T1083", "Discovery", "File and Directory Discovery", "HIGH"),
    "beh_recon": ("T1082", "Discovery", "System Information Discovery", "HIGH"),
    "beh_proc_net": ("T1049", "Discovery", "System Network Connections Discovery", "HIGH"),
    "beh_proc_fd_enum": ("T1057", "Discovery", "Process Discovery", "HIGH"),
    "net_scanning": ("T1046", "Discovery", "Network Service Discovery", "HIGH"),
    "net_scan_telnet": ("T1046", "Discovery", "Network Service Discovery", "MEDIUM"),

    # Initial access and credential access.
    "net_scan_tr069": ("T1190", "Initial Access", "Exploit Public-Facing Application", "HIGH"),
    "net_http_exploit": ("T1190", "Initial Access", "Exploit Public-Facing Application", "HIGH"),
    "net_telnet_brute": ("T1110.001", "Credential Access", "Password Guessing", "HIGH"),

    # Command and control.
    "net_cnc": ("T1071", "Command and Control", "Application Layer Protocol", "HIGH"),
    "net_cnc_tcp": ("T1095", "Command and Control", "Non-Application Layer Protocol", "MEDIUM"),
    "net_cnc_udp": ("T1095", "Command and Control", "Non-Application Layer Protocol", "MEDIUM"),
    "net_dns": ("T1071.004", "Command and Control", "DNS", "HIGH"),
    "net_p2p": ("T1090.003", "Command and Control", "Multi-hop Proxy", "HIGH"),
    "net_blacklisted_ip": ("T1071", "Command and Control", "Application Layer Protocol", "HIGH"),
    "prog_cnc_string": ("T1071", "Command and Control", "Application Layer Protocol", "HIGH"),
    "beh_stage2_drop": ("T1105", "Command and Control", "Ingress Tool Transfer", "HIGH"),

    # Impact. SYN scanning is not treated as a flood unless DDoS evidence is also present.
    "net_ddos": ("T1498", "Impact", "Network Denial of Service", "HIGH"),
    "sys_kill": ("T1489", "Impact", "Service Stop", "MEDIUM"),
}

SUPPORTING_FEATURE_MAP: Dict[str, str] = {
    "arch_arm": "Artifact metadata: ARM build target. Useful for IoT malware characterization, not a standalone ATT&CK technique.",
    "arch_mips": "Artifact metadata: MIPS build target. Useful for IoT malware characterization, not a standalone ATT&CK technique.",
    "arch_x86": "Artifact metadata: x86 build target. Useful for IoT malware characterization, not a standalone ATT&CK technique.",
    "sys_fork": "Supporting execution evidence. Fork alone is generic process activity.",
    "sys_clone": "Supporting execution evidence. Clone alone is generic process activity.",
    "sys_socket": "Supporting network evidence. Socket creation alone does not prove C2.",
    "sys_connect": "Supporting network evidence. Connection alone does not prove C2.",
    "sys_bind": "Supporting network evidence. Bind alone does not prove C2.",
    "sys_listen": "Supporting network evidence. Listen alone does not prove C2.",
    "sys_send": "Supporting network evidence. Send alone does not prove C2.",
    "sys_sendto": "Supporting network evidence. Sendto alone does not prove C2.",
    "sys_recv": "Supporting network evidence. Receive alone does not prove C2.",
    "sys_recvfrom": "Supporting network evidence. Receive alone does not prove C2.",
    "sys_recvmsg": "Supporting network evidence. Receive message alone does not prove C2.",
    "sys_select": "Supporting network multiplexing evidence. Select alone does not prove C2.",
    "sys_open": "Supporting filesystem evidence. Open alone does not prove discovery.",
    "sys_read": "Supporting filesystem evidence. Read alone does not prove discovery.",
    "sys_write": "Supporting filesystem evidence. Write alone does not prove ingress transfer.",
    "sys_mmap": "Supporting memory manipulation evidence. Combined with mprotect it may indicate code injection or unpacking.",
    "sys_mprotect": "Supporting memory protection change evidence. Combined with mmap it may indicate code injection or unpacking.",
    "sys_sleep": "Supporting timing evidence. Sleep alone is not time based evasion.",
    "net_syn_scan": "Supporting scan/flood evidence. Treated as Discovery unless DDoS evidence is also present.",
    "env_dynamic_ran": "Analysis metadata: sample executed dynamically. Not an adversary ATT&CK technique.",
}

CONDITIONAL_TTP_RULES: List[dict] = [
    {
        "id": "syn_scan_as_discovery",
        "requires_any": {"net_syn_scan"},
        "requires_all": set(),
        "excludes": {"net_ddos"},
        "ttp": ("T1046", "Discovery", "Network Service Discovery", "MEDIUM"),
        "reason": "SYN scanning without DDoS evidence is treated as service discovery, not direct flooding.",
    },
    {
        "id": "syn_flood_ddos",
        "requires_any": {"net_syn_scan"},
        "requires_all": {"net_ddos"},
        "excludes": set(),
        "ttp": ("T1498.001", "Impact", "Direct Network Flood", "HIGH"),
        "reason": "SYN scan plus DDoS behavior supports direct network flood.",
    },
    {
        "id": "memory_execution_pattern",
        "requires_any": {"sys_mmap", "sys_mprotect"},
        "requires_all": {"sys_mmap", "sys_mprotect"},
        "excludes": {"beh_proc_inject"},
        "ttp": ("T1055", "Defense Evasion", "Process Injection", "LOW"),
        "reason": "mmap plus mprotect is a weak indicator of code injection or unpacking. Treat as low confidence.",
    },
    {
        "id": "sleep_with_evasion",
        "requires_any": {"sys_sleep"},
        "requires_all": set(),
        "excludes": set(),
        "requires_one_of_sets": [{"beh_antivm"}, {"beh_antidbg"}],
        "ttp": ("T1497.003", "Defense Evasion", "Time Based Checks", "MEDIUM"),
        "reason": "Sleep becomes time based evasion only when paired with anti analysis behavior.",
    },
]

ARCHETYPE_RULES = [
    ({"net_scan_telnet", "net_telnet_brute", "net_scanning"}, "Credential Brute Force Scanner", "Credential guessing and scanning are dominant. Treat as propagation capable.", "HIGH"),
    ({"net_scan_telnet", "net_syn_scan", "net_telnet_brute"}, "Mirai Lineage Scanner", "Mass scanning, Telnet guessing, and flood capability suggest Mirai like behavior.", "HIGH"),
    ({"net_syn_scan", "net_ddos"}, "DDoS Agent", "Traffic generation and flooding behavior create direct availability risk.", "CRITICAL"),
    ({"net_p2p", "net_cnc"}, "P2P C2 Botnet", "Decentralized command and control raises takedown difficulty.", "HIGH"),
    ({"beh_stage2_drop", "net_cnc"}, "Dropper or Downloader", "The sample appears to retrieve or stage additional payloads through C2.", "HIGH"),
    ({"net_scan_tr069", "net_http_exploit"}, "Router Exploit Agent", "Public facing service exploitation suggests CPE or router compromise behavior.", "HIGH"),
    ({"binary_packed", "high_entropy", "beh_antivm", "beh_antidbg"}, "Evasive Packed Malware", "Static and runtime evasion dominate the profile.", "MEDIUM"),
]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Map IoT botnet behavioral features to MITRE ATT&CK TTPs and mitigations.")
    parser.add_argument("--feat-file", default=str(Config.feat_file))
    parser.add_argument("--cluster-file", default=str(Config.cluster_file))
    parser.add_argument("--validation-file", default=str(Config.validation_file), help="Optional Step 4 internal validation summary used to auto select the best clustering run.")
    parser.add_argument("--label-file", default=str(Config.label_file))
    parser.add_argument("--attack-file", default=str(Config.attack_file))
    parser.add_argument("--out-dir", default=str(Config.out_dir))
    parser.add_argument("--cluster-col", default="auto", help="Cluster column to use, or 'auto'.")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--top-heatmap-n", type=int, default=16)
    parser.add_argument("--min-feature-prevalence", type=float, default=0.50)
    parser.add_argument("--figure-dpi", type=int, default=300)
    args = parser.parse_args()
    return Config(
        feat_file=Path(args.feat_file),
        cluster_file=Path(args.cluster_file),
        validation_file=Path(args.validation_file),
        label_file=Path(args.label_file),
        attack_file=Path(args.attack_file),
        out_dir=Path(args.out_dir),
        cluster_col=args.cluster_col,
        top_n=args.top_n,
        top_heatmap_n=args.top_heatmap_n,
        min_feature_prevalence=args.min_feature_prevalence,
        figure_dpi=args.figure_dpi,
    )


def setup_logging(cfg: Config) -> logging.Logger:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (Path("output") / "logs").mkdir(parents=True, exist_ok=True)
    if setup_script_logging:
        setup_script_logging("05_ttp_mitre.py")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.out_dir / "05_ttp_mitre.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger("ttp_mitre")


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "axes.edgecolor": MITRE_BRAND["attack_gray"],
        "grid.color": "#D0D3D4",
        "grid.alpha": 0.35,
    })


def require_inputs(cfg: Config) -> None:
    required = [cfg.feat_file, cfg.cluster_file, cfg.attack_file]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def normalize_tactic_name(name: str) -> str:
    if not name:
        return "Unknown"
    return str(name).replace("-", " ").title().replace("And", "and")


def load_attack_index(attack_file: Path) -> Tuple[dict, dict, dict]:
    with open(attack_file, encoding="utf-8") as f:
        data = json.load(f)

    techniques: Dict[str, dict] = {}
    mitigations: Dict[str, dict] = {}
    uuid_to_tid: Dict[str, str] = {}
    attack_version = data.get("x_mitre_attack_spec_version", data.get("spec_version", "unknown"))

    for obj in data.get("objects", []):
        obj_type = obj.get("type")
        obj_id = obj.get("id", "")
        ext_id = next(
            (ref.get("external_id") for ref in obj.get("external_references", [])
             if ref.get("source_name") == "mitre-attack"),
            None,
        )
        if obj_type == "attack-pattern" and ext_id:
            tactics = [
                normalize_tactic_name(kc.get("phase_name", ""))
                for kc in obj.get("kill_chain_phases", [])
                if kc.get("kill_chain_name") == "mitre-attack"
            ]
            techniques[ext_id] = {
                "name": obj.get("name", "Unknown"),
                "tactics": "; ".join(tactics),
                "deprecated": bool(obj.get("x_mitre_deprecated", False) or obj.get("revoked", False)),
                "description": obj.get("description", ""),
            }
            uuid_to_tid[obj_id] = ext_id
        elif obj_type == "course-of-action" and ext_id:
            mitigations[obj_id] = {"mid": ext_id, "name": obj.get("name", "Unknown")}

    ttp_to_mitigations: Dict[str, List[str]] = {}
    for obj in data.get("objects", []):
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "mitigates":
            source_ref = obj.get("source_ref", "")
            target_ref = obj.get("target_ref", "")
            if source_ref in mitigations and target_ref in uuid_to_tid:
                tid = uuid_to_tid[target_ref]
                ttp_to_mitigations.setdefault(tid, []).append(mitigations[source_ref]["name"])

    for tid in list(ttp_to_mitigations):
        ttp_to_mitigations[tid] = sorted(set(ttp_to_mitigations[tid]))

    return techniques, ttp_to_mitigations, {"attack_version": attack_version}


def read_inputs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    df_feat = pd.read_csv(cfg.feat_file)
    df_clusters = pd.read_csv(cfg.cluster_file)
    df_labels = pd.read_csv(cfg.label_file) if cfg.label_file.exists() else None

    for name, df in [("behavioral_features.csv", df_feat), ("cluster_labels.csv", df_clusters)]:
        if "sample_id" not in df.columns:
            raise ValueError(f"{name} must contain a sample_id column.")
        if df["sample_id"].duplicated().any():
            dupes = df.loc[df["sample_id"].duplicated(), "sample_id"].head(5).tolist()
            raise ValueError(f"{name} contains duplicate sample_id values. Examples: {dupes}")
        df["sample_id"] = df["sample_id"].astype(str)

    if df_labels is not None and "sample_id" in df_labels.columns:
        df_labels["sample_id"] = df_labels["sample_id"].astype(str)

    missing_clusters = sorted(set(df_feat["sample_id"]) - set(df_clusters["sample_id"]))
    if missing_clusters:
        raise ValueError(
            "cluster_labels.csv is missing cluster labels for behavioral samples. "
            f"Missing count={len(missing_clusters)}. First examples={missing_clusters[:5]}"
        )

    return df_feat, df_clusters, df_labels


def read_validation_summary(validation_file: Path, logger: logging.Logger) -> pd.DataFrame:
    """Read Step 4 validation and keep only the current supported algorithms."""
    if not validation_file.exists():
        logger.warning(
            "Validation summary not found at %s. Auto selection will fall back to clustering aliases.",
            validation_file,
        )
        return pd.DataFrame()

    df = pd.read_csv(validation_file)
    required = {"algorithm", "matrix", "label_column", "silhouette"}
    if not required.issubset(df.columns):
        logger.warning(
            "Validation summary exists but lacks required columns: %s. Auto selection will fall back.",
            sorted(required - set(df.columns)),
        )
        return pd.DataFrame()

    df = df.copy()
    df["algorithm"] = df["algorithm"].astype(str)
    df["matrix"] = df["matrix"].astype(str)
    df["label_column"] = df["label_column"].astype(str)
    df = df[df["algorithm"].isin(DISPLAY_TO_ALGO.keys())]
    df = df[df["matrix"].isin(DISPLAY_TO_MATRIX.keys())]
    df = df[df["label_column"].isin(SUPPORTED_CLUSTERING_RUNS)]
    df["silhouette"] = pd.to_numeric(df["silhouette"], errors="coerce")
    return df


def choose_cluster_column(df_clusters: pd.DataFrame, requested: str, validation_df: pd.DataFrame) -> Tuple[str, str, pd.DataFrame]:
    """Choose one clustering output for TTP aggregation.

    Auto mode first uses Step 4 validation and selects the best available
    Hierarchical or Spectral run by silhouette score. If validation is absent,
    it falls back to compatibility aliases from Step 3.
    """
    available = [c for c in CLUSTER_COLUMN_CANDIDATES if c in df_clusters.columns]

    if requested != "auto":
        if requested not in df_clusters.columns:
            raise ValueError(f"Requested cluster column not found: {requested}")
        if requested == "sample_id":
            raise ValueError("sample_id is not a valid cluster label column.")
        selected = pd.DataFrame([{
            "selected_cluster_column": requested,
            "selection_basis": "user supplied",
            "algorithm": requested.split("_")[0].title(),
            "matrix": requested.split("_")[-1].title() if "_" in requested else "Alias",
            "silhouette": np.nan,
        }])
        return requested, "user supplied", selected

    if not available:
        raise ValueError(
            "No supported cluster label column found in cluster_labels.csv. "
            "Expected one of: " + ", ".join(CLUSTER_COLUMN_CANDIDATES)
        )

    if not validation_df.empty:
        valid = validation_df[validation_df["label_column"].isin(available)].copy()
        valid = valid.dropna(subset=["silhouette"])
        if not valid.empty:
            best = valid.sort_values(["silhouette", "algorithm", "matrix"], ascending=[False, True, True]).iloc[0]
            col = str(best["label_column"])
            selected = pd.DataFrame([{
                "selected_cluster_column": col,
                "selection_basis": "best Step 4 internal silhouette among Hierarchical and Spectral runs",
                "algorithm": best["algorithm"],
                "matrix": best["matrix"],
                "silhouette": round(float(best["silhouette"]), 4),
                "n_clusters": int(best["n_clusters"]) if "n_clusters" in best and pd.notna(best["n_clusters"]) else np.nan,
                "davies_bouldin": round(float(best["davies_bouldin"]), 4) if "davies_bouldin" in best and pd.notna(best["davies_bouldin"]) else np.nan,
                "calinski_harabasz": round(float(best["calinski_harabasz"]), 2) if "calinski_harabasz" in best and pd.notna(best["calinski_harabasz"]) else np.nan,
            }])
            return col, "auto selected from Step 4 validation summary", selected

    for col in ["hierarchical", "spectral", "hierarchical_cosine", "hierarchical_jaccard", "spectral_cosine", "spectral_jaccard"]:
        if col in available:
            selected = pd.DataFrame([{
                "selected_cluster_column": col,
                "selection_basis": "fallback compatibility alias because Step 4 validation was unavailable or incomplete",
                "algorithm": col.split("_")[0].title(),
                "matrix": col.split("_")[-1].title() if "_" in col else "Alias",
                "silhouette": np.nan,
            }])
            return col, "fallback auto detected from supported clustering outputs", selected

    col = available[0]
    selected = pd.DataFrame([{
        "selected_cluster_column": col,
        "selection_basis": "fallback first supported column",
        "algorithm": col.split("_")[0].title(),
        "matrix": col.split("_")[-1].title() if "_" in col else "Alias",
        "silhouette": np.nan,
    }])
    return col, "fallback auto detected", selected

def validate_mapping_coverage(df_feat: pd.DataFrame, logger: logging.Logger) -> List[str]:
    feat_cols = [c for c in df_feat.columns if c != "sample_id"]
    known = set(TOKEN_TTP) | set(SUPPORTING_FEATURE_MAP)
    conditional_features = set()
    for rule in CONDITIONAL_TTP_RULES:
        conditional_features |= set(rule.get("requires_any", set()))
        conditional_features |= set(rule.get("requires_all", set()))
        conditional_features |= set(rule.get("excludes", set()))
        for feature_set in rule.get("requires_one_of_sets", []):
            conditional_features |= set(feature_set)
    known |= conditional_features
    unmapped = [c for c in feat_cols if c not in known]
    if unmapped:
        logger.warning("Feature tokens with no direct, conditional, or supporting ATT&CK interpretation: %s", ", ".join(unmapped))
    else:
        logger.info("All %d feature tokens have direct, conditional, or supporting ATT&CK interpretation.", len(feat_cols))
    direct = [c for c in feat_cols if c in TOKEN_TTP]
    supporting = [c for c in feat_cols if c in SUPPORTING_FEATURE_MAP and c not in TOKEN_TTP]
    logger.info("%d features generate direct TTP claims. %d features are supporting evidence only.", len(direct), len(supporting))
    return feat_cols


def parse_active_features(row: pd.Series, feat_cols: Sequence[str]) -> set:
    active = set()
    for feat in feat_cols:
        value = row.get(feat, 0)
        try:
            is_active = float(value) > 0
        except Exception:
            is_active = str(value).strip().lower() in {"1", "true", "yes"}
        if is_active:
            active.add(feat)
    return active


def conditional_rule_applies(rule: dict, active: set) -> bool:
    requires_any = set(rule.get("requires_any", set()))
    requires_all = set(rule.get("requires_all", set()))
    excludes = set(rule.get("excludes", set()))
    if requires_any and not (requires_any & active):
        return False
    if requires_all and not requires_all.issubset(active):
        return False
    if excludes and (excludes & active):
        return False
    for feature_set in rule.get("requires_one_of_sets", []):
        if not (set(feature_set) & active):
            return False
    return True


def add_ttp_record(seen: Dict[str, dict], tid: str, tactic: str, local_name: str,
                   evidence: Iterable[str], confidence: str, mapping_rule: str) -> None:
    key = tid
    seen.setdefault(key, {
        "tactic": tactic,
        "local_name": local_name,
        "evidence": set(),
        "confidence": [],
        "rules": [],
    })
    seen[key]["evidence"].update(evidence)
    seen[key]["confidence"].append(confidence)
    seen[key]["rules"].append(mapping_rule)


def confidence_rank(values: Sequence[str]) -> str:
    order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    if not values:
        return "LOW"
    return sorted(values, key=lambda x: order.get(str(x).upper(), 0), reverse=True)[0]


def extract_ttps(df_feat: pd.DataFrame, feat_cols: Sequence[str], techniques: dict,
                 ttp_to_mitigations: dict) -> pd.DataFrame:
    rows = []
    for _, row in df_feat.iterrows():
        sample_id = row["sample_id"]
        active = parse_active_features(row, feat_cols)
        seen: Dict[str, dict] = {}

        for feat in sorted(active):
            if feat not in TOKEN_TTP:
                continue
            tid, tactic, local_name, confidence = TOKEN_TTP[feat]
            add_ttp_record(seen, tid, tactic, local_name, [feat], confidence, "direct")

        for rule in CONDITIONAL_TTP_RULES:
            if not conditional_rule_applies(rule, active):
                continue
            tid, tactic, local_name, confidence = rule["ttp"]
            evidence = sorted(
                (set(rule.get("requires_any", set())) | set(rule.get("requires_all", set()))) & active
            )
            for feature_set in rule.get("requires_one_of_sets", []):
                evidence.extend(sorted(set(feature_set) & active))
            evidence = sorted(set(evidence))
            add_ttp_record(seen, tid, tactic, local_name, evidence, confidence, rule["id"])

        supporting_features = sorted([f for f in active if f in SUPPORTING_FEATURE_MAP and f not in TOKEN_TTP])
        supporting_notes = [f"{f}: {SUPPORTING_FEATURE_MAP[f]}" for f in supporting_features]

        for tid, info in seen.items():
            official = techniques.get(tid, {})
            rows.append({
                "sample_id": sample_id,
                "ttp_id": tid,
                "tactic": info["tactic"],
                "tactic_phase": TACTIC_TO_PHASE.get(info["tactic"], 99),
                "technique_name": info["local_name"],
                "official_name": official.get("name", info["local_name"]),
                "official_tactics": official.get("tactics", ""),
                "is_deprecated": official.get("deprecated", False),
                "mapping_confidence": confidence_rank(info["confidence"]),
                "mapping_rules": "; ".join(sorted(set(info["rules"]))),
                "evidence_count": len(info["evidence"]),
                "evidence": "; ".join(sorted(info["evidence"])),
                "supporting_features": "; ".join(supporting_features),
                "supporting_feature_notes": " | ".join(supporting_notes),
                "mitigations": "; ".join(ttp_to_mitigations.get(tid, [])),
                "n_mitigations": len(ttp_to_mitigations.get(tid, [])),
            })
    return pd.DataFrame(rows)


def build_supporting_feature_table(df_feat: pd.DataFrame, feat_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feat in feat_cols:
        if feat not in SUPPORTING_FEATURE_MAP:
            continue
        active_count = int(pd.to_numeric(df_feat[feat], errors="coerce").fillna(0).gt(0).sum())
        rows.append({
            "feature": feat,
            "role": "supporting evidence only",
            "active_samples": active_count,
            "coverage_pct": round(active_count / max(len(df_feat), 1) * 100, 2),
            "interpretation": SUPPORTING_FEATURE_MAP[feat],
        })
    return pd.DataFrame(rows).sort_values(["active_samples", "feature"], ascending=[False, True])

def build_feature_attack_mapping_table(feat_cols: Sequence[str], techniques: dict,
                                       ttp_to_mitigations: dict) -> pd.DataFrame:
    """Build a feature-level traceability table for thesis reporting.

    Each row explains how an observed feature token is interpreted, which ATT&CK
    tactic and technique it maps to, and which MITRE mitigations are attached to
    that technique. Direct mappings are separated from conditional mappings and
    supporting-only features to avoid overclaiming generic evidence.
    """
    feat_set = set(feat_cols)
    rows: List[dict] = []

    for feat in sorted(feat_set):
        if feat in TOKEN_TTP:
            tid, tactic, local_name, confidence = TOKEN_TTP[feat]
            official = techniques.get(tid, {})
            rows.append({
                "feature_token": feat,
                "feature": FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "mapping_type": "Direct TTP claim",
                "attack_tactic": tactic,
                "tactic_phase": TACTIC_TO_PHASE.get(tactic, 99),
                "technique_id": tid,
                "technique_name": official.get("name", local_name),
                "local_technique_name": local_name,
                "mapping_confidence": confidence,
                "mitigations": "; ".join(ttp_to_mitigations.get(tid, [])) or "No MITRE mitigation relationship found",
                "mapping_condition": "Feature active in sample",
            })

        if feat in SUPPORTING_FEATURE_MAP and feat not in TOKEN_TTP:
            rows.append({
                "feature_token": feat,
                "feature": FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "mapping_type": "Supporting evidence only",
                "attack_tactic": "Not mapped as standalone tactic",
                "tactic_phase": 99,
                "technique_id": "N/A",
                "technique_name": "N/A",
                "local_technique_name": "N/A",
                "mapping_confidence": "N/A",
                "mitigations": "N/A",
                "mapping_condition": SUPPORTING_FEATURE_MAP[feat],
            })

    for rule in CONDITIONAL_TTP_RULES:
        evidence_features = sorted(
            (set(rule.get("requires_any", set())) | set(rule.get("requires_all", set()))) & feat_set
        )
        for feature_set in rule.get("requires_one_of_sets", []):
            evidence_features.extend(sorted(set(feature_set) & feat_set))
        evidence_features = sorted(set(evidence_features))
        if not evidence_features:
            continue
        tid, tactic, local_name, confidence = rule["ttp"]
        official = techniques.get(tid, {})
        excluded = sorted(set(rule.get("excludes", set())) & feat_set)
        condition_parts = []
        if rule.get("requires_any"):
            condition_parts.append("any of: " + ", ".join(sorted(rule.get("requires_any", set()))))
        if rule.get("requires_all"):
            condition_parts.append("all of: " + ", ".join(sorted(rule.get("requires_all", set()))))
        if rule.get("requires_one_of_sets"):
            condition_parts.append("paired with one of configured feature sets")
        if excluded:
            condition_parts.append("excluding: " + ", ".join(excluded))
        condition = "; ".join(condition_parts) if condition_parts else rule.get("reason", "Conditional rule")
        rows.append({
            "feature_token": " + ".join(evidence_features),
            "feature": " + ".join(FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f in evidence_features),
            "mapping_type": "Conditional TTP claim",
            "attack_tactic": tactic,
            "tactic_phase": TACTIC_TO_PHASE.get(tactic, 99),
            "technique_id": tid,
            "technique_name": official.get("name", local_name),
            "local_technique_name": local_name,
            "mapping_confidence": confidence,
            "mitigations": "; ".join(ttp_to_mitigations.get(tid, [])) or "No MITRE mitigation relationship found",
            "mapping_condition": f"{rule['id']}: {condition}. {rule.get('reason', '')}",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "feature_token", "feature", "mapping_type", "attack_tactic", "tactic_phase",
            "technique_id", "technique_name", "local_technique_name",
            "mapping_confidence", "mitigations", "mapping_condition",
        ])

    out = pd.DataFrame(rows)
    type_order = {"Direct TTP claim": 0, "Conditional TTP claim": 1, "Supporting evidence only": 2}
    out["mapping_type_rank"] = out["mapping_type"].map(type_order).fillna(9)
    out = out.sort_values(["mapping_type_rank", "tactic_phase", "attack_tactic", "feature_token"]).drop(columns=["mapping_type_rank"])
    return out.reset_index(drop=True)


def summarize_frequency(df_ttps: pd.DataFrame, n_total: int) -> pd.DataFrame:
    if df_ttps.empty:
        return pd.DataFrame(columns=["ttp_id", "tactic", "tactic_phase", "official_name", "sample_count", "coverage_pct"])
    freq = (
        df_ttps.groupby(["ttp_id", "tactic", "tactic_phase", "official_name"], as_index=False)["sample_id"]
        .nunique()
        .rename(columns={"sample_id": "sample_count"})
    )
    freq["coverage_pct"] = (freq["sample_count"] / max(n_total, 1) * 100).round(2)
    return freq.sort_values(["sample_count", "tactic_phase"], ascending=[False, True])


def summarize_clusters(df_ttps: pd.DataFrame, df_clusters: pd.DataFrame, cluster_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cluster_sizes = df_clusters[cluster_col].value_counts().rename("cluster_size")
    merged = pd.merge(
        df_ttps,
        df_clusters[["sample_id", cluster_col]].rename(columns={cluster_col: "cluster"}),
        on="sample_id",
        how="inner",
    )
    if merged.empty:
        return merged, pd.DataFrame(), cluster_sizes
    cluster_ttp = (
        merged.groupby(["cluster", "ttp_id", "tactic", "tactic_phase", "official_name"], as_index=False)["sample_id"]
        .nunique()
        .rename(columns={"sample_id": "sample_count"})
    )
    cluster_ttp = cluster_ttp.merge(cluster_sizes, left_on="cluster", right_index=True, how="left")
    cluster_ttp["prevalence"] = (cluster_ttp["sample_count"] / cluster_ttp["cluster_size"]).round(4)
    return merged, cluster_ttp.sort_values(["cluster", "tactic_phase", "prevalence"], ascending=[True, True, False]), cluster_sizes


def summarize_family(df_ttps: pd.DataFrame, df_labels: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_labels is None or df_labels.empty or "family" not in df_labels.columns:
        return None
    labels = df_labels.copy()
    labels["family"] = labels["family"].astype(str).str.lower()
    labels = labels[labels["family"].isin(KNOWN_FAMILIES)]
    if labels.empty:
        return None
    merged = pd.merge(df_ttps, labels[["sample_id", "family"]], on="sample_id", how="inner")
    if merged.empty:
        return None
    fam_sizes = labels["family"].value_counts().rename("family_size")
    family_ttp = (
        merged.groupby(["family", "ttp_id", "tactic", "tactic_phase", "official_name"], as_index=False)["sample_id"]
        .nunique()
        .rename(columns={"sample_id": "sample_count"})
    )
    family_ttp = family_ttp.merge(fam_sizes, left_on="family", right_index=True, how="left")
    family_ttp["prevalence"] = (family_ttp["sample_count"] / family_ttp["family_size"]).round(4)
    return family_ttp.sort_values(["family", "tactic_phase", "prevalence"], ascending=[True, True, False])


def feature_prevalence_for_samples(df_feat: pd.DataFrame, feat_cols: Sequence[str], sample_ids: Sequence,
                                   min_prev: float) -> Tuple[set, pd.Series]:
    subset = df_feat[df_feat["sample_id"].isin(sample_ids)]
    if subset.empty:
        return set(), pd.Series(dtype=float)
    means = subset[list(feat_cols)].apply(pd.to_numeric, errors="coerce").fillna(0).mean().sort_values(ascending=False)
    return set(means[means >= min_prev].index), means


def infer_archetype(features: set) -> Tuple[str, str, str]:
    if not features:
        return "Dormant or Low Activity", "Few active behavioral signals were observed. Review sandbox execution quality and timeout settings.", "LOW"
    for required, name, description, risk in ARCHETYPE_RULES:
        if required & features:
            return name, description, risk
    return "Mixed Behavioral Botnet Profile", "The cluster combines multiple behavior categories without one dominant archetype.", "MEDIUM"


def build_strategic_profiles(df_feat: pd.DataFrame, feat_cols: Sequence[str], df_clusters: pd.DataFrame,
                             cluster_col: str, merged: pd.DataFrame, cluster_ttp: pd.DataFrame,
                             cluster_sizes: pd.Series, cfg: Config) -> pd.DataFrame:
    rows = []
    if cluster_ttp.empty:
        return pd.DataFrame()
    for cluster_id in sorted(cluster_ttp["cluster"].unique()):
        cdata = cluster_ttp[cluster_ttp["cluster"] == cluster_id].copy()
        csize = int(cluster_sizes.get(cluster_id, 0))
        top5 = cdata.sort_values(["prevalence", "tactic_phase"], ascending=[False, True]).head(5)
        sample_ids = df_clusters[df_clusters[cluster_col] == cluster_id]["sample_id"].tolist()
        top_features, feature_means = feature_prevalence_for_samples(df_feat, feat_cols, sample_ids, cfg.min_feature_prevalence)
        archetype, description, risk_level = infer_archetype(top_features)

        tactic_score = {}
        for _, r in cdata.iterrows():
            tactic = r["tactic"]
            tactic_score[tactic] = tactic_score.get(tactic, 0.0) + float(r["sample_count"]) * TACTIC_RISK.get(tactic, 1)
        dominant_tactic = max(tactic_score, key=tactic_score.get) if tactic_score else "Unknown"

        mitigation_counter: Dict[str, int] = {}
        c_mits = merged[merged["cluster"] == cluster_id][["tactic", "mitigations"]]
        for _, r in c_mits.iterrows():
            for mit in str(r["mitigations"]).split("; "):
                mit = mit.strip()
                if mit:
                    mitigation_counter[mit] = mitigation_counter.get(mit, 0) + 1
        top_mitigations = sorted(mitigation_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:7]

        risk_weight = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(risk_level, 1)
        tactic_diversity = cdata["tactic"].nunique()
        ttp_diversity = cdata["ttp_id"].nunique()
        priority_score = risk_weight * (ttp_diversity + tactic_diversity)

        observed_chain = [t for t in ATTACK_TACTIC_ORDER if t in set(cdata["tactic"])]
        rows.append({
            "cluster": cluster_id,
            "cluster_size": csize,
            "pct_of_dataset": round(csize / max(len(df_feat), 1) * 100, 2),
            "archetype": archetype,
            "risk_level": risk_level,
            "priority_score": int(priority_score),
            "dominant_tactic": dominant_tactic,
            "observed_attack_chain": " -> ".join(observed_chain),
            "top_ttp_ids": "; ".join(top5["ttp_id"].tolist()),
            "top_techniques": "; ".join(top5["official_name"].tolist()),
            "top_prevalence": "; ".join(f"{p:.0%}" for p in top5["prevalence"].tolist()),
            "n_unique_tactics": int(tactic_diversity),
            "n_unique_ttps": int(ttp_diversity),
            "top_feature_tokens": "; ".join(feature_means[feature_means > 0].head(10).index.tolist()),
            "recommended_mitigations": "; ".join(m for m, _ in top_mitigations),
            "n_mitigations": len(mitigation_counter),
            "description": description,
        })
    return pd.DataFrame(rows).sort_values(["priority_score", "cluster_size"], ascending=[False, False])


def build_mitigation_table(df_ttps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df_ttps.empty:
        return pd.DataFrame(columns=["mitigation", "dominant_tactic", "frequency", "supporting_ttps", "supporting_tactics"])
    for _, r in df_ttps.iterrows():
        for mit in str(r.get("mitigations", "")).split("; "):
            mit = mit.strip()
            if mit:
                rows.append({
                    "mitigation": mit,
                    "tactic": r["tactic"],
                    "ttp_id": r["ttp_id"],
                    "sample_id": r["sample_id"],
                })
    if not rows:
        return pd.DataFrame(columns=["mitigation", "dominant_tactic", "frequency", "supporting_ttps", "supporting_tactics"])
    raw = pd.DataFrame(rows)
    out_rows = []
    for mit, group in raw.groupby("mitigation"):
        tactic_counts = group["tactic"].value_counts()
        dominant_tactic = sorted(tactic_counts.index, key=lambda t: (-tactic_counts[t], TACTIC_TO_PHASE.get(t, 99)))[0]
        out_rows.append({
            "mitigation": mit,
            "dominant_tactic": dominant_tactic,
            "frequency": len(group),
            "unique_samples": group["sample_id"].nunique(),
            "supporting_ttps": "; ".join(sorted(group["ttp_id"].unique())),
            "supporting_tactics": "; ".join([t for t in ATTACK_TACTIC_ORDER if t in set(group["tactic"])]),
        })
    return pd.DataFrame(out_rows).sort_values(["frequency", "unique_samples"], ascending=[False, False])


def build_attack_flow_by_cluster(cluster_ttp: pd.DataFrame) -> pd.DataFrame:
    if cluster_ttp.empty:
        return pd.DataFrame()
    rows = []
    for (cluster_id, tactic), group in cluster_ttp.groupby(["cluster", "tactic"]):
        phase = TACTIC_TO_PHASE.get(tactic, 99)
        rows.append({
            "cluster": cluster_id,
            "tactic": tactic,
            "tactic_phase": phase,
            "unique_ttps": group["ttp_id"].nunique(),
            "mean_prevalence": round(float(group["prevalence"].mean()), 4),
            "max_prevalence": round(float(group["prevalence"].max()), 4),
            "sample_count_sum": int(group["sample_count"].sum()),
            "top_techniques": "; ".join(group.sort_values("prevalence", ascending=False).head(4)["official_name"].tolist()),
        })
    return pd.DataFrame(rows).sort_values(["cluster", "tactic_phase"])


def build_mapping_trace(df_feat: pd.DataFrame, feat_cols: Sequence[str], df_clusters: pd.DataFrame,
                        cluster_col: str, strategic: pd.DataFrame, family_ttp: Optional[pd.DataFrame],
                        df_labels: Optional[pd.DataFrame], ttp_to_mitigations: dict, cfg: Config) -> pd.DataFrame:
    rows = []
    for _, r in strategic.sort_values("cluster").iterrows():
        sample_ids = df_clusters[df_clusters[cluster_col] == r["cluster"]]["sample_id"].tolist()
        _, feature_means = feature_prevalence_for_samples(df_feat, feat_cols, sample_ids, cfg.min_feature_prevalence)
        features = feature_means[feature_means > 0].head(10).index.tolist()
        pairs = []
        support_notes = []
        for feat in features:
            if feat in TOKEN_TTP:
                tid, tactic, tname, confidence = TOKEN_TTP[feat]
                pairs.append((tactic, f"{tid} {tname} [{confidence}]"))
            elif feat in SUPPORTING_FEATURE_MAP:
                support_notes.append(f"{feat}: supporting")
        rows.append({
            "entity": f"Cluster {r['cluster']}",
            "profile": r["archetype"],
            "dominant_tactic": r["dominant_tactic"],
            "feature_tokens": "; ".join(features),
            "tactics": "; ".join(dict.fromkeys([p[0] for p in pairs])),
            "techniques": "; ".join(dict.fromkeys([p[1] for p in pairs])),
            "mitigations": r["recommended_mitigations"],
        })

    if family_ttp is not None and not family_ttp.empty and df_labels is not None and "family" in df_labels.columns:
        labels = df_labels.copy()
        labels["family"] = labels["family"].astype(str).str.lower()
        for family in sorted(family_ttp["family"].unique()):
            sample_ids = labels[labels["family"] == family]["sample_id"].tolist()
            _, feature_means = feature_prevalence_for_samples(df_feat, feat_cols, sample_ids, cfg.min_feature_prevalence)
            features = feature_means[feature_means > 0].head(10).index.tolist()
            fdata = family_ttp[family_ttp["family"] == family].sort_values("prevalence", ascending=False).head(6)
            tactic_counts = fdata["tactic"].value_counts()
            dominant = sorted(tactic_counts.index, key=lambda t: (-tactic_counts[t], TACTIC_TO_PHASE.get(t, 99)))[0] if not tactic_counts.empty else "Unknown"
            mitigations = []
            for tid in fdata["ttp_id"]:
                mitigations.extend(ttp_to_mitigations.get(tid, []))
            rows.append({
                "entity": f"Family {family}",
                "profile": "Known family label",
                "dominant_tactic": dominant,
                "feature_tokens": "; ".join(features),
                "tactics": "; ".join(dict.fromkeys(fdata["tactic"].tolist())),
                "techniques": "; ".join(f"{x.ttp_id} {x.official_name}" for _, x in fdata.iterrows()),
                "mitigations": "; ".join(dict.fromkeys(mitigations[:10])),
            })
    return pd.DataFrame(rows)


def save_tables(cfg: Config, tables: Dict[str, Optional[pd.DataFrame]]) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df is not None:
            df.to_csv(cfg.out_dir / f"{name}.csv", index=False)


def wrap_text(value: object, width: int = 28, max_lines: Optional[int] = None) -> str:
    text = "" if pd.isna(value) else str(value)
    if not text.strip():
        return "-"
    parts: List[str] = []
    for part in text.split("; "):
        parts.extend(textwrap.wrap(part, width=width) or [part])
    if max_lines is not None and len(parts) > max_lines:
        parts = parts[:max_lines]
        parts[-1] = parts[-1].rstrip(" .,;") + " ..."
    return "\n".join(parts)


def lighten(hex_color: str, amount: float = 0.75) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def tactic_legend(tactics: Iterable[str], max_cols: int = 5):
    unique = [t for t in ATTACK_TACTIC_ORDER if t in set(tactics)]
    if "Unknown" in set(tactics):
        unique.append("Unknown")
    handles = [mpatches.Patch(fc=TACTIC_COLORS.get(t, TACTIC_COLORS["Unknown"]), label=t, ec="white") for t in unique]
    ncol = min(max_cols, max(1, len(handles)))
    return handles, ncol


def savefig(fig, path: Path, cfg: Config) -> None:
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def _attack_cmap():
    """Return one MITRE styled sequential colormap for heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "mitre_attack_orange",
        ["#FFFFFF", "#F7D6CC", "#E8957F", MITRE_BRAND["attack_orange"], "#7F1D1D"],
    )


def _ordered_tactic_values(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    ordered = [t for t in ATTACK_TACTIC_ORDER if t in counts.index]
    if "Unknown" in counts.index:
        ordered.append("Unknown")
    return counts.loc[ordered]


def _style_axes(ax) -> None:
    ax.grid(alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")


def _label_barh(ax, bars, labels, pad: float = 0.5, fontsize: float = 8.0) -> None:
    xmax = 0.0
    for bar, label in zip(bars, labels):
        width = float(bar.get_width())
        xmax = max(xmax, width)
        ax.text(
            width + pad,
            bar.get_y() + bar.get_height() / 2,
            str(label),
            va="center",
            ha="left",
            fontsize=fontsize,
            fontweight="bold",
            color=MITRE_BRAND["attack_black"],
        )
    ax.set_xlim(0, max(xmax * 1.28, xmax + pad * 6, 1))


def _format_table_cells(table, body_fontsize: float = 7.2, header_fontsize: float = 8.2) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(body_fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9DEE3")
        cell.set_linewidth(0.55)
        cell.PAD = 0.055
        txt = cell.get_text()
        txt.set_wrap(True)
        txt.set_va("center")
        if row == 0:
            cell.set_facecolor(MITRE_BRAND["attack_black"])
            txt.set_color("white")
            txt.set_fontweight("bold")
            txt.set_fontsize(header_fontsize)
            txt.set_ha("center")
        else:
            txt.set_fontsize(body_fontsize)
            txt.set_ha("left")


def fig_top_ttps(cfg: Config, freq: pd.DataFrame, n_total: int) -> None:
    top = freq.head(cfg.top_n).copy()
    if top.empty:
        return
    top = top.sort_values(["tactic_phase", "sample_count"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(14.2, max(6.8, len(top) * 0.46 + 1.9)))
    labels = [f"{r.ttp_id}  {wrap_text(r.official_name, 48, 2)}" for _, r in top.iterrows()]
    colors = [TACTIC_COLORS.get(t, TACTIC_COLORS["Unknown"]) for t in top["tactic"]]
    bars = ax.barh(labels, top["coverage_pct"], color=colors, edgecolor="white", linewidth=0.9, height=0.62)
    value_labels = [f"{pct:.1f}%  n={int(count)}" for pct, count in zip(top["coverage_pct"], top["sample_count"])]
    _label_barh(ax, bars, value_labels, pad=0.45)
    ax.set_xlabel("Samples exhibiting technique (%)")
    ax.set_title(
        f"Top MITRE ATT&CK Techniques in IoT Botnet Dataset (N={n_total})\n"
        "Bars use the fixed tactic color policy shared by tactic, technique, and mitigation figures",
        fontweight="bold",
        pad=14,
    )
    _style_axes(ax)
    handles, ncol = tactic_legend(top["tactic"], 4)
    ax.legend(
        handles=handles,
        title="ATT&CK tactic",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.94,
        ncol=1,
    )
    fig.subplots_adjust(left=0.28, right=0.80, top=0.86, bottom=0.12)
    savefig(fig, cfg.out_dir / "fig_top_ttps.png", cfg)


def _heatmap_by_tactic(cfg: Config, pivot: pd.DataFrame, source: pd.DataFrame, index_label: str,
                       value_label: str, title: str, out_name: str) -> None:
    if pivot.empty or sns is None:
        return
    col_tactic = source.drop_duplicates("official_name").set_index("official_name")["tactic"].to_dict()
    top_cols = pivot.sum().sort_values(ascending=False).head(cfg.top_heatmap_n).index
    pivot = pivot[top_cols]
    order_cols = sorted(
        pivot.columns,
        key=lambda c: (TACTIC_TO_PHASE.get(col_tactic.get(c, "Unknown"), 99), -pivot[c].sum(), str(c)),
    )
    pivot = pivot[order_cols]

    fig_w = max(16.5, 0.82 * len(pivot.columns) + 6.2)
    fig_h = max(5.8, 0.72 * len(pivot.index) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap=_attack_cmap(),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7.2},
        linewidths=0.55,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": value_label, "shrink": 0.72, "pad": 0.018},
    )
    xlabels = []
    for col in pivot.columns:
        tactic = col_tactic.get(col, "Unknown")
        phase = TACTIC_TO_PHASE.get(tactic, 99)
        xlabels.append(f"P{phase}: {wrap_text(col, 24, 3)}")
    ax.set_xticklabels(xlabels, rotation=42, ha="right", fontsize=7.7)
    for tick, col in zip(ax.get_xticklabels(), pivot.columns):
        tactic = col_tactic.get(col, "Unknown")
        tick.set_color(TACTIC_COLORS.get(tactic, TACTIC_COLORS["Unknown"]))
        tick.set_fontweight("bold")
    ax.set_xlabel("ATT&CK technique, ordered by tactic phase")
    ax.set_ylabel(index_label)
    ax.set_title(title, fontweight="bold", pad=14)
    handles, _ = tactic_legend([col_tactic.get(c, "Unknown") for c in pivot.columns], 1)
    if handles:
        ax.legend(
            handles=handles,
            title="Technique tactic",
            loc="upper left",
            bbox_to_anchor=(1.16, 1.0),
            fontsize=7.5,
            title_fontsize=8,
            framealpha=0.94,
        )
    fig.subplots_adjust(top=0.86, bottom=0.31, left=0.12, right=0.78)
    savefig(fig, cfg.out_dir / out_name, cfg)


def fig_cluster_heatmap(cfg: Config, cluster_ttp: pd.DataFrame, strategic: pd.DataFrame) -> None:
    if cluster_ttp.empty or sns is None:
        return
    pivot = cluster_ttp.pivot_table(index="cluster", columns="official_name", values="prevalence", aggfunc="max").fillna(0)
    if pivot.empty:
        return
    ylabels = []
    for cid in pivot.index:
        row = strategic[strategic["cluster"] == cid]
        label = f"C{cid}"
        if not row.empty:
            label = f"C{cid}: {wrap_text(row.iloc[0]['archetype'], 24, 2)}"
        ylabels.append(label)
    pivot.index = ylabels
    _heatmap_by_tactic(
        cfg,
        pivot,
        cluster_ttp,
        "Behavioral cluster",
        "Prevalence within cluster",
        "Cluster by ATT&CK Technique Heatmap\nColumns follow ATT&CK tactic phase order and share the fixed tactic color policy",
        "fig_cluster_ttp_heatmap.png",
    )


def fig_family_heatmap(cfg: Config, family_ttp: Optional[pd.DataFrame]) -> None:
    if family_ttp is None or family_ttp.empty or sns is None:
        return
    pivot = family_ttp.pivot_table(index="family", columns="official_name", values="prevalence", aggfunc="max").fillna(0)
    if pivot.empty:
        return
    pivot.index = [str(x).title() for x in pivot.index]
    _heatmap_by_tactic(
        cfg,
        pivot,
        family_ttp,
        "Botnet family",
        "Prevalence within family",
        "Family by ATT&CK Technique Heatmap\nKnown family labels only, columns ordered by ATT&CK tactic phase",
        "fig_family_ttp_heatmap.png",
    )


def fig_top_mitigations(cfg: Config, mitigations: pd.DataFrame) -> None:
    top = mitigations.head(cfg.top_n).copy()
    if top.empty:
        return
    top = top.sort_values(["dominant_tactic", "frequency"], ascending=[True, True])
    top["phase"] = top["dominant_tactic"].map(TACTIC_TO_PHASE).fillna(99)
    top = top.sort_values(["phase", "frequency"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(14.2, max(6.8, len(top) * 0.46 + 1.9)))
    labels = [wrap_text(x, 48, 2) for x in top["mitigation"]]
    colors = [TACTIC_COLORS.get(t, TACTIC_COLORS["Unknown"]) for t in top["dominant_tactic"]]
    bars = ax.barh(labels, top["frequency"], color=colors, edgecolor="white", linewidth=0.9, height=0.62)
    value_labels = [f"{int(val)} hits, {int(samples)} samples" for val, samples in zip(top["frequency"], top["unique_samples"])]
    _label_barh(ax, bars, value_labels, pad=max(float(top["frequency"].max()) * 0.012, 0.8))
    ax.set_xlabel("Frequency across TTP sample instances")
    ax.set_title(
        "Top Strategic Mitigations for Observed IoT Botnet TTPs\n"
        "Mitigation bars inherit the dominant supporting ATT&CK tactic color",
        fontweight="bold",
        pad=14,
    )
    _style_axes(ax)
    handles, _ = tactic_legend(top["dominant_tactic"], 1)
    ax.legend(
        handles=handles,
        title="Dominant tactic",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.94,
    )
    fig.subplots_adjust(left=0.28, right=0.80, top=0.86, bottom=0.12)
    savefig(fig, cfg.out_dir / "fig_top_mitigations.png", cfg)


def fig_tactic_distribution(cfg: Config, df_ttps: pd.DataFrame) -> None:
    if df_ttps.empty:
        return
    counts = _ordered_tactic_values(df_ttps["tactic"])
    colors = [TACTIC_COLORS.get(t, TACTIC_COLORS["Unknown"]) for t in counts.index]
    phase_labels = [f"P{TACTIC_TO_PHASE.get(t, 99):02d}\n{wrap_text(t, 15, 2)}" for t in counts.index]

    fig, ax = plt.subplots(figsize=(16.8, 7.2))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts.values, color=colors, edgecolor="white", linewidth=1.0, width=0.68)
    for bar, tactic, value in zip(bars, counts.index, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.max() * 0.012, 0.6),
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, rotation=0, ha="center", fontsize=8.0)
    ax.set_ylabel("Number of TTP instances")
    ax.set_xlabel("MITRE ATT&CK tactic phase order")
    ax.set_title(
        "MITRE ATT&CK Tactic Distribution Across the IoT Botnet Dataset\n"
        "Frequency follows the ATT&CK kill chain order from Reconnaissance to Impact",
        fontweight="bold",
        pad=14,
    )
    _style_axes(ax)
    ax.set_ylim(0, max(counts.max() * 1.18, 1))
    handles, ncol = tactic_legend(counts.index, 7)
    ax.legend(
        handles=handles,
        title="ATT&CK tactic color policy",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        fontsize=7.8,
        title_fontsize=8,
        framealpha=0.94,
    )
    fig.subplots_adjust(top=0.84, bottom=0.28, left=0.07, right=0.98)
    savefig(fig, cfg.out_dir / "fig_tactic_distribution.png", cfg)


def fig_attack_kill_chain(cfg: Config, df_ttps: pd.DataFrame) -> None:
    sample_counts = df_ttps.groupby("tactic")["sample_id"].nunique().to_dict() if not df_ttps.empty else {}
    technique_counts = df_ttps.groupby("tactic")["ttp_id"].nunique().to_dict() if not df_ttps.empty else {}
    present = [t for t in ATTACK_TACTIC_ORDER if sample_counts.get(t, 0) > 0]
    fig, ax = plt.subplots(figsize=(24, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.linspace(0.055, 0.945, len(ATTACK_TACTIC_ORDER))
    box_w, box_h, y = 0.058, 0.205, 0.56
    for idx, tactic in enumerate(ATTACK_TACTIC_ORDER):
        x = xs[idx] - box_w / 2
        observed = sample_counts.get(tactic, 0) > 0
        color = TACTIC_COLORS.get(tactic, TACTIC_COLORS["Unknown"]) if observed else "#D5D8DC"
        alpha = 0.96 if observed else 0.34
        rect = mpatches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.010,rounding_size=0.012",
                                       fc=color, ec=MITRE_BRAND["attack_black"] if observed else "#AAB7B8", lw=1.2, alpha=alpha)
        ax.add_patch(rect)
        txt_color = "white" if observed and tactic in {"Command and Control", "Impact", "Execution", "Credential Access"} else "#17202A"
        ax.text(x + box_w / 2, y + box_h * 0.64, wrap_text(tactic, 13, 3), ha="center", va="center", fontsize=7.2, fontweight="bold", color=txt_color)
        ax.text(x + box_w / 2, y + box_h * 0.23, f"samples {sample_counts.get(tactic, 0)}\nTTPs {technique_counts.get(tactic, 0)}", ha="center", va="center", fontsize=6.2, color=txt_color)
        if idx < len(ATTACK_TACTIC_ORDER) - 1:
            ax.annotate("", xy=(xs[idx + 1] - box_w / 2 - 0.004, y + box_h / 2), xytext=(x + box_w + 0.004, y + box_h / 2), arrowprops=dict(arrowstyle="->", color="#566573", lw=1.1))

    if present:
        px = [xs[ATTACK_TACTIC_ORDER.index(t)] for t in present]
        sizes = [90 + sample_counts[t] * 1.25 for t in present]
        ax.plot(px, [0.30] * len(px), color=MITRE_BRAND["attack_black"], lw=2.0, alpha=0.72)
        ax.scatter(px, [0.30] * len(px), s=sizes, c=[TACTIC_COLORS[t] for t in present], edgecolors="white", linewidth=1.4, zorder=3)
        for t, px_i in zip(present, px):
            ax.text(px_i, 0.235, str(sample_counts[t]), ha="center", va="center", fontsize=7.4, fontweight="bold")
    chain_text = " -> ".join(present) if present else "No ATT&CK tactics detected"
    ax.text(0.5, 0.93, "MITRE ATT&CK Kill Chain Coverage for IoT Botnet Dataset", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.865, "Top row follows ATT&CK tactic order. Faded boxes were not observed in the dataset.", ha="center", va="center", fontsize=10, color=MITRE_BRAND["attack_gray"])
    ax.text(0.5, 0.12, f"Detected dataset chain: {chain_text}", ha="center", va="center", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.38", fc="#F8F9F9", ec="#D5D8DC"))
    ax.text(0.5, 0.055, "Color rule: a tactic, its techniques, and its associated mitigations use the same tactic color throughout this script.", ha="center", va="center", fontsize=8.8, color=MITRE_BRAND["attack_gray"])
    savefig(fig, cfg.out_dir / "fig_attack_kill_chain.png", cfg)


def fig_attack_flow_by_cluster(cfg: Config, flow: pd.DataFrame) -> None:
    if flow.empty:
        return
    clusters = sorted(flow["cluster"].unique())
    fig, ax = plt.subplots(figsize=(18, max(6, len(clusters) * 0.72 + 2.2)))
    ax.set_xlim(0.5, len(ATTACK_TACTIC_ORDER) + 0.5)
    ax.set_ylim(-0.8, len(clusters) - 0.2)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters])
    ax.set_xticks(range(1, len(ATTACK_TACTIC_ORDER) + 1))
    ax.set_xticklabels([wrap_text(t, 13, 3) for t in ATTACK_TACTIC_ORDER], rotation=35, ha="right")
    ax.grid(axis="x", alpha=0.18)
    for y, cluster_id in enumerate(clusters):
        cdata = flow[flow["cluster"] == cluster_id]
        phases = cdata["tactic_phase"].tolist()
        if phases:
            ax.plot(phases, [y] * len(phases), color="#6C757D", lw=1.2, alpha=0.60, zorder=1)
        for _, r in cdata.iterrows():
            size = 140 + 900 * float(r["mean_prevalence"])
            ax.scatter(r["tactic_phase"], y, s=size, color=TACTIC_COLORS.get(r["tactic"], TACTIC_COLORS["Unknown"]), edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(r["tactic_phase"], y + 0.20, str(int(r["unique_ttps"])), ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xlabel("ATT&CK tactic order")
    ax.set_title("Observed Attack Flow by Behavioral Cluster\nBubble size reflects mean prevalence in the cluster, number shows unique TTP count", fontweight="bold")
    custom = [Line2D([0], [0], marker="o", color="w", label="Higher prevalence", markerfacecolor="#6C757D", markersize=12),
              Line2D([0], [0], marker="o", color="w", label="Lower prevalence", markerfacecolor="#6C757D", markersize=7)]
    ax.legend(handles=custom, loc="upper right", framealpha=0.9)
    savefig(fig, cfg.out_dir / "fig_attack_flow_by_cluster.png", cfg)


def fig_strategic_clusters(cfg: Config, strategic: pd.DataFrame) -> None:
    if strategic.empty:
        return
    fig = plt.figure(figsize=(18.6, 10.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.78, wspace=0.50, height_ratios=[1.0, 1.34])
    ax0 = fig.add_subplot(gs[0, 0])
    risk_counts = strategic["risk_level"].value_counts()
    risk_order = [r for r in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if r in risk_counts.index]
    bars = ax0.bar(risk_order, [risk_counts[r] for r in risk_order], color=[RISK_COLORS[r] for r in risk_order], edgecolor="white", linewidth=1.0, width=0.58)
    for bar, r in zip(bars, risk_order):
        ax0.text(bar.get_x() + bar.get_width() / 2, risk_counts[r] + 0.05, str(int(risk_counts[r])), ha="center", fontweight="bold")
    ax0.set_title("Risk level distribution", fontweight="bold")
    ax0.set_ylabel("Cluster count")
    _style_axes(ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    ordered = strategic.sort_values("priority_score", ascending=True)
    labels = [wrap_text(f"C{r.cluster}: {r.archetype}", 33, 2) for _, r in ordered.iterrows()]
    bar_colors = [TACTIC_COLORS.get(r, TACTIC_COLORS["Unknown"]) for r in ordered["dominant_tactic"]]
    bars = ax1.barh(labels, ordered["priority_score"], color=bar_colors, edgecolor="white", linewidth=0.9, height=0.62)
    for bar, score in zip(bars, ordered["priority_score"]):
        ax1.text(score + 0.2, bar.get_y() + bar.get_height() / 2, str(int(score)), va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Priority score")
    ax1.set_title("Defensive priority ranking, colored by dominant tactic", fontweight="bold")
    _style_axes(ax1)
    handles, _ = tactic_legend(ordered["dominant_tactic"], 1)
    if handles:
        ax1.legend(handles=handles, title="Dominant tactic", loc="lower right", fontsize=7.5, title_fontsize=8, framealpha=0.94)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    table_rows = []
    for _, r in strategic.head(10).iterrows():
        table_rows.append([
            f"C{r['cluster']}",
            f"{int(r['cluster_size'])}\n({r['pct_of_dataset']:.1f}%)",
            wrap_text(r["archetype"], 28, 2),
            r["risk_level"],
            wrap_text(r["dominant_tactic"], 22, 2),
            wrap_text(r["observed_attack_chain"], 48, 3),
            wrap_text(r["recommended_mitigations"], 48, 3),
        ])
    table = ax2.table(cellText=table_rows, colLabels=["Cluster", "Size", "Archetype", "Risk", "Dominant tactic", "Observed attack chain", "Top mitigations"], cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    _format_table_cells(table, body_fontsize=7.3, header_fontsize=7.8)
    table_set_widths(table, [0.07, 0.08, 0.18, 0.08, 0.13, 0.24, 0.22])
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            continue
        risk = table_rows[row - 1][3]
        dominant = table_rows[row - 1][4].replace("\n", " ")
        base = TACTIC_COLORS.get(dominant, RISK_COLORS.get(risk, "#7F8C8D"))
        cell.set_facecolor(lighten(base, 0.86))
        if col == 3:
            cell.set_facecolor(RISK_COLORS.get(risk, "#FFFFFF"))
            cell.get_text().set_color("white" if risk != "MEDIUM" else "#17202A")
            cell.get_text().set_fontweight("bold")
    ax2.set_title("Strategic Cluster Decision Table", fontweight="bold", pad=18)
    fig.suptitle("Strategic Threat Intelligence Summary: IoT Botnet Behavioral Clusters and MITRE ATT&CK", fontsize=13, fontweight="bold")
    savefig(fig, cfg.out_dir / "fig_strategic_clusters.png", cfg)


def _mapping_table_axis(ax, plot_df: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    if plot_df.empty:
        ax.text(0.5, 0.5, f"{title}: no rows available", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=8)
        return
    rows = []
    for _, r in plot_df.iterrows():
        rows.append([
            wrap_text(str(r["entity"]).replace("Cluster ", "C").replace("Family ", ""), 18, 3),
            wrap_text(r["profile"], 24, 3),
            wrap_text(r["feature_tokens"], 28, 5),
            wrap_text(r["tactics"], 24, 4),
            wrap_text(r["techniques"], 38, 6),
            wrap_text(r["mitigations"], 38, 6),
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["Entity", "Profile", "Feature evidence", "Tactics", "Techniques", "Mitigations"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 0.92],
    )
    _format_table_cells(table, body_fontsize=6.8, header_fontsize=7.6)
    table_set_widths(table, [0.09, 0.15, 0.20, 0.15, 0.21, 0.20])
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            continue
        tactic = plot_df.iloc[row - 1]["dominant_tactic"]
        base = TACTIC_COLORS.get(tactic, TACTIC_COLORS["Unknown"])
        cell.set_facecolor(lighten(base, 0.88 if col in [0, 1, 2] else 0.80))
    ax.set_title(title, fontweight="bold", pad=8)


def fig_mapping_table(cfg: Config, mapping: pd.DataFrame) -> None:
    if mapping.empty:
        return
    clusters = mapping[mapping["entity"].astype(str).str.startswith("Cluster")].head(10).copy()
    families = mapping[mapping["entity"].astype(str).str.startswith("Family")].head(10).copy()
    n_rows = max(len(clusters), len(families), 4)
    fig_h = max(10.5, 1.05 * n_rows + 6.0)
    fig = plt.figure(figsize=(24, fig_h))
    gs = GridSpec(2, 1, figure=fig, hspace=0.34, height_ratios=[max(len(clusters), 1), max(len(families), 1)])
    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_family = fig.add_subplot(gs[1, 0])
    _mapping_table_axis(ax_cluster, clusters, "Cluster level MITRE ATT&CK mapping")
    _mapping_table_axis(ax_family, families, "Family level MITRE ATT&CK mapping")
    all_tactics = pd.concat([clusters.get("dominant_tactic", pd.Series(dtype=str)), families.get("dominant_tactic", pd.Series(dtype=str))], ignore_index=True)
    handles, ncol = tactic_legend(all_tactics.dropna().tolist(), 6)
    if handles:
        fig.legend(
            handles=handles,
            title="Row color by dominant tactic",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=ncol,
            fontsize=8,
            title_fontsize=8,
            framealpha=0.94,
        )
    fig.suptitle(
        "Traceable MITRE ATT&CK Mapping Table\n"
        "Cluster and family mappings are separated to avoid hidden or overlapping cell content",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.91, bottom=0.09, left=0.03, right=0.985)
    savefig(fig, cfg.out_dir / "fig_mitre_mapping_table.png", cfg)


def fig_feature_attack_mapping_table(cfg: Config, feature_mapping: pd.DataFrame) -> None:
    """Render a thesis ready feature to ATT&CK mapping table."""
    if feature_mapping.empty:
        return

    direct_first = feature_mapping.copy()
    direct_first["rank"] = direct_first["mapping_type"].map({
        "Direct TTP claim": 0,
        "Conditional TTP claim": 1,
        "Supporting evidence only": 2,
    }).fillna(9)
    plot_df = direct_first.sort_values(["rank", "tactic_phase", "feature_token"]).head(30).copy()

    fig_h = max(11.0, 0.64 * len(plot_df) + 3.8)
    fig, ax = plt.subplots(figsize=(24.0, fig_h))
    ax.axis("off")

    rows = []
    for _, r in plot_df.iterrows():
        technique = "N/A" if r["technique_id"] == "N/A" else f"{r['technique_id']}  {r['technique_name']}"
        feature_text = f"{r['feature']}\n({r['feature_token']})"
        if r["mapping_type"] == "Conditional TTP claim":
            feature_text = "Conditional rule\n" + feature_text
        elif r["mapping_type"] == "Supporting evidence only":
            feature_text = "Supporting evidence\n" + feature_text
        rows.append([
            wrap_text(feature_text, 34, 5),
            wrap_text(r["attack_tactic"], 24, 3),
            wrap_text(technique, 42, 4),
            wrap_text(r["mitigations"], 58, 5),
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Features", "ATT&CK Tactics", "Techniques", "Mitigations"],
        cellLoc="left",
        loc="center",
        bbox=[0.01, 0.05, 0.98, 0.86],
    )
    _format_table_cells(table, body_fontsize=6.9, header_fontsize=8.2)
    table_set_widths(table, [0.25, 0.16, 0.25, 0.34])

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            continue
        tactic = plot_df.iloc[row - 1]["attack_tactic"]
        mapping_type = plot_df.iloc[row - 1]["mapping_type"]
        if tactic in TACTIC_COLORS:
            base = TACTIC_COLORS[tactic]
            cell.set_facecolor(lighten(base, 0.88 if col in [0, 1] else 0.80))
        else:
            cell.set_facecolor("#F3F4F6")
        if mapping_type == "Supporting evidence only":
            cell.set_facecolor("#F1F3F5")
            cell.get_text().set_color("#374151")

    present_tactics = [t for t in plot_df["attack_tactic"].unique() if t in TACTIC_COLORS]
    handles, ncol = tactic_legend(present_tactics, 5)
    if handles:
        ax.legend(
            handles=handles,
            title="Row color by ATT&CK tactic",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=ncol,
            fontsize=8,
            title_fontsize=8,
            framealpha=0.94,
        )

    ax.set_title(
        "Feature to MITRE ATT&CK Mapping Table\n"
        "Observed behavioral features are aligned to tactics, techniques, and mitigations with readable cell wrapping",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.01,
        0.955,
        "Direct and conditional mappings create TTP claims. Supporting-only features are retained as evidence but are not standalone ATT&CK techniques.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.6,
        color=MITRE_BRAND["attack_gray"],
    )
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.025, right=0.985)
    savefig(fig, cfg.out_dir / "fig_feature_attack_mapping_table.png", cfg)


def write_run_metadata(cfg: Config, metadata: dict, cluster_col: str, cluster_note: str, df_feat: pd.DataFrame,
                       df_ttps: pd.DataFrame, strategic: pd.DataFrame) -> None:
    payload = {
        "attack_metadata": metadata,
        "input_files": {
            "features": str(cfg.feat_file),
            "clusters": str(cfg.cluster_file),
            "validation": str(cfg.validation_file) if cfg.validation_file.exists() else None,
            "labels": str(cfg.label_file) if cfg.label_file.exists() else None,
            "attack_file": str(cfg.attack_file),
        },
        "cluster_column": cluster_col,
        "cluster_column_note": cluster_note,
        "n_samples": int(len(df_feat)),
        "n_ttp_instances": int(len(df_ttps)),
        "n_clusters": int(strategic["cluster"].nunique()) if not strategic.empty else 0,
        "pipeline_alignment": "Aligned with Step 3 and Step 4 using only Hierarchical and Spectral clustering over Cosine and Jaccard matrices.",
        "color_policy": {
            "mitre_brand_colors": MITRE_BRAND,
            "tactic_colors": TACTIC_COLORS,
            "note": "MITRE publishes brand colors, not one official color per tactic. Tactic colors are fixed and used consistently for tactics, techniques, and mitigations.",
        },
    }
    with open(cfg.out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_figures(cfg: Config, df_ttps: pd.DataFrame, freq: pd.DataFrame, cluster_ttp: pd.DataFrame,
                   family_ttp: Optional[pd.DataFrame], strategic: pd.DataFrame,
                   mitigations: pd.DataFrame, flow: pd.DataFrame, mapping: pd.DataFrame,
                   feature_mapping: pd.DataFrame, n_total: int) -> None:
    fig_top_ttps(cfg, freq, n_total)
    fig_cluster_heatmap(cfg, cluster_ttp, strategic)
    fig_family_heatmap(cfg, family_ttp)
    fig_top_mitigations(cfg, mitigations)
    fig_tactic_distribution(cfg, df_ttps)
    fig_attack_kill_chain(cfg, df_ttps)
    fig_attack_flow_by_cluster(cfg, flow)
    fig_strategic_clusters(cfg, strategic)
    fig_mapping_table(cfg, mapping)
    fig_feature_attack_mapping_table(cfg, feature_mapping)


def main() -> None:
    cfg = parse_args()
    logger = setup_logging(cfg)
    configure_plotting()
    require_inputs(cfg)

    logger.info("Loading input data.")
    df_feat, df_clusters, df_labels = read_inputs(cfg)
    feat_cols = validate_mapping_coverage(df_feat, logger)
    validation_df = read_validation_summary(cfg.validation_file, logger)
    cluster_col, cluster_note, selected_run_summary = choose_cluster_column(df_clusters, cfg.cluster_col, validation_df)
    logger.info("Using cluster column: %s (%s).", cluster_col, cluster_note)

    logger.info("Loading MITRE ATT&CK STIX index.")
    techniques, ttp_to_mitigations, attack_metadata = load_attack_index(cfg.attack_file)
    used_ids = sorted({v[0] for v in TOKEN_TTP.values()} | {rule["ttp"][0] for rule in CONDITIONAL_TTP_RULES})
    deprecated = [tid for tid in used_ids if techniques.get(tid, {}).get("deprecated", False)]
    if deprecated:
        logger.warning("Deprecated or revoked ATT&CK IDs used in TOKEN_TTP: %s", ", ".join(deprecated))

    logger.info("Extracting per sample TTPs.")
    df_ttps = extract_ttps(df_feat, feat_cols, techniques, ttp_to_mitigations)
    if df_ttps.empty:
        logger.warning("No TTPs were extracted. Check whether feature columns contain binary active indicators.")

    freq = summarize_frequency(df_ttps, len(df_feat))
    merged, cluster_ttp, cluster_sizes = summarize_clusters(df_ttps, df_clusters, cluster_col)
    family_ttp = summarize_family(df_ttps, df_labels)
    strategic = build_strategic_profiles(df_feat, feat_cols, df_clusters, cluster_col, merged, cluster_ttp, cluster_sizes, cfg)
    mitigations = build_mitigation_table(df_ttps)
    flow = build_attack_flow_by_cluster(cluster_ttp)
    mapping = build_mapping_trace(df_feat, feat_cols, df_clusters, cluster_col, strategic, family_ttp, df_labels, ttp_to_mitigations, cfg)
    feature_mapping = build_feature_attack_mapping_table(feat_cols, techniques, ttp_to_mitigations)
    supporting_features = build_supporting_feature_table(df_feat, feat_cols)
    selected_cluster_labels = df_clusters[["sample_id", cluster_col]].rename(columns={cluster_col: "selected_cluster"})

    save_tables(cfg, {
        "sample_ttps": df_ttps,
        "ttp_frequency": freq,
        "cluster_ttp_profile": cluster_ttp,
        "family_ttp_profile": family_ttp,
        "mitigations": mitigations,
        "strategic_cluster_profile": strategic,
        "attack_flow_by_cluster": flow,
        "mitre_mapping_trace_table": mapping,
        "feature_attack_mapping_table": feature_mapping,
        "supporting_feature_mappings": supporting_features,
        "selected_cluster_labels": selected_cluster_labels,
        "selected_clustering_run_summary": selected_run_summary,
    })

    logger.info("Generating figures.")
    create_figures(cfg, df_ttps, freq, cluster_ttp, family_ttp, strategic, mitigations, flow, mapping, feature_mapping, len(df_feat))
    write_run_metadata(cfg, attack_metadata, cluster_col, cluster_note, df_feat, df_ttps, strategic)

    logger.info("Complete. Outputs written to %s", cfg.out_dir)
    print("[DONE] TTP extraction, ATT&CK mapping, attack flow analysis, mitigation ranking, and research figures complete. Step 5 is aligned with Hierarchical and Spectral clustering outputs.")


if __name__ == "__main__":
    main()
