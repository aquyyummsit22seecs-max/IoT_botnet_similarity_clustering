"""
IoT botnet clusteringa & mitigations using similarity measures.

Its our First step 1 of 6:  01_feature_extraction.py, for Behavioral Feature Extraction
Parses IoT botnet sandbox analysis artifacts of all samples in IoT_BDA dataset and produces a binary feature matrix.

Name: ABDUL QUYYUM 
MSIT22SEECS
NUST ISLAMABAD
"""

import os, re, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from log_utils import setup_script_logging

#Provide dataset Path as per your settings/ we set it as per our PC directory / expertimental settings 
DATASET = "../dataset/iot_bda_dataset/tasks/"

#output directory for features related outputs
OUT = "output/features"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
setup_script_logging("01_feature_extraction.py")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "savefig.dpi": 300, "figure.dpi": 100,
})

FAMILY_COLORS = {
    "mirai": "#E74C3C", "hajime": "#3498DB", "gafgyt": "#2ECC71",
    "bashlite": "#2ECC71", "unlabeled": "#95A5A6", "singleton": "#F39C12",
    "tsunami": "#1ABC9C", "dofloo": "#8E44AD",
}
FAMILY_COLOR = lambda f: FAMILY_COLORS.get(f, "#7F8C8D")
CAT_COLORS = {"Syscall": "#E74C3C", "Network": "#3498DB", "Behavioural": "#2ECC71"}

#03_Feature categories we ued in out feature exngineering setup
def feature_category(feat):
    if feat.startswith("sys_"): return "Syscall"
    if feat.startswith("net_"): return "Network"
    if feat.startswith("beh_"): return "Behavioural"
    return "Excluded"

if not os.path.isdir(DATASET):
    print(f"[ERROR] Dataset not found: {os.path.abspath(DATASET)}")
    sys.exit(1)

samples = sorted([s for s in glob.glob(os.path.join(DATASET, "*")) if os.path.isdir(s)])
if not samples:
    print(f"[ERROR] No sample directories in: {DATASET}")
    sys.exit(1)

print(f"[INFO] Found {len(samples)} sample directories")

#syscall tokens n=23 to extract 23 syscall related features from all samples artifacts
SYSCALL_TOKENS = {
    "execve": "sys_exec",    "fork":     "sys_fork",    "clone":    "sys_clone",
    "kill":   "sys_kill",    "prctl":    "sys_prctl",   "setsid":   "sys_daemonize",
    "socket": "sys_socket",  "connect":  "sys_connect", "bind":     "sys_bind",
    "listen": "sys_listen",  "send":     "sys_send",    "sendto":   "sys_sendto",
    "recv":   "sys_recv",    "recvfrom": "sys_recvfrom","recvmsg":  "sys_recvmsg",
    "select": "sys_select",  "open":     "sys_open",    "read":     "sys_read",
    "write":  "sys_write",   "getdents": "sys_dir_enum","mmap2":    "sys_mmap",
    "mprotect": "sys_mprotect", "nanosleep": "sys_sleep",
}


def extract_label(sample_dir):
    try:
        av = json.load(open(os.path.join(sample_dir, "analysis_result.json"))).get("avclass")
        if av is None: return "unlabeled"
        s = str(av).strip().lower()
        return "singleton" if s.startswith("singleton:") else s
    except Exception:
        return "unlabeled"


def parse_analysis_result(sample_dir, tokens):
    path = os.path.join(sample_dir, "analysis_result.json")
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return
    except json.JSONDecodeError as e:
        print(f"  [WARN] Bad JSON {sample_dir}: {e}")
        return

    na = d.get("network_analysis", {})
    scanning = na.get("scanning", [])

    if na.get("cnc"):
        tokens.add("net_cnc")
        for c in na["cnc"]:
            proto = str(c.get("protocol", "")).lower()
            if proto in ("tcp", "udp"):
                tokens.add(f"net_cnc_{proto}")
    if scanning:
        tokens.add("net_scanning")
        ports = {str(s.get("port", "")) for s in scanning}
        if "23" in ports or "2323" in ports: tokens.add("net_scan_telnet")
        if "37215" in ports or "7547" in ports: tokens.add("net_scan_tr069")
    if na.get("telnet_data"):   tokens.add("net_telnet_brute")
    if na.get("dns_questions"): tokens.add("net_dns")
    for anomaly in na.get("anomalies", []):
        name = anomaly.get("name", "").lower()
        if name == "syn_scan":              tokens.add("net_syn_scan")
        elif name == "blacklisted_ip_access": tokens.add("net_blacklisted_ip")
    if na.get("ddos"): tokens.add("net_ddos")
    p2p = na.get("p2p")
    if p2p is True or str(p2p).lower() == "true": tokens.add("net_p2p")
    if na.get("http_exploits") or na.get("http_requests"): tokens.add("net_http_exploit")

    ba = d.get("behavioural_analysis", {})
    open_files = ba.get("open_files", [])
    if any("/dev/watchdog" in x for x in open_files):       tokens.add("beh_watchdog")
    if any("/proc/net" in x for x in open_files):           tokens.add("beh_proc_net")
    if any(re.match(r"/proc/\d+/fd", x) for x in open_files): tokens.add("beh_proc_fd_enum")
    if ba.get("proc_rename"):    tokens.add("beh_proc_masquerade")
    if ba.get("persistence"):    tokens.add("beh_persistence")
    if ba.get("antivm"):         tokens.add("beh_antivm")
    if ba.get("antidbg"):        tokens.add("beh_antidbg")
    if ba.get("kernel_modules"): tokens.add("beh_kernel_module")
    if ba.get("process_inject"): tokens.add("beh_proc_inject")
    if ba.get("removed_files"):  tokens.add("beh_file_removal")
    if ba.get("stage_2_payload"):tokens.add("beh_stage2_drop")
    if ba.get("info_gathering"): tokens.add("beh_recon")


def parse_syscalls(sample_dir, tokens):
    try:
        calls = json.load(open(os.path.join(sample_dir, "syscalls.json")))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if not isinstance(calls, list):
        return
    seen = set()
    for call in calls:
        if isinstance(call, dict):
            name = call.get("name", "")
            if name in SYSCALL_TOKENS and name not in seen:
                tokens.add(SYSCALL_TOKENS[name])
                seen.add(name)


# machine.log: QEMU sandbox boot noise — excluded (zero malware signal)
# prog.log: sandbox orchestration output — excluded (not malware's own behaviour)

# ── Main processing ──
all_tokens, all_labels, empty_list = {}, {}, []
stats = {
    "total": len(samples), "valid": 0, "empty": 0,
    "has_machine_log": 0, "has_prog_log": 0, "has_result_log": 0, "empty_syscalls": 0,
}

for s in samples:
    sid = os.path.basename(s)
    for fname, key in [("machine.log","has_machine_log"),("prog.log","has_prog_log"),("result.log","has_result_log")]:
        if os.path.exists(os.path.join(s, fname)): stats[key] += 1
    try:
        sc = json.load(open(os.path.join(s, "syscalls.json")))
        if not isinstance(sc, list) or len(sc) == 0: stats["empty_syscalls"] += 1
    except Exception:
        stats["empty_syscalls"] += 1

    all_labels[sid] = extract_label(s)
    tokens = set()
    parse_analysis_result(s, tokens)
    parse_syscalls(s, tokens)

    if not tokens:
        empty_list.append(sid); stats["empty"] += 1
    else:
        all_tokens[sid] = tokens; stats["valid"] += 1

# ── Build binary feature matrix ──
feat_names = sorted({t for toks in all_tokens.values() for t in toks})
rows = [{"sample_id": sid, **{f: 1 if f in toks else 0 for f in feat_names}}
        for sid, toks in all_tokens.items()]
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/behavioral_features.csv", index=False)

df_lbl = pd.DataFrame([{"sample_id": sid, "family": lbl} for sid, lbl in all_labels.items()])
df_lbl.to_csv(f"{OUT}/ground_truth_labels.csv", index=False)

feat_cols = [c for c in df.columns if c != "sample_id"]
freq = df[feat_cols].sum().sort_values(ascending=False)
df_summary = pd.DataFrame({
    "feature": freq.index, "count": freq.values,
    "coverage_pct": (freq.values / len(df) * 100).round(1),
})
df_summary["category"] = df_summary["feature"].map(feature_category)
df_summary = df_summary[df_summary["category"] != "Excluded"].copy()
df_summary.to_csv(f"{OUT}/feature_summary.csv", index=False)

if empty_list:
    open(f"{OUT}/empty_samples.txt", "w").write("\n".join(empty_list))

label_dist = df_lbl["family"].value_counts()
n_total = len(df)
cats = {
    "Syscall":     int((df_summary["category"] == "Syscall").sum()),
    "Network":     int((df_summary["category"] == "Network").sum()),
    "Behavioural": int((df_summary["category"] == "Behavioural").sum()),
}


# ── PNG 1 — Feature Extraction Statistics (4-panel) ──
fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.42)

ax_a = fig.add_subplot(gs[0, 0])
comp_labels = ["Total\nSamples", "Valid\n(w/ feats)", "Empty\n(no feats)", "Empty\nSyscalls"]
comp_vals   = [stats["total"], stats["valid"], stats["empty"], stats["empty_syscalls"]]
comp_colors = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12"]
bars = ax_a.bar(comp_labels, comp_vals, color=comp_colors, edgecolor="white", linewidth=1.5, width=0.6)
for bar, v in zip(bars, comp_vals):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comp_vals)*0.02,
              str(v), ha="center", va="bottom", fontweight="bold", fontsize=11)
ax_a.set_title("Dataset Composition", fontweight="bold", pad=10)
ax_a.set_ylabel("Count")
ax_a.set_ylim(0, max(comp_vals) * 1.25)
ax_a.tick_params(axis="x", labelsize=9)
ax_a.grid(axis="y", alpha=0.3)

ax_b = fig.add_subplot(gs[0, 1])
file_labels = ["machine.log", "prog.log", "result.log"]
file_vals   = [stats["has_machine_log"], stats["has_prog_log"], stats["has_result_log"]]
file_colors = ["#9B59B6", "#1ABC9C", "#95A5A6"]
bars_b = ax_b.barh(file_labels, file_vals, color=file_colors, edgecolor="white", linewidth=1.2, height=0.5)
for bar, v in zip(bars_b, file_vals):
    ax_b.text(v + stats["total"]*0.01, bar.get_y() + bar.get_height()/2,
              f"{v} / {stats['total']}", va="center", fontsize=10)
ax_b.set_title("Optional Files Present", fontweight="bold", pad=10)
ax_b.set_xlabel("Number of Samples")
ax_b.set_xlim(0, stats["total"] * 1.3)
ax_b.grid(axis="x", alpha=0.3)

ax_c = fig.add_subplot(gs[0, 2])
cat_vals   = list(cats.values())
cat_labels = list(cats.keys())
cat_colors = [CAT_COLORS[c] for c in cat_labels]
wedges, _ = ax_c.pie(cat_vals, colors=cat_colors, startangle=90,
                      wedgeprops={"edgecolor": "white", "linewidth": 2})
total_feats = sum(cat_vals)
for wedge, val, label in zip(wedges, cat_vals, cat_labels):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x, y = 0.62 * np.cos(np.radians(angle)), 0.62 * np.sin(np.radians(angle))
    ax_c.annotate(f"{val}\n({val/total_feats*100:.0f}%)", xy=(x, y),
                  ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
legend_patches = [mpatches.Patch(color=CAT_COLORS[c], label=f"{c}  (n={cats[c]})") for c in cat_labels]
ax_c.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.38),
            fontsize=8.5, framealpha=0.9, ncol=2, title="Feature Category", title_fontsize=9)
ax_c.set_title(f"Feature Categories\n(Total: {total_feats} features)", fontweight="bold", pad=10)

ax_d = fig.add_subplot(gs[1, :])
top15 = df_summary.head(15)
bar_colors = [CAT_COLORS.get(feature_category(f), "#95A5A6") for f in top15["feature"]]
bars_d = ax_d.barh(top15["feature"][::-1], top15["coverage_pct"][::-1],
                   color=bar_colors[::-1], edgecolor="white", linewidth=0.8, height=0.65)
ax_d.set_xlabel("Coverage (% of Dataset Samples)", fontsize=11)
ax_d.set_title("Top 15 Behavioral Features by Dataset Coverage", fontweight="bold", pad=10)
for bar, pct in zip(bars_d, top15["coverage_pct"][::-1]):
    ax_d.text(pct + 0.4, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%", va="center", fontsize=9)
ax_d.set_xlim(0, 120)
ax_d.grid(axis="x", alpha=0.25)
ax_d.legend(handles=[mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS],
            loc="lower right", fontsize=9, framealpha=0.9, title="Category", title_fontsize=9)

fig.suptitle(f"Feature Extraction Statistics — IoT Botnet Dataset\n"
             f"(N={stats['total']} samples, {total_feats} behavioral features extracted)",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig(f"{OUT}/fig_feature_stats.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[DONE] {OUT}/fig_feature_stats.png")


# ── PNG Figure 2 — for Ground Truth Label Distribution ──
families = label_dist.index.tolist()
counts   = label_dist.values.tolist()
n_total_lbl = len(all_labels)
percentages = [(cnt / n_total_lbl) * 100 for cnt in counts]

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), gridspec_kw={"width_ratios": [1.05, 1.15]})
fig.patch.set_facecolor("#FAFAFA")

ax_bar = axes[0]
colors_bar = [FAMILY_COLOR(f) for f in families]
family_labels_clean = [str(f).title().replace("_", " ") for f in families]
bars = ax_bar.bar(range(len(families)), counts, color=colors_bar, edgecolor="white",
                  linewidth=2, width=0.62, zorder=3)
ax_bar.grid(axis="y", alpha=0.35, zorder=0)
for bar, cnt in zip(bars, counts):
    pct = cnt / n_total_lbl * 100
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                f"n={cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_bar.set_title("Sample Count by Botnet Family\n(AVClass Ground Truth Labels)", fontweight="bold", fontsize=12, pad=12)
ax_bar.set_xlabel("Botnet Family", fontsize=11)
ax_bar.set_ylabel("Number of Samples", fontsize=11)
ax_bar.set_ylim(0, max(counts) * 1.35)
ax_bar.set_xticks(range(len(families)))
ax_bar.set_xticklabels(family_labels_clean, rotation=35, ha="right", fontsize=9)
ax_bar.margins(x=0.04)
ax_bar.legend(handles=[mpatches.Patch(color=FAMILY_COLOR(f), label=f.title()) for f in families],
              fontsize=8.8, loc="upper right", framealpha=0.9, title="Family", title_fontsize=9)

ax_donut = axes[1]
wedges, _ = ax_donut.pie(counts, colors=[FAMILY_COLOR(f) for f in families], startangle=90,
                          counterclock=False, wedgeprops={"edgecolor": "white", "linewidth": 2.3, "width": 0.46})
ax_donut.text(0, 0.09, f"N = {n_total_lbl}", ha="center", va="center",
              fontsize=14, fontweight="bold", color="#2C3E50")
ax_donut.text(0, -0.14, "samples", ha="center", va="center", fontsize=10, color="#7F8C8D")
ax_donut.set_title("Family Distribution (Proportional View)", fontweight="bold", fontsize=12, pad=12)
legend_labels = [f"{fam.title()}  |  {cnt} samples  |  {pct:.1f}%"
                 for fam, cnt, pct in zip(families, counts, percentages)]
ax_donut.legend(wedges, legend_labels, title="Donut legend", title_fontsize=9.5, fontsize=8.9,
                loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95,
                borderaxespad=0.0, labelspacing=1.0)
ax_donut.set_aspect("equal")

labeled   = sum(1 for f in all_labels.values() if f not in ("unlabeled","singleton"))
unlabeled = n_total_lbl - labeled
fig.text(0.5, 0.02,
         f"Known botnet families: {labeled} samples ({labeled/n_total_lbl*100:.1f}%)   |   "
         f"Unlabeled / Singleton: {unlabeled} samples ({unlabeled/n_total_lbl*100:.1f}%)",
         ha="center", fontsize=10, color="#555555",
         bbox=dict(boxstyle="round,pad=0.4", fc="#ECF0F1", ec="#BDC3C7"))
fig.suptitle("Ground Truth Label Distribution — IoT Botnet Dataset", fontsize=14, fontweight="bold", y=0.99)
fig.subplots_adjust(left=0.06, right=0.87, top=0.86, bottom=0.12, wspace=0.28)
plt.savefig(f"{OUT}/fig_label_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[DONE] {OUT}/fig_label_distribution.png")


# ── PNG 3 — Feature Coverage Figure──
fig, axes = plt.subplots(1, 2, figsize=(18, max(9, len(feat_cols) * 0.33 + 2)))
ax_chart = axes[0]
df_plot  = df_summary.sort_values("coverage_pct")
bar_cols_all = [CAT_COLORS.get(feature_category(f), "#95A5A6") for f in df_plot["feature"]]
ax_chart.barh(df_plot["feature"], df_plot["coverage_pct"], color=bar_cols_all, edgecolor="none", height=0.7)
for pct, cnt, y in zip(df_plot["coverage_pct"], df_plot["count"], range(len(df_plot))):
    ax_chart.text(pct + 0.5, y, f"{pct:.1f}% (n={int(cnt)})", va="center", fontsize=8)
ax_chart.axvline(2,  color="#E74C3C", ls="--", lw=1.2, alpha=0.7, label="2% sparse threshold")
ax_chart.axvline(50, color="#27AE60", ls=":",  lw=1.2, alpha=0.7, label="50% threshold")
ax_chart.set_xlabel("Percentage of Samples (%)", fontsize=11)
ax_chart.set_title(f"All {len(feat_cols)} Features — Coverage Across Dataset", fontweight="bold", pad=10)
ax_chart.set_xlim(0, 125)
ax_chart.tick_params(axis="y", labelsize=8)
ax_chart.grid(axis="x", alpha=0.25)
ax_chart.legend(handles=[mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS],
                fontsize=8.5, loc="lower right", framealpha=0.9, title="Category", title_fontsize=9)

ax_tbl = axes[1]
ax_tbl.axis("off")
table_data = [[r["feature"], feature_category(r["feature"]), str(int(r["count"])), f"{r['coverage_pct']:.1f}%"]
              for _, r in df_summary.iterrows()]
tbl = ax_tbl.table(cellText=table_data, colLabels=["Feature Token", "Category", "Count", "Coverage %"],
                   cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1F4E79")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.035)
    else:
        feat_row = table_data[r-1][0] if r <= len(table_data) else ""
        if feat_row.startswith("sys_"):   cell.set_facecolor("#FADBD8")
        elif feat_row.startswith("net_"): cell.set_facecolor("#D6EAF8")
        elif feat_row.startswith("beh_"): cell.set_facecolor("#D5F5E3")
        else:                             cell.set_facecolor("#ECF0F1")
    cell.set_edgecolor("#CCCCCC")
ax_tbl.set_title("Feature Coverage Reference Table\n(Row color = feature category)",
                 fontweight="bold", pad=12, fontsize=11)
fig.suptitle("Behavioral Feature Coverage — IoT Botnet Dataset", fontsize=14, fontweight="bold")
plt.tight_layout(pad=1.5)
plt.savefig(f"{OUT}/fig_feature_coverage.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[DONE] {OUT}/fig_feature_coverage.png")


# ── PNG 4 — Feature Tables by Category ──
fig = plt.figure(figsize=(18, 10.5))
gs = GridSpec(1, 3, figure=fig, wspace=0.18)
for idx, cat in enumerate(["Syscall", "Network", "Behavioural"]):
    ax = fig.add_subplot(gs[0, idx])
    ax.axis("off")
    sub = df_summary[df_summary["category"] == cat].sort_values(["coverage_pct", "feature"], ascending=[False, True])
    cell_text = [[r["feature"], str(int(r["count"])), f"{r['coverage_pct']:.1f}%"] for _, r in sub.iterrows()]
    if not cell_text: cell_text = [["No features", "0", "0.0%"]]
    tbl = ax.table(cellText=cell_text, colLabels=["Feature", "Count", "Coverage"],
                   cellLoc="left", colLoc="left", loc="center", bbox=[0, 0, 1, 0.94])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2 if len(cell_text) <= 18 else 7.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D5D8DC")
        if r == 0:
            cell.set_facecolor(CAT_COLORS[cat])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1: cell.set_facecolor("#F8F9FA")
        else:             cell.set_facecolor("#EEF4FB")
    ax.set_title(f"{cat} features (n={len(sub)})", fontweight="bold", fontsize=12, pad=10)
fig.suptitle("Behavioral Feature Set by Category", fontsize=15, fontweight="bold", y=0.98)
plt.savefig(f"{OUT}/fig_feature_category_tables.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[DONE] {OUT}/fig_feature_category_tables.png")


# ── Summary ──
print("\n" + "="*58)
print("  FEATURE EXTRACTION COMPLETE")
print("="*58)
print(f"  Total samples      : {stats['total']}")
print(f"  Valid (w/features) : {stats['valid']}")
print(f"  Empty (no features): {stats['empty']}")
print(f"  Features extracted : {len(feat_names)}")
print(f"  Empty syscalls     : {stats['empty_syscalls']}")
print()
print("  GROUND TRUTH DISTRIBUTION:")
for fam, cnt in label_dist.items():
    bar = "█" * int(cnt / stats["total"] * 20)
    print(f"    {fam:<15}: {cnt:4d} ({cnt/stats['total']*100:5.1f}%)  {bar}")
print()
print("  FEATURE CATEGORIES:")
for cat, n_cat in cats.items():
    print(f"    {cat:<14}: {n_cat} features")
print("="*58)
