"""Shared IEEE-style visualization utilities for the IoT botnet clustering pipeline."""
import textwrap
import matplotlib.pyplot as plt

ACADEMIC_DPI = 300
IEEE_FONT_FAMILY        = "DejaVu Sans"
IEEE_BASE_FONT_SIZE     = 9
IEEE_TITLE_FONT_SIZE    = 10
IEEE_LABEL_FONT_SIZE    = 9
IEEE_TICK_FONT_SIZE     = 8
IEEE_TABLE_FONT_SIZE    = 7.4
IEEE_TABLE_HEADER_FONT_SIZE = 7.6

CATEGORY_COLORS = {
    "Syscall":     "#E74C3C",
    "Network":     "#3498DB",
    "Behavioural": "#2ECC71",
}
FAMILY_COLORS = {
    "mirai":      "#E74C3C",
    "hajime":     "#3498DB",
    "gafgyt":     "#2ECC71",
    "bashlite":   "#27AE60",
    "unlabeled":  "#95A5A6",
    "singleton":  "#F39C12",
    "tsunami":    "#1ABC9C",
    "dofloo":     "#8E44AD",
    "lightaidra": "#16A085",
}


def set_academic_style():
    """Apply consistent IEEE-style plotting defaults."""
    plt.rcParams.update({
        "font.family":      IEEE_FONT_FAMILY,
        "font.size":        IEEE_BASE_FONT_SIZE,
        "axes.titlesize":   IEEE_TITLE_FONT_SIZE,
        "axes.labelsize":   IEEE_LABEL_FONT_SIZE,
        "xtick.labelsize":  IEEE_TICK_FONT_SIZE,
        "ytick.labelsize":  IEEE_TICK_FONT_SIZE,
        "legend.fontsize":  7.8,
        "figure.dpi":       120,
        "savefig.dpi":      ACADEMIC_DPI,
        "axes.grid":        False,
        "axes.linewidth":   0.8,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
    })


def save_figure(fig, path, pad=0.12):
    fig.savefig(path, dpi=ACADEMIC_DPI, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def family_color(family):
    return FAMILY_COLORS.get(str(family).lower(), "#7F8C8D")


def feature_category(feature):
    feature = str(feature)
    if feature.startswith("sys_"): return "Syscall"
    if feature.startswith("net_"): return "Network"
    if feature.startswith("beh_"): return "Behavioural"
    return "Excluded"


def wrap_label(text, width=18):
    text = str(text)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def wrap_table_text(text, width=24):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def rotate_xticklabels(ax, angle=35):
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_ha("right")


def table_set_widths(tbl, widths):
    for (r, c), cell in tbl.get_celld().items():
        if c < len(widths):
            cell.set_width(widths[c])


def style_table(tbl, header_color="#1F4E79", header_text_color="white",
                alt1="#F8F9FA", alt2="#EEF4FB", edge="#C9D3DD",
                fontsize=None, header_fontsize=None, row_height=0.105, header_height=0.125):
    """Style matplotlib tables so text stays inside cells."""
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize or IEEE_TABLE_FONT_SIZE)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.55)
        cell.PAD = 0.018
        txt = cell.get_text()
        txt.set_wrap(True)
        txt.set_va("center")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color=header_text_color, fontweight="bold", va="center", ha="center")
            cell.set_height(header_height)
            txt.set_fontsize(header_fontsize or IEEE_TABLE_HEADER_FONT_SIZE)
        else:
            cell.set_facecolor(alt1 if r % 2 == 1 else alt2)
            cell.set_height(row_height)


def annotate_bars(ax, bars, fmt="{:.3f}", dy=0.015, fontsize=7.8):
    for bar in bars:
        h = bar.get_height()
        label = fmt.format(h) if isinstance(h, float) else str(h)
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, label,
                ha="center", va="bottom", fontsize=fontsize)
