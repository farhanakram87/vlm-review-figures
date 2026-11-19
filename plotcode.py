r"""Generate plots from the digital pathology VLM review workbooks.

Install the dependencies:
    python -m pip install numpy pandas matplotlib openpyxl

Generate all four plots directly from the deposited extraction workbook:
    python plotcode.py --extraction-workbook VLM_Extracted_Data.xlsx --output-dir figures

Extraction_Clean supplies the task categories, model names, numeric metrics,
and study characteristics. Blank or nonnumeric metric cells are treated as
missing. The explicit N/A labels in the accessibility fields are retained.
The optional --review-workbook argument uses a separate Task_Distribution
and Benchmark_Matrix for the task distribution, heatmap, and violin plots.

Outputs are PNG and PDF versions of the task distribution, benchmark heatmap,
metric violin plots, and study overview. The source named
fig5_parameter_vs_performance.py generates the study overview. These sources
do not include a parameter-versus-performance bubble plot. Output names
describe the plots independently of manuscript numbering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


REVIEW_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# =============================================================================
# COLOR PALETTES
# =============================================================================

# Scientific color palette (colorblind-friendly)
# Based on Nature Color Palette
TASK_COLORS = {
    "Classification": "#0173B2",           # Blue
    "Captioning/Report Generation": "#029E73",  # Green
    "Prognosis/Survival": "#CC78BC",      # Magenta
    "VQA": "#ECE133",                     # Yellow
    "Segmentation": "#D55E00",            # Orange
    "Other": "#999999",                   # Gray
}

TASK_SHORT_LABELS = {
    "Captioning/Report Generation": "Captioning/reporting",
    "Prognosis/Survival": "Prognosis/survival",
}

# Heatmap colormap: low (blue) → high (red)
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "publication_heatmap",
    ["#E8EAEF", "#4472C4", "#70AD47", "#FFC000", "#C5504F"],
    N=256
)

OVERVIEW_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 12,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 600
}

def read_sheet(workbook, sheet, required_columns, *, keep_na_label=False):
    """Read an Excel sheet and report missing inputs with their location."""
    try:
        df = pd.read_excel(
            workbook, sheet_name=sheet, engine="openpyxl",
            keep_default_na=not keep_na_label, na_values=[""],
        )
    except ValueError as error:
        raise ValueError(f"{workbook.name}, sheet {sheet}: {error}") from error

    df = df.dropna(how="all")
    df.columns = df.columns.str.strip()
    missing = [column for column in required_columns if column not in df]
    if missing:
        raise ValueError(f"{workbook.name}, sheet {sheet}: missing columns {', '.join(missing)}")
    if df.empty:
        raise ValueError(f"{workbook.name}, sheet {sheet}: no data rows")
    return df


def count_tasks(studies):
    counts = studies["Task_Category"].value_counts()
    if counts.empty:
        raise ValueError("No task categories found.")
    return counts.rename_axis("Task_Category").reset_index(name="Count")


def prepare_benchmark(studies, sheet):
    metrics = [column for column in BENCHMARK_METRICS if column in studies]
    if not metrics:
        raise ValueError(f"{sheet}: no recognized metric columns found.")
    if "Model_Name" not in studies:
        raise ValueError(f"{sheet}: missing column Model_Name.")

    raw = studies[metrics].replace({"N/A": np.nan, "NA": np.nan, "-": np.nan, "—": np.nan})
    numeric = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ignored = (raw.notna() & numeric.isna()).sum()
    for metric, count in ignored[ignored > 0].items():
        print(f"{sheet}: treating {count} nonnumeric {metric} entries as missing.")
    return studies[["Task_Category", "Model_Name"]].join(numeric)


def save_figure(fig, output_dir, name, *, dpi=300):
    """Save both a raster copy and a PDF of the figure."""
    try:
        for extension in ("png", "pdf"):
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved {path}")
    finally:
        plt.close(fig)


def add_figure_label(fig, letter, title, subtitle="", y_pos=0.98):
    """Add publication-style figure label."""
    fig.text(
        0.05, y_pos,
        f"Figure {letter} | {title}",
        fontsize=12, fontweight="bold", va="top"
    )
    if subtitle:
        fig.text(
            0.05, y_pos - 0.04,
            subtitle,
            fontsize=8.5, style="italic", va="top", color="#555555", wrap=True
        )

# FIGURE A: TASK DISTRIBUTION (SEPARATE SUBFIGURES)
# =============================================================================

def plot_task_distribution(task_counts, output_dir):
    """
    FIGURE A:
    A1 = Donut/Pie chart (distribution)
    A2 = Bar chart (absolute counts)
    """

    # Sort by count descending
    task_counts = task_counts.sort_values("Count", ascending=False)
    total = task_counts["Count"].sum()

    # Colors
    colors = [TASK_COLORS.get(task, "#999999") for task in task_counts["Task_Category"]]

    # Create full figure
    fig = plt.figure(figsize=(10, 4.5))

    # =========================
    # A1: PIE / DONUT CHART
    # =========================
    ax1 = fig.add_subplot(1, 2, 1)

    wedges, texts, autotexts = ax1.pie(
        task_counts["Count"],
        labels=task_counts["Task_Category"],
        colors=colors,
        startangle=90,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        pctdistance=0.75,
        labeldistance=1.05,
        wedgeprops=dict(edgecolor="white", linewidth=1.5, width=0.5)
    )

    # Style labels
    for text in texts:
        text.set_fontsize(8.5)
        text.set_color("#333333")

    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(8)

    ax1.set_title("A1. Task Distribution (%)", fontsize=11, fontweight="bold")

    # =========================
    # A2: BAR CHART
    # =========================
    ax2 = fig.add_subplot(1, 2, 2)

    bars = ax2.barh(
        range(len(task_counts)),
        task_counts["Count"],
        color=colors,
        edgecolor="white",
        linewidth=0.8
    )

    for bar, value in zip(bars, task_counts["Count"]):
        ax2.text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    ax2.set_yticks(range(len(task_counts)))
    ax2.set_yticklabels(task_counts["Task_Category"], fontsize=9)
    ax2.set_xlabel("Number of Studies", fontsize=10)
    ax2.set_xlim(0, max(task_counts["Count"]) * 1.15)
    ax2.grid(axis="x", alpha=0.3)

    ax2.set_title("A2. Task Distribution (Counts)", fontsize=11, fontweight="bold")

    # =========================
    # FIGURE LABEL
    # =========================
    add_figure_label(
        fig,
        "A",
        "Task Category Distribution",
        f"Distribution across {total} studies showing proportional (A1) and absolute (A2) views."
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, output_dir, "task_distribution")

# =============================================================================
# FIGURE B: PERFORMANCE HEATMAP
# =============================================================================

def plot_benchmark_heatmap(benchmark_df, output_dir):
    """
    FIGURE B: Performance Heatmap by Task Category

    Design choices:
    - Models grouped by task category
    - Normalized performance metrics (0-1 scale)
    - Color intensity indicates performance (blue=low, red=high)
    - Missing values shown as light gray
    - A metric column needs two observations, and a row needs one selected metric
    """

    df = benchmark_df.copy()

    # Define task category order
    cat_order = [
        "Classification",
        "Captioning/Report Generation",
        "Prognosis/Survival",
        "Segmentation",
        "VQA",
        "Other"
    ]

    # Filter and sort by task category
    df = df[df["Task_Category"].notna()].copy()
    df["Task_Category"] = pd.Categorical(
        df["Task_Category"], categories=cat_order, ordered=True
    )
    df = df.sort_values("Task_Category").reset_index(drop=True)

    # Select metric columns (exclude sparse metrics like METEOR, CIDEr)
    metric_cols = [c for c in ["ACC", "F1", "AUC", "BLEU", "ROUGE", "C-index", "Dice"]
                   if c in df.columns and df[c].notna().sum() >= 2]

    # Include rows with at least one of the selected metrics.
    df = df[df[metric_cols].notna().sum(axis=1) > 0].reset_index(drop=True)

    if df.empty:
        print("Skipping benchmark heatmap: no metric has at least two observations.")
        return

    # === NORMALIZE METRICS ===
    norm_matrix = np.full((len(df), len(metric_cols)), np.nan)
    val_matrix = np.full((len(df), len(metric_cols)), np.nan)

    for j, metric in enumerate(metric_cols):
        vals = pd.to_numeric(df[metric], errors="coerce")
        vals_valid = vals[vals.notna()]

        if len(vals_valid) > 0:
            lo, hi = vals_valid.min(), vals_valid.max()

            for i, v in enumerate(vals):
                if pd.notna(v):
                    # Normalize to 0-1
                    norm = (v - lo) / (hi - lo + 1e-10)
                    norm_matrix[i, j] = np.clip(norm, 0, 1)
                    val_matrix[i, j] = v

    # === BUILD HEATMAP ===
    nrows, ncols = norm_matrix.shape
    fig_h = max(8, nrows * 0.35 + 3.5)

    fig = plt.figure(figsize=(10, fig_h))
    ax = fig.add_axes([0.25, 0.08, 0.55, 0.82])
    ax.set_facecolor("#ffffff")

    cell_width, cell_height = 0.92, 0.88

    # Draw cells
    for i in range(nrows):
        for j in range(ncols):
            n = norm_matrix[i, j]
            v = val_matrix[i, j]

            # Color based on normalized performance
            if pd.notna(n):
                fc = HEAT_CMAP(n)
            else:
                fc = "#e8e8e8"  # Missing data

            rect = mpatches.FancyBboxPatch(
                (j - 0.46, i - 0.44), cell_width, cell_height,
                boxstyle="round,pad=0.03",
                linewidth=0.3, facecolor=fc, edgecolor="#cccccc"
            )
            ax.add_patch(rect)

            # Value text
            if pd.notna(v):
                txt = f"{v:.2f}" if v <= 1 else f"{v:.1f}"
                ax.text(
                    j, i, txt, ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white" if n > 0.5 else "#333333"
                )
            else:
                ax.text(
                    j, i, "—", ha="center", va="center",
                    fontsize=8, color="#999999"
                )

    # === CATEGORY SEPARATORS ===
    for category, group in df.groupby("Task_Category", observed=True, dropna=False):
        ax.hlines(group.index[0] - 0.5, -0.6, ncols - 0.4, color="white", linewidth=2.0, alpha=0.5)

    # === AXIS SETUP ===
    ax.set_xlim(-1.0, ncols - 0.45)
    ax.set_ylim(-0.6, nrows - 0.4)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(metric_cols, fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")

    ax.set_yticks(range(nrows))
    ax.set_yticklabels(df["Model_Name"], fontsize=7.5)
    ax.invert_yaxis()

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0)

    # Leave enough room for the longest model name before the category labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    model_left = min(label.get_window_extent(renderer).x0 for label in ax.get_yticklabels())
    label_x = ax.transAxes.inverted().transform((model_left - 8 * fig.dpi / 72, 0))[0]
    for category, group in df.groupby("Task_Category", observed=True, dropna=False):
        label = TASK_SHORT_LABELS.get(category, category) if pd.notna(category) else "Uncategorized"
        ax.text(
            label_x, (group.index[0] + group.index[-1]) / 2, label,
            transform=ax.get_yaxis_transform(), va="center", ha="right",
            fontsize=8, fontweight="bold", color=TASK_COLORS.get(category, "#666666"),
        )

    # === COLORBAR ===
    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.82, 0.15, 0.015, 0.60])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="black", labelsize=8, length=2)
    cbar.set_label("Normalized\nPerformance", fontsize=9, fontweight="bold", labelpad=10)

    add_figure_label(
        fig, "B",
        "Performance Heatmap by Task Category",
        "Models grouped by primary task. Cell labels show reported values. Color is scaled within each metric. "
        "Gray cells indicate unreported metrics."
    )

    save_figure(fig, output_dir, "benchmark_heatmap")

# =============================================================================
# FIGURE C: COMPREHENSIVE METRIC SUMMARY
# =============================================================================

def plot_metric_violins(benchmark_df, output_dir):
    """
    FIGURE C: Metric Distribution by Task Category

    Shows distribution of each metric across task categories using violin plots
    with overlaid study/article dots.
    """

    df = benchmark_df[benchmark_df["Task_Category"].notna()]

    # Select key metrics with sufficient data
    metrics_to_plot = []
    for metric in ["ACC", "F1", "AUC", "BLEU", "C-index"]:
        if metric in df.columns and df[metric].notna().sum() >= 3:
            metrics_to_plot.append(metric)

    if len(metrics_to_plot) < 2:
        print("Skipping metric violins: fewer than two metrics have three observations.")
        return

    fig, axes = plt.subplots(
        1,
        len(metrics_to_plot),
        figsize=(16, 5.5),
        sharey=False
    )

    rng = np.random.default_rng(0)

    for ax, metric in zip(axes, metrics_to_plot):

        data_for_plot = df[["Task_Category", metric]].dropna()

        if len(data_for_plot) > 0:

            # =========================
            # PREPARE DATA
            # =========================
            task_cats = sorted(data_for_plot["Task_Category"].unique())

            data_by_cat = [
                data_for_plot[
                    data_for_plot["Task_Category"] == cat
                ][metric].values
                for cat in task_cats
            ]

            # =========================
            # VIOLIN PLOT
            # =========================
            parts = ax.violinplot(
                data_by_cat,
                positions=range(len(task_cats)),
                showmeans=True,
                showmedians=True,
                widths=0.72
            )

            # Style violin bodies
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(
                    TASK_COLORS.get(task_cats[i], "#999999")
                )
                pc.set_alpha(0.55)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.6)

            # Style mean/median lines
            for partname in (
                'cbars',
                'cmins',
                'cmaxes',
                'cmeans',
                'cmedians'
            ):
                vp = parts[partname]
                vp.set_edgecolor("#333333")
                vp.set_linewidth(0.8)

            # =========================
            # OVERLAY STUDY DOTS
            # =========================
            for i, cat in enumerate(task_cats):

                y = data_for_plot[
                    data_for_plot["Task_Category"] == cat
                ][metric].values

                # Add horizontal jitter
                x = rng.normal(
                    loc=i,
                    scale=0.05,
                    size=len(y)
                )

                ax.scatter(
                    x,
                    y,
                    s=24,
                    color="black",
                    alpha=0.75,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3
                )

                # Add sample size
                ax.annotate(
                    f"n={len(y)}",
                    (i, max(y)),
                    xytext=(0, 4),
                    textcoords="offset points",
                    annotation_clip=False,
                    ha="center",
                    fontsize=7,
                    color="#444444"
                )

            # =========================
            # AXIS STYLING
            # =========================
            ax.set_xticks(range(len(task_cats)))

            ax.set_xticklabels(
                [TASK_SHORT_LABELS.get(category, category) for category in task_cats],
                rotation=45,
                ha="right",
                fontsize=8
            )

            ax.set_ylabel(
                metric,
                fontsize=10,
                fontweight="bold"
            )

            ax.set_title(
                metric,
                fontsize=10,
                fontweight="bold",
                pad=8
            )

            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.margins(y=0.15)

            # Clean spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    add_figure_label(
        fig,
        "C",
        "Metric Distribution Across Task Categories",
        "Violin plots with overlaid study-level observations. "
        "Black dots represent individual articles/studies; "
        "violin width indicates density distribution."
    )

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.30, top=0.79, wspace=0.52)
    save_figure(fig, output_dir, "metric_violins")



def plot_study_overview(df, output_dir):
    """Plot the five-panel review overview from the extraction workbook."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], wspace=0.65, hspace=0.35,
        left=0.06, right=0.92, bottom=0.14, top=0.94,
    )

    # Colors
    blue = "#0B7DB5"
    green = "#0BA374"
    pink = "#CC70C2"
    gray = "#9E9E9E"
    orange = "#E67800"
    yellow = "#FFF200"
    light_blue = "#6BAED6"
    pale_blue = "#D7E9F7"
    light_green = "#A8E6C0"
    dark_blue = "#2E4359"
    light_gray = "#B5BEC7"
    purple = "#873FA3"
    light_purple = "#D8B4E2"

    # =========================
    # A. VLM Task Distribution
    # =========================
    ax1 = fig.add_subplot(gs[0, 0])

    task_order = [
        "Classification",
        "Captioning/Report Generation",
        "Prognosis/Survival",
        "Other",
        "Segmentation",
        "VQA"
    ]

    task_counts = df["Task_Category"].value_counts()
    task_vals = [task_counts.get(x, 0) for x in task_order]
    task_labels = [
        "Classification",
        "Captioning/\nReport Generation",
        "Prognosis/\nSurvival",
        "Other",
        "Segmentation",
        "VQA"
    ]
    task_colors = [blue, green, pink, gray, orange, yellow]

    if not any(task_vals):
        plt.close(fig)
        raise ValueError("Extraction_Clean: Task_Category has no recognized values.")

    ax1.pie(
        task_vals,
        labels=task_labels,
        colors=task_colors,
        startangle=90,
        counterclock=True,
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2),
        autopct=lambda p: f"{p:.1f}%" if p % 1 else f"{int(p)}%",
        pctdistance=0.72,
        labeldistance=1.08
    )

    ax1.set_title("A. VLM TASK Distribution", loc="left", fontsize=16, pad=15)
    ax1.set_aspect("equal")

    # =========================
    # B. Dataset Characteristics
    # =========================
    ax2 = fig.add_subplot(gs[0, 1])

    public_counts = df["Public Dataset"].value_counts()
    external_counts = df["External Dataset"].value_counts()

    organs_clean = df["Organs"].replace({"Single-organ": "Single"})
    organ_counts = organs_clean.value_counts()

    rings = [
        (
            [public_counts.get("Public", 0),
             public_counts.get("Private+Public", 0),
             public_counts.get("Private", 0)],
            [blue, light_blue, pale_blue],
            1.00
        ),
        (
            [external_counts.get("Internal", 0),
             external_counts.get("External", 0)],
            [orange, "#F5B77C"],
            0.72
        ),
        (
            [organ_counts.get("Single", 0),
             organ_counts.get("Multiples", 0)],
            [green, light_green],
            0.44
        )
    ]

    for values, colors, radius in rings:
        if not any(values):
            plt.close(fig)
            raise ValueError("Extraction_Clean: a study overview ring has no recognized values.")
        ax2.pie(
            values,
            radius=radius,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.22, edgecolor="white", linewidth=2),
            autopct=lambda p: f"{p:.0f}%",
            pctdistance=1 - 0.11 / radius,
            textprops={"fontsize": 9},
        )

    ax2.set_title("B. Dataset Characteristics", loc="left", fontsize=16, pad=15)
    ax2.set_aspect("equal")

    ax2.legend(
        ["Public", "Private+Public", "Private",
         "Internal Validation", "External Validation",
         "Single Organ", "Multiple Organs"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10
    )

    # =========================
    # C. Accessibility & Reproducibility
    # =========================
    ax3 = fig.add_subplot(gs[0, 2])

    gpu_counts = df["Computing (GPU)"].replace({"Available": "Reported", "N/A": "Not Reported"}).value_counts()
    data_counts = df["Data Accessibility"].replace({"Yes": "Accessible", "N/A": "Not Accessible"}).value_counts()
    code_counts = df["Code Availability"].replace({"Yes": "Available", "N/A": "Not Available"}).value_counts()

    rings_c = [
        (
            [gpu_counts.get("Reported", 0), gpu_counts.get("Not Reported", 0)],
            [dark_blue, light_gray],
            1.00
        ),
        (
            [data_counts.get("Accessible", 0), data_counts.get("Not Accessible", 0)],
            [green, "#A9EAD8"],
            0.72
        ),
        (
            [code_counts.get("Available", 0), code_counts.get("Not Available", 0)],
            [purple, light_purple],
            0.44
        )
    ]

    for values, colors, radius in rings_c:
        if not any(values):
            plt.close(fig)
            raise ValueError("Extraction_Clean: a study overview ring has no recognized values.")
        ax3.pie(
            values,
            radius=radius,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.22, edgecolor="white", linewidth=2),
            autopct=lambda p: f"{p:.0f}%",
            pctdistance=1 - 0.11 / radius,
            textprops={"fontsize": 9},
        )

    ax3.set_title("C. Accessibility & Reproducibility", loc="left", fontsize=16, pad=15)
    ax3.set_aspect("equal")

    ax3.legend(
        ["GPU Reported", "GPU Not Reported",
         "Data Accessible", "Data Not Accessible",
         "Code Available", "Code Not Available"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10
    )

    # =========================
    # D. Publication per Year
    # =========================
    ax4 = fig.add_subplot(gs[1, 0:1])

    year_counts = df["Year"].value_counts().sort_index()
    ax4.bar(year_counts.index.astype(str), year_counts.values, color="#5B9BD5", width=0.32)

    ax4.set_title("D. Publications per year", loc="left", fontsize=16, pad=15)
    ax4.grid(axis="y", color="#D9D9D9", linewidth=1)
    ax4.set_axisbelow(True)
    ax4.spines[["top", "right", "left"]].set_visible(False)
    ax4.tick_params(axis="y", length=0)
    ax4.tick_params(axis="x", length=0, labelsize=14)

    # =========================
    # E. Countries Contribution
    # =========================
    ax5 = fig.add_subplot(gs[1, 1:3])

    country_series = df["Country"].dropna().astype(str)

    country_list = []
    for item in country_series:
        parts = [x.strip() for x in item.split(",")]
        country_list.extend(parts)

    country_counts = pd.Series(country_list).replace({
        "Hong kong": "Hong Kong",
    }).value_counts().sort_values()

    ax5.bar(range(len(country_counts)), country_counts.values, color="#5B9BD5", width=0.5)

    ax5.set_xticks(range(len(country_counts)))
    ax5.set_xticklabels(country_counts.index, rotation=55, ha="right", fontsize=10)

    for i, v in enumerate(country_counts.values):
        ax5.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax5.set_title("E. Countries Contribution Towards VLM Research", loc="left", fontsize=16, pad=15)
    ax5.grid(axis="y", color="#D9D9D9", linewidth=1)
    ax5.set_axisbelow(True)
    ax5.spines[["top", "right", "left"]].set_visible(False)
    ax5.tick_params(axis="y", length=0)
    ax5.tick_params(axis="x", length=0)

    save_figure(fig, output_dir, "study_overview", dpi=600)


BENCHMARK_METRICS = ("ACC", "F1", "AUC", "BLEU", "ROUGE", "C-index", "Dice")
EXTRACTION_COLUMNS = (
    "Task_Category", "Public Dataset", "External Dataset", "Organs",
    "Computing (GPU)", "Data Accessibility", "Code Availability", "Year", "Country",
)


def main():
    parser = argparse.ArgumentParser(description="Generate figures for the digital pathology VLM review.")
    parser.add_argument("--review-workbook", type=Path,
                        help="Excel workbook with the Task_Distribution and Benchmark_Matrix sheets.")
    parser.add_argument("--extraction-workbook", type=Path,
                        help="Excel workbook with Extraction_Clean, such as VLM_Extracted_Data.xlsx.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="Destination for PNG and PDF files (default: figures).")
    args = parser.parse_args()

    if args.review_workbook is None and args.extraction_workbook is None:
        parser.error("provide --review-workbook, --extraction-workbook, or both")

    try:
        benchmark = None
        if args.review_workbook is not None:
            tasks = read_sheet(args.review_workbook, "Task_Distribution", ("Task_Category",))
            task_counts = count_tasks(tasks)
            benchmark_source = read_sheet(
                args.review_workbook, "Benchmark_Matrix", ("Task_Category", "Model_Name"),
            )
            benchmark = prepare_benchmark(benchmark_source, "Benchmark_Matrix")

        if args.extraction_workbook is not None:
            # The source uses literal N/A as an explicit accessibility category.
            extraction = read_sheet(
                args.extraction_workbook, "Extraction_Clean", EXTRACTION_COLUMNS, keep_na_label=True,
            )
            extraction = extraction.map(lambda value: value.strip() if isinstance(value, str) else value)
            years = pd.to_numeric(extraction["Year"], errors="coerce")
            if years.isna().any() or years.mod(1).ne(0).any():
                raise ValueError("Extraction_Clean: each study must have an integer publication year.")
            extraction["Year"] = years.astype(int)
            print(f"Loaded {len(extraction)} studies from {args.extraction_workbook.name}.")
            if benchmark is None and any(metric in extraction for metric in BENCHMARK_METRICS):
                task_counts = count_tasks(extraction)
                benchmark = prepare_benchmark(extraction, "Extraction_Clean")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if benchmark is not None:
            with plt.rc_context(REVIEW_STYLE):
                plot_task_distribution(task_counts, args.output_dir)
                plot_benchmark_heatmap(benchmark, args.output_dir)
                plot_metric_violins(benchmark, args.output_dir)
        if args.extraction_workbook is not None:
            with plt.rc_context(OVERVIEW_STYLE):
                plot_study_overview(extraction, args.output_dir)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
