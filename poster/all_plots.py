import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#0F1117"
PANEL = "#1A1D27"
BORDER = "#2E3347"
ACCENT1 = "#6C8EFF"  # before / 1.7B
ACCENT2 = "#A78BFA"  # after  / 4B
ACCENT3 = "#34D399"  # recovered / 8B
WARN = "#F87171"  # degraded
NEUTRAL = "#94A3B8"  # text secondary
WHITE = "#F1F5F9"
GOLD = "#FBBF24"

MODEL_COLORS = {"1.7B": ACCENT1, "4B": ACCENT2, "8B": ACCENT3}
CAT_COLORS = [ACCENT1, ACCENT2, ACCENT3]

matplotlib.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "text.color": WHITE,
        "axes.labelcolor": WHITE,
        "xtick.color": NEUTRAL,
        "ytick.color": NEUTRAL,
        "axes.edgecolor": BORDER,
        "grid.color": BORDER,
        "font.family": "DejaVu Sans",
    }
)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = "src"
specs = {
    "1.7B": ("qwen3-1.7b_summary.csv", "qwen3-1.7b_rethink_summary.csv"),
    "4B": ("qwen3-4b_summary.csv", "qwen3-4b_rethink_summary.csv"),
    "8B": ("qwen3-8b_summary.csv", "qwen3-8b_rethink_summary.csv"),
}


def load(model_key):
    sf, rf = specs[model_key]
    s = pd.read_csv(f"{DATA_DIR}/{sf}")
    r = pd.read_csv(f"{DATA_DIR}/{rf}")
    n = len(r)
    p2 = r["p2_correct"].sum()
    r["final_correct"] = r.apply(
        lambda x: x["rt_correct"] if x["has_rethink"] else x["p2_correct"], axis=1
    )
    final = r["final_correct"].sum()
    rethink_rows = r[r["has_rethink"]]
    return {
        "model": model_key,
        "n": n,
        "p2_acc": 100 * p2 / n,
        "rt_acc": 100 * final / n,
        "delta": 100 * (final - p2) / n,
        "rethink_n": len(rethink_rows),
        "recovered": int(rethink_rows["recovered"].sum()),
        "degraded": int(rethink_rows["degraded"].sum()),
        "unchanged": int(rethink_rows["unchanged"].sum()),
        "recovery_rate": 100
        * rethink_rows["recovered"].sum()
        / max(len(rethink_rows) - rethink_rows["p2_correct"].sum(), 1),
        "rethink_df": rethink_rows,
        "summary_df": s,
        "full_df": r,
    }


models_data = {k: load(k) for k in ["1.7B", "4B", "8B"]}


# ── Helper ────────────────────────────────────────────────────────────────────
def styled_ax(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=NEUTRAL, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=NEUTRAL, fontsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=BORDER, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Summary Stats Table + Before/After Bars
# ═════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(14, 9), facecolor=BG)
fig1.suptitle(
    "SCALPEL: Dissecting Tool Use in LLM Agents",
    fontsize=15,
    fontweight="bold",
    color=WHITE,
    y=0.97,
)
fig1.text(
    0.5,
    0.935,
    "Qwen3 {1.7B, 4B, 8B}  |  Incorrect tool output injected",
    ha="center",
    fontsize=9,
    color=NEUTRAL,
)

gs = gridspec.GridSpec(
    2,
    2,
    figure=fig1,
    left=0.06,
    right=0.97,
    top=0.90,
    bottom=0.07,
    hspace=0.42,
    wspace=0.30,
)

# ── 1a. Summary table ────────────────────────────────────────────────────────
ax_tbl = fig1.add_subplot(gs[0, :])
ax_tbl.set_facecolor(PANEL)
ax_tbl.axis("off")

col_labels = [
    "Model",
    "Accuracy\n(Before)",
    "Accuracy\n(After)",
    "Δ Accuracy",
    "Rethink\nTriggered",
    "Recovered",
    "Degraded",
    "Recovery\nRate",
]
rows = []
for key in ["1.7B", "4B", "8B"]:
    d = models_data[key]
    rows.append(
        [
            f"Qwen3-{key}",
            f"{d['p2_acc']:.1f}%",
            f"{d['rt_acc']:.1f}%",
            f"+{d['delta']:.1f}pp",
            f"{d['rethink_n']} / {d['n']}",
            str(d["recovered"]),
            str(d["degraded"]),
            f"{d['recovery_rate']:.1f}%",
        ]
    )

tbl = ax_tbl.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.1)

col_widths = [0.14, 0.11, 0.11, 0.12, 0.14, 0.11, 0.11, 0.13]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(BORDER)
    cell.set_linewidth(0.8)
    if r == 0:  # header
        cell.set_facecolor("#252A3D")
        cell.set_text_props(color=GOLD, fontweight="bold", fontsize=9.5)
    else:
        model_key = ["1.7B", "4B", "8B"][r - 1]
        base = "#1E2235"
        cell.set_facecolor(base)
        txt_color = WHITE
        # highlight delta column
        if c == 3:
            cell.set_facecolor("#1C2C23")
            cell.set_text_props(color=ACCENT3, fontweight="bold")
        # highlight recovery rate
        elif c == 7:
            cell.set_facecolor("#1C2234")
            cell.set_text_props(color=ACCENT2, fontweight="bold")
        elif c == 6:  # degraded
            cell.set_text_props(color=WARN)
        else:
            cell.set_text_props(color=WHITE)
    if c < len(col_widths):
        cell.set_width(col_widths[c])

ax_tbl.set_title(
    "Overall Performance Summary",
    color=WHITE,
    fontsize=11,
    fontweight="bold",
    pad=5,
    loc="left",
    x=0.02,
)

# ── 1b. Before / After accuracy bars ─────────────────────────────────────────
ax_acc = fig1.add_subplot(gs[1, 0])
styled_ax(ax_acc, title="Accuracy Before vs After Rethink", ylabel="Accuracy (%)")

model_keys = ["1.7B", "4B", "8B"]
x = np.arange(3)
w = 0.32
before_vals = [models_data[k]["p2_acc"] for k in model_keys]
after_vals = [models_data[k]["rt_acc"] for k in model_keys]

b1 = ax_acc.bar(
    x - w / 2,
    before_vals,
    w,
    color=ACCENT1,
    alpha=0.85,
    label="Before Rethink",
    zorder=3,
)
b2 = ax_acc.bar(
    x + w / 2, after_vals, w, color=ACCENT3, alpha=0.85, label="After Rethink", zorder=3
)

for bar, val in zip(b1, before_vals):
    ax_acc.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=ACCENT1,
        fontweight="bold",
    )
for bar, val in zip(b2, after_vals):
    ax_acc.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=ACCENT3,
        fontweight="bold",
    )

ax_acc.set_xticks(x)
ax_acc.set_xticklabels([f"Qwen3-{k}" for k in model_keys], fontsize=9.5)
ax_acc.set_ylim(0, 105)
ax_acc.legend(
    fontsize=8.5, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE
)

# ── 1c. Rethink outcome stacked bars ─────────────────────────────────────────
ax_out = fig1.add_subplot(gs[1, 1])
styled_ax(ax_out, title="Rethink Outcomes (triggered cases)", ylabel="Count")

recovered = [models_data[k]["recovered"] for k in model_keys]
unchanged = [models_data[k]["unchanged"] for k in model_keys]
degraded = [models_data[k]["degraded"] for k in model_keys]
rethink_n = [models_data[k]["rethink_n"] for k in model_keys]

x2 = np.arange(3)
bars_r = ax_out.bar(
    x2, recovered, 0.5, label="Recovered ✓", color=ACCENT3, alpha=0.9, zorder=3
)
bars_u = ax_out.bar(
    x2,
    unchanged,
    0.5,
    bottom=recovered,
    label="Unchanged",
    color=NEUTRAL,
    alpha=0.55,
    zorder=3,
)
deg_bottom = [r + u for r, u in zip(recovered, unchanged)]
bars_d = ax_out.bar(
    x2,
    degraded,
    0.5,
    bottom=deg_bottom,
    label="Degraded ✗",
    color=WARN,
    alpha=0.9,
    zorder=3,
)

for i, (r, n) in enumerate(zip(recovered, rethink_n)):
    ax_out.text(
        i,
        n + 2.5,
        f"{100 * r / n:.0f}%\nrecovery",
        ha="center",
        va="bottom",
        fontsize=8,
        color=ACCENT3,
        fontweight="bold",
    )

ax_out.set_xticks(x2)
ax_out.set_xticklabels([f"Qwen3-{k}" for k in model_keys], fontsize=9.5)
ax_out.legend(
    fontsize=8.5,
    framealpha=0.2,
    facecolor=PANEL,
    edgecolor=BORDER,
    labelcolor=WHITE,
    loc="upper left",
)

plt.savefig(
    "outputs/fig1_summary.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Fig 1 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-Category Recovery Heatmap
# ═════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig2.suptitle(
    "Category-Wise Recovery Rate",
    fontsize=14,
    fontweight="bold",
    color=WHITE,
    y=1.01,
)

CAT_ORDER = [
    "history",
    "chemistry",
    "physics",
    "biology",
    "geography",
    "mathematics",
    "astronomy",
    "literature",
    "CS",
    "general",
    "economics",
    "music",
    "language",
]
CAT_LABELS = [c.capitalize() for c in CAT_ORDER]

for ax, (key, color) in zip(axes, MODEL_COLORS.items()):
    d = models_data[key]
    rt = d["rethink_df"]

    cat_stats = rt.groupby("category")[["recovered", "degraded", "unchanged"]].sum()
    cat_stats["total"] = rt.groupby("category").size()
    cat_stats["recovery_rate"] = (
        cat_stats["recovered"] / cat_stats["total"] * 100
    ).fillna(0)
    cat_stats["p2_wrong"] = (
        cat_stats["total"] - rt.groupby("category")["p2_correct"].sum()
    )

    rates = []
    totals = []
    recov = []
    for c in CAT_ORDER:
        if c in cat_stats.index:
            rates.append(cat_stats.loc[c, "recovery_rate"])
            totals.append(int(cat_stats.loc[c, "total"]))
            recov.append(int(cat_stats.loc[c, "recovered"]))
        else:
            rates.append(0)
            totals.append(0)
            recov.append(0)

    y = np.arange(len(CAT_ORDER))
    bars = ax.barh(y, rates, 0.65, color=color, alpha=0.85, zorder=3)

    for i, (rate, tot, rec) in enumerate(zip(rates, totals, recov)):
        if tot > 0:
            lbl = f"{rec}/{tot}"
            ax.text(
                min(rate + 2, 102),
                i,
                lbl,
                va="center",
                fontsize=7.5,
                color=WHITE,
                alpha=0.85,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(CAT_LABELS, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Recovery Rate (%)", color=NEUTRAL, fontsize=9)
    ax.axvline(50, color=BORDER, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(100, color=color, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_facecolor(PANEL)
    ax.grid(axis="x", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=NEUTRAL)

    rr = d["recovery_rate"]
    ax.set_title(
        f"Qwen3-{key}\n(overall {rr:.1f}% recovery)",
        color=color,
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.invert_yaxis()

fig2.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(
    "outputs/fig2_category_recovery.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Fig 2 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Conflict Score Distribution + Trust Score by Outcome
# ═════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
fig3.suptitle(
    "Conflict Score Distribution by Rethink Outcome",
    fontsize=13,
    fontweight="bold",
    color=WHITE,
    y=1.01,
)

for ax, (key, color) in zip(axes, MODEL_COLORS.items()):
    d = models_data[key]
    rt = d["rethink_df"]

    recovered_scores = rt[rt["recovered"] == True]["conflict_score"].dropna()
    unchanged_scores = rt[rt["unchanged"] == True]["conflict_score"].dropna()
    degraded_scores = rt[rt["degraded"] == True]["conflict_score"].dropna()

    bins = np.linspace(0, 1, 20)
    ax.hist(
        recovered_scores,
        bins=bins,
        alpha=0.75,
        color=ACCENT3,
        label="Recovered",
        zorder=3,
    )
    ax.hist(
        unchanged_scores,
        bins=bins,
        alpha=0.65,
        color=NEUTRAL,
        label="Unchanged",
        zorder=2,
    )
    if len(degraded_scores):
        ax.hist(
            degraded_scores,
            bins=bins,
            alpha=0.85,
            color=WARN,
            label="Degraded",
            zorder=4,
        )

    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=NEUTRAL)
    ax.set_xlabel("Conflict Score", color=NEUTRAL, fontsize=9)
    ax.set_ylabel("Count", color=NEUTRAL, fontsize=9)
    ax.set_title(f"Qwen3-{key}", color=color, fontsize=11, fontweight="bold", pad=8)
    ax.legend(
        fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE
    )

fig3.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(
    "outputs/fig3_conflict_score_dist.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Fig 3 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Scaling insight: model size vs key metrics
# ═════════════════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig4.suptitle(
    "Scaling Behaviour: Model Size vs Key Metrics",
    fontsize=13,
    fontweight="bold",
    color=WHITE,
    y=1.01,
)

sizes = [1.7, 4, 8]
size_labels = ["1.7B", "4B", "8B"]
colors_list = [ACCENT1, ACCENT2, ACCENT3]

p2_accs = [models_data[k]["p2_acc"] for k in ["1.7B", "4B", "8B"]]
rt_accs = [models_data[k]["rt_acc"] for k in ["1.7B", "4B", "8B"]]
deltas = [models_data[k]["delta"] for k in ["1.7B", "4B", "8B"]]
recov_rates = [models_data[k]["recovery_rate"] for k in ["1.7B", "4B", "8B"]]
rethink_pcts = [
    100 * models_data[k]["rethink_n"] / models_data[k]["n"]
    for k in ["1.7B", "4B", "8B"]
]

metric_sets = [
    (axes[0], p2_accs, rt_accs, "Accuracy (%)", "Before vs After Accuracy"),
    (
        axes[1],
        rethink_pcts,
        recov_rates,
        "Percentage (%)",
        "Rethink Triggered vs Recovery Rate",
    ),
    (axes[2], deltas, None, "Accuracy Gain (pp)", "Accuracy Improvement (Δ pp)"),
]

for ax, y1_vals, y2_vals, ylabel, title in metric_sets:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    ax.plot(
        sizes,
        y1_vals,
        "o-",
        color=ACCENT1,
        linewidth=2,
        markersize=8,
        label="Before" if y2_vals is not None else "Δ Accuracy",
        zorder=3,
    )
    for x, y, lbl in zip(sizes, y1_vals, size_labels):
        ax.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            color=ACCENT1,
            fontweight="bold",
        )

    if y2_vals is not None:
        ax.plot(
            sizes,
            y2_vals,
            "s-",
            color=ACCENT3,
            linewidth=2,
            markersize=8,
            label="After" if title.startswith("Acc") else "Recovery Rate",
            zorder=3,
        )
        for x, y, lbl in zip(sizes, y2_vals, size_labels):
            ax.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                fontsize=8.5,
                color=ACCENT3,
                fontweight="bold",
            )
        ax.legend(
            fontsize=8.5,
            framealpha=0.2,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=WHITE,
        )

    ax.set_xticks(sizes)
    ax.set_xticklabels(size_labels, fontsize=9.5, color=WHITE)
    ax.set_xlabel("Model Size", color=NEUTRAL, fontsize=9)
    ax.set_ylabel(ylabel, color=NEUTRAL, fontsize=9)
    ax.set_title(title, color=WHITE, fontsize=10.5, fontweight="bold", pad=8)
    ax.tick_params(colors=NEUTRAL)

fig4.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(
    "outputs/fig4_scaling.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Fig 4 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Bias verdict before / after rethink (Sankey-style flows)
# ═════════════════════════════════════════════════════════════════════════════

TOOL_COL = "#F87171"  # red  – tool biased
KNOW_COL = "#6C8EFF"  # blue – knowledge biased
UND_COL = "#94A3B8"  # grey – undetermined

VERDICT_COLORS = {
    "TOOL_BIASED": TOOL_COL,
    "KNOWLEDGE_BIASED": KNOW_COL,
    "UNDETERMINED": UND_COL,
}
VERDICT_LABELS = {
    "TOOL_BIASED": "Tool-Biased",
    "KNOWLEDGE_BIASED": "Knowledge-Biased",
    "UNDETERMINED": "Undetermined",
}

fig5, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=BG)
fig5.suptitle(
    "Bias Verdict: Before vs After Rethink Prompt",
    fontsize=14,
    fontweight="bold",
    color=WHITE,
    y=1.01,
)

ORDER = ["TOOL_BIASED", "KNOWLEDGE_BIASED", "UNDETERMINED"]

for ax, (key, mcolor) in zip(axes, MODEL_COLORS.items()):
    r = models_data[key]["full_df"]

    # effective after-verdict: NO_RETHINK → inherit verdict_p2
    r = r.copy()
    r["verdict_after"] = r.apply(
        lambda x: (
            x["verdict_p2"] if x["verdict_rt"] == "NO_RETHINK" else x["verdict_rt"]
        ),
        axis=1,
    )

    before_counts = r["verdict_p2"].value_counts()
    after_counts = r["verdict_after"].value_counts()
    n = len(r)

    # ── draw two stacked bars ─────────────────────────────────────────────
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, n * 1.12)
    ax.axis("off")

    BAR_W = 0.9
    BAR_X = {"before": 0.5, "after": 2.8}
    MID_X = (BAR_X["before"] + BAR_X["after"]) / 2

    def draw_bar(col_x, counts, alpha=1.0):
        """Draw a stacked bar; return {verdict: (y_center, height)}."""
        segments = {}
        y = 0
        for v in ORDER:
            h = counts.get(v, 0)
            if h == 0:
                segments[v] = (y, 0)
                continue
            rect = FancyBboxPatch(
                (col_x - BAR_W / 2, y),
                BAR_W,
                h,
                boxstyle="round,pad=0.5",
                facecolor=VERDICT_COLORS[v],
                alpha=0.88,
                edgecolor=BG,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(rect)
            if h > 6:
                ax.text(
                    col_x,
                    y + h / 2,
                    f"{VERDICT_LABELS[v]}\n{h}  ({100 * h / n:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=WHITE,
                    fontweight="bold",
                    zorder=4,
                )
            segments[v] = (y + h / 2, h)
            y += h
        return segments

    seg_before = draw_bar(BAR_X["before"], before_counts)
    seg_after = draw_bar(BAR_X["after"], after_counts)

    # # ── flow ribbons (bezier) between bars ───────────────────────────────
    # crosstab = pd.crosstab(r["verdict_p2"], r["verdict_after"])

    # def ribbon(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.22):
    #     """Filled bezier ribbon between two vertical spans."""
    #     import matplotlib.patches as mpatches
    #     from matplotlib.path import Path

    #     cp = MID_X
    #     verts = [
    #         (x0, y0_bot),
    #         (cp, y0_bot),
    #         (cp, y1_bot),
    #         (x1, y1_bot),
    #         (x1, y1_top),
    #         (cp, y1_top),
    #         (cp, y0_top),
    #         (x0, y0_top),
    #         (x0, y0_bot),
    #     ]
    #     codes = [
    #         Path.MOVETO,
    #         Path.CURVE4,
    #         Path.CURVE4,
    #         Path.CURVE4,
    #         Path.LINETO,
    #         Path.CURVE4,
    #         Path.CURVE4,
    #         Path.CURVE4,
    #         Path.CLOSEPOLY,
    #     ]
    #     path = Path(verts, codes)
    #     patch = mpatches.PathPatch(
    #         path, facecolor=color, alpha=alpha, edgecolor="none", zorder=2
    #     )
    #     ax.add_patch(patch)

    # # track running offsets for flow splitting
    # before_y_offset = {v: (seg_before[v][0] - seg_before[v][1] / 2) for v in ORDER}
    # after_y_offset = {v: (seg_after[v][0] - seg_after[v][1] / 2) for v in ORDER}

    # for v_from in ORDER:
    #     for v_to in ORDER:
    #         flow = (
    #             crosstab.loc[v_from, v_to]
    #             if (v_from in crosstab.index and v_to in crosstab.columns)
    #             else 0
    #         )
    #         if flow == 0:
    #             continue
    #         y0_bot = before_y_offset[v_from]
    #         y0_top = y0_bot + flow
    #         before_y_offset[v_from] = y0_top

    #         y1_bot = after_y_offset[v_to]
    #         y1_top = y1_bot + flow
    #         after_y_offset[v_to] = y1_top

    #         col = VERDICT_COLORS[v_from]
    #         ribbon(
    #             ax,
    #             BAR_X["before"] + BAR_W / 2,
    #             y0_bot,
    #             y0_top,
    #             BAR_X["after"] - BAR_W / 2,
    #             y1_bot,
    #             y1_top,
    #             col,
    #             alpha=0.28,
    #         )

    # ── labels above bars ────────────────────────────────────────────────
    for label, col_x in [
        ("Before\nRethink", BAR_X["before"]),
        ("After\nRethink", BAR_X["after"]),
    ]:
        ax.text(
            col_x,
            n * 1.055,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=NEUTRAL,
            fontweight="bold",
        )

    ax.set_title(f"Qwen3-{key}", color=mcolor, fontsize=12, fontweight="bold", pad=10)

# ── shared legend ─────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=VERDICT_COLORS[v], label=VERDICT_LABELS[v], edgecolor=BG)
    for v in ORDER
]
fig5.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    fontsize=9.5,
    framealpha=0.15,
    facecolor=PANEL,
    edgecolor=BORDER,
    labelcolor=WHITE,
    bbox_to_anchor=(0.5, -0.04),
)

fig5.tight_layout(rect=[0, 0.04, 1, 0.99])
plt.savefig(
    "outputs/fig5_bias_flow.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Fig 5 saved.")

print("\nAll figures saved to outputs/")
