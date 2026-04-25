import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#ffffff"
PANEL = "#fafafa"
BORDER = "#2E3347"
ACCENT1 = "#ef4f3f"  # before / 1.7B
ACCENT2 = "#2aaf62"  # after  / 4B
ACCENT3 = "#2aaf62"  # recovered / 8B
WARN = "#F87171"  # degraded
NEUTRAL = "#656565"  # text secondary
WHITE = "#000000"

MODEL_COLORS = {"1.5B": ACCENT1, "7B": ACCENT2}
CAT_COLORS = [ACCENT1, ACCENT2]

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
    "1.5B": ("qwen2.5-1.5b_summary.csv", "qwen2.5-1.5b_rethink_summary.csv"),
    "7B": ("qwen2.5-7b_summary.csv", "qwen2.5-7b_rethink_summary.csv"),
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


models_data = {k: load(k) for k in ["1.5B", "7B"]}


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
# FIGURE 1b — Before/After Accuracy Bars Only
# ═════════════════════════════════════════════════════════════════════════════
fig1, ax_acc = plt.subplots(figsize=(9, 6), facecolor=BG)

# ── 1b. Before / After accuracy bars ─────────────────────────────────────────
styled_ax(ax_acc, title="Accuracy Before vs After Rethink", ylabel="Accuracy (%)")

model_keys = ["1.5B", "7B"]
x = np.arange(2)
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
ax_acc.set_xticklabels([f"Qwen2.5-{k}" for k in model_keys], fontsize=9.5)
ax_acc.set_ylim(0, 105)
ax_acc.legend(
    fontsize=8.5, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE
)

plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.90])

plt.savefig(
    "outputs/poster_qwen2.5.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
)
plt.close()
print("Fig 1 saved.")
