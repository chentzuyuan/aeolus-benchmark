"""
fig_benchmark_4panel.py
4-panel horizontal bar chart: GBDT vs DL (probability AUC)
2 rows (ARR / DEP)  x  2 columns (GBDT / DL reimplemented)
Output: fig_aeolus_4panel.png  (saved to thesis figures folder)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, pathlib

OUT = pathlib.Path(
    "/Users/buttegg/Desktop/論文/"
    "LaTeX_template2024_framing_edit_corrected/figures/fig_aeolus_4panel.png"
)

# ── data ────────────────────────────────────────────────────────────────────
DATA = {
    "ARR": {
        "GBDT": {
            "XGBoost (Optuna)":  0.6884,
            "Random Forest":     0.6867,
            "CatBoost":          0.6829,
            "XGBoost (default)": 0.6817,
        },
        "DL": {
            "AutoInt": 0.6797,
            "MLP":     0.6720,
            "ResNet":  0.6718,
        },
    },
    "DEP": {
        "GBDT": {
            "XGBoost (Optuna)":  0.6973,
            "Random Forest":     0.6954,
            "XGBoost (default)": 0.6936,
            "CatBoost":          0.6920,
        },
        "DL": {
            "AutoInt": 0.6864,
            "MLP":     0.6821,
            "ResNet":  0.6735,
        },
    },
}

# Aeolus paper best DL (binary AUC) — shown as dashed reference
PAPER_BEST = {"ARR": 0.623, "DEP": 0.627}

# ── colours & layout ────────────────────────────────────────────────────────
GREEN      = "#2ca02c"
BLUE       = "#1f77b4"
REF_COLOR  = "#d62728"
XLIM       = (0.545, 0.710)
PANEL_LABEL = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

for row_i, task in enumerate(["ARR", "DEP"]):
    for col_i, family in enumerate(["GBDT", "DL"]):
        ax     = axes[row_i][col_i]
        models = DATA[task][family]

        # sort descending (top bar = best)
        sorted_items = sorted(models.items(), key=lambda x: x[1])
        names = [k for k, _ in sorted_items]
        aucs  = [v for _, v in sorted_items]
        color = GREEN if family == "GBDT" else BLUE
        n     = len(names)
        y_pos = list(range(n))

        bars = ax.barh(y_pos, aucs, color=color, alpha=0.82, height=0.55)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10.5)
        ax.set_xlim(XLIM)
        ax.set_xlabel("AUC-ROC (probability score)", fontsize=9)
        ax.grid(axis="x", alpha=0.3, linewidth=0.7)

        # value labels
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_width() + 0.0004,
                    bar.get_y() + bar.get_height() / 2,
                    f"{auc:.4f}", va="center", ha="left",
                    fontweight="bold", fontsize=9.5)

        # dashed reference line (Aeolus paper best DL binary AUC)
        ref = PAPER_BEST[task]
        ax.axvline(ref, color=REF_COLOR, linestyle="--",
                   linewidth=1.4, alpha=0.75,
                   label=f"Aeolus paper best DL (binary AUC ≈ {ref:.3f})")

        family_label = "Tree-based (GBDT)" if family == "GBDT" else "Deep Learning (reimplemented)"
        idx   = row_i * 2 + col_i
        title = f"{PANEL_LABEL[idx]}  {task}_Delay — {family_label}"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

        if row_i == 0 and col_i == 1:          # per-panel legend: ARR DL panel only
            ax.legend(fontsize=8, loc="lower right",
                      framealpha=0.85, edgecolor="gray")

# ── super title & legend patches ─────────────────────────────────────────────
fig.suptitle(
    "Aeolus Benchmark: GBDT vs. Reimplemented DL (Probability AUC)\n"
    "287,845 flights · 22 features · 6:2:2 chronological split",
    fontsize=13, fontweight="bold", y=1.01,
)

green_patch = mpatches.Patch(color=GREEN, alpha=0.82, label="Tree-based (GBDT)")
blue_patch  = mpatches.Patch(color=BLUE,  alpha=0.82, label="DL reimplemented (prob. AUC)")
red_line    = plt.Line2D([0], [0], color=REF_COLOR, linestyle="--",
                         linewidth=1.4, alpha=0.75, label="Aeolus paper best DL (binary AUC ≈ 0.623–0.627)")
fig.legend(handles=[green_patch, blue_patch, red_line],
           loc="lower center", ncol=3, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

# ── save ────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
