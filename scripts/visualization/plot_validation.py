import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("./results/ml/prediction_comparison_v2.csv")

x = df["actual_max_vonMises"]
y = df["predicted_max_vonMises"]

# =========================
# FIGURE STYLE
# =========================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16
})

fig, ax = plt.subplots(figsize=(8, 8))

# =========================
# SCATTER
# =========================

ax.scatter(
    x,
    y,
    s=120,
    edgecolor="black",
    linewidth=1.2,
    zorder=3
)

# =========================
# IDEAL LINE
# =========================

lims = [
    min(x.min(), y.min()) - 0.2,
    max(x.max(), y.max()) + 0.2
]

ax.plot(
    lims,
    lims,
    linestyle="--",
    linewidth=2,
    color="black",
    label="Ideal prediction"
)

# =========================
# LABELS
# =========================

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel("FEM max von Mises stress [MPa]")
ax.set_ylabel("ML predicted stress [MPa]")

ax.set_title(
    "Validation of surrogate model predictions",
    fontsize=22,
    weight="semibold"
)

ax.grid(alpha=0.3)

ax.legend(frameon=True)

plt.tight_layout()

# =========================
# SAVE
# =========================

plt.savefig(
    "./figures/final/fig_validation_fem_vs_ml.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()