import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Allow import from scripts/visualization
sys.path.append("scripts/visualization")
from publication_style import set_publication_style, clean_axes, save_figure

set_publication_style()

# -----------------------------
# PATHS
# -----------------------------
dataset_path = "dataset/processed/enthesis_dataset_v4_clean.csv"
output_dir = "figures/publication"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(dataset_path)

target = "max_vonMises_final"
df = df.dropna(subset=[target])

numeric_features = [
    "enthesis_length_mm",
    "n_exponent",
    "number_of_layers"
]

X_num = df[numeric_features]
X_cat = pd.get_dummies(df["law_type"], prefix="law")

X = pd.concat([X_num, X_cat], axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

y = df[target].astype(float)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)
training_columns = X.columns.tolist()

# -----------------------------
# FUNCTION: prepare candidates
# -----------------------------
def prepare_X(candidates):
    Xc_num = candidates[numeric_features]
    Xc_cat = pd.get_dummies(candidates["law_type"], prefix="law")

    Xc = pd.concat([Xc_num, Xc_cat], axis=1)
    Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(0)
    Xc = Xc.reindex(columns=training_columns, fill_value=0)

    return Xc


# ============================================================
# FIGURE 1 — DESIGN MAP: n vs enthesis length
# ============================================================

n_values = np.linspace(1.5, 5.0, 80)
length_values = np.linspace(4.0, 8.0, 80)

N, L = np.meshgrid(n_values, length_values)

candidate_rows = []

for i in range(N.shape[0]):
    for j in range(N.shape[1]):
        candidate_rows.append({
            "enthesis_length_mm": L[i, j],
            "n_exponent": N[i, j],
            "number_of_layers": 16,
            "law_type": "power_law"
        })

candidates = pd.DataFrame(candidate_rows)
Xc = prepare_X(candidates)

Z = model.predict(Xc).reshape(N.shape)

fig, ax = plt.subplots(figsize=(7.5, 5.2))

contour = ax.contourf(
    N,
    L,
    Z,
    levels=25,
    cmap="cividis"
)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("Predicted max von Mises stress [MPa]")

ax.set_xlabel("Power-law exponent, n")
ax.set_ylabel("Enthesis length [mm]")
ax.set_title("Design map: exponent vs enthesis length")

# Optimum marker moved slightly inside the plot to avoid clipping
ax.scatter(
    [4.85],
    [7.85],
    marker="*",
    s=320,
    facecolor="#FFD700",
    edgecolor="black",
    linewidth=1.2,
    label="Predicted optimum",
    zorder=10
)

ax.annotate(
    "Optimum region",
    xy=(4.85, 7.85),
    xytext=(3.75, 7.55),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.2,
        color="black"
    ),
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec="black",
        alpha=0.9
    )
)

ax.legend(loc="lower right")
clean_axes(ax)

save_figure(
    fig,
    f"{output_dir}/fig_design_map_n_vs_length.png"
)

plt.close(fig)


# ============================================================
# FIGURE 2 — DESIGN MAP: n vs number of layers
# ============================================================

n_values = np.linspace(1.5, 5.0, 80)
layer_values = np.linspace(8, 16, 80)

N, Layers = np.meshgrid(n_values, layer_values)

candidate_rows = []

for i in range(N.shape[0]):
    for j in range(N.shape[1]):
        candidate_rows.append({
            "enthesis_length_mm": 8,
            "n_exponent": N[i, j],
            "number_of_layers": Layers[i, j],
            "law_type": "power_law"
        })

candidates = pd.DataFrame(candidate_rows)
Xc = prepare_X(candidates)

Z = model.predict(Xc).reshape(N.shape)

fig, ax = plt.subplots(figsize=(7.5, 5.2))

contour = ax.contourf(
    N,
    Layers,
    Z,
    levels=25,
    cmap="cividis"
)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("Predicted max von Mises stress [MPa]")

ax.set_xlabel("Power-law exponent, n")
ax.set_ylabel("Number of layers")
ax.set_title("Design map: exponent vs layer discretization")

# Optimum marker moved slightly inside the plot to avoid clipping
ax.scatter(
    [4.85],
    [15.75],
    marker="*",
    s=320,
    facecolor="#FFD700",
    edgecolor="black",
    linewidth=1.2,
    label="Predicted optimum",
    zorder=10
)

ax.annotate(
    "Optimum region",
    xy=(4.85, 15.75),
    xytext=(3.65, 14.95),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.2,
        color="black"
    ),
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec="black",
        alpha=0.9
    )
)

ax.legend(loc="lower right")
clean_axes(ax)

save_figure(
    fig,
    f"{output_dir}/fig_design_map_n_vs_layers.png"
)

plt.close(fig)


# ============================================================
# FIGURE 3 — LAW COMPARISON VS LENGTH
# ============================================================

length_values = np.linspace(4.0, 8.0, 100)
law_types = ["linear", "exponential", "power_law"]

law_labels = {
    "linear": "Linear",
    "exponential": "Exponential",
    "power_law": "Power-law, n = 3"
}

rows = []

for law in law_types:
    for Lval in length_values:
        rows.append({
            "law_type": law,
            "enthesis_length_mm": Lval,
            "n_exponent": 3 if law == "power_law" else 0,
            "number_of_layers": 16
        })

law_df = pd.DataFrame(rows)
Xc = prepare_X(law_df)

law_df["predicted_max_vonMises"] = model.predict(Xc)

fig, ax = plt.subplots(figsize=(7.5, 5.2))

for law in law_types:
    subset = law_df[law_df["law_type"] == law]

    ax.plot(
        subset["enthesis_length_mm"],
        subset["predicted_max_vonMises"],
        marker="o",
        markevery=10,
        linewidth=2.6,
        markersize=5.5,
        label=law_labels[law]
    )

ax.set_xlabel("Enthesis length [mm]")
ax.set_ylabel("Predicted max von Mises stress [MPa]")
ax.set_title("Effect of enthesis length on predicted stress")
ax.legend(loc="best")
clean_axes(ax)

save_figure(
    fig,
    f"{output_dir}/fig_law_comparison_vs_length.png"
)

plt.close(fig)


# -----------------------------
# SAVE DATA
# -----------------------------
law_df.to_csv(
    f"{output_dir}/fig_law_comparison_vs_length_data.csv",
    index=False
)

print("\nPublication-style design maps generated.")
print(f"Saved in: {output_dir}")