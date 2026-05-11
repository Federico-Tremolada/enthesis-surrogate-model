import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# LOAD CLEAN DATASET
# -----------------------------
df = pd.read_csv("dataset/processed/enthesis_dataset_v4_clean.csv")

target = "max_vonMises_final"

df = df.dropna(subset=[target])

# -----------------------------
# FEATURES
# -----------------------------
numeric_features = [
    "enthesis_length_mm",
    "n_exponent",
    "number_of_layers"
]

X_num = df[numeric_features]

X_cat = pd.get_dummies(
    df["law_type"],
    prefix="law"
)

X = pd.concat([X_num, X_cat], axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

y = df[target].astype(float)

# -----------------------------
# TRAIN FINAL SURROGATE
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

training_columns = X.columns.tolist()

# -----------------------------
# OUTPUT FOLDER
# -----------------------------
output_dir = "results/design_maps"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# DESIGN MAP 1
# n exponent vs enthesis length
# power-law only
# -----------------------------
n_values = np.linspace(1.5, 5.0, 50)
length_values = np.linspace(4.0, 8.0, 50)

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

Xc_num = candidates[numeric_features]

Xc_cat = pd.get_dummies(
    candidates["law_type"],
    prefix="law"
)

Xc = pd.concat([Xc_num, Xc_cat], axis=1)
Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(0)
Xc = Xc.reindex(columns=training_columns, fill_value=0)

pred = model.predict(Xc)
Z = pred.reshape(N.shape)

plt.figure(figsize=(7, 5))
contour = plt.contourf(N, L, Z, levels=20)
plt.colorbar(contour, label="Predicted max von Mises [MPa]")
plt.xlabel("Power-law exponent n")
plt.ylabel("Enthesis length [mm]")
plt.title("Design map: n exponent vs enthesis length")
plt.tight_layout()
plt.savefig(
    f"{output_dir}/design_map_n_vs_length_powerlaw.png",
    dpi=300
)
plt.close()

# -----------------------------
# DESIGN MAP 2
# layers vs n exponent
# power-law, fixed length = 8 mm
# -----------------------------
n_values = np.linspace(1.5, 5.0, 50)
layer_values = np.linspace(8, 16, 50)

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

Xc_num = candidates[numeric_features]

Xc_cat = pd.get_dummies(
    candidates["law_type"],
    prefix="law"
)

Xc = pd.concat([Xc_num, Xc_cat], axis=1)
Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(0)
Xc = Xc.reindex(columns=training_columns, fill_value=0)

pred = model.predict(Xc)
Z = pred.reshape(N.shape)

plt.figure(figsize=(7, 5))
contour = plt.contourf(N, Layers, Z, levels=20)
plt.colorbar(contour, label="Predicted max von Mises [MPa]")
plt.xlabel("Power-law exponent n")
plt.ylabel("Number of layers")
plt.title("Design map: n exponent vs number of layers")
plt.tight_layout()
plt.savefig(
    f"{output_dir}/design_map_n_vs_layers_powerlaw.png",
    dpi=300
)
plt.close()

# -----------------------------
# DESIGN MAP 3
# law type comparison vs length
# -----------------------------
length_values = np.linspace(4, 8, 50)
law_types = ["linear", "exponential", "power_law"]

rows = []

for law in law_types:
    for L in length_values:
        rows.append({
            "law_type": law,
            "enthesis_length_mm": L,
            "n_exponent": 3 if law == "power_law" else 0,
            "number_of_layers": 16
        })

law_df = pd.DataFrame(rows)

Xc_num = law_df[numeric_features]

Xc_cat = pd.get_dummies(
    law_df["law_type"],
    prefix="law"
)

Xc = pd.concat([Xc_num, Xc_cat], axis=1)
Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(0)
Xc = Xc.reindex(columns=training_columns, fill_value=0)

law_df["predicted_max_vonMises"] = model.predict(Xc)

plt.figure(figsize=(7, 5))

for law in law_types:
    subset = law_df[law_df["law_type"] == law]
    plt.plot(
        subset["enthesis_length_mm"],
        subset["predicted_max_vonMises"],
        marker="o",
        markersize=3,
        label=law
    )

plt.xlabel("Enthesis length [mm]")
plt.ylabel("Predicted max von Mises [MPa]")
plt.title("Predicted stress vs enthesis length")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{output_dir}/design_map_law_comparison_vs_length.png",
    dpi=300
)
plt.close()

# -----------------------------
# SAVE DESIGN MAP DATA
# -----------------------------
law_df.to_csv(
    f"{output_dir}/law_comparison_vs_length_data.csv",
    index=False
)

print("\nDesign maps generated successfully.")
print(f"Saved in: {output_dir}")