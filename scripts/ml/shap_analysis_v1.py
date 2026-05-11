import pandas as pd
import numpy as np
import os

import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("dataset/processed/enthesis_dataset_v4_complete.csv")

# -----------------------------
# CLEAN COLUMNS
# -----------------------------
if "law_type_x" in df.columns:
    df["law_type_clean"] = df["law_type_x"]
elif "law_type_y" in df.columns:
    df["law_type_clean"] = df["law_type_y"]
elif "law_type" in df.columns:
    df["law_type_clean"] = df["law_type"]

if "enthesis_length_mm_x" in df.columns and "enthesis_length_mm_y" in df.columns:
    df["enthesis_length_mm_clean"] = df["enthesis_length_mm_x"].combine_first(df["enthesis_length_mm_y"])
elif "enthesis_length_mm_x" in df.columns:
    df["enthesis_length_mm_clean"] = df["enthesis_length_mm_x"]
elif "enthesis_length_mm_y" in df.columns:
    df["enthesis_length_mm_clean"] = df["enthesis_length_mm_y"]
else:
    df["enthesis_length_mm_clean"] = 6

# fill NaN
df["enthesis_length_mm_clean"] = df["enthesis_length_mm_clean"].fillna(6)
df["number_of_layers"] = df["number_of_layers"].fillna(16)
df["n_exponent"] = df["n_exponent"].fillna(0)

# -----------------------------
# TARGET
# -----------------------------
target = "max_vonMises"

df = df.dropna(subset=[target])

# -----------------------------
# FEATURES
# -----------------------------
numeric_features = [
    "enthesis_length_mm_clean",
    "n_exponent",
    "number_of_layers"
]

X_num = df[numeric_features]

X_cat = pd.get_dummies(
    df["law_type_clean"],
    prefix="law_type"
)

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

# -----------------------------
# SHAP EXPLAINER
# -----------------------------
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# -----------------------------
# OUTPUT FOLDER
# -----------------------------
os.makedirs("results/shap", exist_ok=True)

# -----------------------------
# SUMMARY PLOT
# -----------------------------
plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()
plt.savefig(
    "results/shap/shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# BAR PLOT
# -----------------------------
plt.figure()

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig(
    "results/shap/shap_bar_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# LOCAL FORCE PLOT EXAMPLE
# -----------------------------
sample_idx = 0

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X.iloc[sample_idx],
    matplotlib=True,
    show=False
)

plt.savefig(
    "results/shap/shap_force_plot_M01.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP analysis completed.")
print("Results saved in results/shap/")