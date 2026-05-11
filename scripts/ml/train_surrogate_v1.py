import pandas as pd
import numpy as np
import os

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("./dataset/processed/enthesis_dataset_v3_extended.csv")

target = "max_vonMises"

features = [
    "law_type",
    "n_exponent",
    "enthesis_length_mm",
    "number_of_layers"
]

# Keep only rows with available target
df = df.dropna(subset=[target])

# Fill missing feature values
df["n_exponent"] = df["n_exponent"].fillna(0)
df["enthesis_length_mm"] = df["enthesis_length_mm"].fillna(6)

# -----------------------------
# ENCODE CATEGORICAL
# -----------------------------
encoder = OneHotEncoder(sparse_output=False)
law_encoded = encoder.fit_transform(df[["law_type"]])
law_cols = encoder.get_feature_names_out(["law_type"])

X_num = df[["n_exponent", "enthesis_length_mm", "number_of_layers"]]
X = np.hstack([X_num.values, law_encoded])

y = df[target].values

feature_names = list(X_num.columns) + list(law_cols)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# -----------------------------
# LEAVE-ONE-OUT VALIDATION
# -----------------------------
loo = LeaveOneOut()
y_pred = cross_val_predict(model, X, y, cv=loo)

mae = mean_absolute_error(y, y_pred)

print("\nLeave-One-Out performance:")
print(f"MAE: {mae:.3f} MPa")

comparison_df = pd.DataFrame({
    "model_id": df["model_id"].values,
    "actual_max_vonMises": y,
    "predicted_max_vonMises": y_pred,
    "error_MPa": y_pred - y,
    "abs_error_MPa": np.abs(y_pred - y)
})

print("\nPrediction comparison:")
print(comparison_df)

# -----------------------------
# FIT FINAL MODEL FOR IMPORTANCE
# -----------------------------
model.fit(X, y)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature importance:")
print(importance_df)

# -----------------------------
# SAVE RESULTS
# -----------------------------
os.makedirs("./results/ml", exist_ok=True)

comparison_df.to_csv("./results/ml/prediction_comparison_v2.csv", index=False)
importance_df.to_csv("./results/ml/feature_importance_v2.csv", index=False)

print("\nSaved:")
print("./results/ml/prediction_comparison_v2.csv")
print("./results/ml/feature_importance_v2.csv")