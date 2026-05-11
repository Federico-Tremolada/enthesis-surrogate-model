import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("dataset/processed/enthesis_dataset_v4_complete.csv")

# -----------------------------
# CLEAN MERGED COLUMNS
# -----------------------------

# law_type
if "law_type_x" in df.columns:
    df["law_type_clean"] = df["law_type_x"]
elif "law_type_y" in df.columns:
    df["law_type_clean"] = df["law_type_y"]
elif "law_type" in df.columns:
    df["law_type_clean"] = df["law_type"]
else:
    raise ValueError("No law type column found.")

# enthesis length
if "enthesis_length_mm_x" in df.columns and "enthesis_length_mm_y" in df.columns:
    df["enthesis_length_mm_clean"] = df["enthesis_length_mm_x"].combine_first(df["enthesis_length_mm_y"])
elif "enthesis_length_mm_x" in df.columns:
    df["enthesis_length_mm_clean"] = df["enthesis_length_mm_x"]
elif "enthesis_length_mm_y" in df.columns:
    df["enthesis_length_mm_clean"] = df["enthesis_length_mm_y"]
elif "enthesis_length_mm" in df.columns:
    df["enthesis_length_mm_clean"] = df["enthesis_length_mm"]
else:
    df["enthesis_length_mm_clean"] = 6

# fill missing
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

# Force everything numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

y = df[target].astype(float)
model_ids = df["model_id"]

print("\nModels used:")
print(model_ids.tolist())

print("\nFeatures used:")
print(X.columns.tolist())

print("\nDataset shape:")
print(X.shape)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# -----------------------------
# LEAVE-ONE-OUT CV
# -----------------------------
loo = LeaveOneOut()

y_true = []
y_pred = []
ids = []

for train_idx, test_idx in loo.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    y_true.append(y_test.values[0])
    y_pred.append(pred[0])
    ids.append(model_ids.iloc[test_idx].values[0])

# -----------------------------
# METRICS
# -----------------------------
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nPerformance:")
print(f"MAE: {mae:.3f} MPa")
print(f"R2: {r2:.3f}")

# -----------------------------
# PREDICTION TABLE
# -----------------------------
results = pd.DataFrame({
    "model_id": ids,
    "actual": y_true,
    "predicted": y_pred
})

results["error"] = results["predicted"] - results["actual"]
results["abs_error"] = results["error"].abs()

print("\nWorst predictions:")
print(results.sort_values("abs_error", ascending=False).head(10))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
model.fit(X, y)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance)

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("results/ml", exist_ok=True)

results.to_csv("results/ml/predictions_v4.csv", index=False)
importance.to_csv("results/ml/feature_importance_v4.csv", index=False)

print("\nSaved:")
print("results/ml/predictions_v4.csv")
print("results/ml/feature_importance_v4.csv")