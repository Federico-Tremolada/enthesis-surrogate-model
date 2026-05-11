import pandas as pd
import numpy as np
import os

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv(
    "dataset/processed/enthesis_dataset_v4_clean.csv"
)

# -----------------------------
# FEATURES
# -----------------------------
target = "max_vonMises_final"

df = df.dropna(subset=[target])

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
# MODELS
# -----------------------------
models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        random_state=42
    )
}

# -----------------------------
# CROSS VALIDATION
# -----------------------------
loo = LeaveOneOut()

results_summary = []

os.makedirs("results/ml_final", exist_ok=True)

for model_name, model in models.items():

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        y_true.append(y_test.values[0])
        y_pred.append(pred[0])

    # metrics
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    results_summary.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # detailed predictions
    pred_df = pd.DataFrame({
        "model_id": df["model_id"],
        "actual": y_true,
        "predicted": y_pred
    })

    pred_df["error"] = (
        pred_df["predicted"]
        - pred_df["actual"]
    )

    pred_df["abs_error"] = pred_df["error"].abs()

    pred_df.to_csv(
        f"results/ml_final/predictions_{model_name}.csv",
        index=False
    )

# -----------------------------
# SAVE SUMMARY
# -----------------------------
summary_df = pd.DataFrame(results_summary)

summary_df = summary_df.sort_values(
    by="MAE"
)

summary_df.to_csv(
    "results/ml_final/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(summary_df)

print("\nSaved:")
print("results/ml_final/model_comparison.csv")