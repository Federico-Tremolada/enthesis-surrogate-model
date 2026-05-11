import pandas as pd
import os

# --------------------------------------------------
# LOAD OPTIMIZATION RESULTS
# --------------------------------------------------

opt_df = pd.read_csv(
    "results/optimization/optimization_results_v1.csv"
)

# --------------------------------------------------
# LOAD FEM VALIDATION RESULTS
# --------------------------------------------------

val_df = pd.read_csv(
    "dataset/raw/summary_validation_models.csv"
)

# --------------------------------------------------
# SELECT VALIDATION MODELS
# --------------------------------------------------

validation_models = ["M27", "M28"]

# --------------------------------------------------
# EXTRACT PREDICTIONS
# --------------------------------------------------

pred_rows = []

for model in validation_models:

    if model == "M27":

        pred = opt_df[
            (opt_df["law_type_clean"] == "exponential")
            &
            (opt_df["enthesis_length_mm_clean"] == 8)
            &
            (opt_df["number_of_layers"] == 16)
        ].iloc[0]

    elif model == "M28":

        pred = opt_df[
            (opt_df["law_type_clean"] == "power_law")
            &
            (opt_df["enthesis_length_mm_clean"] == 8)
            &
            (opt_df["number_of_layers"] == 16)
            &
            (opt_df["n_exponent"] == 5)
        ].iloc[0]

    pred_rows.append({
        "model_id": model,
        "predicted_max_vonMises":
            pred["predicted_max_vonMises"]
    })

pred_df = pd.DataFrame(pred_rows)

# --------------------------------------------------
# CLEAN FEM RESULTS
# --------------------------------------------------

val_df = val_df.rename(columns={
    "Max_Mises_MPa": "fem_max_vonMises",
    "Max_S11_MPa": "fem_max_S11",
    "Min_S11_MPa": "fem_min_S11"
})

# --------------------------------------------------
# MERGE
# --------------------------------------------------

final_df = pred_df.merge(
    val_df,
    on="model_id",
    how="left"
)

# --------------------------------------------------
# ERRORS
# --------------------------------------------------

final_df["absolute_error"] = (
    final_df["predicted_max_vonMises"]
    - final_df["fem_max_vonMises"]
).abs()

final_df["percent_error"] = (
    final_df["absolute_error"]
    / final_df["fem_max_vonMises"]
) * 100

# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs("results/validation", exist_ok=True)

final_df.to_csv(
    "results/validation/final_validation_table.csv",
    index=False
)

# --------------------------------------------------
# PRINT
# --------------------------------------------------

print("\nFINAL VALIDATION TABLE")
print(final_df)

print("\nSaved:")
print("results/validation/final_validation_table.csv")