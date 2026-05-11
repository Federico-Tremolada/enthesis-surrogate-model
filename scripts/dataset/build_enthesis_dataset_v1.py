import pandas as pd
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------

base_path = r"."

design_path = os.path.join(
    base_path,
    "design",
    "design_matrix_existing_models.csv"
)

raw_path = os.path.join(
    base_path,
    "dataset",
    "raw"
)

output_path = os.path.join(
    base_path,
    "dataset",
    "processed",
    "enthesis_dataset_v1.csv"
)

# --------------------------------------------------
# LOAD DESIGN MATRIX
# --------------------------------------------------

design_df = pd.read_csv(design_path)

print("\nDesign matrix loaded:")
print(design_df[["model_id", "law_type", "n_exponent"]].head())
print(f"Total design models: {len(design_df)}")

# --------------------------------------------------
# LOAD SUMMARY FILES
# --------------------------------------------------

summary_files = [
    "summary_all_metrics_phase1.csv",
    "summary_phaseA.csv",
    "summary_phaseB.csv",
    "summary_phaseC.csv"
]

all_summaries = []

for file in summary_files:

    file_path = os.path.join(raw_path, file)

    print(f"\nLoading: {file}")

    df = pd.read_csv(file_path)

    print("Columns found:")
    print(df.columns.tolist())

    print(f"Rows: {len(df)}")

    # -----------------------------
    # Standardize model column
    # -----------------------------
    possible_model_cols = [
        "model_id",
        "Model",
        "model",
        "Model_ID",
        "Case"
    ]

    model_col_found = None

    for col in possible_model_cols:
        if col in df.columns:
            model_col_found = col
            break

    if model_col_found is None:
        print(f"WARNING: No model column found in {file}")
        continue

    df = df.rename(columns={model_col_found: "model_id"})

    all_summaries.append(df)

# --------------------------------------------------
# CONCATENATE ALL STRESS DATA
# --------------------------------------------------

stress_df = pd.concat(
    all_summaries,
    ignore_index=True
)

print("\nCombined stress dataset:")
print(stress_df.head())
print(f"Total stress rows: {len(stress_df)}")

# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

stress_df = stress_df.drop_duplicates(
    subset=["model_id"]
)

print(f"Rows after duplicate removal: {len(stress_df)}")

# --------------------------------------------------
# MERGE WITH DESIGN MATRIX
# --------------------------------------------------

final_df = pd.merge(
    design_df,
    stress_df,
    on="model_id",
    how="left"
)

print("\nMerged dataset preview:")
print(final_df.head())

print(f"\nFinal dataset rows: {len(final_df)}")
print(f"Final dataset columns: {len(final_df.columns)}")

# --------------------------------------------------
# SAVE
# --------------------------------------------------

final_df.to_csv(output_path, index=False)

print("\nDataset saved successfully:")
print(output_path)