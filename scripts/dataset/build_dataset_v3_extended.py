import pandas as pd
import os

# -----------------------------
# PATHS
# -----------------------------
base_dataset_path = "./dataset/processed/enthesis_dataset_v2_clean.csv"
length_dataset_path = "./dataset/processed/length_study_enriched.csv"
output_path = "./dataset/processed/enthesis_dataset_v3_extended.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
base_df = pd.read_csv(base_dataset_path)
length_df = pd.read_csv(length_dataset_path)

print("\nBase dataset columns:")
print(base_df.columns.tolist())

print("\nLength study columns:")
print(length_df.columns.tolist())

# -----------------------------
# PREPARE LENGTH STUDY DATASET
# -----------------------------

# Convert law names to match main dataset
law_map = {
    "linear": "linear",
    "exponential": "exponential",
    "power_n2": "power_law",
    "power_n3": "power_law"
}

n_map = {
    "linear": None,
    "exponential": None,
    "power_n2": 2,
    "power_n3": 3
}

length_df["law_type"] = length_df["law_type"].map(law_map)
length_df["n_exponent"] = length_df["model_id"].map(
    lambda x: 2 if x in ["M21", "M25"] else (3 if x in ["M22", "M26"] else None)
)

length_df["phase"] = "length_study"
length_df["number_of_layers"] = 16
length_df["poisson_mode"] = "constant"
length_df["tendon_E_MPa"] = 200
length_df["bone_E_MPa"] = 20000
length_df["tendon_nu"] = 0.45
length_df["bone_nu"] = 0.30

# Rename output columns to match convention
length_df = length_df.rename(columns={
    "Max_S11_MPa": "max_S11",
    "Min_S11_MPa": "min_S11",
    "Max_Mises_MPa": "max_vonMises"
})

# -----------------------------
# ALIGN COLUMNS
# -----------------------------
all_columns = sorted(set(base_df.columns).union(set(length_df.columns)))

base_aligned = base_df.reindex(columns=all_columns)
length_aligned = length_df.reindex(columns=all_columns)

extended_df = pd.concat(
    [base_aligned, length_aligned],
    ignore_index=True
)

# -----------------------------
# CLEAN DUPLICATES
# -----------------------------
extended_df = extended_df.drop_duplicates(subset=["model_id"], keep="last")

# -----------------------------
# SAVE
# -----------------------------
extended_df.to_csv(output_path, index=False)

print("\nExtended dataset created:")
print(output_path)

print("\nShape:")
print(extended_df.shape)

print("\nModels included:")
print(extended_df["model_id"].tolist())