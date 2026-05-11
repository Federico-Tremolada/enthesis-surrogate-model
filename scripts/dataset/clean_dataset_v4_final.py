import pandas as pd
import os

input_path = "dataset/processed/enthesis_dataset_v4_complete.csv"
output_path = "dataset/processed/enthesis_dataset_v4_clean.csv"

df = pd.read_csv(input_path)

# -----------------------------
# CLEAN law_type
# -----------------------------
if "law_type_x" in df.columns and "law_type_y" in df.columns:
    df["law_type"] = df["law_type_x"].combine_first(df["law_type_y"])
elif "law_type_x" in df.columns:
    df["law_type"] = df["law_type_x"]
elif "law_type_y" in df.columns:
    df["law_type"] = df["law_type_y"]

# -----------------------------
# CLEAN enthesis_length
# -----------------------------
if "enthesis_length_mm_x" in df.columns and "enthesis_length_mm_y" in df.columns:
    df["enthesis_length_mm"] = df["enthesis_length_mm_x"].combine_first(df["enthesis_length_mm_y"])
elif "enthesis_length_mm_x" in df.columns:
    df["enthesis_length_mm"] = df["enthesis_length_mm_x"]
elif "enthesis_length_mm_y" in df.columns:
    df["enthesis_length_mm"] = df["enthesis_length_mm_y"]

# -----------------------------
# CLEAN target columns
# -----------------------------
if "max_S11" in df.columns and "max_S11_from_odb" in df.columns:
    df["max_S11_final"] = df["max_S11"].combine_first(df["max_S11_from_odb"])
elif "max_S11" in df.columns:
    df["max_S11_final"] = df["max_S11"]

if "min_S11" in df.columns and "min_S11_from_odb" in df.columns:
    df["min_S11_final"] = df["min_S11"].combine_first(df["min_S11_from_odb"])
elif "min_S11" in df.columns:
    df["min_S11_final"] = df["min_S11"]

df["max_vonMises_final"] = df["max_vonMises"]

# -----------------------------
# STANDARD FIELDS
# -----------------------------
df["n_exponent"] = df["n_exponent"].fillna(0)
df["enthesis_length_mm"] = df["enthesis_length_mm"].fillna(6)
df["number_of_layers"] = df["number_of_layers"].fillna(16)
df["poisson_mode"] = df["poisson_mode"].fillna("constant")

# -----------------------------
# FINAL COLUMNS
# -----------------------------
final_cols = [
    "model_id",
    "phase",
    "source",
    "law_type",
    "n_exponent",
    "number_of_layers",
    "enthesis_length_mm",
    "poisson_mode",
    "tendon_E_MPa",
    "bone_E_MPa",
    "tendon_nu",
    "bone_nu",
    "max_S11_final",
    "min_S11_final",
    "max_vonMises_final",
    "S11_Interface_MPa",
    "S11_Range_MPa",
    "MaxAbs_dSdx",
    "MeanAbs_dSdx",
    "MaxAbs_d2Sdx2",
    "MeanAbs_d2Sdx2",
    "AreaAbs_S11",
    "Area_vs_Sharp",
    "notes"
]

final_cols = [c for c in final_cols if c in df.columns]

clean_df = df[final_cols].copy()

# -----------------------------
# QUALITY CHECKS
# -----------------------------
print("\nDATASET CLEANED")
print("Shape:", clean_df.shape)

print("\nModels:")
print(clean_df["model_id"].tolist())

print("\nMissing values in key columns:")
key_cols = [
    "law_type",
    "n_exponent",
    "number_of_layers",
    "enthesis_length_mm",
    "max_S11_final",
    "min_S11_final",
    "max_vonMises_final"
]

for col in key_cols:
    if col in clean_df.columns:
        print(col, ":", clean_df[col].isna().sum())

print("\nLaw types:")
print(clean_df["law_type"].value_counts())

print("\nEnthesis lengths:")
print(clean_df["enthesis_length_mm"].value_counts().sort_index())

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("dataset/processed", exist_ok=True)
clean_df.to_csv(output_path, index=False)

print("\nSaved:")
print(output_path)