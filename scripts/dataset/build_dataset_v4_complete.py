import pandas as pd
import os
import re

RAW = "./dataset/raw"
PROCESSED = "./dataset/processed"
DESIGN = "./design"

# -----------------------------
# LOAD DESIGN MATRICES
# -----------------------------
design_existing = pd.read_csv(f"{DESIGN}/design_matrix_existing_models.csv")

design_v2_path = f"{DESIGN}/design_matrix_v2_new_simulations.csv"
if os.path.exists(design_v2_path):
    design_new = pd.read_csv(design_v2_path)
else:
    length_tmp = pd.read_csv(f"{PROCESSED}/length_study_enriched.csv")
    design_new = pd.DataFrame({
        "model_id": length_tmp["model_id"],
        "phase": "length_study",
        "law_type": length_tmp["law_type"].replace({
            "power_n2": "power_law",
            "power_n3": "power_law"
        }),
        "n_exponent": length_tmp["model_id"].map(
            lambda x: 2 if x in ["M21", "M25"] else (3 if x in ["M22", "M26"] else None)
        ),
        "number_of_layers": 16,
        "enthesis_length_mm": length_tmp["enthesis_length_mm"],
        "poisson_mode": "constant",
        "tendon_E_MPa": 200,
        "bone_E_MPa": 20000,
        "tendon_nu": 0.45,
        "bone_nu": 0.30,
        "notes": "Length study model"
    })

design_df = pd.concat([design_existing, design_new], ignore_index=True)
design_df["model_id"] = design_df["model_id"].astype(str)

# -----------------------------
# PHASE 1: M01-M05
# -----------------------------
phase1 = pd.read_csv(f"{RAW}/summary_all_metrics_phase1.csv")

phase1_map = {
    "Sharp": "M01",
    "Linear": "M02",
    "Exponential": "M03",
    "Power_n05": "M04",
    "Power_n2": "M05",
    "Power n=0.5": "M04",
    "Power n=2": "M05"
}

phase1["model_id"] = phase1["Model"].map(phase1_map)

phase1_metrics = phase1.rename(columns={
    "Max_S11_MPa": "max_S11",
    "Min_S11_MPa": "min_S11"
})[[
    "model_id",
    "max_S11",
    "min_S11",
    "S11_Interface_MPa",
    "S11_Range_MPa",
    "MaxAbs_dSdx",
    "MeanAbs_dSdx",
    "MeanAbs_d2Sdx2",
    "MaxAbs_d2Sdx2",
    "Area_vs_Sharp",
    "AreaAbs_S11"
]]

# -----------------------------
# PHASE A/B/C: extract Mxx from model_name
# -----------------------------
def extract_model_id(name):
    match = re.search(r"M\d+", str(name))
    return match.group(0) if match else None

def load_phase_summary(filename):
    df = pd.read_csv(f"{RAW}/{filename}")
    df["model_id"] = df["model_name"].apply(extract_model_id)

    rename_dict = {
        "max_s11": "max_S11",
        "min_s11": "min_S11",
        "max_mises": "max_vonMises_existing"
    }

    df = df.rename(columns=rename_dict)

    keep_cols = ["model_id"]

    for col in [
        "max_S11",
        "min_S11",
        "max_vonMises_existing",
        "max_mises_x",
        "max_mises_y",
        "max_s11_x",
        "max_s11_y",
        "min_s11_x",
        "min_s11_y"
    ]:
        if col in df.columns:
            keep_cols.append(col)

    return df[keep_cols]

phaseA = load_phase_summary("summary_phaseA.csv")
phaseB = load_phase_summary("summary_phaseB.csv")
phaseC = load_phase_summary("summary_phaseC.csv")

original_metrics = pd.concat(
    [phase1_metrics, phaseA, phaseB, phaseC],
    ignore_index=True
)

# -----------------------------
# MERGE OLD VON MISES FROM ODB EXTRACTION
# -----------------------------
mises_old = pd.read_csv(f"{RAW}/summary_original_project_mises.csv")
mises_old["model_id"] = mises_old["model_id"].astype(str)

mises_old = mises_old.rename(columns={
    "Max_Mises_MPa": "max_vonMises",
    "Max_S11_MPa_from_odb": "max_S11_from_odb",
    "Min_S11_MPa_from_odb": "min_S11_from_odb"
})

original_metrics["model_id"] = original_metrics["model_id"].astype(str)

original_metrics = original_metrics.merge(
    mises_old[[
        "model_id",
        "odb_file",
        "max_vonMises",
        "max_S11_from_odb",
        "min_S11_from_odb"
    ]],
    on="model_id",
    how="left"
)

original_metrics["source"] = "original_project"

# -----------------------------
# LENGTH STUDY M19-M26
# -----------------------------
length_df = pd.read_csv(f"{PROCESSED}/length_study_enriched.csv")

length_metrics = length_df.rename(columns={
    "Max_S11_MPa": "max_S11",
    "Min_S11_MPa": "min_S11",
    "Max_Mises_MPa": "max_vonMises"
})

length_metrics["source"] = "length_study"

# -----------------------------
# COMBINE METRICS
# -----------------------------
all_metric_cols = sorted(set(original_metrics.columns).union(set(length_metrics.columns)))

original_metrics = original_metrics.reindex(columns=all_metric_cols)
length_metrics = length_metrics.reindex(columns=all_metric_cols)

metrics_df = pd.concat([original_metrics, length_metrics], ignore_index=True)
metrics_df = metrics_df.drop_duplicates(subset=["model_id"], keep="last")

# -----------------------------
# FINAL MERGE WITH DESIGN MATRIX
# -----------------------------
full_df = design_df.merge(
    metrics_df,
    on="model_id",
    how="left"
)

# -----------------------------
# SAVE
# -----------------------------
output_path = f"{PROCESSED}/enthesis_dataset_v4_complete.csv"
os.makedirs(PROCESSED, exist_ok=True)

full_df.to_csv(output_path, index=False)

print("\nDataset v4 complete created:")
print(output_path)

print("\nShape:")
print(full_df.shape)

print("\nModels included:")
print(sorted(full_df["model_id"].dropna().unique()))

print("\nColumns:")
print(full_df.columns.tolist())

print("\nMissing max_vonMises count:")
print(full_df["max_vonMises"].isna().sum())

print("\nMissing max_S11 count:")
print(full_df["max_S11"].isna().sum())