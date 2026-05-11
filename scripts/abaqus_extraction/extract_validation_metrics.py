# ------------------------------------------------------------
# Script: extract_validation_metrics.py
# Purpose:
#   Extract validation metrics from Abaqus ODB files
#   for M27 and M28 validation models.
#
# Output:
#   dataset/raw/summary_validation_models.csv
# ------------------------------------------------------------

from odbAccess import openOdb
import os
import csv

BASE_DIR = os.path.abspath(os.getcwd())

ODB_DIR = os.path.join(BASE_DIR, "models", "odb_results")

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "summary_validation_models.csv"
)

odb_files = [
    "M27_exp_L8_L16_validation.odb",
    "M28_power_n5_L8_L16_validation.odb"
]

def extract_metrics(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)

    step_name = list(odb.steps.keys())[-1]
    step = odb.steps[step_name]
    last_frame = step.frames[-1]

    stress_field = last_frame.fieldOutputs["S"]

    s11_values = []
    mises_values = []

    for value in stress_field.values:
        s11_values.append(value.data[0])
        mises_values.append(value.mises)

    max_s11 = max(s11_values)
    min_s11 = min(s11_values)
    max_mises = max(mises_values)

    odb.close()

    return max_s11, min_s11, max_mises


rows = []

for odb_file in odb_files:
    odb_path = os.path.join(ODB_DIR, odb_file)

    model_id = odb_file.split("_")[0]

    print("Processing:", odb_file)

    if not os.path.exists(odb_path):
        print("WARNING: file not found:", odb_path)
        continue

    max_s11, min_s11, max_mises = extract_metrics(odb_path)

    rows.append({
        "model_id": model_id,
        "odb_file": odb_file,
        "Max_S11_MPa": max_s11,
        "Min_S11_MPa": min_s11,
        "Max_Mises_MPa": max_mises
    })


with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = [
        "model_id",
        "odb_file",
        "Max_S11_MPa",
        "Min_S11_MPa",
        "Max_Mises_MPa"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(row)

print("\nValidation extraction completed.")
print("Saved file:", OUTPUT_CSV)