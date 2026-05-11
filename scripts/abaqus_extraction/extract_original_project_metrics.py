from odbAccess import openOdb
import os
import csv

BASE_DIR = os.path.abspath(os.getcwd())

ODB_DIR = os.path.join(BASE_DIR, "models", "odb_original_project")
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset", "raw", "summary_original_project_mises.csv")

# -----------------------------
# AUTO-DETECT ODB FILES
# -----------------------------
odb_files = [f for f in os.listdir(ODB_DIR) if f.endswith(".odb")]

# Filtra fuori eventuali file sener (non servono)
odb_files = [f for f in odb_files if "sener" not in f.lower()]

print("\nFound ODB files:")
for f in odb_files:
    print(f)

# -----------------------------
# EXTRACTION FUNCTION
# -----------------------------
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

# -----------------------------
# MAIN LOOP
# -----------------------------
rows = []

for odb_file in odb_files:
    odb_path = os.path.join(ODB_DIR, odb_file)

    # Estrae M01, M02, ecc.
    model_id = odb_file.split("_")[0]

    print("\nProcessing:", odb_file)

    max_s11, min_s11, max_mises = extract_metrics(odb_path)

    rows.append({
        "model_id": model_id,
        "odb_file": odb_file,
        "Max_S11_MPa_from_odb": max_s11,
        "Min_S11_MPa_from_odb": min_s11,
        "Max_Mises_MPa": max_mises
    })

# -----------------------------
# SAVE CSV
# -----------------------------
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = [
        "model_id",
        "odb_file",
        "Max_S11_MPa_from_odb",
        "Min_S11_MPa_from_odb",
        "Max_Mises_MPa"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(row)

print("\nExtraction completed.")
print("Saved file:", OUTPUT_CSV)