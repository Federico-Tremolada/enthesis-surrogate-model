import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("dataset/processed/enthesis_dataset_v4_complete.csv")

# -----------------------------
# CLEAN COLUMNS
# -----------------------------
if "law_type_x" in df.columns:
    df["law_type_clean"] = df["law_type_x"]
elif "law_type_y" in df.columns:
    df["law_type_clean"] = df["law_type_y"]
elif "law_type" in df.columns:
    df["law_type_clean"] = df["law_type"]
else:
    raise ValueError("No law type column found.")

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

df["enthesis_length_mm_clean"] = df["enthesis_length_mm_clean"].fillna(6)
df["number_of_layers"] = df["number_of_layers"].fillna(16)
df["n_exponent"] = df["n_exponent"].fillna(0)

target = "max_vonMises"
df = df.dropna(subset=[target])

# -----------------------------
# TRAIN SURROGATE MODEL
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
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

y = df[target].astype(float)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

training_columns = X.columns.tolist()

# -----------------------------
# DESIGN SPACE FOR OPTIMIZATION
# -----------------------------
candidate_rows = []

law_types = ["linear", "exponential", "power_law"]
lengths = np.linspace(4, 8, 17)     # 4.0, 4.25, ..., 8.0
layers_list = [8, 12, 16]
n_values = [0, 1.5, 2, 3, 5]

for law in law_types:
    for L in lengths:
        for layers in layers_list:

            if law == "power_law":
                for n in n_values:
                    if n == 0:
                        continue

                    candidate_rows.append({
                        "law_type_clean": law,
                        "enthesis_length_mm_clean": L,
                        "n_exponent": n,
                        "number_of_layers": layers
                    })

            else:
                candidate_rows.append({
                    "law_type_clean": law,
                    "enthesis_length_mm_clean": L,
                    "n_exponent": 0,
                    "number_of_layers": layers
                })

candidates = pd.DataFrame(candidate_rows)

# -----------------------------
# PREPARE CANDIDATES
# -----------------------------
Xc_num = candidates[numeric_features]

Xc_cat = pd.get_dummies(
    candidates["law_type_clean"],
    prefix="law_type"
)

Xc = pd.concat([Xc_num, Xc_cat], axis=1)
Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(0)

# align columns with training
Xc = Xc.reindex(columns=training_columns, fill_value=0)

# -----------------------------
# PREDICT
# -----------------------------
candidates["predicted_max_vonMises"] = model.predict(Xc)

# -----------------------------
# RANK RESULTS
# -----------------------------
best = candidates.sort_values(
    by="predicted_max_vonMises",
    ascending=True
)

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("results/optimization", exist_ok=True)

best.to_csv(
    "results/optimization/optimization_results_v1.csv",
    index=False
)

best.head(20).to_csv(
    "results/optimization/top20_designs_v1.csv",
    index=False
)

print("\nOptimization completed.")
print("\nTop 20 candidate designs:")
print(best.head(20))

print("\nSaved:")
print("results/optimization/optimization_results_v1.csv")
print("results/optimization/top20_designs_v1.csv")