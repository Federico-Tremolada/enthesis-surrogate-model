import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# PATHS
# -----------------------------
input_path = "./dataset/raw/summary_length_study.csv"
output_dir = "./results/length_study"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(input_path)

# -----------------------------
# ADD MODEL METADATA
# -----------------------------
metadata = {
    "M19": {"law_type": "linear", "enthesis_length_mm": 4},
    "M20": {"law_type": "exponential", "enthesis_length_mm": 4},
    "M21": {"law_type": "power_n2", "enthesis_length_mm": 4},
    "M22": {"law_type": "power_n3", "enthesis_length_mm": 4},
    "M23": {"law_type": "linear", "enthesis_length_mm": 8},
    "M24": {"law_type": "exponential", "enthesis_length_mm": 8},
    "M25": {"law_type": "power_n2", "enthesis_length_mm": 8},
    "M26": {"law_type": "power_n3", "enthesis_length_mm": 8},
}

df["law_type"] = df["model_id"].map(lambda x: metadata[x]["law_type"])
df["enthesis_length_mm"] = df["model_id"].map(lambda x: metadata[x]["enthesis_length_mm"])

# Use absolute compressive S11 peak
df["Abs_Min_S11_MPa"] = df["Min_S11_MPa"].abs()

# -----------------------------
# SAVE ENRICHED TABLE
# -----------------------------
df.to_csv("./dataset/processed/length_study_enriched.csv", index=False)

print("\nLength study enriched table:")
print(df)

# -----------------------------
# BAR PLOT: MAX VON MISES
# -----------------------------
plt.figure(figsize=(8, 5))

for length in [4, 8]:
    subset = df[df["enthesis_length_mm"] == length]
    plt.bar(
        subset["law_type"] + f"_L{length}",
        subset["Max_Mises_MPa"]
    )

plt.ylabel("Max von Mises stress [MPa]")
plt.xlabel("Material transition law")
plt.title("Maximum von Mises stress - Length study")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/bar_max_mises_length_study.png", dpi=300)
plt.close()

# -----------------------------
# BAR PLOT: ABS MIN S11
# -----------------------------
plt.figure(figsize=(8, 5))

for length in [4, 8]:
    subset = df[df["enthesis_length_mm"] == length]
    plt.bar(
        subset["law_type"] + f"_L{length}",
        subset["Abs_Min_S11_MPa"]
    )

plt.ylabel("|Min S11| [MPa]")
plt.xlabel("Material transition law")
plt.title("Compressive S11 peak - Length study")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/bar_abs_min_s11_length_study.png", dpi=300)
plt.close()

# -----------------------------
# LINE PLOT: EFFECT OF LENGTH
# -----------------------------
plt.figure(figsize=(7, 5))

for law in df["law_type"].unique():
    subset = df[df["law_type"] == law].sort_values("enthesis_length_mm")
    plt.plot(
        subset["enthesis_length_mm"],
        subset["Max_Mises_MPa"],
        marker="o",
        label=law
    )

plt.xlabel("Enthesis length [mm]")
plt.ylabel("Max von Mises stress [MPa]")
plt.title("Effect of enthesis length on peak von Mises stress")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/line_length_vs_mises.png", dpi=300)
plt.close()

print("\nPlots saved in:")
print(output_dir)