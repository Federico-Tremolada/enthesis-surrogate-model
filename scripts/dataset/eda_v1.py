import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# PATH
# -----------------------------
dataset_path = r"./dataset/processed/enthesis_dataset_v1.csv"

results_path = r"./results/eda"

os.makedirs(results_path, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(dataset_path)

print("\nDATASET INFO")
print(df.info())

print("\nHEAD")
print(df.head())

print("\nDESCRIPTIVE STATS")
print(df.describe(include="all"))

# -----------------------------
# MISSING VALUES
# -----------------------------
print("\nMISSING VALUES")
print(df.isnull().sum())

# -----------------------------
# TARGET DISTRIBUTIONS
# -----------------------------
targets = [
    "max_S11",
    "min_S11",
    "max_vonMises"
]

for target in targets:
    if target in df.columns:
        plt.figure(figsize=(6,4))
        plt.hist(df[target].dropna(), bins=10)
        plt.title(f"Distribution of {target}")
        plt.xlabel(target)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{results_path}/{target}_distribution.png")
        plt.close()

# -----------------------------
# LAW TYPE COMPARISON
# -----------------------------
if "law_type" in df.columns and "max_S11" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="law_type", y="max_S11")
    plt.title("max_S11 by law type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_path}/maxS11_by_lawtype.png")
    plt.close()

# -----------------------------
# N EXPONENT EFFECT
# -----------------------------
if "n_exponent" in df.columns and "max_S11" in df.columns:
    power_df = df[df["law_type"] == "power_law"]

    plt.figure(figsize=(6,4))
    plt.scatter(power_df["n_exponent"], power_df["max_S11"])
    plt.title("Effect of n exponent on max_S11")
    plt.xlabel("n exponent")
    plt.ylabel("max_S11")
    plt.tight_layout()
    plt.savefig(f"{results_path}/n_vs_maxS11.png")
    plt.close()

# -----------------------------
# CORRELATION MATRIX
# -----------------------------
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{results_path}/correlation_matrix.png")
plt.close()

print("\nEDA completed successfully.")
print(f"Results saved in: {results_path}")