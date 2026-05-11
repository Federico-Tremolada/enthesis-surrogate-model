import pandas as pd

# load
df = pd.read_csv("./dataset/processed/enthesis_dataset_v1.csv")

print("Original columns:")
print(df.columns.tolist())

# rename target columns
df = df.rename(columns={
    "Max_S11_MPa": "max_S11",
    "Min_S11_MPa": "min_S11"
})

# remove constant columns
constant_cols = []

for col in df.columns:
    if df[col].nunique() <= 1:
        constant_cols.append(col)

print("\nConstant columns removed:")
print(constant_cols)

df = df.drop(columns=constant_cols)

print("\nFinal columns:")
print(df.columns.tolist())

# save
df.to_csv(
    "./dataset/processed/enthesis_dataset_v2_clean.csv",
    index=False
)

print("\nClean dataset saved.")