import pandas as pd

# Step 1: Load the datasets
vision_df = pd.read_csv("vision_symptom_dataset.csv")           # your generated vision dataset
diabetes_df = pd.read_csv("diabetes_filtered.csv")             # filtered diabetes dataset

# Strip spaces from column names
vision_df.columns = vision_df.columns.str.strip()
diabetes_df.columns = diabetes_df.columns.str.strip()

# Rename outcome to diabetes if still named outcome
if 'outcome' in diabetes_df.columns:
    diabetes_df.rename(columns={'outcome': 'diabetes'}, inplace=True)

# Step 2: Merge datasets on 'age'
merged_df = pd.merge(vision_df, diabetes_df, on="age", how="left")

# Step 3: Handle missing values safely
for col in ['diabetes', 'smoking', 'family_history']:
    if col in merged_df.columns:
        merged_df[col].fillna(0, inplace=True)
        merged_df[col] = merged_df[col].astype(int)

# Step 4: Save the merged dataset
merged_df.to_csv("clinical_dataset.csv", index=False)

print("✅ Vision + Diabetes datasets merged successfully! 'clinical_dataset.csv' is ready for training.")
print("Columns:", merged_df.columns)
