import pandas as pd

# Step 1: Load the diabetes dataset
df = pd.read_csv("Diabetes Dataset.csv")  # make sure file is in the same folder

# Step 2: Select only the needed columns with correct names
filtered_df = df[['Age', 'Smoking Status', 'Family History', 'Outcome']]

# Step 3: Rename columns to standard names
filtered_df.rename(columns={
    'Age': 'age',
    'smoking status': 'smoking',
    'family history': 'family_history',
    'outcome': 'diabetes'
}, inplace=True)

# Step 4: Save the filtered dataset
filtered_df.to_csv("diabetes_filtered.csv", index=False)

print("✅ Filtered diabetes dataset with columns: age, smoking, family_history, diabetes is ready!")
