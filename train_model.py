import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv("clinical_dataset.csv")  # full dataset

# -------------------------------
# Step 2: Sample ~20k rows if dataset is huge
# -------------------------------
if len(df) > 20000:
    df_sample = df.groupby('age', group_keys=False).apply(
        lambda x: x.sample(frac=min(1, 20000/len(df)), random_state=42)
    )
else:
    df_sample = df.copy()
df = df_sample.reset_index(drop=True)

# -------------------------------
# Step 3: Encode categorical fields
# -------------------------------
cat_cols = ['Smoking Status', 'Family History', 'Diabetes']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # store label encoder for possible future use

# -------------------------------
# Step 4: Compute risk label and reason
# -------------------------------
def assign_risk(row):
    # Include age in risk scoring
    score = row['blurred_vision'] + row['glare'] + row['night_vision_issue'] + row['Diabetes']
    if row['age'] >= 60:
        score += 1  # age is a major factor

    # Build reason string
    reason_parts = []
    if row['blurred_vision']: reason_parts.append("Blurred vision")
    if row['glare']: reason_parts.append("Glare")
    if row['night_vision_issue']: reason_parts.append("Night vision issues")
    if row['Diabetes']: reason_parts.append("Diabetes")
    if row['age'] >= 60: reason_parts.append("Age >= 60")

    reason = ", ".join(reason_parts) if reason_parts else "No major symptoms"

    # Risk thresholds (tunable)
    if score >= 4:
        risk = "High"
    elif score == 3:
        risk = "Medium"
    else:
        risk = "Low"

    return pd.Series([risk, reason])

df[['risk_label', 'reason']] = df.apply(assign_risk, axis=1)

# -------------------------------
# Step 5: Prepare ML training
# -------------------------------
feature_cols = ['age', 'blurred_vision', 'glare', 'night_vision_issue', 'Smoking Status', 'Family History', 'Diabetes']
X = df[feature_cols]
y = df['risk_label']

# Encode target
le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

# -------------------------------
# Step 6: Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# -------------------------------
# Step 7: Train Random Forest
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 8: Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy on test set: {accuracy:.2%}")

# -------------------------------
# Step 9: Save model and label encoders
# -------------------------------
with open("cataract_risk_model.pkl", "wb") as f:
    pickle.dump({'model': model, 'le_target': le_target, 'le_features': le_dict}, f)

# -------------------------------
# Step 10: Save the labeled dataset (optional)
# -------------------------------
df.to_csv("clinical_dataset_sampled_labeled.csv", index=False)

print("✅ Training complete. Model saved as 'cataract_risk_model.pkl'")