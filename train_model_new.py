import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv("clinical_dataset.csv")

# -------------------------------
# Step 2: Sampling (optional)
# -------------------------------
if len(df) > 20000:
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)

# -------------------------------
# Step 3: Encode categorical fields
# -------------------------------
cat_cols = ['Smoking Status', 'Family History', 'Diabetes']
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# -------------------------------
# Step 4: BINNING (SMOOTHING)
# -------------------------------
df['age_bin'] = pd.cut(
    df['age'],
    bins=[0, 40, 55, 70, 100],
    labels=[0, 1, 2, 3]
)

df['symptom_score'] = (
    df['blurred_vision'] +
    df['glare'] +
    df['night_vision_issue']
)

# -------------------------------
# Step 5: Create risk label (WITH NOISE)
# -------------------------------
def assign_risk_with_noise(row):
    score = row['symptom_score'] + row['Diabetes'] + int(row['age_bin'])

    # add uncertainty (noise)
    score += np.random.choice([0, 0, 1, -1])

    if score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"

df['risk_label'] = df.apply(assign_risk_with_noise, axis=1)

# -------------------------------
# Step 6: Prepare ML data
# -------------------------------
feature_cols = [
    'age_bin',
    'symptom_score',
    'Smoking Status',
    'Family History',
    'Diabetes'
]

X = df[feature_cols]
y = df['risk_label']

le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

# -------------------------------
# Step 7: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# -------------------------------
# Step 8: Train REGULARIZED Random Forest
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,          # prevents memorization
    min_samples_leaf=15,  # smoothing
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Step 9: Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}\n")

print("📊 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=le_target.classes_
))

# -------------------------------
# Step 10: Save model
# -------------------------------
with open("cataract_risk_model.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'le_target': le_target,
        'le_features': le_dict
    }, f)

print("\n✅ Model saved as cataract_risk_model.pkl")