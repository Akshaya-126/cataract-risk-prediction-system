import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
data = pd.read_csv("cataract_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

# -------------------------------
# 2. SPLIT DATA
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. TRAIN MODEL
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# 4. MODEL EVALUATION
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------------
# 5. NEW PATIENT INPUT
# -------------------------------
new_patient = pd.DataFrame(
    [[68, 1, 1, 1, 1, 0, 1]],
    columns=X.columns
)

# -------------------------------
# 6. PREDICTION
# -------------------------------
probability = model.predict_proba(new_patient)[0][1]

# -------------------------------
# 7. RESULT & RISK LEVEL
# -------------------------------
if probability > 0.5:
    result = "Cataract Detected"
else:
    result = "No Cataract Detected"

if probability < 0.4:
    risk = "Low"
elif probability < 0.7:
    risk = "Medium"
else:
    risk = "High"

# -------------------------------
# 8. OUTPUT
# -------------------------------
print("\n--- PREDICTION RESULT ---")
print("Result:", result)
print("Risk Level:", risk)
print("Probability:", round(probability, 2))

if risk in ["Medium", "High"]:
    print("Suggestion: Visit an ophthalmologist for clinical confirmation.")
else:
    print("Suggestion: Regular eye checkup recommended.")
