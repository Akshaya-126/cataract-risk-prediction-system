import pandas as pd
import pickle
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv("clinical_dataset_sampled_labeled.csv")

# -------------------------------
# Step 2: Feature Engineering (MUST MATCH TRAINING)
# -------------------------------

# Age binning
df["age_bin"] = pd.cut(
    df["age"],
    bins=[0, 40, 60, 120],
    labels=[0, 1, 2]
).astype(int)

# Symptom score
df["symptom_score"] = (
    df["blurred_vision"] +
    df["glare"] +
    df["night_vision_issue"] +
    df["Diabetes"]
)

# -------------------------------
# Step 3: Prepare features & target
# -------------------------------
X_test = df[["age_bin", "symptom_score"]]
y_test = df["risk_label"]

# -------------------------------
# Step 4: Load trained model
# -------------------------------
with open("cataract_risk_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le_target = data["le_target"]

# Encode target
y_test_enc = le_target.transform(y_test)

# -------------------------------
# Step 5: Predictions
# -------------------------------
y_pred_enc = model.predict(X_test)
y_pred = le_target.inverse_transform(y_pred_enc)
y_true = le_target.inverse_transform(y_test_enc)

# -------------------------------
# Step 6: Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred, labels=["Low", "Medium", "High"])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# Step 7: Classification Report
# -------------------------------
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Low", "Medium", "High"]))

# -------------------------------
# Step 8: Accuracy
# -------------------------------
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# Step 9: Precision, Recall, F1-score
# -------------------------------
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=["Low", "Medium", "High"]
)

df_metrics = pd.DataFrame({
    "Class": ["Low", "Medium", "High"],
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})

print("\n📌 Precision, Recall, F1-Score per Class:\n")
print(df_metrics)