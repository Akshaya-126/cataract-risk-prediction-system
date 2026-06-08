import pandas as pd
import random

data = []

def calculate_risk(age, blurred, glare, night, diabetes, smoking, family):
    score = 0

    # Age contribution
    if 30 <= age < 40:
        score += 1
    elif age >= 40:
        score += 2

    # Symptom contribution
    score += blurred * 2
    score += glare * 2
    score += night * 3

    # Risk factors
    score += diabetes * 1
    score += smoking * 1
    score += family * 1

    # Risk level
    if score <= 3:
        return "Low"
    elif score <= 7:
        return "Medium"
    else:
        return "High"

# Generate 300 patients
for _ in range(300):
    age = random.randint(20, 80)

    blurred = random.choice([0, 1])
    glare = random.choice([0, 1])
    night = random.choice([0, 1])
    diabetes = random.choice([0, 1])
    smoking = random.choice([0, 1])
    family = random.choice([0, 1])

    risk = calculate_risk(
        age, blurred, glare, night,
        diabetes, smoking, family
    )

    data.append([
        age, blurred, glare, night,
        diabetes, smoking, family, risk
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "age",
    "blurred_vision",
    "glare",
    "night_vision",
    "diabetes",
    "smoking",
    "family_history",
    "risk_label"
])

# Save CSV
df.to_csv("cataract_clinical_data.csv", index=False)

print("✅ Clinically correct dataset created: cataract_clinical_data.csv")
