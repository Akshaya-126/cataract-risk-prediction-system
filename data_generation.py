import numpy as np
import pandas as pd
import random

np.random.seed(42)

N = 20000  # total samples

data = []

for _ in range(N):
    age = np.random.randint(35, 80)
    iop = np.random.randint(10, 35)  # intraocular pressure
    diabetes = np.random.choice([0, 1], p=[0.7, 0.3])
    family_history = np.random.choice([0, 1], p=[0.6, 0.4])
    blurred_vision = np.random.choice([0, 1], p=[0.65, 0.35])
    night_vision = np.random.choice([0, 1], p=[0.7, 0.3])

    # Add FEATURE NOISE (real-world uncertainty)
    if random.random() < 0.1:
        blurred_vision = 1 - blurred_vision
    if random.random() < 0.1:
        night_vision = 1 - night_vision

    # Risk score
    score = (
        (age > 55) +
        (iop > 21) +
        diabetes +
        family_history +
        blurred_vision +
        night_vision
    )

    # LABEL SMOOTHING (key part)
    if score <= 2:
        risk = np.random.choice(["Low", "Medium"], p=[0.8, 0.2])
    elif score <= 4:
        risk = np.random.choice(["Medium", "Low", "High"], p=[0.7, 0.15, 0.15])
    else:
        risk = np.random.choice(["High", "Medium"], p=[0.85, 0.15])

    data.append([
        age, iop, diabetes, family_history,
        blurred_vision, night_vision, risk
    ])

df = pd.DataFrame(data, columns=[
    "Age", "IOP", "Diabetes", "FamilyHistory",
    "BlurredVision", "NightVision", "Risk"
])

df.to_csv("glaucoma_data.csv", index=False)

print("✅ Dataset generated and saved as glaucoma_data.csv")