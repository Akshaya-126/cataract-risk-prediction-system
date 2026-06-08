import pandas as pd
import random
import numpy as np

data = []

for _ in range(20000):
    # Age distribution (realistic shape)
    # majority 20-70, with more older people
    age = int(np.clip(random.gauss(50, 15), 18, 85))

    # Base symptom probability increases with age
    p_blurred = min(0.4 + (age - 20) / 100, 0.9)
    p_glare = min(0.3 + (age - 20) / 130, 0.85)
    p_night = min(0.2 + (age - 20) / 150, 0.8)

    # Introduce outliers randomness
    blur = 1 if random.random() < p_blurred else 0
    glare = 1 if random.random() < p_glare else 0
    night = 1 if random.random() < p_night else 0

    # Inject some controlled outliers
    # 1% unusually symptomatic young
    if age < 30 and random.random() < 0.01:
        blur, glare, night = 1, 1, 1

    # 2% older people with no symptoms
    if age > 70 and random.random() < 0.02:
        blur, glare, night = 0, 0, 0

    data.append([age, blur, glare, night])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "age", "blurred_vision", "glare", "night_vision_issue"
])

# Save to CSV
df.to_csv("vision_symptom_dataset_20k.csv", index=False)
print("✅ Vision dataset with 20,000+ records created successfully!")
