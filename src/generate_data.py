import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

corridors = ["Megenagna", "Bole", "Kazanchis", "Piazza", "Merkato", "Sarbet", "Ayat", "Gotera"]

data = {
    "corridor_id": np.random.choice(corridors, n_samples),
    "hour_of_day": np.random.randint(0, 24, n_samples),
    "ambient_light": np.random.uniform(0.0, 1.0, n_samples),
    "street_light_func": np.random.uniform(0.0, 1.0, n_samples),
    "crime_risk_index": np.random.uniform(0.0, 10.0, n_samples),
    "foot_traffic": np.random.uniform(0.0, 100.0, n_samples),
}

df = pd.DataFrame(data)

def assign_risk(row):
    score = (
        (10.0 - row["crime_risk_index"]) * 0.3 +
        (row["ambient_light"] * 10) * 0.2 +
        (row["street_light_func"] * 10) * 0.2 +
        (row["foot_traffic"] / 10) * 0.3
    )
    if row["hour_of_day"] >= 22 or row["hour_of_day"] <= 4:
        score -= 2.0  # Late night risk penalty
    
    if score > 6.0:
        return 0  # Safe
    elif score > 3.5:
        return 1  # Escort Needed
    else:
        return 2  # Unsafe

df["risk_code"] = df.apply(assign_risk, axis=1)

# Save output dataset
df.to_csv("data/processed/corridor_safety_data.csv", index=False)
print("SUCCESS: Generated 1,000 corridor records -> data/processed/corridor_safety_data.csv")