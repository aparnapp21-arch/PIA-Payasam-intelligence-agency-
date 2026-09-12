import pandas as pd
import numpy as np

# Make the results reproducible
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate basic recipe parameters
payasam_types = np.random.choice(
    ["Palada", "Semiya", "Rice", "Parippu", "Paal"],
    size=num_samples
)

milk_ml = np.random.randint(200, 801, num_samples)
water_ml = np.random.randint(100, 601, num_samples)
main_ingredient_g = np.random.randint(30, 151, num_samples)
sugar_g = np.random.randint(50, 201, num_samples)
cooking_time_min = np.random.randint(15, 61, num_samples)
temperature_c = np.random.randint(60, 91, num_samples)

# Calculate a synthetic consistency score
liquid = milk_ml + water_ml

score = (
    50
    + 0.25 * main_ingredient_g
    + 0.05 * sugar_g
    + 0.35 * cooking_time_min
    - 0.035 * liquid
    + 0.10 * (temperature_c - 60)
)

# Add a small amount of natural variation
score += np.random.normal(0, 5, num_samples)

# Keep the score between 0 and 100
consistency_score = np.clip(score, 0, 100)

# Create the dataset
data = pd.DataFrame({
    "payasam_type": payasam_types,
    "milk_ml": milk_ml,
    "water_ml": water_ml,
    "main_ingredient_g": main_ingredient_g,
    "sugar_g": sugar_g,
    "cooking_time_min": cooking_time_min,
    "temperature_c": temperature_c,
    "consistency_score": consistency_score.round(2)
})

# Save the dataset
data.to_csv("data/payasam_data.csv", index=False)

print("PIA dataset created successfully!")
print(f"Number of samples: {len(data)}")
print("\nFirst 5 samples:")
print(data.head())