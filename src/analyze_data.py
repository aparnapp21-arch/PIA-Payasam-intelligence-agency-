import pandas as pd
import matplotlib.pyplot as plt

# Load the PIA dataset
data = pd.read_csv("data/payasam_data.csv")

# Basic information
print("Number of samples:", len(data))
print("Number of columns:", len(data.columns))

print("\nColumn names:")
print(data.columns.tolist())

# First 5 rows
print("\nFirst 5 rows:")
print(data.head())

# Dataset information
print("\nDataset information:")
data.info()

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

# Payasam type distribution
print("\nPayasam types:")
print(data["payasam_type"].value_counts())

# Consistency score
print("\nConsistency score:")
print("Minimum:", data["consistency_score"].min())
print("Maximum:", data["consistency_score"].max())
print("Average:", data["consistency_score"].mean())

# Calculate total liquid
data["total_liquid_ml"] = data["milk_ml"] + data["water_ml"]

# Correlation analysis
print("\nCorrelation with consistency score:")

numeric_data = data.select_dtypes(include="number")

correlations = numeric_data.corr()["consistency_score"].sort_values(
    ascending=False
)

print(correlations)

# Graph 1: Cooking time vs consistency
plt.figure()
plt.scatter(data["cooking_time_min"], data["consistency_score"])
plt.xlabel("Cooking Time (minutes)")
plt.ylabel("Consistency Score")
plt.title("Cooking Time vs Consistency")
plt.show()

# Graph 2: Total liquid vs consistency
plt.figure()
plt.scatter(data["total_liquid_ml"], data["consistency_score"])
plt.xlabel("Total Liquid (ml)")
plt.ylabel("Consistency Score")
plt.title("Total Liquid vs Consistency")
plt.show()

# Graph 3: Distribution of consistency scores
plt.figure()
plt.hist(data["consistency_score"], bins=20)
plt.xlabel("Consistency Score")
plt.ylabel("Number of Samples")
plt.title("Distribution of Consistency Scores")
plt.show()