import pandas as pd
from pathlib import Path

# Locate the analytics folder
folder = Path(__file__).resolve().parent

# Load the saved original dataset
df = pd.read_csv(folder / "titanic.csv")

print("Original dataset shape:", df.shape)

# Calculate missing values and percentages
missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

print("\nMissing Value Report:")
print(missing_report)

# Remove rows where missing percentage is below 5%
# embarked and embark_town each have less than 5% missing data
df = df.dropna(subset=["embarked", "embark_town"])

# Impute age using the median
age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)

# Drop deck because more than 75% of its values are missing
df = df.drop(columns=["deck"])

# Check remaining missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)

# Save the cleaned dataset
df.to_csv(folder / "titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
