import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "titanic_cleaned.csv")

# Select age and fare
columns = ["age", "fare"]

# Apply Z-score standardization
scaler = StandardScaler()

standardized_data = scaler.fit_transform(df[columns])

standardized_df = pd.DataFrame(
    standardized_data,
    columns=["age_zscore", "fare_zscore"]
)

# Display results
print("Original data:")
print(df[columns].head())

print("\nStandardized data:")
print(standardized_df.head())

print("\nStandardized means:")
print(standardized_df.mean())

print("\nStandardized standard deviations:")
print(standardized_df.std())

# Save standardized data
standardized_df.to_csv(
    folder / "standardized_data.csv",
    index=False
)

print("\nStandardization completed successfully!")
