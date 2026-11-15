
import pandas as pd
import seaborn as sns
from pathlib import Path

# Find the analytics folder
folder = Path(__file__).resolve().parent

# Load the Titanic dataset exactly once
df = sns.load_dataset("titanic")

# Immediately save the original dataset
df.to_csv(folder / "titanic.csv", index=False)

print("Titanic dataset loaded and saved successfully!")

# Display basic dataset information
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))
