
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate the analytics folder
folder = Path(__file__).resolve().parent

# Load cleaned dataset
df = pd.read_csv(folder / "titanic_cleaned.csv")

print("Dataset shape:", df.shape)

# Basic information
print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# -----------------------------------
# 1. Age distribution
# -----------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(folder / "age_distribution.png")
plt.show()

# -----------------------------------
# 2. Fare distribution
# -----------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Distribution of Ticket Fares")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(folder / "fare_distribution.png")
plt.show()

# -----------------------------------
# 3. Age boxplot
# -----------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Age Boxplot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(folder / "age_boxplot.png")
plt.show()

# -----------------------------------
# 4. Fare boxplot
# -----------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"])
plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(folder / "fare_boxplot.png")
plt.show()

# -----------------------------------
# 5. Fare statistics
# -----------------------------------
print("\nFare statistics:")
print("Mean:", df["fare"].mean())
print("Median:", df["fare"].median())
print("Mode:", df["fare"].mode()[0])
print("Skewness:", df["fare"].skew())

# -----------------------------------
# 6. IQR outlier calculation
# -----------------------------------
for column in ["age", "fare"]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\n{column.upper()} outlier analysis:")
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Number of outliers:", len(outliers))

print("\nEDA analysis completed!")
