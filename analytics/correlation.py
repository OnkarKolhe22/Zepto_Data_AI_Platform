import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load cleaned dataset
folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "titanic_cleaned.csv")

# Select exactly the six required columns
columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_data = df[columns]

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Draw heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Titanic Dataset Correlation Heatmap")
plt.tight_layout()

plt.savefig(folder / "correlation_heatmap.png")
plt.show()

# Find the top two absolute off-diagonal correlations
correlation_pairs = correlation_matrix.abs().where(
    ~pd.DataFrame(
        [[i == j for j in range(len(columns))]
         for i in range(len(columns))],
        index=columns,
        columns=columns
    )
).stack().sort_values(ascending=False)

# Keep each pair only once
seen = set()

print("\nTop 2 strongest correlation pairs:")

count = 0

for (column1, column2), value in correlation_pairs.items():
    pair = tuple(sorted([column1, column2]))

    if pair not in seen:
        seen.add(pair)
        actual_value = correlation_matrix.loc[column1, column2]

        print(
            f"{column1} and {column2}:",
            round(actual_value, 3)
        )

        count += 1

        if count == 2:
            break

print("\nCorrelation analysis completed!")
