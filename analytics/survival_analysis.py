import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned dataset
folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "titanic_cleaned.csv")

# ----------------------------------
# 1. Survival rate by sex
# ----------------------------------
print("\nSURVIVAL RATE BY SEX")

male_survival = df[df["sex"] == "male"]["survived"].mean()
female_survival = df[df["sex"] == "female"]["survived"].mean()

print("Male survival rate:", round(male_survival * 100, 2), "%")
print("Female survival rate:", round(female_survival * 100, 2), "%")

# ----------------------------------
# 2. Survival rate by passenger class
# ----------------------------------
print("\nSURVIVAL RATE BY PASSENGER CLASS")

for passenger_class in sorted(df["pclass"].unique()):
    rate = df[df["pclass"] == passenger_class]["survived"].mean()
    print(
        f"Class {passenger_class} survival rate:",
        round(rate * 100, 2), "%"
    )

# ----------------------------------
# 3. Survival rate by sex and class
# ----------------------------------
print("\nSURVIVAL RATE BY SEX AND CLASS")

survival_table = df.groupby(
    ["sex", "pclass"],
    observed=True
)["survived"].mean() * 100

print(survival_table)

# ----------------------------------
# 4. Visualize survival by sex
# ----------------------------------
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(folder / "survival_by_sex.png")
plt.show()

# ----------------------------------
# 5. Visualize survival by class
# ----------------------------------
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(folder / "survival_by_class.png")
plt.show()

# ----------------------------------
# 6. Visualize survival by sex and class
# ----------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(folder / "survival_by_sex_class.png")
plt.show()

print("\nSurvival analysis completed!")
