
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "titanic_cleaned.csv")

# Chart 1: Survival by sex and passenger class
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(folder / "multivariate_1.png")
plt.show()

# Chart 2: Age and fare by survival
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.6
)
plt.title("Age and Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(folder / "multivariate_2.png")
plt.show()

# Chart 3: Survival by embarkation port and sex
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="embark_town",
    y="survived",
    hue="sex"
)
plt.title("Survival by Embarkation Port and Sex")
plt.xlabel("Embarkation Town")
plt.ylabel("Survival Rate")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(folder / "multivariate_3.png")
plt.show()

# Chart 4: Family size and survival
df["family_size"] = df["sibsp"] + df["parch"] + 1

plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="family_size",
    y="survived"
)
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(folder / "multivariate_4.png")
plt.show()

# Chart 5: Fare by class and survival
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived"
)
plt.title("Fare Distribution by Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(folder / "multivariate_5.png")
plt.show()

print("Five multivariate charts created successfully!")
