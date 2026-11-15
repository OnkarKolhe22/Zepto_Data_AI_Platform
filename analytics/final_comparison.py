import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

# ---------------------------------------
# 1. Classification comparison
# ---------------------------------------
classification = pd.read_csv(
    folder / "classification_results.csv"
)

imbalance = pd.read_csv(
    folder / "class_imbalance_results.csv"
)

print("\nCLASSIFICATION MODEL COMPARISON")
print("=" * 65)
print(classification.round(4).to_string(index=False))

print("\nCLASS IMBALANCE COMPARISON")
print("=" * 65)
print(imbalance.round(4).to_string(index=False))

classification.to_csv(
    folder / "classification_comparison.csv",
    index=False
)

imbalance.to_csv(
    folder / "imbalance_comparison.csv",
    index=False
)

# ---------------------------------------
# 2. Regression comparison
# ---------------------------------------
regression = pd.read_csv(
    folder / "regression_results.csv"
)

print("\nREGRESSION MODEL RESULTS")
print("=" * 65)
print(regression.round(4).to_string(index=False))

regression.to_csv(
    folder / "regression_comparison.csv",
    index=False
)

print("\nAll comparison tables saved successfully!")
