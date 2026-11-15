import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------
# 1. Load cleaned Titanic dataset
# ---------------------------------------
folder = Path(__file__).resolve().parent

df = pd.read_csv(folder / "titanic_cleaned.csv")

df = df.drop(columns=[
    "alive",
    "class",
    "who",
    "adult_male",
    "embark_town",
    "alone"
])

# ---------------------------------------
# 2. Separate features and target
# ---------------------------------------
X = df.drop(columns=["survived", "fare"])
y = df["fare"]

numeric_features = [
    "pclass", "age", "sibsp", "parch"
]

categorical_features = [
    "sex", "embarked"
]

# ---------------------------------------
# 3. Split data
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------
# 4. Preprocessing
# ---------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# ---------------------------------------
# 5. Build regression pipeline
# ---------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# ---------------------------------------
# 6. Train and predict
# ---------------------------------------
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ---------------------------------------
# 7. Calculate metrics
# ---------------------------------------
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)

n = len(y_test)
p = X_test.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\nREGRESSION MODEL RESULTS")
print("-" * 40)

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))
print("Adjusted R2 Score:", round(adjusted_r2, 4))

# ---------------------------------------
# 8. Residual analysis
# ---------------------------------------
residuals = y_test - predictions

plt.figure(figsize=(8, 5))

plt.scatter(predictions, residuals, alpha=0.6)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")

plt.title("Residual Plot - Fare Prediction")

plt.tight_layout()

plt.savefig(
    folder / "regression_residual_plot.png",
    dpi=300
)

plt.close()

# ---------------------------------------
# 9. Save results
# ---------------------------------------
results = pd.DataFrame([{
    "Model": "Random Forest Regressor",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted R2": adjusted_r2
}])

results.to_csv(
    folder / "regression_results.csv",
    index=False
)

print("\nResidual plot saved!")
print("Regression results saved!")
print("\nRegression analysis completed!")
