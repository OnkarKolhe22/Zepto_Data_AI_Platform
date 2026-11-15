
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------
# 1. Load cleaned dataset
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

X = df.drop(columns=["survived"])
y = df["survived"]

numeric_features = [
    "pclass", "age", "sibsp", "parch", "fare"
]

categorical_features = [
    "sex", "embarked"
]

# ---------------------------------------
# 2. Split dataset
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# 3. Preprocessing
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
# 4. Build tuned Random Forest pipeline
# ---------------------------------------
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        max_features="sqrt",
        random_state=42
    ))
])

# ---------------------------------------
# 5. Train final model
# ---------------------------------------
final_pipeline.fit(X_train, y_train)

predictions = final_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Final model test accuracy:", round(accuracy, 4))

print("\nClassification report:")
print(classification_report(y_test, predictions))

# ---------------------------------------
# 6. Save complete pipeline
# ---------------------------------------
model_path = folder / "titanic_survival_pipeline.joblib"

joblib.dump(final_pipeline, model_path)

print("\nModel saved successfully!")
print("File:", model_path)

# ---------------------------------------
# 7. Reload saved pipeline
# ---------------------------------------
loaded_model = joblib.load(model_path)

print("\nSaved model loaded successfully!")

# ---------------------------------------
# 8. Test with a new raw passenger
# ---------------------------------------
new_passenger = pd.DataFrame([{
    "pclass": 3,
    "sex": "male",
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S"
}])

prediction = loaded_model.predict(new_passenger)[0]

probability = loaded_model.predict_proba(new_passenger)[0][1]

print("\nNEW PASSENGER PREDICTION")

if prediction == 1:
    print("Predicted result: Survived")
else:
    print("Predicted result: Did not survive")

print("Survival probability:", round(probability, 4))

print("\nModel save, reload and prediction test completed!")
