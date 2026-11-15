import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
# 2. Stratified train/test split
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
numeric_transformer = SklearnPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = SklearnPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# ---------------------------------------
# 4. Three experiments
# ---------------------------------------
experiments = {
    "Baseline Random Forest": SklearnPipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "Balanced Random Forest": SklearnPipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "SMOTE Random Forest": ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])
}

# ---------------------------------------
# 5. Train and evaluate
# ---------------------------------------
results = []

for name, pipeline in experiments.items():

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "Experiment": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probabilities)
    }

    results.append(result)

    print("\n", name)
    print("-" * 40)

    for metric, value in result.items():
        if metric != "Experiment":
            print(f"{metric}: {value:.4f}")

# ---------------------------------------
# 6. Save comparison
# ---------------------------------------
results_df = pd.DataFrame(results)

print("\nCLASS IMBALANCE COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    folder / "class_imbalance_results.csv",
    index=False
)

print("\nClass imbalance experiments completed!")
