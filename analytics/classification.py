import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ---------------------------------------
# 1. Load cleaned Titanic dataset
# ---------------------------------------
folder = Path(__file__).resolve().parent

df = pd.read_csv(folder / "titanic_cleaned.csv")

# Remove columns that duplicate information or
# reveal the target directly.
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
X = df.drop(columns=["survived"])
y = df["survived"]

# Identify numerical and categorical columns
numeric_features = [
    "pclass", "age", "sibsp", "parch", "fare"
]

categorical_features = [
    "sex", "embarked"
]

# ---------------------------------------
# 3. Stratified train/test split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining survival proportions:")
print(y_train.value_counts(normalize=True))

print("\nTesting survival proportions:")
print(y_test.value_counts(normalize=True))

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
# 5. Create three classification models
# ---------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# ---------------------------------------
# 6. Train and evaluate models
# ---------------------------------------
results = []

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train using training data only
    pipeline.fit(X_train, y_train)

    # Predict on unseen testing data
    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

# ---------------------------------------
# 7. Compare all models
# ---------------------------------------
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    folder / "classification_results.csv",
    index=False
)

print("\nClassification analysis completed!")
