import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------
# 1. Load dataset
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
# 2. Train/test split
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

# Fit preprocessing on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

# ---------------------------------------
# 4. Decision Tree visualization
# ---------------------------------------
decision_tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

decision_tree.fit(X_train_processed, y_train)

plt.figure(figsize=(22, 12))

plot_tree(
    decision_tree,
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Titanic Decision Tree Classification")
plt.tight_layout()

plt.savefig(
    folder / "decision_tree_visualization.png",
    dpi=300
)

plt.close()

print("Decision Tree visualization saved!")

# ---------------------------------------
# 5. Random Forest GridSearchCV
# ---------------------------------------
random_forest = RandomForestClassifier(
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", random_forest)
])

parameter_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBEST RANDOM FOREST PARAMETERS")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION F1 SCORE")
print(round(grid_search.best_score_, 4))

# ---------------------------------------
# 6. Evaluate best model on test data
# ---------------------------------------
best_model = grid_search.best_estimator_

test_score = best_model.score(X_test, y_test)

print("\nTEST ACCURACY")
print(round(test_score, 4))

# ---------------------------------------
# 7. Out-of-Bag score
# ---------------------------------------
oob_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=grid_search.best_params_[
        "classifier__max_depth"
    ],
    max_features=grid_search.best_params_[
        "classifier__max_features"
    ],
    oob_score=True,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

oob_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", oob_forest)
])

oob_pipeline.fit(X_train, y_train)

oob_score = oob_pipeline.named_steps[
    "classifier"
].oob_score_

print("\nRANDOM FOREST OOB SCORE")
print(round(oob_score, 4))

print("\nModel tuning completed successfully!")
