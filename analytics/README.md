# Module 2 – Titanic Data Analysis and Machine Learning

## 1. Project Overview

This module performs exploratory data analysis, data preprocessing, visualization, classification and regression using the Titanic dataset.

The project uses Python, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn.

## 2. Dataset

The Titanic dataset is loaded using Seaborn's `load_dataset()` function.

- Original dataset: `titanic.csv`
- Cleaned dataset: `titanic_cleaned.csv`
- Original shape: 891 rows and 15 columns.
- Cleaned shape: 889 rows and 14 columns.

The original dataset is saved as a CSV file to provide an offline fallback.

## 3. Data Cleaning

- Missing age values were replaced using the median.
- Rows with missing embarked and embark_town values were removed.
- The deck column was removed because approximately 77.22% of its values were missing.
- The cleaned dataset contains no remaining missing values.

## 4. Exploratory Data Analysis

EDA includes:

- Dataset information and descriptive statistics.
- Age and fare histograms.
- Age and fare boxplots.
- IQR-based outlier analysis.
- Fare mean, median, mode and skewness.
- Survival analysis by sex and passenger class.
- Correlation heatmap.
- Five multivariate visualizations.
- Z-score standardization of age and fare.

The standardized age and fare columns have means approximately equal to zero and standard deviations approximately equal to one.

## 5. Classification Models

Three classification models were trained using a stratified train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The tuned Random Forest achieved:

- Test accuracy: 83.15%
- Cross-validation F1 score: 0.7408
- Out-of-bag score: 0.8214

The final model was evaluated using accuracy, precision, recall, F1 score, confusion matrix and ROC AUC.

## 6. Class Imbalance

Three experiments were compared:

- Baseline Random Forest
- Balanced Random Forest
- SMOTE Random Forest

Results:

- Baseline Random Forest achieved the highest accuracy of approximately 82.02%.
- Balanced Random Forest achieved the highest recall of 75%.
- SMOTE Random Forest achieved the highest ROC AUC of 82.50%.

The baseline model provided the strongest overall accuracy and F1 score among these experiments.

## 7. Hyperparameter Tuning

GridSearchCV with five-fold cross-validation was used to tune the Random Forest.

Best parameters:

- `n_estimators`: 200
- `max_depth`: 5
- `max_features`: sqrt

Results:

- Best cross-validation F1 score: 0.7408
- Test accuracy: 0.8315
- OOB score: 0.8214

The Decision Tree visualization is saved as `decision_tree_visualization.png`.

## 8. Regression Analysis

A Random Forest Regressor was trained to predict passenger fare.

Regression results:

- MAE: 15.6061
- RMSE: 45.4321
- R²: 0.2263
- Adjusted R²: 0.1992

The residual plot is saved as `regression_residual_plot.png`.

### Residual Plot and Heteroscedasticity Interpretation

The residual plot shows that residuals are not evenly distributed around zero across all predicted fare values. Their spread generally increases at higher predicted fares, indicating possible heteroscedasticity (non-constant error variance). Some large positive and negative residuals also suggest that the model struggles to predict certain passenger fares accurately. This may be related to the wide variation in ticket prices and the presence of high-fare outliers. Therefore, the regression model has limited predictive performance and may benefit from further feature engineering or alternative modelling techniques.

## 9. Saved Model

The final classification pipeline, including preprocessing and the trained Random Forest, is saved as:

`titanic_survival_pipeline.joblib`

The saved pipeline was successfully reloaded and tested using a new passenger input.

The example passenger received a predicted survival probability of 0.0946 and a predicted class of 0 (Did not survive).

## 10. How to Run

Run the scripts from the project root directory:

```bash
python analytics/titanic_analysis.py
python analytics/cleaning.py
python analytics/eda.py
python analytics/survival_analysis.py
python analytics/correlation.py
python analytics/multivariate.py
python analytics/standardization.py
python analytics/classification.py
python analytics/class_imbalance.py
python analytics/model_tuning.py
python analytics/regression.py
python analytics/final_model.py
python analytics/final_comparison.py
```

Install the required Python libraries using pip if they are not already installed.

## 11. Conclusion

The tuned Random Forest classifier achieved 83.15% test accuracy and was the strongest classification model based on the selected evaluation results. The regression model showed limited predictive performance, with an R² score of 0.2263. The complete classification pipeline was saved, reloaded and tested successfully. Overall, this module demonstrates data cleaning, exploratory analysis, visualization, classification, regression and model evaluation using the Titanic dataset.