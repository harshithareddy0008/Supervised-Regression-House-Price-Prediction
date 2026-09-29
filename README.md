# End-to-End Supervised Regression — House Price Prediction

 *Live Demo:* [House Price Prediction App](https://supervised-regression-house-price.onrender.com/)

A portfolio-ready supervised machine-learning project that compares major regression algorithms on one reproducible housing dataset.

## What this project demonstrates

- Data understanding and exploratory analysis
- Missing-value handling
- Numeric scaling and categorical one-hot encoding
- Train/test split and pipelines
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression (degrees 2–4)
- Ridge, Lasso, ElasticNet
- K-Nearest Neighbors Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression
- Extra Trees Regression
- AdaBoost Regression
- Gradient Boosting Regression
- HistGradientBoosting Regression
- XGBoost Regression (when installed)
- CatBoost Regression (when installed)
- MAE, MSE, RMSE, R², Adjusted R², MAPE
- Cross-validation option
- Hyperparameter tuning with RandomizedSearchCV
- Permutation feature importance
- Model persistence with Joblib
- Command-line prediction
- Streamlit prediction app

## Dataset

The bundled dataset contains **6,000 synthetic housing records** created specifically for regression benchmarking. It contains numeric and categorical predictors, nonlinear effects, feature interactions, noise, and a small amount of missing predictor data.

This dataset is **synthetic**, not a real housing-market dataset. The generation is reproducible using `src/generate_dataset.py` with random seed 42.

## Folder structure

```text
Supervised_Regression_House_Price/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── housing_regression.csv
│   ├── processed/
│   └── README.md
├── models/
│   ├── best_model.joblib
│   └── best_model_name.txt
├── notebooks/
│   └── 01_complete_regression_project.ipynb
├── reports/
│   ├── figures/
│   │   ├── actual_vs_predicted.png
│   │   ├── linear_vs_polynomial.png
│   │   ├── model_rmse_comparison.png
│   │   └── permutation_feature_importance.png
│   ├── best_model_test_predictions.csv
│   ├── model_comparison.csv
│   ├── permutation_importance.csv
│   └── simple_polynomial_results.csv
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── explain_model.py
│   ├── generate_dataset.py
│   ├── linear_experiments.py
│   ├── predict.py
│   ├── train_models.py
│   └── tune_models.py
├── tests/
│   └── test_smoke.py
├── .gitignore
├── LICENSE
├── requirements.txt
├── run_project.bat
└── run_project.sh
```

## Verified benchmark results

The packaged project was executed successfully on the bundled dataset. Current top test-set results:

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Multiple Linear Regression | $34,525 | $44,395 | 0.9752 | 7.28% |
| Ridge Regression | $34,516 | $44,396 | 0.9752 | 7.27% |
| Lasso Regression | $34,511 | $44,449 | 0.9752 | 7.28% |
| CatBoost Regressor | $35,992 | $46,772 | 0.9725 | 7.71% |
| XGBoost Regressor | $37,185 | $48,043 | 0.9710 | 7.94% |

> These numbers are a reproducible educational benchmark for the bundled synthetic dataset; they are not real-market performance claims.

## Setup

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the complete project

Run the educational simple/polynomial experiment:

```bash
python src/linear_experiments.py
```

Train and compare the main regression models:

```bash
python src/train_models.py
```

Generate feature-importance analysis:

```bash
python src/explain_model.py
```

Run a sample prediction:

```bash
python src/predict.py
```

Or, on Windows, run:

```bash
run_project.bat
```

## Launch the Streamlit app

```bash
streamlit run app/app.py
```

## Hyperparameter tuning

Random Forest:

```bash
python src/tune_models.py --model random_forest
```

Gradient Boosting:

```bash
python src/tune_models.py --model gradient_boosting
```

## Why Simple, Multiple and Polynomial Regression are separated

**Simple Linear Regression** intentionally uses only `Area_sqft` so the relationship `x -> y` is easy to visualize and explain.

**Polynomial Regression** uses the same one-dimensional feature with polynomial degrees 2–4, making nonlinear behavior visible without creating thousands of polynomial interaction columns.

**Multiple Linear Regression** uses all numeric and categorical predictors through a preprocessing pipeline and is included in the full model comparison.

## Metrics

- **MAE**: average absolute prediction error.
- **MSE**: squared-error average.
- **RMSE**: error in the same units as the target; lower is better.
- **R²**: proportion of target variance explained; higher is better.
- **Adjusted R²**: R² adjusted for the transformed feature count; most interpretable for linear-model discussion.
- **MAPE**: average absolute percentage error.

## Interview explanation

A concise explanation:

> I built an end-to-end supervised regression project for house-price prediction. I created a reproducible dataset with numeric and categorical features, missing values, nonlinear relationships and noise. I built preprocessing pipelines for imputation, scaling and one-hot encoding; implemented simple, multiple and polynomial linear regression; regularized models such as Ridge, Lasso and ElasticNet; nonlinear models such as KNN, SVR and decision trees; and ensemble/boosting methods including Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost and CatBoost. I compared the models using MAE, RMSE, R² and MAPE, persisted the best model, analyzed feature importance and exposed predictions through a Streamlit app.

## Important discussion points for interviews

1. Why scaling matters for KNN, SVR and regularized linear models.
2. Why tree models generally do not require feature scaling.
3. Difference between L1 and L2 regularization.
4. Why polynomial regression may overfit at high degree.
5. Bias–variance trade-off.
6. Why Random Forest uses bagging while Gradient Boosting/XGBoost use boosting.
7. Why a train/test split alone is not enough for reliable model selection.
8. Why preprocessing must be inside a Pipeline to prevent data leakage.
9. Difference between MAE and RMSE.
10. Why synthetic data should not be represented as real production market data.

## Reproducibility

Dataset seed: `42`  
Train/test split seed: `42`

## License

Code is provided under the MIT License. The bundled dataset is generated by this project's own reproducible data-generation script for educational use.
