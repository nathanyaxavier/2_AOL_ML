# PADDY YIELD PREDICTION


# Rice Yield Prediction using Optimized Random Forest

A Machine Learning application for predicting rice yield based on agricultural, environmental, and cultivation factors. This project compares multiple regression algorithms and deploys the best-performing model using Streamlit for interactive prediction.

---

## Project Overview

Accurate rice yield prediction is essential for agricultural planning, fertilizer management, and improving farming productivity. This project develops a machine learning model capable of estimating rice yield from cultivation parameters.

Three regression algorithms were evaluated:

- Linear Regression (Baseline)
- K-Nearest Neighbors (KNN) Regressor
- Optimized Random Forest Regressor

Among them, the Optimized Random Forest achieved the best predictive performance and was selected for deployment.

---

## Objectives

- Predict rice yield from agricultural input variables.
- Compare multiple regression algorithms.
- Optimize model performance through hyperparameter tuning.
- Deploy the best model as an interactive Streamlit application.

---

## Dataset

**Source**

UCI Machine Learning Repository – Paddy Crop Dataset

After preprocessing:

- Total samples: **2,338**
- Total original features: **45**
- Missing values: **0**
- Duplicate records removed

---

## Feature Selection

Feature importance was calculated using a baseline Random Forest model.

The final model uses the most important agricultural features:

- Hectares
- Variety
- Soil Types
- Seedrate (Kg)
- Urea_40Days
- Potassh_50Days
- 30DRain (mm)
- Relative Humidity_D1_D30

These features were selected because they contributed most significantly to rice yield prediction while keeping the model efficient.

---

## Machine Learning Pipeline

1. Data Cleaning
2. Duplicate Removal
3. Feature Selection
4. One-Hot Encoding
5. Robust Scaling
6. Train-Test Split (80:20)
7. Hyperparameter Tuning
8. 5-Fold Cross Validation
9. Model Evaluation
10. Streamlit Deployment

---

## Models Compared

Linear Regression: Baseline model 
KNN Regressor: Distance-based regression 
Random Forest: Ensemble learning model 

### Hyperparameter Tuning

### Random Forest

- n_estimators = 50, 100, 150
- max_depth = None, 10, 20

Best Parameters:

```text
n_estimators = 150
max_depth = None
```

### KNN

- n_neighbors = 3, 5, 7
- weights = uniform, distance

Best Parameters:

```text
n_neighbors = 7
weights = uniform
```

---

## Model Performance

| Model | MAE (kg) | RMSE (kg) | P90 Error (kg) | R² |
|--------|----------:|----------:|---------------:|----:|
| Optimized Random Forest | **657.62** | **912.78** | **1426.74** | **0.9903** |
| Optimized KNN | 682.38 | 937.67 | 1458.57 | 0.9897 |
| Linear Regression | 762.88 | 1021.45 | 1584.16 | 0.9878 |

---

## Overfitting Check

The gap between Train R² and Test R² was analyzed to verify the model's generalization capability.

| Model | Train R² | Test R² | Gap |
|--------|----------:|---------:|----:|
| Random Forest | 0.9922 | 0.9903 | 0.0019 |
| KNN | 0.9916 | 0.9897 | 0.0019 |
| Linear Regression | 0.9893 | 0.9878 | 0.0015 |

All models achieved gaps below **0.01**, indicating no significant overfitting.

---

## Streamlit Application

The deployed application allows users to:

- Input agricultural parameters
- Predict total rice yield
- View productivity (ton/ha)
- Read prediction interpretation
- Obtain recommendations based on prediction results

---

## Project Structure

```
Rice-Yield-Prediction/
│
├── app.py
├── random_forest_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
└── images/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/nathanyaxavier/2_AOL_ML.git
```

Move into the project directory

```bash
cd Rice-Yield-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Requirements

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

Install all packages using

```bash
pip install -r requirements.txt
```

---

## User Evaluation

The application was evaluated by five independent users.

### Average Ratings

| Evaluation Aspect | Score |
|-------------------|------:|
| Ease of Use | 4.8 / 5 |
| Interface Clarity | 4.6 / 5 |
| Relevant Prediction Factors | 4.6 / 5 |
| Prediction Clarity | 4.2 / 5 |
| Supports Farming Planning | 4.4 / 5 |
| Long-term Usefulness | 4.6 / 5 |

Overall feedback indicated that the application is easy to use, provides relevant prediction factors, and has potential to support rice cultivation planning.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## License

This project was developed for academic purposes.
