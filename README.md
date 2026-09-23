# 🩺 Healthcare Diabetes Decision Support System

A machine learning-based Healthcare Decision Support System (DSS) for diabetes risk prediction and healthcare analytics.

## 🚀 Live Application

👉 [Open Healthcare Diabetes DSS](https://healthcare-diabetes-dss-cd9m4afdhd5vzjnpxzn8mp.streamlit.app/)

## 📌 Project Overview

This project develops a machine learning framework for diabetes-risk prediction using healthcare indicators.

Multiple machine learning algorithms are evaluated and compared, followed by a Decision Support System (DSS) layer that converts prediction probabilities into interpretable risk categories.

The final selected model is **CatBoost**.

## 🎯 Objectives

1. Analyze healthcare indicators and identify important factors associated with diabetes risk using exploratory and statistical analytics.

2. Develop and compare machine learning models for diabetes-risk prediction using appropriate preprocessing and evaluation metrics.

3. Design a Healthcare Decision Support System layer that converts model predictions into interpretable risk categories and analytics outputs.

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

### Input Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target

- `0` → No Diabetes
- `1` → Diabetes

## 🔧 Data Preprocessing

Healthcare indicator values recorded as zero for the following variables were treated as missing values:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

Median imputation was then applied.

The dataset was divided using an **80:20 stratified train-test split**.

## 🤖 Machine Learning Models

The following models were evaluated:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 70.78% | 58.82% | 55.56% | 57.14% | 0.8263 |
| Random Forest | 86.36% | 81.13% | 79.63% | 80.37% | 0.9446 |
| Extra Trees | 85.06% | 79.25% | 77.78% | 78.50% | 0.9309 |
| XGBoost | 88.31% | 82.14% | 85.19% | 83.64% | 0.9494 |
| LightGBM | 87.66% | 83.02% | 81.48% | 82.24% | 0.9524 |
| **CatBoost** | **88.96%** | **83.64%** | **85.19%** | **84.40%** | **0.9524** |

## 🏆 Final Model

**CatBoost** was selected as the final model for the deployed DSS prototype.

### Test Performance

- Accuracy: **88.96%**
- Precision: **83.64%**
- Recall: **85.19%**
- F1-Score: **84.40%**
- ROC-AUC: **0.9524**

### Confusion Matrix

| | Predicted No Diabetes | Predicted Diabetes |
|---|---:|---:|
| Actual No Diabetes | 91 | 9 |
| Actual Diabetes | 8 | 46 |

## 🧠 Decision Support System

The application converts the predicted diabetes probability into three prototype risk categories:

| Probability | Risk Category |
|---|---|
| `< 0.30` | Low Risk |
| `0.30 – < 0.60` | Moderate Risk |
| `≥ 0.60` | High Risk |

These thresholds are **project-design thresholds for an academic prototype and are not clinically validated**.

## 🔍 Explainability

The project uses:

- CatBoost feature importance
- SHAP analysis

to examine the contribution of healthcare indicators to model predictions.

## 💻 Streamlit Application

The deployed application allows users to enter healthcare indicators and obtain:

- Diabetes probability
- Diabetes prediction
- Prototype risk category
- Input summary
- Model performance information
- Decision-support interpretation

### Live Demo

👉 **[Healthcare Diabetes DSS](https://healthcare-diabetes-dss-cd9m4afdhd5vzjnpxzn8mp.streamlit.app/)**

## 📁 Project Structure

```text
healthcare-diabetes-dss/
│
├── app.py
├── healthcare_diabetes_catboost.pkl
├── feature_names.pkl
├── requirements.txt
└── README.md


