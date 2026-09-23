# Healthcare Diabetes Decision Support System

A machine learning-based Healthcare Decision Support System for diabetes risk prediction and healthcare analytics.

## Project Overview

This project develops a machine learning framework for early diabetes-risk prediction using healthcare indicators. Multiple machine learning algorithms are evaluated and compared, followed by a Decision Support System (DSS) layer.

The final model uses CatBoost for diabetes prediction.

## Objectives

1. Analyze healthcare indicators and identify important factors associated with diabetes risk using exploratory and statistical analytics.

2. Develop and compare machine learning models for early diabetes-risk prediction using appropriate preprocessing and evaluation metrics.

3. Design a Healthcare Decision Support System layer that converts model predictions into interpretable risk categories and analytics outputs.

## Dataset

The project uses the Pima Indians Diabetes Dataset.

### Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target

- 0 = No Diabetes
- 1 = Diabetes

## Data Preprocessing

Healthcare indicator values recorded as zero for Glucose, BloodPressure, SkinThickness, Insulin, and BMI were treated as missing values.

Missing values were replaced using median imputation.

The dataset was divided into training and testing sets using an 80:20 stratified split.

## Machine Learning Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 70.78% | 58.82% | 55.56% | 57.14% | 0.8263 |
| Random Forest | 86.36% | 81.13% | 79.63% | 80.37% | 0.9446 |
| Extra Trees | 85.06% | 79.25% | 77.78% | 78.50% | 0.9309 |
| XGBoost | 88.31% | 82.14% | 85.19% | 83.64% | 0.9494 |
| LightGBM | 87.66% | 83.02% | 81.48% | 82.24% | 0.9524 |
| CatBoost | 88.96% | 83.64% | 85.19% | 84.40% | 0.9524 |

## Final Model

The final selected model is CatBoost.

### Test Set Performance

- Accuracy: 88.96%
- Precision: 83.64%
- Recall: 85.19%
- F1-Score: 84.40%
- ROC-AUC: 0.9524

### Confusion Matrix

- True Negative: 91
- False Positive: 9
- False Negative: 8
- True Positive: 46

## Decision Support System

The DSS converts predicted diabetes probability into three prototype risk categories:

- Low Risk: probability < 0.30
- Moderate Risk: probability 0.30 to < 0.60
- High Risk: probability >= 0.60

These thresholds are project-design thresholds for an academic prototype and are not clinically validated.

## Explainability

CatBoost feature importance and SHAP analysis are used to understand the contribution of healthcare indicators to model predictions.

## Streamlit Application

The project includes a Streamlit dashboard where users can enter healthcare indicators and receive:

- Diabetes probability
- Prediction
- Prototype risk category
- Input summary
- Model performance information
- Decision-support interpretation

## Project Structure

```text
healthcare-diabetes-dss/
│
├── app.py
├── healthcare_diabetes_catboost.pkl
├── feature_names.pkl
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/healthcare-diabetes-dss.git
cd healthcare-diabetes-dss
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Requirements

- Python
- Streamlit
- CatBoost
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Important Note

This project is an academic prototype for machine learning and healthcare decision-support research.

It is NOT a medical diagnosis system and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Author

Sangram Dey

M.Tech in Artificial Intelligence and Data Science

Alliance University, Bangalore, India
