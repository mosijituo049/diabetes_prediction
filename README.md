# Diabetes Prediction Using Machine Learning

A machine learning project for predicting diabetes risk using demographic and health-related indicators. This project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, hyperparameter optimization, model evaluation, and feature importance analysis.

---

## Project Overview

The objective of this project is to develop and compare multiple machine learning models for diabetes prediction and identify the most effective approach based on classification performance.

The project includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Multiple machine learning models
* Hyperparameter optimization
* Class imbalance handling
* Model evaluation and comparison
* Feature importance analysis

---

## Key Results

### Final Selected Model: Tuned Random Forest

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **97.22%** |
| Precision | **98.99%** |
| Recall    | 69.16%     |
| F1 Score  | **81.43%** |
| ROC-AUC   | 97.27%     |

Although Tuned XGBoost achieved a slightly higher ROC-AUC score (97.72%), Tuned Random Forest was selected as the final model because it achieved the highest Accuracy, Precision, and F1-score, providing the best overall balance of classification performance.

---

## Dataset

The dataset contains demographic and health-related factors commonly associated with diabetes risk.

### Features

| Feature             | Description              |
| ------------------- | ------------------------ |
| gender              | Gender of the individual |
| age                 | Age in years             |
| hypertension        | Hypertension diagnosis   |
| heart_disease       | Heart disease diagnosis  |
| smoking_history     | Smoking status           |
| bmi                 | Body Mass Index          |
| HbA1c_level         | Hemoglobin A1c level     |
| blood_glucose_level | Blood glucose level      |

### Target Variable

| Variable | Description                          |
| -------- | ------------------------------------ |
| diabetes | Diabetes diagnosis (0 = No, 1 = Yes) |

---

## Project Structure

```text
diabetes_prediction/
│
├── data/
│   ├── raw/
│   └── clean/
│
├── models/
│
├── notebooks/
│   ├── zhiwen_EDA.ipynb
│   ├── models_evaluation.ipynb
│   └── zidene.ipynb
│
├── sql_scripts/
│
├── src/
│   └── diabetes_prediction/
│       ├── __init__.py
│       ├── function.py
│       └── train.py
│
├── config.yaml
├── main.py
├── pyproject.toml
└── README.md
```

---

## Data Preprocessing

The following preprocessing steps were performed before model training:

* Removed duplicate records
* Removed rows with rare "Other" gender category
* One-hot encoded categorical features
* Standardized numerical features
* Stratified train-test split
* Applied SMOTE oversampling for imbalance experiments

---

## Exploratory Data Analysis (EDA)

EDA was conducted to better understand feature distributions and their relationships with diabetes.

The analysis included:

* Class distribution analysis
* Age distribution analysis
* BMI distribution analysis
* Blood glucose level analysis
* HbA1c level analysis
* Diabetes prevalence by gender
* Diabetes prevalence by smoking history
* Correlation analysis

---

## Machine Learning Models

The following classification models were implemented and evaluated:

* Logistic Regression
* Random Forest
* Tuned Random Forest
* Tuned Random Forest (Class Balanced)
* Tuned Random Forest (SMOTE)
* XGBoost
* Tuned XGBoost
* K-Nearest Neighbors (KNN)
* Tuned KNN

---

## Hyperparameter Optimization

Different optimization strategies were applied depending on the model.

| Model         | Optimization Method |
| ------------- | ------------------- |
| Random Forest | RandomizedSearchCV  |
| KNN           | GridSearchCV        |
| XGBoost       | Optuna              |

### Random Forest Parameters

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf
* max_features

### XGBoost Parameters

* n_estimators
* learning_rate
* max_depth
* subsample
* colsample_bytree

### KNN Parameters

* n_neighbors
* weights
* metric

---

## Handling Class Imbalance

To investigate the impact of class imbalance on diabetes prediction, three Random Forest variants were evaluated:

1. Tuned Random Forest
2. Tuned Random Forest with Class Weights
3. Tuned Random Forest with SMOTE Oversampling

### Findings

* Class weighting and SMOTE substantially increased recall.
* However, they reduced precision and overall accuracy.
* The standard Tuned Random Forest provided the best overall balance across evaluation metrics and was therefore selected as the final model.

---

## Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve

---

## Performance Comparison

| Model               | Accuracy   | Precision  | Recall | F1 Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ------ | ---------- | ---------- |
| Tuned RF            | **0.9722** | **0.9899** | 0.6916 | **0.8143** | 0.9727     |
| Tuned XGBoost       | 0.9714     | 0.9736     | 0.6952 | 0.8111     | **0.9772** |
| XGBoost             | 0.9706     | 0.9520     | 0.7022 | 0.8083     | 0.9754     |
| Random Forest       | 0.9695     | 0.9419     | 0.6975 | 0.8015     | 0.9569     |
| Logistic Regression | 0.9588     | 0.8587     | 0.6380 | 0.7321     | 0.9595     |
| Tuned RF Balanced   | 0.9288     | 0.5656     | 0.8337 | 0.6740     | 0.9730     |
| Tuned RF SMOTE      | 0.9287     | 0.5648     | 0.8379 | 0.6747     | 0.9726     |
| Tuned KNN           | 0.9586     | 0.8750     | 0.6191 | 0.7251     | 0.9016     |
| KNN                 | 0.9602     | 0.9550     | 0.5761 | 0.7186     | 0.9283     |

### Key Findings

* Tuned Random Forest achieved the highest Accuracy, Precision, and F1-score.
* Tuned XGBoost achieved the highest ROC-AUC score.
* Class balancing and SMOTE significantly improved recall but reduced precision.
* Tree-based ensemble methods consistently outperformed Logistic Regression and KNN models.
* HbA1c level and blood glucose level were the strongest predictors of diabetes.

---

## Feature Importance Analysis

Feature importance was extracted from the best-performing tree-based model.

The most influential predictors were:

1. HbA1c Level
2. Blood Glucose Level
3. Age
4. BMI
5. Hypertension

These findings are consistent with established clinical risk factors for diabetes.

---

## Reproducing Results

### Clone the Repository

```bash
git clone https://github.com/mosijituo049/diabetes_prediction.git
cd diabetes_prediction
```

### Install Dependencies

Using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python src/diabetes_prediction/train.py
```

or

```bash
python main.py
```

The training pipeline will:

1. Load and preprocess the data
2. Train machine learning models
3. Perform hyperparameter optimization
4. Evaluate model performance
5. Save trained models to the `models/` directory

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Optuna
* Imbalanced-Learn (SMOTE)
* Matplotlib
* Seaborn
* Jupyter Notebook
* Joblib

---

## Future Improvements

* Deploy the model using Streamlit or FastAPI
* Implement experiment tracking with MLflow
* Explore advanced ensemble techniques
* Investigate cost-sensitive learning methods
* Build an end-to-end prediction API

---

## Authors

* Zidene
* Zhiwen
