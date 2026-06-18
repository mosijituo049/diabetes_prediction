import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
from pathlib import Path
from joblib import load

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
RAW_DATA_DIR = BASE_DIR / "data/raw"

def prepare_train_data(file):
    df = pd.read_csv(RAW_DATA_DIR/file)
    df = df.drop_duplicates()
    #there are only 18 rows of other gender, drop then
    df = df[df['gender'] != 'Other']

    #Onehot encoding
    df_encoded = pd.get_dummies(
        df,
        columns=['gender','smoking_history'],
        drop_first=True,
        dtype=int
    )

    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #standarize numerical columns
    num_cols = [
        'age',
        'bmi',
        'HbA1c_level',
        'blood_glucose_level'
    ]

    scaler = StandardScaler()

    X_train[num_cols] = scaler.fit_transform(
        X_train[num_cols]
    )

    X_test[num_cols] = scaler.transform(
        X_test[num_cols]
    )

    dump(X_train, MODEL_DIR/"X_train.joblib")
    dump(X_test, MODEL_DIR/"X_test.joblib")
    dump(y_train, MODEL_DIR/"y_train.joblib")
    dump(y_test, MODEL_DIR/"y_test.joblib")

def train_model(model):
    TRAIN_DATA_FILES = [
        "X_train.joblib",
        "X_test.joblib",
        "y_train.joblib",
        "y_test.joblib"
    ]

    missing = any(
        not (MODEL_DIR / f).exists()
        for f in TRAIN_DATA_FILES
    )

    if missing:
        prepare_train_data("diabetes_prediction_dataset.csv")
        
    X_train = load(MODEL_DIR/"X_train.joblib")
    y_train = load(MODEL_DIR/"y_train.joblib")

    #logistic regression
    if model == "logistic_regression":
        print("trainning model logistic regression...")
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        dump(lr, MODEL_DIR/"logistic_regression.joblib")
        print("model is ready!")

    #random forest
    if model == "random_forest":
        print("trainning model random forest...")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        dump(rf, MODEL_DIR/"random_forest.joblib")
        print("model is ready!")

    #turned random forest
    if model == "tuned_random_forest":
        print("trainning model tuned random forest...")
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),

            param_distributions=param_dist,

            n_iter=30,

            cv=5,

            scoring='roc_auc',

            n_jobs=-1,

            random_state=42
        )

        random_search.fit(X_train, y_train)

        best_rf = random_search.best_estimator_

        dump(best_rf, MODEL_DIR/"tuned_random_forest.joblib")
        dump(random_search, MODEL_DIR/"random_search.joblib")
        print("model is ready!")

    #xgboost
    if model == "xgboost":
        print("trainning model xgboost...")
        xgb = XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(X_train, y_train)
        dump(xgb, MODEL_DIR/"xgboost.joblib")
        print("model is ready!")

if __name__ == "__main__":
    MODEL_NAME = [
        "logistic_regression",
        "random_forest",
        "turned_random_forest",
        "xgboost"
    ]

    for model in MODEL_NAME:
        train_model(model)
    