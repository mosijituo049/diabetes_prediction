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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import optuna
from sklearn.model_selection import cross_val_score

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

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    dump(X_train, MODEL_DIR/"X_train.joblib")
    dump(X_train_smote, MODEL_DIR/"X_train_smote.joblib")
    dump(X_test, MODEL_DIR/"X_test.joblib")
    dump(y_train_smote, MODEL_DIR/"y_train_smote.joblib")
    dump(y_test, MODEL_DIR/"y_test.joblib")

def train_model(model):
    TRAIN_DATA_FILES = [
        "X_train.joblib",
        "X_train_smote.joblib",
        "X_test.joblib",
        "y_train.joblib",
        "y_train_smote.joblib",
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
    X_train_smote = load(MODEL_DIR/"X_train_smote.joblib")
    y_train_smote = load(MODEL_DIR/"y_train_smote.joblib")

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

            #scoring='roc_auc',
            scoring='f1',
            
            n_jobs=-1,

            random_state=42
        )

        random_search.fit(X_train, y_train)

        best_rf = random_search.best_estimator_

        best_params = random_search.best_params_

        best_rf_balanced = RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=42
        )

        best_rf_balanced.fit(
            X_train,
            y_train
        )

        best_rf_smote = RandomForestClassifier(
            **best_params,
            random_state=42
        )

        best_rf_smote.fit(
            X_train_smote,
            y_train_smote
        )

        dump(best_rf, MODEL_DIR/"tuned_random_forest.joblib")
        dump(best_rf_balanced , MODEL_DIR/"tuned_random_forest_balanced.joblib")
        dump(best_rf_smote , MODEL_DIR/"tuned_random_forest_smote.joblib")
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
    
    #knn
    if model == "knn":
        print("trainning model knn...")
        knn = KNeighborsClassifier(n_neighbors =10, n_jobs=-1)
        knn.fit(X_train, y_train)
        dump(knn, MODEL_DIR/"knn.joblib")
        print("model is ready!")

    #tuned knn
    if model == "tuned_knn":
        print("trainning model tuned_knn...")
        # Define the best hyperparameter grid: with 'weights': ['uniform', 'distance'] it takes 960sec
        param_grid = {
            'n_neighbors': [5,7,9,11,13,15]
            #'weights': ['uniform','distance'],
            #'metric': ['euclidean','manhattan']
        }

        # Run Grid Search
        gs = GridSearchCV(
            estimator=KNeighborsClassifier(n_neighbors =10, n_jobs=-1),
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=1,
            verbose = 10
        )

        gs.fit(X_train, y_train)

        tuned_knn = gs.best_estimator_

        dump(tuned_knn, MODEL_DIR/"tuned_knn.joblib")
        dump(gs, MODEL_DIR/"gs.joblib")
        print("model is ready!")

def objective(trial):
    params = {

        "n_estimators":
        trial.suggest_int(
            "n_estimators",
            100,
            1000
        ),

        "max_depth":
        trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate":
        trial.suggest_float(
            "learning_rate",
            0.01,
            0.3
        ),

        "subsample":
        trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree":
        trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        )
    }

    model = XGBClassifier(
        **params,
        random_state=42
    )

    X_train = load(MODEL_DIR/"X_train.joblib")
    y_train = load(MODEL_DIR/"y_train.joblib")

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='f1'
    ).mean()

    return score

def xgb_optuna():
    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=50
    )

    best_xgb = XGBClassifier(
        **study.best_params,
        random_state=42
    )

    X_train = load(MODEL_DIR/"X_train.joblib")
    y_train = load(MODEL_DIR/"y_train.joblib")

    best_xgb.fit(
        X_train,
        y_train
    )

    dump(best_xgb , MODEL_DIR/"tuned_xgb.joblib")
    dump(study , MODEL_DIR/"study.joblib")

if __name__ == "__main__":
    MODEL_NAME = [
        "logistic_regression",
        "random_forest",
        "turned_random_forest",
        "xgboost"
    ]

    for model in MODEL_NAME:
        train_model(model)
    
