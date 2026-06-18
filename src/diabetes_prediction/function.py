from joblib import load
import pandas as pd
from pathlib import Path
import diabetes_prediction.train as train
from pickle import UnpicklingError

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

def load_train_data():
    print("loading train data...")

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
        train.prepare_train_data("diabetes_prediction_dataset.csv")

    X_train = load(MODEL_DIR/"X_train.joblib")
    y_train = load(MODEL_DIR/"y_train.joblib")
    X_test = load(MODEL_DIR/"X_test.joblib")
    y_test = load(MODEL_DIR/"y_test.joblib")

    train_data = {
        "X_train":X_train,
        "y_train":y_train,
        "X_test":X_test,
        "y_test":y_test
    }

    print("train data loaded!")

    return train_data

def load_models():
    MODEL_NAME = [
        "logistic_regression",
        "random_forest",
        "tuned_random_forest",
        "xgboost"
    ]

    models = dict()

    print("loading models...")

    for model in MODEL_NAME:

        model_path = MODEL_DIR / f"{model}.joblib"

        try:
            models[model] = load(model_path)

        except (
            FileNotFoundError,
            EOFError,
            UnpicklingError
        ):
            print(
                f"{model} not found or corrupted. training..."
            )

            train.train_model(model)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"training finished but {model_path} was not created."
                )

            models[model] = load(model_path)

    print("all models loaded!")

    return models

if __name__ == "__main__":
    print()