import os
from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.model_utils import create_vectorizer, save_pickle, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_legal_texts.csv"
MODEL_DIR = PROJECT_ROOT / "models"
TEXT_COL = "cleaned_text"
LABEL_COL = "category"
PRIMARY_MODEL_PATH = MODEL_DIR / "intent_classifier.pkl"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    print("Accuracy:", accuracy)
    print(report)

    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
    }


def main():
    print("Loading data:", DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    print("\nCategory distribution:")
    print(df[LABEL_COL].value_counts())

    X = df[TEXT_COL]
    y = df[LABEL_COL]

    vectorizer = create_vectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = split_data(X_vec, y)

    print("\nTraining Logistic Regression...")
    lr_result = train_and_evaluate(
        LogisticRegression(max_iter=1000),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    save_pickle(lr_result["model"], LOGISTIC_MODEL_PATH)

    print("\nTraining Random Forest...")
    rf_result = train_and_evaluate(
        RandomForestClassifier(n_estimators=200, random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    save_pickle(rf_result["model"], RANDOM_FOREST_MODEL_PATH)

    best_model_name, best_result = max(
        (
            ("logistic_regression", lr_result),
            ("random_forest", rf_result),
        ),
        key=lambda item: item[1]["accuracy"],
    )

    save_pickle(best_result["model"], PRIMARY_MODEL_PATH)
    save_pickle(vectorizer, VECTORIZER_PATH)

    print(
        f"\nBest model saved to {PRIMARY_MODEL_PATH.name}: "
        f"{best_model_name} ({best_result['accuracy']:.4f})"
    )
    print("\nTRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
