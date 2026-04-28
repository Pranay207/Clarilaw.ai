import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# COMMON UTILS
# =========================

def save_report(report, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =========================
# ML MODEL EVALUATION
# =========================

def evaluate_ml_model(model_path, X_test, y_test, label_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    save_report(report, f"{output_dir}/metrics.txt")
    save_confusion_matrix(
        cm,
        label_names,
        f"{output_dir}/confusion_matrix.png"
    )

    print("✅ ML Model Evaluation Complete")
    print(f"Accuracy: {acc:.4f}")


# =========================
# TRANSFORMER EVALUATION
# =========================

def evaluate_transformer(
    model_dir,
    texts,
    labels,
    label_names,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    preds = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            preds.append(pred)

    preds = np.array(preds)

    acc = accuracy_score(labels, preds)
    report = classification_report(
        labels,
        preds,
        target_names=label_names,
        zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    save_report(report, f"{output_dir}/metrics.txt")
    save_confusion_matrix(
        cm,
        label_names,
        f"{output_dir}/confusion_matrix.png"
    )

    print("✅ Transformer Evaluation Complete")
    print(f"Accuracy: {acc:.4f}")


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    """
    Example only – normally called from notebooks or pipelines
    """
    pass

