import inspect
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "dl_training_data.pt"
MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = PROJECT_ROOT / "models" / "transformer_intent_classifier"


def tokenize_function(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train_transformer(texts, labels, num_labels):
    texts = ["" if t is None else str(t) for t in texts]
    labels = [int(label) for label in labels]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    for param in model.base_model.parameters():
        param.requires_grad = False

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    args_kwargs = {
        "output_dir": "models/checkpoints",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 6,
        "save_strategy": "no",
        "logging_steps": 50,
        "report_to": "none",
    }

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        args_kwargs["eval_strategy"] = "no"
    else:
        args_kwargs["evaluation_strategy"] = "no"

    training_args = TrainingArguments(**args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics,
    }

    if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("Training transformer model...")
    trainer.train()

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("OK: Transformer model saved successfully.")
    print(f"Saved at: {SAVE_PATH}")


if __name__ == "__main__":
    assert os.path.exists(DATA_PATH), f"File not found: {DATA_PATH}"

    payload = torch.load(DATA_PATH, weights_only=False)
    if isinstance(payload, dict):
        texts = payload["texts"]
        labels = payload["labels"]
        label_names = payload.get("label_names", [])
        num_labels = len(label_names) if label_names else len(set(labels))
    else:
        texts, labels = payload
        num_labels = len(set(labels))

    train_transformer(
        texts=texts,
        labels=labels,
        num_labels=num_labels,
    )
