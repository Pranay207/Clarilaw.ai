# src/inference/load_model.py

import warnings
from pathlib import Path

import joblib
import torch
from sklearn.exceptions import InconsistentVersionWarning

MODEL_PATH = "models/intent_classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
SUMMARIZER_PATH = "models/summarizer.pt"


def _resolve_path(path: str) -> Path:
    # Resolve relative to project root (two levels up from this file).
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / path).resolve()


def _load_sklearn_artifact(path: Path, label: str):
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            f"{label} file is missing or empty: '{path}'. "
            "Regenerate it by running: python -m src.models.train_ml_models"
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {label.lower()} from '{path}'. "
            "Re-train with your current scikit-learn version."
        ) from exc


def load_intent_model():
    return _load_sklearn_artifact(_resolve_path(MODEL_PATH), "Intent model")


def load_vectorizer():
    return _load_sklearn_artifact(_resolve_path(VECTORIZER_PATH), "Vectorizer")


def load_summarizer():
    summarizer_path = _resolve_path(SUMMARIZER_PATH)
    if not summarizer_path.exists() or summarizer_path.stat().st_size == 0:
        raise RuntimeError(
            f"Summarizer file is missing or empty: '{summarizer_path}'. "
            "Regenerate it with your training pipeline."
        )
    try:
        return torch.load(summarizer_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load summarizer from '{summarizer_path}'. "
            "Check that the file exists and matches your torch version."
        ) from exc


def load_all_models():
    intent_model = load_intent_model()
    vectorizer = load_vectorizer()

    summarizer = None
    try:
        summarizer = load_summarizer()
    except RuntimeError:
        pass

    return intent_model, vectorizer, summarizer
