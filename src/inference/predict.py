from pathlib import Path
import sys
import re

# Allow running this file directly by ensuring the project root is on sys.path
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.load_model import (
    load_intent_model,
    load_vectorizer,
)

intent_model = None
vectorizer = None
load_error = None


def _get_model_and_vectorizer():
    global intent_model, vectorizer, load_error
    if intent_model is not None and vectorizer is not None:
        return intent_model, vectorizer

    if load_error is not None:
        raise RuntimeError(load_error)

    try:
        intent_model = load_intent_model()
        vectorizer = load_vectorizer()
        return intent_model, vectorizer
    except Exception as exc:
        load_error = str(exc)
        raise RuntimeError(load_error) from exc

# =========================
# Intent label mapping
# =========================
LABEL_MAP = {
    0: "legal_query",
    1: "general",
    2: "other",
}

# =========================
# Text preprocessing
# =========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.strip()

# =========================
# Single intent prediction
# =========================
def predict_intent(text: str):
    model_obj, vectorizer_obj = _get_model_and_vectorizer()
    cleaned_text = clean_text(text)

    X = vectorizer_obj.transform([cleaned_text])

    probabilities = model_obj.predict_proba(X)[0]
    class_index = probabilities.argmax()

    intent = LABEL_MAP.get(class_index, "unknown")
    confidence = float(probabilities[class_index])

    return {
        "intent": intent,
        "confidence": round(confidence, 3)
    }

# =========================
# Batch intent prediction
# =========================
def predict_batch_intent(texts: list):
    model_obj, vectorizer_obj = _get_model_and_vectorizer()
    cleaned_texts = [clean_text(t) for t in texts]
    X = vectorizer_obj.transform(cleaned_texts)

    probabilities = model_obj.predict_proba(X)
    class_indices = probabilities.argmax(axis=1)

    results = []
    for idx, probs in zip(class_indices, probabilities):
        results.append({
            "intent": LABEL_MAP[idx],
            "confidence": round(float(probs[idx]), 3)
        })

    return results

# =========================
# Summary prediction (placeholder)
# =========================
def predict_summary(text: str):
    return {
        "summary": text[:200] + "..."
    }

def predict_batch_summary(texts: list):
    return [
        {"summary": text[:200] + "..."}
        for text in texts
    ]

# =========================
# Chat prediction (placeholder)
# =========================
def predict_chat_response(query: str):
    return {
        "reply": "This feature will answer legal queries in future versions."
    }

def predict_batch_chat_responses(queries: list):
    return [
        {"reply": "This feature will answer legal queries in future versions."}
        for _ in queries
    ]

# =========================
# Combined prediction
# =========================
def predict_all(text: str):
    return {
        "intent": predict_intent(text),
        "summary": predict_summary(text),
        "chat_response": predict_chat_response(text)
    }

def predict_batch_all(texts: list):
    results = []
    for text in texts:
        results.append({
            "intent": predict_intent(text),
            "summary": predict_summary(text),
            "chat_response": predict_chat_response(text)
        })
    return results
