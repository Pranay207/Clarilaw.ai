from pathlib import Path
import re
import sys

# Allow running this file directly by ensuring the project root is on sys.path.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.load_model import load_intent_model, load_vectorizer


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


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.strip()


def predict_intent(text: str):
    model_obj, vectorizer_obj = _get_model_and_vectorizer()
    cleaned_text = clean_text(text)

    X = vectorizer_obj.transform([cleaned_text])
    prediction = model_obj.predict(X)[0]
    probabilities = model_obj.predict_proba(X)[0]
    class_index = int(probabilities.argmax())
    confidence = float(probabilities[class_index])

    return {
        "intent": str(prediction),
        "confidence": round(confidence, 3),
    }


def predict_batch_intent(texts: list):
    model_obj, vectorizer_obj = _get_model_and_vectorizer()
    cleaned_texts = [clean_text(t) for t in texts]
    X = vectorizer_obj.transform(cleaned_texts)

    predictions = model_obj.predict(X)
    probabilities = model_obj.predict_proba(X)

    results = []
    for prediction, probs in zip(predictions, probabilities):
        class_index = int(probs.argmax())
        results.append(
            {
                "intent": str(prediction),
                "confidence": round(float(probs[class_index]), 3),
            }
        )

    return results


def predict_summary(text: str):
    return {
        "summary": text[:200] + "...",
    }


def predict_batch_summary(texts: list):
    return [{"summary": text[:200] + "..."} for text in texts]


def predict_chat_response(query: str):
    return {
        "reply": "This feature will answer legal queries in future versions.",
    }


def predict_batch_chat_responses(queries: list):
    return [
        {"reply": "This feature will answer legal queries in future versions."}
        for _ in queries
    ]


def predict_all(text: str):
    return {
        "intent": predict_intent(text),
        "summary": predict_summary(text),
        "chat_response": predict_chat_response(text),
    }


def predict_batch_all(texts: list):
    results = []
    for text in texts:
        results.append(
            {
                "intent": predict_intent(text),
                "summary": predict_summary(text),
                "chat_response": predict_chat_response(text),
            }
        )
    return results
