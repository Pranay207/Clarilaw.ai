from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.inference.load_model import load_intent_model, load_vectorizer

router = APIRouter()

model = None
vectorizer = None
load_error = None


def _get_model_and_vectorizer():
    global model, vectorizer, load_error
    if model is not None and vectorizer is not None:
        return model, vectorizer

    if load_error is not None:
        raise RuntimeError(load_error)

    try:
        model = load_intent_model()
        vectorizer = load_vectorizer()
        return model, vectorizer
    except Exception as exc:
        load_error = str(exc)
        raise RuntimeError(load_error) from exc


class TextRequest(BaseModel):
    text: str


@router.post("/classify")
def classify_text(request: TextRequest):
    text = request.text

    try:
        model_obj, vectorizer_obj = _get_model_and_vectorizer()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Classification model unavailable: {exc}") from exc

    vector = vectorizer_obj.transform([text])
    prediction = model_obj.predict(vector)[0]
    proba = model_obj.predict_proba(vector)

    confidence = float(max(proba[0]))

    return {
        "intent": prediction,
        "confidence": round(confidence, 2)
    }
