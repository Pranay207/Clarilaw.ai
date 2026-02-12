from fastapi import APIRouter
from pydantic import BaseModel
from src.inference.load_model import load_intent_model, load_vectorizer

router = APIRouter()

model = load_intent_model()
vectorizer = load_vectorizer()


class TextRequest(BaseModel):
    text: str


@router.post("/classify")
def classify_text(request: TextRequest):
    text = request.text

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)

    confidence = float(max(proba[0]))

    return {
        "intent": prediction,
        "confidence": round(confidence, 2)
    }
