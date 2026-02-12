from pathlib import Path
import sys

# Allow running this file directly by ensuring the project root is on sys.path.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.predict import predict_intent


def classify_text(text: str):
    return predict_intent(text)


def summarize_text(text: str):
    # placeholder logic
    return {
        "summary": text[:200] + "..."
    }


def chat_response(query: str):
    return {
        "reply": "This feature will answer legal queries in future versions."
    }


def classify_service(text: str):
    result = predict_intent(text)
    return {
        "intent": result["intent"],
        "confidence": result["confidence"]
    }

    
def answer_query(query, context=None):
    vector = vectorizer.transform([query])
    prediction = model.predict(vector)

    answer = retrieve_relevant_law(query)
    confidence = max(model.predict_proba(vector)[0])

    return {
        "answer": answer,
        "source": "IPC / CrPC / Final_IC",
        "confidence": round(confidence, 2)
    }
