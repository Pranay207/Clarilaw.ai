from pathlib import Path
import sys

# Allow running this file directly by ensuring the project root is on sys.path.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import APIRouter
from api.schemas.request_response import TextRequest
from api.services.legal_ai_service import chat_response

router = APIRouter()

@router.post("/chat")
def chat(request: TextRequest):
    return chat_response(request.text)
