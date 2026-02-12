from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    intent: str
    confidence: float

class SummaryResponse(BaseModel):
    summary: str
