from pathlib import Path
import sys
from contextlib import asynccontextmanager

# Ensure the project root is on sys.path whether run as a script or module.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api.routes import classify, summarize, chat
from src.models.verify_models import check_models


def _raise_if_models_missing() -> None:
    missing, empty = check_models()
    if not missing and not empty:
        return

    details = []
    if missing:
        details.append("Missing: " + ", ".join(str(path) for path in missing))
    if empty:
        details.append("Empty: " + ", ".join(str(path) for path in empty))

    raise RuntimeError(
        "Model files are missing or empty. "
        + " | ".join(details)
        + ". Regenerate with: python -m src.models.train_ml_models"
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _raise_if_models_missing()
    yield


app = FastAPI(
    title="Clarilaw-AI",
    description="AI-powered Legal Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(summarize.router, prefix="/api")
app.include_router(classify.router, prefix="/api")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Avoid noisy 404s for browser favicon requests.
    return Response(status_code=204)

@app.get("/")
def read_root():
    return {"message": "Welcome to Clarilaw-AI! Visit /docs for API documentation."}
