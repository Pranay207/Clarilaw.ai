# Clarilaw.ai

Legal AI service for intent classification and related utilities, with a FastAPI backend and supporting training/evaluation scripts.

**Showcase**
- Production‑ready FastAPI backend with clean routing and startup validation.
- End‑to‑end ML workflow: preprocessing, training, evaluation, serving.
- Dual pipeline support: classical ML (TF‑IDF + scikit‑learn) and transformer fine‑tuning.
- Reproducible scripts and project‑relative paths for consistent local or CI runs.
- Clear artifacts: models saved under `models/`, reports under `evaluation/`.

**Tech Stack**
- Python, FastAPI, scikit‑learn, Transformers, PyTorch
- pandas, NumPy, matplotlib, seaborn

**API Endpoints**
- `POST /api/classify` intent classification
- `POST /api/summarize` text summarization (placeholder logic)
- `POST /api/chat` chat response (placeholder logic)

**Example Requests**
```powershell
curl -X POST "http://127.0.0.1:8000/api/classify" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Explain Section 420 of IPC\"}"
```

```powershell
curl -X POST "http://127.0.0.1:8000/api/summarize" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Long legal text here...\"}"
```

**Evaluation Snapshot**
- Metrics and artifacts are tracked under `evaluation/`.
- See `evaluation/metrics.txt` and `evaluation/results.md` for the latest run.

**Highlights**
- FastAPI API with `/api/classify`, `/api/summarize`, and `/api/chat` routes.
- Scikit-learn intent classifier with TF‑IDF vectorization.
- Optional transformer training pipeline for intent classification.
- Utilities for data validation, training, and evaluation.

**Requirements**
- Python 3.10+
- Windows, macOS, or Linux

**Quick Start**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the API:
```powershell
.\.venv\Scripts\python.exe -m api
```

Or with Uvicorn:
```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open:
- `http://127.0.0.1:8000/` for the root message
- `http://127.0.0.1:8000/docs` for Swagger UI

**Project Layout**
- `api/` FastAPI app and routes
- `src/` training, inference, and utilities
- `data/` datasets and processed artifacts
- `models/` trained model artifacts
- `evaluation/` evaluation outputs and reports

**Training (ML)**
Train the scikit‑learn model and vectorizer:
```powershell
.\.venv\Scripts\python.exe -m src.models.train_ml_models
```

This writes:
- `models/intent_classifier.pkl`
- `models/vectorizer.pkl`

**Training (Transformer)**
Prepare DL training data:
```powershell
.\.venv\Scripts\python.exe -m src.models.prepare_dl_data
```

Train the transformer model:
```powershell
.\.venv\Scripts\python.exe -m src.models.train_dl_models
```

This writes:
- `models/transformer_intent_classifier/`

**Evaluation**
```powershell
.\.venv\Scripts\python.exe -m src.models.evaluate
```

Artifacts are saved under `evaluation/`.

**Troubleshooting**
- Missing model files: run `python -m src.models.train_ml_models`.
- Missing packages: install from `requirements.txt`.
- If you see warnings about sklearn pickle versions, retrain with your current scikit‑learn version.

**Development Notes**
- Paths are project‑relative to support any working directory.
- For production, pin versions in `requirements.txt` and run with a WSGI/ASGI server.




-# Clarilaw.ai
-
-## Running the API from anywhere
-
-Install the project in editable mode once:
-
-```powershell
-cd C:\Users\Shiva\OneDrive\Desktop\Clarilaw.ai-main\Clarilaw.ai-main
-pip install -e .
-```
-
-After that, these both work from any directory:
-
-```powershell
-python -m api
-```
-
-```powershell
-uvicorn api.main:app --reload
-```
