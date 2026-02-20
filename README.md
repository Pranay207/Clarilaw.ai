# Clarilaw.ai ⚖️🤖
### AI-Powered Legal Understanding Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-orange)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## 📘 Overview

**Clarilaw.ai** is an AI-powered legal intelligence service that helps users understand legal queries through **intent classification and NLP-driven utilities**. The project is designed as a **production-ready Machine Learning system**, focusing on clean architecture, reproducibility, and deployable ML inference APIs.

> 🎯 **Focus:** Machine Learning Engineering, NLP pipelines, and API-based model serving (no frontend by design).

---

## 🚀 Why Clarilaw.ai?

Legal language is complex and inaccessible to non-experts.  
Clarilaw.ai demonstrates how **Machine Learning and NLP** can be used to:

- ⚖️ Understand legal intent from user queries  
- 🧠 Classify legal questions accurately  
- 🔌 Serve ML models through scalable REST APIs  
- ♻️ Build reproducible, industry-style ML pipelines  

This project reflects **real-world ML engineering practices**, not just academic experimentation.

---

## ✨ Key Highlights (Recruiter-Focused)

* ✅ **Production-ready FastAPI backend** with modular routing and validation
* ✅ **End-to-end ML workflow**: data preprocessing → training → evaluation → inference
* ✅ **Dual modeling pipelines**:

  * Classical ML (TF-IDF + scikit-learn)
  * Transformer-based intent classification (PyTorch + Hugging Face)
* ✅ **Reproducible experiments** with project-relative paths
* ✅ **Clear artifacts management**:

  * Trained models → `models/`
  * Metrics & reports → `evaluation/`
* ✅ **Docker-ready & deployment-friendly** design

---

## 🧠 Tech Stack

* **Languages & Frameworks**: Python, FastAPI
* **Machine Learning**: scikit-learn, Transformers
* **Deep Learning**: PyTorch
* **Data Processing**: pandas, NumPy
* **Evaluation & Visualization**: matplotlib, seaborn

---

## 🔌 API Endpoints

| Endpoint         | Method | Description                                  |
| ---------------- | ------ | -------------------------------------------- |
| `/api/classify`  | POST   | Legal intent classification                  |
| `/api/summarize` | POST   | Legal text summarization (placeholder logic) |
| `/api/chat`      | POST   | Chat-style response (placeholder logic)      |

---

## 📥 Example Requests

### Intent Classification

```powershell
curl -X POST "http://127.0.0.1:8000/api/classify" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Explain Section 420 of IPC\"}"
```

### Summarization

```powershell
curl -X POST "http://127.0.0.1:8000/api/summarize" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Long legal text here...\"}"
```

---

## 📊 Evaluation Snapshot

* Model metrics and artifacts are stored under `evaluation/`
* Refer to:

  * `evaluation/metrics.txt`
  * `evaluation/results.md`

These include accuracy, precision, recall, F1-score, and error analysis summaries.

---

## 📁 Project Structure

```
api/            # FastAPI application and routes
src/            # Training, inference, and ML utilities
data/           # Raw and processed datasets
models/         # Saved ML and DL model artifacts
evaluation/     # Metrics, plots, and evaluation reports
deployment/     # Docker and deployment configs
```

---

## ⚙️ Requirements

* Python **3.10+**
* Windows / macOS / Linux

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## ▶️ Running the API

### Option 1: Run as a Python module

```powershell
.\.venv\Scripts\python.exe -m api
```

### Option 2: Run with Uvicorn (recommended)

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open in browser:

* Root: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Model Training

### Train Classical ML Model (TF-IDF + scikit-learn)

```powershell
.\.venv\Scripts\python.exe -m src.models.train_ml_models
```

**Outputs:**

* `models/intent_classifier.pkl`
* `models/vectorizer.pkl`

---

### Train Transformer Model

Prepare data:

```powershell
.\.venv\Scripts\python.exe -m src.models.prepare_dl_data
```

Train model:

```powershell
.\.venv\Scripts\python.exe -m src.models.train_dl_models
```

**Output:**

* `models/transformer_intent_classifier/`

---

## 📈 Evaluation

```powershell
.\.venv\Scripts\python.exe -m src.models.evaluate
```

Evaluation artifacts are saved under `evaluation/`.

---

## 🐳 Deployment

* Dockerfile and docker-compose configuration available under `deployment/`
* The API can be containerized and deployed on platforms like **Render, AWS, or Azure**

---

## 🛠️ Troubleshooting

* **Missing model files** → Run `train_ml_models`
* **Dependency issues** → Reinstall from `requirements.txt`
* **Pickle version warnings** → Retrain models with current scikit-learn version

---

## 📌 Design Philosophy

* No frontend by design — focus is on **ML system engineering**
* APIs + Swagger UI provide sufficient interaction
* Clean separation between training, inference, and deployment

---

## 🎓 Resume Value

This project demonstrates:

* Real-world NLP problem solving
* ML model experimentation and evaluation
* API-based model serving
* Deployment-ready ML systems

> **Clarilaw.ai** is suitable for **Machine Learning Engineer** and **AI Engineer** roles.

---

📫 *Built as a major academic and portfolio project with industry-aligned ML practices.*
