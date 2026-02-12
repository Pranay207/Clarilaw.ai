FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV SKIP_MODEL_CHECK=1

COPY api/ api/
COPY src/ src/
COPY requirements-render.txt .
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install --no-cache-dir --retries 10 --timeout 120 --progress-bar off -r requirements-render.txt

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:${PORT:-8000}"]
