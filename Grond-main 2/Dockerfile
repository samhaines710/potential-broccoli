# ── Base image & workdir ────────────────────────────────────────────────────────
FROM python:3.10-slim
WORKDIR /app

# ── Environment ──────────────────────────────────────────────────────────────────
ENV ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib

# ── Dependencies ────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy your application code ──────────────────────────────────────────────────
COPY . .

# ── Prepare models folder (no COPY of the model itself) ─────────────────────────
RUN mkdir -p /app/models

# ── Expose ports ────────────────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 10000

# ── Entrypoint ──────────────────────────────────────────────────────────────────
CMD ["python", "grond_orchestrator.py"]