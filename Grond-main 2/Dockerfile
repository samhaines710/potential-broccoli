# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

WORKDIR /app

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tini bash \
 && rm -rf /var/lib/apt/lists/*

# ── Runtime env ────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_DEFAULT_REGION=eu-north-1 \
    MODEL_URI=/app/models/xgb_classifier.pipeline.joblib \
    ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib \
    PORT=10000 \
    WEB_CONCURRENCY=1 \
    APP_SERVER=uvicorn \
    APP_MODULE=server:app \
    PYTHONPATH=/app

# ── Copy code FROM subfolder (because Root Directory = repo root) ─────────────
# IMPORTANT: keep the quotes because the folder name contains a space.
COPY "Grond-main 2/." /app

# Ensure model dir exists (if you download a model at runtime)
RUN mkdir -p /app/models

# ── Python deps ────────────────────────────────────────────────────────────────
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir fastapi uvicorn gunicorn requests boto3 \
 && if [ -f /app/requirements.txt ] && [ -s /app/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/requirements.txt ; \
    fi

# ── Import checker (prints full traceback if APP_MODULE can't import) ─────────
RUN cat > /usr/local/bin/import_check.py <<'PY'
import os, sys, importlib, traceback
mod, _, attr = os.environ.get("APP_MODULE","server:app").partition(":")
print("sys.path:", sys.path)
print("[DEBUG] Checking import:", mod, "attr:", attr)
try:
    m = importlib.import_module(mod)
    if attr:
        getattr(m, attr)
    print("[DEBUG] Import OK:", mod)
except Exception as e:
    print("[DEBUG] Import FAILED:", e)
    traceback.print_exc()
    raise
PY
RUN chmod 0755 /usr/local/bin/import_check.py

# ── Entrypoint ────────────────────────────────────────────────────────────────
RUN cat > /usr/local/bin/entrypoint.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

echo "[DEBUG] PYTHONPATH=$PYTHONPATH"
echo "[DEBUG] Listing /app top-level:"
ls -la /app || true
echo "[DEBUG] Expecting /app/server.py or a module that provides FastAPI app"

# Quick sanity: ensure server.py exists (your APP_MODULE points to server:app)
if [ ! -f /app/server.py ]; then
  echo "[WARN] /app/server.py not found. Will still try to import ${APP_MODULE}."
fi

echo "[DEBUG] Import check for ${APP_MODULE}..."
python /usr/local/bin/import_check.py

PORT="${PORT:-10000}"
echo "[BOOT] ${APP_SERVER} ${APP_MODULE} on :${PORT}"

if [ "${APP_SERVER}" = "uvicorn" ]; then
  exec uvicorn "${APP_MODULE}" --host 0.0.0.0 --port "${PORT}"
else
  exec gunicorn "${APP_MODULE}" -b "0.0.0.0:${PORT}" --workers "${WEB_CONCURRENCY}" --timeout 120
fi
SH
RUN chmod 0755 /usr/local/bin/entrypoint.sh

# ── Drop privileges ───────────────────────────────────────────────────────────
RUN addgroup --system app && adduser --system --ingroup app app \
 && chown -R app:app /app
USER app

EXPOSE 8000 10000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]
