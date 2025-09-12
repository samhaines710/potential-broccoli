# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tini bash \
 && rm -rf /var/lib/apt/lists/*

# Runtime env
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

# Copy code first so files are present for installs/import checks
COPY . /app

# Ensure model dir exists
RUN mkdir -p /app/models

# Install deps: FastAPI/servers + your project requirements if present
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir fastapi uvicorn gunicorn requests boto3 \
 && if [ -f /app/requirements.txt ] && [ -s /app/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/requirements.txt ; \
    fi

# Inline entrypoint with pre-import check; prints full traceback if import fails
RUN printf '%s\n' '#!/usr/bin/env bash' \
 'set -euo pipefail' \
 'echo "[DEBUG] PYTHONPATH=$PYTHONPATH"' \
 'echo "[DEBUG] Listing /app:"' \
 'ls -la /app || true' \
 'echo "[DEBUG] Import check for ${APP_MODULE}..."' \
 'python - <<PY\nimport os, sys, importlib, traceback\nprint(\"sys.path:\", sys.path)\nmod, _, attr = os.environ[\"APP_MODULE\"].partition(\":\")\ntry:\n    m = importlib.import_module(mod)\n    if attr:\n        getattr(m, attr)\n    print(\"[DEBUG] Import OK:\", mod)\nexcept Exception as e:\n    print(\"[DEBUG] Import FAILED:\", e)\n    traceback.print_exc()\n    raise\nPY' \
 'PORT="${PORT:-10000}"' \
 'echo "[BOOT] ${APP_SERVER} ${APP_MODULE} on :${PORT}"' \
 'if [ "${APP_SERVER}" = "uvicorn" ]; then' \
 '  exec uvicorn "${APP_MODULE}" --host 0.0.0.0 --port "${PORT}"' \
 'else' \
 '  exec gunicorn "${APP_MODULE}" -b "0.0.0.0:${PORT}" --workers "${WEB_CONCURRENCY}" --timeout 120' \
 'fi' \
 > /usr/local/bin/entrypoint.sh \
 && chmod 0755 /usr/local/bin/entrypoint.sh

# Non-root
RUN addgroup --system app && adduser --system --ingroup app app \
 && chown -R app:app /app
USER app

EXPOSE 8000 10000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]
