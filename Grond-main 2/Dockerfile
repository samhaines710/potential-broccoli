# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates tini bash \
 && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_DEFAULT_REGION=eu-north-1 \
    MODEL_URI=/app/models/xgb_classifier.pipeline.joblib \
    ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib \
    PORT=10000 \
    WEB_CONCURRENCY=2

# Copy application first (ensures requirements.txt is available in /app)
COPY . /app

# Ensure models dir exists
RUN mkdir -p /app/models

# Install deps
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir gunicorn uvicorn boto3 requests \
 && if [ -f /app/requirements.txt ] && [ -s /app/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/requirements.txt ; \
    fi

# Inline entrypoint (no external file)
RUN printf '%s\n' '#!/usr/bin/env bash' \
 'set -euo pipefail' \
 'PORT="${PORT:-10000}"' \
 'APP_MODULE="${APP_MODULE:-main:app}"' \
 'APP_SERVER="${APP_SERVER:-uvicorn}"' \
 'echo "[BOOT] ${APP_SERVER} ${APP_MODULE} on :${PORT}"' \
 'if [ "${APP_SERVER}" = "uvicorn" ]; then' \
 '  exec uvicorn "${APP_MODULE}" --host 0.0.0.0 --port "${PORT}"' \
 'else' \
 '  exec gunicorn "${APP_MODULE}" -b "0.0.0.0:${PORT}" --workers "${WEB_CONCURRENCY}" --timeout 120' \
 'fi' \
 > /usr/local/bin/entrypoint.sh \
 && chmod 0755 /usr/local/bin/entrypoint.sh

# Drop privileges
RUN addgroup --system app && adduser --system --ingroup app app \
 && chown -R app:app /app
USER app

EXPOSE 8000 10000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]