# syntax=docker/dockerfile:1

FROM python:3.11-slim
WORKDIR /app

# System deps: xgboost/numba runtime, TLS, proper init, bash for entrypoint
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 ca-certificates tini bash \
    && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_DEFAULT_REGION=eu-north-1 \
    # Unified model variable used by ml_classifier and entrypoint
    MODEL_URI=/app/models/xgb_classifier.pipeline.joblib \
    # Back-compat in case other modules still read this
    ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib

# Python dependencies (expects requirements.txt to be in this folder)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && if [ -s /tmp/requirements.txt ]; then pip install --no-cache-dir -r /tmp/requirements.txt; fi \
 && pip install --no-cache-dir boto3 requests

# Application code
COPY . /app

# Non-root and model dir
RUN mkdir -p /app/models \
 && addgroup --system app && adduser --system --ingroup app app \
 && chown -R app:app /app
USER app

# Entry script (copied, not inlined)
COPY scripts/docker_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000 10000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]
