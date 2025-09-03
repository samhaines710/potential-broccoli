# syntax=docker/dockerfile:1

FROM python:3.11-slim
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
    ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib

# Python deps (expects requirements.txt in this folder/context)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && if [ -s /tmp/requirements.txt ]; then pip install --no-cache-dir -r /tmp/requirements.txt; fi \
 && pip install --no-cache-dir boto3 requests

# Application code
COPY . /app

# Create model dir (as root) so it's present regardless of user
RUN mkdir -p /app/models

# Entry script: copy and chmod as root (must be BEFORE USER switch)
COPY scripts/docker_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

# Non-root user & ownership of app dir
RUN addgroup --system app && adduser --system --ingroup app app \
 && chown -R app:app /app
USER app

EXPOSE 8000 10000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]
