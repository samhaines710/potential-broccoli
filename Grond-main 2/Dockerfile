# ── Base image & workdir ────────────────────────────────────────────────────────
FROM python:3.10-slim
WORKDIR /app

# ── System deps (slim-friendly) ────────────────────────────────────────────────
# libgomp1: XGBoost runtime; ca-certificates: TLS; tini: clean PID 1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# ── Runtime env ────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    AWS_DEFAULT_REGION=eu-north-1 \
    # Where the model will live inside the container
    MODEL_LOCAL_PATH=/app/models/xgb_classifier.pipeline.joblib \
    # If this is s3://bucket/key the entrypoint will download it at boot
    ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib

# ── Dependencies ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # ensure boto3 exists even if requirements.txt forgot it
    && pip install --no-cache-dir boto3

# ── Copy application ───────────────────────────────────────────────────────────
COPY . .

# ── Prepare models folder ──────────────────────────────────────────────────────
RUN mkdir -p /app/models

# ── Non-root user ──────────────────────────────────────────────────────────────
RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app
USER app

# ── Bootstrap: download model from S3 if needed, then exec app ────────────────
# We install a tiny entrypoint that:
#   - if ML_MODEL_PATH starts with s3:// and local file missing, downloads it
#   - then execs the orchestrator
# Note: relies on AWS creds via env (GitHub Actions secrets or ECS task role).
RUN printf '%s\n' '#!/bin/sh' \
    'set -euo pipefail' \
    '' \
    'MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-/app/models/xgb_classifier.pipeline.joblib}"' \
    'ML_MODEL_PATH="${ML_MODEL_PATH:-}"' \
    '' \
    'mkdir -p "$(dirname "$MODEL_LOCAL_PATH")"' \
    '' \
    'case "$ML_MODEL_PATH" in' \
    '  s3://*)' \
    '    if [ ! -f "$MODEL_LOCAL_PATH" ]; then' \
    '      python - <<'"'"'PY'"'"'' \
    'import os, pathlib, boto3' \
    'uri = os.environ["ML_MODEL_PATH"]' \
    'dst = os.environ.get("MODEL_LOCAL_PATH", "/app/models/xgb_classifier.pipeline.joblib")' \
    'bucket, key = uri[5:].split("/", 1)' \
    'pathlib.Path(os.path.dirname(dst) or ".").mkdir(parents=True, exist_ok=True)' \
    'boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION","eu-north-1")).download_file(bucket, key, dst)' \
    'print(f"Downloaded {uri} -> {dst}")' \
    'PY' \
    '    fi' \
    '    ;;' \
    'esac' \
    '' \
    'exec python grond_orchestrator.py' \
    > /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

# ── Networking ─────────────────────────────────────────────────────────────────
EXPOSE 8000 10000

# ── Use tini as PID1, then run our entrypoint ──────────────────────────────────
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]
