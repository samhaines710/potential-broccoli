# syntax=docker/dockerfile:1

# ── Base image & workdir ───────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# ── System deps ────────────────────────────────────────────────────────────────
# libgomp1: xgboost/sklearn; ca-certificates: TLS; tini: PID 1; bash: entrypoint shell
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 ca-certificates tini bash \
    && rm -rf /var/lib/apt/lists/*

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_DEFAULT_REGION=eu-north-1 \
    # Unified model path env used by the app (ml_classifier expects MODEL_URI)
    MODEL_URI=/app/models/xgb_classifier.pipeline.joblib \
    # Back-compat: expose same path under legacy name if other code reads it
    ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && if [ -s /tmp/requirements.txt ]; then pip install --no-cache-dir -r /tmp/requirements.txt; fi \
 && pip install --no-cache-dir boto3

# ── Application code ───────────────────────────────────────────────────────────
COPY . /app

# ── Model dir & non-root user ─────────────────────────────────────────────────
RUN mkdir -p /app/models \
 && addgroup --system app && adduser --system --ingroup app app \
 && chown -R app:app /app
USER app

# ── Entry point: optional S3 download, then run orchestrator ───────────────────
RUN printf '%s\n' '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  '' \
  'MODEL_URI="${MODEL_URI:-}"' \
  'if [[ -z "$MODEL_URI" ]]; then' \
  '  echo "MODEL_URI not set. Refusing to start." >&2; exit 2' \
  'fi' \
  '' \
  'mkdir -p "$(dirname "$MODEL_URI")"' \
  '' \
  '# If MODEL_URI points to S3, download once into the standard local path' \
  'if [[ "$MODEL_URI" == s3://* ]]; then' \
  '  if [[ ! -f "/app/models/model.loaded" ]]; then' \
  '    python - <<'"'"'PY'"'"'' \
  'import os, pathlib, boto3' \
  'uri = os.environ["MODEL_URI"]' \
  'bucket, key = uri[5:].split("/", 1)' \
  'dst = "/app/models/xgb_classifier.pipeline.joblib"' \
  'pathlib.Path("/app/models").mkdir(parents=True, exist_ok=True)' \
  'boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION","eu-north-1")).download_file(bucket, key, dst)' \
  'open("/app/models/model.loaded","w").write("ok")' \
  'print(f"Downloaded {uri} -> {dst}")' \
  'PY' \
  '  fi' \
  'fi' \
  '' \
  'exec python -u grond_orchestrator.py' \
  > /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

# ── Networking & init ─────────────────────────────────────────────────────────
EXPOSE 8000 10000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/entrypoint.sh"]
