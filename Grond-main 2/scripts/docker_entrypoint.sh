#!/usr/bin/env bash
set -euo pipefail

# MODEL_URI can be a local path (default in Dockerfile) or s3://bucket/key
MODEL_URI="${MODEL_URI:-/app/models/xgb_classifier.pipeline.joblib}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-eu-north-1}"

mkdir -p "$(dirname "$MODEL_URI")"

# If MODEL_URI is an S3 URI, download once to the canonical local model path
if [[ "$MODEL_URI" == s3://* ]]; then
  if [[ ! -f "/app/models/model.loaded" ]]; then
    python - <<PY
import os, pathlib, boto3
uri = os.environ["MODEL_URI"]
bucket, key = uri[5:].split("/", 1)
dst = "/app/models/xgb_classifier.pipeline.joblib"
pathlib.Path("/app/models").mkdir(parents=True, exist_ok=True)
boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION","eu-north-1")).download_file(bucket, key, dst)
open("/app/models/model.loaded","w").write("ok")
print(f"Downloaded {uri} -> {dst}")
PY
  fi
fi

# Start the application
exec python -u grond_orchestrator.py
