import os
import hashlib
import time
import logging
from typing import Optional

import requests

LOG = logging.getLogger("model_fetch")

MODEL_DIR = "/app/models"
MODEL_PATH = "/app/models/xgb_classifier.pipeline.joblib"
ENV_URL = "MODEL_PRESIGNED_URL"
ENV_SHA256 = "MODEL_SHA256"
ENV_TIMEOUT = "MODEL_DOWNLOAD_TIMEOUT"  # seconds
ENV_RETRIES = "MODEL_DOWNLOAD_RETRIES"  # int

DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3
CHUNK = 1024 * 1024  # 1MB


def _sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out_path: str, timeout: int) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, out_path)


def ensure_model() -> None:
    """Ensure the model exists at MODEL_PATH.
    If MODEL_PRESIGNED_URL is set, download/verify it. Otherwise, do nothing.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    url = os.getenv(ENV_URL, "").strip()
    if not url:
        LOG.warning("No %s set; expecting model already baked at %s", ENV_URL, MODEL_PATH)
        return

    want_sha = os.getenv(ENV_SHA256, "").lower().strip()
    timeout = int(os.getenv(ENV_TIMEOUT, str(DEFAULT_TIMEOUT)))
    retries = int(os.getenv(ENV_RETRIES, str(DEFAULT_RETRIES)))

    have_sha = _sha256(MODEL_PATH)
    if have_sha and want_sha and have_sha == want_sha:
        LOG.info("Model present and checksum matches: %s", MODEL_PATH)
        return
    if have_sha and not want_sha:
        LOG.info("Model present (no checksum provided), skipping download: %s", MODEL_PATH)
        return

    for attempt in range(1, retries + 1):
        try:
            LOG.info("Downloading model (attempt %d/%d) to %s", attempt, retries, MODEL_PATH)
            _download(url, MODEL_PATH, timeout)
            if want_sha:
                got_sha = _sha256(MODEL_PATH)
                if got_sha != want_sha:
                    raise ValueError(f"Checksum mismatch: expected {want_sha}, got {got_sha}")
            LOG.info("Model download complete.")
            return
        except Exception as e:
            LOG.warning("Model download failed (attempt %d/%d): %s", attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(min(5 * attempt, 20))
