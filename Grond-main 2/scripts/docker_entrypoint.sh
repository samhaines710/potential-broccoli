#!/usr/bin/env bash
set -euo pipefail

# Render injects $PORT; default for local runs
PORT="${PORT:-10000}"

# APP_MODULE: dotted path to ASGI/WSGI app. Defaults to "main:app"
APP_MODULE="${APP_MODULE:-main:app}"

# APP_SERVER: uvicorn (ASGI) or gunicorn (WSGI)
APP_SERVER="${APP_SERVER:-uvicorn}"

echo "[BOOT] Starting ${APP_SERVER} with module ${APP_MODULE} on :${PORT}"

if [ "$APP_SERVER" = "uvicorn" ]; then
  # FastAPI / Starlette / any ASGI app
  exec uvicorn "$APP_MODULE" --host 0.0.0.0 --port "${PORT}"
else
  # Flask / Django (WSGI)
  # For Flask, set APP_MODULE=app:app (or your module)
  exec gunicorn "$APP_MODULE" -b "0.0.0.0:${PORT}" --workers "${WEB_CONCURRENCY:-2}" --timeout 120
fi
