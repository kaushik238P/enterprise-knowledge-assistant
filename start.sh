#!/bin/sh

set -e

echo "Starting FastAPI..."

uv run uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 &

echo "Waiting for FastAPI to become ready..."

until curl -fs http://localhost:8000/health >/dev/null 2>&1; do
    sleep 1
done

echo "FastAPI is ready."

echo "Starting Streamlit..."

exec uv run streamlit run frontend/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501