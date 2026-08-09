#!/bin/sh
set -eu

uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
backend_pid=$!

cleanup() {
    kill -TERM "$backend_pid" 2>/dev/null || true
    wait "$backend_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 10000 \
    --server.headless true
