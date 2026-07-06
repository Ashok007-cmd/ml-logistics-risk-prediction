#!/bin/bash
set -euo pipefail

export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"

# Start FastAPI backend in the background
echo "Starting FastAPI backend on port 8000..."
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start Streamlit frontend in the background too, so we can wait on both
# and exit non-zero (and take the other process down) if either one dies.
echo "Starting Streamlit frontend on port 8501..."
python -m streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &
STREAMLIT_PID=$!

trap 'kill "$API_PID" "$STREAMLIT_PID" 2>/dev/null || true' EXIT
wait -n "$API_PID" "$STREAMLIT_PID"
