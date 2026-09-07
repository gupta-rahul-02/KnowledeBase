# ---------- Stage 1: build the React frontend ----------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


# ---------- Stage 2: Python runtime ----------
FROM python:3.12-slim AS runtime

# curl for HEALTHCHECK; build-essential for any sdists that need compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    DATA_DIR=/data \
    HF_HOME=/data/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/data/hf_cache \
    LOG_LEVEL=INFO \
    FRONTEND_DIST=/app/frontend_dist

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY --from=frontend-build /app/frontend/dist/ ./frontend_dist/

# Persistent volume mount point
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# --workers 1 is REQUIRED: Chroma's PersistentClient is not multi-process safe.
CMD ["sh", "-c", "uvicorn src.knowledebase.api:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]
