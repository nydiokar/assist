# assist image (spec A03 §8.1, hardened in S8): python:3.12-slim + assist (no torch) + OCR tools.
# Runs as a non-root user; designed for a read-only root filesystem (compose: read_only + tmpfs
# /tmp); writable state only under /state. Model inference is served by llama-server over HTTP.
#
#   docker build -f deploy/Dockerfile -t assist:dev .
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# OCR (D-10/A11): tesseract with Bulgarian + English, ocrmypdf for scanned PDFs, antiword for
# legacy .doc. curl for the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-bul tesseract-ocr-eng ocrmypdf ghostscript antiword curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra serve --no-install-project --no-editable
COPY src ./src
COPY configs ./configs
COPY scripts/state_backup.py ./scripts/state_backup.py
RUN uv sync --frozen --extra serve --no-editable \
    && useradd --uid 10001 --create-home --home-dir /home/assist --shell /usr/sbin/nologin assist \
    && mkdir -p /state /corpus /models \
    && chown assist:assist /state

USER 10001
VOLUME ["/state"]
EXPOSE 8000 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["python", "-m", "project_rag.cli"]
