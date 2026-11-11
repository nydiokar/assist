# assist image (spec A03 §8.1). S4 builds it for OCR-enabled ingests on the dev PC; S8 hardens it
# (non-root user, read-only root filesystem, healthcheck, GHCR publish). No torch: model inference
# is served by llama-server over HTTP.
#
#   docker build -f deploy/Dockerfile -t assist:dev .
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# OCR (D-10/A11): tesseract with Bulgarian + English, ocrmypdf for scanned PDFs, antiword for
# legacy .doc. curl for container healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-bul tesseract-ocr-eng ocrmypdf ghostscript antiword curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUTF8=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra serve --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --extra serve

ENTRYPOINT ["python", "-m", "project_rag.cli"]
