# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Instala tudo em /install usando wheels pré-compilados do PyPI
RUN pip install --no-cache-dir --prefix=/install .

# ─────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia pacotes instalados pelo builder
COPY --from=builder /install /usr/local

COPY src/ src/
COPY migrations/ migrations/

RUN useradd -m -u 1000 carq
USER carq

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

EXPOSE 8000

CMD ["uvicorn", "carq.main:app", "--host", "0.0.0.0", "--port", "8000"]
