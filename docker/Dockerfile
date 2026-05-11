# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata + source so all deps can be resolved
COPY pyproject.toml .
COPY src/ src/

# Build wheels for the project AND all transitive dependencies
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels .

# ─────────────────────────────────────────────────────────────────────────
# Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels

# Install all packages from pre-built wheels (no internet needed)
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-index --find-links=/wheels carq

# Copy application
COPY src/ src/
COPY migrations/ migrations/

# Create non-root user
RUN useradd -m -u 1000 carq
USER carq

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app:${PATH}"

# Health check — uses curl (installed above), no Python import needed
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

EXPOSE 8000

# Default to uvicorn
CMD ["uvicorn", "carq.main:app", "--host", "0.0.0.0", "--port", "8000"]
