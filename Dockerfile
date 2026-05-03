# syntax=docker/dockerfile:1.7

# ---- Builder stage --------------------------------------------------------
# Resolves Python deps in a fat image, then we discard the build chain.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build tools needed for some sklearn / scipy wheel paths on slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src/ src/

# Install CPU-only PyTorch from the dedicated index, then everything else.
# Two passes so the constraint resolver doesn't fight over cuda metadata.
RUN python -m pip install --upgrade pip \
 && python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install -r requirements.txt \
 && python -m pip install --no-deps -e .


# ---- Runtime stage --------------------------------------------------------
# Slim base; copy only the installed packages + the source layout.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db \
    MODEL_NAME=churn_classifier \
    MODEL_STAGE=Production

WORKDIR /app

# Site-packages from the builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Application source. The model itself is loaded from the MLflow registry at
# startup, *not* baked into the image — keeps images model-agnostic and
# decouples deploys from model rollouts.
COPY src/ /app/src/

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash app \
 && mkdir -p /app/mlruns /app/data \
 && chown -R app:app /app
USER app

EXPOSE 8000

# A simple curl-based healthcheck — Docker / Kubernetes can use this as the
# liveness probe; the API itself reports degraded vs. ok in the body.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "churn.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
