# ReefScan backend — Hugging Face Spaces (Docker SDK, CPU). Phase 5 deploy.
# Build context = repo root:  docker build -f deploy/Dockerfile -t reefscan .
FROM python:3.11-slim

# system libs: git (sam2 from source), ffmpeg + libgl (opencv/video)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch wheels + app deps, then SAM2 from source (no PyPI release)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
 && pip install --no-cache-dir "git+https://github.com/facebookresearch/sam2.git"

COPY backend ./backend

# HF model repo to load weights + conformal from (overridable via Space secrets).
# HF_HOME under /tmp so the (non-root) Spaces runtime can write the model cache.
ENV HF_MODEL_REPO=HrishiKabra/reefscan-dinov2-coral \
    HF_MODEL_STAGE=finetune \
    HF_HOME=/tmp/huggingface \
    PORT=7860

# Spaces routes traffic to 7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
