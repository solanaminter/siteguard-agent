# SiteGuard Agent — container image (CPU; OpenCV 5 headless)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STORAGE_BACKEND=local

WORKDIR /app

# System deps for OpenCV headless + video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY siteguard/ siteguard/
COPY assets/models/SHA256SUMS assets/models/
COPY scripts/download_models.sh scripts/

# Fetch the pinned detection model at build time (verifies SHA256)
RUN bash scripts/download_models.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "siteguard.app:app", "--host", "0.0.0.0", "--port", "8000"]
