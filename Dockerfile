# CPU-only deployment image for the Section 10 API + frontend.
# IMPORTANT: run `git lfs pull` BEFORE `docker build` -- the checkpoints under
# outputs/checkpoints/ are Git LFS pointers, not real files, until that runs;
# building without it ships broken (pointer-text) model weight files.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY src/ src/
COPY backend/ backend/
COPY frontend/ frontend/
COPY outputs/checkpoints/yolov8n_player/weights/best.pt outputs/checkpoints/yolov8n_player/weights/best.pt
COPY outputs/checkpoints/person_finetune_b3a.pt outputs/checkpoints/person_finetune_b3a.pt
COPY outputs/checkpoints/b8mt_lam1.0.pt outputs/checkpoints/b8mt_lam1.0.pt

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
