"""Deploy the FastAPI backend (backend/app.py) to Modal.

    pip install modal
    modal setup                           # one-time browser login
    modal deploy deploy/modal/modal_app.py

Modal prints the public https URL; put it in frontend/config.js as API_BASE.
CPU-only. max_containers=1 on purpose: backend/app.py keeps job state in
process memory, so every request must reach the same container.
"""
import os

import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REMOTE = "/root/app"
WEIGHTS = [
    "outputs/checkpoints/yolov8n_player/weights/best.pt",
    "outputs/checkpoints/person_finetune_b3a.pt",
    "outputs/checkpoints/b8mt_lam1.0.pt",
]
IGNORE = ["**/__pycache__/**", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("fastapi", "uvicorn[standard]", "python-multipart",
                 "opencv-python-headless", "ultralytics", "yt-dlp", "tqdm", "numpy")
    .add_local_dir(os.path.join(ROOT, "backend"), f"{REMOTE}/backend", ignore=IGNORE)
    .add_local_dir(os.path.join(ROOT, "src"), f"{REMOTE}/src", ignore=IGNORE)
)
for w in WEIGHTS:
    image = image.add_local_file(os.path.join(ROOT, w), f"{REMOTE}/{w}")

app = modal.App("volleyball-api")


@app.function(image=image, cpu=4, memory=6144, timeout=3600,
              max_containers=1, scaledown_window=600)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def web():
    import sys

    os.chdir(REMOTE)
    sys.path.insert(0, REMOTE)
    from backend.app import app as fastapi_app

    return fastapi_app
