"""FastAPI backend for the Section 10 deliverable: upload a volleyball
match video, get back a timestamped event timeline.

Models (YOLOv8n fine-tuned detector, B3-A backbone, B8-mt) load ONCE at
startup, not per request -- re-loading a ResNet50 + LSTM stack on every
upload would make each request take minutes instead of seconds.

Run locally:
    .venv\\Scripts\\python.exe -m uvicorn backend.app:app --reload --port 8000

Then open http://localhost:8000/ (serves the frontend) or POST a video
directly to http://localhost:8000/api/analyze.
"""
import os
import shutil
import sys
import tempfile
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB -- generous for a short clip, not a full match
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

app = FastAPI(title="Volleyball Event Timeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev default -- restrict to the real frontend origin before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None  # lazy-loaded once, on first request or at startup


def get_predictor():
    global _predictor
    if _predictor is None:
        from src.inference.sliding_window import WindowPredictor
        print("loading models (B3-A backbone + B8-mt) ...")
        _predictor = WindowPredictor()
        print("models loaded.")
    return _predictor


@app.on_event("startup")
def _warm_up():
    # Load models at startup, not on the first upload, so the first real
    # request isn't the one that pays the ~seconds-long load cost.
    get_predictor()


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": _predictor is not None}


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), stride: int = 9, n_frames: int = 9):
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"unsupported file type {ext!r}; use one of {sorted(ALLOWED_EXTENSIONS)}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            total = 0
            while chunk := await video.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
                tmp.write(chunk)

        from src.inference.timeline import build_timeline
        predictor = get_predictor()
        result = build_timeline_with_predictor(tmp_path, predictor, n_frames=n_frames, stride=stride)
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"failed to process video: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_timeline_with_predictor(video_path, predictor, n_frames=9, stride=9):
    """Same as timeline.build_timeline() but reuses an already-loaded
    WindowPredictor instead of constructing (and re-loading models) a new
    one per request."""
    from src.inference.video_io import get_fps
    from src.inference.sliding_window import iter_windows

    fps = get_fps(video_path)
    raw = []
    for start_f, end_f, pred in iter_windows(video_path, predictor, n_frames=n_frames, stride=stride):
        if pred is None:
            continue
        raw.append({"start_frame": start_f, "end_frame": end_f, **pred})

    events = []
    for w in raw:
        if events and events[-1]["_scene_class"] == w["scene_class"]:
            events[-1]["end_frame"] = w["end_frame"]
            events[-1]["confidence"] = max(events[-1]["confidence"], w["scene_confidence"])
        else:
            events.append({
                "start_frame": w["start_frame"], "end_frame": w["end_frame"],
                "team": w["team"], "event": w["event"], "_scene_class": w["scene_class"],
                "confidence": w["scene_confidence"],
            })
    for e in events:
        e["start_sec"] = round(e.pop("start_frame") / fps, 2)
        e["end_sec"] = round((e.pop("end_frame") + 1) / fps, 2)
        e.pop("_scene_class")

    return {"fps": fps, "n_windows": len(raw), "events": events}


# Serve the frontend as static files at "/" (same origin as the API --
# simplest thing to deploy as one unit).
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
