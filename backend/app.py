"""FastAPI backend for the Volleyball Group Activity Recognition & Match Analytics Platform.
Features:
- Models loaded once at startup: Fine-tuned YOLOv8n detector, ByteTrack tracker, B3-A backbone, B8-mt Hierarchical LSTM.
- End-to-end inference + annotated H.264 video rendering with broadcast-style HUD overlays.
- Real-time job status & progress polling.
- Built-in video streaming for source and AI-annotated videos.
- Video library browser for both "from internet" (real 3-5min matches) and "from our dataset" (smooth continuous plays).
- One-click analysis of library videos.
- Integrated YouTube downloader for custom matches.
- Serves the frontend platform as static files at root "/".
"""
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
import cv2
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.auth import require_user  # noqa: E402

FRONTEND_DIR = os.path.join(ROOT, "frontend")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
ANNOTATED_DIR = os.path.join(OUTPUTS_DIR, "annotated_videos")
INTERNET_DATA_DIR = os.path.join(ROOT, "production Data", "from internet")
DATASET_DATA_DIR = os.path.join(ROOT, "production Data", "from our dataset")

os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(INTERNET_DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DATA_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Set DISABLE_YOUTUBE=1 to switch off the two endpoints that download from YouTube.
YOUTUBE_ENABLED = os.environ.get("DISABLE_YOUTUBE") != "1"
YOUTUBE_OFF_MSG = ("YouTube downloads are switched off: the model was trained on one fixed camera view and "
                   "has no 'not gameplay' class, so results on full YouTube matches are not reliable yet.")

app = FastAPI(title="Volleyball AI Analytics Platform API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None
_JOBS = {}  # job_id -> job dict


def get_predictor():
    global _predictor
    if _predictor is None:
        from src.inference.sliding_window import WindowPredictor
        print("[Engine] Loading Vision Backbone (ResNet50) & Hierarchical Temporal Model (B8-mt)...")
        _predictor = WindowPredictor()
        print("[Engine] Models loaded successfully into memory.")
    return _predictor


@app.on_event("startup")
def _warm_up():
    try:
        get_predictor()
    except Exception as e:
        print(f"[Engine Startup Warning] Could not preload models: {e}")


def _new_job(filename: str, size_bytes: int, source_type: str = "upload"):
    return {
        "job_id": None,
        "status": "queued",       # queued -> probing -> processing -> rendering -> done | error
        "phase": "Initialized",
        "progress": 0,
        "filename": filename,
        "source_type": source_type,
        "size_bytes": size_bytes,
        "fps": None,
        "total_frames": None,
        "duration_sec": None,
        "total_windows": None,
        "windows_done": 0,
        "events": [],
        "latest_event": None,
        "annotated_video_url": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }


def _run_annotated_job(job_id: str, video_path: str, delete_input: bool = False, n_frames: int = 9, stride: int = 9):
    from src.inference.annotator import render_annotated_pipeline

    job = _JOBS[job_id]
    try:
        job["status"] = "processing"
        job["phase"] = "Detecting Players & Extracting Features"

        predictor = get_predictor()
        out_video_path = os.path.join(ANNOTATED_DIR, f"{job_id}.mp4")

        def on_progress(percent, phase_desc):
            job["progress"] = percent
            job["phase"] = phase_desc

        result = render_annotated_pipeline(
            video_path=video_path,
            out_path=out_video_path,
            predictor=predictor,
            n_frames=n_frames,
            stride=stride,
            progress_callback=on_progress,
        )

        job["fps"] = result["fps"]
        job["total_frames"] = result["total_frames"]
        job["duration_sec"] = result["duration_sec"]
        job["events"] = result["events"]
        job["annotated_video_url"] = f"/api/jobs/{job_id}/video"
        job["status"] = "done"
        job["phase"] = "Analysis & Video Rendering Complete"
        job["progress"] = 100
        job["finished_at"] = time.time()

        # Calculate analytics summary
        event_counts = {}
        team_counts = {"left": 0, "right": 0}
        for ev in result["events"]:
            act = ev["event"].lower()
            team = ev["team"].lower()
            event_counts[act] = event_counts.get(act, 0) + 1
            if team in team_counts:
                team_counts[team] += 1

        job["analytics"] = {
            "total_events": len(result["events"]),
            "event_breakdown": event_counts,
            "team_distribution": team_counts,
        }

    except Exception as e:
        traceback.print_exc()
        job["status"] = "error"
        job["phase"] = "Failed"
        job["error"] = str(e)
        job["finished_at"] = time.time()
    finally:
        if delete_input and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "models_loaded": _predictor is not None,
        "version": "2.0.0",
        "device": "cuda" if _predictor and _predictor.device.type == "cuda" else "cpu"
    }


@app.get("/api/stats")
def model_stats():
    """Returns architecture specifications and benchmark performance."""
    return {
        "architecture": {
            "detector": "YOLOv8n Fine-Tuned on Volleyball Dataset (98.0% mAP50)",
            "tracker": "ByteTrack (97.7% GT Identity Preservation Match)",
            "backbone": "ResNet50 Stage B3-A Feature Extractor",
            "temporal_model": "Multi-Task Hierarchical LSTM (B8-mt)",
            "window_size": "9 frames (~0.36s at 25fps)",
        },
        "benchmarks": {
            "scene_activity_accuracy": "87.4%",
            "player_action_accuracy": "75.7%",
            "detector_precision": "95.3%",
            "detector_recall": "94.2%",
        },
        "supported_classes": {
            "scenes": ["Pass", "Set", "Spike", "Winpoint"],
            "player_actions": ["Blocking", "Digging", "Falling", "Jumping", "Moving", "Setting", "Spiking", "Standing", "Waiting"]
        }
    }


@app.post("/api/analyze")
async def analyze_upload(
    video: UploadFile = File(...),
    stride: int = 9,
    n_frames: int = 9,
    _user: Optional[str] = Depends(require_user),
):
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {sorted(ALLOWED_EXTENSIONS)}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            total = 0
            while chunk := await video.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Video too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
                tmp.write(chunk)
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    job_id = uuid.uuid4().hex
    job = _new_job(video.filename, total, source_type="upload")
    job["job_id"] = job_id
    _JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_annotated_job,
        args=(job_id, tmp_path, True, n_frames, stride),
        daemon=True
    )
    thread.start()
    return {"job_id": job_id, "filename": video.filename}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str):
    video_path = os.path.join(ANNOTATED_DIR, f"{job_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(404, "Annotated video not found or still generating")
    return FileResponse(video_path, media_type="video/mp4", filename=f"volleyball_annotated_{job_id[:8]}.mp4")


def _get_video_info(file_path: str):
    try:
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        dur = round(frames / fps, 1) if fps > 0 else 0
        return {"duration_sec": dur, "fps": round(fps, 1), "frames": int(frames)}
    except Exception:
        return {"duration_sec": 0, "fps": 25.0, "frames": 0}


@app.get("/api/library")
def get_library():
    """Lists all available match videos from internet and dataset."""
    internet_files = []
    if os.path.isdir(INTERNET_DATA_DIR):
        for f in os.listdir(INTERNET_DATA_DIR):
            if f.endswith((".mp4", ".mov", ".avi", ".mkv")) and not f.endswith(".part"):
                p = os.path.join(INTERNET_DATA_DIR, f)
                size_mb = round(os.path.getsize(p) / (1024 * 1024), 1)
                info = _get_video_info(p)
                mtime = os.path.getmtime(p)
                internet_files.append({
                    "filename": f,
                    "category": "from internet",
                    "title": f.replace("_", " ").replace(".mp4", ""),
                    "size_mb": size_mb,
                    "duration_sec": info["duration_sec"],
                    "fps": info["fps"],
                    "mtime": mtime,
                    "stream_url": f"/api/library/stream/from_internet/{f}",
                })

    dataset_files = []
    if os.path.isdir(DATASET_DATA_DIR):
        for f in os.listdir(DATASET_DATA_DIR):
            if f.endswith((".mp4", ".mov", ".avi", ".mkv")):
                p = os.path.join(DATASET_DATA_DIR, f)
                size_mb = round(os.path.getsize(p) / (1024 * 1024), 1)
                info = _get_video_info(p)
                mtime = os.path.getmtime(p)
                dataset_files.append({
                    "filename": f,
                    "category": "from our dataset",
                    "title": f.replace("_", " ").replace(".mp4", ""),
                    "size_mb": size_mb,
                    "duration_sec": info["duration_sec"],
                    "fps": info["fps"],
                    "mtime": mtime,
                    "stream_url": f"/api/library/stream/from_dataset/{f}",
                })

    # Sort internet by newest downloaded first (mtime desc), dataset by longest first
    internet_files.sort(key=lambda x: -x["mtime"])
    dataset_files.sort(key=lambda x: -x["duration_sec"])

    return {
        "internet_matches": internet_files,
        "dataset_plays": dataset_files,
        "total_videos": len(internet_files) + len(dataset_files),
    }


@app.get("/api/library/stream/{folder}/{filename}")
def stream_library_video(folder: str, filename: str):
    if folder == "from_internet":
        path = os.path.join(INTERNET_DATA_DIR, filename)
    elif folder == "from_dataset":
        path = os.path.join(DATASET_DATA_DIR, filename)
    else:
        raise HTTPException(400, "Invalid folder")

    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


class AnalyzeLibraryRequest(BaseModel):
    category: str  # "from internet" or "from our dataset"
    filename: str
    stride: int = 9
    n_frames: int = 9


@app.post("/api/library/analyze")
def analyze_library_video(req: AnalyzeLibraryRequest, _user: Optional[str] = Depends(require_user)):
    if req.category == "from internet" or req.category == "from_internet":
        vpath = os.path.join(INTERNET_DATA_DIR, req.filename)
    elif req.category == "from our dataset" or req.category == "from_dataset":
        vpath = os.path.join(DATASET_DATA_DIR, req.filename)
    else:
        raise HTTPException(400, f"Invalid category: {req.category}")

    if not os.path.exists(vpath):
        raise HTTPException(404, f"File {req.filename} not found in library")

    size_bytes = os.path.getsize(vpath)
    job_id = uuid.uuid4().hex
    job = _new_job(req.filename, size_bytes, source_type="library")
    job["job_id"] = job_id
    _JOBS[job_id] = job

    # Don't delete library video upon completion (delete_input=False)
    thread = threading.Thread(
        target=_run_annotated_job,
        args=(job_id, vpath, False, req.n_frames, req.stride),
        daemon=True
    )
    thread.start()
    return {"job_id": job_id, "filename": req.filename}


class YouTubeDownloadRequest(BaseModel):
    url: Optional[str] = None
    query: Optional[str] = None
    name: Optional[str] = None
    max_duration_sec: Optional[int] = None


@app.post("/api/analyze/url")
def analyze_url(req: YouTubeDownloadRequest, _user: Optional[str] = Depends(require_user)):
    """Downloads a video from YouTube / web URL and immediately runs full AI analysis pipeline."""
    if not YOUTUBE_ENABLED:
        raise HTTPException(403, YOUTUBE_OFF_MSG)
    from src.inference.download_youtube_matches import download_url, search_and_download

    if not req.url and not req.query:
        raise HTTPException(400, "Provide either url or query")

    job_id = uuid.uuid4().hex
    display_title = req.name or (req.url if req.url else req.query)
    job = _new_job(display_title, 0, source_type="url")
    job["job_id"] = job_id
    job["status"] = "processing"
    job["phase"] = "Downloading match footage from URL..."
    job["progress"] = 5
    _JOBS[job_id] = job

    max_sec = req.max_duration_sec if (req.max_duration_sec is not None and req.max_duration_sec > 0) else None

    def _worker():
        try:
            if req.url:
                res = download_url(req.url, out_name=req.name, out_dir=INTERNET_DATA_DIR, max_seconds=max_sec)
            else:
                res = search_and_download(req.query, out_name=req.name, out_dir=INTERNET_DATA_DIR, max_seconds=max_sec)

            vpath = res["file"]
            job["filename"] = res["filename"]
            job["size_bytes"] = os.path.getsize(vpath) if os.path.exists(vpath) else 0
            job["phase"] = "Detecting Players & Extracting Features"
            job["progress"] = 10

            _run_annotated_job(job_id, vpath, delete_input=False)
        except Exception as e:
            traceback.print_exc()
            job["status"] = "error"
            job["phase"] = "Failed to download / process URL"
            job["error"] = str(e)
            job["finished_at"] = time.time()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return {"job_id": job_id, "title": display_title}


_DOWNLOADS = {}  # download_id -> dict


@app.get("/api/youtube/progress/{download_id}")
def get_download_progress(download_id: str):
    if download_id not in _DOWNLOADS:
        raise HTTPException(404, "Download not found")
    return _DOWNLOADS[download_id]


@app.post("/api/youtube/download")
def download_youtube_video(req: YouTubeDownloadRequest, background_tasks: BackgroundTasks):
    if not YOUTUBE_ENABLED:
        raise HTTPException(403, YOUTUBE_OFF_MSG)
    from src.inference.download_youtube_matches import download_url, search_and_download

    if not req.url and not req.query:
        raise HTTPException(400, "Provide either url or query")

    download_id = uuid.uuid4().hex
    _DOWNLOADS[download_id] = {
        "download_id": download_id,
        "status": "starting",
        "percent": 0.0,
        "speed": "Connecting...",
        "eta": "--",
        "filename": None,
        "error": None,
    }

    max_sec = req.max_duration_sec if (req.max_duration_sec is not None and req.max_duration_sec > 0) else None

    def _bg_download():
        def _prog(info):
            _DOWNLOADS[download_id].update(info)

        try:
            if req.url:
                res = download_url(req.url, out_name=req.name, out_dir=INTERNET_DATA_DIR, max_seconds=max_sec, progress_callback=_prog)
            else:
                res = search_and_download(req.query, out_name=req.name, out_dir=INTERNET_DATA_DIR, max_seconds=max_sec, progress_callback=_prog)
            _DOWNLOADS[download_id]["status"] = "finished"
            _DOWNLOADS[download_id]["percent"] = 100.0
            _DOWNLOADS[download_id]["filename"] = res["filename"]
            _DOWNLOADS[download_id]["title"] = res.get("title", res["filename"])
        except Exception as e:
            traceback.print_exc()
            _DOWNLOADS[download_id]["status"] = "error"
            _DOWNLOADS[download_id]["error"] = str(e)

    background_tasks.add_task(_bg_download)
    return {
        "status": "started",
        "download_id": download_id,
        "message": "Downloading YouTube match with real-time progress"
    }


# Mount static frontend
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
