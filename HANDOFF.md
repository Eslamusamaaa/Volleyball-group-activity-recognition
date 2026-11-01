# Handoff — Volleyball Group Activity Recognition + Video→Timeline App

One place that explains everything built so far, why, and what's left —
written so someone picking this up cold (including future-you) doesn't
have to re-derive it from commit history.

## 1. What this project is

Reproduces and extends Ibrahim et al.'s CVPR 2016 "Hierarchical Deep
Temporal Model for Group Activity Recognition" on the official Volleyball
dataset (PyTorch, ResNet50 instead of the paper's AlexNet), then goes
beyond the paper with a mandatory per-player action head, and a full
video → timestamped-event-timeline app. Full baseline-by-baseline detail,
numbers, and ablation reasoning: [`README.md`](README.md). This file
covers status against the original project brief and the newer
web-app/deployment work the README doesn't fully narrate yet.

## 2. Status against the original brief (`IMPLEMENTATION_PROMPT.md`, not committed to git)

| Section | What it asks for | Status |
|---|---|---|
| 1–4 | Data loading, splits, 4GB-VRAM 3-phase training strategy | ✅ done |
| 5 | Baseline ladder B1→B8 | ✅ done, all evaluated — see `outputs/results/RESULTS.md` |
| 6 | Multi-task person-action head (B8-mt) | ✅ done (87.4% scene / 75.7% person). λ sweep: only 1.0 and 2.0 tried, not the full {0.3, 0.5, 1.0, 2.0} — not blocking, 1.0 already strong |
| 7.1–7.2 | Frame ordering, player-identity-across-frames | ✅ done (real IoU-matched tracking, static-box fallback also implemented and compared) |
| 7.3 | Per-video FPS extraction | ✅ done — `src/inference/video_io.py`, `cv2.CAP_PROP_FPS`, never assumed |
| 8 | Ball component | ❌ not started. No ball annotations exist anywhere in the dataset (verified). Only used by Section 10's optional "attribute winpoint to nearest player" detail — not required for the timeline itself |
| 9 | Full evaluation (confusion matrices, per-class P/R/F1) | ✅ done for every baseline |
| **10** | **Video → timestamped event timeline** | ✅ **all 4 stages built and wired end-to-end this session** (was 0% at the start of this session) |
| 11 | Repo structure | ✅ matches, `src/inference/` now filled in |

## 3. What got built this session, in order

### 3.1 Pipeline diagrams (README visuals)
Replaced the old icon-only B1–B8-mt pipeline diagrams with ones built from
real dataset frames/crops (`src/evaluation/make_pipeline_diagrams_v2.py`).
Purely documentation — no model/behavior change.

### 3.2 Section 10, stage 1 — person detector
- First pass: stock COCO YOLOv8n. Measured, not assumed: 77.8% match vs.
  ground truth on an 18-clip sweep (also fires on referees/staff).
- **Fine-tuned instead** on this project's own ~2,150 train-split clips'
  ground-truth player boxes (single class `player`).
  `src/inference/prepare_yolo_dataset.py` exports the data,
  `src/inference/train_detector.py` runs the fine-tune (30 epochs, RTX
  3050, `workers=2` — the default `workers=8` crashed with a Windows
  `cv2 OutOfMemoryError` on this machine's 16GB RAM mid-validation).
  Result: 95.3% precision / 94.2% recall / 0.98 mAP50 on held-out val.
  Weights: `outputs/checkpoints/yolov8n_player/weights/best.pt`.

### 3.3 Section 10, stage 2 — tracker
`src/inference/tracker.py`: YOLO's built-in ByteTrack. Validated against
ground truth (not assumed good) the same way as stage 1:

| Detector | detector-only match | detector+tracker match |
|---|---|---|
| stock COCO YOLOv8n | 77.8% | 72.1% |
| fine-tuned (3.2) | — | **97.7%** |

A `max_players` top-N-by-confidence cap was tried as a cheaper fix before
the fine-tune and measured to make things *worse* (64.6%) — a mid-swing
player is often lower-confidence than a standing official, so confidence
alone can't separate them. Left in the code, off by default, documented
rather than silently dropped.

### 3.4 Section 10, stages 3–4 — sliding window + timeline
- `src/inference/model_io.py`: loads the B3-A backbone + B8-mt model
  exactly the way `train_temporal.py` builds them, so training checkpoints
  load here with zero surprises.
- `src/inference/sliding_window.py`: slides a 9-frame window over the
  tracked video; only players tracked through the *entire* window are
  used (a dropped track has no full sequence for LSTM1); crops → B3-A
  features → B8-mt → scene class + per-player action, per window.
- `src/inference/timeline.py`: frame→second conversion via the video's
  own FPS, merges consecutive same-class windows into one event, writes
  `{start_sec, end_sec, team, event, confidence}` JSON.
- **End-to-end smoke test**: stitched clip 94540's real 41 frames (ground
  truth `r_spike`) into a test video, ran all 4 stages — first event:
  `{"team": "right", "event": "spike", "confidence": 0.999}`, matching the
  real label. This is one integration smoke test, **not** a dataset-scale
  accuracy number for the pipeline as a whole, and not yet tested on a
  genuinely new raw video (none exists on disk — every source video here
  is already split into annotated clip frames).

### 3.5 Backend + frontend (this session, for the deployment plan)
- **Backend** (`backend/app.py`, FastAPI): loads both models once at
  startup (not per request), exposes `POST /api/analyze` (multipart video
  upload → timeline JSON) and `GET /api/health`, serves the frontend as
  static files from the same origin. Tested directly with `curl` — same
  JSON output as the CLI smoke test above.
- **Frontend** (`frontend/`): single static page, drag-and-drop or
  click-to-choose upload, calls `/api/analyze`, renders the returned
  events as a simple timeline list. No framework/build step — plain
  HTML/CSS/JS, served by the backend's `StaticFiles` mount.
- Verified: backend starts, `/api/health` returns `models_loaded: true`,
  `/api/analyze` returns correct events for the test video, frontend page
  renders and is wired to call the API (upload click-through not scripted
  end-to-end, since that opens a native OS file picker, but both halves
  are independently confirmed working).

## 4. Running it locally

```bash
cd "E:\volleyball datasets"
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000
```
Open `http://localhost:8000/` — upload a `.mp4`/`.avi`/`.mov`/`.mkv` and
click "Analyze video". Or hit the API directly:
```bash
curl -X POST http://localhost:8000/api/analyze -F "video=@/path/to/clip.mp4"
```
CLI-only, no server:
```bash
.venv\Scripts\python.exe -m src.inference.timeline "C:\path\to\video.mp4"
```

## 5. Deployment notes

- **`Dockerfile`** at repo root builds a CPU-only image. **Run `git lfs
  pull` before `docker build`** — the checkpoints under
  `outputs/checkpoints/` are Git LFS pointers until that runs; building
  without it ships broken (pointer-text) weight files that will fail to
  `torch.load()`.
- Inference falls back to CPU automatically if no GPU is present
  (`WindowPredictor` in `sliding_window.py` checks `torch.cuda.is_available()`)
  — it'll just be slower per video, nothing to configure.
- `CORSMiddleware` in `backend/app.py` currently allows all origins
  (`allow_origins=["*"]`) — fine for local dev, **tighten this to the
  real frontend's origin before deploying publicly.**
- Upload size is capped at 200MB (`MAX_UPLOAD_BYTES` in `backend/app.py`)
  — a short clip is fine, a full match recording is not; raise this
  deliberately if needed, with awareness that `/api/analyze` is currently
  **synchronous** (the request blocks until the whole video is processed).
  For a full match-length video this needs a background job queue
  (Celery/RQ/a simple polling status endpoint) instead of a blocking
  request — not built, since no real match-length video exists yet to
  size this against.
- No auth, no rate limiting, no HTTPS termination — add whatever the
  actual deployment target (a reverse proxy, a PaaS's built-in TLS, etc.)
  expects.

## 6. Known gaps / honest limitations

- **No real raw match video tested yet.** Every video source in this
  project is pre-split into annotated clip frames; the only end-to-end
  test is the one synthetic clip described in 3.4. Get a real match
  recording and run it through before trusting the pipeline's accuracy
  on genuinely new footage.
- **Stages 3–4 have no dataset-scale accuracy number**, unlike stages 1–2
  (which have an 18-clip IoU sweep) and the B1–B8-mt baselines (which have
  full test-set metrics). One smoke test confirms the pipeline *runs
  correctly*, not that its event boundaries/classifications are accurate
  at scale.
- **Ball component (Section 8) not started.** Only needed for attributing
  a `winpoint` to a specific nearest player; the timeline works without it.
- **λ sweep (Section 6)** only covers 1.0 and 2.0 of the suggested
  {0.3, 0.5, 1.0, 2.0}.
- **Synchronous upload endpoint** — see Section 5, needs a job queue for
  anything longer than a short clip.
- **Windows-specific gotcha already hit once**: `ultralytics` training
  defaults to 8 dataloader workers, which crashed with an out-of-memory
  error on this machine's 16GB RAM. `train_detector.py` already sets
  `workers=2` — don't revert that without re-checking available RAM.

## 7. File map (everything new this session)

```
src/inference/
  video_io.py             real per-video FPS + frame iteration (file or frame-dir)
  detector.py              YOLOv8n player detector (fine-tuned weights by default)
  tracker.py                ByteTrack wrapper + validate_on_clip() ground-truth check
  prepare_yolo_dataset.py    exports GT player boxes -> YOLO training format
  train_detector.py          fine-tunes YOLOv8n on this dataset's own players
  model_io.py                 loads B3-A backbone + B8-mt the way training does
  sliding_window.py            9-frame window -> B8-mt prediction per window
  timeline.py                   video -> JSON event timeline (CLI entry point)
backend/
  app.py                 FastAPI: /api/analyze, /api/health, serves frontend/
  requirements.txt
frontend/
  index.html, style.css, script.js   upload UI, no build step
Dockerfile                CPU deployment image (needs `git lfs pull` first)
outputs/checkpoints/
  yolov8n.pt                   stock COCO YOLOv8n (kept as a fallback/reference)
  yolov8n_player/weights/best.pt   fine-tuned detector (now the default)
outputs/results/
  tracker_validation.json   18-clip IoU match-rate sweep (stock vs. fine-tuned)
outputs/yolo_player_data/  (gitignored, regenerate with prepare_yolo_dataset.py)
```
