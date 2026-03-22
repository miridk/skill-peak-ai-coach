# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

After completing any meaningful unit of work — a bug fix, a feature change, a tuning session — commit and push immediately. Never leave work uncommitted at the end of a session.

```bash
git add <specific files>
git commit -m "short, clear description of what changed and why"
git push
```

Commit messages should be specific: `fix occlusion EMA freeze during player overlap` is good, `update code` is not. Never use `git add -A` or `git add .` without checking `git status` first to avoid committing large model files (`.pt`, `.task`) or output folders (`save/`, `exports/`, `metrics_out/`).

## Environment Setup

Python 3.11 is required. The venv is at `.venv/`. PyTorch is installed with CUDA 12.4 support.

```bash
# Activate venv (bash)
source .venv/Scripts/activate

# Verify GPU is working
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

To rebuild the environment from scratch (PowerShell):
```powershell
winget install Python.Python.3.11
Remove-Item .venv -Recurse -Force
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Running the Pipeline

The full analysis pipeline has these stages, run in order:

**Stage 1 – Cut raw footage into rallies** (`01_Cutting Tool/skill_peak_clip_cutter.py`)

**Stage 2 – Extract frames for training data** (`02_Frame Ripper for AI coach training/`)
```bash
bash "02_Frame Ripper for AI coach training/run_pipeline.sh"
```

**Stage 3 – Generate dataset labels** (`03_Dataset generation/`)
```bash
python "03_Dataset generation/auto_label_to_mot.py" --video input.mp4 --model yolov8n.pt --output dataset_out --calibration calibration/min_fil.json --draw-preview
```

**Stage 4 – Main tracker** (`V2/badminton_analyzer.py`) — the core of the project:
```bash
cd V2
python badminton_analyzer.py
# A file dialog opens to select the video. Then click 4 court corners (TL, TR, BR, BL), then click each player.
```
Outputs go to `V2/save/` (annotated video) and `V2/exports/` (CSV + JSONL per run).

**Stage 5 – Metrics and coaching report** (run from repo root, reads latest CSV from `exports/`):
```bash
python metrics.py          # generates heatmaps, speed plots, HTML report → metrics_out/
python generate_events.py  # detects coaching events from series CSVs → metrics_out/events/
```

**Auto-clipping** (`TODO_Auto Clip/`):
```bash
cd "TODO_Auto Clip"
RUN_ALL.bat   # runs auto_segments.py then clip_and_prep.py
```

## Architecture Overview

### V2/badminton_analyzer.py — the main tracker (~1700 lines)

This is a single-file real-time badminton tracking pipeline. The data flow per frame is:

```
VideoCapture (FrameReader thread) → YOLO11m (FP16, GPU) → sv.ByteTrack
→ per-detection: CLIP crop encoding (GPU, FP16) + color histogram + MediaPipe pose (CPU, ThreadPoolExecutor)
→ IdentityManager.assign() → court metrics + drawing → VideoWriter + CSV/JSONL
```

**Key architectural components:**

- **`IdentityManager`** — the heart of the system. Maintains 4 stable player IDs (`PLAYER_IDS = [1,2,3,4]`) across the entire video, fighting against ByteTrack's raw ID swaps. Uses a Hungarian algorithm cost matrix combining 8+ signals: CLIP visual similarity, motion prediction, color histograms, pose signature, court side, depth role, direction history, and edge proximity.

- **`IdentityTrack` dataclass** — per-player state. Holds EMA features, memory banks (CLIP/color/pose), velocity history (300 frames), occlusion recovery snapshots, and side/role preferences.

- **Two-layer tracking**: ByteTrack handles frame-to-frame association (fast), IdentityManager handles long-term identity (robust to occlusion, reappearance, similar-looking players).

- **Calibration**: A perspective homography `H` maps pixel coordinates → court meters (6.10m × 13.40m). Stored in `calibration/` as JSON. The `COURT_SETUP_NAME` constant controls which file is loaded/saved. On first run with a new setup, the user clicks 4 court corners.

- **Bootstrap**: On the first frame with ≥4 detections, the user clicks each player to initialize stable IDs. All subsequent assignment is automatic.

### Occlusion Handling

When two players overlap (`bbox IOU ≥ OCCLUSION_IOU_THRESH`), several protections activate:
- **Position freeze**: tracks dead-reckon along pre-occlusion velocity instead of updating from the noisy merged detection
- **Feature freeze**: both EMA and memory banks stop updating (prevents identity bleed)
- **Snapshot**: the pre-occlusion CLIP/color/pose features are saved the moment overlap starts
- **Post-occlusion recovery** (`POST_OCCLUSION_RECOVERY_FRAMES = 25`): for 25 frames after separation, cost matrix adds an extra term comparing detections to the pre-occlusion snapshot

### Cost Function Weights (tunable in CONFIG section)

All weights starting with `W_` control the assignment cost matrix. The key ones to tune for ID stability:
- `W_CLIP = 0.34` — visual appearance (most reliable long-term signal)
- `W_DIRECTION = 0.28` — trajectory consistency
- `W_POSE = 0.20` — body pose
- `W_MOTION = 0.18` — predicted position
- `DIRECTION_COST = 1.35` — penalty multiplier for motion in opposite direction

`SWITCH_CONFIRM_FRAMES = 8` — how many consecutive frames an ID must "want to switch" before it actually switches. This is the primary anti-swap mechanism.

### Output Format

The CSV/JSONL from each run contains one row per detected player per frame with: `frame_idx`, `timestamp_s`, `stable_id` (1-4), court coordinates `x_m`/`y_m`, `zone` (e.g. `FAR-FRONT-LEFT`), `speed_kmh`, `distance_m`, `knee_angle_avg`, `ready_flag`, `ready_pct`.

`metrics.py` reads the latest CSV from `exports/`, writes per-player series CSVs to `metrics_out/series/` which `generate_events.py` then reads.

### Required Model Files

Both files must be present in `V2/` (not included in repo):
- `yolo11m.pt` — YOLO11 medium, person detection
- `pose_landmarker_full.task` — MediaPipe pose model

CLIP model (`ViT-B-32`, `laion2b_s34b_b79k`) downloads automatically from open_clip on first run.
