# ForgetMeNot — Care-PC Fall Worker

Local **fall detection worker** for the ForgetMeNot (FMN) FYP.  
Runs on a care-PC / laptop: watches a room camera (or a demo video), detects falls, and posts a small JSON event to the FMN backend.

This repo is meant for **backend (and frontend) integration testing**.  
Training notebooks and dataset experiments live in a separate thesis repo.

> **Backend teammates (start here):**  
> Step-by-step clone, Python setup, and how to hit your real API — written for beginners:  
> `[docs/BACKEND_README.md](docs/BACKEND_README.md)`

---

## What this does

```text
Room camera / video
  → YOLO-pose
  → motion features
  → LSTM + persistence (NEED)
  → POST /api/v1/events  { fall_detected, patient_id, ... }
```

- **Video stays on the care PC.** Only the event JSON is sent.
- `patient_id` **is an integer** (your `patients.id` PK).
- Push notifications (FCM) are **your** concern after you write `activity_logs`.

More detail:

- `[docs/PIPELINE.md](docs/PIPELINE.md)` — detection pipeline  
- `[docs/EVENT_API.md](docs/EVENT_API.md)` — backend contract  
- `[docs/care_pc_flowchart.svg](docs/care_pc_flowchart.svg)` — system diagram

---

## Requirements


| Item     | Notes                                                  |
| -------- | ------------------------------------------------------ |
| OS       | Windows / macOS / Linux                                |
| Python   | **3.11 or 3.12** (see `.python-version`)               |
| Hardware | CPU works; CUDA optional (`--device cuda`)             |
| Camera   | USB webcam **or** an `.mp4` fall clip for offline demo |


---

## Project layout

```text
FMN_fall_worker/
├── README.md
├── requirements.txt          # worker runtime
├── requirements-mock.txt     # + FastAPI mock server
├── artifacts/                # LSTM checkpoint
├── fall_cv/                  # pose, features, LSTM, HTTP client
├── workers/fall_worker.py    # main entrypoint
├── mock_api/                 # local POST /api/v1/events stand-in
├── scripts/smoke_post.py     # POST without camera
├── docs/                     # pipeline + API contract
└── data/live_logs/           # local session / alert clips (gitignored)
```

---

## Setup

```bash
git clone <this-repo-url> FMN_fall_worker
cd FMN_fall_worker

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: mock API for offline contract testing
pip install -r requirements-mock.txt
```

LSTM weights are already in `artifacts/fall_lstm_urfd_all.pt`.  
YOLO-pose weights download automatically on first run (or see `artifacts/README.md`).

---

## Quick start — mock API (no real backend yet)

**Terminal 1 — mock server**

```bash
uvicorn mock_api.app:app --host 127.0.0.1 --port 5155
```

**Terminal 2 — smoke POST (no camera)**

```bash
python scripts/smoke_post.py --api-url http://127.0.0.1:5155 --patient-id 1
```

**Or run the worker on a video / camera**

```bash
# video file
python workers/fall_worker.py --source path/to/fall_clip.mp4 --api-url http://127.0.0.1:5155 --patient-id 1

# live webcam
python workers/fall_worker.py --cam 0 --api-url http://127.0.0.1:5155 --patient-id 1
```

On alert you should see `POST fall_detected ok id=...` in the worker log.  
Mock events are also written to `data/mock_events/events.json`.

---

## Test against the **real** FMN backend

1. Implement `POST /api/v1/events` per `[docs/EVENT_API.md](docs/EVENT_API.md)`.
2. Use a real patient integer id from your DB.
3. Point the worker at your API base URL (no trailing path — worker appends `/api/v1/events`):

```bash
python workers/fall_worker.py ^
  --source path/to/fall_clip.mp4 ^
  --api-url https://YOUR_API_HOST ^
  --patient-id 1 ^
  --no-preview
```

(macOS/Linux: use `\` line continuations instead of `^`.)

1. Confirm a new `activity_logs` row:
  - `type = fall_detected`
  - `patient_id = <that int>`
  - `source = room_cam`
  - `label = floor_down`

**API-only smoke (still no camera):**

```bash
python scripts/smoke_post.py --api-url https://YOUR_API_HOST --patient-id 1
```

Equivalent curl:

```bash
curl -X POST https://YOUR_API_HOST/api/v1/events ^
  -H "Content-Type: application/json" ^
  -d "{\"patient_id\":1,\"type\":\"fall_detected\",\"label\":\"floor_down\",\"confidence\":0.91,\"source\":\"room_cam\"}"
```

---

## Useful flags


| Flag           | Example          | Meaning                          |
| -------------- | ---------------- | -------------------------------- |
| `--api-url`    | `https://api...` | Base URL; `""` disables POST     |
| `--patient-id` | `1`              | Required for POST                |
| `--source`     | `clip.mp4`       | Offline demo                     |
| `--cam`        | `0`              | Webcam index                     |
| `--device`     | `cpu` / `cuda`   | Inference device                 |
| `--skip`       | `2`              | Every Nth frame (help slow CPUs) |
| `--cooldown`   | `60`             | Seconds between alerts           |
| `--no-preview` |                  | Headless                         |


Local annotated session + alert clips land in `data/live_logs/`.

---

## Environment notes

- First YOLO run may download weights and feel slow; later runs are faster.
- On CPU, use `--skip 2` (or higher) if FPS is low.
- Preview window: press `q` or `Esc` to quit.
- Network errors on POST are warnings only — the worker keeps running.

---

## Out of scope (this repo)

- Training / URFD notebooks (thesis repo)
- Face / door-cam worker (same event contract; separate process later)
- Uploading video to the cloud

