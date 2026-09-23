# Fall worker pipeline

How the care-PC process turns camera frames into a `fall_detected` event.

Also see the system diagram: [`care_pc_flowchart.svg`](care_pc_flowchart.svg).

---

## High-level flow

```text
Camera / video file
        │
        ▼
   YOLO-pose (person + keypoints)
        │
        ▼
   5 features per frame
   (angle, height, drop speed, shape, confidence)
        │
        ▼
   Rolling window (20 frames, stride 10)
        │
        ▼
   LSTM → fall / not-fall + confidence
        │
        ▼
   NEED consecutive fall windows (default 2)
        │
        ▼
   Cooldown (default 60s) ──► skip duplicate alerts
        │
        ▼
   POST /api/v1/events   +   optional local alert clip
```

---

## Modules in this repo

| Path | Role |
| --- | --- |
| `workers/fall_worker.py` | Entrypoint: capture, overlay, logging, POST |
| `fall_cv/pose.py` | Load YOLO-pose; pick largest person |
| `fall_cv/features.py` | Keypoints → 5 floats |
| `fall_cv/lstm_model.py` | Small LSTM architecture |
| `fall_cv/live.py` | Window buffer, NEED, cooldown, checkpoint load |
| `fall_cv/fmn_client.py` | HTTP POST client |
| `mock_api/app.py` | Local stand-in for the FMN event endpoint |
| `artifacts/fall_lstm_urfd_all.pt` | Trained LSTM weights + normalize stats |

---

## Alert policy (important for testers)

- **NEED:** several consecutive LSTM “fall” windows before we alert (cuts brief crouches).
- **Cooldown:** after an alert, wait before posting again (default 60s).
- **POST failure:** logged as a warning; the camera loop does **not** crash.
- **Video:** session + alert MP4s stay under `data/live_logs/` — never uploaded by this worker.

---

## CLI knobs you’ll use most

| Flag | Meaning |
| --- | --- |
| `--cam 0` | USB camera index |
| `--source clip.mp4` | File instead of camera |
| `--api-url URL` | FMN base URL (empty `""` = no POST) |
| `--patient-id N` | Integer `patients.id` |
| `--device cpu\|cuda` | Inference device |
| `--skip N` | Process every Nth frame if CPU is slow |
| `--no-preview` | Headless |
| `--cooldown 60` | Seconds between alerts |
| `--need 2` | Override persistence |
