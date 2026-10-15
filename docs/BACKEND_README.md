# Backend guide — ForgetMeNot Fall Worker

This guide is for the **FMN backend team**.  
You do **not** need to understand machine learning. You only need to:

1. Run this small program on a laptop.
2. Point it at **your** real API.
3. Confirm a fall event shows up in `activity_logs`.

If you have never cloned a GitHub repo or used Python before, follow this page top to bottom.

---

## What is this project? (in plain words)

ForgetMeNot has:

- A **caregiver app** (Flutter)
- A **backend API** (your FastAPI / Supabase stack)
- A **care-PC worker** (this repo) that watches a room camera in the patient’s home

When the worker thinks a fall happened, it sends a tiny JSON message to your server:

```http
POST /api/v1/events
```

Your job on the backend:

- Accept that JSON
- Save it into `activity_logs`
- (Later) notify the caregiver however you already do — FCM, etc.

**Important:** the worker does **not** upload video. Only the event JSON.

Full field list and rules: [`EVENT_API.md`](EVENT_API.md)

---

## What you care about vs what you can ignore

| Relevant to backend | You can ignore |
| --- | --- |
| `docs/EVENT_API.md` — request/response contract | How YOLO / LSTM work inside |
| `scripts/smoke_post.py` — test your endpoint with no camera | Training / research code (not in this repo) |
| `workers/fall_worker.py` — real fall → POST | `data/live_logs/` videos (local only) |
| `patient_id` = **integer** patient PK | Caregiver UUID (you resolve it from the patient row) |
| Inserting into `activity_logs` | Building the ML models |

Visitor / face events use the **same** endpoint shape later (`visitor_recognized`, `visitor_unknown`). This repo ships the **fall** worker.

---

## What you need installed first

Do this once on your laptop.

### 1) Git

Git lets you download the project from GitHub.

- Windows: install [Git for Windows](https://git-scm.com/download/win)
- macOS: install Xcode Command Line Tools, or Git from [git-scm.com](https://git-scm.com)
- Check it works. Open a terminal and type:

```bash
git --version
```

You should see a version number, not an error.

### 2) Python 3.11 or 3.12

- Windows: install from [python.org](https://www.python.org/downloads/)  
  During setup, tick **“Add python.exe to PATH”**.
- Check:

```bash
python --version
```

You want something like `Python 3.11.x` or `3.12.x`.  
On some Macs the command is `python3` instead of `python`.

### 3) A terminal

- Windows: **PowerShell** or **Command Prompt** (or Git Bash)
- macOS / Linux: **Terminal**

All commands below are typed in that terminal.

### 4) (Optional) A short fall video

For the full worker test you need either:

- a USB webcam, **or**
- any short `.mp4` where a person falls / lies down

You can still test your API **without** a camera using `smoke_post.py` (recommended first).

---

## Step-by-step: get the repo on your computer

### Step A — Create a folder for the project

Example (Windows):

```bash
cd %USERPROFILE%\Documents
mkdir FMN
cd FMN
```

macOS / Linux example:

```bash
cd ~
mkdir -p FMN
cd FMN
```

### Step B — Clone (download) the repo

Replace the URL with the real GitHub link you were given:

```bash
git clone https://github.com/YOUR_ORG/FMN_fall_worker.git
cd FMN_fall_worker
```

What this means:

- `git clone` = download a copy of the project
- `cd FMN_fall_worker` = enter that folder

You should now see files like `README.md`, `requirements.txt`, `workers/`, `docs/`.

---

## Step-by-step: create a Python “virtual environment”

A **virtual environment** (venv) is a private folder of Python packages for this project only.  
It avoids breaking other Python projects on your machine.

### Step C — Create the venv

Still inside `FMN_fall_worker`:

```bash
python -m venv .venv
```

If `python` fails, try:

```bash
python3 -m venv .venv
```

This creates a folder named `.venv`.

### Step D — Activate the venv

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run this once as Administrator, then retry activate:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt):**

```bash
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

When it worked, your prompt usually starts with `(.venv)`.

To leave the venv later:

```bash
deactivate
```

### Step E — Install dependencies

With the venv **activated**:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This can take several minutes (PyTorch / OpenCV are large). Wait until it finishes with no red errors.

Optional — only if you want the **local mock** API on your laptop:

```bash
pip install -r requirements-mock.txt
```

---

## Path 1 (start here): test YOUR real API with no camera

This is the fastest way to check your endpoint.

### What your backend must already have

- A running server with:

```http
POST /api/v1/events
```

- Body shape described in [`EVENT_API.md`](EVENT_API.md)
- A real patient row whose **integer** `id` you can use (example: `1`)

### Run the smoke script

Replace the URL and patient id:

**Windows (PowerShell):**

```bash
python scripts/smoke_post.py --api-url https://YOUR_API_HOST --patient-id 1
```

Examples of `--api-url`:

- `http://127.0.0.1:5155` (API on your own machine)
- `https://api.yourdomain.com` (deployed API)

Do **not** put `/api/v1/events` in `--api-url`.  
The script adds that path for you.  
So if your full URL is `https://x.com/api/v1/events`, pass only `https://x.com`.

### Success looks like

- Terminal prints JSON with an `id` and `created_at`
- In your database / Supabase, a new `activity_logs` row exists:
  - `type` = `fall_detected`
  - `patient_id` = the int you passed
  - `label` = `floor_down`
  - `source` = `room_cam`

### If it fails

| Symptom | Likely cause |
| --- | --- |
| `connection` / timeout | Wrong host, server not running, firewall |
| `404` | Path is not `/api/v1/events` |
| `422` | Body validation — check field types (`patient_id` must be int) |
| `400` | Empty / bad JSON handling on your side |

Same test with curl (optional):

```bash
curl -X POST https://YOUR_API_HOST/api/v1/events ^
  -H "Content-Type: application/json" ^
  -d "{\"patient_id\":1,\"type\":\"fall_detected\",\"label\":\"floor_down\",\"confidence\":0.91,\"source\":\"room_cam\"}"
```

(On macOS/Linux use `\` at end of lines instead of `^`, and a single-quoted `-d '...'` JSON string.)

---

## Path 2: run the real fall worker against YOUR API

Use this after smoke POST works. This uses the camera / a video file.

### Video file demo

```bash
python workers/fall_worker.py --source path\to\your_fall_clip.mp4 --api-url https://YOUR_API_HOST --patient-id 1 --no-preview
```

### Live webcam

```bash
python workers/fall_worker.py --cam 0 --api-url https://YOUR_API_HOST --patient-id 1
```

- `--cam 0` = first webcam. Try `1` if you have multiple cameras.
- Press `q` or `Esc` in the preview window to quit (if preview is on).
- On a slow CPU, add `--skip 2`.

### What you should see

1. Worker log line: `FALL ALERT ...`
2. Then: `POST fall_detected ok id=...`
3. Matching row in `activity_logs`

If POST fails, the worker **keeps running** and only prints a warning. Fix the API, then try again.

Local debug videos (not uploaded) appear under `data/live_logs/`.

---

## Path 3 (optional): practice with the mock API first

If your real API is not ready yet, this repo includes a fake server.

**Terminal 1:**

```bash
uvicorn mock_api.app:app --host 127.0.0.1 --port 5155
```

Leave it running.

**Terminal 2** (venv activated, same folder):

```bash
python scripts/smoke_post.py --api-url http://127.0.0.1:5155 --patient-id 1
```

Events are saved to `data/mock_events/events.json`.  
When your real API is ready, change `--api-url` to your host. Nothing else changes.

---

## Your backend checklist

- [ ] `POST /api/v1/events` implemented as in [`EVENT_API.md`](EVENT_API.md)
- [ ] `patient_id` accepted as **integer**
- [ ] Unknown patient → `422` (or your chosen validation error)
- [ ] Valid event → insert `activity_logs` → `201` + created row
- [ ] `smoke_post.py` succeeds against your server
- [ ] (Optional) worker with video/camera also posts successfully
- [ ] Caregiver notification path (FCM / Realtime) wired the same way as your other activity logs

You do **not** need to run YOLO or LSTM on the server.

---

## Common beginner mistakes

1. **Forgot to activate `.venv`** — `pip` / `python` then use the wrong install. Activate first (prompt shows `(.venv)`).
2. **Put the full path in `--api-url`** — use base URL only, not `.../api/v1/events`.
3. **Used a UUID for `patient_id`** — must be the patient’s **int** primary key.
4. **Ran commands from the wrong folder** — you must be inside `FMN_fall_worker` (where `requirements.txt` lives).
5. **First run is slow** — YOLO may download weights once; that is normal.

---

## Need more technical detail?

| Doc | For |
| --- | --- |
| [`EVENT_API.md`](EVENT_API.md) | Exact JSON contract |
| [`PIPELINE.md`](PIPELINE.md) | How the worker decides “fall” |
| [`../README.md`](../README.md) | Shorter project overview + flags |
| [`care_pc_flowchart.svg`](care_pc_flowchart.svg) | How care PC, app, and backend connect |

If something in this guide is unclear, ask the fall-detection teammate which `--api-url` and which patient id to use in your environment.
