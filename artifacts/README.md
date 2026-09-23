# Artifacts

| File | Purpose |
| --- | --- |
| `fall_lstm_urfd_all.pt` | Live LSTM checkpoint (weights + feature mean/std + default NEED). **Tracked in git** (~75 KB). |

## YOLO-pose weights (not tracked)

On first run Ultralytics downloads `yolo11n-pose.pt` if missing, **or** place a local copy in the repo root / cwd:

```text
yolo11n-pose.pt
```

Override path with:

```bash
python workers/fall_worker.py --pose-model path/to/yolo11n-pose.pt ...
```
