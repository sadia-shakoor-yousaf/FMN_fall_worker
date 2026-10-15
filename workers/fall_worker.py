"""
ForgetMeNot care-PC fall worker.

USB webcam (or video file) → YOLO-pose → 5 features → LSTM + NEED → alert
→ optional POST to FMN event API (mock or real).

  python workers/fall_worker.py --cam 0 --patient-id 1
  python workers/fall_worker.py --source path/to/clip.mp4 --api-url http://127.0.0.1:5155 --patient-id 1
  python workers/fall_worker.py --cam 0 --api-url ""   # local alert only (no POST)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fall_cv.camera import enumerate_named_cameras, open_camera, resolve_camera_index
from fall_cv.features import features_from_pose, to_vector
from fall_cv.fmn_client import DEFAULT_BASE_URL, FmnClientError, post_event
from fall_cv.live import DEFAULT_NEED, LiveFallDetector, load_lstm_checkpoint
from fall_cv.pose import largest_person, load_pose_model

DEFAULT_CKPT = ROOT / "artifacts" / "fall_lstm_urfd_all.pt"
DEFAULT_LOG_DIR = ROOT / "data" / "live_logs"
ALERT_PRE_S = 3.0
ALERT_POST_S = 4.0


def _setup_log(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"fall_worker_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    log = logging.getLogger("fall_worker")
    log.info("log file: %s", log_path)
    return log


def _draw_overlay(
    frame: np.ndarray,
    *,
    pred: int,
    conf: float,
    run: int,
    need: int,
    fps: float,
    alert: bool,
) -> np.ndarray:
    out = frame.copy()
    status = "FALL?" if pred == 1 else "ok"
    color = (0, 0, 255) if pred == 1 else (0, 180, 0)
    lines = [
        f"LSTM: {status}  conf={conf:.2f}  NEED {run}/{need}",
        f"FPS: {fps:.1f}",
    ]
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28
    if alert:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 8)
        cv2.putText(
            out,
            "FALL ALERT",
            (12, out.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
    return out


def _open_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter: {path}")
    return writer


def _write_clip(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    writer = _open_writer(path, frames[0], fps)
    h, w = frames[0].shape[:2]
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h))
        writer.write(fr)
    writer.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ForgetMeNot live fall worker")
    p.add_argument(
        "--cam",
        type=int,
        default=None,
        help="Camera index. Default: USB/JLC webcam (not the laptop camera).",
    )
    p.add_argument(
        "--cam-name",
        type=str,
        default=None,
        help='Substring of the device name, e.g. "USB 2.0" or "JLC"',
    )
    p.add_argument(
        "--list-cams",
        action="store_true",
        help="Print camera indexes and names, then exit",
    )
    p.add_argument("--width", type=int, default=1920, help="Capture width (JLC 1080p)")
    p.add_argument("--height", type=int, default=1080, help="Capture height")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file instead of camera (demo / offline test)",
    )
    p.add_argument(
        "--patient-id",
        type=int,
        default=None,
        help="Patient integer id (required when posting to API)",
    )
    p.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"FMN base URL for POST /api/v1/events (default {DEFAULT_BASE_URL}). "
        'Empty string "" disables POST.',
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help="LSTM checkpoint (default: artifacts/fall_lstm_urfd_all.pt)",
    )
    p.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    p.add_argument("--need", type=int, default=None, help=f"default {DEFAULT_NEED}")
    p.add_argument("--cooldown", type=float, default=60.0, help="seconds between alerts")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all). Raise if CPU FPS is low.",
    )
    p.add_argument("--no-preview", action="store_true", help="Headless (log only)")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Local logs + alert clips",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames (tests)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = _setup_log(args.log_dir)

    if args.list_cams:
        cams = enumerate_named_cameras()
        if not cams:
            log.error("no cameras found")
            return 1
        for idx, name in cams:
            log.info("cam %s: %s", idx, name)
        return 0

    if not args.ckpt.is_file():
        log.error(
            "checkpoint missing: %s — see artifacts/README.md",
            args.ckpt,
        )
        return 1

    api_url = (args.api_url or "").strip()
    post_enabled = bool(api_url)
    if post_enabled and not args.patient_id:
        log.warning(
            "POST enabled (%s) but --patient-id missing; alerts will not be posted",
            api_url,
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; using CPU")
        device = "cpu"

    lstm, mean, std, ckpt_need = load_lstm_checkpoint(args.ckpt, device=device)
    need = args.need if args.need is not None else ckpt_need
    detector = LiveFallDetector(
        lstm,
        mean,
        std,
        need=need,
        cooldown_s=args.cooldown,
        device=device,
    )
    pose = load_pose_model(args.pose_model)

    cam_label = None
    if args.source is not None:
        source = args.source
        cap = cv2.VideoCapture(source)
    else:
        try:
            cam_index, cam_label = resolve_camera_index(args.cam, args.cam_name)
        except RuntimeError as e:
            log.error("%s", e)
            return 1
        source = cam_index
        cap = open_camera(cam_index, args.width, args.height)
    if not cap.isOpened():
        log.error("cannot open source: %s", source)
        return 1
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if cam_label:
        log.info("camera %s (%s) %sx%s", source, cam_label, got_w, got_h)

    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if cam_fps < 1:
        cam_fps = 30.0
    skip = max(int(args.skip), 1)
    write_fps = max(cam_fps / skip, 1.0)
    pre_frames = max(int(ALERT_PRE_S * write_fps), 1)
    post_frames = max(int(ALERT_POST_S * write_fps), 1)
    ring: deque[np.ndarray] = deque(maxlen=pre_frames)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_path = args.log_dir / f"session_{run_stamp}.mp4"
    session_writer: cv2.VideoWriter | None = None

    log.info(
        "start source=%s patient_id=%s need=%s cooldown=%.0fs preview=%s skip=%s "
        "api_url=%s",
        source,
        args.patient_id or "(none)",
        need,
        args.cooldown,
        not args.no_preview,
        skip,
        api_url or "(disabled)",
    )
    log.info("full annotated session → %s", session_path)

    prev_h: float | None = None
    t0 = time.perf_counter()
    n_proc = 0
    n_read = 0
    fps_ema = 0.0
    alert_flash_until = 0.0
    pending_alert: dict | None = None

    def _flush_alert(pending: dict) -> None:
        _write_clip(pending["frames"], pending["path"], write_fps)
        log.info(
            "alert clip saved (%d frames, ~%.1fs): %s",
            len(pending["frames"]),
            len(pending["frames"]) / write_fps,
            pending["path"],
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.info("end of stream")
                break
            n_read += 1
            if (n_read - 1) % skip != 0:
                continue

            t_frame = time.perf_counter()
            h, _w = frame.shape[:2]
            pred = pose.predict(
                frame, verbose=False, device=device, imgsz=args.imgsz
            )[0]
            got = largest_person(pred)
            plotted = pred.plot()

            if got is None:
                feat_vec = np.array(
                    [np.nan, np.nan, 0.0, np.nan, 0.0], dtype=np.float64
                )
                prev_h = None
            else:
                xy, conf, box = got
                feat = features_from_pose(xy, conf, box, h, prev_h)
                feat_vec = to_vector(feat)
                prev_h = feat.body_height if feat.valid else None

            now = time.perf_counter()
            event = detector.push(feat_vec, now_s=now)

            n_proc += 1
            inst = 1.0 / max(time.perf_counter() - t_frame, 1e-6)
            fps_ema = inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * inst)

            flashing = now < alert_flash_until
            if event is not None and pending_alert is None:
                alert_flash_until = now + 2.5
                flashing = True
                clip_path = args.log_dir / f"alert_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                pending_alert = {
                    "path": clip_path,
                    "frames": list(ring),
                    "post_left": post_frames,
                    "seen_alert_frame": False,
                    "conf": event.confidence,
                    "frame_idx": event.frame_idx,
                }
                log.warning(
                    "FALL ALERT patient_id=%s conf=%.3f frame=%d "
                    "(recording +%.0fs post-roll → %s)",
                    args.patient_id or "(none)",
                    event.confidence,
                    event.frame_idx,
                    ALERT_POST_S,
                    clip_path,
                )
                if post_enabled and args.patient_id:
                    try:
                        posted = post_event(
                            api_url,
                            patient_id=args.patient_id,
                            event_type="fall_detected",
                            label="floor_down",
                            confidence=float(event.confidence),
                            source="room_cam",
                        )
                        log.info(
                            "POST fall_detected ok id=%s",
                            posted.get("id", "?"),
                        )
                    except FmnClientError as e:
                        log.warning("POST fall_detected failed: %s", e)
                elif post_enabled and not args.patient_id:
                    log.warning("skip POST: --patient-id required")

            vis = _draw_overlay(
                plotted,
                pred=detector.last_pred,
                conf=detector.last_confidence,
                run=detector.consecutive_hits,
                need=need,
                fps=fps_ema,
                alert=flashing,
            )
            ring.append(vis)

            if session_writer is None:
                session_writer = _open_writer(session_path, vis, write_fps)
            session_writer.write(vis)

            if pending_alert is not None:
                pending_alert["frames"].append(vis)
                if not pending_alert["seen_alert_frame"]:
                    pending_alert["seen_alert_frame"] = True
                else:
                    pending_alert["post_left"] -= 1
                    if pending_alert["post_left"] <= 0:
                        _flush_alert(pending_alert)
                        pending_alert = None

            if not args.no_preview:
                cv2.imshow("ForgetMeNot fall worker", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    log.info("quit by user")
                    break

            if args.max_frames is not None and n_proc >= args.max_frames:
                log.info("hit --max-frames=%s", args.max_frames)
                break
    finally:
        if pending_alert is not None:
            _flush_alert(pending_alert)
            pending_alert = None
        if session_writer is not None:
            session_writer.release()
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    avg = n_proc / elapsed if elapsed > 0 else 0.0
    log.info(
        "done processed=%d read=%d elapsed=%.1fs avg_FPS=%.2f ema_FPS=%.2f",
        n_proc,
        n_read,
        elapsed,
        avg,
        fps_ema,
    )
    if session_path.is_file():
        log.info("full annotated video: %s", session_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
