"""Pick a Windows USB webcam (JLB / USB 2.0 Camera) instead of the laptop cam."""

from __future__ import annotations

import os

import cv2

# Laptop / IR cameras we skip when auto-picking a room webcam.
_SKIP = ("wide vision", "integrated", "infrared", "ir camera")
# "jlb" is the room webcam used on the care PC. "jlc" stays as a fallback
# for older device strings.
_PREFER = ("jlb", "jlc", "usb2.0", "usb 2.0", "uvc", "usb camera")


def enumerate_named_cameras() -> list[tuple[int, str]]:
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        return []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    try:
        cams = list(enumerate_cameras(backend))
    except Exception:
        cams = list(enumerate_cameras())
    return [(int(c.index), str(c.name or "")) for c in cams]


def resolve_camera_index(
    cam: int | None,
    cam_name: str | None,
) -> tuple[int, str]:
    named = enumerate_named_cameras()
    if cam_name:
        needle = cam_name.lower()
        for idx, name in named:
            if needle in name.lower():
                return idx, name
        have = ", ".join(f"{i}:{n}" for i, n in named) or "(none)"
        raise RuntimeError(f"no camera matching {cam_name!r}; have {have}")
    if cam is not None:
        name = next((n for i, n in named if i == cam), f"index {cam}")
        return cam, name
    # Prefer JLB over a generic "USB Camera" even when the laptop cam is index 0.
    for needle in _PREFER:
        for idx, name in named:
            low = name.lower()
            if needle in low and not any(s in low for s in _SKIP):
                return idx, name
    for idx, name in named:
        if not any(s in name.lower() for s in _SKIP):
            return idx, name
    if named:
        return named[0]
    return 0, "index 0"


def open_camera(index: int, width: int = 0, height: int = 0) -> cv2.VideoCapture:
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Width/height 0 keeps the webcam's own mode. Forcing 1920x1080 MJPG
    # stretches JLB frames sideways; the Windows Camera app does not do that.
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap
