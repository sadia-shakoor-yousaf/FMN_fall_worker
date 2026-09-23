"""Pose keypoints → 5 motion features used by the live LSTM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO-17 indices used by YOLO-pose
NOSE, L_HIP, R_HIP = 0, 11, 12

FEATURE_COLS = [
    "posture_angle_deg",
    "body_height",
    "drop_speed",
    "body_shape",
    "pose_confidence",
]


@dataclass
class FrameFeatures:
    posture_angle_deg: float  # ~0 standing, larger when lying
    body_height: float  # 1 = high in frame, 0 = near floor
    drop_speed: float  # positive = moving down the image
    body_shape: float  # bbox width / height
    pose_confidence: float
    valid: bool


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def angle_from_vertical_deg(hip: np.ndarray, nose: np.ndarray) -> float:
    """Angle between hip→nose and image-up (0, -1)."""
    vec = nose - hip
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return float("nan")
    vec = vec / n
    up = np.array([0.0, -1.0])
    cos = float(np.clip(np.dot(vec, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def features_from_pose(
    xy: np.ndarray,
    conf: np.ndarray,
    bbox_xyxy: np.ndarray,
    frame_h: int,
    prev_height: float | None,
) -> FrameFeatures:
    """
    xy: (17, 2) pixel coords
    conf: (17,)
    bbox_xyxy: (4,) x1,y1,x2,y2
    """
    kconf = np.asarray(conf, dtype=np.float64).reshape(-1)
    pose_confidence = float(np.nanmean(kconf)) if kconf.size else 0.0
    if (
        kconf.size < 17
        or kconf[NOSE] < 0.2
        or kconf[L_HIP] < 0.2
        or kconf[R_HIP] < 0.2
    ):
        return FrameFeatures(
            posture_angle_deg=float("nan"),
            body_height=float("nan"),
            drop_speed=0.0,
            body_shape=float("nan"),
            pose_confidence=pose_confidence,
            valid=False,
        )

    xy = np.asarray(xy, dtype=np.float64).reshape(17, 2)
    hip = _mid(xy[L_HIP], xy[R_HIP])
    nose = xy[NOSE]
    angle = angle_from_vertical_deg(hip, nose)

    cog_y = (hip[1] + nose[1]) / 2.0
    body_height = 1.0 - float(np.clip(cog_y / max(frame_h, 1), 0.0, 1.0))

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    body_shape = bw / bh

    drop_speed = 0.0
    if prev_height is not None and np.isfinite(prev_height) and np.isfinite(body_height):
        drop_speed = float(prev_height - body_height)

    return FrameFeatures(
        posture_angle_deg=angle,
        body_height=body_height,
        drop_speed=drop_speed,
        body_shape=body_shape,
        pose_confidence=pose_confidence,
        valid=True,
    )


def to_vector(f: FrameFeatures) -> np.ndarray:
    return np.array(
        [
            f.posture_angle_deg,
            f.body_height,
            f.drop_speed,
            f.body_shape,
            f.pose_confidence,
        ],
        dtype=np.float64,
    )
