"""YOLO-pose helpers for the live worker."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO


def largest_person(result) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (keypoints_xy, keypoint_conf, bbox_xyxy) for the largest person box."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None
    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(areas))
    return xy[i], conf[i], boxes[i]


def load_pose_model(model_name: str = "yolo11n-pose.pt") -> YOLO:
    """Load YOLO-pose once and reuse for every frame."""
    return YOLO(model_name)
