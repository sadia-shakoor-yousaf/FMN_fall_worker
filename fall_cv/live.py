"""Live / streaming LSTM fall decisions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .features import FEATURE_COLS
from .lstm_model import FallLSTM

# Consecutive fall-windows required before an alert (reduces brief crouch FPs).
DEFAULT_NEED = 2

WINDOW = 20
STRIDE = 10
N_FEAT = len(FEATURE_COLS)


@dataclass
class AlertEvent:
    confidence: float
    need_hit: int
    frame_idx: int


def load_lstm_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[FallLSTM, np.ndarray, np.ndarray, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FallLSTM()
    model.load_state_dict(ckpt["state"])
    model.to(device)
    model.eval()
    mean = np.asarray(ckpt["mean"], dtype=np.float64).reshape(1, N_FEAT)
    std = np.asarray(ckpt["std"], dtype=np.float64).reshape(1, N_FEAT)
    need = int(ckpt.get("need", DEFAULT_NEED))
    return model, mean, std, need


def _fill_window(chunk: np.ndarray) -> np.ndarray | None:
    """Require mostly finite values; fill leftover NaNs with column median."""
    if np.isfinite(chunk).mean() < 0.7:
        return None
    out = np.where(np.isfinite(chunk), chunk, np.nan).astype(np.float64)
    for c in range(out.shape[1]):
        col = out[:, c]
        if not np.isfinite(col).all():
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = 0.0
            out[:, c] = np.where(np.isfinite(col), col, med)
    return out


class LiveFallDetector:
    """Rolling window → LSTM → NEED consecutive hits → cooldown."""

    def __init__(
        self,
        model: FallLSTM,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        need: int = DEFAULT_NEED,
        window: int = WINDOW,
        stride: int = STRIDE,
        cooldown_s: float = 60.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.mean = mean
        self.std = std
        self.need = need
        self.window = window
        self.stride = stride
        self.cooldown_s = cooldown_s
        self.device = device

        self._feats: deque[np.ndarray] = deque(maxlen=window)
        self._run = 0
        self._last_alert_t: float | None = None
        self._last_pred = 0
        self._last_conf = 0.0
        self.frame_idx = 0

    def reset(self) -> None:
        self._feats.clear()
        self._run = 0
        self._last_pred = 0
        self._last_conf = 0.0
        self.frame_idx = 0

    @property
    def last_pred(self) -> int:
        return self._last_pred

    @property
    def last_confidence(self) -> float:
        return self._last_conf

    @property
    def consecutive_hits(self) -> int:
        return self._run

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def _predict_window(self, chunk: np.ndarray) -> tuple[int, float]:
        filled = _fill_window(chunk)
        if filled is None:
            return 0, 0.0
        x = self._normalize(filled)
        tens = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tens)
            prob = torch.softmax(logits, dim=1)[0]
            pred = int(logits.argmax(dim=1).item())
            conf = float(prob[pred].item())
        return pred, conf

    def push(
        self,
        feat_vec: np.ndarray,
        now_s: float,
    ) -> AlertEvent | None:
        """
        Push one frame's 5-feature vector.
        Returns AlertEvent when NEED consecutive fall-windows fire and cooldown is clear.
        """
        vec = np.asarray(feat_vec, dtype=np.float64).reshape(N_FEAT)
        self._feats.append(vec)
        self.frame_idx += 1

        if len(self._feats) < self.window:
            return None
        offset = self.frame_idx - self.window
        if offset % self.stride != 0:
            return None

        chunk = np.stack(self._feats, axis=0)
        pred, conf = self._predict_window(chunk)
        self._last_pred = pred
        self._last_conf = conf

        if pred == 1:
            self._run += 1
        else:
            self._run = 0

        if self._run < self.need:
            return None

        if (
            self._last_alert_t is not None
            and (now_s - self._last_alert_t) < self.cooldown_s
        ):
            return None

        self._last_alert_t = now_s
        self._run = 0
        return AlertEvent(
            confidence=conf,
            need_hit=self.need,
            frame_idx=self.frame_idx,
        )
