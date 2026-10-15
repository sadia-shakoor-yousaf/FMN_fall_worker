"""HTTP client for ForgetMeNot event API (mock or real)."""

from __future__ import annotations

from typing import Any

import httpx

# Same process as: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 5155
DEFAULT_BASE_URL = "http://127.0.0.1:5155"
EVENTS_PATH = "/api/v1/events"


class FmnClientError(Exception):
    """Raised when the event POST fails (network or HTTP error)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def post_event(
    base_url: str,
    *,
    patient_id: int,
    event_type: str = "fall_detected",
    label: str = "floor_down",
    confidence: float = 0.0,
    source: str = "room_cam",
    timeout: float = 5.0,
) -> dict[str, Any]:
    """
    POST one activity event to the single FMN backend (port 5155).

    Returns the JSON body on 2xx.
    Raises FmnClientError on timeout, connection failure, or non-2xx.
    """
    url = base_url.rstrip("/") + EVENTS_PATH
    payload = {
        "patient_id": int(patient_id),
        "type": event_type,
        "label": label,
        "confidence": min(1.0, max(0.0, float(confidence))),
        "source": source,
    }
    # trust_env=False: a machine HTTP(S)_PROXY must not swallow 127.0.0.1.
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            resp = client.post(url, json=payload)
    except httpx.TimeoutException as e:
        raise FmnClientError(f"timeout posting to {url}") from e
    except httpx.HTTPError as e:
        raise FmnClientError(f"HTTP error posting to {url}: {e}") from e

    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise FmnClientError(
            f"POST {url} → {resp.status_code}: {detail}",
            status_code=resp.status_code,
        )
    body = resp.json()
    if not isinstance(body, dict):
        raise FmnClientError(
            f"POST {url} → {resp.status_code}: expected a JSON object",
            status_code=resp.status_code,
        )
    return body
