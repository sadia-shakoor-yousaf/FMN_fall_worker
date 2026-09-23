"""
Mock ForgetMeNot event API for local contract testing.

No auth. In-memory store + JSON dump under data/mock_events/.

  uvicorn mock_api.app:app --host 127.0.0.1 --port 5155
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DUMP_DIR = ROOT / "data" / "mock_events"

EventType = Literal["fall_detected", "visitor_recognized", "visitor_unknown"]
SourceType = Literal["room_cam", "door_cam"]


class EventCreate(BaseModel):
    patient_id: int = Field(gt=0)
    type: EventType
    label: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    source: SourceType


class EventOut(EventCreate):
    id: str
    created_at: str


app = FastAPI(title="FMN mock events", version="0.1.0")
_store: list[EventOut] = []


def _dump() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    path = DUMP_DIR / "events.json"
    path.write_text(
        json.dumps([e.model_dump(mode="json") for e in _store], indent=2),
        encoding="utf-8",
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    body = await request.body()
    if not body:
        return JSONResponse(status_code=400, content={"detail": "empty body"})
    try:
        json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"detail": "malformed JSON"})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/events", status_code=201, response_model=EventOut)
def create_event(body: EventCreate) -> EventOut:
    event = EventOut(
        id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        patient_id=body.patient_id,
        type=body.type,
        label=body.label,
        confidence=body.confidence,
        source=body.source,
    )
    _store.append(event)
    _dump()
    return event


@app.get("/api/v1/events", response_model=list[EventOut])
def list_events() -> list[EventOut]:
    """Dev-only list. Not required of production FMN."""
    return list(_store)
