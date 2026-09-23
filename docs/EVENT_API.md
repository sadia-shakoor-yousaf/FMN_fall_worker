# FMN Event API — Backend Contract

**Caller:** care-PC fall worker (this repo)  
**Auth:** not required for this endpoint  
**Scope:** accept one event JSON, persist it, return the created row. Video never leaves the care PC.

---

## Context — what runs on the care PC (not on your servers)

ForgetMeNot caregivers use UUID; patients use integer `id`. On a home **care PC** this worker watches a room camera, decides a fall happened, then calls **only** `POST /api/v1/events`. Raw video and model inference stay on that machine.

### Fall detection (`source: room_cam` → `type: fall_detected`)

1. USB / room camera frames are read on the care PC.
2. **YOLO-pose** finds the person and body keypoints.
3. From pose we build a short sequence of motion features.
4. A small **LSTM** scores each window as fall vs not-fall.
5. We require **persistence** (several consecutive fall windows) before alerting.
6. On alert we POST:
   - `type`: `fall_detected`
   - `label`: typically `floor_down`
   - `confidence`: model score in `[0, 1]`
   - `source`: `room_cam`
   - `patient_id`: that patient’s integer PK
7. Optionally we save a short local clip for demo — **never uploaded** via this API.

### Face / visitor (same endpoint, future / sibling worker)

Same JSON shape with `source: door_cam` and `type` `visitor_recognized` or `visitor_unknown`. This repo ships the **fall** worker; visitor types are part of the shared contract.

### What this means for backend

| You receive | You do **not** receive / run |
| --- | --- |
| Small JSON event about a patient | Video frames or alert clips |
| `patient_id` as **int** | Caregiver UUID in this payload (derive via patient → caregiver) |
| Activity types for `activity_logs` | YOLO / LSTM on the server |

`care PC (detect) → POST /api/v1/events → insert activity_logs → (your FCM / UI path)`

---

## Endpoint

| | |
| --- | --- |
| Method | `POST` |
| Path | `/api/v1/events` |

---

## Request body

```json
{
  "patient_id": 1,
  "type": "fall_detected",
  "label": "floor_down",
  "confidence": 0.91,
  "source": "room_cam"
}
```

| Field | Type | Rules |
| --- | --- | --- |
| `patient_id` | integer | Existing patient primary key (`patients.id`) |
| `type` | string enum | `fall_detected` \| `visitor_recognized` \| `visitor_unknown` |
| `label` | string | Fall: `floor_down`. Face: visitor name or unknown label |
| `confidence` | float | `0.0`–`1.0` |
| `source` | string enum | `room_cam` \| `door_cam` |

---

## Success — `201 Created`

```json
{
  "id": "<activity_log id>",
  "created_at": "2026-09-23T09:00:00+00:00",
  "patient_id": 1,
  "type": "fall_detected",
  "label": "floor_down",
  "confidence": 0.91,
  "source": "room_cam"
}
```

---

## Errors

| Status | When |
| --- | --- |
| `400` | Empty body or malformed JSON |
| `422` | Invalid / missing fields, unknown enums, bad confidence, unknown `patient_id` |

---

## Server-side processing

1. Validate the body.
2. Resolve `patient_id` → patient (and linked caregiver). Reject if missing.
3. Insert into `activity_logs`.
4. Return `201` with the created row.

**Must land in `activity_logs`:** `fall_detected`, `visitor_recognized`, `visitor_unknown`.

FCM / UI reactions are out of scope for this contract — use your existing `activity_logs` path.

---

## Out of scope for this endpoint

- Video / clip upload
- Running detection models on the server
- Specifying FCM payload details
- `GET /api/v1/events` (mock-only helper in this repo)
