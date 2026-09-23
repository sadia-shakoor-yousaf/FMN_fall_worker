"""
POST one fall_detected event with no camera (API smoke test).

  python scripts/smoke_post.py
  python scripts/smoke_post.py --api-url https://YOUR_HOST --patient-id 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fall_cv.fmn_client import DEFAULT_BASE_URL, FmnClientError, post_event


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test POST /api/v1/events")
    p.add_argument("--api-url", default=DEFAULT_BASE_URL)
    p.add_argument("--patient-id", type=int, default=1)
    p.add_argument("--confidence", type=float, default=0.91)
    p.add_argument("--label", default="floor_down")
    p.add_argument("--source", default="room_cam")
    args = p.parse_args()

    try:
        out = post_event(
            args.api_url,
            patient_id=args.patient_id,
            event_type="fall_detected",
            label=args.label,
            confidence=args.confidence,
            source=args.source,
        )
    except FmnClientError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
