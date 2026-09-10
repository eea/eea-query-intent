"""Serialize the hand-authored English bank to JSONL.

The query text comes verbatim from ``english_bank_data.py`` (authored, not
slot-filled). This script only adds the record envelope: a stable id /
template_id (``enx-NNNN``), the en language, source/review metadata, and a
deterministic 90/10 train/calibration split.

Output: data/english/expanded_v1.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from english_bank_data import (  # noqa: E402
    CLAIM,
    EXPLORATORY,
    QUESTION,
    RETRIEVAL,
    UNKNOWN,
)

OUT = ROOT / "data" / "english" / "expanded_v1.jsonl"
INTENT_BANKS = (
    ("question", QUESTION),
    ("exploratory", EXPLORATORY),
    ("claim", CLAIM),
    ("retrieval", RETRIEVAL),
    ("unknown", UNKNOWN),
)


def _split(template_id: str) -> str:
    digest = hashlib.sha256(template_id.encode("utf-8")).digest()
    return "calibration" if digest[0] % 10 == 0 else "train"


def main() -> int:
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    n = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for intent, texts in INTENT_BANKS:
            for text in texts:
                key = " ".join(text.split()).lower()
                if key in seen:
                    continue
                seen.add(key)
                n += 1
                template_id = f"enx-{n:04d}"
                counts[intent] += 1
                handle.write(
                    json.dumps(
                        {
                            "id": template_id,
                            "language": "en",
                            "text": " ".join(text.split()),
                            "intent": intent,
                            "template_id": template_id,
                            "source_type": "synthetic_generated",
                            "review_status": "policy_reviewed",
                            "split": _split(template_id),
                        }
                    )
                    + "\n"
                )
    print(f"wrote {OUT} ({n} rows)")
    for intent, _ in INTENT_BANKS:
        print(f"  {intent:12s} {counts.get(intent, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
