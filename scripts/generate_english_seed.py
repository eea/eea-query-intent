"""Generate the English seed corpus from the volto-searchlib policy corpus.

The seed converts the 118 labeled query rows from
classifyQueryIntent.corpus.test.js into the JSONL dataset format so the
benchmark starts from the exact same policy decisions the frontend shipped.

Usage:
    python scripts/generate_english_seed.py [corpus_test_js] [output_jsonl]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_CORPUS = (
    Path(__file__).resolve().parent.parent.parent
    / "eea-website-frontend"
    / "src"
    / "addons"
    / "volto-searchlib"
    / "searchlib"
    / "components"
    / "AnswerBox"
    / "classifyQueryIntent.corpus.test.js"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "data" / "seed" / "english.jsonl"
)

# Matches lines like: ['What is the main cause of air pollution?', 'question'],
# or: ["Who's the capital of France", 'question'],
ROW_PATTERN = re.compile(
    r"^\s*\[(?:'(?P<q1>(?:[^'\\]|\\.)*)'|\"(?P<q2>(?:[^\"\\]|\\.)*)\"),"
    r"\s*'(?P<intent>[a-z]+)'\],\s*$"
)


def unescape(text: str) -> str:
    return text.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def main() -> int:
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    rows: list[tuple[str, str]] = []
    seen_texts: set[str] = set()
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        match = ROW_PATTERN.match(line)
        if not match:
            continue
        text = unescape(match.group("q1") or match.group("q2") or "")
        intent = match.group("intent")
        if intent not in {"question", "exploratory", "claim", "retrieval", "unknown"}:
            continue
        if text in seen_texts:
            raise SystemExit(f"duplicate corpus text: {text!r}")
        seen_texts.add(text)
        rows.append((text, intent))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, (text, intent) in enumerate(rows, start=1):
            record = {
                "id": f"en-{index:04d}",
                "language": "en",
                "text": text,
                "intent": intent,
                "template_id": f"template-en-{index:04d}",
                "source_type": "human_authored",
                "review_status": "policy_reviewed",
                "split": "train",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
