"""Live smoke test for the running eea-query-intent service.

Usage:
    uv run python scripts/smoke_service.py [http://127.0.0.1:8100]
"""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8100"

QUERIES = [
    # (query, expected eligible)
    ("What are the main sources of air pollution in Europe?", True),
    ("Was sind die Hauptquellen der Luftverschmutzung in Europa?", True),
    ("Quelles sont les principales sources de pollution de l'air en Europe ?", True),
    ("¿Cuáles son las principales fuentes de contaminación del aire en Europa?", True),
    ("Quais são as principais fontes de poluição do ar na Europa?", True),
    ("Jakie są główne źródła zanieczyszczenia powietrza w Europie?", True),
    ("Avrupa'da hava kirliliğinin başlıca kaynakları nelerdir?", True),
    ("Hvað eru helstu uppsprettur loftmengunar í Evrópu?", False),  # abstain p~0.93
    ("L'inquinamento atmosferico causa problemi di salute", False),  # abstain p~0.92
    ("Oorzaken van klimaatverandering", False),  # abstain p~0.81
    ("climate", False),
    ("SOER 2025", False),
    ("water quality", False),
    ("PDF sur la qualité de l'eau", False),
    ("waste statistics", False),
    ("https://eea.example/air", False),  # URL guard
    ("", False),
    ("kwalità arja pollution", False),
]


def main() -> int:
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    health = json.load(urllib.request.urlopen(f"{base}/health", timeout=10))
    print(f"health: {health}\n")

    ok = 0
    for query, expected in QUERIES:
        body = json.dumps({"query": query}).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/v1/classify",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = json.load(urllib.request.urlopen(request, timeout=30))
        got = response["eligible"]
        marker = "ok" if got == expected else "XX"
        ok += got == expected
        label = query if len(query) <= 52 else query[:49] + "..."
        print(
            f"  [{marker}] elig={str(got):5s} "
            f"intent={response['intent']:<12s} "
            f"p={response['eligible_probability']:.2f} "
            f"abst={response['abstained']!s:5s} "
            f"{response['latency_ms']:>6}ms  {label!r}"
        )
    print(f"\n{ok}/{len(QUERIES)} matched expectations")
    return 0 if ok == len(QUERIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
