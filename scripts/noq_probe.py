"""Side-by-side probe: current setfit-v4 vs the noq-short candidate.

Replicates the service eligibility logic exactly
(probabilities = predict_proba, eligible_probability = sum over
question/exploratory/claim, gate at 0.98) and classifies three
batteries:

  NOQ     - no-question-mark fact-lookup patterns that abstained before
  SHORT   - 2-4 word short questions (the length cliff)
  CONTROL - keyword searches (must stay abstained), long questions
            (must stay eligible), out-of-domain (must stay abstained)

Usage: uv run python scripts/noq_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "models" / "setfit"
CANDIDATE = ROOT / "models" / "setfit-noq-short"
THRESHOLD = 0.98

NOQ = [
    "what are the co2 emissions in europe",
    "what is the water framework directive",
    "why are co2 emissions increasing in europe",
    "was sind die co2 emissionen in europa",
    "quelles sont les émissions de co2 en europe",
    "cuales son las emisiones de co2 en europa",
    "quali sono le emissioni di co2 in europa",
    "jakie są emisje co2 w europie",
]
SHORT = [
    "what is ets?",
    "what is ets",
    "air quality?",
    "why forests?",
    "biodiversity loss?",
    "what is the eea?",
    "forest fires?",
    "plastic waste?",
    "why are forests important",
    "how is co2 measured",
    "do wetlands filter water",
    "was ist ets?",
    "qu'est-ce que l'ets?",
    "que es el ets?",
]
CONTROL = [
    "water framework directive",
    "co2 emissions in europe",
    "what is the eu emissions trading system?",
    "where is romania located?",
]


class Model:
    def __init__(self, name: str, path: Path, lowercase: bool = False) -> None:
        from setfit import SetFitModel

        manifest = json.loads((path / "manifest.json").read_text("utf-8"))
        self.name = name
        self.labels = tuple(manifest["labels"])
        self.eligible = tuple(manifest["eligible_labels"])
        self.lowercase = lowercase
        self.model = SetFitModel.from_pretrained(str(path), device="mps")
        print(f"loaded {name} from {path}", flush=True)

    def classify(self, text: str) -> tuple[str, float]:
        if self.lowercase:
            text = text.lower()
        probs = self.model.predict_proba([text], as_numpy=True)[0]
        prob_map = dict(zip(self.labels, (float(p) for p in probs), strict=True))
        top = max(prob_map, key=prob_map.get)
        eligible = min(1.0, max(0.0, sum(prob_map[label] for label in self.eligible)))
        return top, eligible


def run(m: Model, probes: list[str]) -> list[tuple[str, float, bool]]:
    return [
        (text, m.classify(text)[1], m.classify(text)[1] >= THRESHOLD) for text in probes
    ]


def main() -> int:
    if not CANDIDATE.exists():
        print(f"candidate model missing: {CANDIDATE}")
        return 1
    current = Model("current setfit-v4", CURRENT)
    candidate = Model("noq-short candidate", CANDIDATE, lowercase=True)

    for title, probes in [
        ("NO-QUESTION-MARK", NOQ),
        ("SHORT QUESTIONS", SHORT),
        ("CONTROL", CONTROL),
    ]:
        print(f"\n=== {title} ===")
        print(f"{'probe':44} {'cur':>6} {'elig':>5}  {'cand':>6} {'elig':>5}  delta")
        for probe in probes:
            _, c_prob, c_ok = run(current, [probe])[0]
            _, k_prob, k_ok = run(candidate, [probe])[0]
            flag = ""
            if c_ok != k_ok:
                flag = "  <-- CHANGED"
            print(
                f"{probe[:44]:44} {c_prob:6.3f} {'Y' if c_ok else 'n':>5}  "
                f"{k_prob:6.3f} {'Y' if k_ok else 'n':>5}  "
                f"{k_prob - c_prob:+.3f}{flag}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
