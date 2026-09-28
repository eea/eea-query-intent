"""Shared pipeline for the GPT QA scripts.

``gpt_train_qa.py`` and ``gpt_acceptance_qa.py`` run the same resumable
batch pipeline over their respective raw shards: load batch progress,
QA the unreviewed batches, apply ok/fix/drop verdicts, sanitize, top up
under-quota intents, finalize, check shortfalls, and write the report
log plus the final shard. The shared pipeline lives here so the two
scripts (and ``gpt_train_gen.py`` for text scanning) keep only their own
directories, quotas, prompts, and avoid sets.

Both scripts run as ``uv run python scripts/<name>.py <lang>``, so
``scripts/`` is on sys.path and this module imports directly.
"""

import json
from collections.abc import Callable
from pathlib import Path

from gpt_acceptance_gen import build_prompt, call_gpt

INTENTS = ("question", "exploratory", "claim", "retrieval", "unknown")
MAX_WORDS = 20
BATCH_SIZE = 200


def texts_from_dir(d: Path, exclude: frozenset[Path] = frozenset()) -> list[str]:
    """Casefolded ``text`` values of every ``*.jsonl`` row under ``d``."""
    texts: list[str] = []
    for path in d.glob("*.jsonl"):
        if path in exclude:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and "text" in rec:
                texts.append(rec["text"].casefold())
    return texts


def texts_from_dirs(
    dirs: list[Path], exclude: frozenset[Path] = frozenset()
) -> set[str]:
    seen: set[str] = set()
    for d in dirs:
        seen.update(texts_from_dir(d, exclude))
    return seen


def load_progress(path: Path) -> dict[str, dict[int, dict]]:
    """Return ``{batch_start: {row_index: verdict}}`` from prior runs.

    The QA pass is resumable: each completed batch is appended to the
    progress file, so a killed process restarts where it left off
    instead of re-paying for already-reviewed batches.
    """
    done: dict[str, dict[int, dict]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[str(rec["start"])] = {int(k): v for k, v in rec["verdicts"].items()}
    return done


def save_batch_progress(path: Path, start: int, verdicts: dict[int, dict]) -> None:
    rec = {"start": start, "verdicts": {str(k): v for k, v in verdicts.items()}}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        handle.flush()


def run_qa_batches(
    lang: str,
    rows: list[dict],
    progress: dict[str, dict[int, dict]],
    verdicts: dict[int, dict],
    save_path: Path,
    label: str,
    qa_batch: Callable[[list[tuple[int, str, str]], str], dict],
) -> None:
    """QA every unreviewed batch of ``rows``.

    Freshly reviewed batches are persisted to ``save_path`` and folded
    into both in-memory maps as they complete: ``progress`` so a later
    resume skips them and ``verdicts`` so the caller can apply the full
    set immediately (the original inline-main semantics).
    """
    for start in range(0, len(rows), BATCH_SIZE):
        if str(start) in progress:
            resumed = min(start + BATCH_SIZE, len(rows))
            print(
                f"{lang} {label} batch {start + 1}-{resumed}/{len(rows)} "
                f"resumed from progress",
                flush=True,
            )
            continue
        chunk = rows[start : start + BATCH_SIZE]
        numbered = [
            (start + j, rec["intent"], rec["text"]) for j, rec in enumerate(chunk)
        ]
        res = qa_batch(numbered, lang)
        by_i = {r["i"]: r for r in res.get("rows", [])}
        batch_verdicts: dict[int, dict] = {}
        for i, _intent, _text in numbered:
            v = by_i.get(i)
            verdict = (
                v
                if v is not None
                else {"v": "drop", "t": "", "r": "missing-from-output"}
            )
            batch_verdicts[i] = verdict
        save_batch_progress(save_path, start, batch_verdicts)
        progress[str(start)] = batch_verdicts
        verdicts.update(batch_verdicts)
        print(
            f"{lang} {label} batch {start + 1}-{start + len(chunk)}/{len(rows)}",
            flush=True,
        )


def apply_verdicts(
    rows: list[dict],
    verdicts: dict[int, dict],
    intents: dict[str, int],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    # apply verdicts, grouped by intent in original order
    kept: dict[str, list[str]] = {intent: [] for intent in intents}
    stats = {"ok": 0, "fix": 0, "drop": 0}
    for idx, rec in enumerate(rows):
        v = verdicts[idx]
        kind = v.get("v", "drop")
        if kind == "ok":
            kept[rec["intent"]].append(rec["text"].strip())
            stats["ok"] += 1
        elif kind == "fix":
            corrected = (v.get("t") or "").strip()
            if corrected:
                kept[rec["intent"]].append(corrected)
                stats["fix"] += 1
            else:
                stats["drop"] += 1
        else:
            stats["drop"] += 1
    return kept, stats


def sanitize_kept(kept: dict[str, list[str]], avoid: set[str]) -> None:
    # Sanitize kept BEFORE the top-up so it sees the true post-filter
    # count: the final pass below drops exactly these shapes, and if the
    # top-up ran first it would silently fall below target.
    for intent in kept:
        seen: set[str] = set()
        clean: list[str] = []
        for t in kept[intent]:
            key = t.casefold()
            if not t or len(t.split()) > MAX_WORDS or key in seen or key in avoid:
                continue
            seen.add(key)
            clean.append(t)
        kept[intent] = clean


def top_up_intent(
    lang: str,
    intent: str,
    n: int,
    kept_texts: list[str],
    avoid: set[str],
    qa_batch: Callable[[list[tuple[int, str, str]], str], dict],
) -> int:
    generated = call_gpt(build_prompt(lang, intent, n))
    res = qa_batch(
        [(j, intent, text) for j, text in enumerate(generated, start=1)], lang
    )
    by_i = {r["i"]: r for r in res.get("rows", [])}
    seen = {t.casefold() for t in kept_texts}
    added = 0
    for j, text in enumerate(generated, start=1):
        v = by_i.get(j)
        if v and v.get("v") in ("ok", "fix"):
            t = (v.get("t") or text).strip()
            if t and t.casefold() not in seen and t.casefold() not in avoid:
                kept_texts.append(t)
                seen.add(t.casefold())
                added += 1
    return added


def top_up(
    lang: str,
    kept: dict[str, list[str]],
    avoid: set[str],
    cycles: int,
    quotas: dict[str, int],
    qa_batch: Callable[[list[tuple[int, str, str]], str], dict],
) -> int:
    topups = 0
    for _cycle in range(cycles):
        need = {
            intent: quotas[intent] - len(texts)
            for intent, texts in kept.items()
            if len(texts) < quotas[intent]
        }
        if not need:
            break
        for intent, n in need.items():
            topups += top_up_intent(lang, intent, n, kept[intent], avoid, qa_batch)
        print(
            f"{lang} after top-up: "
            + ", ".join(f"{i}={len(t)}" for i, t in kept.items()),
            flush=True,
        )
    return topups


def finalize_kept(
    kept: dict[str, list[str]],
    quotas: dict[str, int],
    avoid: set[str] | None = None,
) -> tuple[dict[str, list[str]], int]:
    # final safety: dedupe (case-insensitive), cap word count, prune to
    # target; with an avoid set, also drop anything that collides with
    # another dataset and count it as leaked.
    final: dict[str, list[str]] = {}
    leaked = 0
    for intent, texts in kept.items():
        seen: set[str] = set()
        clean: list[str] = []
        for t in texts:
            key = t.casefold()
            if (
                not t
                or len(t.split()) > MAX_WORDS
                or key in seen
                or (avoid is not None and key in avoid)
            ):
                leaked += int(avoid is not None and key in avoid)
                continue
            seen.add(key)
            clean.append(t)
        final[intent] = clean[: quotas[intent]]
    return final, leaked


def shortfall_splits(
    counts: dict[str, int],
    quotas: dict[str, int],
    tolerance: dict[str, float] | None = None,
    default_tolerance: float = 0.05,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    # Quotas are data-shape targets, not hard requirements: the
    # collision-aware top-up can legitimately run dry when the model
    # keeps producing canonical phrases that hit the corpus avoid-set.
    # Shortfalls up to a per-intent tolerance are accepted with a loud
    # warning by the caller; more is a hard failure.
    shortfall = {
        intent: quotas[intent] - counts[intent]
        for intent in quotas
        if counts[intent] < quotas[intent]
    }
    tol = tolerance or {}
    tolerated = {
        intent: short
        for intent, short in shortfall.items()
        if counts[intent] >= (1 - tol.get(intent, default_tolerance)) * quotas[intent]
    }
    hard = {
        intent: short for intent, short in shortfall.items() if intent not in tolerated
    }
    return shortfall, tolerated, hard


def write_report_log(report_dir: Path, name: str, lang: str, log: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{name}_{lang}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )


def write_shard(
    out_path: Path,
    lang: str,
    final: dict[str, list[str]],
    id_prefix: str,
    split: str,
) -> Path:
    with out_path.open("w", encoding="utf-8") as handle:
        for intent in INTENTS:
            for seq, text in enumerate(final[intent], start=1):
                row_id = f"{id_prefix}-{lang}-{intent}-{seq:04d}"
                handle.write(
                    json.dumps(
                        {
                            "id": row_id,
                            "template_id": row_id,
                            "language": lang,
                            "text": text,
                            "intent": intent,
                            "source_type": "synthetic_generated",
                            "review_status": "llm_reviewed",
                            "split": split,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return out_path
