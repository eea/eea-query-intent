"""Deterministic policy guards that run before any linguistic model.

Empty and overlong input are input validation, not classification, so they
stay local and synchronous. Every guard fails closed: if the query must not
be sent to the classifier, the query is not AI-eligible.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_WORDS = 20


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    word_count: int
    should_classify: bool
    eligible: bool
    reason: str


def _count_words(query: str) -> int:
    """Count whitespace-delimited tokens containing a letter or digit.

    Punctuation attached to a token does not start a new word, so European
    diacritics, apostrophes, and trailing question marks are handled without
    per-language rules.
    """
    return sum(1 for token in query.split() if any(ch.isalnum() for ch in token))


def evaluate_local_policy(
    query: str, *, max_words: int = DEFAULT_MAX_WORDS
) -> PolicyDecision:
    if not isinstance(query, str) or not query.strip():
        return PolicyDecision(
            word_count=0,
            should_classify=False,
            eligible=False,
            reason="empty",
        )

    word_count = _count_words(query)
    if word_count == 0:
        return PolicyDecision(
            word_count=0,
            should_classify=False,
            eligible=False,
            reason="empty",
        )

    if word_count > max_words:
        return PolicyDecision(
            word_count=word_count,
            should_classify=False,
            eligible=False,
            reason="too_long",
        )

    return PolicyDecision(
        word_count=word_count,
        should_classify=True,
        eligible=False,
        reason="classify",
    )
