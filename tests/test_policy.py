from eea_query_intent.policy import evaluate_local_policy


def test_empty_query_fails_closed() -> None:
    decision = evaluate_local_policy("  \n ")

    assert decision.should_classify is False
    assert decision.eligible is False
    assert decision.reason == "empty"


def test_query_at_word_limit_is_sent_to_classifier() -> None:
    query = " ".join(f"word{index}" for index in range(20))

    decision = evaluate_local_policy(query)

    assert decision.should_classify is True
    assert decision.reason == "classify"


def test_query_over_word_limit_fails_closed() -> None:
    query = " ".join(f"word{index}" for index in range(21))

    decision = evaluate_local_policy(query)

    assert decision.should_classify is False
    assert decision.eligible is False
    assert decision.reason == "too_long"


def test_pasted_url_fails_closed_without_the_model() -> None:
    for query in ("https://eea.example/air", "http://eea.example", "www.eea.example"):
        decision = evaluate_local_policy(query)
        assert decision.should_classify is False
        assert decision.eligible is False
        assert decision.reason == "url"


def test_query_mentioning_a_site_is_not_treated_as_a_url() -> None:
    decision = evaluate_local_policy("What does the EEA website say about PM2.5?")

    assert decision.should_classify is True
    assert decision.reason == "classify"


def test_word_count_handles_european_diacritics_and_apostrophes() -> None:
    query = "Cén fáth a bhfuil cáilíocht an aeir ag dul in olcas?"

    decision = evaluate_local_policy(query, max_words=12)

    assert decision.word_count == 11
    assert decision.should_classify is True
