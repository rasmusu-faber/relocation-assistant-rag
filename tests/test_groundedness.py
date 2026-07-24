"""Tests for the groundedness sentence splitter (pure Python, no LLM/ML stack)."""
from __future__ import annotations

from eval.groundedness import split_sentences


def test_splits_on_sentence_punctuation():
    sentences = split_sentences("Go to the office. Bring your passport with you.")
    assert sentences == [
        "Go to the office.",
        "Bring your passport with you.",
    ]


def test_strips_numbered_list_markers():
    """Enumeration ("1.", "2.") must not become standalone one-token 'sentences'."""
    text = "1. Go to the municipal office.\n2. Submit the application form."
    sentences = split_sentences(text)
    assert sentences == [
        "Go to the municipal office.",
        "Submit the application form.",
    ]
    assert all(s not in {"1.", "2."} for s in sentences)


def test_strips_bullet_markers():
    text = "- A valid passport or ID card\n* Proof of the legal basis for the stay"
    sentences = split_sentences(text)
    assert sentences == [
        "A valid passport or ID card",
        "Proof of the legal basis for the stay",
    ]


def test_drops_fragments_below_min_words():
    # "OK." is one word -> dropped; the longer claim is kept.
    sentences = split_sentences("OK. You must register within thirty days of arrival.")
    assert sentences == ["You must register within thirty days of arrival."]


def test_empty_or_marker_only_input_yields_nothing():
    assert split_sentences("") == []
    assert split_sentences("   \n  1.  \n - \n") == []
