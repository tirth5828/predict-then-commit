"""Unit tests for visible-stream metrics."""

from ptc.metrics import detokenized_erasure_step, lcp_length, normalized_detokenized_erasure, total_detokenized_erasure


def test_lcp_length() -> None:
    assert lcp_length("Die Bank", "Die Bank wird") == len("Die Bank")
    assert lcp_length("Das Ufer", "Die Bank") == 1
    assert lcp_length("abc", "xyz") == 0


def test_detokenized_erasure_step_append_only() -> None:
    assert detokenized_erasure_step("Die Bank", "Die Bank wird") == 0


def test_detokenized_erasure_step_retraction() -> None:
    assert detokenized_erasure_step("Das Ufer", "Die Bank") == len("Das Ufer") - 1


def test_total_erasure() -> None:
    stream = ["", "Das Ufer", "Die Bank", "Die Bank wird"]
    assert total_detokenized_erasure(stream) == len("Das Ufer") - 1
    assert normalized_detokenized_erasure(stream) > 0.0
