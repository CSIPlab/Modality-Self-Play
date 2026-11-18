"""Tests for the one-real-anchor rule.

A source sample containing both modalities must be used through exactly one
of them; it must never appear as the real anchor in two subsets.
"""

from src.data.multimodal_dataset import CompositionDataset


class _Subset:
    """Minimal stand-in for CompositionSubset."""

    def __init__(self, name, keys):
        self.name = name
        self.samples = [{"key": k} for k in keys]
        self.sample_key_fn = lambda sample: sample["key"]


def _audit(*subsets):
    dataset = object.__new__(CompositionDataset)
    dataset.subsets = list(subsets)
    return CompositionDataset.audit_one_real_anchor(dataset)


def test_disjoint_anchors_pass():
    assert _audit(_Subset("vision", ["img1", "img2"]),
                  _Subset("audio", ["aud1", "aud2"])) == {}


def test_shared_anchor_is_reported():
    violations = _audit(_Subset("vision", ["clip7", "img2"]),
                        _Subset("audio", ["clip7", "aud2"]))
    assert violations == {"clip7": ["vision", "audio"]}


def test_repeat_within_one_subset_is_not_a_violation():
    # The rule is about a sample anchoring in two modalities, not about
    # the same key appearing twice within a single modality's data.
    assert _audit(_Subset("vision", ["img1", "img1"])) == {}


def test_empty_keys_ignored():
    assert _audit(_Subset("vision", ["", None]),
                  _Subset("audio", ["", None])) == {}


def test_eval_benchmarks_are_not_trainable_formats():
    """The benchmarks the paper evaluates on must never be training formats."""
    from src.data.multimodal_dataset import EVAL_ONLY_QA_FORMATS, QA_PARSERS

    for benchmark in ("avqa", "music_avqa", "omnibench"):
        assert benchmark in QA_PARSERS, f"{benchmark} parser missing"
        assert benchmark in EVAL_ONLY_QA_FORMATS, f"{benchmark} is trainable"

    trainable = set(QA_PARSERS) - EVAL_ONLY_QA_FORMATS
    assert trainable, "no trainable formats left"
    assert not (trainable & EVAL_ONLY_QA_FORMATS)


def test_lenient_match_rejects_substring_false_positives():
    """Short answers must not match by containment."""
    from src.evaluation.metrics_multimodal import _lenient_match

    assert _lenient_match("no", "no")
    assert _lenient_match("one", "1")
    assert _lenient_match("two cats", "2 cats")
    # "no" is a substring of "nothing"; the old matcher scored this correct
    assert not _lenient_match("no", "nothing")
    assert not _lenient_match("one", "someone")
