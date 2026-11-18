"""Utilities for split-aware multimodal evaluation metrics."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUBSET_ORDER = ["overall", "S1", "S2", "S3", "S4"]


_WORD2NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
_NUM2WORD = {v: k for k, v in _WORD2NUM.items()}
def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    text = text.strip().lower()
    text = re.sub(r"[.,!?;:\"'()\[\]]", "", text)
    text = text.replace("_", " ").replace("-", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
def _canonical_forms(text: str) -> set[str]:
    """All forms a normalised answer can take (number ↔ word, both)."""
    norm = normalize_answer(text)
    forms = {norm, norm.replace(" ", "_")}
    if norm in _WORD2NUM:
        forms.add(_WORD2NUM[norm])
    if norm in _NUM2WORD:
        forms.add(_NUM2WORD[norm])
    # Token-wise mapping for short multi-word phrases like "two cats"
    toks = norm.split()
    mapped = " ".join(_WORD2NUM.get(t, _NUM2WORD.get(t, t)) for t in toks)
    if mapped != norm:
        forms.add(mapped)
    return {f for f in forms if f}
def _lenient_match(pred: str, gt: str) -> bool:
    """Resilient EM with exact canonical-form matching.

    The earlier substring-based version produced false positives for
    short answers like "no" and "one". Canonicalization already handles
    underscores, whitespace, and number-word variants, so exact set
    intersection is the safer policy here.
    """
    p_forms = _canonical_forms(pred)
    g_forms = _canonical_forms(gt)
    if not p_forms or not g_forms:
        return False
    return not p_forms.isdisjoint(g_forms)



def load_split_index(split_path: str | Path) -> dict[tuple[str, Any], dict[str, Any]]:
    rows = json.loads(Path(split_path).read_text())
    index: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in rows:
        index[(row["dataset_split"], row["question_id"])] = row
    return index


def attach_split_metadata(records: list[dict[str, Any]], split_index: dict[tuple[str, Any], dict[str, Any]], dataset_split: str) -> None:
    for record in records:
        meta = split_index.get((dataset_split, record.get("question_id")))
        if meta is None:
            record["modality_split"] = "unknown"
            record["split_confidence"] = "unknown"
        else:
            record["modality_split"] = meta.get("assigned_split", "unknown")
            record["split_confidence"] = meta.get("confidence", "unknown")


def _acc_entry(correct: int = 0, total: int = 0) -> dict[str, float | int]:
    acc = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall_correct = sum(1 for r in records if r["correct"])
    overall_total = len(records)

    by_subset: dict[str, dict[str, float | int]] = {}
    for subset in ["S1", "S2", "S3", "S4", "unknown"]:
        subset_records = [r for r in records if r.get("modality_split") == subset]
        by_subset[subset] = _acc_entry(
            correct=sum(1 for r in subset_records if r["correct"]),
            total=len(subset_records),
        )

    by_qtype_raw: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in records:
        key = record.get("question_type", "unknown")
        by_qtype_raw[key]["total"] += 1
        by_qtype_raw[key]["correct"] += int(record["correct"])
    by_question_type = {
        key: _acc_entry(correct=value["correct"], total=value["total"])
        for key, value in sorted(by_qtype_raw.items())
    }

    return {
        "overall": _acc_entry(overall_correct, overall_total),
        "by_subset": by_subset,
        "by_question_type": by_question_type,
        "num_records": overall_total,
    }


def _get_accuracy(summary: dict[str, Any], subset: str) -> float:
    if subset == "overall":
        return float(summary["overall"]["accuracy"])
    return float(summary["by_subset"].get(subset, {}).get("accuracy", 0.0))


def compute_interaction_metrics(
    mode_summaries: dict[str, dict[str, Any]],
    va_mode: str,
    vision_mode: str = "vision_only",
    audio_mode: str = "audio_only",
    text_mode: str = "text_only",
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for subset in SUBSET_ORDER:
        va = _get_accuracy(mode_summaries[va_mode], subset)
        v0 = _get_accuracy(mode_summaries[vision_mode], subset)
        a0 = _get_accuracy(mode_summaries[audio_mode], subset)
        t0 = _get_accuracy(mode_summaries[text_mode], subset)
        metrics[subset] = {
            "va_accuracy": va,
            "vision_only_accuracy": v0,
            "audio_only_accuracy": a0,
            "text_only_accuracy": t0,
            "G": va - max(v0, a0),
            "I_va": va - v0 - a0 + t0,
            "Gamma": va - max(v0, a0),
        }
    return metrics


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


