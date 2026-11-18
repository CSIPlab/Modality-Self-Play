"""MCQ-corpus composition dataset for the cross-modal MCQ training regime.

Each MCQ file is a dict keyed by `<prefix>:<id>` topic identifiers
from the shared 519K topic pool (LLaVA / COCO image filenames,
AudioCaps ytids). Every value is a list of
multiple-choice QA entries with the schema:

    {
      "q": "What sound is heard?",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."]
                   or  ["...", "...", "...", "..."],
      "answer": "A",
      "answer_text": "..."
    }

The *question* is about one modality (e.g. audio) — determined by
which MCQ file the sample came from, passed in as `question_modality`.
The *anchor* modality (the one with the real feature) is determined
by the key prefix — so a question about audio can land on a sample
whose anchor is vision, which forces the LoRA to attend to the audio
proxy to answer. That's the composition-forcing training signal the
current 1-real-N-proxies contract lacks.

Real features are looked up per-modality from the existing caches on
this cluster; proxies come from `topic_{audio,vision}_proxy.pt`.

Usage sketch:

    subsets = [
        ("data/mcq/qa_audio_mcq.json",  "audio"),
        ("data/mcq/qa_vision_mcq.json", "vision"),
    ]
    ds = MCQCompositionDataset(
        mcq_subsets=subsets,
        real_features=real_features_by_modality,
        proxy_features=proxy_features_by_modality,
        tokenizer=tok,
        ...
    )
"""
from __future__ import annotations

import gzip
import json
import logging
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.multimodal_dataset import (
    MODALITY_DEFAULTS,
    mask_assistant_labels,
)
from src.models.multimodal_model import get_modality_placeholder


logger = logging.getLogger(__name__)


# Map topic-pool key prefix → anchor modality.
PREFIX_TO_ANCHOR = {
    "audio": "audio",
    "coco":  "vision",
    "llava": "vision",
}


def _strip_prefix(key: str) -> tuple[str, str]:
    """`audio:-2GlU3e0nTU` → ('audio', '-2GlU3e0nTU')."""
    if ":" in key:
        head, bare = key.split(":", 1)
        return head.strip(), bare.strip()
    return "", key


def _strip_ext(key: str) -> str:
    for ext in (".jpg", ".jpeg", ".png", ".wav", ".mp4", ".npy", ".obj", ".ply"):
        if key.endswith(ext):
            return key[:-len(ext)]
    return key


def _resolve_real_key(bare_key: str, anchor: str,
                      real_features: dict[str, torch.Tensor]) -> str | None:
    """Find a real-feature-cache key that matches `bare_key` for the
    given anchor modality. Returns the key to use or None if no match.

    Handles the common mismatches:
      * Vision cache uses basename like '000000000034.jpg'; topic-pool
        also uses basename → direct match.
      * Audio topic-pool uses bare ytid '-2GlU3e0nTU'; real cache uses
        ytid with optional '.wav' and/or '_<segment>' suffix. Try
        bare, bare+.wav, bare+segment variants.
    """
    if bare_key in real_features:
        return bare_key
    # Strip extension
    stripped = _strip_ext(bare_key)
    if stripped != bare_key and stripped in real_features:
        return stripped
    # Audio: try with .wav suffix (some caches keep it)
    if anchor == "audio":
        for cand in (bare_key + ".wav", stripped + ".wav"):
            if cand in real_features:
                return cand
        # Try any real-cache key that starts with this ytid (segmented audio)
        prefix = stripped + "_"
        for k in real_features.keys():
            if isinstance(k, str) and k.startswith(prefix):
                return k
    return None


def _resolve_proxy_key(bare_key: str, prefix: str,
                       proxy_features: dict[str, torch.Tensor]) -> str | None:
    """Look up a proxy feature. Proxy dicts may be keyed by the full
    `<prefix>:<id>` form or (post alias-expansion) by bare `<id>`.
    Try both."""
    full = f"{prefix}:{bare_key}" if prefix else bare_key
    if full in proxy_features:
        return full
    if bare_key in proxy_features:
        return bare_key
    stripped = _strip_ext(bare_key)
    if stripped != bare_key and stripped in proxy_features:
        return stripped
    return None


def _parse_mcq_entry(entry: dict) -> dict | None:
    """Validate + normalise one MCQ entry. Returns a normalised dict or
    None if the entry is malformed."""
    q = (entry.get("q") or entry.get("question") or "").strip()
    opts = entry.get("options") or entry.get("multi_choice") or []
    ans_text = (entry.get("answer_text") or "").strip()
    letter = (entry.get("answer") or "").strip()
    if not q or len(opts) != 4 or not ans_text:
        return None
    # Strip leading letter markers like "A) ..." if present.
    norm_opts = []
    for o in opts:
        s = str(o).strip()
        m = re.match(r"^[A-Da-d][\)\.\:\s-]+(.+)$", s)
        norm_opts.append(m.group(1).strip() if m else s)
    if not letter:
        # Reconstruct from answer_text.
        for i, o in enumerate(norm_opts):
            if o.lower() == ans_text.lower():
                letter = "ABCD"[i]
                break
    return {"q": q, "options": norm_opts,
            "answer_text": ans_text, "letter": letter or ""}


def _build_mcq_prompt(question: str, options: list[str],
                      all_placeholders: dict[str, str],
                      modality_order: list[str],
                      system_prompt: str,
                      answer_text: str) -> str:
    """Build an MCQ-format chat prompt. All active modalities are
    prefixed in a fixed order (anchor-invariant prompt) so the LoRA
    sees consistent attention positions regardless of which modality
    actually carries the real feature."""
    prefix = " ".join(all_placeholders[m] for m in modality_order)
    letters = "ABCD"
    opt_block = "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(options))
    user = (
        f"{prefix}\n{question}\n"
        f"Choose the correct answer from:\n{opt_block}"
    )
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
    parts.append(f"<|im_start|>assistant\n{answer_text}<|im_end|>\n")
    return "".join(parts)


class MCQCompositionDataset(Dataset):
    """Prefix-anchored MCQ composition dataset.

    Parameters
    ----------
    mcq_subsets : list[tuple[str, str]]
        (mcq_file_path, question_modality) pairs. Each MCQ file holds
        questions about one modality; the key prefix determines the
        anchor modality per-sample.
    real_features : dict[str, dict[str, torch.Tensor]]
        Per-modality real-feature caches.
    proxy_features : dict[str, dict[str, torch.Tensor]]
        Per-modality topic-proxy caches. Alias-expanded upstream so
        both `<prefix>:<id>` and bare `<id>` keys hit.
    tokenizer : transformers tokenizer
    active_modalities : list[str]
        Which modalities appear in each sample's prompt. Typically
        ['audio', 'vision'].
    num_tokens : dict[str, int]
        Per-modality injection-token count.
    feature_dims : dict[str, int]
        Per-modality real feature dim.
    max_seq_length : int
    system_prompt : str
    strict_coverage : bool
        Drop samples whose anchor real feature or any non-anchor proxy
        is missing.
    """

    def __init__(
        self,
        mcq_subsets: list[tuple[str, str]],
        real_features: dict[str, dict[str, torch.Tensor]],
        proxy_features: dict[str, dict[str, torch.Tensor]],
        tokenizer,
        active_modalities: list[str] | None = None,
        num_tokens: dict[str, int] | None = None,
        feature_dims: dict[str, int] | None = None,
        max_seq_length: int = 512,
        system_prompt: str = "You are a helpful assistant.",
        strict_coverage: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.real_features = real_features
        self.proxy_features = proxy_features
        self.strict_coverage = strict_coverage

        mods = active_modalities or ["audio", "vision"]
        self.all_modalities = sorted(mods)
        self.num_tokens = num_tokens or {
            m: MODALITY_DEFAULTS.get(m, {}).get("tokens", 1) for m in mods
        }
        self.feature_dims = feature_dims or {
            m: MODALITY_DEFAULTS.get(m, {}).get("dim", 512) for m in mods
        }

        self.placeholders = {
            m: get_modality_placeholder(m, self.num_tokens[m])
            for m in self.all_modalities
        }

        # Load MCQ subsets and build the flat index.
        self._mcq_entries: list[dict] = []
        per_subset_counts = Counter()
        per_anchor_counts = Counter()
        drop_anchor_missing = 0
        drop_proxy_missing = 0

        for mcq_path, question_mod in mcq_subsets:
            opener = gzip.open if str(mcq_path).endswith(".gz") else open
            with opener(mcq_path, "rt") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(
                    f"{mcq_path}: expected dict-of-lists keyed by "
                    f"topic_id; got {type(data).__name__}")
            logger.info(f"  {Path(mcq_path).name}: "
                        f"{len(data)} topic keys (question_modality={question_mod})")
            for topic_key, entries in data.items():
                if not isinstance(entries, list):
                    continue
                prefix, bare = _strip_prefix(topic_key)
                anchor = PREFIX_TO_ANCHOR.get(prefix)
                if anchor is None:
                    continue

                # Resolve real feature for the anchor modality.
                real_key = _resolve_real_key(
                    bare, anchor, self.real_features.get(anchor, {}))
                if real_key is None:
                    drop_anchor_missing += 1
                    if self.strict_coverage:
                        continue

                # Resolve proxy features for every non-anchor modality.
                proxy_keys = {}
                missed = False
                for m in self.all_modalities:
                    if m == anchor:
                        continue
                    pk = _resolve_proxy_key(
                        bare, prefix, self.proxy_features.get(m, {}))
                    if pk is None:
                        missed = True
                        if self.strict_coverage:
                            break
                    proxy_keys[m] = pk
                if missed and self.strict_coverage:
                    drop_proxy_missing += 1
                    continue

                for entry in entries:
                    norm = _parse_mcq_entry(entry)
                    if norm is None:
                        continue
                    self._mcq_entries.append({
                        "topic_key": topic_key,
                        "bare": bare,
                        "prefix": prefix,
                        "anchor": anchor,
                        "real_key": real_key,
                        "proxy_keys": proxy_keys,
                        "question_modality": question_mod,
                        **norm,
                    })
                    per_subset_counts[question_mod] += 1
                    per_anchor_counts[anchor] += 1

        logger.info(
            f"MCQCompositionDataset: {len(self._mcq_entries)} samples, "
            f"question-modality {dict(per_subset_counts)}, "
            f"anchor-modality {dict(per_anchor_counts)}, "
            f"dropped: anchor_missing={drop_anchor_missing} "
            f"proxy_missing={drop_proxy_missing}")

    def __len__(self) -> int:
        return len(self._mcq_entries)

    def _zero_feature(self, modality: str) -> torch.Tensor:
        ntok = self.num_tokens[modality]
        dim = self.feature_dims[modality]
        return torch.zeros(ntok, dim) if ntok > 1 else torch.zeros(dim)

    def _coerce_shape(self, feat: torch.Tensor, modality: str) -> torch.Tensor:
        ntok = self.num_tokens[modality]
        dim = self.feature_dims[modality]
        if ntok == 1:
            if feat.dim() == 2:
                if feat.shape[0] == 1:
                    return feat.squeeze(0)
                return feat.mean(dim=0)
            return feat
        if feat.dim() == 1 and feat.shape[0] == dim:
            return feat.unsqueeze(0).expand(ntok, dim).contiguous()
        if feat.dim() == 2 and feat.shape == (ntok, dim):
            return feat
        if feat.dim() == 2 and feat.shape[0] == 1 and feat.shape[1] == dim:
            return feat.expand(ntok, dim).contiguous()
        if feat.dim() == 2 and feat.shape[1] == dim:
            src = feat.shape[0]
            if src > ntok:
                return feat[:ntok]
            pad = feat[-1:].expand(ntok - src, dim)
            return torch.cat([feat, pad], dim=0)
        return feat

    def __getitem__(self, idx: int) -> dict:
        entry = self._mcq_entries[idx]
        anchor = entry["anchor"]

        # Real anchor feature.
        real_feat = self.real_features.get(anchor, {}).get(entry["real_key"]) \
            if entry["real_key"] is not None else None
        if real_feat is None:
            real_feat = self._zero_feature(anchor)
        modality_features = {
            anchor: self._coerce_shape(real_feat.float(), anchor)
        }

        # Proxies for every non-anchor modality.
        for m in self.all_modalities:
            if m == anchor:
                continue
            pk = entry["proxy_keys"].get(m)
            if pk is not None and pk in self.proxy_features.get(m, {}):
                pf = self.proxy_features[m][pk].float()
            else:
                pf = self._zero_feature(m)
            modality_features[m] = self._coerce_shape(pf, m)

        prompt = _build_mcq_prompt(
            question=entry["q"],
            options=entry["options"],
            all_placeholders=self.placeholders,
            modality_order=self.all_modalities,
            system_prompt=self.system_prompt,
            answer_text=entry["answer_text"],
        )
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=self.max_seq_length,
        )
        input_ids = encoded.input_ids.squeeze(0)
        labels = mask_assistant_labels(input_ids, self.tokenizer)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "modality_features": modality_features,
            "sample_key": entry["topic_key"],
            "anchor": anchor,
            "question_modality": entry["question_modality"],
            "answer_text": entry["answer_text"],
        }
