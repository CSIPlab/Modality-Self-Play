"""Cross-modal composition dataset: freeform-answer joint-reasoning training.

Trains MSP on questions that *require* two modalities to answer. Unlike
the MCQ-corpus regime where the question is about one modality and other
slots can be ignored, here the answer depends on both modalities listed
per sample. This is the cleanest test of MSP's joint-reasoning claim:
the LoRA cannot answer correctly without using both the real-anchor
modality and the proxy-substituted secondary modality.

Schema (one entry per topic, multiple per topic):

    "audio: Ambience Birds.wav": [
        {"q": "What bird produces the trilling sound?",
         "a": "The sparrow on the branches",
         "modalities": ["audio", "vision"]},
        ...
    ]

Anchor modality comes from the topic-key prefix.
Each sample's `modalities` list contains the anchor + one other modality.
The "other" modality is filled with a proxy feature derived from the
same topic. Unused third modality (not in the entry's `modalities`) is
filled with the corresponding topic-pool proxy for consistency with the
fixed 3-slot prompt structure.

Slot fill policy:

    anchor in entry.modalities  → real feature
    other  in entry.modalities  → proxy feature for that modality
    third  NOT in entry.modalities → proxy (kept benign so the prompt
                                     has 3 populated placeholders)

The answer is freeform text. Loss is the same label-masked CE used by
the existing CompositionDataset / MCQCompositionDataset.
"""
from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.mcq_composition_dataset import (
    PREFIX_TO_ANCHOR,
    _resolve_proxy_key,
    _resolve_real_key,
    _strip_prefix,
)
from src.data.multimodal_dataset import (
    MODALITY_DEFAULTS,
    mask_assistant_labels,
)
from src.models.multimodal_model import get_modality_placeholder

logger = logging.getLogger(__name__)


def _parse_xm_entry(entry: dict) -> dict | None:
    """Validate one cross-modal entry. Returns normalised dict or None."""
    q = (entry.get("q") or entry.get("question") or "").strip()
    a = (entry.get("a") or entry.get("answer") or
         entry.get("answer_text") or "").strip()
    mods = entry.get("modalities") or entry.get("mods") or []
    if not q or not a or not isinstance(mods, list) or len(mods) < 2:
        return None
    norm_mods = []
    for m in mods:
        m = str(m).strip()
        if m in PREFIX_TO_ANCHOR.values():
            norm_mods.append(m)
    norm_mods = sorted(set(norm_mods))
    if len(norm_mods) < 2:
        return None
    return {"q": q, "a": a, "modalities": norm_mods}


def _build_xm_prompt(question: str, answer: str,
                     all_placeholders: dict[str, str],
                     modality_order: list[str],
                     system_prompt: str) -> str:
    """Freeform-answer prompt. Three modality placeholders always present
    in fixed order, then the question, then the freeform answer in the
    assistant turn."""
    prefix = " ".join(all_placeholders[m] for m in modality_order)
    user = f"{prefix}\n{question}"
    parts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n",
        f"<|im_start|>user\n{user}<|im_end|>\n",
        f"<|im_start|>assistant\n{answer}<|im_end|>\n",
    ]
    return "".join(parts)


class CrossModalCompositionDataset(Dataset):
    """Freeform-answer cross-modal composition dataset.

    Each sample's `modalities` list must contain ≥2 modalities. The
    anchor (per topic-key prefix) gets its real feature; the other
    listed modality gets a proxy. The third (unlisted) modality also
    gets a proxy so the prompt's 3 placeholders are uniformly populated.

    Parameters
    ----------
    qa_paths : list[str | Path]
        Paths to cross-modal `.json` or `.json.gz` files. Each file is
        expected to be `{topic_key: [entry, entry, ...]}`.
    real_features, proxy_features, tokenizer : same as MCQCompositionDataset
    active_modalities : list[str], default ['audio','vision']
    max_samples : int | None
        If set, randomly subsample to this many entries (post-filter).
        Useful for B1-scale training.
    seed : int
        For subsampling reproducibility.
    """

    def __init__(
        self,
        qa_paths: list,
        real_features: dict[str, dict[str, torch.Tensor]],
        proxy_features: dict[str, dict[str, torch.Tensor]],
        tokenizer,
        active_modalities: list[str] | None = None,
        num_tokens: dict[str, int] | None = None,
        feature_dims: dict[str, int] | None = None,
        max_seq_length: int = 512,
        system_prompt: str = "You are a helpful assistant.",
        strict_coverage: bool = True,
        max_samples: int | None = None,
        seed: int = 42,
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

        self._entries: list[dict] = []
        per_pair = Counter()           # which (anchor, other) pairs
        per_anchor = Counter()
        drop_anchor_missing = 0
        drop_proxy_missing = 0
        drop_malformed = 0

        for p in qa_paths:
            opener = gzip.open if str(p).endswith(".gz") else open
            with opener(p, "rt") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(
                    f"{p}: expected dict-of-lists keyed by topic_id; "
                    f"got {type(data).__name__}")
            logger.info(f"  {Path(p).name}: {len(data)} topic keys")
            for topic_key, entries in data.items():
                if not isinstance(entries, list) or not entries:
                    continue
                prefix, bare = _strip_prefix(topic_key)
                anchor = PREFIX_TO_ANCHOR.get(prefix)
                if anchor is None:
                    continue

                real_key = _resolve_real_key(
                    bare, anchor, self.real_features.get(anchor, {}))
                if real_key is None:
                    drop_anchor_missing += len(entries)
                    if self.strict_coverage:
                        continue

                # Resolve proxies for every non-anchor modality.
                proxy_keys: dict[str, str | None] = {}
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
                    drop_proxy_missing += len(entries)
                    continue

                for entry in entries:
                    norm = _parse_xm_entry(entry)
                    if norm is None:
                        drop_malformed += 1
                        continue
                    # Sanity: anchor should be one of the listed modalities
                    if anchor not in norm["modalities"]:
                        drop_malformed += 1
                        continue
                    other = [m for m in norm["modalities"] if m != anchor]
                    if not other:
                        drop_malformed += 1
                        continue
                    self._entries.append({
                        "topic_key": topic_key,
                        "bare": bare,
                        "prefix": prefix,
                        "anchor": anchor,
                        "real_key": real_key,
                        "proxy_keys": proxy_keys,
                        "primary_other": other[0],
                        "modalities_required": norm["modalities"],
                        "q": norm["q"],
                        "a": norm["a"],
                    })
                    per_pair[(anchor, other[0])] += 1
                    per_anchor[anchor] += 1

        logger.info(
            f"CrossModalCompositionDataset built: {len(self._entries)} entries, "
            f"per-anchor {dict(per_anchor)}, per-pair {dict(per_pair)}, "
            f"dropped malformed={drop_malformed} "
            f"anchor_missing={drop_anchor_missing} proxy_missing={drop_proxy_missing}"
        )

        if max_samples is not None and len(self._entries) > max_samples:
            import random
            rng = random.Random(seed)
            # Stratified subsample: balance across (anchor, primary_other) pairs
            by_pair: dict[tuple, list[dict]] = {}
            for e in self._entries:
                by_pair.setdefault((e["anchor"], e["primary_other"]), []).append(e)
            n_pairs = len(by_pair)
            per_pair_quota = max_samples // n_pairs
            sampled: list[dict] = []
            for k, lst in by_pair.items():
                rng.shuffle(lst)
                sampled.extend(lst[:per_pair_quota])
            rng.shuffle(sampled)
            self._entries = sampled
            stratified = Counter((e["anchor"], e["primary_other"])
                                 for e in self._entries)
            logger.info(
                f"  Stratified subsample to {len(self._entries)} entries; "
                f"per-pair counts: {dict(stratified)}")

    def __len__(self) -> int:
        return len(self._entries)

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
        e = self._entries[idx]
        anchor = e["anchor"]

        # Anchor real feature.
        real_feat = (self.real_features.get(anchor, {}).get(e["real_key"])
                     if e["real_key"] is not None else None)
        if real_feat is None:
            real_feat = self._zero_feature(anchor)
        modality_features = {anchor: self._coerce_shape(real_feat.float(), anchor)}

        # Proxies for every non-anchor modality (covers both the
        # "other required" modality and the unlisted third).
        for m in self.all_modalities:
            if m == anchor:
                continue
            pk = e["proxy_keys"].get(m)
            if pk is not None and pk in self.proxy_features.get(m, {}):
                pf = self.proxy_features[m][pk].float()
            else:
                pf = self._zero_feature(m)
            modality_features[m] = self._coerce_shape(pf, m)

        prompt = _build_xm_prompt(
            question=e["q"],
            answer=e["a"],
            all_placeholders=self.placeholders,
            modality_order=self.all_modalities,
            system_prompt=self.system_prompt,
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
            "sample_key": e["topic_key"],
            "anchor": anchor,
            "primary_other": e["primary_other"],
            "modalities_required": e["modalities_required"],
            "answer_text": e["a"],
        }
