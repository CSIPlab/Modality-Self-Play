"""General multimodal dataset for any modality.

Handles Phase 1 (unimodal, real paired data) and Phase 2 (self-play
composition) training. Supports multiple QA formats via pluggable parsers.

Usage:
    from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn

    dataset = MultiModalDataset(
        samples=llava_samples,
        tokenizer=tokenizer,
        modality_name="vision",
        features=vision_features,
        qa_format="llava",
        num_modality_tokens=1,
    )
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from src.models.multimodal_model import get_modality_placeholder

from src.models.qwen_tokens import (
    IM_END as IM_END_ID,
    IM_START as IM_START_ID,
    PAD as PAD_ID,
)

# Every modality is a single 512-d token, so real and proxy features share
# the same representation format.
MODALITY_DEFAULTS = {
    "vision": {"dim": 512, "tokens": 1},
    "audio": {"dim": 512, "tokens": 1},
}


# ── QA format parsers ───────────────────────────────────────────────────
# Each parser takes a raw sample dict and returns a list of
# (user_text, assistant_text) turn pairs.

def parse_llava(sample: dict) -> list[tuple[str, str]]:
    """Parse LLaVA conversation format.

    Expected: {"conversations": [{"from": "human", "value": ...},
                                  {"from": "gpt", "value": ...}, ...]}
    """
    conversations = sample.get("conversations", [])
    turns = []
    for i in range(0, len(conversations), 2):
        human = conversations[i].get("value", "")
        gpt = conversations[i + 1].get("value", "") if i + 1 < len(conversations) else ""
        turns.append((human, gpt))
    return turns


def parse_audiocaps(sample: dict) -> list[tuple[str, str]]:
    """Parse AudioCaps format.

    Expected: {"sound_name": "xxx.wav", "captions": ["description..."]}
    """
    captions = sample.get("captions", [])
    caption = captions[0] if captions else ""
    question = "Describe the sounds you hear."
    return [(question, caption)]


def parse_simple_qa(sample: dict) -> list[tuple[str, str]]:
    """Parse simple QA format.

    Expected: {"question": "...", "answer": "..."}
    or: {"q": "...", "a": "..."}
    """
    question = sample.get("question", sample.get("q", ""))
    answer = sample.get("answer", sample.get("a", ""))
    return [(question, answer)]


def parse_avqa(sample: dict) -> list[tuple[str, str]]:
    """Parse AVQA format with template resolution and MCQ choices.

    AVQA is multiple-choice. Without exposing the four candidate
    choices to the model, it produces freeform short answers that
    rarely string-match the ground-truth choice. We append the choices
    to the prompt and instruct the model to pick one.

    Field semantics in val_qa.json:
      - `multi_choice`: list of 4 candidate answer strings
      - `answer` (int) is an INDEX into `multi_choice`
      - `answer_text` is the actual answer string (= `multi_choice[answer]`)
    Older AVQA exports used `anser` (typo) or `answer` as a string.
    """
    question = sample.get("question_content", sample.get("question_text", ""))
    answer = sample.get(
        "answer_text",
        sample.get("anser", sample.get("answer", "")))

    # Resolve templates
    templ_values = sample.get("templ_values", "[]")
    if isinstance(templ_values, str):
        templ_values = json.loads(templ_values)
    if templ_values:
        idx = [0]
        def replacer(match):
            if idx[0] < len(templ_values):
                val = templ_values[idx[0]]
                idx[0] += 1
                return str(val)
            return match.group(0)
        question = re.sub(r"<[A-Za-z]+>", replacer, question)

    # Append the four MCQ candidates if available, so the model can
    # actually pick one rather than guessing a freeform short answer.
    choices = sample.get("multi_choice")
    if isinstance(choices, list) and len(choices) > 0:
        choice_lines = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        prompt = (
            f"{question}\nChoose the correct answer from:\n"
            f"{choice_lines}\n"
            "Respond with the exact text of the correct option."
        )
    else:
        prompt = question + "\nAnswer the question using a single word."
    return [(prompt, str(answer))]


def parse_music_avqa(sample: dict) -> list[tuple[str, str]]:
    """Parse MUSIC-AVQA format with template resolution."""
    question = sample.get("question_content", sample.get("question_text", ""))
    answer = sample.get("anser", sample.get("answer", ""))

    templ_values = sample.get("templ_values", "[]")
    if isinstance(templ_values, str):
        templ_values = json.loads(templ_values)
    if templ_values:
        idx = [0]

        def replacer(match):
            if idx[0] < len(templ_values):
                val = templ_values[idx[0]]
                idx[0] += 1
                return str(val)
            return match.group(0)
        question = re.sub(r"<[A-Za-z]+>", replacer, question)

    prompt = question + "\nAnswer the question using a single word."
    return [(prompt, str(answer))]


def parse_omnibench(sample: dict) -> list[tuple[str, str]]:
    """Parse OmniBench format (audio + image + MCQ).

    Expected: {"question_content": str, "multi_choice": [str]*4,
               "answer_text": str}
    Similar MCQ-form prompt as AVQA; lenient-match + MCQ-snap on
    the eval side selects the right letter/text.
    """
    question = sample.get("question_content", sample.get("question", ""))
    answer = sample.get("answer_text", sample.get("answer", ""))
    choices = sample.get("multi_choice") or sample.get("options") or []
    if isinstance(choices, list) and len(choices) > 0:
        choice_lines = "\n".join(
            f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        prompt = (
            f"{question}\nChoose the correct answer from:\n"
            f"{choice_lines}\n"
            "Respond with the exact text of the correct option."
        )
    else:
        prompt = question + "\nAnswer the question using a single word."
    return [(prompt, str(answer))]


def parse_videoinst(sample: dict) -> list[tuple[str, str]]:
    """Parse VideoInstruct-100K format.

    Expected: {"video_id": "...", "q": "...", "a": "..."}
    """
    question = sample.get("q", sample.get("question", ""))
    answer = sample.get("a", sample.get("answer", ""))
    return [(question, answer)]


def parse_videomme_mc(sample: dict) -> list[tuple[str, str]]:
    """Video-MME multi-choice format (lmms-lab/Video-MME).

    Expected: {"question": str, "options": [str]*4, "answer": "A"|"B"|"C"|"D"}.
    The options come pre-formatted as 'A. xxx' strings.
    """
    q = (sample.get("question") or "").strip()
    opts = sample.get("options") or []
    ans_letter = (sample.get("answer") or "").strip().upper()
    # Try to derive the answer text from the option string
    ans_text = ans_letter
    for o in opts:
        if isinstance(o, str) and o.strip().upper().startswith(ans_letter + "."):
            ans_text = o.split(".", 1)[1].strip()
            break
    options_block = "\n".join(opts) if opts else ""
    prompt = (
        f"{q}\n{options_block}\n"
        "Respond with the letter of the correct option."
    )
    return [(prompt, ans_letter)]


def _videomme_key(sample: dict) -> str:
    return str(sample.get("videoID") or sample.get("video_id") or "")


def parse_nextqa_mc(sample: dict) -> list[tuple[str, str]]:
    """NExT-QA multiple-choice format (lmms-lab/NExTQA MC variant).

    Expected: {"question": "...", "a0"..."a4": str, "answer": int 0-4}.
    Builds a 5-choice MC prompt and uses the gold-text answer (a<idx>)
    as the target so evaluation can lenient-match against the option
    string. OneLLM Tab.2 reports 57.3% on this benchmark.
    """
    q = (sample.get("question") or "").strip()
    choices = [sample.get(f"a{i}", "") for i in range(5)]
    answer_idx = int(sample.get("answer", 0))
    answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
    choice_lines = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices) if c)
    prompt = (
        f"{q}\nChoose the correct answer from:\n{choice_lines}\n"
        "Respond with the exact text of the correct option."
    )
    return [(prompt, str(answer_text))]


def _nextqa_key(sample: dict) -> str:
    return str(sample.get("video_id") or sample.get("video") or "")


def parse_videoinst_short(sample: dict) -> list[tuple[str, str]]:
    """Same as videoinst but instructs the model to answer in 1-2 words.

    MSVD-QA / MSRVTT-QA gold answers are typically single words ('piano',
    'man', 'running'). Without a hint, our caption-trained model emits
    full sentences and exact-match accuracy collapses. The hint nudges
    output toward the eval format.
    """
    question = sample.get("q", sample.get("question", ""))
    answer = sample.get("a", sample.get("answer", ""))
    prompt = question.strip()
    if not prompt.endswith("?"):
        prompt = prompt + "?"
    prompt = prompt + " Answer with one or two words."
    return [(prompt, answer)]


# Benchmarks used only for evaluation. MSP is never trained on these, and
# their training splits are never used, so they are rejected by the training
# entry points.
EVAL_ONLY_QA_FORMATS = frozenset({
    "avqa",
    "music_avqa",
    "omnibench",
    "nextqa_mc",
    "videomme_mc",
})


QA_PARSERS = {
    "llava": parse_llava,
    "audiocaps": parse_audiocaps,
    "audio_conversations": parse_llava,  # conversations[] array keyed by sound_name/ytid
    "simple_qa": parse_simple_qa,
    "avqa": parse_avqa,
    "music_avqa": parse_music_avqa,
    "omnibench": parse_omnibench,
    "videoinst": parse_videoinst,
    "videoinst_short": parse_videoinst_short,
    "nextqa_mc": parse_nextqa_mc,
    "videomme_mc": parse_videomme_mc,
}


# ── Default sample key extractors ───────────────────────────────────────

def _llava_key(sample: dict) -> str:
    img = sample.get("image", "")
    return img.split("/")[-1] if "/" in img else img


def _audiocaps_key(sample: dict) -> str:
    return sample.get("sound_name", sample.get("ytid", ""))


def _avqa_key(sample: dict) -> str:
    # AVQA feature dicts are keyed by the YouTube video_name string
    # (e.g. "06GyG4-wONQ_000006"), not the integer video_id.
    return sample.get("video_name", sample.get("video_id", ""))


def _music_avqa_key(sample: dict) -> str:
    return sample.get("video_id", "")


def _videoinst_key(sample: dict) -> str:
    return sample.get("video_id", "")


def _omnibench_key(sample: dict) -> str:
    return str(sample.get("video_name",
                          sample.get("index", "")))


DEFAULT_KEY_FNS = {
    "llava": _llava_key,
    "audiocaps": _audiocaps_key,
    "audio_conversations": _audiocaps_key,  # sound_name/ytid lookup, same as audiocaps
    "simple_qa": _llava_key,
    "avqa": _avqa_key,
    "music_avqa": _music_avqa_key,
    "omnibench": _omnibench_key,
    "videoinst": _videoinst_key,
    "videoinst_short": _videoinst_key,
    "nextqa_mc": _nextqa_key,
    "videomme_mc": _videomme_key,
}


# ── Label masking utility ───────────────────────────────────────────────

def mask_assistant_labels(input_ids: torch.LongTensor, tokenizer) -> torch.LongTensor:
    """Create labels tensor that only unmasks assistant response tokens.

    Scans for <|im_start|>assistant\\n patterns and unmasks tokens from
    after \\n to the next <|im_end|> (inclusive). Everything else = -100.
    """
    labels = torch.full_like(input_ids, -100)
    ids = input_ids.tolist()
    n = len(ids)
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

    i = 0
    while i < n:
        if ids[i] == IM_START_ID:
            j = i + 1
            while j < n and ids[j] != newline_id:
                j += 1
            if j < n:
                role_ids = ids[i + 1:j]
                if all(aid in role_ids for aid in assistant_ids):
                    start = j + 1
                    k = start
                    while k < n and ids[k] != IM_END_ID:
                        k += 1
                    if k < n:
                        labels[start:k + 1] = input_ids[start:k + 1]
                    i = k + 1
                    continue
        i += 1

    return labels


# ── Dataset ─────────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """General dataset for any modality, any QA format.

    Args:
        samples: Raw QA records (list of dicts, any format).
        tokenizer: Qwen 3 tokenizer.
        modality_name: Primary modality ("vision" or "audio")
        features: Dict mapping sample_key → feature tensor.
        qa_format: Parser name ("llava" or "audiocaps")
        num_modality_tokens: Number of placeholder tokens for this modality.
        sample_key_fn: Extracts feature dict key from sample. Default per format.
        max_seq_length: Maximum token sequence length.
        system_prompt: System message.
        extra_modalities: For Phase 2 — additional modalities to inject.
            Dict of {name: {"features": dict, "num_tokens": int}}
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        modality_name: str,
        features: dict[str, torch.Tensor],
        qa_format: str = "llava",
        num_modality_tokens: int | None = None,
        sample_key_fn: Callable | None = None,
        max_seq_length: int = 512,
        system_prompt: str = "You are a helpful assistant.",
        extra_modalities: dict[str, dict] | None = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.modality_name = modality_name
        self.features = features
        self.qa_format = qa_format
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.extra_modalities = extra_modalities or {}

        if num_modality_tokens is None:
            num_modality_tokens = MODALITY_DEFAULTS.get(modality_name, {}).get("tokens", 1)
        self.num_modality_tokens = num_modality_tokens

        if sample_key_fn is None:
            sample_key_fn = DEFAULT_KEY_FNS.get(qa_format, _llava_key)
        self.sample_key_fn = sample_key_fn

        if qa_format not in QA_PARSERS:
            raise ValueError(f"Unknown QA format: {qa_format}. "
                             f"Available: {list(QA_PARSERS.keys())}")
        self.parser = QA_PARSERS[qa_format]

        # Build modality placeholder string
        self.modality_placeholder = get_modality_placeholder(
            modality_name, num_modality_tokens)

        # Build extra modality placeholders
        self.extra_placeholders = {}
        for name, cfg in self.extra_modalities.items():
            ntok = cfg.get("num_tokens", MODALITY_DEFAULTS.get(name, {}).get("tokens", 1))
            self.extra_placeholders[name] = get_modality_placeholder(name, ntok)

        # Encoder dim for zero features
        dim = MODALITY_DEFAULTS.get(modality_name, {}).get("dim", 512)
        self.encoder_dim = dim

    def __len__(self):
        return len(self.samples)

    def _build_prompt(self, turns: list[tuple[str, str]]) -> str:
        """Build chat-formatted prompt with modality token injection."""
        # Build all modality placeholders to prepend
        all_placeholders = [self.modality_placeholder]
        for name in sorted(self.extra_modalities.keys()):
            all_placeholders.append(self.extra_placeholders[name])
        modality_prefix = " ".join(all_placeholders)

        parts = [f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"]

        for i, (user_text, assistant_text) in enumerate(turns):
            # Replace <image> or <audio> etc with modality tokens
            if "<image>" in user_text:
                user_text = user_text.replace("<image>\n", f"{modality_prefix}\n")
                user_text = user_text.replace("\n<image>", f"\n{modality_prefix}")
                user_text = user_text.replace("<image>", modality_prefix)
            elif i == 0:
                # No placeholder tag in raw text — prepend modality tokens
                user_text = f"{modality_prefix}\n{user_text}"

            parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
            if assistant_text:
                parts.append(f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n")

        return "".join(parts)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_key = self.sample_key_fn(sample)

        # Parse QA into turns
        turns = self.parser(sample)

        # Build prompt and tokenize
        prompt = self._build_prompt(turns)
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=self.max_seq_length,
        )
        input_ids = encoded.input_ids.squeeze(0)

        # Build labels
        labels = mask_assistant_labels(input_ids, self.tokenizer)

        # Load primary modality features
        # Strip common extensions for key matching
        clean_key = sample_key
        for ext in (".jpg", ".jpeg", ".png", ".wav", ".mp4", ".npy", ".obj"):
            if clean_key.endswith(ext):
                clean_key = clean_key[:-len(ext)]
                break

        feature = self.features.get(
            sample_key,
            self.features.get(clean_key, None))

        if feature is None:
            # Zero features for missing samples
            if self.num_modality_tokens > 1:
                feature = torch.zeros(self.num_modality_tokens, self.encoder_dim)
            else:
                feature = torch.zeros(self.encoder_dim)

        modality_features = {self.modality_name: feature}

        # Load extra modality features (for Phase 2)
        for name, cfg in self.extra_modalities.items():
            extra_feats = cfg.get("features", {})
            extra_dim = MODALITY_DEFAULTS.get(name, {}).get("dim", 512)
            extra_ntok = cfg.get("num_tokens", 1)
            ef = extra_feats.get(sample_key, extra_feats.get(clean_key, None))
            if ef is None:
                ef = torch.zeros(extra_ntok, extra_dim) if extra_ntok > 1 else torch.zeros(extra_dim)
            modality_features[name] = ef

        return {
            "input_ids": input_ids,
            "labels": labels,
            "modality_features": modality_features,
            "sample_key": sample_key,
        }


def multimodal_collate_fn(batch: list[dict]) -> dict:
    """Collate multimodal samples with right-padding."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = []
    labels = []
    sample_keys = []

    # Collect all modality names
    all_modalities = set()
    for b in batch:
        all_modalities.update(b["modality_features"].keys())

    # Initialize per-modality feature lists
    modality_features = {name: [] for name in all_modalities}
    feature_presence = {name: [] for name in all_modalities}

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([
            b["input_ids"],
            torch.full((pad_len,), PAD_ID, dtype=torch.long),
        ]))
        labels.append(torch.cat([
            b["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ]))
        sample_keys.append(b["sample_key"])

        sample_presence = b.get("feature_presence", {})
        for name in all_modalities:
            feat = b["modality_features"].get(name)
            if feat is None:
                # Should not happen, but handle gracefully
                feat = torch.zeros(1)
            modality_features[name].append(feat)
            feature_presence[name].append(bool(
                sample_presence.get(name, name in b["modality_features"])) )

    # Stack features per modality
    stacked_features = {}
    stacked_presence = {}
    for name, feat_list in modality_features.items():
        stacked_features[name] = torch.stack(feat_list)
        stacked_presence[name] = torch.tensor(
            feature_presence[name], dtype=torch.bool)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": (torch.stack(input_ids) != PAD_ID).long(),
        "modality_features": stacked_features,
        "feature_presence": stacked_presence,
        "sample_keys": sample_keys,
    }


# ── Phase 2 composition dataset ─────────────────────────────────────────
# Every sample grounds exactly one modality in real features (the anchor);
# the others come from proxy features encoded through the contrastive text
# encoders, so no sample carries two real non-text modalities.


class CompositionSubset:
    """One (modality, QA-format, real-features) triple for composition training."""

    def __init__(
        self,
        name: str,
        samples: list[dict],
        qa_format: str,
        real_features: dict[str, torch.Tensor],
        num_tokens: int | None = None,
        sample_key_fn: Callable | None = None,
    ):
        self.name = name
        self.samples = samples
        self.qa_format = qa_format
        self.real_features = real_features
        if num_tokens is None:
            num_tokens = MODALITY_DEFAULTS.get(name, {}).get("tokens", 1)
        self.num_tokens = num_tokens
        if sample_key_fn is None:
            sample_key_fn = DEFAULT_KEY_FNS.get(qa_format, _llava_key)
        self.sample_key_fn = sample_key_fn
        if qa_format not in QA_PARSERS:
            raise ValueError(f"Unknown QA format: {qa_format}")
        self.parser = QA_PARSERS[qa_format]


class CompositionDataset(Dataset):
    """Phase 2 composition dataset with the 1-real-N-proxies rule.

    Combines multiple per-modality subsets. Each sample is drawn from one
    subset, so the "anchor" modality is determined by which subset the
    sample came from:
      - vision anchor → LLaVA sample → real image features, proxy audio
      - audio anchor  → AudioCaps sample → proxy vision, real audio

    All samples inject ALL modality placeholders so the model sees the same
    token structure regardless of which modality is the anchor. This is
    critical: at inference we swap proxies for real features without
    changing the prompt shape.

    Proxy features come from a per-modality dict passed at construction.
    These are expected to be encoded from imagined descriptions (MSP-0 or
    self-play cache). If a sample has no proxy feature available for a
    non-anchor modality, a zero feature vector is used.

    Args:
        subsets: One CompositionSubset per modality participating in composition.
        tokenizer: Qwen 3 tokenizer.
        proxy_features: dict of {modality_name: {sample_key: proxy_tensor}}.
            Proxy features for non-anchor modalities.
        max_seq_length: Max token sequence length.
        system_prompt: System message.
        interleave: "round_robin" (default) samples subsets in rotation,
            preserving balanced anchor distribution. "weighted" draws
            proportional to subset size. "concat" concatenates subsets
            (anchor distribution = subset sizes).
    """

    def __init__(
        self,
        subsets: list[CompositionSubset],
        tokenizer,
        proxy_features: dict[str, dict[str, torch.Tensor]],
        max_seq_length: int = 512,
        system_prompt: str = "You are a helpful assistant.",
        interleave: str = "round_robin",
        unimodal_only: bool = False,
        shuffled_proxies: bool = False,
        shuffle_seed: int = 0,
        strict_proxy_coverage: bool = False,
        require_anchor_coverage: bool = False,
        audit_coverage: bool = False,
    ):
        """
        unimodal_only: when True, every sample carries ONLY its anchor
            modality — the prompt has only the anchor placeholder, and
            modality_features contains only the anchor entry. This is the
            true shuffled co-training baseline: the model never sees
            placeholder tokens for any non-anchor modality, so it cannot
            learn "ignore zero features for the absent modality." Use this
            instead of an empty proxy_features dict, which still leaves
            zero-feature placeholders in the prompt.
        """
        if not subsets:
            raise ValueError("CompositionDataset requires at least one subset")
        self.subsets = subsets
        self.tokenizer = tokenizer
        self.proxy_features = proxy_features or {}
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.interleave = interleave
        self.unimodal_only = unimodal_only
        self.shuffled_proxies = shuffled_proxies
        self.shuffle_seed = shuffle_seed
        self.strict_proxy_coverage = strict_proxy_coverage
        self.require_anchor_coverage = require_anchor_coverage
        self.audit_coverage = audit_coverage

        if self.shuffled_proxies and not self.unimodal_only:
            self.proxy_features = self._shuffle_proxy_features(
                self.proxy_features,
                self.shuffle_seed,
            )

        # All participating modalities (anchors + guests with proxies)
        self.all_modalities = sorted({s.name for s in subsets}
                                     | set(self.proxy_features.keys()))

        # Per-modality num_tokens — anchor subsets take priority, else proxy shape
        self.num_tokens = {}
        for s in subsets:
            self.num_tokens[s.name] = s.num_tokens
        for name in self.all_modalities:
            if name not in self.num_tokens:
                # Derive from proxy feature shape
                pf = self.proxy_features.get(name, {})
                if pf:
                    sample_val = next(iter(pf.values()))
                    if sample_val.dim() == 2:
                        self.num_tokens[name] = sample_val.shape[0]
                    else:
                        self.num_tokens[name] = MODALITY_DEFAULTS.get(
                            name, {}).get("tokens", 1)
                else:
                    self.num_tokens[name] = MODALITY_DEFAULTS.get(
                        name, {}).get("tokens", 1)

        # Feature dim for the zero fallback, inferred from a real or proxy
        # feature where possible.
        self.feature_dims = {}
        for name in self.all_modalities:
            inferred = None
            for subset in self.subsets:
                if subset.name == name and subset.real_features:
                    sample = next(iter(subset.real_features.values()))
                    inferred = sample.shape[-1]
                    break
            if inferred is None and name in self.proxy_features and self.proxy_features[name]:
                sample = next(iter(self.proxy_features[name].values()))
                inferred = sample.shape[-1]
            if inferred is None:
                inferred = MODALITY_DEFAULTS.get(name, {}).get("dim", 512)
            self.feature_dims[name] = int(inferred)

        # Pre-computed modality placeholders (one per modality)
        self.placeholders = {
            name: get_modality_placeholder(name, self.num_tokens[name])
            for name in self.all_modalities
        }

        # Build interleaved index list: (subset_idx, sample_idx)
        self._index = self._build_index()

        # Coverage enforcement, weakest to strongest: audit only, drop
        # samples missing their anchor feature, or additionally drop samples
        # missing a proxy. All no-ops in unimodal_only mode.
        if not self.unimodal_only and (
                self.strict_proxy_coverage
                or self.require_anchor_coverage
                or self.audit_coverage):
            self._index = self._filter_for_coverage(
                self._index,
                drop_missing_anchor=(
                    self.strict_proxy_coverage or self.require_anchor_coverage),
                drop_missing_proxy=self.strict_proxy_coverage,
            )

    def _build_index(self) -> list[tuple[int, int]]:
        if self.interleave == "round_robin":
            # Visit each sample of each subset exactly once. Smaller subsets
            # drop out of rotation once exhausted (the larger subsets keep
            # going). Name is historical — it does NOT recycle.
            max_len = max(len(s.samples) for s in self.subsets)
            index = []
            for i in range(max_len):
                for j, s in enumerate(self.subsets):
                    if i < len(s.samples):
                        index.append((j, i))
            return index
        elif self.interleave == "balanced":
            # Recycle smaller subsets so every cycle has all modalities.
            # Driver = largest subset; each sample there is visited once,
            # and the smaller subsets wrap (i % len) so they get more
            # exposures. Index size = len(subsets) × max_len.
            max_len = max(len(s.samples) for s in self.subsets)
            index = []
            for i in range(max_len):
                for j, s in enumerate(self.subsets):
                    index.append((j, i % len(s.samples)))
            return index
        elif self.interleave == "concat":
            index = []
            for j, s in enumerate(self.subsets):
                for i in range(len(s.samples)):
                    index.append((j, i))
            return index
        elif self.interleave == "weighted":
            # Same as concat but shuffled (DataLoader shuffles anyway)
            index = []
            for j, s in enumerate(self.subsets):
                for i in range(len(s.samples)):
                    index.append((j, i))
            return index
        else:
            raise ValueError(f"Unknown interleave mode: {self.interleave}")

    def __len__(self):
        return len(self._index)

    def _shuffle_proxy_features(
        self,
        proxy_features: dict[str, dict[str, torch.Tensor]],
        seed: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Permute proxy assignments within each modality while preserving coverage."""
        shuffled: dict[str, dict[str, torch.Tensor]] = {}
        for modality, feats in proxy_features.items():
            clean_to_feat: dict[str, torch.Tensor] = {}
            for key, feat in feats.items():
                clean_to_feat.setdefault(self._strip_key(key), feat)

            keys = list(clean_to_feat.keys())
            if len(keys) <= 1:
                shuffled[modality] = dict(clean_to_feat)
                continue

            src_keys = keys[:]
            rng = random.Random(f"{seed}:{modality}")
            rng.shuffle(src_keys)
            if src_keys == keys:
                src_keys = src_keys[1:] + src_keys[:1]

            shuffled[modality] = {
                dst_key: clean_to_feat[src_key]
                for dst_key, src_key in zip(keys, src_keys)
            }
        return shuffled

    def _filter_for_coverage(
        self, index: list[tuple[int, int]],
        drop_missing_anchor: bool = False,
        drop_missing_proxy: bool = False,
    ) -> list[tuple[int, int]]:
        """Audit / filter by coverage.

        Always logs per-anchor counts for (total, missing_anchor,
        missing_any_proxy). When drop_missing_anchor is True, samples with
        no real anchor feature are removed. When drop_missing_proxy is
        True, samples missing ANY non-anchor proxy are removed. With both
        False this is audit-only (index unchanged).
        """
        import logging
        logger = logging.getLogger(__name__)

        per_anchor_total: dict[str, int] = {}
        per_anchor_no_anchor: dict[str, int] = {}
        per_anchor_no_proxy: dict[str, int] = {}
        kept: list[tuple[int, int]] = []

        for subset_idx, sample_idx in index:
            subset = self.subsets[subset_idx]
            anchor = subset.name
            per_anchor_total[anchor] = per_anchor_total.get(anchor, 0) + 1

            sample = subset.samples[sample_idx]
            sample_key = subset.sample_key_fn(sample)
            clean_key = self._strip_key(sample_key)

            anchor_missing = (
                self._lookup_feature(
                    subset.real_features, sample_key, clean_key) is None)
            if anchor_missing:
                per_anchor_no_anchor[anchor] = \
                    per_anchor_no_anchor.get(anchor, 0) + 1

            proxy_missing = False
            for m in self.all_modalities:
                if m == anchor:
                    continue
                pd = self.proxy_features.get(m, {})
                if self._lookup_feature(pd, sample_key, clean_key) is None:
                    proxy_missing = True
                    break
            if proxy_missing:
                per_anchor_no_proxy[anchor] = \
                    per_anchor_no_proxy.get(anchor, 0) + 1

            drop = (drop_missing_anchor and anchor_missing) \
                or (drop_missing_proxy and proxy_missing)
            if drop:
                continue
            kept.append((subset_idx, sample_idx))

        for name, total in per_anchor_total.items():
            na = per_anchor_no_anchor.get(name, 0)
            npx = per_anchor_no_proxy.get(name, 0)
            logger.info(
                f"coverage[{name}]: total={total} "
                f"missing_anchor={na} ({na / max(total,1):.1%}) "
                f"missing_proxy={npx} ({npx / max(total,1):.1%})"
            )
        mode = "audit" if not (drop_missing_anchor or drop_missing_proxy) \
            else (f"drop_anchor={drop_missing_anchor}"
                  f" drop_proxy={drop_missing_proxy}")
        logger.info(
            f"coverage[{mode}]: kept {len(kept)}/{len(index)} "
            f"({len(kept) / max(len(index), 1):.1%})"
        )
        return kept

    def _strip_key(self, key: str) -> str:
        for ext in (".jpg", ".jpeg", ".png", ".wav", ".mp4", ".npy", ".obj", ".ply"):
            if key.endswith(ext):
                return key[:-len(ext)]
        return key

    def _lookup_feature(
        self, feature_dict: dict, key: str, clean_key: str,
    ) -> torch.Tensor | None:
        return feature_dict.get(key, feature_dict.get(clean_key, None))

    def audit_one_real_anchor(self) -> dict[str, list[str]]:
        """Check the one-real-anchor rule.

        A source sample that happens to contain both modalities must be used
        through exactly one of them. This returns any sample key that appears
        as the real anchor in more than one subset, keyed by the subsets it
        was found in.
        """
        seen: dict[str, list[str]] = {}
        unkeyed = 0
        for subset in self.subsets:
            for sample in subset.samples:
                try:
                    key = subset.sample_key_fn(sample)
                except (KeyError, TypeError, AttributeError):
                    key = None
                if not key:
                    unkeyed += 1
                    continue
                seen.setdefault(key, [])
                if subset.name not in seen[key]:
                    seen[key].append(subset.name)
        if unkeyed:
            logging.getLogger(__name__).warning(
                "One-real-anchor audit skipped %d sample(s) with no usable "
                "key; the rule was not checked for them.", unkeyed)
        return {k: v for k, v in seen.items() if len(v) > 1}

    def audit_feature_coverage(
        self,
        max_per_subset: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Audit real-anchor and proxy coverage across subsets.

        Returns per-anchor-subset coverage counts so callers can warn or
        fail fast when MSP training would silently fall back to zeros.
        """
        report: dict[str, dict[str, Any]] = {}
        for subset in self.subsets:
            samples = subset.samples
            if max_per_subset is not None:
                samples = samples[:max_per_subset]

            subset_total = len(samples)
            anchor_present = 0
            proxy_present = {
                name: 0 for name in self.all_modalities if name != subset.name
            }

            for sample in samples:
                sample_key = subset.sample_key_fn(sample)
                clean_key = self._strip_key(sample_key)
                if self._lookup_feature(subset.real_features, sample_key, clean_key) is not None:
                    anchor_present += 1

                for name in proxy_present:
                    proxy_dict = self.proxy_features.get(name, {})
                    if self._lookup_feature(proxy_dict, sample_key, clean_key) is not None:
                        proxy_present[name] += 1

            report[subset.name] = {
                "sampled": subset_total,
                "anchor_present": anchor_present,
                "anchor_missing": subset_total - anchor_present,
                "anchor_coverage": (anchor_present / subset_total) if subset_total else 0.0,
                "proxy_present": proxy_present,
                "proxy_missing": {
                    name: subset_total - count for name, count in proxy_present.items()
                },
                "proxy_coverage": {
                    name: (count / subset_total) if subset_total else 0.0
                    for name, count in proxy_present.items()
                },
            }
        return report

    def _zero_feature(self, modality: str) -> torch.Tensor:
        ntok = self.num_tokens[modality]
        dim = self.feature_dims[modality]
        if ntok > 1:
            return torch.zeros(ntok, dim)
        return torch.zeros(dim)

    def _coerce_shape(self, feat: torch.Tensor, modality: str) -> torch.Tensor:
        """Normalise a feature tensor to the shape the projector expects.

        A single embedding is tiled when more tokens are wanted, several are
        mean-pooled when one is wanted, and correctly shaped features pass
        through. Without this, torch.stack in multimodal_collate_fn fails on
        a batch whose real and proxy features disagree on token count.
        """
        ntok = self.num_tokens[modality]
        dim = self.feature_dims[modality]
        if ntok == 1:
            if feat.dim() == 2:
                if feat.shape[0] == 1:
                    return feat.squeeze(0)
                # Mean-pool across the token dim for multi-token proxies.
                return feat.mean(dim=0)
            return feat
        # ntok > 1
        if feat.dim() == 1 and feat.shape[0] == dim:
            return feat.unsqueeze(0).expand(ntok, dim).contiguous()
        if feat.dim() == 2 and feat.shape == (ntok, dim):
            return feat
        if feat.dim() == 2 and feat.shape[0] == 1 and feat.shape[1] == dim:
            return feat.expand(ntok, dim).contiguous()
        if feat.dim() == 2 and feat.shape[1] == dim:
            # Pad or truncate along the token dim.
            src = feat.shape[0]
            if src > ntok:
                return feat[:ntok]
            pad = feat[-1:].expand(ntok - src, dim)
            return torch.cat([feat, pad], dim=0)
        return feat

    def _build_prompt(
        self, turns: list[tuple[str, str]], anchor: str,
    ) -> str:
        """Prepend modality placeholders.

        Default (composition) mode: all modality placeholders, sorted, so the
        token IDs are consistent across samples regardless of which one is the
        anchor. This is the "anchor-invariant prompt" rule.

        Unimodal mode (`self.unimodal_only`): only the anchor's placeholder is
        included. Each sample looks like a pure single-modality QA example.
        Used for the shuffled co-training baseline so the model never sees
        placeholder tokens for any modality whose features are absent.
        """
        if self.unimodal_only:
            prefix = self.placeholders[anchor]
        else:
            prefix = " ".join(self.placeholders[m] for m in self.all_modalities)

        parts = [f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"]
        for i, (user_text, assistant_text) in enumerate(turns):
            if "<image>" in user_text:
                user_text = user_text.replace("<image>\n", f"{prefix}\n")
                user_text = user_text.replace("\n<image>", f"\n{prefix}")
                user_text = user_text.replace("<image>", prefix)
            elif i == 0:
                user_text = f"{prefix}\n{user_text}"

            parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
            if assistant_text:
                parts.append(
                    f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n")
        return "".join(parts)

    def __getitem__(self, idx):
        subset_idx, sample_idx = self._index[idx]
        subset = self.subsets[subset_idx]
        sample = subset.samples[sample_idx]
        anchor = subset.name

        sample_key = subset.sample_key_fn(sample)
        clean_key = self._strip_key(sample_key)

        # Parse QA for this sample
        turns = subset.parser(sample)

        # Build anchor-invariant prompt
        prompt = self._build_prompt(turns, anchor)
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=self.max_seq_length,
        )
        input_ids = encoded.input_ids.squeeze(0)
        labels = mask_assistant_labels(input_ids, self.tokenizer)

        # Build per-modality features. _coerce_shape normalises every
        # feature (real or proxy) to the shape the projector expects, so
        # the collate stacker never sees mixed ranks. Track which entries
        # were actually found so training can audit how much MSP is falling
        # back to zeros.
        modality_features = {}
        feature_presence = {}
        anchor_feat = self._lookup_feature(
            subset.real_features, sample_key, clean_key)
        feature_presence[anchor] = anchor_feat is not None
        if anchor_feat is None:
            anchor_feat = self._zero_feature(anchor)
        modality_features[anchor] = self._coerce_shape(anchor_feat, anchor)

        # Non-anchor modalities: in composition mode, look up proxy
        # features (fall back to zeros if missing). In unimodal mode, the
        # prompt has no placeholder for them so the model never reads
        # them — but we still emit per-modality zero tensors of the
        # correct shape so the collator can stack batches with mixed
        # anchor modalities without hitting a shape-fallback path.
        for name in self.all_modalities:
            if name == anchor:
                continue
            if self.unimodal_only:
                feat = self._zero_feature(name)
                feature_presence[name] = False
            else:
                proxy_dict = self.proxy_features.get(name, {})
                feat = self._lookup_feature(proxy_dict, sample_key, clean_key)
                feature_presence[name] = feat is not None
                if feat is None:
                    feat = self._zero_feature(name)
            modality_features[name] = self._coerce_shape(feat, name)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "modality_features": modality_features,
            "feature_presence": feature_presence,
            "sample_key": sample_key,
            "anchor": anchor,
        }


def composition_collate_fn(batch: list[dict]) -> dict:
    """Collate for CompositionDataset. Same as multimodal_collate_fn plus anchors."""
    out = multimodal_collate_fn(batch)
    out["anchors"] = [b.get("anchor") for b in batch]
    return out
