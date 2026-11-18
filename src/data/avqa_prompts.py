"""Prompt construction for AVQA and MUSIC-AVQA evaluation.

These benchmarks are used for evaluation only; MSP is never trained on them
and their training splits are never used. This module builds the evaluation
prompts, the reserved modality token blocks, and the text-control payloads
used by the inference-time controls.
"""

import json
import random
import re
from pathlib import Path


from src.models.qwen_tokens import (
    FIM_PAD as AUDIO_CONTEXT_ID,
    IMAGE_PAD as VISION_CONTEXT_ID,
    IM_END as IM_END_ID,
    IM_START as IM_START_ID,
    PAD as PAD_ID,
)
QWEN_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
QWEN_AUDIO_BLOCK = "<|fim_pad|>"

METADATA_DIR = "data/music_avqa/metadata"
IMAGE_CAPTIONS_PATH = "data/music_avqa/image_captions.json"
TEXT_CONTROL_MODES = (
    "none",
    "proxy_question",
    "shuffled_question",
    "unrelated_question",
    "generic_only",
    "proxy_no_slot",
)
GENERIC_CONTROL_TEXT = (
    "Review the available evidence carefully and answer with a single word after "
    "considering the relevant objects, actions, and sounds."
)
UNRELATED_CONTROL_BANK = (
    "A librarian sorted invoices beside weather reports while a train schedule lay open "
    "next to a stack of blue folders on a wooden desk."
)


def resolve_templates(question, templ_values):
    """Replace <LR>, <Object>, etc. with values from templ_values list."""
    if isinstance(templ_values, str):
        templ_values = json.loads(templ_values)
    if not templ_values:
        return question

    idx = [0]

    def replacer(match):
        if idx[0] < len(templ_values):
            val = templ_values[idx[0]]
            idx[0] += 1
            return str(val)
        return match.group(0)

    return re.sub(r"<[A-Za-z]+>", replacer, question)


def get_modality_prompt_blocks(tokenizer, num_vision_tokens=1, num_audio_tokens=1):
    """Return the prompt blocks for injected vision and audio tokens."""
    audio_tokens = QWEN_AUDIO_BLOCK * num_audio_tokens
    if num_vision_tokens == 1:
        return QWEN_VISION_BLOCK, audio_tokens
    image_pad = tokenizer.decode([VISION_CONTEXT_ID])
    vision_tokens = image_pad * num_vision_tokens
    return f"<|vision_start|>{vision_tokens}<|vision_end|>", audio_tokens


def load_image_captions(captions_path=IMAGE_CAPTIONS_PATH):
    path = Path(captions_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if v}


def video_id_to_image_key(video_id: str) -> str:
    return f"{video_id}.jpg"


def get_image_caption(image_captions, video_id: str) -> str:
    return image_captions.get(video_id_to_image_key(video_id), image_captions.get(video_id, ""))


def _match_word_length(text: str, reference_text: str) -> str:
    words = text.split()
    if not words:
        return text
    target_words = max(1, len(reference_text.split()))
    out = []
    while len(out) < target_words:
        out.extend(words)
    return " ".join(out[:target_words])


def build_text_shuffle_map(video_ids, seed=0):
    unique_ids = list(dict.fromkeys(video_ids))
    if len(unique_ids) <= 1:
        return {vid: vid for vid in unique_ids}
    shuffled = unique_ids.copy()
    rng = random.Random(seed)
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(unique_ids, shuffled)):
            break
    return dict(zip(unique_ids, shuffled))


def get_text_control_payload(video_id, text_control_mode, image_captions, text_shuffle_map=None):
    if text_control_mode == "none":
        return ""

    caption = get_image_caption(image_captions, video_id)
    if text_control_mode in ("proxy_question", "proxy_no_slot"):
        return caption
    if text_control_mode == "generic_only":
        return GENERIC_CONTROL_TEXT
    if text_control_mode == "unrelated_question":
        return _match_word_length(UNRELATED_CONTROL_BANK, caption or GENERIC_CONTROL_TEXT)
    if text_control_mode == "shuffled_question":
        shuffled_id = video_id
        if text_shuffle_map is not None:
            shuffled_id = text_shuffle_map.get(video_id, video_id)
        shuffled_caption = get_image_caption(image_captions, shuffled_id)
        if not shuffled_caption:
            shuffled_caption = UNRELATED_CONTROL_BANK
        return _match_word_length(shuffled_caption, caption or shuffled_caption)
    raise ValueError(f"Unsupported text_control_mode: {text_control_mode}")


def build_user_content(
    question,
    vision_block,
    audio_block,
    text_control_mode="none",
    control_text="",
    include_vision=True,
    include_audio=True,
):
    """Build prompt body for train/eval under a modality/text-control condition."""
    lines = []
    prefix_parts = []
    use_vision_block = include_vision and text_control_mode != "proxy_no_slot"
    if use_vision_block:
        prefix_parts.append(vision_block)
    if include_audio:
        prefix_parts.append(audio_block)
    if prefix_parts:
        lines.append(" ".join(prefix_parts))

    if text_control_mode in ("proxy_question", "shuffled_question", "unrelated_question") and control_text:
        lines.append(f"Caption: {control_text}")
    elif text_control_mode == "proxy_no_slot" and control_text:
        lines.append(f"Caption: {control_text}")
    elif text_control_mode == "generic_only":
        lines.append(f"Instruction: {control_text or GENERIC_CONTROL_TEXT}")

    lines.append(question)
    lines.append("Answer the question using a single word.")
    return "\n".join(lines)


