"""Cross-modal imagination: generate descriptions of one modality from another.

Given a MultiModalModel that understands modality M1, generate text
descriptions of what modality M2 would contain. The same model that
learned vision generates audio descriptions — no external captioner.

Usage:
    from src.selfplay.imagine_modality import imagine_modality, SelfPlayCache

    description = imagine_modality(
        model, tokenizer,
        source_modality="vision",
        target_modality="audio",
        source_features=vision_features,  # (1, 512) or (1, N, 512)
    )

    # With caching:
    cache = SelfPlayCache(
        "data/selfplay/vision_to_audio.json",
        model=model, tokenizer=tokenizer,
        source_modality="vision", target_modality="audio",
    )
    description = cache.get("sample_key", source_feature_tensor)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from src.models.multimodal_model import get_modality_placeholder

logger = logging.getLogger("selfplay")


# ── Imagination prompts ─────────────────────────────────────────────────
# Organized as PROMPTS[source_modality][target_modality]

# Each prompt asks for a caption in the target encoder's own style, forbids
# source-modality vocabulary, and gives one short example. Wording like
# "visible in the image" would pull the embedding away from the real-modality
# region, since the text encoders were trained on single-modality captions.

IMAGINATION_PROMPTS = {
    "vision": {
        "audio": (
            "{modality_placeholder}\n"
            "Describe the sounds in this scene as a single AudioCaps-style "
            "caption: ONE short sentence, 8–15 words, present tense, naming "
            "1–3 specific sound sources directly. Do not use multiple "
            "sentences and do not describe pitch, timbre, or volume.\n\n"
            "Do NOT use: image, picture, photo, visible, scene, look, see, "
            "shown, view, visually.\n\n"
            "Examples:\n"
            "  \"A man is speaking while water is flowing.\"\n"
            "  \"A train engine is running and blowing its horn.\"\n"
            "  \"Music is playing as a vehicle passes by.\"\n"
            "  \"An animal is chewing loudly in the background.\"\n"
            "  \"Wind is blowing and leaves are rustling.\"\n\n"
            "Write ONE such sentence. No other text."
        ),
    },
    "audio": {
        "vision": (
            "{modality_placeholder}\n"
            "Write a single COCO-style image caption describing what the "
            "source of the sound would look like as a photograph. Use one "
            "short sentence, 8\u201318 words, present tense.\n\n"
            "Do NOT use any of these words or concepts: sound, audio, "
            "noise, hear, hearing, listen, volume, pitch, timbre, loud, "
            "quiet, rhythmic, echo.\n\n"
            "Name the main object, person, or animal and the setting. "
            "Mention visible attributes only.\n\n"
            "Examples:\n"
            "  \"A dog stands on grass with its mouth open.\"\n"
            "  \"A man sits indoors beside a running kitchen sink.\"\n"
            "  \"A train passes along tracks near a platform.\"\n\n"
            "Write ONE such sentence. No other text."
        ),
    },
}


def get_imagination_prompt(
    source_modality: str,
    target_modality: str,
    num_source_tokens: int = 1,
) -> str:
    """Get the prompt template for imagining target from source."""
    if source_modality not in IMAGINATION_PROMPTS:
        raise ValueError(
            f"No imagination prompts from '{source_modality}'. "
            f"Available sources: {list(IMAGINATION_PROMPTS.keys())}")

    targets = IMAGINATION_PROMPTS[source_modality]
    if target_modality not in targets:
        raise ValueError(
            f"No imagination prompt from '{source_modality}' to "
            f"'{target_modality}'. Available targets: {list(targets.keys())}")

    template = targets[target_modality]
    placeholder = get_modality_placeholder(source_modality, num_source_tokens)
    return template.replace("{modality_placeholder}", placeholder)


@torch.no_grad()
def imagine_modality(
    model,
    tokenizer,
    source_modality: str,
    target_modality: str,
    source_features: torch.Tensor,
    num_source_tokens: int = 1,
    max_new_tokens: int = 256,
) -> str:
    """Generate a text description of target_modality from source features.

    Uses model.generate() with source features injected at the modality
    placeholder tokens. No external captioner — the same model that learned
    the source modality does the imagining.

    Args:
        model: MultiModalModel with source_modality projector trained.
        tokenizer: Qwen 3 tokenizer.
        source_modality: Modality the model understands ("vision", "audio", etc.)
        target_modality: Modality to imagine ("audio" or "vision").
        source_features: (1, dim) or (1, N, dim) source modality features.
        num_source_tokens: Number of source modality tokens in the prompt.
        max_new_tokens: Max generation length.

    Returns:
        Generated text description of the target modality.
    """
    prompt = get_imagination_prompt(
        source_modality, target_modality, num_source_tokens)

    # Prepend </think> to skip Qwen 3's thinking mode, which otherwise
    # spends the generation budget on internal monologue.
    chat = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n</think>\n"
    )

    encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    device = source_features.device
    input_ids = encoded.input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    # Ensure features have batch dim
    if source_features.dim() == 1:
        source_features = source_features.unsqueeze(0)

    # Match model dtype to avoid BFloat16/Float mismatch in the projector.
    # The checkpoint loader puts the LLM in bfloat16; features from disk are
    # float32 by default. Cast to whatever dtype the LLM parameters are in.
    try:
        model_dtype = next(model.llm.parameters()).dtype
        if source_features.dtype != model_dtype:
            source_features = source_features.to(model_dtype)
    except StopIteration:
        pass

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        modality_features={source_modality: source_features},
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=eos_id,
    )

    output_ids = generated[0][input_ids.shape[1]:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    # Safety strip: even with </think>\n prefix, Qwen 3 sometimes emits
    # a leftover closing tag or a stray "<think>...</think>" pair.
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.lstrip("</think>").strip()
    return text


class SelfPlayCache:
    """Lazy cache for cross-modal imagination descriptions.

    On first access for a sample key, generates the description using the
    model. Subsequent accesses return the cached result. Persists to disk.

    Generalized for any modality pair — not just vision→audio.
    """

    def __init__(
        self,
        cache_path: str,
        model=None,
        tokenizer=None,
        source_modality: str = "vision",
        target_modality: str = "audio",
        num_source_tokens: int = 1,
        max_new_tokens: int = 256,
    ):
        self.cache_path = Path(cache_path)
        self.model = model
        self.tokenizer = tokenizer
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.num_source_tokens = num_source_tokens
        self.max_new_tokens = max_new_tokens

        # Load existing cache
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self.cache = json.load(f)
            logger.info(
                f"Self-play cache loaded: {len(self.cache)} entries "
                f"({source_modality}→{target_modality})")
        else:
            self.cache = {}
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._dirty = 0

    def get(self, sample_key: str,
            source_feature: torch.Tensor | None = None) -> str:
        """Get or generate description for a sample."""
        if sample_key in self.cache:
            return self.cache[sample_key]

        if self.model is None or source_feature is None:
            return ""

        # Generate on-the-fly
        try:
            if source_feature.dim() == 1:
                source_feature = source_feature.unsqueeze(0)

            description = imagine_modality(
                self.model, self.tokenizer,
                self.source_modality, self.target_modality,
                source_feature, self.num_source_tokens,
                self.max_new_tokens,
            )
        except Exception as e:
            logger.warning(
                f"Imagination failed for {sample_key}: {e}")
            description = ""

        self.cache[sample_key] = description
        self._dirty += 1

        if self._dirty >= 100:
            self.save()

        return description

    def save(self):
        """Persist cache to disk."""
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)
        self._dirty = 0

    def __len__(self):
        return len(self.cache)


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    r"""Generate and cache self-play proxy descriptions for one modality pair.

    The Phase-1 model is conditioned on the real anchor features and prompted
    to describe the modality it does not observe. Proxies are generated once
    here and cached; nothing is generated online during Phase 2.

    Example:
        python -m src.selfplay.imagine_modality \
            --checkpoint outputs/phase1/final \
            --source_features data/features/image_features_1token.pt \
            --source_modality vision --target_modality audio \
            --output data/selfplay/vision_to_audio.json
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Phase-1 checkpoint directory.")
    parser.add_argument("--model_path", default="models/qwen3-8b",
                        help="Base LLM the checkpoint was trained from.")
    parser.add_argument("--source_features", required=True,
                        help=".pt dict {sample_key: feature} for the real anchor.")
    parser.add_argument("--source_modality", default="vision", choices=["vision", "audio"])
    parser.add_argument("--target_modality", default="audio", choices=["vision", "audio"])
    parser.add_argument("--output", required=True, help="Proxy cache JSON to write.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    if args.source_modality == args.target_modality:
        parser.error("--source_modality and --target_modality must differ")

    from transformers import AutoTokenizer

    from src.models.multimodal_model import MultiModalModel

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)

    features = torch.load(args.source_features, map_location="cpu", weights_only=False)
    keys = sorted(features.keys())
    if args.max_samples:
        keys = keys[:args.max_samples]
    logger.info(f"{len(keys)} {args.source_modality} anchors from {args.source_features}")

    model = MultiModalModel(args.model_path)
    model.add_modality(args.source_modality, input_dim=512, num_tokens=1)
    model.load_checkpoint(args.checkpoint)
    model.to(device).eval()

    cache = SelfPlayCache(
        args.output, model=model, tokenizer=tokenizer,
        source_modality=args.source_modality,
        target_modality=args.target_modality,
        max_new_tokens=args.max_new_tokens,
    )

    try:
        from tqdm import tqdm
        keys_iter = tqdm(keys, desc=f"{args.source_modality}->{args.target_modality}")
    except ImportError:
        keys_iter = keys

    for key in keys_iter:
        if key in cache.cache:
            continue
        cache.get(key, features[key].to(device))

    cache.save()
    logger.info(f"Wrote {len(cache)} proxies to {args.output}")


if __name__ == "__main__":
    main()
