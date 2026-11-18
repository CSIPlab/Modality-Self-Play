"""Encode an imagination cache JSON through a target modality's text encoder.

Takes a SelfPlayCache JSON ({sample_key: text_description}) produced by
imagine_modality.py / generate_imaginations.sh, and encodes each description
through the contrastive text encoder of the TARGET modality, producing a
.pt feature file usable as proxy features in CompositionDataset.

Encoder mapping:
    target=vision      → MetaCLIP text encoder       (512-d)
    target=audio       → CLAP text encoder           (512-d)

Example:
    # Encode the audio→vision imagination cache through MetaCLIP text:
    python -m src.selfplay.encode_imaginations \\
        --input data/selfplay/audio_to_vision_cache.json \\
        --target_modality vision \\
        --output data/selfplay/audio_to_vision_proxy.pt

    # Optionally merge several proxy files into one per-modality file:
    python -m src.selfplay.encode_imaginations --merge \\
        --inputs data/selfplay/audio_to_vision_proxy.pt \\
        --output data/selfplay/vision_proxy_for_composition.pt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


MODALITY_ENCODERS = {
    "vision": "vision",         # MetaCLIP
    "audio": "audio",           # CLAP
}

MODALITY_DIMS = {
    "vision": 512,
    "audio": 512,
}


def split_into_sentences(text: str, target_count: int) -> list[str]:
    """Split a description into N sentences, padding by repeating the last."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return [""] * target_count
    while len(sents) < target_count:
        sents.append(sents[-1])
    return sents[:target_count]


def encode_cache(
    cache_path: Path,
    target_modality: str,
    num_tokens: int,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Encode an imagination cache through the target modality's text encoder."""
    from src.models.encoders import build_encoder

    encoder_name = MODALITY_ENCODERS[target_modality]
    dim = MODALITY_DIMS[target_modality]

    print(f"Loading {target_modality} text encoder ({encoder_name})...")
    encoder = build_encoder(encoder_name)
    encoder_model = encoder.model if hasattr(encoder, "model") else encoder
    if hasattr(encoder_model, "to"):
        encoder_model = encoder_model.to(device)

    print(f"Loading cache from {cache_path}...")
    with open(cache_path) as f:
        cache = json.load(f)
    print(f"  {len(cache)} entries")

    # Cache values are a single caption string, or a list of captions for
    # callers that generated several per sample.
    items: list[tuple[str, object]] = []
    n_list = 0
    for k, v in cache.items():
        if isinstance(v, str) and v.strip():
            items.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(x, str) and x.strip() for x in v):
            items.append((k, v))
            n_list += 1
    print(f"  {len(items)} non-empty entries ({n_list} multi-caption lists)")
    if len(items) < len(cache):
        print(f"  Skipping {len(cache) - len(items)} empty entries")
    is_list_cache = n_list > 0
    if is_list_cache and n_list != len(items):
        raise ValueError(
            f"Mixed cache format: {n_list}/{len(items)} entries are lists. "
            "All entries must be the same form (all strings or all lists).")

    def _to_tensor(output):
        """Extract tensor from encoder output (handles transformers 5.x
        returning BaseModelOutputWithPooling instead of raw tensor)."""
        if isinstance(output, torch.Tensor):
            return output
        # transformers model outputs: try common attributes
        for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(output, attr):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    if attr == "last_hidden_state":
                        return val[:, 0]  # CLS token
                    return val
        # Fallback: index into the output
        if hasattr(output, "__getitem__"):
            return output[0]
        raise TypeError(f"Cannot extract tensor from {type(output)}")

    features: dict[str, torch.Tensor] = {}

    for i in tqdm(range(0, len(items), batch_size),
                  desc=f"Encoding {target_modality}"):
        batch = items[i:i + batch_size]
        keys = [k for k, _ in batch]
        texts = [v for _, v in batch]

        if is_list_cache:
            # Each texts[j] is a list of captions (length N_per_key).
            # Each caption becomes one token → features[key] shape = [N, D].
            # N must be consistent across all entries (validated once below).
            n_per_key = len(texts[0])
            all_sents: list[str] = []
            for caps in texts:
                if len(caps) != n_per_key:
                    raise ValueError(
                        f"Inconsistent multi-caption length: expected "
                        f"{n_per_key}, got {len(caps)}")
                all_sents.extend(caps)
            with torch.no_grad():
                embs = _to_tensor(encoder.get_text_features(all_sents))
                embs = F.normalize(embs, dim=-1).cpu()
            for j, key in enumerate(keys):
                features[key] = embs[j * n_per_key:(j + 1) * n_per_key]
        elif num_tokens > 1:
            all_sents: list[str] = []
            for text in texts:
                all_sents.extend(split_into_sentences(text, num_tokens))
            with torch.no_grad():
                embs = _to_tensor(encoder.get_text_features(all_sents))
                embs = F.normalize(embs, dim=-1).cpu()
            for j, key in enumerate(keys):
                token_embs = embs[j * num_tokens:(j + 1) * num_tokens]
                features[key] = token_embs
        else:
            with torch.no_grad():
                embs = _to_tensor(encoder.get_text_features(texts))
                embs = F.normalize(embs, dim=-1).cpu()
            for j, key in enumerate(keys):
                features[key] = embs[j]

    sample_shape = next(iter(features.values())).shape
    print(f"Encoded {len(features)} {target_modality} proxy features, "
          f"shape per sample: {tuple(sample_shape)}")
    return features


def merge_feature_files(input_paths: list[Path], output_path: Path):
    """Merge multiple proxy feature .pt files into one dict.

    Used to combine proxies from multiple sources
    into a single vision_proxy_for_composition.pt that covers all anchor
    sample keys from several sources.
    """
    print(f"Merging {len(input_paths)} feature files...")
    merged: dict[str, torch.Tensor] = {}
    expected_shape = None
    collisions = 0

    for p in input_paths:
        d = torch.load(p, weights_only=True, map_location="cpu")
        if not d:
            print(f"  Warning: {p} is empty")
            continue
        k0 = next(iter(d))
        shape = tuple(d[k0].shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: {p} has {shape}, expected {expected_shape}")

        for k, v in d.items():
            if k in merged:
                collisions += 1
            merged[k] = v
        print(f"  {p.name}: +{len(d)} features (total: {len(merged)})")

    if collisions:
        print(f"  Warning: {collisions} key collisions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(f"Saved {len(merged)} merged features → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    # Default mode: encode a single cache
    parser.add_argument("--input", type=str, default=None,
                        help="Path to imagination cache JSON")
    parser.add_argument("--target_modality", type=str, default=None,
                        choices=list(MODALITY_ENCODERS.keys()))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_tokens", type=int, default=1,
                        help="Tokens per sample (1 default; 4 for vision quadrant style)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    # Merge mode
    parser.add_argument("--merge", action="store_true",
                        help="Merge multiple proxy .pt files into one")
    parser.add_argument("--inputs", type=str, nargs="+", default=None,
                        help="Input .pt files for merge mode")

    args = parser.parse_args()

    if args.merge:
        if not args.inputs or not args.output:
            parser.error("--merge requires --inputs and --output")
        merge_feature_files(
            [Path(p) for p in args.inputs], Path(args.output))
        return

    # Encode mode
    if not args.input or not args.target_modality or not args.output:
        parser.error("Encode mode requires --input, --target_modality, --output")

    features = encode_cache(
        cache_path=Path(args.input),
        target_modality=args.target_modality,
        num_tokens=args.num_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, out_path)
    print(f"Saved {len(features)} features → {out_path}")


if __name__ == "__main__":
    main()
