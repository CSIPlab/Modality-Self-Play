"""Pre-compute MetaCLIP encoder features for all captions and images.

Saves features to disk so the training dataset can load them without
running the encoder (avoids CUDA fork issues with DataLoader workers).

Usage:
    # Pre-compute text proxy features from captions
    python -m src.precompute --mode proxy

    # Pre-compute image features (for real-image control)
    python -m src.precompute --mode real_image

    # Both
    python -m src.precompute --mode both
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.models.config import load_config
from src.models.encoders import VisionEncoder


def precompute_text_features(encoder, captions, output_path, batch_size=128):
    """Encode all captions and save as {filename: tensor} dict."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filenames = list(captions.keys())
    texts = [captions[f] for f in filenames]
    features = {}

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        batch_files = filenames[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            feats = encoder.get_text_features(batch_texts).cpu()
        for j, fname in enumerate(batch_files):
            features[fname] = feats[j]

    torch.save(features, output_path)
    print(f"Saved {len(features)} text features to {output_path}")


def precompute_image_features(encoder, image_dir, image_files, output_path, batch_size=64):
    """Encode all images and save as {filename: tensor} dict."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = {}

    for i in tqdm(range(0, len(image_files), batch_size), desc="Encoding images"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        valid_files = []
        for fname in batch_files:
            img_path = Path(image_dir) / fname
            if img_path.exists():
                batch_images.append(Image.open(img_path).convert("RGB"))
                valid_files.append(fname)

        if batch_images:
            with torch.no_grad():
                feats = encoder.get_modality_features(batch_images).cpu()
            for j, fname in enumerate(valid_files):
                features[fname] = feats[j]

    torch.save(features, output_path)
    print(f"Saved {len(features)} image features to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute encoder features")
    parser.add_argument("--mode", choices=["proxy", "real_image", "both"], default="both")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading encoder on {device}...")
    encoder = VisionEncoder(cfg.encoder.model_name, cfg.encoder.pretrained)
    encoder = encoder.to(device)

    # Load captions
    captions_path = Path(cfg.dataset.captions_path)
    if captions_path.exists():
        with open(captions_path) as f:
            captions = {k: v for k, v in json.load(f).items() if v}
        print(f"Loaded {len(captions)} captions")
    else:
        captions = {}
        print("No captions file found")

    output_dir = Path(cfg.paths.data_dir) / "features"

    if args.mode in ("proxy", "both") and captions:
        precompute_text_features(
            encoder, captions,
            output_dir / "proxy_features.pt",
        )

    if args.mode in ("real_image", "both"):
        # Get all image files referenced by LLaVA
        with open(cfg.dataset.json_path) as f:
            all_samples = json.load(f)
        image_files = sorted(set(s["image"] for s in all_samples))
        print(f"Found {len(image_files)} unique images in LLaVA data")

        precompute_image_features(
            encoder, cfg.dataset.image_dir, image_files,
            output_dir / "image_features.pt",
        )

    print("Done.")


if __name__ == "__main__":
    main()
