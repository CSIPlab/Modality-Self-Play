"""Pre-compute CLAP audio and MetaCLIP vision features for AVQA.

AVQA and MUSIC-AVQA are evaluation benchmarks: MSP is never trained on
them and their training splits are never used. This extracts features for
the evaluation split only, in the same format the evaluation scripts read.

Usage:
    python -m src.data.precompute_avqa --data-dir <avqa-root> \
        --output-dir <features-out>
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from tqdm import tqdm
from src.models.encoders import CLAP_MODEL, CLAP_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------

# AVQA's evaluation split; the dataset authors label this file the test set.
SPLIT_CONFIG = {
    "val": {
        "audio_dir": "audio",
        "image_dir": "image",
    },
}


# ---------------------------------------------------------------------------
# Vision features (MetaCLIP)
# ---------------------------------------------------------------------------

def extract_vision_features(image_dir: Path, output_path: Path,
                            batch_size: int = 64, device: str = "cuda"):
    """Encode images with MetaCLIP ViT-B/16 and save as {video_name: tensor(512)}.

    Features are L2-normalised to match the existing pipeline.
    """
    from src.models.encoders import VisionEncoder

    print(f"Loading MetaCLIP encoder on {device}...")
    encoder = VisionEncoder()
    encoder = encoder.to(device)

    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(image_dir.glob("*.png"))
    print(f"Found {len(image_files)} image files in {image_dir}")

    if len(image_files) == 0:
        print("No images found, skipping vision extraction.")
        return

    features = {}
    skipped = 0

    for i in tqdm(range(0, len(image_files), batch_size), desc="Encoding vision"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        valid_names = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                batch_images.append(img)
                valid_names.append(f.stem)
            except Exception as e:
                print(f"  Skipping corrupt image {f.name}: {e}")
                skipped += 1

        if not batch_images:
            continue

        with torch.no_grad():
            feats = encoder.get_modality_features(batch_images)  # (B, 512)
            feats = F.normalize(feats, dim=-1).cpu()

        for j, name in enumerate(valid_names):
            features[name] = feats[j]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    print(f"Saved {len(features)} vision features to {output_path}")
    if skipped:
        print(f"  ({skipped} files skipped due to errors)")

    _print_stats(features)


# ---------------------------------------------------------------------------
# Audio features (CLAP)
# ---------------------------------------------------------------------------

def extract_audio_features(audio_dir: Path, output_path: Path,
                           batch_size: int = 16, device: str = "cuda"):
    """Encode audio with CLAP and save as {video_name: tensor(512)}.

    Uses LAION-CLAP via HuggingFace transformers.
    """
    from transformers import ClapModel, ClapProcessor

    print(f"Loading CLAP model on {device}...")
    model = ClapModel.from_pretrained(CLAP_MODEL)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model = model.to(device).eval()

    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    if len(audio_files) == 0:
        print("No audio found, skipping audio extraction.")
        return

    features = {}
    skipped = 0

    for i in tqdm(range(0, len(audio_files), batch_size), desc="Encoding audio"):
        batch_files = audio_files[i:i + batch_size]
        batch_waveforms = []
        valid_names = []

        for f in batch_files:
            try:
                waveform, sr = torchaudio.load(str(f))
                # Resample to 48kHz if needed (CLAP expectation)
                if sr != CLAP_SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, sr, CLAP_SAMPLE_RATE)
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                batch_waveforms.append(waveform.squeeze(0).numpy())
                valid_names.append(f.stem)
            except Exception as e:
                print(f"  Skipping corrupt audio {f.name}: {e}")
                skipped += 1

        if not batch_waveforms:
            continue

        inputs = processor(
            audios=batch_waveforms, sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_feats = model.get_audio_features(**inputs).cpu()  # (B, 512)

        for j, name in enumerate(valid_names):
            features[name] = audio_feats[j]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    print(f"Saved {len(features)} audio features to {output_path}")
    if skipped:
        print(f"  ({skipped} files skipped due to errors)")

    _print_stats(features)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_stats(features: dict):
    """Print feature dimension and norm statistics."""
    if not features:
        return
    tensors = torch.stack(list(features.values()))
    norms = tensors.norm(dim=-1)
    print(f"  Feature dim: {tensors.shape[1]}")
    print(f"  Norm: mean={norms.mean():.3f}  std={norms.std():.3f}")


def process_split(split: str, data_dir: Path, output_dir: Path,
                  batch_size: int, device: str):
    """Extract both audio and vision features for one split."""
    cfg = SPLIT_CONFIG[split]
    audio_dir = data_dir / cfg["audio_dir"]
    image_dir = data_dir / cfg["image_dir"]

    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"  Audio dir:  {audio_dir}")
    print(f"  Image dir:  {image_dir}")
    print(f"{'='*60}\n")

    # Audio
    if audio_dir.exists():
        extract_audio_features(
            audio_dir,
            output_dir / f"{split}_audio_features.pt",
            batch_size=batch_size,
            device=device,
        )
    else:
        print(f"Audio directory not found: {audio_dir}")

    # Vision
    if image_dir.exists():
        extract_vision_features(
            image_dir,
            output_dir / f"{split}_vision_features.pt",
            batch_size=batch_size * 4,  # vision batches are lighter
            device=device,
        )
    else:
        print(f"Image directory not found: {image_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLAP audio and MetaCLIP vision features for AVQA",
    )
    parser.add_argument(
        "--split", choices=["val"], default="val",
        help="Evaluation split (the only split this script processes).",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/avqa",
        help="Root AVQA data directory (default: data/avqa)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for .pt files (default: <data-dir>/features)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for audio encoding; vision uses 4x this (default: 16)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "features"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    splits = [args.split]
    for split in splits:
        process_split(split, data_dir, output_dir, args.batch_size, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
