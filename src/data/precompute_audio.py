"""Pre-compute CLAP features for MUSIC-AVQA.

Encodes real audio files through CLAP audio encoder, and optionally
text captions through CLAP text encoder for proxy features.

Usage:
    # Real audio features (for training and eval)
    python -m src.precompute_audio --mode audio \
        --audio_dir data/music_avqa/audio \
        --output data/music_avqa/features/real_audio_features.pt

    # Text proxy features (for ablation only)
    python -m src.precompute_audio --mode text \
        --captions data/music_avqa/audio_captions.json \
        --output data/music_avqa/features/proxy_audio_features.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from src.models.encoders import CLAP_MODEL, CLAP_SAMPLE_RATE




def precompute_real_audio_features(audio_dir, output_path, batch_size=16):
    """Encode real audio files through CLAP audio encoder."""
    from transformers import ClapModel, ClapProcessor

    print("Loading CLAP model...")
    model = ClapModel.from_pretrained(CLAP_MODEL)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    features = {}
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Encoding audio"):
        batch_files = audio_files[i:i + batch_size]
        batch_waveforms = []

        for f in batch_files:
            waveform, sr = torchaudio.load(str(f))
            # Resample to 48kHz if needed
            if sr != CLAP_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, CLAP_SAMPLE_RATE)
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            batch_waveforms.append(waveform.squeeze(0).numpy())

        inputs = processor(audios=batch_waveforms, sampling_rate=CLAP_SAMPLE_RATE,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_features = model.get_audio_features(**inputs).cpu()

        for j, f in enumerate(batch_files):
            video_id = f.stem  # e.g. "00008214"
            features[video_id] = audio_features[j]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    print(f"Saved {len(features)} real audio features to {output_path}")

    norms = torch.stack(list(features.values())).norm(dim=-1)
    print(f"Feature dim: {list(features.values())[0].shape[0]}")
    print(f"Norm: mean={norms.mean():.3f} std={norms.std():.3f}")


def precompute_clap_text_features(captions, output_path, batch_size=64):
    """Encode audio descriptions through CLAP text encoder (proxy features)."""
    from transformers import ClapModel, ClapProcessor

    print("Loading CLAP model...")
    model = ClapModel.from_pretrained(CLAP_MODEL)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    video_ids = list(captions.keys())
    texts = [captions[vid] for vid in video_ids]
    features = {}

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding CLAP text"):
        batch_ids = video_ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]

        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**inputs).cpu()

        for j, vid in enumerate(batch_ids):
            features[vid] = text_features[j]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    print(f"Saved {len(features)} CLAP text features to {output_path}")

    norms = torch.stack(list(features.values())).norm(dim=-1)
    print(f"Feature dim: {list(features.values())[0].shape[0]}")
    print(f"Norm: mean={norms.mean():.3f} std={norms.std():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLAP features")
    parser.add_argument("--mode", choices=["audio", "text", "both"],
                        default="both")
    parser.add_argument("--audio_dir", type=str, default="data/music_avqa/audio")
    parser.add_argument("--captions", type=str, default="data/music_avqa/audio_captions.json")
    parser.add_argument("--output_dir", type=str, default="data/music_avqa/features")
    parser.add_argument("--output_filename", type=str, default=None,
                        help="Override output filename (in output_dir).")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "audio":
        out = output_dir / (args.output_filename or "real_audio_features.pt")
        precompute_real_audio_features(args.audio_dir, out, batch_size=args.batch_size)
    elif args.mode == "text":
        with open(args.captions) as f:
            captions = {k: v for k, v in json.load(f).items() if v}
        print(f"Loaded {len(captions)} audio captions")
        precompute_clap_text_features(
            captions, output_dir / (args.output_filename or "proxy_audio_features.pt")
        )
    elif args.mode == "both":
        precompute_real_audio_features(args.audio_dir, output_dir / "real_audio_features.pt")
        with open(args.captions) as f:
            captions = {k: v for k, v in json.load(f).items() if v}
        precompute_clap_text_features(captions, output_dir / "proxy_audio_features.pt")


if __name__ == "__main__":
    main()
