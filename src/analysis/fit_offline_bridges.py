"""Fit the frozen text-to-modality bridges used in Phase 2.

Each bridge is a linear ridge map from the contrastive text-tower space into
the corresponding modality-tower space:

    W_m = (T^T T + lambda I)^-1 T^T R,   B_m(t) = W_m^T t,   lambda = 1.0

The bridge is fit once on 1,000 paired (caption, modality) calibration
samples -- 500 used for the fit and 500 held out to report calibration
quality -- and is kept frozen for the remainder of training.

Calibration data: AudioCaps for audio (CLAP audio and CLAP text towers) and
LLaVA-665K for vision (MetaCLIP vision and MetaCLIP text towers).

Outputs:
  <output_dir>/{modality}_bridge_fit.pt  -- state_dict
  <output_dir>/bridge_config.json        -- {modality: "ridge"}

Usage:
  python -m src.analysis.fit_offline_bridges \
      --modalities audio,vision \
      --output_dir outputs/offline_bridges
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.analysis.gap_bridging import LinearRidgeBridge

LAMBDA = 1.0
N_SAMPLES = 1000
N_TRAIN = 500


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1) if x.dim() == 3 else x


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(F.normalize(a, dim=-1),
                               F.normalize(b, dim=-1), dim=-1).mean().item()


def _fit_and_save(name: str, real: torch.Tensor, text: torch.Tensor,
                  out_dir: Path, lam: float, n_train: int) -> None:
    """Fit on the first n_train pairs, report cosine on the held-out rest."""
    real, text = _flatten(real).float(), _flatten(text).float()
    r_tr, t_tr = real[:n_train], text[:n_train]
    r_ho, t_ho = real[n_train:], text[n_train:]
    print(f"  fit on {len(r_tr)}, held out {len(r_ho)}")
    print(f"  raw cos(text, {name}) held-out: {_cos(t_ho, r_ho):.4f}")

    bridge = LinearRidgeBridge(dim=real.shape[-1], lam=lam, normalize_output=True)
    bridge.fit(r_tr, t_tr)
    bridge.eval()

    with torch.no_grad():
        bridged = _cos(bridge.transform(t_ho), r_ho)
    print(f"  bridged cos(text, {name}) held-out: {bridged:.4f}")

    torch.save(bridge.state_dict(), out_dir / f"{name}_bridge_fit.pt")
    print(f"  saved {out_dir / f'{name}_bridge_fit.pt'}")


def fit_audio(out_dir: Path, lam: float, n_samples: int, n_train: int,
              audio_features: str, audio_text_features: str) -> None:
    """Audio bridge: paired (CLAP text tower -> CLAP audio tower) features."""
    print("=== Audio bridge ===")
    audio = torch.load(audio_features, map_location="cpu", weights_only=False)
    text = torch.load(audio_text_features, map_location="cpu", weights_only=False)

    common = sorted(set(audio.keys()) & set(text.keys()))[:n_samples]
    if len(common) < n_samples:
        raise RuntimeError(f"need {n_samples} paired samples, found {len(common)}")

    _fit_and_save("audio",
                  torch.stack([audio[k].float() for k in common]),
                  torch.stack([text[k].float() for k in common]),
                  out_dir, lam, n_train)


def fit_vision(out_dir: Path, device: torch.device, lam: float,
               n_samples: int, n_train: int,
               vision_features: str, vision_captions: str) -> None:
    """Vision bridge: paired (MetaCLIP text tower -> MetaCLIP vision tower)."""
    print("=== Vision bridge ===")
    vis = torch.load(vision_features, map_location="cpu", weights_only=False)

    with open(vision_captions) as f:
        llava = json.load(f)

    # Vision features are keyed by basename; LLaVA uses full paths.
    captions: dict[str, str] = {}
    for entry in llava:
        image, convs = entry.get("image"), entry.get("conversations", [])
        if not image or not convs:
            continue
        for turn in convs:
            if turn.get("from") == "human":
                captions[Path(image).name] = (
                    turn.get("value", "").replace("<image>", "").strip())
                break

    common = [k for k in vis.keys() if k in captions][:n_samples]
    if len(common) < n_samples:
        raise RuntimeError(f"need {n_samples} paired samples, found {len(common)}")

    from src.models.encoders import VisionEncoder
    encoder = VisionEncoder().to(device).eval()
    text = torch.cat(
        [encoder.get_text_features([captions[k][:220] for k in common[i:i + 64]]).cpu()
         for i in range(0, len(common), 64)], dim=0)

    _fit_and_save("vision",
                  torch.stack([vis[k].float() for k in common]),
                  text, out_dir, lam, n_train)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modalities", default="audio,vision")
    parser.add_argument("--audio_features", default="data/audio/audiocaps_clap_features.pt",
                        help="Paired audio-tower features, {sample_key: tensor}.")
    parser.add_argument("--audio_text_features",
                        default="data/audio/audiocaps_clap_text_features.pt",
                        help="Matching text-tower features for the same keys.")
    parser.add_argument("--vision_features",
                        default="data/features/image_features_665k_1token.pt",
                        help="Paired vision-tower features, {sample_key: tensor}.")
    parser.add_argument("--vision_captions",
                        default="data/llava_instruct_150k/llava_v1_5_mix665k.json",
                        help="Caption JSON whose entries carry image and conversations.")
    parser.add_argument("--output_dir", default="outputs/offline_bridges")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES,
                        help="Total calibration pairs (default: 1000).")
    parser.add_argument("--n_train", type=int, default=N_TRAIN,
                        help="Pairs used for the fit; the rest are held out.")
    parser.add_argument("--lam", type=float, default=LAMBDA,
                        help="Ridge regularization strength (default: 1.0).")
    args = parser.parse_args()

    if not 0 < args.n_train < args.n_samples:
        parser.error("--n_train must be between 1 and --n_samples")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mods = [m.strip() for m in args.modalities.split(",")]
    cfg = {}
    if "audio" in mods:
        fit_audio(out, args.lam, args.n_samples, args.n_train,
                  args.audio_features, args.audio_text_features)
        cfg["audio"] = "ridge"
    if "vision" in mods:
        fit_vision(out, device, args.lam, args.n_samples, args.n_train,
                   args.vision_features, args.vision_captions)
        cfg["vision"] = "ridge"

    with open(out / "bridge_config.json", "w") as f:
        json.dump({"method": "offline_fit", "lambda": args.lam, "bridges": cfg},
                  f, indent=2)
    print(f"\n=== Done. bridge_config at {out / 'bridge_config.json'} ===")


if __name__ == "__main__":
    main()
