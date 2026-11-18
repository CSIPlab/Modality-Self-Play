"""Unified contrastive encoder wrappers for all modalities.

Each encoder provides:
  - get_text_features(texts) → Tensor(batch, dim)
  - get_modality_features(inputs) → Tensor(batch, dim)
  - embedding_dim: int

Supported modalities:
  - vision: MetaCLIP ViT-B/16 (512-d)
  - audio: LAION CLAP (512-d)

Usage:
    from src.models.encoders import build_encoder

    encoder = build_encoder("vision")
    text_feats = encoder.get_text_features(["a cat sitting on a mat"])
    img_feats = encoder.get_modality_features(pil_images)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


CLAP_MODEL = "laion/larger_clap_music_and_speech"
CLAP_SAMPLE_RATE = 48000


class BaseEncoder(nn.Module):
    """Base class for all contrastive encoders."""

    modality_name: str = "base"

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_text_features(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def get_modality_features(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError


class VisionEncoder(BaseEncoder):
    """MetaCLIP ViT-B/16 for vision-text alignment."""

    modality_name = "vision"

    def __init__(self, model_name="ViT-B-16-quickgelu", pretrained="metaclip_fullcc"):
        super().__init__()
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self.model.visual.output_dim

    @property
    def context_length(self) -> int:
        return self.model.context_length

    @torch.no_grad()
    def get_text_features(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts)
        device = next(self.model.parameters()).device
        return self.model.encode_text(tokens.to(device))

    @torch.no_grad()
    def get_modality_features(self, images) -> torch.Tensor:
        if not isinstance(images, (list, tuple)):
            images = [images]
        device = next(self.model.parameters()).device
        pixels = torch.stack([self.preprocess(img) for img in images]).to(device)
        return self.model.encode_image(pixels)


class AudioEncoder(BaseEncoder):
    """LAION CLAP for audio-text alignment."""

    modality_name = "audio"

    def __init__(self, model_name=CLAP_MODEL):
        super().__init__()
        from transformers import ClapModel, ClapProcessor
        # use_safetensors=True avoids the transformers>=4.54 torch.load
        # safety check that requires torch>=2.6 (CVE-2025-32434).
        self.model = ClapModel.from_pretrained(
            model_name, use_safetensors=True).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return 512

    @torch.no_grad()
    def get_text_features(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True,
            truncation=True, max_length=77).to(device)
        return self.model.get_text_features(**inputs)

    @torch.no_grad()
    def get_modality_features(self, audio_arrays, sampling_rate=CLAP_SAMPLE_RATE) -> torch.Tensor:
        """Encode audio waveforms.

        Args:
            audio_arrays: List of numpy arrays or single array.
            sampling_rate: Audio sample rate CLAP expects.
        """
        device = next(self.model.parameters()).device
        if not isinstance(audio_arrays, (list, tuple)):
            audio_arrays = [audio_arrays]
        inputs = self.processor(
            audios=audio_arrays, sampling_rate=sampling_rate,
            return_tensors="pt").to(device)
        return self.model.get_audio_features(**inputs)


# ── Registry ────────────────────────────────────────────────────────────

ENCODER_REGISTRY = {
    "vision": VisionEncoder,
    "audio": AudioEncoder,
}


def build_encoder(modality: str, **kwargs) -> BaseEncoder:
    """Build a contrastive encoder for the given modality."""
    if modality not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown modality: {modality}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}")
    return ENCODER_REGISTRY[modality](**kwargs)
