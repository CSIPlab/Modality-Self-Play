"""Audio projector: maps CLAP features into the LLM embedding space."""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioProjector(nn.Module):
    """Two-layer MLP projecting CLAP audio features to LLM token space.

    LayerNorm → Linear → GELU → Linear.
    Supports expanding a single CLAP vector into N audio tokens.

    Args:
        input_dim: CLAP embedding dimension (default: 512).
        output_dim: LLM hidden dimension (default: 4096 for Qwen3-8B).
        hidden_dim: Intermediate MLP dimension (default: input_dim).
        num_audio_tokens: Number of audio tokens to produce per sample (default: 1).
        normalize: L2-normalize input features before projection (default: True).
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 4096,
        hidden_dim: int | None = None,
        num_audio_tokens: int = 1,
        normalize: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.num_audio_tokens = num_audio_tokens
        self.normalize = normalize
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * num_audio_tokens),
        )
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project CLAP features to LLM space.

        Args:
            x: Either (batch, input_dim) global features or
               (batch, tokens, input_dim) tokenized features.

        Returns:
            (batch, num_audio_tokens, output_dim) projected features.
        """
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)

        if x.dim() == 2:
            out = self.mlp(x)
            return out.view(-1, self.num_audio_tokens, self.output_dim)

        if x.dim() == 3:
            batch, tokens, dim = x.shape
            flat = x.view(batch * tokens, dim)
            out = self.token_mlp(flat)
            return out.view(batch, tokens, self.output_dim)

        raise ValueError(f"Unsupported audio feature shape: {tuple(x.shape)}")
