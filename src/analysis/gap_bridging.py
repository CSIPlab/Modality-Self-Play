"""Gap bridging techniques for reducing the CLAP text-audio modality gap.

Each technique transforms CLAP text proxy features to be closer to the
distribution of real CLAP audio features. All follow the same interface:

    bridge = MeanShiftBridge()
    bridge.fit(calibration_audio_feats, calibration_text_feats)
    corrected = bridge.transform(text_proxy_feats)

Available techniques:
  1. MeanShiftBridge — subtract modality-specific means (GR-CLIP style)
  2. EmbeddingDropoutBridge — stochastic dropout during training
  3. ResidualMLPBridge — learned residual correction network
  4. WhiteningBridge — ZCA whitening + re-coloring with audio statistics
  5. AffineAlignBridge — learned affine transformation (rotation + shift)
  6. LinearRidgeBridge — regularized linear map fit on matched pairs
  7. CompositeBridge — chain multiple techniques

Usage:
    from src.gap_bridging import MeanShiftBridge, ResidualMLPBridge, build_bridge

    # Simple:
    bridge = MeanShiftBridge()
    bridge.fit(audio_feats, text_feats)
    corrected = bridge.transform(text_proxy_feats)

    # From config:
    bridge = build_bridge("mean_shift+dropout", audio_feats, text_feats)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBridge(nn.Module):
    """Base class for gap bridging techniques."""

    def fit(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        """Fit on a calibration set of matched audio-text feature pairs."""
        pass

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        """Transform text features to be closer to audio distribution."""
        raise NotImplementedError

    def forward(self, text_feats: torch.Tensor) -> torch.Tensor:
        return self.transform(text_feats)


class IdentityBridge(BaseBridge):
    """No transformation. Baseline."""

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        return text_feats


class MeanShiftBridge(BaseBridge):
    """Shift text features by the mean gap vector.

    The simplest possible correction: compute the average difference between
    audio and text centroids, then add it to all text features. Zero parameters.
    Based on GR-CLIP (2025).
    """

    def __init__(self, normalize_output: bool = True):
        super().__init__()
        self.normalize_output = normalize_output
        self.register_buffer("gap_vector", torch.zeros(512))
        self._fitted = False

    def fit(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        # Flatten multi-token features
        if audio_feats.dim() == 3:
            audio_feats = audio_feats.reshape(-1, audio_feats.shape[-1])
        if text_feats.dim() == 3:
            text_feats = text_feats.reshape(-1, text_feats.shape[-1])

        audio_mean = audio_feats.mean(dim=0)
        text_mean = text_feats.mean(dim=0)
        self.gap_vector = (audio_mean - text_mean).detach()
        self._fitted = True

        cos_before = F.cosine_similarity(audio_mean.unsqueeze(0),
                                          text_mean.unsqueeze(0)).item()
        shifted_mean = text_mean + self.gap_vector
        cos_after = F.cosine_similarity(audio_mean.unsqueeze(0),
                                         shifted_mean.unsqueeze(0)).item()
        print(f"MeanShiftBridge: gap_norm={self.gap_vector.norm():.4f}, "
              f"centroid_cos {cos_before:.4f} → {cos_after:.4f}")

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        shifted = text_feats + self.gap_vector.to(text_feats.device)
        if self.normalize_output:
            shifted = F.normalize(shifted, dim=-1)
        return shifted


class EmbeddingDropoutBridge(BaseBridge):
    """Apply dropout to features during training to prevent modality overfitting.

    Based on Saijo et al. (MERL, ICASSP 2025). Prevents the downstream
    projector from overfitting to text-specific embedding structure.
    """

    def __init__(self, dropout_prob: float = 0.1, feature_noise: float = 0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.feature_noise = feature_noise

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout_prob > 0:
            text_feats = F.dropout(text_feats, p=self.dropout_prob, training=True)
        if self.training and self.feature_noise > 0:
            noise = torch.randn_like(text_feats) * self.feature_noise
            text_feats = text_feats + noise
            text_feats = F.normalize(text_feats, dim=-1)
        return text_feats


class ResidualMLPBridge(BaseBridge):
    """Learned residual correction: predict the gap as a function of the input.

    audio_feat ≈ text_feat + residual_mlp(text_feat)

    The MLP learns to predict the per-sample correction needed to map text
    features closer to their corresponding audio features. Can be pre-trained
    on a small calibration set or co-trained with the audio projector.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 normalize_output: bool = True):
        super().__init__()
        self.normalize_output = normalize_output
        self.residual_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Initialize near-zero so initial output ≈ input
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def fit(self, audio_feats: torch.Tensor, text_feats: torch.Tensor,
            epochs: int = 100, lr: float = 1e-3):
        """Pre-train the residual MLP on a calibration set."""
        if audio_feats.dim() == 3:
            audio_feats = audio_feats.reshape(-1, audio_feats.shape[-1])
        if text_feats.dim() == 3:
            text_feats = text_feats.reshape(-1, text_feats.shape[-1])

        device = next(self.parameters()).device
        audio_feats = audio_feats.to(device)
        text_feats = text_feats.to(device)

        optimizer = torch.optim.Adam(self.residual_net.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            residual = self.residual_net(text_feats)
            predicted = text_feats + residual
            if self.normalize_output:
                predicted = F.normalize(predicted, dim=-1)
                target = F.normalize(audio_feats, dim=-1)
            else:
                target = audio_feats

            # Cosine similarity loss
            loss = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                cos = F.cosine_similarity(predicted, target, dim=-1).mean().item()
                print(f"  ResidualMLP epoch {epoch+1}: cos={cos:.4f}, loss={loss.item():.4f}")

        self.eval()

        # Report final quality
        with torch.no_grad():
            predicted = self.transform(text_feats)
            target = F.normalize(audio_feats, dim=-1)
            final_cos = F.cosine_similarity(predicted, target, dim=-1).mean().item()
            raw_cos = F.cosine_similarity(
                F.normalize(text_feats, dim=-1), target, dim=-1).mean().item()
        print(f"ResidualMLPBridge: cos {raw_cos:.4f} → {final_cos:.4f}")

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        residual = self.residual_net(text_feats)
        corrected = text_feats + residual
        if self.normalize_output:
            corrected = F.normalize(corrected, dim=-1)
        return corrected


class WhiteningBridge(BaseBridge):
    """ZCA whitening of text features, then re-coloring with audio statistics.

    Aligns first and second-order statistics without learned parameters.
    1. Compute text mean and covariance → ZCA whiten
    2. Compute audio mean and covariance → re-color
    """

    def __init__(self, normalize_output: bool = True, reg: float = 1e-5):
        super().__init__()
        self.normalize_output = normalize_output
        self.reg = reg
        self.register_buffer("text_mean", torch.zeros(512))
        self.register_buffer("audio_mean", torch.zeros(512))
        self.register_buffer("whiten_matrix", torch.eye(512))
        self.register_buffer("color_matrix", torch.eye(512))

    def fit(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        if audio_feats.dim() == 3:
            audio_feats = audio_feats.reshape(-1, audio_feats.shape[-1])
        if text_feats.dim() == 3:
            text_feats = text_feats.reshape(-1, text_feats.shape[-1])

        # Text statistics
        t_mean = text_feats.mean(dim=0)
        t_centered = text_feats - t_mean
        t_cov = (t_centered.T @ t_centered) / (len(text_feats) - 1)

        # Audio statistics
        a_mean = audio_feats.mean(dim=0)
        a_centered = audio_feats - a_mean
        a_cov = (a_centered.T @ a_centered) / (len(audio_feats) - 1)

        # ZCA whitening matrix for text
        U_t, S_t, _ = torch.linalg.svd(t_cov)
        whiten = U_t @ torch.diag(1.0 / (S_t + self.reg).sqrt()) @ U_t.T

        # Coloring matrix from audio
        U_a, S_a, _ = torch.linalg.svd(a_cov)
        color = U_a @ torch.diag((S_a + self.reg).sqrt()) @ U_a.T

        self.text_mean = t_mean.detach()
        self.audio_mean = a_mean.detach()
        self.whiten_matrix = whiten.detach()
        self.color_matrix = color.detach()

        # Report
        with torch.no_grad():
            corrected = self.transform(text_feats)
            target = F.normalize(audio_feats, dim=-1)
            raw_cos = F.cosine_similarity(
                F.normalize(text_feats, dim=-1), target, dim=-1).mean().item()
            new_cos = F.cosine_similarity(corrected, target, dim=-1).mean().item()
        print(f"WhiteningBridge: cos {raw_cos:.4f} → {new_cos:.4f}")

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        device = text_feats.device
        centered = text_feats - self.text_mean.to(device)
        whitened = centered @ self.whiten_matrix.to(device).T
        colored = whitened @ self.color_matrix.to(device).T + self.audio_mean.to(device)
        if self.normalize_output:
            colored = F.normalize(colored, dim=-1)
        return colored


class AffineAlignBridge(BaseBridge):
    """Learned affine transformation: W @ text_feat + b.

    More expressive than mean-shift, less expressive than ResidualMLP.
    Learns a rotation/scaling to align text features to audio space.
    """

    def __init__(self, dim: int = 512, normalize_output: bool = True):
        super().__init__()
        self.normalize_output = normalize_output
        self.W = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def fit(self, audio_feats: torch.Tensor, text_feats: torch.Tensor,
            epochs: int = 200, lr: float = 1e-3):
        if audio_feats.dim() == 3:
            audio_feats = audio_feats.reshape(-1, audio_feats.shape[-1])
        if text_feats.dim() == 3:
            text_feats = text_feats.reshape(-1, text_feats.shape[-1])

        device = next(self.parameters()).device
        audio_feats = F.normalize(audio_feats, dim=-1).to(device)
        text_feats = text_feats.to(device)

        optimizer = torch.optim.Adam(self.W.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            predicted = self.transform(text_feats)
            loss = 1.0 - F.cosine_similarity(predicted, audio_feats, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.eval()

        with torch.no_grad():
            final_cos = F.cosine_similarity(
                self.transform(text_feats), audio_feats, dim=-1).mean().item()
            raw_cos = F.cosine_similarity(
                F.normalize(text_feats, dim=-1), audio_feats, dim=-1).mean().item()
        print(f"AffineAlignBridge: cos {raw_cos:.4f} → {final_cos:.4f}")

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        out = self.W(text_feats)
        if self.normalize_output:
            out = F.normalize(out, dim=-1)
        return out


class LinearRidgeBridge(BaseBridge):
    """Regularized linear proxy→real map fit on matched feature pairs."""

    bridge_method = "ridge"

    def __init__(self, dim: int = 512, lam: float = 1.0, normalize_output: bool = True):
        super().__init__()
        self.normalize_output = normalize_output
        self.lam = lam
        self.W = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def fit(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        if audio_feats.dim() == 3:
            audio_feats = audio_feats.reshape(-1, audio_feats.shape[-1])
        if text_feats.dim() == 3:
            text_feats = text_feats.reshape(-1, text_feats.shape[-1])

        x = text_feats.float()
        y = audio_feats.float()
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        xb = torch.cat([x, ones], dim=1)
        reg = torch.eye(xb.shape[1], dtype=xb.dtype, device=xb.device)
        reg[-1, -1] = 0.0
        beta = torch.linalg.solve(xb.T @ xb + self.lam * reg, xb.T @ y)
        weight = beta[:-1, :]
        bias = beta[-1, :]
        with torch.no_grad():
            self.W.weight.copy_(weight.T)
            self.W.bias.copy_(bias)

        with torch.no_grad():
            raw_cos = F.cosine_similarity(
                F.normalize(text_feats.float(), dim=-1),
                F.normalize(audio_feats.float(), dim=-1),
                dim=-1,
            ).mean().item()
            new_cos = F.cosine_similarity(
                self.transform(text_feats.float()),
                F.normalize(audio_feats.float(), dim=-1),
                dim=-1,
            ).mean().item()
        print(f"LinearRidgeBridge(lam={self.lam:g}): cos {raw_cos:.4f} -> {new_cos:.4f}")

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        dtype = self.W.weight.dtype
        out = self.W(text_feats.to(dtype))
        if self.normalize_output:
            out = F.normalize(out, dim=-1)
        return out


class CompositeBridge(BaseBridge):
    """Chain multiple bridging techniques in sequence."""

    def __init__(self, bridges: list[BaseBridge]):
        super().__init__()
        self.bridges = nn.ModuleList(bridges)

    def fit(self, audio_feats: torch.Tensor, text_feats: torch.Tensor, **kwargs):
        current = text_feats
        for bridge in self.bridges:
            bridge.fit(audio_feats, current, **kwargs)
            with torch.no_grad():
                current = bridge.transform(current)

    def transform(self, text_feats: torch.Tensor) -> torch.Tensor:
        x = text_feats
        for bridge in self.bridges:
            x = bridge.transform(x)
        return x


# ── Factory ─────────────────────────────────────────────────────────────

class CoTrainedResidualBridge(ResidualMLPBridge):
    """Residual MLP that skips pre-fitting and learns end-to-end from task loss.

    No audio calibration data needed. The bridge parameters are included in
    the main training optimizer and learn to transform CLAP text features into
    whatever representation the downstream LLM finds most useful.
    """

    def fit(self, audio_feats=None, text_feats=None, **kwargs):
        # Skip pre-fitting — parameters will be trained end-to-end
        pass


class CoTrainedAffineBridge(AffineAlignBridge):
    """Affine alignment that learns end-to-end from task loss only."""

    def fit(self, audio_feats=None, text_feats=None, **kwargs):
        pass


BRIDGE_REGISTRY = {
    "none": IdentityBridge,
    "mean_shift": MeanShiftBridge,
    "dropout": EmbeddingDropoutBridge,
    "residual_mlp": ResidualMLPBridge,
    "whiten": WhiteningBridge,
    "affine": AffineAlignBridge,
    "ridge": LinearRidgeBridge,
    "co_residual": CoTrainedResidualBridge,
    "co_affine": CoTrainedAffineBridge,
}


def build_bridge(
    method: str,
    audio_feats: Optional[torch.Tensor] = None,
    text_feats: Optional[torch.Tensor] = None,
    dropout_prob: float = 0.1,
    feature_noise: float = 0.0,
    residual_hidden: int = 256,
    residual_epochs: int = 100,
    ridge_lam: float = 1.0,
    input_dim: Optional[int] = None,
    **kwargs,
) -> BaseBridge:
    """Build and optionally fit a gap bridge from a method string.

    Supports composite methods with '+': e.g., "mean_shift+dropout"

    Args:
        input_dim: explicit feature dim for bridges that need it. Overrides
            inference from audio_feats. Defaults: shape[-1] of audio_feats if
            provided, else 512. Must be set for co-trained bridges when no
            calibration data is passed (e.g., in train_composition.py).
    """
    methods = [m.strip() for m in method.split("+")]

    def _resolve_dim(default: int = 512) -> int:
        if input_dim is not None:
            return input_dim
        if audio_feats is not None:
            return audio_feats.shape[-1]
        return default

    bridges = []
    for m in methods:
        if m == "none":
            bridges.append(IdentityBridge())
        elif m == "mean_shift":
            bridges.append(MeanShiftBridge())
        elif m == "dropout":
            bridges.append(EmbeddingDropoutBridge(
                dropout_prob=dropout_prob, feature_noise=feature_noise))
        elif m == "residual_mlp":
            bridges.append(ResidualMLPBridge(
                input_dim=_resolve_dim(), hidden_dim=residual_hidden))
        elif m == "whiten":
            bridges.append(WhiteningBridge())
        elif m == "affine":
            bridges.append(AffineAlignBridge(dim=_resolve_dim()))
        elif m == "ridge":
            bridges.append(LinearRidgeBridge(dim=_resolve_dim(), lam=ridge_lam))
        elif m == "co_residual":
            bridges.append(CoTrainedResidualBridge(
                input_dim=_resolve_dim(), hidden_dim=residual_hidden))
        elif m == "co_affine":
            bridges.append(CoTrainedAffineBridge(dim=_resolve_dim()))
        else:
            raise ValueError(f"Unknown bridge method: {m}")

    if len(bridges) == 1:
        bridge = bridges[0]
    else:
        bridge = CompositeBridge(bridges)

    # Fit if calibration data provided
    if audio_feats is not None and text_feats is not None:
        if isinstance(bridge, CompositeBridge):
            bridge.fit(audio_feats, text_feats)
        else:
            fit_kwargs = {}
            if isinstance(bridge, ResidualMLPBridge):
                fit_kwargs["epochs"] = residual_epochs
            elif isinstance(bridge, AffineAlignBridge):
                fit_kwargs["epochs"] = residual_epochs
            bridge.fit(audio_feats, text_feats, **fit_kwargs)

    return bridge
