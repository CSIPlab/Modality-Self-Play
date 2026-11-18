"""Tests for the MetaCLIP encoder wrapper.

Uses MetaCLIP ViT-B/16 (metaclip_fullcc) via OpenCLIP.
Tests require the open_clip_torch package.
"""

import pytest


@pytest.fixture(scope="module")
def encoder():
    from src.models.encoders import VisionEncoder

    return VisionEncoder()


def test_text_features_single(encoder):
    """Single text input should produce (1, 512) features."""
    features = encoder.get_text_features(["a photo of a cat"])
    assert features.shape == (1, 512)


def test_text_features_batch(encoder):
    """Batch of N texts should produce (N, 512) features."""
    texts = ["a photo of a cat", "a dog running in a park", "sunset over the ocean"]
    features = encoder.get_text_features(texts)
    assert features.shape == (3, 512)


def test_embedding_dim(encoder):
    """embedding_dim property should return 512."""
    assert encoder.embedding_dim == 512


def test_context_length(encoder):
    """context_length property should return 77."""
    assert encoder.context_length == 77
