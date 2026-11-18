"""Verify that all required dependencies are installed and configured."""

import importlib
import re
from pathlib import Path

import pytest

# Import names for the distributions listed in requirements.txt, so the two
# cannot drift apart.
IMPORT_NAMES = {
    "open_clip_torch": "open_clip",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
}


def _required_packages() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    names = []
    for line in (root / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        dist = re.split(r"[<>=\[]", line)[0].strip()
        names.append(IMPORT_NAMES.get(dist, dist.replace("-", "_")))
    return names


REQUIRED_PACKAGES = _required_packages()


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_import(package):
    """Each required package should be importable."""
    importlib.import_module(package)


def test_transformers_version():
    """transformers >= 4.51.0 is required for Qwen3 support."""
    import transformers
    from packaging.version import Version

    assert Version(transformers.__version__) >= Version("4.51.0"), (
        f"transformers {transformers.__version__} < 4.51.0; "
        "upgrade with: pip install 'transformers>=4.51.0'"
    )


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_cuda_available():
    """CUDA should be available for training."""
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_bf16_support():
    """BFloat16 should be supported on the available GPU."""
    import torch

    assert torch.cuda.is_bf16_supported(), (
        "BFloat16 not supported on this GPU; "
        "Ampere (A100) or newer is recommended"
    )
