"""Tests for the MSP configuration system."""

from src.models.config import LLM_REGISTRY, MSPConfig, load_config


def test_default_config_loads():
    """Default config should load without errors."""
    cfg = load_config()
    assert isinstance(cfg, MSPConfig)
    assert cfg.project_name == "msp"
    assert cfg.seed == 42


def test_projector_dims_auto_derived():
    """Projector dims should be auto-derived from encoder/LLM pairing."""
    cfg = load_config()
    # Default: MetaCLIP (512) -> Qwen3-8B (4096)
    assert cfg.projector.input_dim == 512
    assert cfg.projector.output_dim == 4096


def test_llm_override_4b():
    """Overriding LLM to 4B should update hidden_size and projector output."""
    cfg = load_config(overrides=["llm.name=qwen3-4b"])
    assert cfg.llm.name == "qwen3-4b"
    assert cfg.llm.hidden_size == 2560
    assert cfg.llm.hf_id == "Qwen/Qwen3-4B"
    assert cfg.projector.output_dim == 2560


def test_llm_override_8b():
    """Overriding LLM to 8B should update hidden_size and projector output."""
    cfg = load_config(overrides=["llm.name=qwen3-8b"])
    assert cfg.llm.name == "qwen3-8b"
    assert cfg.llm.hidden_size == 4096
    assert cfg.llm.hf_id == "Qwen/Qwen3-8B"
    assert cfg.projector.output_dim == 4096


def test_registry_completeness():
    """LLM registry should contain all three Qwen3 variants."""
    expected_models = {"qwen3-1.7b", "qwen3-4b", "qwen3-8b"}
    assert set(LLM_REGISTRY.keys()) == expected_models

    for name, entry in LLM_REGISTRY.items():
        assert "hf_id" in entry, f"Missing hf_id for {name}"
        assert "hidden_size" in entry, f"Missing hidden_size for {name}"
        assert isinstance(entry["hidden_size"], int)


def test_encoder_config_defaults():
    """Encoder config should have correct MetaCLIP defaults."""
    cfg = load_config()
    assert cfg.encoder.name == "metaclip-b16-fullcc"
    assert cfg.encoder.hidden_size == 512
    assert cfg.encoder.context_length == 77
    assert cfg.encoder.num_patches == 196
