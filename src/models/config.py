"""MSP configuration: dataclasses, YAML loading, and LLM registry."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# LLM Registry
# ---------------------------------------------------------------------------

LLM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3-1.7b": {"hf_id": "Qwen/Qwen3-1.7B", "hidden_size": 2048},
    "qwen3-4b": {"hf_id": "Qwen/Qwen3-4B", "hidden_size": 2560},
    "qwen3-8b": {"hf_id": "Qwen/Qwen3-8B", "hidden_size": 4096},
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    name: str = "metaclip-b16-fullcc"
    model_name: str = "ViT-B-16-quickgelu"
    pretrained: str = "metaclip_fullcc"
    hidden_size: int = 512
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 196
    context_length: int = 77
    loss: str = "softmax"
    freeze: bool = True


@dataclass
class LLMConfig:
    name: str = "qwen3-8b"
    hf_id: str = "Qwen/Qwen3-8B"
    local_path: str = "models/qwen3-8b"
    hidden_size: int = 4096


@dataclass
class ProjectorConfig:
    input_dim: int = 512
    output_dim: int = 4096
    hidden_dim: int = 512
    activation: str = "gelu"
    normalize: bool = True  # L2-normalize encoder features before projection


@dataclass
class PathsConfig:
    root: str = "."
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "outputs"
    log_dir: str = "logs"


@dataclass
class DatasetConfig:
    json_path: str = "data/llava_instruct_150k/llava_instruct_150k.json"
    captions_path: str = "data/captions/captions.json"
    image_dir: str = "data/coco/train2017"
    val_split: float = 0.02
    max_seq_length: int = 2048
    max_samples: int = 0  # 0 = use all, >0 = limit total samples
    system_prompt: str = "You are a helpful assistant."


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 512
    alpha: int = 16
    dropout: float = 0.05
    # LoRA is applied to the attention projections.
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])


@dataclass
class TrainingConfig:
    mode: str = "proxy"  # proxy | real_image | text_only
    stage: int = 1  # 1 = projector only, 2 = LoRA/full
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 50
    output_dir: str = "outputs"
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class EvalConfig:
    batch_size: int = 16
    max_new_tokens: int = 256
    temperature: float = 0.0


@dataclass
class MSPConfig:
    project_name: str = "msp"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(
    path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> MSPConfig:
    """Load configuration from a YAML file with optional CLI-style overrides.

    Args:
        path: Path to a YAML config file. If None, uses configs/base.yaml
              relative to the project root.
        overrides: List of dotlist overrides, e.g. ["llm.name=qwen3-4b"].

    Returns:
        A fully-resolved MSPConfig with projector dims auto-derived
        from the encoder/LLM pairing.
    """
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = str(repo_root / "configs" / "base.yaml")

    cfg = OmegaConf.load(path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve LLM from registry if name was overridden
    llm_name = OmegaConf.select(cfg, "llm.name", default="qwen3-8b")
    if llm_name in LLM_REGISTRY:
        registry_entry = LLM_REGISTRY[llm_name]
        if OmegaConf.is_missing(cfg, "llm.hf_id") or cfg.llm.get("hf_id") is None:
            cfg.llm.hf_id = registry_entry["hf_id"]
        if OmegaConf.is_missing(cfg, "llm.hidden_size") or cfg.llm.get("hidden_size") is None:
            cfg.llm.hidden_size = registry_entry["hidden_size"]
        cfg.llm.hidden_size = registry_entry["hidden_size"]
        cfg.llm.hf_id = registry_entry["hf_id"]
        cfg.llm.local_path = f"models/{llm_name}"

    # Auto-derive projector dimensions from encoder/LLM pairing
    encoder_dim = OmegaConf.select(cfg, "encoder.hidden_size", default=512)
    llm_dim = OmegaConf.select(cfg, "llm.hidden_size", default=2048)
    cfg.projector.input_dim = encoder_dim
    cfg.projector.output_dim = llm_dim

    # Convert to structured config
    schema = OmegaConf.structured(MSPConfig)
    merged = OmegaConf.merge(schema, cfg)
    return OmegaConf.to_object(merged)
