"""Multi-modality model: Qwen3-8B (text-only) + N modality projectors.

Adds any number of modalities to a frozen text-only LLM via per-modality
projectors that inject features at reserved special token positions.

Each modality M gets:
  - A contrastive encoder E_M (frozen, not part of this model)
  - A projector P_M (trainable MLP: encoder_dim → LLM hidden)
  - A reserved token ID for injection
  - A token placeholder string for prompt construction

The model handles:
  - Stage 1: projector-only training (frozen LLM)
  - Stage 2: projector + LoRA (LLM adapts)
  - Multi-modality: multiple projectors active simultaneously
  - Self-play: proxy features in one slot, real features in another

Usage:
    from src.models.multimodal_model import MultiModalModel

    model = MultiModalModel("models/qwen3-8b")
    model.add_modality("vision", input_dim=512, num_tokens=1)
    model.add_modality("audio", input_dim=512, num_tokens=1)
    model.configure_training(stage=1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.models.audio_projector import AudioProjector

from src.models.qwen_tokens import (  # noqa: F401  (re-exported)
    MODALITY_TOKEN_IDS,
    MODALITY_TOKEN_STRINGS,
)


class MultiModalModel(nn.Module):
    """Qwen3-8B with pluggable modality projectors.

    The LLM is frozen. Each modality gets its own trainable projector that
    maps contrastive encoder features into the LLM's embedding space.
    Features are injected at reserved special token positions via forward hooks.
    """

    def __init__(self, model_path: str = "models/qwen3-8b"):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        self.hidden_size = self.llm.config.hidden_size
        self.model_path = model_path

        # Per-modality projectors (added via add_modality())
        self.projectors = nn.ModuleDict()
        self.modality_token_ids = {}
        self.modality_num_tokens = {}

    def add_modality(
        self,
        name: str,
        input_dim: int = 512,
        num_tokens: int = 1,
        normalize: bool = True,
        token_id: int | None = None,
    ):
        """Register a new modality with its projector.

        Args:
            name: Modality name ("vision" or "audio").
            input_dim: Contrastive encoder embedding dimension.
            num_tokens: Number of LLM tokens per sample for this modality.
            normalize: L2-normalize input features before projection.
            token_id: Override token ID (default: from MODALITY_TOKEN_IDS).
        """
        if name in self.projectors:
            print(f"Warning: overwriting existing projector for {name}")

        self.projectors[name] = AudioProjector(
            input_dim=input_dim,
            output_dim=self.hidden_size,
            num_audio_tokens=1,  # expansion handled by multi-token input
            normalize=normalize,
        )

        if token_id is None:
            if name not in MODALITY_TOKEN_IDS:
                raise ValueError(
                    f"No default token ID for modality '{name}'. "
                    f"Available: {list(MODALITY_TOKEN_IDS.keys())}. "
                    f"Pass token_id explicitly.")
            token_id = MODALITY_TOKEN_IDS[name]

        self.modality_token_ids[name] = token_id
        self.modality_num_tokens[name] = num_tokens

        print(f"Added modality '{name}': dim={input_dim}, tokens={num_tokens}, "
              f"token_id={token_id}, projector params="
              f"{sum(p.numel() for p in self.projectors[name].parameters()):,}")

    @property
    def modalities(self) -> list[str]:
        return list(self.projectors.keys())

    def configure_training(
        self,
        stage: int = 1,
        trainable_modalities: list[str] | None = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_preset: str = "qkvo",
    ):
        """Configure which parameters are trainable.

        Stage 1: projectors only (LLM frozen)
        Stage 2: projectors + LoRA on LLM
        """
        # Freeze everything
        for p in self.llm.parameters():
            p.requires_grad = False
        for p in self.projectors.parameters():
            p.requires_grad = False

        # Unfreeze specified projectors
        if trainable_modalities is None:
            trainable_modalities = self.modalities

        for name in trainable_modalities:
            if name in self.projectors:
                for p in self.projectors[name].parameters():
                    p.requires_grad = True

        # Stage 2: add LoRA
        if stage == 2:
            from peft import LoraConfig, TaskType, get_peft_model

            if lora_target_preset == "qkvo":
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif lora_target_preset == "all_linear":
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
            else:
                target_modules = lora_target_preset.split(",")

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.llm = get_peft_model(self.llm, peft_config)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        modality_str = ", ".join(trainable_modalities)
        print(f"Stage {stage} [{modality_str}]: {trainable:,} trainable / "
              f"{total:,} total ({100 * trainable / total:.2f}%)")

    def _find_token_positions(self, input_ids: torch.LongTensor,
                              token_id: int) -> list[list[int]]:
        """Find positions of a token ID in each batch element."""
        positions = []
        for i in range(input_ids.shape[0]):
            pos = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]
            positions.append(pos.tolist())
        return positions

    def _make_injection_hook(self, injections: dict):
        """Create embedding hook that injects projected features.

        Used ONLY under `torch.no_grad()` (generate path). Under autograd,
        the forward-hook + in-place assignment on the embedding output
        can break the gradient chain when the LLM is fully frozen
        (frozen base + frozen LoRA) — the in-place mutation on a
        non-grad-tracked tensor silently disconnects the projectors
        from the loss. The training path in `forward()` computes
        `inputs_embeds` manually to avoid this.
        """
        def hook(module, args, output):
            for token_id, (projected, positions_list) in injections.items():
                for i in range(output.shape[0]):
                    positions = positions_list[i]
                    for j, pos in enumerate(positions):
                        if 0 <= pos < output.shape[1] and j < projected.shape[1]:
                            output[i, pos] = projected[i, j].to(output.dtype)
            return output
        return hook

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        modality_features: dict[str, torch.Tensor] | None = None,
    ):
        """Forward pass with multi-modality feature injection.

        Builds `inputs_embeds` from `input_ids` and substitutes projected
        modality features at placeholder positions using `torch.where` +
        `gather`, both fully differentiable. Passes `inputs_embeds=` to
        the LLM. Autograd chain: loss → LLM output (frozen or not,
        differentiable either way) → `inputs_embeds` → `projected_bc` →
        `projected` → projector weights.

        Crucially: this works when every LLM parameter is frozen (frozen
        base + frozen LoRA during projector warmup). In-place assignment
        into `inputs_embeds` would not propagate grad in that regime
        because the cloned tensor inherits `requires_grad=False` from the
        frozen embedding.
        """
        if modality_features is None:
            modality_features = {}

        dtype = next(self.llm.parameters()).dtype

        embed_layer = self.llm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # (B, T, H)
        B, T, H = inputs_embeds.shape

        for name, features in modality_features.items():
            if name not in self.projectors:
                continue
            if features is None:
                continue
            # Zero-sum features are not skipped: the projector still has to
            # enter the graph so the loss keeps a grad_fn. Non-anchor
            # modalities with zeros carry an all-false mask, so torch.where
            # ignores them.

            projector = self.projectors[name]
            token_id = self.modality_token_ids[name]

            # Project features. Cast input to projector dtype so the
            # linear op stays in native precision; cast output to LLM
            # dtype for embedding compatibility.
            proj_dtype = next(projector.parameters()).dtype
            f = features.to(proj_dtype)
            if f.dim() == 3:
                # Multi-token: (B, N, D) → project each token
                Bf, N, D = f.shape
                projected = projector(f.reshape(Bf * N, D)).reshape(Bf, N, -1)
            else:
                # Single token: (B, D) → (B, 1, H)
                projected = projector(f)
            projected = projected.to(dtype)  # (B, N, H)
            N = projected.shape[1]

            # Build per-batch mask and "which-of-N" index tensors so we can
            # scatter projected values at placeholder positions in a fully
            # differentiable way.
            positions = self._find_token_positions(input_ids, token_id)
            j_index = torch.zeros(
                B, T, dtype=torch.long, device=inputs_embeds.device)
            mask = torch.zeros(
                B, T, dtype=torch.bool, device=inputs_embeds.device)
            for i in range(B):
                for j, pos in enumerate(positions[i]):
                    if 0 <= pos < T and j < N:
                        j_index[i, pos] = j
                        mask[i, pos] = True

            # Broadcast projected (B, N, H) → (B, T, H) by gathering along
            # dim=1 with j_index. Positions where mask is False still
            # gather index 0 (valid shape) but are masked out by where.
            j_index_exp = j_index.unsqueeze(-1).expand(B, T, H)
            projected_bc = projected.gather(1, j_index_exp)  # (B, T, H)

            inputs_embeds = torch.where(
                mask.unsqueeze(-1), projected_bc, inputs_embeds)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        modality_features: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        """Generate with multi-modality feature injection."""
        if modality_features is None:
            modality_features = {}

        dtype = next(self.llm.parameters()).dtype
        injections = {}

        for name, features in modality_features.items():
            if name not in self.projectors or features is None:
                continue

            projector = self.projectors[name]
            token_id = self.modality_token_ids[name]

            proj_dtype = next(projector.parameters()).dtype
            f = features.to(proj_dtype)
            if f.dim() == 3:
                B, N, D = f.shape
                projected = projector(f.reshape(B * N, D)).reshape(B, N, -1)
            else:
                projected = projector(f)
            projected = projected.to(dtype)

            positions = self._find_token_positions(input_ids, token_id)
            injections[token_id] = (projected, positions)

        handle = None
        if injections:
            embed_layer = self.llm.get_input_embeddings()
            handle = embed_layer.register_forward_hook(
                self._make_injection_hook(injections))

        try:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

    def _is_peft_enabled(self):
        return hasattr(self.llm, "peft_config")

    def save_checkpoint(self, path: str):
        """Save all projectors, LoRA (if PEFT mode) or full LLM weights (full-FT)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for name, projector in self.projectors.items():
            torch.save(projector.state_dict(), path / f"{name}_projector.pt")

        if self._is_peft_enabled():
            from peft import get_peft_model_state_dict
            torch.save(
                get_peft_model_state_dict(self.llm),
                path / "llm_lora.pt")
        else:
            # Full-FT path: save the full LLM state dict. Note this can be
            # ~16 GB (8B bf16) per checkpoint — the
            # rolling `latest/` convention in train_composition.py overwrites
            # to keep disk use bounded.
            torch.save(
                self.llm.state_dict(),
                path / "llm_full.pt")

        # Save modality config
        config = {
            "model_path": self.model_path,
            "modalities": {
                name: {
                    "input_dim": self.projectors[name].mlp[0].normalized_shape[0],
                    "num_tokens": self.modality_num_tokens[name],
                    "token_id": self.modality_token_ids[name],
                }
                for name in self.modalities
            },
        }
        import json
        with open(path / "modality_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Checkpoint saved to {path} "
              f"(modalities: {', '.join(self.modalities)})")

    def load_checkpoint(self, path: str, load_lora: bool = True):
        """Load projectors and optionally LoRA weights."""
        path = Path(path)

        for name in self.modalities:
            pt = path / f"{name}_projector.pt"
            if pt.exists():
                self.projectors[name].load_state_dict(
                    torch.load(pt, map_location="cpu", weights_only=True))
                print(f"Loaded {name} projector from {pt}")

        if load_lora:
            lora_pt = path / "llm_lora.pt"
            full_pt = path / "llm_full.pt"
            if lora_pt.exists():
                if not self._is_peft_enabled():
                    print(f"Warning: LoRA weights at {lora_pt} but model not "
                          f"configured for stage 2")
                else:
                    from peft import set_peft_model_state_dict
                    state = torch.load(
                        lora_pt, map_location="cpu", weights_only=True)
                    set_peft_model_state_dict(self.llm, state)
                    print(f"Loaded LoRA weights from {lora_pt}")
            elif full_pt.exists():
                # Full-FT checkpoint — load full LLM state dict.
                state = torch.load(full_pt, map_location="cpu", weights_only=True)
                missing, unexpected = self.llm.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"Full LLM load: missing={len(missing)} unexpected={len(unexpected)}")
                print(f"Loaded full LLM weights from {full_pt}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_path: str | None = None):
        """Load a model from a saved checkpoint including modality config."""
        import json
        config_path = Path(checkpoint_path) / "modality_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No modality_config.json in {checkpoint_path}")

        with open(config_path) as f:
            config = json.load(f)

        if model_path is None:
            model_path = config["model_path"]

        model = cls(model_path)
        for name, mod_config in config["modalities"].items():
            model.add_modality(
                name,
                input_dim=mod_config["input_dim"],
                num_tokens=mod_config["num_tokens"],
                token_id=mod_config["token_id"],
            )

        model.load_checkpoint(checkpoint_path, load_lora=False)
        return model


def get_modality_placeholder(name: str, num_tokens: int = 1) -> str:
    """Get the prompt placeholder string for a modality."""
    token_str = MODALITY_TOKEN_STRINGS.get(name, f"<|{name}_pad|>")
    return token_str * num_tokens
