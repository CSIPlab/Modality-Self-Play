"""Qwen3 with audio and vision feature injection.

Audio and vision features (real or bridged proxy) are injected at reserved
special token positions in the prompt.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.models.audio_projector import AudioProjector

from src.models.qwen_tokens import (
    FIM_PAD as AUDIO_PAD_ID,
    IMAGE_PAD as IMAGE_PAD_ID,
    VISION_END as VISION_END_ID,
    VISION_START as VISION_START_ID,
)


class QwenWithAudio(nn.Module):
    """Vanilla Qwen3 with audio + vision proxy injection.

    The LLM is a plain Qwen3 backbone that was not pre-trained for
    multimodal tasks; modality features enter only through the projectors.

    Args:
        qwen_path: Path to Qwen3 model.
        audio_input_dim: CLAP feature dimension (default: 512).
        vision_proxy_dim: MetaCLIP feature dimension (default: 512).
        num_audio_tokens: Audio tokens per sample (default: 1).
        normalize: L2-normalize features before projection (default: True).
    """

    def __init__(
        self,
        qwen_path: str = "models/qwen3-8b",
        audio_input_dim: int = 512,
        vision_proxy_dim: int = 512,
        num_audio_tokens: int = 1,
        normalize: bool = True,
    ):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        for param in self.llm.parameters():
            param.requires_grad = False

        llm_hidden_size = self.llm.config.hidden_size

        self.audio_projector = AudioProjector(
            input_dim=audio_input_dim,
            output_dim=llm_hidden_size,
            num_audio_tokens=num_audio_tokens,
            normalize=normalize,
        )

        self.vision_proxy_projector = AudioProjector(
            input_dim=vision_proxy_dim,
            output_dim=llm_hidden_size,
            num_audio_tokens=1,
            normalize=normalize,
        )

        self.image_pad_id = IMAGE_PAD_ID
        self.audio_pad_id = AUDIO_PAD_ID

    @property
    def llm_hidden_size(self):
        return self.llm.config.hidden_size

    def _is_peft_enabled(self):
        return hasattr(self.llm, "peft_config") and bool(getattr(self.llm, "peft_config", None))

    def configure_training(self, stage, lora_cfg=None):
        for p in self.llm.parameters():
            p.requires_grad = False

        for p in self.audio_projector.parameters():
            p.requires_grad = True
        for p in self.vision_proxy_projector.parameters():
            p.requires_grad = True

        if stage == 2 and lora_cfg and lora_cfg.enabled:
            from peft import LoraConfig, TaskType, get_peft_model
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
            )
            self.llm = get_peft_model(self.llm, peft_config)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Stage {stage}: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    def _make_injection_hook(self, projections_dict):
        def hook(module, args, output):
            for token_id, (projected, positions_list) in projections_dict.items():
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
        audio_features: Optional[torch.FloatTensor] = None,
        vision_proxy_features: Optional[torch.FloatTensor] = None,
    ):
        dtype = self.audio_projector.mlp[1].weight.dtype
        injections = {}

        has_audio = audio_features is not None and (audio_features.abs().sum(dim=-1) > 0).any()
        if has_audio:
            proj_audio = self.audio_projector(audio_features.to(dtype))
            audio_positions = []
            for i in range(input_ids.shape[0]):
                pos = (input_ids[i] == self.audio_pad_id).nonzero(as_tuple=True)[0]
                audio_positions.append(pos.tolist())
            injections[self.audio_pad_id] = (proj_audio, audio_positions)

        has_vision = vision_proxy_features is not None and (vision_proxy_features.abs().sum() > 0).item()
        if has_vision:
            vf = vision_proxy_features.to(dtype)
            if vf.dim() == 3:
                B, N, D = vf.shape
                proj_vision = self.vision_proxy_projector(vf.reshape(B * N, D)).reshape(B, N, -1)
            else:
                proj_vision = self.vision_proxy_projector(vf)
            vision_positions = []
            for i in range(input_ids.shape[0]):
                pos = (input_ids[i] == self.image_pad_id).nonzero(as_tuple=True)[0]
                vision_positions.append(pos.tolist())
            injections[self.image_pad_id] = (proj_vision, vision_positions)

        handle = None
        if injections:
            embed_layer = self.llm.get_input_embeddings()
            handle = embed_layer.register_forward_hook(self._make_injection_hook(injections))

        try:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            if handle is not None:
                handle.remove()

        return outputs

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, audio_features=None,
                 vision_proxy_features=None, **kwargs):
        dtype = self.audio_projector.mlp[1].weight.dtype
        injections = {}

        if audio_features is not None and (audio_features.abs().sum(dim=-1) > 0).any():
            proj_audio = self.audio_projector(audio_features.to(dtype))
            audio_positions = []
            for i in range(input_ids.shape[0]):
                pos = (input_ids[i] == self.audio_pad_id).nonzero(as_tuple=True)[0]
                audio_positions.append(pos.tolist())
            injections[self.audio_pad_id] = (proj_audio, audio_positions)

        if vision_proxy_features is not None and (vision_proxy_features.abs().sum() > 0).item():
            vf = vision_proxy_features.to(dtype)
            if vf.dim() == 3:
                B, N, D = vf.shape
                proj_vision = self.vision_proxy_projector(vf.reshape(B * N, D)).reshape(B, N, -1)
            else:
                proj_vision = self.vision_proxy_projector(vf)
            vision_positions = []
            for i in range(input_ids.shape[0]):
                pos = (input_ids[i] == self.image_pad_id).nonzero(as_tuple=True)[0]
                vision_positions.append(pos.tolist())
            injections[self.image_pad_id] = (proj_vision, vision_positions)

        handle = None
        if injections:
            embed_layer = self.llm.get_input_embeddings()
            handle = embed_layer.register_forward_hook(self._make_injection_hook(injections))

        try:
            result = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()

        return result

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        audio_features: Optional[torch.FloatTensor] = None,
        vision_proxy_features: Optional[torch.FloatTensor] = None,
    ):
        """Return layerwise hidden states for the injected multimodal prompt."""
        dtype = self.audio_projector.mlp[1].weight.dtype
        injections = {}

        if audio_features is not None and (audio_features.abs().sum(dim=-1) > 0).any():
            proj_audio = self.audio_projector(audio_features.to(dtype))
            audio_positions = []
            for i in range(input_ids.shape[0]):
                pos = (input_ids[i] == self.audio_pad_id).nonzero(as_tuple=True)[0]
                audio_positions.append(pos.tolist())
            injections[self.audio_pad_id] = (proj_audio, audio_positions)

        if vision_proxy_features is not None and (vision_proxy_features.abs().sum() > 0).item():
            vf = vision_proxy_features.to(dtype)
            if vf.dim() == 3:
                bsz, n_tok, dim = vf.shape
                proj_vision = self.vision_proxy_projector(vf.reshape(bsz * n_tok, dim)).reshape(bsz, n_tok, -1)
            else:
                proj_vision = self.vision_proxy_projector(vf)
            vision_positions = []
            for i in range(input_ids.shape[0]):
                pos = (input_ids[i] == self.image_pad_id).nonzero(as_tuple=True)[0]
                vision_positions.append(pos.tolist())
            injections[self.image_pad_id] = (proj_vision, vision_positions)

        handle = None
        if injections:
            embed_layer = self.llm.get_input_embeddings()
            handle = embed_layer.register_forward_hook(self._make_injection_hook(injections))

        try:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        finally:
            if handle is not None:
                handle.remove()

        return outputs.hidden_states

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.audio_projector.state_dict(), path / "audio_projector.pt")
        torch.save(self.vision_proxy_projector.state_dict(), path / "vision_proxy_projector.pt")
        if self._is_peft_enabled():
            from peft import get_peft_model_state_dict

            torch.save(get_peft_model_state_dict(self.llm), path / "llm_lora.pt")
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        path = Path(path)
        for name, module in [("audio_projector", self.audio_projector),
                             ("vision_proxy_projector", self.vision_proxy_projector)]:
            pt_path = path / f"{name}.pt"
            if pt_path.exists():
                state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
                module.load_state_dict(state_dict, strict=False)
        lora_path = path / "llm_lora.pt"
        if lora_path.exists():
            if not self._is_peft_enabled():
                raise RuntimeError(
                    f"LoRA weights found at {lora_path}, but the model was not configured "
                    "with stage-2 LoRA before load_checkpoint()."
                )
            from peft import set_peft_model_state_dict

            lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
            adapter_name = next(iter(self.llm.peft_config.keys()))
            set_peft_model_state_dict(self.llm, lora_state, adapter_name=adapter_name)
        print(f"Checkpoint loaded from {path}")
