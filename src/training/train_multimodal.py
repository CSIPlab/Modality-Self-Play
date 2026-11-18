"""General multimodal training for Phase 1 (unimodal) and Phase 2 (self-play).

Phase 1: Train a single modality projector on real (modality, text) paired data.
Phase 2: Self-play composition — model imagines target modality from source,
  encodes via contrastive text encoder, trains with proxy features.

Usage:
    # Phase 1: train the vision projector
    python -m src.training.train_multimodal \\
        --phase 1 --modality vision \\
        --qa_data <vision-qa.json> --qa_format <format> \\
        --features_path <vision-features.pt>

    # Phase 1: train the audio projector
    python -m src.training.train_multimodal \\
        --phase 1 --modality audio \\
        --qa_data <audio-qa.jsonl> --qa_format <format> \\
        --features_path <audio-features.pt>

    # Phase 2: self-play vision→audio composition
    python -m src.training.train_multimodal \\
        --phase 2 --modality audio \\
        --source_modality vision \\
        --source_checkpoint <phase1-vision-ckpt> \\
        --qa_data <vision-qa.json> --qa_format <format> \\
        --features_path <vision-features.pt>
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.data.multimodal_dataset import (
    EVAL_ONLY_QA_FORMATS,
    MODALITY_DEFAULTS,
    QA_PARSERS,
    MultiModalDataset,
    multimodal_collate_fn,
)
from src.models.multimodal_model import MultiModalModel

logger = logging.getLogger("multimodal")


def setup_logging(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{output_dir}/train.log"),
        ],
    )


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qa_data(path: str, qa_format: str, max_samples: int | None = None):
    """Load QA data from JSON or JSONL."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    logger.info(f"Loaded {len(data)} QA samples from {path} (format: {qa_format})")
    return data


# ── Feature providers ───────────────────────────────────────────────────

class RealFeatureProvider:
    """Pass-through: features already loaded by the dataset."""

    def get_features(self, batch: dict) -> dict[str, torch.Tensor]:
        return batch["modality_features"]


class SelfPlayFeatureProvider:
    """Generates proxy features for Phase 2 via self-play imagination.

    For each sample:
    1. Check self-play cache for target modality description
    2. On miss: use model.generate() to imagine target from source
    3. Encode description via target encoder's text encoder
    4. Apply gap bridging
    5. Return both source (real) and target (proxy) features
    """

    def __init__(
        self,
        model: MultiModalModel,
        tokenizer,
        source_modality: str,
        target_modality: str,
        target_encoder,
        cache,
        bridge=None,
        num_target_tokens: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.encoder = target_encoder
        self.cache = cache
        self.bridge = bridge
        self.num_target_tokens = num_target_tokens

    def get_features(self, batch: dict) -> dict[str, torch.Tensor]:
        """Generate proxy features for the target modality."""
        sample_keys = batch["sample_keys"]
        source_features = batch["modality_features"][self.source_modality]
        device = source_features.device

        # Generate descriptions for each sample
        descriptions = []
        for i, key in enumerate(sample_keys):
            src_feat = source_features[i]
            desc = self.cache.get(key, src_feat.to(device))
            descriptions.append(desc if desc else "Ambient silence.")

        # Encode descriptions through target encoder
        with torch.no_grad():
            if self.num_target_tokens > 1:
                # Split into sentences, encode each
                import re
                all_sents = []
                for desc in descriptions:
                    sents = re.split(r'(?<=[.!?])\s+', desc.strip())
                    sents = [s.strip() for s in sents if s.strip()]
                    while len(sents) < self.num_target_tokens:
                        sents.append(sents[-1] if sents else "")
                    sents = sents[:self.num_target_tokens]
                    all_sents.extend(sents)

                proxy_flat = self.encoder.get_text_features(all_sents)
                proxy_flat = F.normalize(proxy_flat, dim=-1)
                proxy = proxy_flat.reshape(
                    len(descriptions), self.num_target_tokens, -1)
            else:
                proxy = self.encoder.get_text_features(descriptions)
                proxy = F.normalize(proxy, dim=-1).unsqueeze(1)

        # Apply gap bridging
        if self.bridge is not None:
            orig_shape = proxy.shape
            flat = proxy.reshape(-1, proxy.shape[-1])
            flat = self.bridge(flat)
            proxy = flat.reshape(orig_shape)

        # Return both source (real) and target (proxy) features
        features = dict(batch["modality_features"])
        features[self.target_modality] = proxy.to(device)
        return features


# ── Training loop ───────────────────────────────────────────────────────

def train_loop(
    model: MultiModalModel,
    dataloader: DataLoader,
    provider,
    optimizer,
    scheduler,
    args,
    gap_bridge=None,
    deferred_lora=None,
):
    """Shared training loop for Phase 1 and Phase 2."""
    device = next(model.parameters()).device
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs

    logger.info(f"Training: {len(dataloader.dataset)} samples, "
                f"{steps_per_epoch} steps/epoch, {total_steps} total")

    global_step = 0
    running_loss = 0.0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        # Keep LLM in eval if frozen (no dropout in frozen layers)
        if not model._is_peft_enabled():
            model.llm.eval()

        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Move batch features to device
            for name in batch["modality_features"]:
                batch["modality_features"][name] = \
                    batch["modality_features"][name].to(device)

            # Get features (pass-through for Phase 1, generate for Phase 2)
            modality_features = provider.get_features(batch)

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    modality_features=modality_features,
                )

            loss = outputs.loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping NaN/Inf loss at step {step}")
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                trainable = [p for p in model.parameters() if p.requires_grad]
                if gap_bridge is not None:
                    trainable += [p for p in gap_bridge.parameters()
                                  if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                running_loss += outputs.loss.item()

                if deferred_lora and global_step == args.projector_warmup_steps:
                    for p in deferred_lora:
                        p.requires_grad = True
                    model.llm.train()
                    logger.info(
                        f"Step {global_step}: unfreezing "
                        f"{sum(p.numel() for p in deferred_lora):,} LoRA params")

                if global_step % args.log_steps == 0:
                    avg = running_loss / args.log_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    logger.info(
                        f"[{global_step}/{total_steps}] loss={avg:.4f} "
                        f"lr={lr:.2e} elapsed={elapsed:.0f}s")
                    running_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = f"{args.output_dir}/checkpoint-{global_step}"
                    model.save_checkpoint(ckpt)
                    if hasattr(provider, "cache"):
                        provider.cache.save()

        # End of epoch
        logger.info(f"Epoch {epoch + 1}/{args.epochs} complete")
        model.save_checkpoint(f"{args.output_dir}/epoch-{epoch + 1}")
        if hasattr(provider, "cache"):
            provider.cache.save()

    # Final save
    model.save_checkpoint(f"{args.output_dir}/final")
    if hasattr(provider, "cache"):
        provider.cache.save()
    logger.info("Training complete.")


# ── Main ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for Phase 1 and Phase 2 training."""
    parser = argparse.ArgumentParser(
        description="General multimodal training (Phase 1 & Phase 2)")

    # Mode
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="1 = unimodal (real data), 2 = self-play composition")
    parser.add_argument("--modality", type=str, required=True,
                        choices=list(MODALITY_DEFAULTS.keys()),
                        help="Target modality to train")

    # Model
    parser.add_argument("--model_path", default="models/qwen3-8b")
    parser.add_argument("--init_checkpoint", default=None,
                        help="Resume from checkpoint")

    # Data
    parser.add_argument("--qa_data", type=str, required=True,
                        help="Path to QA JSON/JSONL")
    parser.add_argument("--qa_format", type=str, required=True,
                        choices=sorted(set(QA_PARSERS) - EVAL_ONLY_QA_FORMATS),
                        help="Schema of --qa_data. Evaluation benchmarks are "
                             "not accepted here; they are never trained on.")
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to precomputed features .pt dict")
    parser.add_argument("--num_tokens", type=int, default=None,
                        help="Modality tokens (default: from MODALITY_DEFAULTS)")

    # Phase 2 specific
    parser.add_argument("--source_modality", type=str, default=None,
                        help="Phase 2: source modality the model already understands")
    parser.add_argument("--source_checkpoint", type=str, default=None,
                        help="Phase 2: checkpoint with trained source projector")
    parser.add_argument("--source_num_tokens", type=int, default=None)
    parser.add_argument("--selfplay_cache_path", type=str, default=None,
                        help="Phase 2: path for self-play description cache")
    parser.add_argument("--gap_bridge_method", type=str, default="none",
                        help="Gap bridging: none, mean_shift, dropout, co_residual, etc.")
    parser.add_argument("--gap_bridge_calibration", type=str, default=None)
    parser.add_argument("--gap_bridge_text_calibration", type=str, default=None)
    parser.add_argument("--gap_bridge_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="1 = projector only, 2 = projector + LoRA")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=512)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--projector_warmup_steps", type=int, default=2000,
        help="Freeze the LoRA params for the first N optimizer steps so only "
             "the modality projectors are updated; the LoRA params are "
             "unfrozen afterwards. The Phase 1 schedule uses 2000.")
    parser.add_argument("--lora_target_preset", type=str, default="qkvo")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default="")

    return parser


def main():
    args = build_parser().parse_args()

    # Default output dir
    if args.output_dir is None:
        phase_str = "unimodal" if args.phase == 1 else "selfplay"
        args.output_dir = f"outputs/{phase_str}_{args.modality}_stage{args.stage}"
        if args.phase == 2 and args.source_modality:
            args.output_dir += f"_from_{args.source_modality}"
        if args.run_tag:
            args.output_dir += f"_{args.run_tag}"

    # Default tokens
    if args.num_tokens is None:
        args.num_tokens = MODALITY_DEFAULTS.get(
            args.modality, {}).get("tokens", 1)

    # Default self-play cache path
    if args.phase == 2 and args.selfplay_cache_path is None:
        args.selfplay_cache_path = (
            f"data/selfplay/{args.source_modality}_to_{args.modality}_cache.json")

    setup_logging(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Phase {args.phase}: {args.modality} "
                f"({'unimodal' if args.phase == 1 else 'self-play'})")
    logger.info(f"Stage: {args.stage}, Model: {args.model_path}")
    if args.phase == 2:
        logger.info(f"Source: {args.source_modality} "
                    f"(checkpoint: {args.source_checkpoint})")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)

    # Load features
    features = torch.load(args.features_path, weights_only=True)
    logger.info(f"Features: {len(features)} entries")

    # Load QA data
    qa_data = load_qa_data(args.qa_data, args.qa_format, args.max_samples)

    # Build model
    logger.info("Building MultiModalModel...")
    model = MultiModalModel(args.model_path)

    target_dim = MODALITY_DEFAULTS.get(args.modality, {}).get("dim", 512)
    model.add_modality(args.modality, input_dim=target_dim,
                       num_tokens=args.num_tokens)

    if args.phase == 2:
        # Also add source modality (already trained)
        if args.source_modality is None:
            raise ValueError("Phase 2 requires --source_modality")
        source_dim = MODALITY_DEFAULTS.get(
            args.source_modality, {}).get("dim", 512)
        source_tokens = args.source_num_tokens or MODALITY_DEFAULTS.get(
            args.source_modality, {}).get("tokens", 1)
        model.add_modality(args.source_modality, input_dim=source_dim,
                           num_tokens=source_tokens)

    # Configure training
    model.configure_training(
        stage=args.stage,
        trainable_modalities=[args.modality],
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_preset=args.lora_target_preset,
    )

    # Load checkpoints
    if args.init_checkpoint:
        model.load_checkpoint(args.init_checkpoint)
    if args.phase == 2 and args.source_checkpoint:
        model.load_checkpoint(args.source_checkpoint)

    model.to(device)

    # Build dataset
    extra_modalities = None
    if args.phase == 2:
        # Source modality features come from the dataset
        # Target modality features are generated by SelfPlayFeatureProvider
        extra_modalities = {
            args.modality: {
                "features": {},  # empty — filled by provider
                "num_tokens": args.num_tokens,
            }
        }
        # Dataset's primary modality is the source
        dataset = MultiModalDataset(
            samples=qa_data,
            tokenizer=tokenizer,
            modality_name=args.source_modality,
            features=features,
            qa_format=args.qa_format,
            num_modality_tokens=MODALITY_DEFAULTS.get(
                args.source_modality, {}).get("tokens", 1),
            max_seq_length=args.max_seq_length,
            extra_modalities=extra_modalities,
        )
    else:
        dataset = MultiModalDataset(
            samples=qa_data,
            tokenizer=tokenizer,
            modality_name=args.modality,
            features=features,
            qa_format=args.qa_format,
            num_modality_tokens=args.num_tokens,
            max_seq_length=args.max_seq_length,
        )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=multimodal_collate_fn, num_workers=2,
        pin_memory=True, drop_last=True,
    )

    # Build feature provider
    gap_bridge = None
    if args.phase == 1:
        provider = RealFeatureProvider()
    else:
        from src.models.encoders import build_encoder
        from src.selfplay.imagine_modality import SelfPlayCache
        from src.analysis.gap_bridging import build_bridge

        target_encoder = build_encoder(args.modality)
        target_encoder = target_encoder.to(device)

        cache = SelfPlayCache(
            cache_path=args.selfplay_cache_path,
            model=model,
            tokenizer=tokenizer,
            source_modality=args.source_modality,
            target_modality=args.modality,
            num_source_tokens=MODALITY_DEFAULTS.get(
                args.source_modality, {}).get("tokens", 1),
        )

        # Gap bridging
        if args.gap_bridge_method != "none":
            calib_audio, calib_text = None, None
            if args.gap_bridge_calibration:
                calib_audio = torch.stack(list(
                    torch.load(args.gap_bridge_calibration,
                               weights_only=True).values()))
            if args.gap_bridge_text_calibration:
                calib_text = torch.stack(list(
                    torch.load(args.gap_bridge_text_calibration,
                               weights_only=True).values()))
            gap_bridge = build_bridge(
                args.gap_bridge_method,
                audio_feats=calib_audio,
                text_feats=calib_text,
                dropout_prob=args.gap_bridge_dropout,
            ).to(device)
            logger.info(f"Gap bridge: {args.gap_bridge_method}")

        provider = SelfPlayFeatureProvider(
            model=model,
            tokenizer=tokenizer,
            source_modality=args.source_modality,
            target_modality=args.modality,
            target_encoder=target_encoder,
            cache=cache,
            bridge=gap_bridge,
            num_target_tokens=args.num_tokens,
        )

    # Phase 1 schedule: hold the LLM frozen for the first
    # --projector_warmup_steps optimizer steps so only the modality
    # projectors are updated, then unfreeze the LoRA parameters.
    deferred_lora = []
    if args.stage == 2 and args.projector_warmup_steps > 0:
        deferred_lora = [p for n, p in model.llm.named_parameters()
                         if "lora_" in n]
        for p in deferred_lora:
            p.requires_grad = False
        logger.info(
            f"Phase 1 schedule: LLM frozen for the first "
            f"{args.projector_warmup_steps} optimizer steps "
            f"({sum(p.numel() for p in deferred_lora):,} LoRA params deferred)")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += deferred_lora
    if gap_bridge is not None:
        bridge_params = [p for p in gap_bridge.parameters() if p.requires_grad]
        if bridge_params:
            trainable_params += bridge_params
            logger.info(f"Co-training gap bridge: "
                        f"{sum(p.numel() for p in bridge_params):,} params")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.lr_warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    # Train
    train_loop(model, dataloader, provider, optimizer, scheduler, args,
               gap_bridge=gap_bridge, deferred_lora=deferred_lora)


if __name__ == "__main__":
    main()
