"""Phase 2: composition training under the one-real-anchor rule.

Loads the Phase 1 projector checkpoints into a single MultiModalModel,
configures a fresh stage-2 LoRA for the joint model, and trains on a
CompositionDataset where every sample grounds exactly one modality in real
features and injects the other as a proxy feature encoded from the cached
self-play descriptions.

Phase 1 LoRA checkpoints are NOT loaded — they were trained for different
unimodal objectives. The fresh LoRA learns the cross-modal composition
language modeling on top of the (frozen LLM + Phase 1 projectors).

The text-to-modality bridges map proxy features toward the real-feature
distribution before they are projected. They are fit beforehand and stay
frozen here.

Usage:
    python -m src.training.train_composition \\
        --vision_qa <vision-qa.json> --vision_qa_format <format> \\
        --vision_real_features <vision-features.pt> \\
        --audio_qa <audio-qa.jsonl> --audio_qa_format <format> \\
        --audio_real_features <audio-features.pt> \\
        --vision_proxy <vision-proxy.pt> \\
        --audio_proxy <audio-proxy.pt> \\
        --vision_phase1 <phase1-vision-ckpt> \\
        --audio_phase1 <phase1-audio-ckpt> \\
        --frozen_bridge_dir outputs/offline_bridges \\
        --epochs 1 --batch_size 2 --grad_accum 8 \\
        --run_tag composition_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.data.multimodal_dataset import (
    EVAL_ONLY_QA_FORMATS,
    QA_PARSERS,
    MODALITY_DEFAULTS,
    CompositionDataset,
    CompositionSubset,
    composition_collate_fn,
)
from src.models.multimodal_model import MultiModalModel

logger = logging.getLogger("composition")


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


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qa(path: str, max_samples: int | None = None) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p) as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(p) as f:
            data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    return data


def load_features(path: str | None) -> dict[str, torch.Tensor] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"Features file not found: {path}")
        return None
    return torch.load(p, weights_only=True, map_location="cpu")


def bridge_method_name(bridge: torch.nn.Module) -> str:
    if hasattr(bridge, "bridge_method"):
        return str(getattr(bridge, "bridge_method"))
    cls = type(bridge).__name__
    if cls in {"ResidualMLPBridge", "CoTrainedResidualBridge"}:
        return "residual_mlp"
    if cls in {"AffineAlignBridge", "CoTrainedAffineBridge"}:
        return "affine"
    if cls == "LinearRidgeBridge":
        return "ridge"
    if cls == "MeanShiftBridge":
        return "mean_shift"
    if cls == "WhiteningBridge":
        return "whiten"
    if cls == "IdentityBridge":
        return "none"
    return cls.lower()


def _save_all(model, ds_engine, ckpt_dir, bridges, args, optimizer,
              scheduler, global_step, resume_epoch, resume_step_in_epoch,
              running_loss, anchor_counts, feature_coverage, t_start):
    """Unified save path for DS and non-DS.

    Writes:
      - per-modality projectors (.pt)
      - llm_lora.pt OR llm_full.pt (depending on mode)
      - bridges
      - trainer_state (json + optimizer/scheduler state)
      - (DS only) ds_engine.save_checkpoint tag for optimizer-state resume
    """
    # Projectors + LLM weights (the model.save_checkpoint path handles
    # both LoRA and full-FT; for DS ZeRO-2 the full LLM state is
    # replicated, so a direct state_dict dump works).
    model.save_checkpoint(ckpt_dir)
    save_bridges(bridges, ckpt_dir, args.gap_bridge)
    save_trainer_state(
        ckpt_dir, args, optimizer, scheduler, global_step,
        resume_epoch=resume_epoch,
        resume_step_in_epoch=resume_step_in_epoch,
        running_loss=running_loss,
        anchor_counts=anchor_counts,
        feature_coverage=feature_coverage,
        t_start=t_start,
    )
    # DeepSpeed engine state (sharded optimizer + grads) — enables true
    # resume of training dynamics across restarts. Lives alongside our
    # existing trainer_state.pt. DS writes a "ds_engine" subdir.
    if ds_engine is not None:
        from pathlib import Path
        ds_path = Path(ckpt_dir) / "ds_engine"
        try:
            ds_engine.save_checkpoint(str(ds_path), tag="latest")
        except Exception as exc:
            # Don't crash the training run on save failure; log loudly.
            logger.warning("ds_engine.save_checkpoint failed: %s", exc)


def save_bridges(bridges: dict, ckpt_dir: str, method: str):
    """Save bridges + a sidecar config so eval can reconstruct them."""
    from pathlib import Path
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    modal_cfg = {}
    method_set = set()
    for name, b in bridges.items():
        torch.save(b.state_dict(), p / f"{name}_bridge.pt")
        state = b.state_dict()
        input_dim = int(next(iter(state.values())).shape[-1] if state else 0)
        bmethod = bridge_method_name(b)
        method_set.add(bmethod)
        modal_cfg[name] = {
            "input_dim": input_dim,
            "class": type(b).__name__,
            "method": bmethod,
        }
        if hasattr(b, "lam"):
            modal_cfg[name]["lam"] = float(getattr(b, "lam"))
    cfg = {
        "method": (next(iter(method_set)) if len(method_set) == 1 else "mixed"),
        "modalities": modal_cfg,
    }
    with open(p / "bridge_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def capture_rng_state() -> dict:
    import random
    import numpy as np

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None):
    if not state:
        return

    import random
    import numpy as np

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer,
                                   device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_trainer_state(
    ckpt_dir: str,
    args,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    resume_epoch: int,
    resume_step_in_epoch: int,
    running_loss: float,
    anchor_counts: dict[str, int],
    feature_coverage: dict[str, dict[str, int]],
    t_start: float,
):
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    elapsed_sec = max(0.0, time.time() - t_start)
    state = {
        "global_step": global_step,
        "resume_epoch": resume_epoch,
        "resume_step_in_epoch": resume_step_in_epoch,
        "running_loss": running_loss,
        "anchor_counts": anchor_counts,
        "feature_coverage": feature_coverage,
        "elapsed_sec": elapsed_sec,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "args": vars(args),
    }
    torch.save(state, p / "trainer_state.pt")

    summary = {
        "global_step": global_step,
        "resume_epoch": resume_epoch,
        "resume_step_in_epoch": resume_step_in_epoch,
        "elapsed_sec": elapsed_sec,
        "output_dir": args.output_dir,
    }
    with open(p / "trainer_state.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_trainer_state(
    checkpoint_dir: str,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> dict:
    state_path = Path(checkpoint_dir) / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Trainer state not found at {state_path}. "
            "This checkpoint predates resumable training support."
        )

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state.get("rng_state"))
    return state


def load_checkpoint_bridges(bridges: dict, checkpoint_dir: str):
    ckpt_dir = Path(checkpoint_dir)
    for name, bridge in bridges.items():
        bridge_path = ckpt_dir / f"{name}_bridge.pt"
        if not bridge_path.exists():
            continue
        state = torch.load(bridge_path, map_location="cpu", weights_only=True)
        bridge.load_state_dict(state)
        logger.info(f"  Loaded {name} bridge from {bridge_path}")


def load_phase1_projector(model: MultiModalModel, modality: str,
                          checkpoint_dir: str):
    """Load just the modality projector from a Phase 1 checkpoint dir.

    Skips Phase 1 LoRA — Phase 2 trains a fresh LoRA on top of the
    composed model.
    """
    ckpt_dir = Path(checkpoint_dir)
    proj_path = ckpt_dir / f"{modality}_projector.pt"
    if not proj_path.exists():
        raise FileNotFoundError(
            f"Projector not found: {proj_path}\n"
            f"Available files: {list(ckpt_dir.glob('*'))}")
    state = torch.load(proj_path, map_location="cpu", weights_only=True)
    model.projectors[modality].load_state_dict(state)
    logger.info(f"  Loaded {modality} projector from {proj_path}")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for Phase 2 composition training."""
    parser = argparse.ArgumentParser(description="Phase 2 composition training")

    # Model
    parser.add_argument("--model_path", default="models/qwen3-8b")
    # DeepSpeed launcher passes --local_rank automatically; accept it here
    # so argparse doesn't choke. Actual rank info is available via
    # torch.distributed.get_rank() after deepspeed.init_distributed().
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="DeepSpeed launcher arg; do not set manually.")

    # QA data per modality (each subset's anchor data)
    parser.add_argument("--vision_qa", default=None,
                        help="QA file for vision-anchored samples.")
    parser.add_argument("--audio_qa", default=None,
                        help="QA file for audio-anchored samples.")

    # Real features per modality
    parser.add_argument("--vision_real_features", default=None)
    parser.add_argument("--audio_real_features", default=None)

    # Proxy features per modality (for non-anchor injection)
    parser.add_argument("--vision_proxy", default=None,
                        help="Vision proxy features (for audio anchors).")
    parser.add_argument("--audio_proxy", default=None,
                        help="Audio proxy features (for vision anchors).")

    # Phase 1 unimodal checkpoints (projectors only, LoRA discarded)
    parser.add_argument("--vision_phase1", default=None)
    parser.add_argument("--audio_phase1", default=None)

    # Per-modality token counts (must match Phase 1 training)
    parser.add_argument("--vision_tokens", type=int, default=1)
    parser.add_argument(
        "--audio_tokens", type=int, default=1,
        help="Audio tokens injected per sample.")

    # Schema of each QA file; must name a parser in QA_PARSERS.
    parser.add_argument("--vision_qa_format", default=None,
                        help="QA schema of --vision_qa.")
    parser.add_argument("--audio_qa_format", default=None,
                        help="QA schema of --audio_qa.")

    # Gap bridging (co-trained)
    parser.add_argument("--gap_bridge", default="co_residual",
                        help="none, mean_shift, residual, co_residual, ...")
    parser.add_argument(
        "--frozen_bridge_dir", default=None,
        help="Directory with pre-fit {modality}_bridge_fit.pt files. When "
             "set, bridges are loaded from this directory, set to "
             "requires_grad=False, and not co-trained. This is the fix for "
             "the co-trained-bridge-collapses-to-identity issue.")

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Lower than Phase 1 since projectors are pretrained")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_samples_per_subset", type=int, default=None,
                        help="Cap each subset for dry-runs")
    parser.add_argument(
        "--coverage_audit_samples", type=int, default=5000,
        help="How many samples per subset to audit for real-anchor/proxy "
             "coverage before training. 0 disables the audit.")
    parser.add_argument(
        "--require_full_anchor_coverage", action="store_true",
        help="Fail fast if the coverage audit finds missing real anchor "
             "features in any sampled subset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--interleave", default="round_robin",
                        choices=["round_robin", "balanced", "concat", "weighted"])

    # Ablation / comparison flags
    parser.add_argument(
        "--unimodal_interleave", action="store_true",
        help="Unimodal co-training baseline. Each sample carries only its "
             "anchor modality, with no placeholder tokens or zero-feature "
             "fallback for the others.")
    parser.add_argument(
        "--shuffle_proxy_assignments", action="store_true",
        help="Shuffle non-anchor proxy assignments within each modality while "
             "keeping the real anchor, prompt structure, and proxy coverage "
             "fixed. Control for whether matched proxy content drives the "
             "composition gain.")
    parser.add_argument(
        "--paired_mode", action="store_true",
        help="Paired upper bound. Every sample carries real features for "
             "each listed modality, with no anchor/proxy distinction. Needs "
             "matched sample keys across the real-feature dicts.")
    parser.add_argument(
        "--strict_proxy_coverage", action="store_true",
        help="Drop samples at index-build time when the anchor real feature "
             "is missing OR any non-anchor modality's proxy lookup misses. "
             "Prevents silently training 'real anchor + zero proxy' batches "
             "that break the 1-real-N-proxies contract.")
    parser.add_argument(
        "--require_anchor_coverage", action="store_true",
        help="Weaker filter: drop only samples whose anchor real feature is "
             "missing. Proxy gaps are still allowed (non-anchor modality "
             "falls back to zero for those samples). Useful when proxy "
             "coverage is partial but anchor fidelity is non-negotiable.")
    parser.add_argument(
        "--audit_coverage", action="store_true",
        help="Report per-anchor missing-anchor / missing-proxy counts at "
             "index-build time without filtering. No effect on training.")
    parser.add_argument(
        "--mcq_subsets", default=None,
        help="Comma-separated list of MCQ files in 'path:question_modality' "
             "form, e.g. 'data/selfplay/qa_audio_mcq_merged.json.gz:audio,"
             "data/selfplay/qa_vision_mcq_merged.json.gz:vision'. When set, "
             "bypasses the per-modality QA subset builder and uses "
             "MCQCompositionDataset: each sample's anchor modality is "
             "determined by its topic-key prefix, "
             "while the question is about whichever modality the MCQ file "
             "covers. --{vision,audio}_real_features and "
             "--{vision,audio}_proxy are still needed for real + proxy "
             "feature caches.")
    parser.add_argument(
        "--crossmodal_subsets", default=None,
        help="Comma-separated list of cross-modal QA files (.json or "
             ".json.gz) for the joint-reasoning training regime. Schema: "
             "{topic_key: [{q, a, modalities}, ...]}. Each sample's "
             "answer requires both modalities listed. Mutually "
             "exclusive with --mcq_subsets and the per-modality QA "
             "subset path.")
    parser.add_argument(
        "--crossmodal_max_samples", type=int, default=None,
        help="If --crossmodal_subsets is set, cap total entries to this "
             "via stratified subsampling across (anchor, other) pairs.")
    parser.add_argument(
        "--modality_dropout_prob", type=float, default=0.15,
        help="Per-sample probability of zeroing out a non-anchor "
             "modality's feature during training, so that absent "
             "modalities are not treated as malformed input. Anchor "
             "features are never dropped. 0.0 disables.")
    parser.add_argument(
        "--anchor_dropout_prob", type=float, default=0.0,
        help="Per-sample probability of zeroing the ANCHOR feature so the "
             "model must answer from the other (proxy) modality alone. "
             "Composition regularizer: breaks single-modality collapse so "
             "both real modalities are used at eval (both >> single). "
             "0.0 disables.")

    # LoRA (fresh, on top of composed model)
    parser.add_argument("--lora_r", type=int, default=512)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_preset", default="qkvo")
    parser.add_argument("--full_ft", action="store_true",
                        help="Full fine-tune the LLM instead of LoRA. Requires "
                             "DeepSpeed ZeRO-2+ at 8B+ scale. Projectors still "
                             "train jointly; LoRA adapter is NOT applied.")
    parser.add_argument("--deepspeed", default=None,
                        help="Path to DeepSpeed config JSON. Required for "
                             "--full_ft at 8B+. Training loop switches to "
                             "deepspeed.initialize() when set.")
    parser.add_argument(
        "--projector_warmup_steps", type=int, default=0,
        help="If > 0, freeze LoRA params (requires_grad=False) for the first "
             "N optimizer steps so only projectors (and bridges, if any) "
             "train. Intended for fresh-projector runs where LoRA-on-random-"
             "projector is unstable at step 0. LoRA is unfrozen after N steps.")
    parser.add_argument(
        "--init_lora_from", default=None,
        help="Path to a checkpoint directory containing `llm_lora.pt`. If "
             "set, the composition LoRA is initialized from that checkpoint "
             "(e.g. the joint unimodal LoRA from Phase 2) instead of fresh "
             "random. Projector weights at that path are also loaded on top "
             "of Phase-1 projectors. This is the hook for the 3-phase recipe: "
             "Phase 2 trains a joint unimodal LoRA (--unimodal_interleave, "
             "no --init_lora_from), Phase 3 points --init_lora_from at its "
             "output and trains on proxies.")
    parser.add_argument(
        "--resume_from", default=None,
        help="Path to a checkpoint directory containing model weights plus "
             "`trainer_state.pt`. Restores optimizer, scheduler, RNG state, "
             "and loop position for exact-ish continuation after preemption "
             "or timeout.")
    parser.add_argument(
        "--fresh_composition_lora", action="store_true",
        help="Explicitly acknowledge running composition training with a "
             "freshly-initialised LoRA (no warm-start). Required when "
             "--init_lora_from is not set, so the recipe is never "
             "ambiguous in logs. Mutually exclusive with --init_lora_from.")

    # Output
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_tag", default="")

    return parser


def main():
    args = build_parser().parse_args()

    if args.resume_from and args.init_lora_from:
        raise ValueError("--resume_from and --init_lora_from are mutually exclusive")

    for qa, fmt, flag in ((args.vision_qa, args.vision_qa_format, "vision"),
                          (args.audio_qa, args.audio_qa_format, "audio")):
        if qa and not fmt:
            raise ValueError(f"--{flag}_qa requires --{flag}_qa_format")
        if fmt in EVAL_ONLY_QA_FORMATS:
            raise ValueError(
                f"--{flag}_qa_format={fmt} is an evaluation benchmark and is "
                f"never used for training. Choose one of: "
                f"{sorted(set(QA_PARSERS) - EVAL_ONLY_QA_FORMATS)}")

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        args.resume_from = str(resume_path)

    if args.output_dir is None:
        if args.resume_from:
            args.output_dir = str(Path(args.resume_from).parent)
        else:
            args.output_dir = "outputs/composition_phase2"
            if args.run_tag:
                args.output_dir += f"_{args.run_tag}"

    # Auto-resume: if a `latest` checkpoint exists at output_dir and the
    # user didn't explicitly set --resume_from, pick it up automatically.
    # This lets SLURM requeue a crashed job and continue from the last
    # rolling checkpoint without manual intervention.
    if not args.resume_from and not args.init_lora_from:
        auto_latest = Path(args.output_dir) / "latest"
        if (auto_latest / "trainer_state.json").exists():
            args.resume_from = str(auto_latest)
            print(f"Auto-resume: found {auto_latest}/trainer_state.json, "
                  f"resuming from this checkpoint.")

    setup_logging(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recipe clarity: the paper's composition recipe warm-starts from the
    # joint Phase-1 multimodal LoRA. Fresh-LoRA runs are valid (baselines
    # and pre-joint-checkpoint iterations) but should be logged loudly so
    # they cannot be mistaken for the canonical recipe.
    if args.resume_from:
        logger.info(f"Recipe: resuming training state from {args.resume_from}.")
    elif args.unimodal_interleave:
        logger.info("Recipe: joint Phase-1 (fresh projectors + fresh LoRA, "
                    "interleaved unimodal).")
    elif args.init_lora_from:
        logger.info(
            f"Recipe: composition warm-start from {args.init_lora_from} "
            "(paper-canonical).")
    else:
        logger.warning(
            "Recipe: composition with FRESH LoRA (not paper-canonical). "
            "The paper recipe warm-starts from joint Phase-1 — pass "
            "--init_lora_from <joint_phase1_ckpt> once that checkpoint "
            "exists. Continuing with fresh LoRA as a baseline.")

    logger.info("=" * 60)
    logger.info("Phase 2: Multimodal composition training")
    logger.info(f"Output: {args.output_dir}")
    if args.frozen_bridge_dir:
        logger.info(f"Frozen bridge dir: {args.frozen_bridge_dir}")
    else:
        logger.info(f"Gap bridge: {args.gap_bridge}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)

    # ── Build CompositionSubsets ────────────────────────────────────────
    subsets: list[CompositionSubset] = []

    # In MCQ mode the MCQ file carries the questions, so QA paths are
    # optional — only real-feature caches are needed. Stub a placeholder
    # QA list so the CompositionSubset path can still wire real_features
    # into `subsets` for downstream loading.
    mcq_mode = bool(args.mcq_subsets) or bool(args.crossmodal_subsets)

    if (args.vision_qa or mcq_mode) and args.vision_real_features:
        logger.info("Loading vision subset...")
        v_qa = load_qa(args.vision_qa, args.max_samples_per_subset) \
            if args.vision_qa else []
        # Some QA schemas interleave text-only entries that carry no image;
        # drop those so every vision anchor has a real feature to ground on.
        if v_qa and any("image" in d for d in v_qa):
            v_qa = [d for d in v_qa if d.get("image")]
        v_real = load_features(args.vision_real_features)
        logger.info(f"  vision: {len(v_qa)} QA, {len(v_real)} real features")
        subsets.append(CompositionSubset(
            name="vision",
            samples=v_qa,
            qa_format=args.vision_qa_format,
            real_features=v_real,
            num_tokens=args.vision_tokens,
        ))

    if (args.audio_qa or mcq_mode) and args.audio_real_features:
        logger.info("Loading audio subset...")
        a_qa = load_qa(args.audio_qa, args.max_samples_per_subset) \
            if args.audio_qa else []
        a_real = load_features(args.audio_real_features)
        logger.info(f"  audio: {len(a_qa)} QA, {len(a_real)} real features")
        subsets.append(CompositionSubset(
            name="audio",
            samples=a_qa,
            qa_format=args.audio_qa_format,
            real_features=a_real,
            num_tokens=args.audio_tokens,
        ))

    if not subsets:
        raise ValueError(
            "Need at least one real-feature cache (--vision_real_features / "
            "--audio_real_features).")

    active_modalities = [s.name for s in subsets]
    logger.info(f"Active anchors: {active_modalities}")

    # ── Load proxy features (for non-anchor modality injection) ─────────
    proxy_features: dict[str, dict[str, torch.Tensor]] = {}
    unimodal_interleave = args.unimodal_interleave
    if unimodal_interleave:
        logger.info("Unimodal co-training baseline: each prompt carries only "
                    "the anchor modality's placeholder.")
    elif args.paired_mode:
        # Paired upper bound: for every modality we have real features,
        # wire those SAME real features as the non-anchor feed so the
        # dataset's anchor/proxy split becomes (anchor=real, others=real)
        # using sample_key lookup across feature dicts.
        logger.info("--paired_mode set: real features double as paired "
                    "cross-modal supervision.")
        real_bindings = {
            "vision": args.vision_real_features,
            "audio": args.audio_real_features,
        }
        for name, path in real_bindings.items():
            if path:
                proxy_features[name] = load_features(path) or {}
                logger.info(
                    f"  {name} paired real: {len(proxy_features[name])} entries")
    else:
        if args.vision_proxy:
            proxy_features["vision"] = load_features(args.vision_proxy) or {}
            logger.info(
                f"  vision proxy: {len(proxy_features['vision'])} entries")
        if args.audio_proxy:
            proxy_features["audio"] = load_features(args.audio_proxy) or {}
            logger.info(
                f"  audio proxy: {len(proxy_features['audio'])} entries")

        # Topic-based proxies are keyed as `<namespace>:<id>`, but
        # CompositionDataset looks up by bare
        # sample_key. Expand each dict so both the namespaced AND the
        # unnamespaced key hit. The same id across namespaces points to
        # the same underlying sample, so later-seen duplicates are benign.
        for mod, pd in list(proxy_features.items()):
            added = 0
            for k in list(pd.keys()):
                if ":" in k:
                    bare = k.split(":", 1)[1]
                    if bare and bare not in pd:
                        pd[bare] = pd[k]
                        added += 1
            if added:
                logger.info(
                    f"  {mod} proxy: +{added} unnamespaced aliases "
                    f"(now {len(pd)} entries)")

    # ── Build dataset ───────────────────────────────────────────────────
    if args.crossmodal_subsets:
        from src.data.crossmodal_composition_dataset import CrossModalCompositionDataset
        logger.info("Cross-modal mode: building CrossModalCompositionDataset")
        qa_paths = [p.strip() for p in args.crossmodal_subsets.split(",")
                    if p.strip()]
        for p in qa_paths:
            logger.info(f"  cross-modal QA: {p}")
        real_by_mod = {s.name: s.real_features for s in subsets}
        num_tokens_by_mod = {s.name: s.num_tokens for s in subsets}
        all_mods = sorted(set(real_by_mod) | set(proxy_features))
        dataset = CrossModalCompositionDataset(
            qa_paths=qa_paths,
            real_features=real_by_mod,
            proxy_features=proxy_features,
            tokenizer=tokenizer,
            active_modalities=all_mods,
            num_tokens={
                m: num_tokens_by_mod.get(
                    m, MODALITY_DEFAULTS.get(m, {}).get("tokens", 1))
                for m in all_mods
            },
            max_seq_length=args.max_seq_length,
            strict_coverage=args.strict_proxy_coverage,
            max_samples=args.crossmodal_max_samples,
            seed=args.seed,
        )
        logger.info(f"Cross-modal composition dataset: {len(dataset)} samples")
        logger.info(f"Active modalities: {all_mods}")
    elif args.mcq_subsets:
        # MCQ-corpus path: parse --mcq_subsets spec, extract the real-
        # feature caches we already stashed in CompositionSubset
        # instances (so we don't re-load them), and build an
        # MCQCompositionDataset where each sample's anchor comes from
        # its topic-key prefix rather than from a per-modality QA
        # subset. Questions can be about any modality.
        from src.data.mcq_composition_dataset import MCQCompositionDataset
        logger.info("MCQ-corpus mode: building MCQCompositionDataset")
        mcq_pairs: list[tuple[str, str]] = []
        for spec in args.mcq_subsets.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if ":" not in spec:
                raise ValueError(
                    f"--mcq_subsets entry must be 'path:question_modality', "
                    f"got {spec!r}")
            path, qmod = spec.rsplit(":", 1)
            mcq_pairs.append((path.strip(), qmod.strip()))
            logger.info(f"  {path.strip()}  → question_modality={qmod.strip()}")

        real_by_mod = {s.name: s.real_features for s in subsets}
        num_tokens_by_mod = {s.name: s.num_tokens for s in subsets}
        all_mods = sorted(set(real_by_mod) | set(proxy_features))
        dataset = MCQCompositionDataset(
            mcq_subsets=mcq_pairs,
            real_features=real_by_mod,
            proxy_features=proxy_features,
            tokenizer=tokenizer,
            active_modalities=all_mods,
            num_tokens={
                m: num_tokens_by_mod.get(
                    m, MODALITY_DEFAULTS.get(m, {}).get("tokens", 1))
                for m in all_mods
            },
            max_seq_length=args.max_seq_length,
            strict_coverage=args.strict_proxy_coverage,
        )
        logger.info(f"MCQ composition dataset: {len(dataset)} samples")
        logger.info(f"Active modalities: {all_mods}")
    else:
        dataset = CompositionDataset(
            subsets=subsets,
            tokenizer=tokenizer,
            proxy_features=proxy_features,
            max_seq_length=args.max_seq_length,
            interleave=args.interleave,
            unimodal_only=unimodal_interleave,
            shuffled_proxies=args.shuffle_proxy_assignments,
            shuffle_seed=args.seed,
            strict_proxy_coverage=args.strict_proxy_coverage,
            require_anchor_coverage=args.require_anchor_coverage,
            audit_coverage=args.audit_coverage,
        )
        logger.info(f"Composition dataset: {len(dataset)} samples")
        logger.info(f"All modalities: {dataset.all_modalities}")
    logger.info(f"Per-modality num_tokens: {dataset.num_tokens}")

    if hasattr(dataset, "audit_one_real_anchor"):
        violations = dataset.audit_one_real_anchor()
        if violations:
            shown = list(violations.items())[:5]
            detail = "; ".join(f"{k} in {v}" for k, v in shown)
            msg = (f"One-real-anchor rule violated: {len(violations)} sample(s) "
                   f"appear as the real anchor in more than one subset "
                   f"({detail}{', ...' if len(violations) > 5 else ''})")
            if args.require_full_anchor_coverage:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info("One-real-anchor rule: OK (no sample anchors twice)")

    if args.coverage_audit_samples != 0 and hasattr(dataset, "audit_feature_coverage"):
        audit_n = None if args.coverage_audit_samples < 0 else args.coverage_audit_samples
        coverage = dataset.audit_feature_coverage(max_per_subset=audit_n)
        for anchor, stats in coverage.items():
            sampled = stats["sampled"]
            logger.info(
                f"Coverage audit [{anchor}] sampled={sampled} "
                f"anchor={stats['anchor_present']}/{sampled} "
                f"({100.0 * stats['anchor_coverage']:.1f}%)")
            for mod, cov in stats["proxy_coverage"].items():
                logger.info(
                    f"Coverage audit [{anchor}->{mod}] "
                    f"{stats['proxy_present'][mod]}/{sampled} "
                    f"({100.0 * cov:.1f}%)")
            if stats["anchor_missing"] > 0:
                msg = (
                    f"Coverage audit found {stats['anchor_missing']} missing real "
                    f"anchor features for subset '{anchor}' out of {sampled} "
                    f"sampled examples. MSP would silently fall back to zeros "
                    f"for those anchors."
                )
                if args.require_full_anchor_coverage:
                    raise ValueError(msg)
                logger.warning(msg)

    shuffle_batches = args.interleave == "weighted"
    if args.interleave == "round_robin":
        logger.info(
            "Preserving round-robin anchor order with DataLoader shuffle=False")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle_batches,
        collate_fn=composition_collate_fn, num_workers=2,
        pin_memory=True, drop_last=True,
    )

    # ── Build model ─────────────────────────────────────────────────────
    logger.info("Building MultiModalModel...")
    model = MultiModalModel(args.model_path)

    for s in subsets:
        dim = MODALITY_DEFAULTS.get(s.name, {}).get("dim", 512)
        model.add_modality(s.name, input_dim=dim, num_tokens=s.num_tokens)

    # Load Phase 1 projector weights (NOT LoRAs)
    logger.info("Loading Phase 1 projector checkpoints...")
    phase1_paths = {
        "vision": args.vision_phase1,
        "audio": args.audio_phase1,
    }
    for name in active_modalities:
        path = phase1_paths.get(name)
        if path:
            load_phase1_projector(model, name, path)
        else:
            logger.warning(f"No Phase 1 checkpoint for {name} — using random init")

    # ── Configure training: full fine-tune or a fresh stage-2 LoRA ──────
    if args.full_ft:
        logger.info("FULL FINE-TUNE mode: unfreezing LLM + projectors. "
                    "Requires DeepSpeed ZeRO-2+ at 8B+ scale.")
        # Unfreeze everything. Projectors + LLM all trainable.
        for p in model.llm.parameters():
            p.requires_grad = True
        for name in active_modalities:
            if name in model.projectors:
                for p in model.projectors[name].parameters():
                    p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Stage FULL-FT: {trainable:,} trainable / "
                    f"{total:,} total ({100 * trainable / total:.2f}%)")
    else:
        model.configure_training(
            stage=2,
            trainable_modalities=active_modalities,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_preset=args.lora_target_preset,
        )
    # Optional warm-start: initialise LoRA (and re-load projectors) from a
    # previous checkpoint such as the joint unimodal LoRA from Phase 2.
    if args.resume_from:
        logger.info(
            f"Restoring model/projectors/LoRA from {args.resume_from}")
        model.load_checkpoint(args.resume_from, load_lora=True)
    elif args.init_lora_from:
        logger.info(
            f"Warm-starting LoRA + projectors from {args.init_lora_from}")
        model.load_checkpoint(args.init_lora_from, load_lora=True)
    model.to(device)

    # ── Build bridges: either frozen offline-fit, or co-trained ─────────
    bridges: dict[str, torch.nn.Module] = {}
    if args.frozen_bridge_dir:
        from src.analysis.gap_bridging import build_bridge
        frozen_dir = Path(args.frozen_bridge_dir)
        logger.info(f"Loading FROZEN bridges from {frozen_dir}")
        llm_dtype = next(model.llm.parameters()).dtype
        cfg = {}
        cfg_path = frozen_dir / "bridge_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
        mod_cfgs = cfg.get("modalities", {}) if isinstance(cfg, dict) else {}
        for name in active_modalities:
            pt = frozen_dir / f"{name}_bridge_fit.pt"
            if not pt.exists():
                logger.warning(
                    f"No frozen bridge for {name} at {pt} — skipping "
                    f"(proxy features for {name} will go through identity).")
                continue
            mod_dim = MODALITY_DEFAULTS.get(name, {}).get("dim", 512)
            frozen_state = torch.load(pt, map_location="cpu", weights_only=False)
            mod_cfg = mod_cfgs.get(name, {})
            bridge_method = (
                mod_cfg.get("method")
                or cfg.get("method")
                or args.gap_bridge
                or "residual_mlp"
            )
            build_kwargs = {"input_dim": mod_dim}
            if bridge_method == "residual_mlp":
                first_w = next((v for v in frozen_state.values()
                                if isinstance(v, torch.Tensor) and v.dim() == 2),
                               None)
                build_kwargs["residual_hidden"] = int(first_w.shape[0]) if first_w is not None else 256
            if bridge_method == "ridge" and "lam" in mod_cfg:
                build_kwargs["ridge_lam"] = float(mod_cfg["lam"])
            bridge = build_bridge(bridge_method, **build_kwargs)
            bridge.load_state_dict(frozen_state, strict=False)
            bridge = bridge.to(device).to(llm_dtype).eval()
            for p in bridge.parameters():
                p.requires_grad = False
            bridges[name] = bridge
            logger.info(f"  {name}: loaded + frozen ({bridge_method})")
    elif args.gap_bridge != "none":
        from src.analysis.gap_bridging import build_bridge
        for name in active_modalities:
            mod_dim = MODALITY_DEFAULTS.get(name, {}).get("dim", 512)
            bridge = build_bridge(args.gap_bridge, input_dim=mod_dim)
            bridge = bridge.to(device)
            try:
                bridge = bridge.to(next(model.llm.parameters()).dtype)
            except StopIteration:
                pass
            bridges[name] = bridge
        logger.info(f"Bridges ({args.gap_bridge}, co-trained): {list(bridges.keys())}")

    if args.resume_from:
        load_checkpoint_bridges(bridges, args.resume_from)

    # ── Optimizer ────────────────────────────────────────────────────────
    # Record LoRA params up-front so we can freeze/unfreeze them for the
    # projector warmup phase without rebuilding the optimizer.
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    ]
    if args.projector_warmup_steps > 0:
        for p in lora_params:
            p.requires_grad = False
        logger.info(
            f"Projector warmup: LoRA frozen for first "
            f"{args.projector_warmup_steps} optimizer steps "
            f"({len(lora_params)} LoRA tensors)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    for b in bridges.values():
        trainable += [p for p in b.parameters() if p.requires_grad]
    # Include LoRA in optimizer even when currently frozen — AdamW skips
    # params whose .grad is None, so these contribute nothing until unfrozen.
    if args.projector_warmup_steps > 0:
        trainable += lora_params

    n_trainable = sum(p.numel() for p in trainable if p.requires_grad)
    logger.info(f"Trainable params (excl. frozen LoRA): {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, len(dataloader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.lr_warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    # ── DeepSpeed initialize (if requested) ─────────────────────────────
    ds_engine = None
    if args.deepspeed:
        import deepspeed
        import json as _json
        logger.info(f"Initialising DeepSpeed with config {args.deepspeed}")
        with open(args.deepspeed) as _f:
            _ds_cfg = _json.load(_f)
        # If the DS config defines its own optimizer/scheduler (required
        # when ZeRO-Offload is on), let DS construct them. Passing a
        # client-side AdamW alongside CPU-offload triggers a hard error.
        _ds_init_kwargs = dict(model=model, config=args.deepspeed,
                               model_parameters=trainable)
        if "optimizer" not in _ds_cfg:
            _ds_init_kwargs["optimizer"] = optimizer
        if "scheduler" not in _ds_cfg:
            _ds_init_kwargs["lr_scheduler"] = scheduler
        ds_engine, optimizer, _, scheduler = deepspeed.initialize(
            **_ds_init_kwargs
        )
        # After DeepSpeed wraps, model forward should go through ds_engine.
        # We keep `model` as the underlying MultiModalModel for custom
        # forward logic; backward + step go through ds_engine below.
        logger.info(f"DeepSpeed engine ready (ZeRO stage "
                    f"{ds_engine.zero_optimization_stage()})")

    logger.info(f"Training: {len(dataset)} samples, {steps_per_epoch} steps/epoch, "
                f"{total_steps} total")

    # ── Training loop ───────────────────────────────────────────────────
    global_step = 0
    running_loss = 0.0
    t_start = time.time()
    resume_epoch = 0
    resume_step_in_epoch = 0
    anchor_counts = {name: 0 for name in active_modalities}
    feature_coverage = {
        name: {
            "present": 0,
            "missing": 0,
            "anchor_missing": 0,
            "proxy_missing": 0,
        }
        for name in active_modalities
    }

    if args.resume_from:
        resume_state = load_trainer_state(
            args.resume_from, optimizer, scheduler, device)
        global_step = int(resume_state.get("global_step", 0))
        running_loss = float(resume_state.get("running_loss", 0.0))
        resume_epoch = int(resume_state.get("resume_epoch", 0))
        resume_step_in_epoch = int(
            resume_state.get("resume_step_in_epoch", 0))
        t_start = time.time() - float(resume_state.get("elapsed_sec", 0.0))

        for name, count in resume_state.get("anchor_counts", {}).items():
            if name in anchor_counts:
                anchor_counts[name] = int(count)

        for name, vals in resume_state.get("feature_coverage", {}).items():
            if name not in feature_coverage:
                continue
            for key in feature_coverage[name]:
                if key in vals:
                    feature_coverage[name][key] = int(vals[key])

        if (args.projector_warmup_steps > 0
                and global_step >= args.projector_warmup_steps):
            for p in lora_params:
                p.requires_grad = True

        logger.info(
            f"Resumed trainer state from {args.resume_from}: "
            f"global_step={global_step}, resume_epoch={resume_epoch}, "
            f"resume_step_in_epoch={resume_step_in_epoch}")

    for epoch in range(resume_epoch, args.epochs):
        model.train()
        if not model._is_peft_enabled():
            model.llm.eval()

        if epoch == resume_epoch and resume_step_in_epoch > 0:
            logger.info(
                f"Fast-forwarding dataloader to batch {resume_step_in_epoch} "
                f"within epoch {epoch + 1}/{args.epochs}")

        if ds_engine is None:
            optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            if epoch == resume_epoch and step < resume_step_in_epoch:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            anchors = batch["anchors"]
            for a in anchors:
                if a in anchor_counts:
                    anchor_counts[a] += 1

            batch_presence = batch.get("feature_presence", {})
            for name, present_mask in batch_presence.items():
                if name not in feature_coverage:
                    continue
                feature_coverage[name]["present"] += int(present_mask.sum().item())
                feature_coverage[name]["missing"] += int((~present_mask).sum().item())
                for idx, anchor_name in enumerate(anchors):
                    if present_mask[idx]:
                        continue
                    if anchor_name == name:
                        feature_coverage[name]["anchor_missing"] += 1
                    else:
                        feature_coverage[name]["proxy_missing"] += 1

            # Move features to device, apply bridges to proxy entries only
            mod_features = {}
            for name in batch["modality_features"]:
                feat = batch["modality_features"][name].to(device)

                if name in bridges:
                    # Per-sample mask: True where this sample's anchor != name
                    is_proxy = torch.tensor(
                        [a != name for a in anchors],
                        dtype=torch.bool, device=feat.device,
                    )
                    if is_proxy.any():
                        proxy_feat = feat[is_proxy]
                        orig_shape = proxy_feat.shape
                        flat = proxy_feat.reshape(-1, proxy_feat.shape[-1])
                        # Match bridge dtype
                        bdtype = next(bridges[name].parameters()).dtype
                        flat = flat.to(bdtype)
                        bridged = bridges[name](flat).reshape(orig_shape)
                        bridged = bridged.to(feat.dtype)
                        new_feat = feat.clone()
                        new_feat[is_proxy] = bridged
                        feat = new_feat

                mod_features[name] = feat

            # Modality dropout: with probability p, per-sample, zero out
            # a non-anchor modality's feature so the composition LoRA
            # also sees batches with "missing" modalities. The
            # placeholder stays in the prompt; the projector receives
            # zeros and outputs its bias — the same pattern a
            # deployment call sees when a modality is absent. Anchor
            # features are never dropped.
            if args.modality_dropout_prob > 0 and len(anchors) > 0:
                for name in list(mod_features.keys()):
                    feat = mod_features[name]
                    for i, anchor_i in enumerate(anchors):
                        if anchor_i == name:
                            continue  # never drop the anchor
                        if torch.rand(1).item() < args.modality_dropout_prob:
                            feat[i] = torch.zeros_like(feat[i])
                    mod_features[name] = feat

            # Anchor dropout (composition regularizer): with prob p, zero the
            # ANCHOR feature so the model must answer from the OTHER (proxy)
            # modality alone. Breaks single-modality collapse so that at eval
            # the model genuinely fuses both real modalities (both >> single).
            # Only drops when >=1 other modality slot exists for that sample.
            if getattr(args, "anchor_dropout_prob", 0.0) > 0 and len(anchors) > 0:
                names = list(mod_features.keys())
                for i, anchor_i in enumerate(anchors):
                    if anchor_i not in mod_features:
                        continue
                    if not [n for n in names if n != anchor_i]:
                        continue
                    if torch.rand(1).item() < args.anchor_dropout_prob:
                        mod_features[anchor_i][i] = torch.zeros_like(
                            mod_features[anchor_i][i])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    modality_features=mod_features,
                )

            loss = outputs.loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping NaN/Inf loss at step {step}")
                if ds_engine is None:
                    optimizer.zero_grad()
                continue

            if ds_engine is not None:
                # DeepSpeed handles backward/step/grad-accum/clip internally
                # via its config. grad_accum_steps comes from the DS JSON
                # (should match args.grad_accum); DS updates params when
                # internal accumulator hits that count.
                ds_engine.backward(loss)
                ds_engine.step()
                # DS counts one "step" per micro-batch; advance global_step
                # only on full grad_accum boundary to match the non-DS path.
                if (step + 1) % args.grad_accum == 0:
                    global_step += 1
                    running_loss += outputs.loss.item()
            else:
                loss.backward()
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    running_loss += outputs.loss.item()
            if (step + 1) % args.grad_accum == 0:
                # Unfreeze LoRA exactly at the end of the warmup window.
                if (args.projector_warmup_steps > 0
                        and global_step == args.projector_warmup_steps):
                    for p in lora_params:
                        p.requires_grad = True
                    n_now = sum(p.numel() for p in trainable if p.requires_grad)
                    logger.info(
                        f"[{global_step}/{total_steps}] Projector warmup "
                        f"complete — LoRA unfrozen. "
                        f"Trainable params now: {n_now:,}")

                if global_step % args.log_steps == 0:
                    avg = running_loss / args.log_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    anchor_str = " ".join(
                        f"{n}={anchor_counts[n]}" for n in active_modalities)
                    coverage_str = " ".join(
                        f"{name}=cov:{(100.0 * vals['present'] / max(vals['present'] + vals['missing'], 1)):.1f}%"
                        f" real_miss:{vals['anchor_missing']}"
                        f" proxy_miss:{vals['proxy_missing']}"
                        for name, vals in feature_coverage.items())
                    logger.info(
                        f"[{global_step}/{total_steps}] loss={avg:.4f} "
                        f"lr={lr:.2e} elapsed={elapsed:.0f}s | anchors: {anchor_str} "
                        f"| coverage: {coverage_str}")
                    running_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Rolling latest checkpoint — overwrites the previous
                    # copy every save_steps. Prevents unbounded disk growth
                    # on long training runs (can be TBs for full-FT).
                    ckpt_dir = f"{args.output_dir}/latest"
                    _save_all(model, ds_engine, ckpt_dir, bridges, args,
                              optimizer, scheduler, global_step, epoch,
                              step + 1, running_loss, anchor_counts,
                              feature_coverage, t_start)

        resume_step_in_epoch = 0
        logger.info(f"Epoch {epoch + 1}/{args.epochs} complete")
        ep_dir = f"{args.output_dir}/epoch-{epoch + 1}"
        _save_all(model, ds_engine, ep_dir, bridges, args, optimizer,
                  scheduler, global_step, epoch + 1, 0, running_loss,
                  anchor_counts, feature_coverage, t_start)

    final_dir = f"{args.output_dir}/final"
    _save_all(model, ds_engine, final_dir, bridges, args, optimizer,
              scheduler, global_step, args.epochs, 0, running_loss,
              anchor_counts, feature_coverage, t_start)
    logger.info(f"Final anchor distribution: {anchor_counts}")
    logger.info(f"Final feature coverage: {feature_coverage}")
    logger.info("Composition training complete.")


if __name__ == "__main__":
    main()
