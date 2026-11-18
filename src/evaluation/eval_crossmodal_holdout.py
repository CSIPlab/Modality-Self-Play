"""Evaluate a checkpoint on the held-out cross-modal val set.

Pre-registered claim + scope documented in
paper/CROSSMODAL_EVAL_PREREGISTRATION.md. This is a DIAGNOSTIC of
training effectiveness on held-out samples from the same generation
distribution as training — NOT a benchmark of cross-pipeline
generalization.

Per entry: topic_key prefix determines the anchor modality (which gets
its real feature). The other modalities listed in `modalities` get
their topic-pool proxy features, bridged into the modality's feature
space via the frozen bridges. Prompt places all 3 modality
placeholders in fixed order (anchor-invariant prompt), then the
question; freeform answer is generated and compared via lenient-match.

Usage:
    python -m src.evaluation.eval_crossmodal_holdout \\
        --checkpoint outputs/composition_phase2_msp_xmodal_n50000_s42/final \\
        --val data/selfplay/qa_crossmodal_val_holdout.json.gz \\
        --real_vision data/features/image_features_665k_1token.pt \\
        --real_audio data/audio/merged_clap_1token_v2.pt \\
        --proxy_audio data/selfplay/proxies/topic_audio_proxy.pt \\
        --proxy_vision data/selfplay/proxies/topic_vision_proxy.pt \\
        --proxy_pc data/selfplay/proxies/topic_pc_proxy.pt \\
        --override_bridge_dir outputs/default_bridges \\
        --output_json results/crossmodal_val_msp_xmodal_n50k_s42.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch
from src.evaluation.metrics_multimodal import _lenient_match

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_path", default="models/qwen3-8b")
    ap.add_argument("--val", required=True)
    ap.add_argument("--real_vision", required=True)
    ap.add_argument("--real_audio", required=True)
    ap.add_argument("--proxy_audio", required=True)
    ap.add_argument("--proxy_vision", required=True)
    ap.add_argument("--override_bridge_dir", default="outputs/default_bridges")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from src.models.multimodal_model import MultiModalModel, get_modality_placeholder
    from src.evaluation.eval_multimodal import load_bridges_from_checkpoint
    from src.data.mcq_composition_dataset import PREFIX_TO_ANCHOR, _strip_prefix, _resolve_proxy_key, _resolve_real_key

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)

    print(f"Loading features...")
    real_features = {
        "vision": torch.load(args.real_vision, weights_only=False),
        "audio": torch.load(args.real_audio, weights_only=False),
    }
    proxy_features = {
        "vision": torch.load(args.proxy_vision, weights_only=False),
        "audio": torch.load(args.proxy_audio, weights_only=False),
    }
    for m in real_features:
        print(f"  {m}: real={len(real_features[m])} proxy={len(proxy_features[m])}")

    print(f"Loading val set...")
    opener = gzip.open if args.val.endswith(".gz") else open
    with opener(args.val, "rt") as f:
        val = json.load(f)
    all_items: list[dict] = []
    for tk, entries in val.items():
        prefix, bare = _strip_prefix(tk)
        anchor = PREFIX_TO_ANCHOR.get(prefix)
        if anchor is None:
            continue
        real_key = _resolve_real_key(bare, anchor, real_features[anchor])
        if real_key is None:
            continue
        proxy_keys = {}
        for m in ["vision", "audio"]:
            if m == anchor:
                continue
            pk = _resolve_proxy_key(bare, prefix, proxy_features[m])
            if pk is None:
                pk = None
            proxy_keys[m] = pk
        for e in entries:
            all_items.append({
                "topic_key": tk, "bare": bare, "anchor": anchor,
                "real_key": real_key, "proxy_keys": proxy_keys,
                "primary_other": sorted([m for m in e.get("modalities", [])
                                          if m != anchor])[0] if [m for m in e.get("modalities", []) if m != anchor] else None,
                "q": e.get("q", ""), "a": e.get("a", ""),
                "modalities": e.get("modalities", []),
            })
    if args.max_samples and args.max_samples < len(all_items):
        all_items = all_items[:args.max_samples]
    print(f"  {len(all_items)} val items after feature resolution")

    print(f"Loading model + bridges from {args.checkpoint}...")
    model = MultiModalModel.from_checkpoint(args.checkpoint, args.model_path)
    model.configure_training(stage=2, trainable_modalities=[])
    model.load_checkpoint(args.checkpoint)
    model.to(device).eval()

    bridge_source = args.override_bridge_dir or args.checkpoint
    bridges = load_bridges_from_checkpoint(
        bridge_source, ["vision", "audio"], device)
    print(f"  Bridges loaded: {list(bridges.keys())}")

    all_mods = sorted(["vision", "audio"])
    num_tokens = {"vision": 1, "audio": 1}
    feature_dims = {"vision": 512, "audio": 512}
    placeholders = {m: get_modality_placeholder(m, num_tokens[m]) for m in all_mods}

    def _coerce(feat: torch.Tensor, m: str) -> torch.Tensor:
        if feat.dim() == 2 and feat.shape[0] > 1:
            feat = feat.mean(dim=0)
        elif feat.dim() == 2 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        return feat.float()

    records = []
    correct = 0
    per_pair = defaultdict(lambda: {"n": 0, "c": 0})

    print(f"Running eval...")
    for i, item in enumerate(all_items):
        anchor = item["anchor"]
        mf = {}
        # Real anchor
        rf = real_features[anchor].get(item["real_key"])
        mf[anchor] = _coerce(rf, anchor)
        # Proxies for others (bridge them)
        for m in all_mods:
            if m == anchor:
                continue
            pk = item["proxy_keys"].get(m)
            if pk is not None and pk in proxy_features[m]:
                pf = _coerce(proxy_features[m][pk], m)
                if m in bridges:
                    with torch.no_grad():
                        pf = bridges[m](pf.unsqueeze(0).to(device)).squeeze(0).cpu()
                mf[m] = pf
            else:
                mf[m] = torch.zeros(feature_dims[m])

        # Build prompt
        prefix = " ".join(placeholders[m] for m in all_mods)
        user = f"{prefix}\n{item['q']}"
        prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(device)

        # Feature dict to shape [1, T, D]
        batched = {m: v.unsqueeze(0) for m, v in mf.items()}
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                modality_features=batched,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        pred = gen.split("<|im_end|>")[0].strip()

        ok = _lenient_match(pred, item["a"])
        if ok:
            correct += 1
        pair = (anchor, item["primary_other"])
        per_pair[pair]["n"] += 1
        if ok:
            per_pair[pair]["c"] += 1

        records.append({
            "topic_key": item["topic_key"], "anchor": anchor,
            "primary_other": item["primary_other"],
            "q": item["q"], "a": item["a"],
            "prediction": pred, "correct": ok,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_items)} acc={correct/(i+1)*100:.2f}%")

    overall = correct / max(len(all_items), 1)
    print(f"\nOverall: {overall*100:.2f}% ({correct}/{len(all_items)})")
    print("Per-pair:")
    per_pair_out = {}
    for (a, o), s in sorted(per_pair.items()):
        acc = s["c"] / max(s["n"], 1)
        print(f"  ({a}, {o}): n={s['n']} acc={acc*100:.2f}%")
        per_pair_out[f"{a}__{o}"] = {"n": s["n"], "correct": s["c"], "accuracy": acc}

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "val_file": args.val,
        "bridge_source": bridge_source,
        "metrics": {"accuracy": overall, "correct": correct, "total": len(all_items)},
        "per_pair": per_pair_out,
        "records": records,
    }, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
