"""Evaluate audio-visual composition on MUSIC-AVQA.

Modes:
  1. text_only: no features (lower bound)
  2. audio_only: real CLAP audio, no vision
  3. vision_only: real vision features when available, else proxy fallback
  4. proxy_v_real_a: proxy vision + real audio (training distribution)
  5. real_v_real_a: real vision + real audio (composition test)

This evaluator now reports:
- overall accuracy
- S1/S2/S3/S4 accuracy from the frozen modality split
- interaction metrics G and I_va
- per-question-type accuracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.avqa_prompts import (
    METADATA_DIR,
    TEXT_CONTROL_MODES,
    build_text_shuffle_map,
    build_user_content,
    get_modality_prompt_blocks,
    get_text_control_payload,
    load_image_captions,
    resolve_templates,
)
from src.evaluation.metrics_multimodal import (
    SUBSET_ORDER,
    attach_split_metadata,
    compute_interaction_metrics,
    format_pct,
    load_split_index,
    summarize_records,
)

# MUSIC-AVQA answer vocabulary
ANSWER_VOCAB = {
    "yes", "no",
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "left", "right", "middle",
    "indoor", "outdoor",
    "simultaneously",
    "accordion", "acoustic_guitar", "bagpipe", "banjo", "bassoon",
    "cello", "clarinet", "congas", "drum", "electric_bass", "erhu",
    "flute", "guzheng", "piano", "pipa", "saxophone", "trumpet",
    "tuba", "ukulele", "violin", "xylophone",
}


def extract_answer(text: str) -> str:
    """Extract answer from model output, constrained to AVQA vocabulary."""
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|]+\|>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()

    if text in ANSWER_VOCAB:
        return text

    first_word = text.split()[0] if text.split() else ""
    first_word = first_word.strip(".,!?;:")
    if first_word in ANSWER_VOCAB:
        return first_word

    for word in ANSWER_VOCAB:
        if word in text:
            return word
    return first_word


def evaluate(
    model,
    tokenizer,
    qa_pairs: list[dict[str, Any]],
    audio_features,
    vision_proxy_features,
    real_vision_features,
    device,
    mode: str = "proxy_v_real_a",
    max_samples: int | None = None,
    num_vision_tokens: int = 1,
    num_audio_tokens: int = 1,
    text_control_mode: str = "none",
    image_captions: dict[str, str] | None = None,
    text_shuffle_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run evaluation under a specific mode and return per-example records."""
    model.eval()
    samples = qa_pairs[:max_samples] if max_samples else qa_pairs
    records: list[dict[str, Any]] = []

    for qa in tqdm(samples, desc=mode):
        video_id = qa["video_id"]
        question = resolve_templates(qa["question_content"], qa.get("templ_values", "[]"))
        gt_answer = qa.get("anser", qa.get("answer", "")).lower().strip()
        q_type = json.loads(qa["type"]) if isinstance(qa["type"], str) else qa["type"]
        q_type_str = "_".join(q_type)

        vision_block, audio_block = get_modality_prompt_blocks(tokenizer, num_vision_tokens=num_vision_tokens, num_audio_tokens=num_audio_tokens)
        control_text = get_text_control_payload(
            video_id,
            text_control_mode,
            image_captions or {},
            text_shuffle_map,
        )

        if mode == "text_only":
            user_content = build_user_content(
                question, vision_block, audio_block,
                text_control_mode=text_control_mode,
                control_text=control_text,
                include_vision=False, include_audio=False,
            )
        elif mode == "audio_only":
            user_content = build_user_content(
                question, vision_block, audio_block,
                text_control_mode=text_control_mode,
                control_text=control_text,
                include_vision=False, include_audio=True,
            )
        elif mode == "vision_only":
            user_content = build_user_content(
                question, vision_block, audio_block,
                text_control_mode=text_control_mode,
                control_text=control_text,
                include_vision=True, include_audio=False,
            )
        else:
            user_content = build_user_content(
                question, vision_block, audio_block,
                text_control_mode=text_control_mode,
                control_text=control_text,
                include_vision=True, include_audio=True,
            )

        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        audio_feat = None
        vision_feat = None

        if mode in ("audio_only", "proxy_v_real_a", "real_v_real_a") and video_id in audio_features:
            audio_feat = audio_features[video_id].unsqueeze(0).to(device)

        if text_control_mode == "none":
            if mode == "proxy_v_real_a":
                if video_id in vision_proxy_features:
                    vision_feat = vision_proxy_features[video_id].unsqueeze(0).to(device)
            elif mode == "vision_only":
                if real_vision_features and video_id in real_vision_features:
                    vision_feat = real_vision_features[video_id].unsqueeze(0).to(device)
                elif video_id in vision_proxy_features:
                    vision_feat = vision_proxy_features[video_id].unsqueeze(0).to(device)
            elif mode == "real_v_real_a":
                if real_vision_features and video_id in real_vision_features:
                    vision_feat = real_vision_features[video_id].unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            eos_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>")]
            think_end = tokenizer.convert_tokens_to_ids("</think>")
            if think_end is not None:
                eos_ids.append(think_end)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_feat,
                vision_proxy_features=vision_feat,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=eos_ids,
            )

        gen_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=False)
        pred = extract_answer(gen_text)
        records.append({
            "question_id": qa.get("question_id"),
            "video_id": video_id,
            "question": question,
            "question_type": q_type_str,
            "ground_truth": gt_answer,
            "prediction": pred,
            "correct": pred == gt_answer,
            "mode": mode,
            "used_audio": audio_feat is not None,
            "used_vision": vision_feat is not None,
        })

    return records


def print_subset_table(mode_summaries: dict[str, dict[str, Any]], modes: list[str]) -> None:
    header = f"{'Subset':10s}"
    for mode in modes:
        header += f" | {mode:>15s}"
    print(header)
    print("-" * len(header))

    for subset in SUBSET_ORDER:
        row = f"{subset:10s}"
        for mode in modes:
            summary = mode_summaries[mode]
            if subset == "overall":
                acc = float(summary["overall"]["accuracy"])
            else:
                acc = float(summary["by_subset"].get(subset, {}).get("accuracy", 0.0))
            row += f" | {format_pct(acc):>15s}"
        print(row)


def print_question_type_table(mode_summaries: dict[str, dict[str, Any]], modes: list[str]) -> None:
    q_types = set()
    for summary in mode_summaries.values():
        q_types.update(summary["by_question_type"].keys())
    q_types = sorted(q_types)

    header = f"{'Question Type':30s}"
    for mode in modes:
        header += f" | {mode:>15s}"
    print(header)
    print("-" * len(header))

    for q_type in q_types:
        row = f"{q_type:30s}"
        for mode in modes:
            acc = float(mode_summaries[mode]["by_question_type"].get(q_type, {}).get("accuracy", 0.0))
            row += f" | {format_pct(acc):>15s}"
        print(row)


def print_interaction_table(title: str, metrics: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    header = f"{'Subset':10s} | {'VA':>8s} | {'V0':>8s} | {'A0':>8s} | {'T0':>8s} | {'G':>8s} | {'I_va':>8s}"
    print(header)
    print("-" * len(header))
    for subset in SUBSET_ORDER:
        row = metrics[subset]
        print(
            f"{subset:10s} | "
            f"{100.0 * row['va_accuracy']:7.1f}% | "
            f"{100.0 * row['vision_only_accuracy']:7.1f}% | "
            f"{100.0 * row['audio_only_accuracy']:7.1f}% | "
            f"{100.0 * row['text_only_accuracy']:7.1f}% | "
            f"{100.0 * row['G']:7.1f} | "
            f"{100.0 * row['I_va']:7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Composition evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=512,
                        help="LoRA rank of the checkpoint being loaded.")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument(
        "--num_vision_tokens",
        type=int,
        default=None,
        help="Number of proxy/real vision tokens to use. If omitted, infer from checkpoint name.",
    )
    parser.add_argument(
        "--num_audio_tokens",
        type=int,
        default=1,
        help="Number of audio tokens to use.",
    )
    parser.add_argument(
        "--modality_split_path",
        type=str,
        default="data/music_avqa/analysis/modality_split.json",
        help="Path to frozen S1-S4 split metadata.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save summary JSON.",
    )
    parser.add_argument(
        "--text_control_mode",
        type=str,
        default="none",
        choices=TEXT_CONTROL_MODES,
        help="Prompt-based extra-text control variant to evaluate.",
    )
    parser.add_argument(
        "--proxy_vision_features_path",
        type=str,
        default=None,
        help="Optional explicit path to proxy vision features.",
    )
    parser.add_argument(
        "--real_vision_features_path",
        type=str,
        default=None,
        help="Optional explicit path to real vision features.",
    )
    parser.add_argument(
        "--audio_features_path",
        type=str,
        default=None,
        help="Optional explicit path to audio features.",
    )
    parser.add_argument(
        "--image_captions_path",
        type=str,
        default="data/music_avqa/image_captions.json",
        help="Path to per-video image captions used by text-control experiments.",
    )
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = "models/qwen3-8b"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    if args.num_vision_tokens is None:
        args.num_vision_tokens = 4 if "4tok" in args.checkpoint or "4token" in args.checkpoint else 1

    features_dir = Path("data/music_avqa/features")
    if args.audio_features_path:
        audio_path = Path(args.audio_features_path)
    elif args.num_audio_tokens > 1:
        audio_path = features_dir / f"real_audio_{args.num_audio_tokens}token.pt"
    else:
        audio_path = features_dir / "real_audio_features.pt"
    audio_features = torch.load(audio_path, weights_only=True)
    if args.proxy_vision_features_path:
        proxy_path = Path(args.proxy_vision_features_path)
    elif args.num_vision_tokens > 1:
        proxy_path = features_dir / f"vision_proxy_{args.num_vision_tokens}token.pt"
    else:
        proxy_path = features_dir / "vision_proxy_features.pt"

    if args.real_vision_features_path:
        real_vision_path = Path(args.real_vision_features_path)
    elif args.num_vision_tokens > 1:
        real_vision_path = features_dir / f"real_vision_{args.num_vision_tokens}token.pt"
    else:
        real_vision_path = features_dir / "real_vision_features.pt"

    vision_proxy_features = torch.load(proxy_path, weights_only=True)
    real_vision_features = torch.load(real_vision_path, weights_only=True) if real_vision_path.exists() else {}
    print(
        f"Audio: {len(audio_features)}, Vision proxy: {len(vision_proxy_features)}, "
        f"Vision real: {len(real_vision_features)}, num_vision_tokens={args.num_vision_tokens}"
    )

    with open(Path(METADATA_DIR) / f"avqa-{args.split}.json") as f:
        qa_pairs = json.load(f)
    qa_pairs = [q for q in qa_pairs if q["video_id"] in audio_features]
    print(f"Eval samples: {len(qa_pairs)}")

    split_index = load_split_index(args.modality_split_path)
    print(f"Loaded modality split metadata: {len(split_index)} rows from {args.modality_split_path}")
    image_captions = load_image_captions(args.image_captions_path)
    text_shuffle_map = build_text_shuffle_map([q["video_id"] for q in qa_pairs], seed=0) if args.text_control_mode == "shuffled_question" else None

    print(f"Loading model from {args.checkpoint}...")
    from src.models.qwen_audio_model import QwenWithAudio
    model = QwenWithAudio(qwen_path=args.model_path, num_audio_tokens=args.num_audio_tokens)
    if args.stage == 2:
        from types import SimpleNamespace
        lora_cfg = SimpleNamespace(
            enabled=True, r=args.lora_r, alpha=args.lora_alpha,
            dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model.configure_training(2, lora_cfg)
    model.load_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    modes = ["real_v_real_a", "proxy_v_real_a", "audio_only", "vision_only", "text_only"]
    if args.text_control_mode != "none":
        print("Note: text control mode disables vision feature injection; real/proxy vision mode names remain for table compatibility.")
    mode_records: dict[str, list[dict[str, Any]]] = {}
    mode_summaries: dict[str, dict[str, Any]] = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {mode}")
        print(f"{'=' * 60}")
        records = evaluate(
            model, tokenizer, qa_pairs, audio_features, vision_proxy_features,
            real_vision_features, device, mode=mode, max_samples=args.max_samples,
            num_vision_tokens=args.num_vision_tokens, num_audio_tokens=args.num_audio_tokens,
            text_control_mode=args.text_control_mode, image_captions=image_captions,
            text_shuffle_map=text_shuffle_map,
        )
        attach_split_metadata(records, split_index, args.split)
        summary = summarize_records(records)
        mode_records[mode] = records
        mode_summaries[mode] = summary

        print(
            f"\n>>> {mode}: {format_pct(float(summary['overall']['accuracy']))} "
            f"({summary['overall']['correct']}/{summary['overall']['total']})"
        )
        subset_bits = []
        for subset in ["S1", "S2", "S3", "S4"]:
            info = summary["by_subset"].get(subset, {})
            subset_bits.append(f"{subset}={format_pct(float(info.get('accuracy', 0.0)))}")
        print("    " + ", ".join(subset_bits))

    real_interaction = compute_interaction_metrics(mode_summaries, va_mode="real_v_real_a")
    proxy_interaction = compute_interaction_metrics(mode_summaries, va_mode="proxy_v_real_a")

    print(f"\n{'=' * 90}")
    print("Composition results by modality split")
    print(f"{'=' * 90}")
    print_subset_table(mode_summaries, modes)

    print(f"\n{'=' * 90}")
    print("Question type breakdown")
    print(f"{'=' * 90}")
    print_question_type_table(mode_summaries, modes)

    print(f"\n{'=' * 90}")
    print("Interaction metrics")
    print(f"{'=' * 90}")
    print_interaction_table("Real Vision + Real Audio vs baselines", real_interaction)
    print_interaction_table("Proxy Vision + Real Audio vs baselines", proxy_interaction)

    if args.output_json:
        payload = {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "num_vision_tokens": args.num_vision_tokens,
            "num_audio_tokens": args.num_audio_tokens,
            "text_control_mode": args.text_control_mode,
            "mode_summaries": mode_summaries,
            "interaction_metrics": {
                "real_v_real_a": real_interaction,
                "proxy_v_real_a": proxy_interaction,
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved summary JSON to {out_path}")


if __name__ == "__main__":
    main()
