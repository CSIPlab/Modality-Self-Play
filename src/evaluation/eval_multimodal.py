"""General evaluation for MultiModalModel on any QA benchmark.

Loads a checkpoint, runs generation with modality features injected,
extracts answers, and computes accuracy metrics.

Usage:
    # Evaluate vision-only
    python -m src.evaluation.eval_multimodal \\
        --checkpoint outputs/unimodal_vision/final \\
        --modalities vision \\
        --feature_paths vision:data/features/image_features.pt \\
        --qa_data data/llava_instruct_150k/llava_instruct_150k.json \\
        --qa_format llava

    # Evaluate audio-visual composition
    python -m src.evaluation.eval_multimodal \\
        --checkpoint outputs/selfplay_audio_from_vision/final \\
        --modalities vision,audio \\
        --feature_paths vision:data/features/image_features.pt,audio:data/music_avqa/features/real_audio_features.pt \\
        --qa_data /path/to/avqa-val.json \\
        --qa_format avqa
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.multimodal_dataset import MODALITY_DEFAULTS, QA_PARSERS, DEFAULT_KEY_FNS
from src.models.multimodal_model import MultiModalModel, get_modality_placeholder


# ── Answer extraction ───────────────────────────────────────────────────

COMMON_VOCAB = {
    "yes", "no",
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "left", "right", "middle", "center",
    "indoor", "outdoor",
    "simultaneously",
}

# MUSIC-AVQA instrument vocabulary
INSTRUMENT_VOCAB = {
    "accordion", "acoustic_guitar", "bagpipe", "banjo", "bassoon",
    "cello", "clarinet", "congas", "drum", "electric_bass", "erhu",
    "flute", "guzheng", "piano", "pipa", "saxophone", "trumpet",
    "tuba", "ukulele", "violin", "xylophone",
}

# Number words to digits mapping
NUMBER_WORDS = {
    "zero": "zero", "none": "zero", "0": "zero",
    "one": "one", "single": "one", "1": "one",
    "two": "two", "2": "two", "pair": "two",
    "three": "three", "3": "three",
    "four": "four", "4": "four",
    "five": "five", "5": "five",
    "six": "six", "6": "six",
    "seven": "seven", "7": "seven",
}

ALL_VOCAB = COMMON_VOCAB | INSTRUMENT_VOCAB


def _word_in_text_match(pred: str, gt: str) -> bool:
    """Word-boundary containment match.

    Used for benchmarks where the model is captioning-trained but the
    eval expects a single-word answer (e.g. MSVD-QA: gold "man" is
    correct if pred is "the man opens a box"). Strict EM on canonical
    forms is checked first; otherwise each canonical form of gt must
    appear as a complete word/phrase in pred (word-boundary regex
    anchored on \b). Multi-word gt phrases are matched as a contiguous
    word sequence, not character substring — so gt "no" never matches
    pred "not", and gt "one" never matches pred "alone".
    """
    if _lenient_match(pred, gt):
        return True
    p_norm = " " + re.sub(r"\s+", " ", pred.lower()) + " "
    g_forms = _canonical_forms(gt)
    for g in g_forms:
        g = g.strip()
        if not g:
            continue
        # Word-boundary contains: gt as a complete word/phrase in pred.
        if re.search(r"\b" + re.escape(g) + r"\b", p_norm):
            return True
    return False


# Per-qa_format matchers. Default = strict canonical EM. Override for
# benchmarks where the gold answer is a single content word that the
# captioning-trained model emits inside a sentence.
_MATCHERS_BY_FORMAT: dict[str, Any] = {
    "videoinst_short": _word_in_text_match,  # MSVD-QA / MSRVTT-QA
}


def _reference_answers(sample: dict[str, Any], fallback: str) -> list[str]:
    """Collect all acceptable answers for a sample.

    Some benchmarks provide multiple human references in `answers`;
    the rest fall back to the parser's single target answer.
    """
    refs: list[str] = []
    raw_answers = sample.get("answers") if isinstance(sample, dict) else None
    if isinstance(raw_answers, list):
        for ans in raw_answers:
            ans_text = str(ans).strip().lower()
            if ans_text:
                refs.append(ans_text)
    # Captioning sources: dedup files for COCO/Clotho ship a
    # `references` list with the multi-reference ground truth (5 captions
    # per image for COCO; ~5 per audio clip for Clotho).
    raw_refs = sample.get("references") if isinstance(sample, dict) else None
    if isinstance(raw_refs, list):
        for ans in raw_refs:
            ans_text = str(ans).strip().lower()
            if ans_text:
                refs.append(ans_text)
    fallback = fallback.strip().lower()
    if fallback:
        refs.append(fallback)

    deduped: list[str] = []
    seen: set[str] = set()
    for ans in refs:
        if ans not in seen:
            seen.add(ans)
            deduped.append(ans)
    return deduped


def extract_answer(text: str, vocab: set[str] | None = None) -> str:
    """Extract a short answer from model output with fuzzy matching.

    Handles common variations:
    - "The answer is violin" → "violin"
    - "It's a cello" → "cello"
    - "There are two instruments" → "two"
    - "Yes, there is" → "yes"
    - "acoustic guitar" → "acoustic_guitar"
    """
    if vocab is None:
        vocab = ALL_VOCAB

    # Strip thinking tags and special tokens
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|]+\|>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = normalize_answer(text)

    if not text:
        return ""

    # Direct match
    if text in vocab:
        return text

    # Check underscore variants (acoustic guitar → acoustic_guitar)
    text_underscored = text.replace(" ", "_")
    if text_underscored in vocab:
        return text_underscored

    # Strip common prefixes
    for prefix in ["the answer is ", "it is ", "it's ", "i think it's ",
                    "i think ", "there is ", "there are ", "the ",
                    "a ", "an ", "they are "]:
        if text.startswith(prefix):
            stripped = text[len(prefix):]
            stripped_u = stripped.replace(" ", "_")
            if stripped in vocab or stripped_u in vocab:
                return stripped_u if stripped_u in vocab else stripped

    # Number word normalization via token-wise exact matching
    for token in text.split():
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]

    # Check each word in the response against vocab
    words = text.split()
    for word in words:
        word_clean = re.sub(r"[.,!?;:]", "", word)
        if word_clean in vocab:
            return word_clean
        word_u = word_clean.replace(" ", "_")
        if word_u in vocab:
            return word_u

    # Check multi-word vocab entries (acoustic guitar, electric bass).
    # Use whole-phrase matching only; substring checks like "one" in
    # "someone" create false positives on open-ended QA benchmarks.
    for v in vocab:
        v_spaced = v.replace("_", " ")
        if " " not in v_spaced and "_" not in v:
            continue
        if re.search(rf"\b{re.escape(v_spaced)}\b", text):
            return v

    # Fallback: first word
    first = words[0] if words else ""
    return re.sub(r"[.,!?;:]", "", first)


# ── Evaluation ──────────────────────────────────────────────────────────

def load_bridges_from_checkpoint(
    checkpoint_path: str,
    modalities: list[str],
    device: torch.device,
    model=None,
) -> dict:
    """Load co-trained gap bridges saved alongside a Phase 2 composition
    checkpoint. Returns a dict {modality_name: bridge_module}.

    Phase 2 training saves `{modality}_bridge.pt` state dicts plus a
    `bridge_config.json` sidecar recording the bridge method used. We
    reconstruct each bridge via `build_bridge(method)` and load state.
    """
    from pathlib import Path
    bridges: dict = {}
    ckpt_dir = Path(checkpoint_path)
    config_path = ckpt_dir / "bridge_config.json"
    if not config_path.exists():
        # No bridges saved (gap_bridge was "none" or legacy checkpoint)
        return bridges

    with open(config_path) as f:
        cfg = json.load(f)
    method = cfg.get("method", "none")
    if method == "none":
        return bridges

    from src.analysis.gap_bridging import build_bridge
    mod_cfgs = cfg.get("modalities", {})

    for name in modalities:
        # Checkpoint-saved bridges use `{name}_bridge.pt`; offline-fit
        # bridge dirs (e.g. outputs/default_bridges) use
        # `{name}_bridge_fit.pt`. Try both.
        bridge_path = ckpt_dir / f"{name}_bridge.pt"
        if not bridge_path.exists():
            bridge_path = ckpt_dir / f"{name}_bridge_fit.pt"
        if not bridge_path.exists():
            continue
        try:
            state = torch.load(
                bridge_path, map_location="cpu", weights_only=True)
            mod_cfg = mod_cfgs.get(name, {})
            input_dim = mod_cfg.get("input_dim") or \
                MODALITY_DEFAULTS.get(name, {}).get("dim", 512)
            bridge_method = mod_cfg.get("method") or method
            build_kwargs = {"input_dim": input_dim}
            if bridge_method == "residual_mlp":
                first_w = next((v for v in state.values()
                                if isinstance(v, torch.Tensor) and v.dim() == 2),
                               None)
                build_kwargs["residual_hidden"] = int(first_w.shape[0]) if first_w is not None else 256
            if bridge_method == "ridge" and "lam" in mod_cfg:
                build_kwargs["ridge_lam"] = float(mod_cfg["lam"])
            bridge = build_bridge(bridge_method, **build_kwargs)
            bridge.load_state_dict(state, strict=False)
            bridge = bridge.to(device).eval()
            bridges[name] = bridge
            print(f"  Loaded {name} bridge ({bridge_method}, dim={input_dim}) "
                  f"from {bridge_path.name}")
        except Exception as e:
            print(f"  Warning: could not load {bridge_path.name}: {e}")
    return bridges


@torch.no_grad()
def evaluate(
    model: MultiModalModel,
    tokenizer,
    qa_data: list[dict],
    features: dict[str, dict[str, torch.Tensor]],
    qa_format: str = "llava",
    modalities: list[str] | None = None,
    max_samples: int | None = None,
    answer_vocab: set[str] | None = None,
    bridges: dict | None = None,
    bridge_modalities: list[str] | None = None,
    proxy_features: dict[str, dict[str, torch.Tensor]] | None = None,
    proxy_key_strip_prefixes: list[str] | None = None,
    prompt_modalities_mode: str = "checkpoint_all",
) -> list[dict[str, Any]]:
    """Run generation evaluation.

    Args:
        model: Loaded MultiModalModel.
        tokenizer: Qwen 3 tokenizer.
        qa_data: List of QA sample dicts.
        features: {modality_name: {sample_key: tensor}}.
        qa_format: QA parser format name.
        modalities: Which modalities to inject (default: all in features).
        max_samples: Limit evaluation samples.
        answer_vocab: Optional vocabulary constraint.

    Returns:
        List of per-sample record dicts.
    """
    model.eval()
    device = next(model.parameters()).device
    parser = QA_PARSERS[qa_format]
    key_fn = DEFAULT_KEY_FNS.get(qa_format)
    matcher = _MATCHERS_BY_FORMAT.get(qa_format, _lenient_match)
    samples = qa_data[:max_samples] if max_samples else qa_data

    if modalities is None:
        modalities = list(features.keys())

    records = []

    trace_samples = os.environ.get("EVAL_TRACE_SAMPLES", "").lower() in {"1", "true", "yes"}

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        sample_key = key_fn(sample) if key_fn else ""
        turns = parser(sample)
        if not turns:
            continue

        question = turns[0][0]
        gt_answer = turns[0][1].strip().lower() if turns[0][1] else ""
        reference_answers = _reference_answers(sample, gt_answer)

        # Build prompt with modality placeholders.
        # Two prompt regimes are used in training:
        #   checkpoint_all: all trained modalities, sorted. This matches the
        #     anchor-invariant MSP/composition prompt convention.
        #   caller_only: only the caller-requested modalities. This matches
        #     unimodal-only training where just the anchor placeholder is
        #     present in the prompt.
        if prompt_modalities_mode == "checkpoint_all":
            prompt_modalities = sorted(modalities)
        elif prompt_modalities_mode == "caller_only":
            prompt_modalities = list(modalities)
        else:
            raise ValueError(
                f"Unknown prompt_modalities_mode={prompt_modalities_mode!r}")

        all_placeholders = []
        for mod in prompt_modalities:
            ntok = model.modality_num_tokens.get(mod, 1)
            all_placeholders.append(get_modality_placeholder(mod, ntok))
        prefix = " ".join(all_placeholders)

        # Replace <image> in question or prepend
        if "<image>" in question:
            question_mod = question.replace("<image>", prefix)
        else:
            question_mod = f"{prefix}\n{question}"

        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{question_mod}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        # Load features
        modality_features = {}
        for mod in modalities:
            feat_dict = features.get(mod, {})
            # Try key with and without extension. Some QA formats (AVQA)
            # emit integer video_ids; coerce to str before the suffix check.
            str_key = str(sample_key)
            clean_key = str_key
            for ext in (".jpg", ".jpeg", ".png", ".wav", ".mp4", ".npy"):
                if clean_key.endswith(ext):
                    clean_key = clean_key[:-len(ext)]
                    break

            feat = feat_dict.get(sample_key,
                                 feat_dict.get(str_key,
                                               feat_dict.get(clean_key, None)))
            if feat is not None:
                feat = feat.unsqueeze(0).to(device)
                # Apply bridge if this modality is marked as proxy + bridge loaded
                if (bridges is not None and mod in bridges
                        and bridge_modalities is not None
                        and mod in bridge_modalities):
                    orig_shape = feat.shape
                    flat = feat.reshape(-1, feat.shape[-1])
                    bdtype = next(bridges[mod].parameters()).dtype
                    flat = flat.to(bdtype)
                    flat = bridges[mod](flat).reshape(orig_shape).to(feat.dtype)
                    feat = flat
                modality_features[mod] = feat
            else:
                # Try the proxy bank first (cross-modal imagined features).
                # The MSP-trained model expects all modalities present in
                # every batch — zero-filling missing modalities at eval
                # time is OOD and degenerates generation. Proxies match
                # the training distribution.
                proxy_feat = None
                if proxy_features and mod in proxy_features:
                    pdict = proxy_features[mod]
                    # Try sample_key, then key with stripped prefixes.
                    proxy_feat = pdict.get(str_key, pdict.get(clean_key))
                    if proxy_feat is None and proxy_key_strip_prefixes:
                        for prefix in proxy_key_strip_prefixes:
                            if str_key.startswith(prefix):
                                stripped = str_key[len(prefix):]
                                proxy_feat = pdict.get(
                                    stripped, pdict.get(stripped.split('.')[0]))
                                if proxy_feat is not None:
                                    break
                if proxy_feat is not None:
                    feat = proxy_feat.unsqueeze(0).to(device)
                    if (bridges is not None and mod in bridges
                            and bridge_modalities is not None
                            and mod in bridge_modalities):
                        orig_shape = feat.shape
                        flat = feat.reshape(-1, feat.shape[-1])
                        bdtype = next(bridges[mod].parameters()).dtype
                        flat = flat.to(bdtype)
                        flat = bridges[mod](flat).reshape(orig_shape).to(feat.dtype)
                        feat = flat
                    modality_features[mod] = feat
                    continue
                # Fall back to projector's actual input dim (zero-fill).
                if mod in model.projectors:
                    dim = model.projectors[mod].mlp[0].normalized_shape[0]
                else:
                    dim = MODALITY_DEFAULTS.get(mod, {}).get("dim", 512)
                ntok = model.modality_num_tokens.get(mod, 1)
                if ntok > 1:
                    modality_features[mod] = torch.zeros(1, ntok, dim, device=device)
                else:
                    modality_features[mod] = torch.zeros(1, dim, device=device)

        # Generate — disable thinking, short answer only
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        use_cuda_autocast = os.environ.get("EVAL_DISABLE_CUDA_AUTOCAST", "").lower() not in {
            "1", "true", "yes"
        }
        use_kv_cache = os.environ.get("EVAL_DISABLE_KV_CACHE", "").lower() not in {
            "1", "true", "yes"
        }
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda_autocast else
            contextlib.nullcontext()
        )
        with autocast_ctx:
            if trace_samples:
                print(f"[trace] idx={idx} key={sample_key}", flush=True)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                modality_features=modality_features,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=eos_id,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=use_kv_cache,
            )

        gen_text = tokenizer.decode(
            generated[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # With a multi_choice list, snap the freeform output to the closest
        # option: match the answer text first, then a bare letter, then fall
        # back to the vocab extractor.
        choices = sample.get("multi_choice") if isinstance(sample, dict) else None
        if isinstance(choices, list) and choices:
            pred = None
            stripped = re.sub(
                r"^\s*[\(\[]?([A-Da-d])[\)\].:\-]?\s+", "", gen_text).strip()
            for c in choices:
                if _lenient_match(stripped, str(c)) \
                        or _lenient_match(gen_text, str(c)):
                    pred = str(c)
                    break
            if pred is None:
                m = re.match(r"^\s*[\(\[]?([A-Da-d])[\)\].:\-]?\s*$",
                             gen_text.strip())
                if m:
                    idx = ord(m.group(1).upper()) - ord("A")
                    if 0 <= idx < len(choices):
                        pred = str(choices[idx])
            if pred is None:
                pred = extract_answer(gen_text, answer_vocab)
        else:
            pred = extract_answer(gen_text, answer_vocab)

        # Most QA formats use canonical-EM. videoinst_short (MSVD/MSRVTT-QA)
        # additionally accepts the gold word appearing as a complete word
        # in the model's full generated text — captioning-trained models
        # answer "the man opens a box" for gold "man".
        correct = any(matcher(pred, ans) for ans in reference_answers)
        if matcher is _word_in_text_match:
            correct = correct or any(
                matcher(gen_text, ans) for ans in reference_answers)

        records.append({
            "sample_key": sample_key,
            "question": question[:100],
            "ground_truth": gt_answer,
            "reference_answers": reference_answers,
            "prediction": pred,
            "correct": correct,
            "modalities": modalities,
            "generated_text": gen_text[:200],
        })

    return records


def _clean_caption_metric_text(text: str) -> str:
    """Normalize caption text for line-oriented caption scorers."""
    text = str(text).replace("|||", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(records: list[dict]) -> dict:
    """Compute accuracy + (when references present) caption metrics.

    For QA evals, accuracy is the headline. For captioning evals, the
    sample dicts carry a `references` list (multi-reference ground truth)
    and we score the generated text against them with corpus BLEU-1..4
    and ROUGE-L (mean over rows). CIDEr requires pycocoevalcap which
    isn't installed in this env; reporting BLEU + ROUGE-L is enough to
    distinguish working vs degenerate generation.
    """
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    accuracy = correct / total if total > 0 else 0.0
    out = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    # Captioning metrics — fire when we have at least one reference per
    # most rows. Uses pycocoevalcap (the standard captioning suite that
    # OneLLM and most papers report against): BLEU-1..4, METEOR, ROUGE-L,
    # CIDEr. Skips SPICE (needs Stanford CoreNLP setup).
    has_refs = sum(1 for r in records if r.get("reference_answers"))
    if has_refs >= max(10, total // 2):
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider

            gts: dict[str, list[str]] = {}
            res: dict[str, list[str]] = {}
            for i, r in enumerate(records):
                gen = _clean_caption_metric_text(
                    r.get("generated_text") or r.get("prediction") or "")
                ref_list = [
                    _clean_caption_metric_text(x)
                    for x in (r.get("reference_answers") or [])
                    if x and _clean_caption_metric_text(x)
                ]
                if not gen or not ref_list:
                    continue
                key = str(r.get("sample_key") or i)
                gts[key] = ref_list
                res[key] = [gen]

            if gts:
                scorers = [
                    (Bleu(4),  ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
                    (Rouge(),  "rouge_l"),
                    (Cider(),  "cider"),
                ]
                for scorer, names in scorers:
                    try:
                        score, _ = scorer.compute_score(gts, res)
                    except Exception as inner:
                        out[f"{names if isinstance(names, str) else 'bleu'}_error"] = str(inner)
                        continue
                    if isinstance(names, list):
                        for n, s in zip(names, score):
                            out[n] = float(s)
                    else:
                        out[names] = float(score)
                # METEOR is the only pycocoevalcap scorer here that shells
                # out to a long-lived Java subprocess. It is stable on the
                # smaller UM benches, but the 5k-image COCO run has been
                # observed to wedge after generation inside METEOR. Keep the
                # reliable BLEU/ROUGE/CIDEr metrics for large evals and skip
                # METEOR there so the job can finish and unblock the queue.
                if len(gts) > 2000:
                    out["meteor_skipped"] = (
                        f"skipped on large eval (n={len(gts)}) to avoid "
                        "known METEOR Java hangs; BLEU/ROUGE/CIDEr reported")
                else:
                    try:
                        score, _ = Meteor().compute_score(gts, res)
                        out["meteor"] = float(score)
                    except Exception as inner:
                        out["meteor_error"] = str(inner)
                out["captioning_n"] = len(gts)
        except Exception as e:
            out["captioning_metric_error"] = str(e)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="General multimodal evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_path", default="models/qwen3-8b")
    parser.add_argument("--modalities", type=str, required=True,
                        help="Comma-separated: vision,audio")
    parser.add_argument("--feature_paths", type=str, required=True,
                        help="Comma-separated modality:path pairs")
    parser.add_argument(
        "--proxy_paths", type=str, default="",
        help="Comma-separated <mod>:<path> pairs of cross-modal proxy "
             "feature dicts. When the model was trained with all modalities "
             "present in every batch (round-robin MSP), single-modality "
             "captioning evals zero-fill the missing modalities and the "
             "model degenerates. Pointing this at the training proxy banks "
             "(or fresh imaginations for the eval set) keeps eval-time "
             "input distribution aligned with training.")
    parser.add_argument(
        "--proxy_key_strip_prefixes", type=str, default="",
        help="Comma-separated prefixes to strip from sample keys when "
             "looking up proxies (e.g. 'COCO_val2014_,COCO_train2014_' "
             "to match Karpathy-style image names against COCO-id-keyed "
             "proxy banks).")
    parser.add_argument(
        "--prompt_modalities_mode",
        default="checkpoint_all",
        choices=["checkpoint_all", "caller_only"],
        help="Prompt placeholder regime. `checkpoint_all` auto-expands to "
             "all trained modalities and matches composition/MSP prompts. "
             "`caller_only` keeps only the caller-requested modalities in "
             "the prompt and matches unimodal-only training prompts.")
    parser.add_argument("--qa_data", required=True)
    parser.add_argument("--qa_format", default="llava",
                        choices=list(QA_PARSERS.keys()))
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_saved_records", type=int, default=0,
        help="Cap on per-record entries saved to --output_json. "
             "0 = save everything. Was historically 100 for quick "
             "inspection; set to 0 now so diagnostics can analyse the "
             "full eval.")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--bridge_modalities", type=str, default=None,
                        help="Comma-separated modalities to apply bridges to "
                             "(for proxy-feature eval). Bridges are loaded "
                             "from {checkpoint}/{name}_bridge.pt.")
    parser.add_argument("--override_bridge_dir", type=str, default=None,
                        help="If set, load bridges from this directory "
                             "instead of the checkpoint. Useful for testing "
                             "a newer bridge (e.g. outputs/default_bridges) "
                             "against an existing checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)

    # Parse modalities and feature paths. `--modalities none` (or empty) runs a
    # question-only / text-prior baseline: no modality placeholders, no features.
    _raw_mods = args.modalities.strip()
    text_only = _raw_mods.lower() in ("", "none")
    modalities = [] if text_only else [m.strip() for m in _raw_mods.split(",")]
    feature_paths = {}
    if not text_only and args.feature_paths.strip().lower() not in ("", "none"):
        for pair in args.feature_paths.split(","):
            mod, path = pair.strip().split(":")
            feature_paths[mod.strip()] = path.strip()

    # `checkpoint_all` matches the anchor-invariant composition prompt:
    # auto-expand to the full checkpoint modality set. `caller_only`
    # preserves the caller-provided subset and matches unimodal-only
    # training where the prompt includes only the anchor placeholder.
    from pathlib import Path as _Path
    mc_path = _Path(args.checkpoint) / "modality_config.json"
    if not text_only and args.prompt_modalities_mode == "checkpoint_all" and mc_path.exists():
        try:
            mc = json.load(open(mc_path))
            trained_mods = sorted(mc.get("modalities", {}).keys())
            missing = [m for m in trained_mods if m not in modalities]
            if missing:
                print(f"Note: checkpoint was trained with {trained_mods}, "
                      f"caller passed {modalities}. Auto-expanding prompt "
                      f"to cover missing modalities with zero features: {missing}")
                for m in missing:
                    modalities.append(m)
        except Exception as e:
            print(f"Warning: could not read {mc_path}: {e}")

    # Load model
    model = MultiModalModel.from_checkpoint(args.checkpoint, args.model_path)
    if args.stage == 2:
        # Detect full-FT vs LoRA: llm_full.pt → full FT (no PEFT wrap);
        # llm_lora.pt → LoRA (need PEFT wrap with matching rank).
        from pathlib import Path as _P
        is_full_ft = (_P(args.checkpoint) / "llm_full.pt").exists()
        if is_full_ft:
            print(f"Detected full-FT checkpoint (llm_full.pt present); "
                  f"skipping LoRA wrap.")
            # Still need to mark projectors as trainable=False (eval mode).
            # configure_training with stage=1 freezes LLM and unfreezes
            # projectors but doesn't wrap in PEFT. That's fine for eval.
            model.configure_training(stage=1, trainable_modalities=[])
        else:
            model.configure_training(
                stage=2, trainable_modalities=[],
                lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        model.load_checkpoint(args.checkpoint)
    model.to(device).eval()

    # Load co-trained bridges if any are saved in the checkpoint dir,
    # OR from --override_bridge_dir if set (e.g. newer ridge bridges).
    bridge_source = args.override_bridge_dir or args.checkpoint
    if args.override_bridge_dir:
        print(f"Bridge override: loading from {args.override_bridge_dir} "
              f"(checkpoint's own bridges ignored)")
    bridges = load_bridges_from_checkpoint(bridge_source, modalities, device)
    bridge_modalities = None
    if args.bridge_modalities:
        bridge_modalities = [m.strip() for m in args.bridge_modalities.split(",")]
        missing = [m for m in bridge_modalities if m not in bridges]
        if missing:
            print(f"Warning: --bridge_modalities requested {missing} but "
                  f"no bridges found for those in checkpoint")

    # Load features
    features = {}
    for mod in modalities:
        if mod in feature_paths:
            features[mod] = torch.load(
                feature_paths[mod], weights_only=True)
            print(f"  {mod}: {len(features[mod])} entries")

    # Load QA data
    qa_path = Path(args.qa_data)
    if qa_path.suffix == ".jsonl":
        with open(qa_path) as f:
            qa_data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(qa_path) as f:
            qa_data = json.load(f)
    print(f"  QA: {len(qa_data)} samples")

    # Optional proxy banks for missing modalities at eval time.
    proxy_features: dict[str, dict[str, torch.Tensor]] = {}
    if args.proxy_paths:
        for pair in args.proxy_paths.split(","):
            mod, path = pair.strip().split(":")
            print(f"  Loading proxy bank for {mod.strip()} from {path.strip()}")
            proxy_features[mod.strip()] = torch.load(
                path.strip(), weights_only=False)
    proxy_key_strip_prefixes = (
        [s.strip() for s in args.proxy_key_strip_prefixes.split(",") if s.strip()]
        if args.proxy_key_strip_prefixes else None)

    # Build an answer vocabulary from the dataset for open-ended QA
    # formats. This keeps extraction aligned with the benchmark's actual
    # label space rather than the small default AVQA/MUSIC-AVQA vocab.
    answer_vocab = None
    if args.qa_format in {"videoinst", "videoinst_short", "simple_qa"}:
        vocab: set[str] = set()
        for sample in qa_data:
            refs = _reference_answers(sample, str(sample.get("answer", "")))
            for ans in refs:
                norm = normalize_answer(ans)
                if norm:
                    vocab.add(norm)
                    vocab.add(norm.replace(" ", "_"))
        if vocab:
            answer_vocab = vocab
            print(f"  answer_vocab: {len(answer_vocab)} entries from qa_data")

    # Run eval
    records = evaluate(
        model, tokenizer, qa_data, features,
        qa_format=args.qa_format,
        modalities=modalities,
        max_samples=args.max_samples,
        answer_vocab=answer_vocab,
        bridges=bridges if bridges else None,
        bridge_modalities=bridge_modalities,
        proxy_features=proxy_features or None,
        proxy_key_strip_prefixes=proxy_key_strip_prefixes,
        prompt_modalities_mode=args.prompt_modalities_mode,
    )

    metrics = compute_metrics(records)
    print(f"\n{'=' * 40}")
    print(f"Accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"Modalities: {modalities}")
    print(f"{'=' * 40}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "modalities": modalities,
                "prompt_modalities_mode": args.prompt_modalities_mode,
                "metrics": metrics,
                "records": records[:args.max_saved_records]
                           if args.max_saved_records > 0 else records,
            }, f, indent=2)
        print(f"Saved {len(records) if args.max_saved_records == 0 else min(len(records), args.max_saved_records)} records to {out}")


if __name__ == "__main__":
    main()
