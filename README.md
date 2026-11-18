# Modality Self-Play

**MSP: Modality Self-Play for Training Multimodal LLMs Without Paired Cross-Modal Data**
<br>NeurIPS 2026

Robert Moseley¹ · Md Kaykobad Reza¹˒² · Ameya Patil² · Edward Ayrapetian² · M. Salman Asif¹
<br>¹University of California, Riverside &nbsp;&nbsp; ²Amazon

**[Project page](https://csiplab.github.io/Modality-Self-Play/)** ·
**[Paper (PDF)](https://arxiv.org/pdf/2610.20965)** · **[arXiv](https://arxiv.org/abs/2610.20965)** · **[Code](https://github.com/CSIPlab/Modality-Self-Play)**

---

## The idea

A model that has only ever seen images, and only ever heard sounds — never the
two together — can still learn to reason over both at once. It gets there by
writing down, in words, what the missing modality would be.

Teaching a model to reason across modalities normally means collecting data
where those modalities appear together: the same recording, seen and heard. That
data is expensive, hard to scale, and the cost grows combinatorially as
modalities are added. MSP removes that requirement.

During training every sample carries exactly one real modality — an image *or* a
sound, never both. For whatever is missing, the model is asked what it would be,
and its own answer is used in that slot. The description is text, so a frozen
contrastive text encoder can read it; a small frozen **bridge** then moves that
text embedding into the region of the space where real audio or real vision
features live, and the result enters the language model through the same
interface a real modality would use.

> **Why it works.** The training signal is *anchor consistency*, not literal
> reconstruction of the held-out scene. A proxy that is wrong about the specific
> scene but compatible with what the model actually observed still teaches the
> composition.

## How it works

| | | |
|---|---|---|
| **Phase 1** — Unimodal captioning | **Self-play** — Modality imagination | **Phase 2** — Composition training |
| Each modality gets its own projector into the language model, trained on ordinary modality–text pairs. Encoders stay frozen; projectors and LoRA adapters train. No paired cross-modal data. | The frozen Phase-1 model is shown one real modality and asked what the other would be. Its answer becomes the text proxy. Generated once, then cached. | One real modality plus a proxy for the missing one. Only the proxy passes through the bridge. Never two real non-text modalities. |

**The bridge.** The text-encoder and modality-encoder outputs of a contrastive
encoder lie in related but shifted regions of a shared space — the *modality
gap*. MSP fits a linear map by ridge regression on 1,000 caption–modality
samples (500 train / 500 held out), freezes it, and uses it thereafter:

```
W_m = (TᵀT + λI)⁻¹ Tᵀ R        B_m(t) = W_mᵀ t        λ = 1.0
```

**One real anchor.** Some source datasets do contain both modalities. MSP never
uses both halves of such a sample: each one is assigned to *either* its audio
*or* its vision, and is never revisited through the other.

## Results

Zero-shot throughout — MSP is never fine-tuned on these benchmarks, and their
training splits are never used.

| MUSIC-AVQA | Params | Acc. |
|---|---|---|
| OneLLM | 7B | 47.60 |
| Qwen2.5-VL | 7B | 47.75 |
| GPT-4o | — | 50.08 |
| **MSP (r = 512)** | 8B | **65.65** |

| AVQA | Params | Acc. |
|---|---|---|
| ImageBind-LLM *(zero-shot)* | 7B | 55.00 |
| HGA *(trained on AVQA)* | — | 82.20 |
| HCRN *(trained on AVQA)* | — | 82.50 |
| **MSP (r = 256)** | 8B | **85.73** |

| OmniBench | Params | Acc. |
|---|---|---|
| Baichuan-Omni-1.5 | 7B | 42.90 |
| Gemini-1.5-Pro | — | 42.91 |
| **MSP (r = 256)** | 8B | **43.26** |
| Qwen2.5-Omni | 7B | 56.13 |

**What each piece contributes** (audio–video, r = 512, validation splits):

| Configuration | AVQA | MUSIC-AVQA |
|---|---|---|
| Without proxy and bridge | 70.21 | 45.75 |
| With proxy, without bridge | 82.91 *(+12.70)* | 58.60 *(+12.85)* |
| **With proxy and bridge** | **85.58** *(+2.67)* | **65.65** *(+7.05)* |

## What MSP does and does not assume

**Removed:** the need for real paired non-text supervision — audio and vision
originating from the same underlying source.

**Still relied upon:** modality–text pairs, pretrained contrastive encoders, and
modality-distribution bridge calibration.

We therefore do not claim the absence of cross-modal priors. Contrastively
pretrained encoders already carry transferable cross-modal structure, and MSP
leans on it deliberately.

**Limitations.** MSP depends on the quality and semantic consistency of its
proxies; noisy or weakly aligned proxies degrade performance. It inherits the
coverage limitations and biases of the contrastive encoders it builds on. And
proxies are a training-time scaffold — real modalities still outperform their
proxy counterparts at inference.

## Repository

```
index.html    project page (static, no build step)
assets/       figures, animation, qualitative frames
src/          the method: encoders, projectors, bridges, self-play, training, evaluation
configs/      model and training configuration
tests/        unit tests
```

## Code layout

```
src/
  models/       encoders, per-modality MLP projectors, LLM wrapper
  data/         datasets and feature precomputation
  selfplay/     proxy generation - the model describes the modality it cannot observe
  training/     Phase 1 and Phase 2 training loops
  analysis/     bridge fitting and the bridge modules
  evaluation/   AVQA, MUSIC-AVQA, OmniBench and composition evaluation
configs/        model and training configuration
tests/          unit tests
```

Every runnable module is a `python -m` entry point with `--help`.

## Pipeline

The stages below correspond to the three panels of Figure 1 in the paper.

**1 — Precompute features.** The encoders (MetaCLIP ViT-B/16 for vision,
LAION-CLAP for audio) are frozen, so every modality feature is computed once as
a single 512-d vector and cached.

```bash
python -m src.data.precompute        --help   # vision
python -m src.data.precompute_audio  --help   # audio
```

**2 — Phase 1: per-modality projectors.** Each modality gets its own MLP
projector (512 -> 4096), trained on modality-text pairs only -- AudioCaps and
WavCaps for audio, LLaVA-665K for vision. No paired cross-modal data is used.
Each batch carries features from exactly one modality, and the two modalities
alternate. For the first `--projector_warmup_steps` optimizer steps (2,000 by
default) only the projectors are updated; the LoRA parameters are unfrozen
afterwards.

```bash
python -m src.training.train_multimodal --phase 1 --modality vision \
    --qa_data <llava-665k.json> --qa_format llava \
    --features_path <vision_features.pt>

python -m src.training.train_multimodal --phase 1 --modality audio \
    --qa_data <audiocaps.jsonl> --qa_format audiocaps \
    --features_path <audio_features.pt>
```

**3 — Generate self-play proxies.** The frozen Phase-1 model is conditioned on
the real anchor and prompted to describe the modality it cannot observe. There
is no external captioner: the same model that learned vision writes the audio
description. Proxies are generated once and cached.

```bash
python -m src.selfplay.imagine_modality     --help   # anchor -> proxy text
python -m src.selfplay.encode_imaginations  --help   # proxy text -> features
```

**4 — Fit the bridges.** One linear ridge map per modality, fit in closed form
on 1,000 calibration pairs (500 fit / 500 held out) and frozen thereafter.

```bash
python -m src.analysis.fit_offline_bridges --modalities audio,vision \
    --output_dir outputs/offline_bridges
```

**5 — Phase 2: composition training.** One real anchor plus a bridged proxy for
the missing modality. Only the proxy passes through the bridge. Modality
dropout (`--modality_dropout_prob`, 0.15 by default) zeroes a non-anchor
feature now and then so absent modalities are not treated as malformed input;
the anchor is never dropped.

```bash
python -m src.training.train_composition --help
```

`train_composition` enforces the one-real-anchor rule: a source sample that
happens to contain both modalities is assigned to one of them and is never
revisited through the other.

**6 — Evaluate.** MUSIC-AVQA, AVQA and OmniBench are evaluation benchmarks
only: MSP is never trained on them and their training splits are never used.
The training entry points reject these formats outright.

```bash
python -m src.data.precompute_avqa               --help   # benchmark features
python -m src.evaluation.eval_composition        --help   # MUSIC-AVQA, AVQA
python -m src.evaluation.eval_crossmodal_holdout --help
python -m src.evaluation.eval_multimodal         --help
```

## Setup

Requires **Python 3.10 or newer** (the code uses PEP 604 `X | None` annotations).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU for training. Paths to datasets, checkpoints and output
directories are passed as arguments or set in `configs/` — nothing is hardcoded.

## What is not in this repository

- **Datasets.** AVQA, MUSIC-AVQA, OmniBench, AudioCaps, WavCaps, Clotho-AQA,
  VQAv2 and LLaVA-665K must be obtained from their original sources under their
  own licences.
- **Checkpoints and cached features.** Not distributed here.
- **Cluster job scripts.** The original scheduler scripts were specific to our
  compute environment and are not included. Entry points above take plain
  arguments and run anywhere.
- **Dataset preparation.** The scripts that assemble the QA files and extract
  raw media -- including uniformly sampling four frames per video and
  mean-pooling them into a single vision token -- are not included. The
  entry points above consume the resulting QA JSON and cached feature files.
- **Analysis and figure code.** The scripts that produced the paper's tables,
  plots and mechanism probes are not included. This repository is the method
  and the training and evaluation pipeline only.

## Notes

- The code covers the audio and vision pathways reported in the paper. Other
  modalities explored during development are not included.

## Licence

Released under the MIT Licence - see [LICENSE](LICENSE).

Third-party model weights and datasets used with this code (MetaCLIP,
LAION-CLAP, Qwen3, AVQA, MUSIC-AVQA, OmniBench, AudioCaps, WavCaps, Clotho-AQA,
VQAv2, LLaVA-665K) carry their own licences and are not covered by this one.

## Citation

```bibtex
@inproceedings{moseley2026msp,
  title     = {{MSP}: Modality Self-Play for Training Multimodal {LLM}s
               Without Paired Cross-Modal Data},
  author    = {Moseley, Robert and Reza, Md Kaykobad and Patil, Ameya and
               Ayrapetian, Edward and Asif, M. Salman},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

This work is partially supported by a gift award from Amazon and an NSF award 2406199.
This work used Anvil at Purdue University through allocation ELE260028 from the Advanced
Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is
supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307,
#2137603, and #2138296.

Ameya Patil and Edward Ayrapetian are employed by Amazon. The authors declare no other
competing interests.

Qualitative examples are drawn from the AVQA and MUSIC-AVQA benchmarks, which
carry their own licences.

