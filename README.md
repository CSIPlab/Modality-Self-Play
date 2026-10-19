# Modality Self-Play

**MSP: Modality Self-Play for Training Multimodal LLMs Without Paired Cross-Modal Data**
<br>NeurIPS 2026

Robert Moseley¹ · Md Kaykobad Reza¹˒² · Ameya Patil² · Edward Ayrapetian² · Salman Asif¹
<br>¹University of California, Riverside &nbsp;&nbsp; ²Amazon

**[Project page](https://csiplab.github.io/Modality-Self-Play/)** ·
**[Paper (PDF)](https://arxiv.org/pdf/2610.20965)** · **[arXiv](https://arxiv.org/abs/2610.20965)** · Code *(coming soon)*

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
```

Code release to follow.

## Citation

```bibtex
@inproceedings{moseley2026msp,
  title     = {{MSP}: Modality Self-Play for Training Multimodal {LLM}s
               Without Paired Cross-Modal Data},
  author    = {Moseley, Robert and Reza, Md Kaykobad and Patil, Ameya and
               Ayrapetian, Edward and Asif, Salman},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

This work is partially supported by a gift award from Amazon and an NSF award 2406199.

Qualitative examples are drawn from the AVQA and MUSIC-AVQA benchmarks, which
carry their own licences.
