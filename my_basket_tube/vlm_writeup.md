# BasketTube VLM — Development and Usage

## Overview

BasketTube includes two parallel vision-language pipelines for analyzing basketball broadcast footage. Both pipelines share a common foundation: a Whisper medium transcription of game commentary producing 1,153 timestamped subtitle segments aligned to 30fps video frames.

---

## Track 1 — Visual Action Retrieval

### Motivation

The goal of Track 1 is to retrieve video moments that visually match a natural language query — for example, "Curry hits a three pointer" — without relying on commentary text at retrieval time. This requires learning a shared embedding space between video frames and natural language.

### Architecture

```
Frame t-start ┐
Frame t-mid   ├→ frozen CLIP ViT-L/14 (1024-dim CLS each) → concat (3072-dim) → ProjectionLayer → 768-dim
Frame t-end   ┘
                                                              ↕ CosineEmbeddingLoss
                                              CLIP text encoder (subtitle text) → 768-dim
```

**Key design decisions:**

- **CLIP ViT-L/14** serves as the frozen vision backbone. Its vision encoder outputs a 1024-dimensional CLS token per frame; its text encoder outputs a 768-dimensional embedding. Both encoders share a joint semantic space by design.
- **Three frames** are sampled per subtitle segment: the start, midpoint, and end of a context window centered on the subtitle timestamp. This captures temporal context without the computational cost of processing all frames in the window.
- **ProjectionLayer** is a two-layer MLP (3072→768) trained to project the concatenated three-frame visual representation into the CLIP text embedding space.
- **CosineEmbeddingLoss** is used as the training objective, pushing visual embeddings toward the CLIP text embedding of the corresponding subtitle.

### Context Window Sweep

The parameter `CONTEXT_SECONDS` controls the spread of the three-frame window. Five models were trained across context windows of 3, 5, 10, 15, and 20 seconds, allowing comparison of how much surrounding context benefits retrieval quality.

| Context | Interpretation |
|---|---|
| 3s | Tight — captures the immediate action |
| 5s | Moderate — captures action plus immediate context |
| 10–20s | Wide — captures broader game state |

### Training

- **Dataset:** 1,153 subtitle segments from a single Lakers vs. Warriors game
- **Optimizer:** AdamW with linear warmup schedule
- **Batch size:** 8
- **Training:** 9 epochs total (3 at lr=1e-4, 6 at lr=1e-5)
- **Results:** Cosine similarity improved from ~0.0 (random) to 0.54–0.64; loss plateaued at ~0.408

The plateau is consistent with the ceiling expected from a single-game dataset of 1,153 samples.

### Retrieval at Query Time

At inference, a user text query is embedded via the CLIP text encoder and compared against the precomputed visual embedding index using cosine similarity. The top-K matching segments are returned along with an M3U playlist for zero-quality-loss VLC playback.

---

## Track 2 — Commentary-Based Q&A

### Motivation

Track 2 answers natural language questions about player performance using commentary as evidence. The initial implementation uses a RAG (Retrieval-Augmented Generation) architecture: retrieve relevant subtitle segments, then synthesize an answer. The goal of the VLM fine-tuning stage is to replace the API-based synthesis with a fully local model.

### Phase 1 — RAG with Sonnet

The baseline system:

1. CLIP text encoder embeds all 1,153 subtitle segments → precomputed index
2. User query → CLIP text encoder → top-K nearest subtitle chunks
3. Retrieved chunks → `claude-sonnet-4-6` → structured 4-section response

The structured response format enforced by the system prompt:
1. **DIRECT RESPONSE** — factual statement of what is happening
2. **SUMMARY** — player action and game context
3. **EVIDENCE** — 2–5 observations with timestamps citing visual or commentary source
4. **DISCLAIMER** — note that analysis is limited to 3 frames and one commentary segment

### Phase 2 — Dataset Generation (Teacher)

To eliminate API costs at inference time, `claude-sonnet-4-6` was used as a teacher to generate a training dataset. For each of the 1,153 subtitle segments:

- Three frames were sampled (start/mid/end of a 5-second context window)
- All three frames were base64-encoded and sent to Sonnet alongside the subtitle text
- The full game commentary transcript was provided as cached context in a prefilled assistant turn
- Sonnet's structured 4-section response was stored as the training target

The full commentary transcript was loaded as a cached first user turn using Anthropic's prompt caching (`cache_control: ephemeral`), reducing per-request costs significantly. The dataset was generated incrementally with save-after-every-entry for safe resumption.

**Dataset statistics:**
- 1,153 entries
- Each entry: frame paths (×3), subtitle text, timestamp, Sonnet response
- Approximate generation cost: ~$20–25

### Phase 3 — PaliGemma Fine-tuning (Student)

**Model:** `google/paligemma-3b-pt-224` — a 3B parameter vision-language model from Google.

**Approach:** QLoRA (Quantized Low-Rank Adaptation)
- 4-bit NF4 quantization via bitsandbytes keeps the model within 16GB VRAM
- LoRA adapters trained on attention layers only (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- LoRA rank r=16, alpha=32

**Input construction:**
The three sampled frames are concatenated horizontally into a single wide image before being passed to PaliGemma, which expects a single image input. The prompt includes the commentary segment text and timestamp.

**Training configuration:**
- Batch size: 1 (VRAM constraint)
- Gradient checkpointing + `enable_input_require_grads()` to enable backprop through frozen layers
- 3 epochs, lr=2e-4

**Training results:**

| Epoch | Loss |
|---|---|
| 1 | 1.5849 |
| 2 | 1.2659 |
| 3 | 1.1241 |

Consistent downward trend across all three epochs indicates the model is learning the structured response format.

### Inference

At query time (`player_performance_vlm.py`):

1. CLIP retrieves the top-K most visually similar subtitle segments
2. For each retrieved segment, three frames are sampled and concatenated
3. PaliGemma generates a structured response locally — no API calls
4. Results are cached to `query_cache_vlm.json`
5. An M3U playlist is generated for VLC playback

**Run command:**
```bash
BNB_CUDA_VERSION=128 python my_basket_tube/inference/player_performance_vlm.py
```

---

## Hardware and Environment Notes

- **GPU:** NVIDIA RTX 3080 16GB
- **CUDA:** 13.0 (no bitsandbytes pre-compiled binary — use `BNB_CUDA_VERSION=128`)
- **Key version pins:** `transformers==4.44.0`, `accelerate==0.34.0`
- **HuggingFace:** PaliGemma is a gated model requiring license acceptance and a fine-grained token with "Access public gated repositories" enabled

---

## Summary

| Component | Approach | Status |
|---|---|---|
| Transcription | Whisper medium (local) | Complete |
| Visual retrieval VLM | CLIP + trained ProjectionLayer | Complete — 5 models |
| Commentary RAG (API) | CLIP retrieval + Sonnet synthesis | Complete |
| Training dataset | Sonnet teacher, 1153 entries | Complete |
| Local synthesis VLM | PaliGemma-3B QLoRA | Complete — 3 epochs |
| Local query UI | player_performance_vlm.py | Complete |
