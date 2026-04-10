# Gemma 4 E2B Support — Implementation Plan

**Status:** In Progress (~75% complete)
**Date:** 2026-04-04 (updated 2026-04-06)
**Effort:** ~10 working days (6 elapsed, ~3 remaining)
**Risk:** Medium

---

## Context

Google released Gemma 4 on April 2, 2026. The E2B variant (2.3B effective params, 5.1B total with PLE tables) is the smallest model and a natural upgrade from Gemma 3 1B IT for swellmd.

Gemma 4 introduces six architectural changes over Gemma 3. This plan covers text-only inference support — vision and audio encoders are deferred to a later phase.

### Model Variants

| Model | Total Params | Effective Params | Type | Notes |
|-------|-------------|-----------------|------|-------|
| `google/gemma-4-E2B` | 5.1B | 2.3B | Dense | PLE inflates total param count |
| `google/gemma-4-E4B` | ~8B | ~4B | Dense | Larger dense |
| `google/gemma-4-26B-A4B` | 26B | 4B active | MoE | Mixture of Experts |
| `google/gemma-4-31B` | 31B | 31B | Dense | Full dense |

This plan targets E2B first. E4B follows for free (same architecture, larger dims). MoE (26B-A4B) requires additional `MoeLayer` routing but rustml already has Mixtral MoE support.

### Gemma 4 E2B Config (text model)

```
Architecture:    gemma4_text
Layers:          35
Hidden size:     1536
Attention heads: 8
KV heads:        1 (GQA)
Head dim:        256 (sliding), 512 (global)
Intermediate:    6144
Vocab:           262,144
Context:         131,072 tokens
Sliding window:  512 tokens
KV shared layers: 20 (last 20 layers reuse KV from earlier layers)
Activation:      gelu_pytorch_tanh (GeGLU)
Norm:            RMSNorm (eps=1e-6)
Embeddings:      Tied (token_embedding = lm_head)
```

---

## Architectural Changes from Gemma 3

### 1. Explicit Layer Type Array

**Gemma 3:** Pattern-based — `(i+1) % pattern == 0` determines global layers.

**Gemma 4:** Explicit array in config:

```json
"layer_types": [
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "full_attention",
    ... (repeats, every 5th layer is global, 35 layers total = 7 global layers)
]
```

**Impact:** Config parsing only. Replace `(i+1) % pattern == 0` with `layer_types[i] == "full_attention"`.

**Files:** `rustml-nlp/api/types.rs` (ModelConfig parsing)

### 2. Dual RoPE with Proportional Encoding (p-RoPE)

**Gemma 3:** Two RoPE configs (local_theta, global_theta), same application.

**Gemma 4:** Per-layer-type RoPE with new parameters:

```json
"rope_parameters": {
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10000.0
    },
    "full_attention": {
        "rope_type": "proportional",
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 0.25
    }
}
```

New concepts:
- **`partial_rotary_factor: 0.25`** — only rotate the first 25% of head dimensions. For `global_head_dim=512`, rotate dims 0-127, leave 128-511 unrotated.
- **`rope_type: "proportional"`** — scale position indices for long-context extrapolation.

**Impact:** Modify `RoPEFreqs` to support partial rotation. Add `partial_rotary_factor` field.

**Files:** `rustml-nn/core/rope.rs`, `rustml-nlp/api/types.rs`

### 3. Different Head Dimensions per Layer Type

**Gemma 3:** Uniform `head_dim` across all layers.

**Gemma 4:**

```json
"head_dim": 256,          // sliding_attention layers
"global_head_dim": 512    // full_attention layers
```

**Impact:** Per-layer head_dim in `MultiHeadAttention`. Q/K/V projection sizes differ between sliding and global layers. KV cache must allocate variable slot sizes per layer.

**Files:** `rustml-nn/core/attention.rs`, `rustml-nn/core/kv_cache.rs`, `rustml-nlp/core/model.rs`

### 4. KV Sharing Across Layers

**Gemma 3:** Every layer computes its own K, V projections.

**Gemma 4:**

```json
"num_kv_shared_layers": 20
```

The last 20 layers (layers 15-34) reuse KV states from earlier layers (layers 0-14) instead of computing their own K/V projections. Only Q projection and attention computation happen in shared layers.

```
Layer 0:  Q, K, V computed → KV stored in cache slot 0
Layer 1:  Q, K, V computed → KV stored in cache slot 1
...
Layer 14: Q, K, V computed → KV stored in cache slot 14
Layer 15: Q computed, K/V reused from cache slot 0   ← shared
Layer 16: Q computed, K/V reused from cache slot 1   ← shared
...
Layer 34: Q computed, K/V reused from cache slot 19  ← shared
```

**Impact:** Skip K/V linear projections for shared layers. KV cache reads from source layer index. Reduces memory by ~57% (20 of 35 layers share KV).

**Files:** `rustml-nn/core/kv_cache.rs`, `rustml-nn/core/attention.rs`, `rustml-nlp/core/model.rs`

### 5. Per-Layer Embeddings (PLE)

**Gemma 3:** Single shared token embedding.

**Gemma 4:**

```json
"vocab_size_per_layer_input": 262144,
"hidden_size_per_layer_input": 256
```

Each of the 35 transformer layers has its own embedding table `[262144, 256]` that maps token IDs to a 256-dim vector. This vector is projected to `hidden_size` (1536) and added to the hidden state at the start of each layer.

```
Per layer:
    ple_output = per_layer_embed[layer_i][token_ids]   → [B, S, 256]
    ple_proj   = linear_proj(ple_output)               → [B, S, 1536]
    hidden     = hidden + ple_proj
```

This is why total params (5.1B) >> effective params (2.3B) — the 35 PLE tables are `35 × 262144 × 256 × 4 bytes ≈ 9.4 GB` in F32, but each is just an O(1) lookup per token. In quantized formats (Q8_0), this shrinks to ~2.4 GB.

**Impact:** New `PerLayerEmbedding` struct. Load 35 embedding tables + projection weights. Add PLE lookup + add at each layer's forward pass.

**Files:** `rustml-nn/core/embedding.rs` (new PLE type), `rustml-nlp/core/model.rs`

### 6. Double-Wide MLP

**Gemma 3:** Standard GeGLU with `intermediate_size`.

**Gemma 4:**

```json
"use_double_wide_mlp": true
```

FFN intermediate dimension is doubled for all layers: `intermediate_size * 2 = 12288`.

**Impact:** Conditional intermediate size in `FeedForward::from_weights_geglu`.

**Files:** `rustml-nlp/core/model.rs` (construction), potentially `rustml-nn/core/feed_forward.rs`

---

## What We Skip (Phase 1 — Text Only)

| Component | Params | Deferred Because |
|-----------|--------|-----------------|
| Vision encoder (ViT) | ~150M | Requires image preprocessing pipeline |
| Audio encoder (conformer) | ~300M | Requires audio preprocessing pipeline |
| Multimodal token routing | — | Depends on vision/audio encoders |
| Thinking mode (`<|think|>`) | — | Works automatically if tokenizer handles control tokens |

Text-only inference covers the primary use case for swellmd chat completions.

---

## Reusable from Gemma 3 (~70%)

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| RMSNorm with offset | Yes | Same implementation |
| GeGLU FFN | Yes | Same structure, different size |
| QK normalization | Yes | Gemma 4 uses it — loading NOT yet implemented |
| Sliding window attention | Yes | Same mechanism, config-driven |
| Logit softcapping | Yes | `final_logit_softcapping: 30.0` still present |
| Dual RoPE structure | Partially | Need partial rotation, p-RoPE |
| Weight mapping pattern | Partially | New tensor names for PLE, shared KV |
| GGUF loader | Partially | Depends on llama.cpp's gemma4 support |

---

## Implementation Phases

### Phase 1: Config Parsing + Layer Types (Day 1) — DONE

**Goal:** Parse `gemma4_text` config into `ModelConfig`.

**Tasks:**
- [x] Add `layer_types: Option<Vec<String>>` to `ModelConfig`
- [x] Parse `rope_parameters` map (per-layer-type RoPE configs)
- [x] Parse `global_head_dim`, `num_kv_shared_layers`, `hidden_size_per_layer_input`, `vocab_size_per_layer_input`
- [x] Parse `use_double_wide_mlp`
- [x] Add `"gemma4" | "gemma4_text"` match arm in `ModelConfig::from_json_value()`
- [x] Merge `text_config` for multimodal wrapper configs
- [x] Unit tests for config parsing (`test_gemma4_config_from_json`)

**Files:**
- `llm/nlp/main/src/api/types.rs` (lines 219-231, 512-554, 680-714)

### Phase 2: Partial RoPE + Dual Head Dim (Days 2-3) — PARTIAL

**Goal:** Support `partial_rotary_factor` and per-layer head dimensions.

**Tasks:**
- [x] `RoPEFreqs::with_partial_rotation(head_dim, partial_factor, ...)` — only compute frequencies for first `head_dim * factor` dimensions
- [x] `RoPEFreqs::apply()` — rotate partial dims, pass through remaining dims unchanged
- [x] SIMD optimizations (AVX2/NEON) for rope_apply
- [x] Unit test: `test_partial_rope` (rope.rs:333-352)
- [ ] **GAP: Per-layer head dimensions not wired up** — `global_head_dim` parsed but attention built with uniform `head_dim`. Sliding layers (head_dim=256) and global layers (head_dim=512) must use different Q/K/V projection sizes.
- [ ] Verify Q projection output size matches per-layer head_dim

**Files:**
- `llm/nn/main/src/core/rope.rs` (lines 136-163)
- `llm/nn/main/src/core/attention.rs`

### Phase 3: KV Sharing (Days 3-5) — PARTIAL

**Goal:** Last N layers reuse KV states from earlier layers.

**Tasks:**
- [x] `KVCache` — `with_kv_sharing()` constructor with `layer_to_slot` mapping
- [x] `KVCache::get_slot_idx(layer_idx)` — maps virtual layer to physical slot
- [x] `KVCache::is_slot_owner(layer_idx)` — detects first writer to cache slot
- [x] `KVCache::get_view(layer_idx, len)` — retrieve cached KV for attention
- [x] `KVCache::update(layer_idx, key, value)` — store new KV in cache
- [ ] **GAP: No auto-setup from config** — no helper to generate `layer_to_slot` from `config.num_kv_shared_layers`. Users must manually call `with_kv_sharing()`.
- [ ] **GAP: Forward pass doesn't skip K/V projection for shared layers** — `forward_with_cache_pass` computes K/V for all layers regardless of sharing.
- [ ] Unit tests: verify KV reuse produces correct attention output

**Files:**
- `llm/nn/main/src/core/kv_cache.rs` (lines 1-213)
- `llm/nn/main/src/core/attention.rs`
- `llm/nn/main/src/core/transformer_block.rs`

### Phase 4: Per-Layer Embeddings (Days 5-6) — DONE

**Goal:** Each layer adds a per-layer embedding lookup to the hidden state.

**Tasks:**
- [x] `PerLayerEmbedding` struct — wraps `Embedding` [vocab, ple_dim] + `Linear` [ple_dim, hidden_size]
- [x] `PerLayerEmbedding::forward(token_ids)` — lookup + project + return
- [x] `LlmModel` — store `ple_tables: Vec<PerLayerEmbedding>` (one per layer)
- [x] Forward pass — add PLE output to hidden state before each transformer block
- [x] Unit test: `test_ple_lookup` (embedding.rs:129-133)

**Files:**
- `llm/nn/main/src/core/embedding.rs` (lines 89-122)
- `llm/nlp/main/src/core/model.rs` (lines 1553-1557, 1638-1642)

### Phase 5: Weight Loading + Constructor (Days 6-8) — PARTIAL

**Goal:** Load Gemma 4 weights from GGUF and SafeTensors.

**Tasks:**
- [x] `WeightMap::gemma4(n_layers)` — map internal names to Gemma 4 tensor names
- [x] Handle PLE tensor naming (HF: `model.language_model.layers.{i}.per_layer_projection.weight`)
- [x] `LlmModel::from_pretrained_gemma4(config, weights)` constructor
- [x] Wire into `build_safetensors_model()` with `"gemma4" | "gemma4_text"` dispatch
- [x] GGUF weight mapping: `gguf_gemma4_weight_map()` (extends gemma3 map with PLE entries)
- [ ] **GAP: QK normalization not loaded** — Gemma 3 constructor loads QK norms (model.rs:800-806) but `from_pretrained_gemma4` omits them entirely. Must load `q_norm.weight` and `k_norm.weight` per layer.
- [ ] **GAP: `use_double_wide_mlp` not wired** — config field parsed but never applied. FFN intermediate size should be doubled when `use_double_wide_mlp == true`.
- [ ] **GAP: GGUF config bridge missing Gemma 4** — `gguf_config_to_model_config()` in `gguf_bridge.rs` only branches for Gemma 3. Gemma 4 fields (layer_types, global_head_dim, num_kv_shared_layers, PLE dims) not extracted from GGUF metadata.
- [ ] **GAP: GGUF tensor names unverified** — `blk.{i}.ple_embd.weight` / `blk.{i}.ple_proj.weight` assumed but not confirmed against actual GGUF files from llama.cpp.
- [ ] `swellmd` loader: add gemma4 branch alongside gemma3/nomic-bert

**Files:**
- `llm/nlp/main/src/core/weight_map.rs` (lines 350-444)
- `llm/nlp/main/src/core/model.rs` (lines 898-1063, 1969-1973)
- `llm/nlp/main/src/core/gguf_bridge.rs` (lines 46-111 — needs update)
- `llm/gguf/main/src/core/weight_map.rs` (lines 167-182)

### Phase 6: Validation + Testing (Days 8-10) — NOT STARTED

**Goal:** Verify correctness and measure performance.

**Tasks:**
- [ ] Generate reference logits from HuggingFace `transformers` (Python)
- [ ] Compare top-5 token predictions for known prompts
- [ ] Interactive chat test via swellmd
- [ ] Performance benchmark: tok/s, memory usage
- [ ] Profile PLE lookup overhead
- [ ] Verify KV sharing memory savings vs Gemma 3
- [ ] Update docs: add Gemma 4 to supported models in deployment guide, operations guide, user manual

**Integration tests needed (currently missing):**
- [ ] `test_gemma4_model_construction` — build model from config + random weights
- [ ] `test_gemma4_forward_pass_shapes` — verify output shape through full forward
- [ ] `test_gemma4_forward_with_cache` — verify cached inference path
- [ ] `test_gemma4_kv_sharing_correctness` — shared layers produce same KV as source
- [ ] `test_gemma4_ple_integration` — PLE modifies hidden state at each layer
- [ ] `test_gemma4_gguf_config_bridge` — GGUF metadata → ModelConfig round-trip

**Acceptance Criteria:**
- [ ] `swellmd --safetensors google/gemma-4-E2B-it` starts and serves chat completions
- [ ] Logit comparison within tolerance (< 0.01 for same quantization)
- [ ] Coherent conversational responses
- [ ] Memory footprint documented
- [ ] All existing tests still pass (no regression)

---

## Architecture Diagram

```
token_ids [B, S]
    ↓
Token Embedding [262K, 1536]          (shared, tied with lm_head)
    ↓
hidden [B, S, 1536]
    ↓
┌─── 35x Transformer Block ──────────────────────────────────────┐
│                                                                 │
│  ┌─ Per-Layer Embedding (PLE) ──────────────────────┐          │
│  │  ple = per_layer_embed[i][token_ids] → [B,S,256] │          │
│  │  hidden += linear_proj(ple)          → [B,S,1536] │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  RMSNorm (pre-attention)                                       │
│  ↓                                                             │
│  if layer_types[i] == "sliding_attention":                     │
│  │  head_dim = 256                                             │
│  │  window = 512 tokens                                        │
│  │  RoPE: default, theta=10K, full rotation                   │
│  │                                                             │
│  elif layer_types[i] == "full_attention":                      │
│  │  head_dim = 512                                             │
│  │  no window (full context)                                   │
│  │  p-RoPE: proportional, theta=1M, 25% partial rotation      │
│  │                                                             │
│  if i >= (n_layers - num_kv_shared_layers):  [layers 15-34]   │
│  │  Q = q_proj(hidden)                                         │
│  │  K, V = reuse from KV cache[i - 20]     ← KV sharing      │
│  else:                                       [layers 0-14]    │
│  │  Q = q_proj(hidden)                                         │
│  │  K = k_proj(hidden)  → store in KV cache                   │
│  │  V = v_proj(hidden)  → store in KV cache                   │
│                                                                 │
│  Attention(Q, K, V) + logit_softcapping(30.0)                  │
│  ↓                                                             │
│  Residual add                                                  │
│  RMSNorm (pre-FFN)                                             │
│  ↓                                                             │
│  GeGLU FFN (intermediate = 12288 if double_wide, else 6144)   │
│  ↓                                                             │
│  Residual add                                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
RMSNorm (final)
    ↓
lm_head (tied embedding weights) → logits [B, S, 262K]
    ↓
Softcapping(30.0) → softmax → next token
```

---

## Memory Estimate (Gemma 4 E2B, Q8_0)

| Component | Size | Notes |
|-----------|------|-------|
| Token embedding | 262K × 1536 × 1 byte = ~384 MB | Q8_0, tied with lm_head |
| PLE tables (35 layers) | 35 × 262K × 256 × 1 byte = ~2.3 GB | Q8_0, O(1) lookups |
| PLE projections (35 layers) | 35 × 256 × 1536 × 1 byte = ~13 MB | Small |
| Transformer weights (15 unique) | ~800 MB | Q/K/V/O, FFN, norms |
| Transformer weights (20 shared) | ~400 MB | Q/O only, no K/V |
| KV cache (128K context) | ~varies | Depends on context length used |
| **Total model** | **~3.9 GB** | Q8_0 quantized |

For comparison, Gemma 3 1B is ~500 MB in Q8_0. The PLE tables dominate Gemma 4's memory. Q4_0 quantization would bring it to ~2.5 GB.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PLE memory exceeds available RAM | Medium | High | Support Q4_0 PLE tables; test on 16 GB machine |
| KV sharing indexing bugs | Medium | Medium | Thorough unit tests; compare with HF reference |
| GGUF weight names unknown | Medium | Medium | Wait for llama.cpp gemma4 support or inspect GGUF files directly |
| Partial RoPE correctness | Low | High | Numerical gradient check against HF reference |
| Gemma 4 config changes in HF | Low | Low | Pin to specific model revision |

---

## Dependencies

- **GGUF support:** Depends on `ggml-org/gemma-4-E2B-it-GGUF` or `unsloth/gemma-4-E2B-it-GGUF` having stable weight naming. Already available on HF.
- **SafeTensors:** `google/gemma-4-E2B-it` available, requires HF token (gated model).
- **No external library changes needed** — all work is within rustml crates.

---

## See Also

- [Gemma 4 HF Blog](https://huggingface.co/blog/gemma4) — Release announcement
- [google/gemma-4-E2B config.json](https://huggingface.co/google/gemma-4-E2B/raw/main/config.json) — Full model config
- [GPT-2 Architecture Guide](../3-design/guides/GPT-2.md) — Base model reference
- [ADR-001: Unified LlmModel](../3-design/adr/adr-001-unified-llmmodel-for-gpt2.md) — Config-driven model dispatch
- [Developer Guide](../4-development/developer_guide.md) — Crate architecture
