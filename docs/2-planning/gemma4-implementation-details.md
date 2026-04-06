# Implementation Plan - Gemma 4 Support

**Status:** In Progress (~75% complete, updated 2026-04-06)

Implement text-only inference support for the Gemma 4 E2B model (2.3B effective parameters) in the `rustml` ecosystem. This involves significant architectural updates to support features like per-layer embeddings, KV sharing, and proportional RoPE.

## User Review Required

> [!IMPORTANT]
> Gemma 4 E2B requires ~3.9 GB of RAM in Q8_0 format due to its 35 Per-Layer Embedding (PLE) tables. This is a significant increase over Gemma 3 1B (~500 MB). Please confirm if this memory footprint is acceptable for your target environment.

## Implementation Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Config Parsing + Layer Types | **DONE** |
| 2 | Partial RoPE + Dual Head Dim | **PARTIAL** — head_dim per layer not wired |
| 3 | KV Sharing | **PARTIAL** — infrastructure done, auto-setup + forward skip missing |
| 4 | Per-Layer Embeddings (PLE) | **DONE** |
| 5 | Weight Loading + Constructor | **PARTIAL** — QK norms, double MLP, GGUF bridge missing |
| 6 | Validation + Testing | **NOT STARTED** |

## Remaining Work (Priority Order)

### P0 — Correctness Blockers

1. **Load QK norms in Gemma 4 constructor** (`model.rs`)
   - Gemma 3 loads `q_norm.weight` / `k_norm.weight` at lines 800-806 but `from_pretrained_gemma4` omits them
   - Without these, attention logits will diverge from reference
   - Fix: mirror Gemma 3's QK norm loading in the Gemma 4 constructor

2. **Wire `global_head_dim` for per-layer attention sizing** (`model.rs`)
   - Config parses `global_head_dim: 512` but all layers built with uniform `head_dim: 256`
   - Global (full_attention) layers need 512-dim Q/K/V projections
   - Fix: in `from_pretrained_gemma4`, select head_dim per layer based on `layer_types[i]`

3. **Honor `use_double_wide_mlp`** (`model.rs`)
   - Config field parsed but never read during FFN construction
   - Fix: apply `intermediate_size * 2` when building FeedForward for layers with this flag

### P1 — Functional Gaps

4. **Auto-generate KV cache sharing map from config** (`kv_cache.rs` or `model.rs`)
   - Add `KVCache::from_gemma4_config(config)` or equivalent helper
   - Computes `layer_to_slot` from `num_kv_shared_layers` and `n_layers`
   - Without this, users must manually construct the sharing map

5. **Skip K/V projection for shared layers in forward pass** (`model.rs`, `attention.rs`)
   - Layers 15-34 should only compute Q and reuse K/V from their source slot
   - Currently all layers compute Q, K, and V regardless
   - Fix: check `cache.is_slot_owner(layer_idx)` and branch

6. **GGUF config bridge for Gemma 4** (`gguf_bridge.rs`)
   - `gguf_config_to_model_config()` only handles Gemma 3
   - Add `is_gemma4` branch to extract: layer_types, global_head_dim, num_kv_shared_layers, PLE dimensions
   - Blocked partially on confirming GGUF metadata key names

### P2 — Verification

7. **Verify GGUF tensor names** against actual llama.cpp GGUF files
   - Current assumptions: `blk.{i}.ple_embd.weight`, `blk.{i}.ple_proj.weight`
   - May need to download and inspect a GGUF file to confirm

8. **Integration tests** (see Phase 6 in support plan)

9. **End-to-end logit comparison** against HuggingFace reference

## Proposed Changes (Original Plan — Annotated)

### 1. `rustml-nlp` (API & Configuration) — DONE

- **File:** `rustml/nlp/main/src/api/types.rs`
    - [x] Update `ModelConfig` struct with Gemma 4 specific fields:
        - `layer_types: Option<Vec<String>>`
        - `global_head_dim: Option<usize>`
        - `num_kv_shared_layers: Option<usize>`
        - `hidden_size_per_layer_input: Option<usize>`
        - `vocab_size_per_layer_input: Option<usize>`
        - `use_double_wide_mlp: Option<bool>`
        - `rope_parameters: Option<HashMap<String, RopeParameters>>`
    - [x] Add `RopeParameters` struct for per-layer-type RoPE configs (types.rs:8-15)
    - [x] Update `ModelConfig::from_json_value` to support `"gemma4"` and `"gemma4_text"` architectures (types.rs:512-554)
    - [x] Handle `text_config` merging for multimodal wrapper configs
    - [x] Unit test: `test_gemma4_config_from_json` (types.rs:680-714)

### 2. `rustml-nn` (Core Infrastructure) — PARTIAL

- **File:** `rustml/nn/main/src/core/rope.rs` — DONE
    - [x] `RoPEFreqs::with_partial_rotation` constructor (rope.rs:136-163)
    - [x] Partial rotation in `apply()` — rotate first N dims, pass through rest
    - [x] SIMD optimizations (AVX2/NEON)
    - [x] Unit test: `test_partial_rope` (rope.rs:333-352)
- **File:** `rustml/nn/main/src/core/kv_cache.rs` — PARTIAL
    - [x] `KVCache::with_kv_sharing()` constructor with `layer_to_slot` mapping
    - [x] `get_slot_idx()`, `is_slot_owner()`, `get_view()`, `update()` APIs
    - [ ] **Missing:** Auto-generate sharing map from `num_kv_shared_layers` config
- **File:** `rustml/nn/main/src/core/embedding.rs` — DONE
    - [x] `PerLayerEmbedding` struct: wraps `Embedding` + `Linear` projection (embedding.rs:89-122)
    - [x] Unit test: `test_ple_lookup` (embedding.rs:129-133)
- **File:** `rustml/nn/main/src/core/transformer_block.rs` — PARTIAL
    - [ ] **Missing:** Skipping K/V projections when operating in shared-KV mode

### 3. `rustml-nlp` (Model Implementation) — PARTIAL

- **File:** `rustml/nlp/main/src/core/model.rs`
    - [x] `ple_tables: Vec<PerLayerEmbedding>` on `LlmModel`
    - [x] `from_pretrained_gemma4` constructor (model.rs:898-1063)
    - [x] `forward_pass` and `forward_with_cache_pass` incorporate PLE lookup/add
    - [x] Per-layer RoPE selection based on `layer_types[i]`
    - [x] Wire into `build_safetensors_model()` dispatch (model.rs:1969-1973)
    - [ ] **Missing:** QK norm loading (Gemma 3 does it at model.rs:800-806, Gemma 4 skips it)
    - [ ] **Missing:** Per-layer `head_dim` vs `global_head_dim` — all layers use uniform head_dim
    - [ ] **Missing:** `use_double_wide_mlp` not applied to FFN intermediate size
- **File:** `rustml/nlp/main/src/core/weight_map.rs` — DONE
    - [x] `WeightMap::gemma4` maps HF tensor names to internal names (weight_map.rs:350-444)

### 4. `rustml-gguf` (GGUF Support) — PARTIAL

- **File:** `rustml/gguf/main/src/core/weight_map.rs` — DONE
    - [x] `gguf_gemma4_weight_map()` extends Gemma 3 map with PLE entries (weight_map.rs:167-182)
    - [ ] **Unverified:** GGUF tensor names (`blk.{i}.ple_embd.weight`) not confirmed against actual files
- **File:** `rustml/nlp/main/src/core/gguf_bridge.rs` — NOT DONE
    - [ ] **Missing:** `gguf_config_to_model_config()` has no Gemma 4 branch (only handles Gemma 3)

## Verification Plan

### Unit Tests (Status)
- [x] `test_gemma4_config_from_json`: All new fields parsed correctly from JSON (types.rs:680-714)
- [x] `test_partial_rope`: Partial rotation output verified (rope.rs:333-352)
- [x] `test_ple_lookup`: PLE forward pass shape/output verified (embedding.rs:129-133)
- [ ] `test_kv_sharing`: Verify that Layer 15 correctly reuses KV states from Layer 0
- [ ] `test_qk_norm_gemma4`: Verify QK normalization applied correctly in attention

### Integration Tests (Not Started)
- [ ] `test_gemma4_model_construction`: Build model from config + random weights, verify layer count and types
- [ ] `test_gemma4_forward_pass_shapes`: End-to-end forward, verify output shape `[B, S, vocab_size]`
- [ ] `test_gemma4_forward_with_cache`: Cached inference produces same logits as non-cached for first token
- [ ] `test_gemma4_kv_sharing_correctness`: Shared layers read from source layer's cache, not their own
- [ ] `test_gemma4_ple_integration`: PLE modifies hidden state differently at each layer
- [ ] `test_gemma4_gguf_config_bridge`: GGUF metadata → ModelConfig round-trip preserves all fields
- [ ] `test_gemma4_dual_head_dim`: Global layers use 512 head_dim, sliding use 256

### Manual Verification (Not Started)
- [ ] **Logit Comparison:** Compare top token predictions for a set of prompts against a reference HuggingFace `transformers` implementation (using a provided script).
- [ ] **Memory Audit:** Use `RUST_LOG=rustml=debug` to verify memory allocation for PLE tables and savings from KV sharing.
- [ ] **Performance Benchmarking:** Run `rustml-infer` with Gemma 4 E2B and measure tokens/sec.
