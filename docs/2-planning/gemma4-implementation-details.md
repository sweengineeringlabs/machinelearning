# Implementation Plan - Gemma 4 Support

Implement text-only inference support for the Gemma 4 E2B model (2.3B effective parameters) in the `rustml` ecosystem. This involves significant architectural updates to support features like per-layer embeddings, KV sharing, and proportional RoPE.

## User Review Required

> [!IMPORTANT]
> Gemma 4 E2B requires ~3.9 GB of RAM in Q8_0 format due to its 35 Per-Layer Embedding (PLE) tables. This is a significant increase over Gemma 3 1B (~500 MB). Please confirm if this memory footprint is acceptable for your target environment.

## Proposed Changes

### 1. `rustml-nlp` (API & Configuration)
- **File:** `rustml/nlp/main/src/api/types.rs`
    - Update `ModelConfig` struct with Gemma 4 specific fields:
        - `layer_types: Option<Vec<String>>`
        - `global_head_dim: Option<usize>`
        - `num_kv_shared_layers: Option<usize>`
        - `hidden_size_per_layer_input: Option<usize>`
        - `vocab_size_per_layer_input: Option<usize>`
        - `use_double_wide_mlp: Option<bool>`
        - `rope_parameters: Option<HashMap<String, RopeParameters>>`
    - Add `HFGemma4Config` internal struct for parsing HuggingFace `config.json`.
    - Update `ModelConfig::from_json_value` to support `"gemma4"` and `"gemma4_text"` architectures.

### 2. `rustml-nn` (Core Infrastructure)
- **File:** `rustml/nn/main/src/core/rope.rs`
    - Update `RoPEFreqs` to support `partial_rotary_factor` (only rotating a fraction of head dimensions).
    - Add `with_partial_rotation` constructor to `RoPEFreqs`.
- **File:** `rustml/nn/main/src/core/kv_cache.rs`
    - Update `KVCache` to support layer-to-slot mapping.
    - Implement redirection logic so that shared layers read from/write to the source layer's cache slot.
- **File:** `rustml/nn/main/src/core/embedding.rs`
    - Implement `PerLayerEmbedding` struct: wraps a lookup `Embedding` and a `Linear` projection.
- **File:** `rustml/nn/main/src/core/transformer_block.rs`
    - Update `TransformerBlock` to optionally include a `PerLayerEmbedding` path.
    - Support skipping K/V projections when operating in shared-KV mode.

### 3. `rustml-nlp` (Model Implementation)
- **File:** `rustml/nlp/main/src/core/model.rs`
    - Add `ple_tables: Vec<PerLayerEmbedding>` to `LlmModel`.
    - Implement `from_pretrained_gemma4` constructor to handle complex weight mapping and construction.
    - Update `forward_pass` and `forward_with_cache_pass` to incorporate PLE lookup/add at each layer's start.
- **File:** `rustml/nlp/main/src/core/weight_map.rs`
    - Add `WeightMap::gemma4` to handle Gemma 4's tensor naming conventions (PLE, shared KV, etc.).

### 4. `rustml-gguf` (GGUF Support)
- **File:** `rustml/gguf/main/src/core/weight_map.rs`
    - Update GGUF metadata parsing to handle new Gemma 4 keys.

## Verification Plan

### Automated Tests
- **Unit Tests:**
    - `test_gemma4_config_parsing`: Verify all new fields are parsed correctly from JSON.
    - `test_partial_rope`: Compare `RoPEFreqs::with_partial_rotation` output against a manual scalar implementation.
    - `test_kv_sharing`: Verify that Layer 15 correctly reuses KV states from Layer 0.
    - `test_ple_lookup`: Verify shape and output of `PerLayerEmbedding` forward pass.
- **Integration Tests:**
    - `test_gemma4_tiny_forward`: Construct a tiny Gemma 4 model with random weights and verify end-to-end forward pass shapes.

### Manual Verification
- **Logit Comparison:** Compare top token predictions for a set of prompts against a reference HuggingFace `transformers` implementation (using a provided script).
- **Memory Audit:** Use `RUST_LOG=rustml=debug` to verify memory allocation for PLE tables and savings from KV sharing.
- **Performance Benchmarking:** Run `rustml-infer` with Gemma 4 E2B and measure tokens/sec.
