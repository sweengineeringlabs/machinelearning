# ADR-003: L2 Normalization in the Embedding Endpoint

**Status:** Accepted
**Date:** 2026-04-04

## Context

The `swellmd` daemon serves an OpenAI-compatible `/v1/embeddings` endpoint that returns raw embedding vectors from Nomic-BERT (768-dim, Mean pooling). These vectors are consumed by downstream services — primarily `swevecdb` (vector database) via the `swerag` RAG orchestrator.

Testing revealed that raw (unnormalized) embeddings produce compressed cosine similarity scores with poor separation:

```
"deep learning" vs "neural networks" (related):   0.90
"deep learning" vs "pizza on fridays" (unrelated): 0.87
Gap: 0.03
```

A gap of 0.03 is insufficient for reliable retrieval ranking. The root cause is that unnormalized vectors cluster in one region of the embedding space, compressing angular differences.

### Current embedding pipeline

```
swerag → swembed/swellmd (raw vectors) → swevecdb (stores as-is)
```

No service in the pipeline applies L2 normalization:

| Service | Normalizes? | Notes |
|---------|-------------|-------|
| swellmd `/v1/embeddings` | No | Returns raw model output |
| swembed (external) | No | Pass-through to model |
| swerag (orchestrator) | No | Treats vectors as opaque |
| swevecdb | Has `TransformType::Normalize` migration op, not applied on ingest | Manual, after-the-fact |

### What L2 normalization does

Scales each vector to unit length (magnitude = 1.0) while preserving direction:

```
v_normalized = v / ||v||₂

where ||v||₂ = sqrt(v₁² + v₂² + ... + vₙ²)
```

After normalization, cosine similarity equals the dot product (faster to compute) and scores spread across the full [-1, 1] range instead of clustering near 1.0.

### Alternative normalization methods considered

| Method | Formula | Produces | Fit for embeddings |
|--------|---------|----------|--------------------|
| **L2 (Euclidean)** | `v / sqrt(Σ vᵢ²)` | Unit vectors (magnitude = 1) | Standard for embedding search |
| L1 (Manhattan) | `v / Σ|vᵢ|` | Elements sum to 1 | Probability distributions, not embeddings |
| Max (L∞) | `v / max(|vᵢ|)` | Largest element = ±1 | Feature scaling, not search |
| Min-Max | `(vᵢ - min) / (max - min)` | Values in [0, 1] | Neural network inputs |
| Z-score | `(vᵢ - μ) / σ` | Mean=0, std=1 | Statistical analysis |

L2 is the industry standard for embedding models: Nomic, OpenAI, Cohere, Sentence-Transformers all specify L2-normalized output for cosine similarity search.

## Decision

Apply L2 normalization to embedding vectors in the `swellmd` `/v1/embeddings` endpoint before returning them to the caller.

### Why normalize at the embedding service (not downstream)

1. **Single responsibility**: The embedding service knows its model requires normalization. Downstream consumers should receive ready-to-use vectors.
2. **Correctness by default**: Every consumer gets normalized vectors without needing to know about the model's requirements.
3. **Performance**: Normalizing once at the source is cheaper than normalizing at every consumer or applying a retroactive migration in the vector database.
4. **Consistency with industry APIs**: OpenAI's `/v1/embeddings` returns normalized vectors. Matching this behavior ensures drop-in compatibility.

### Implementation

Add L2 normalization to the embeddings handler in `daemon/main/src/core/router.rs`, applied per-vector after pooling:

```rust
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}
```

Applied in the embeddings handler after `model.embed()` returns and before serializing the response.

### Scope

- Applies to `/v1/embeddings` only (not `/v1/chat/completions`)
- Always-on (no opt-out flag) — this matches OpenAI API behavior
- Zero-copy: normalizes the vector in place before serialization

## Consequences

### Positive

- **Better retrieval quality**: Cosine similarity scores spread across a meaningful range, improving ranking separation for RAG pipelines.
- **Drop-in OpenAI compatibility**: Consumers expecting normalized embeddings (standard practice) will work correctly.
- **No downstream changes needed**: swerag, swevecdb, and any other consumer receive ready-to-use vectors.
- **Negligible performance cost**: One pass over 768 floats (~3 microseconds) per embedding.

### Negative

- **Not configurable**: If a consumer specifically needs raw embeddings, they cannot opt out. This is acceptable because raw embeddings are rarely useful for search, and no current consumer requires them.

### Neutral

- **swevecdb's `TransformType::Normalize` migration**: Remains useful for retroactively normalizing vectors that were ingested before this change, or for vectors from other embedding sources.
- **Existing swembed service**: If swembed also starts normalizing, double-normalization is a no-op on unit vectors (idempotent). No harm if both normalize.

## References

- [Nomic Embed v1.5 documentation](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) — specifies L2 normalization for search tasks
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings) — returns normalized vectors
- swevecdb `normalize_vector()` — existing implementation in `vecdb/main/backend/features/client/client-python/src/core/numpy_conv.rs`
