# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`0-ideation` is the concept/prototype phase of the WeightScope frontend. It contains two files:

- `weights_analyzer.html` — a self-contained, single-file SPA prototype. No build system, no dependencies beyond Chart.js loaded from CDN. Open directly in a browser.
- `weightscope-openapi.yaml` — the target REST API spec (WeightScope v2.5.0) that the real frontend will eventually call.

The HTML file exists to explore UX and layout. All data in it is simulated — there are no real API calls.

## Prototype architecture

The HTML is one file: all CSS in `<style>`, all JS in `<script>`, all markup inline.

**Page model:** Five top-level pages toggled by `switchPage()`. Only one is visible at a time via `.page.active`. The pages are:

| Tab | ID | What it shows |
|---|---|---|
| Analysis | `page-analysis` | Weight distribution, heatmap, per-layer stats, anomaly list, quant format table |
| Quantize Engine | `page-quantize` | Per-layer format overrides, pipeline steps, before/after comparison, result |
| Tensor Details | `page-tensors` | 7 tensor sub-views (overview, embeddings, attention, FFN, vocab, norms, positional, LM head) |
| Embedding Query | `page-embedquery` | Similarity search, vector arithmetic, analogy, 2D projection |
| Quant Eval | `page-quanteval` | Perplexity benchmarks, task accuracy, speed/memory profiling, leaderboard |

Tensor Details has its own nested navigation via `switchTensor()` and `.tsub.active`.

**Mock data:** All model data is driven by the `MODELS` constant (lines ~1345–1362), which holds four models: GPT-2 Small (117M), GPT-2 Medium (345M), GPT-2 Large (774M), ViT-B/16. Layer counts, hidden dims, and param counts are taken from there. `generateLayerData(name)` and `buildHeatmap(name)` derive everything else from the model name via a seeded pseudo-random function `sr(seed)`.

**Embedding simulation:** `getEmbedding(word)` builds a 64-dim vector from a seeded hash. `injectSemantics()` then nudges specific words (king, queen, man, woman, etc.) to produce realistic cosine-similarity results. `VOCAB_WORDS` (~line 2299) is the fixed vocabulary the query operates over.

**Quantization simulation:** `startQuantization()` drives an animated pipeline (Load → Calibrate → Sensitivity → Quantize → Verify → Export) using `setTimeout` chains. No real computation happens.

## API spec structure

`weightscope-openapi.yaml` defines the real API the frontend will eventually integrate with:

- **Base URL:** `https://api.weightscope.ai/v2` (also staging and `localhost:8080`)
- **Auth:** Bearer JWT on every request
- **Rate limits:** 60 req/min (free), 600 (pro), unlimited (enterprise)
- **Async jobs:** Both quantization (`POST /models/{id}/quantize`) and evaluation (`POST /models/{id}/eval`) return a job ID immediately; status is polled via GET. Poll until `status` is `done` or `failed`.
- **Supported checkpoint formats:** `.pt`, `.pth`, `.bin`, `.safetensors`, `.gguf` (up to 50 GB via multipart)
- **Quantization formats enum:** `FP32 | FP16 | BF16 | INT8 | INT4 | NF4 | GPTQ-4 | AWQ-4 | BnB-4`
- **Similarity metrics enum:** `cosine | dot | euclidean | manhattan`
- **Projection methods enum:** `tsne | pca | umap`

The six resource groups map directly to the five UI tabs (Analysis and Tensors are split in the API but merged into two tabs in the UI).

## When moving from prototype to real frontend

The mock data functions (`MODELS`, `generateLayerData`, `getEmbedding`, `VOCAB_WORDS`, etc.) are the seam. Replacing each with a real `fetch()` to the corresponding OpenAPI endpoint is the migration path. The HTML's ID conventions (`statParams`, `layerTableBody`, `quantResult`, etc.) are the DOM contract that new rendering code must honour.
