# Inference Pipeline — How Text Generation Works

This guide traces a single prediction through the inference stack, mapping each step to the crate that implements it.

## Input → Output

Given input `"The cat sat"`, the model predicts the next token `" on"`.

## Pipeline

```mermaid
flowchart TD
    Input["&quot;The cat sat&quot;"]

    subgraph EMB["EMBEDDING · embedding/"]
        E1["Token IDs: [The]=0 [cat]=1 [sat]=2"]
        E2["Look up 3 rows from weight matrix"]
        E3["→ 3 dense vectors, each 768 floats"]
        E1 --> E2 --> E3
    end

    subgraph LAYER["TRANSFORMER LAYER × n_layers"]
        subgraph NORM1["NORMALIZATION · normalization/"]
            N1["RMSNorm: divide each vector by its RMS<br/>Stabilizes values before attention"]
        end

        subgraph ATTN["ATTENTION · inference/layers/"]
            A1["Q = input × W_q · what am I seeking?"]
            A2["K = input × W_k · what do I contain?"]
            A3["V = input × W_v · what is my value?"]

            subgraph ROPE["ROPE · inference/layers/rope.rs"]
                R1["Rotate Q and K by position angle"]
                R2["pos 0 → 0° · pos 1 → θ° · pos 2 → 2θ°"]
                R3["SIMD: AVX2 processes 8 floats/cycle"]
                R1 --> R2 --> R3
            end

            A4["Scores = Q × Kᵀ / √d"]
            A5["&quot;sat&quot; attends to:<br/>The → 0.1 · cat → 0.7 · sat → 0.2"]
            A6["Output = softmax(scores) × V"]

            A1 --> ROPE
            A2 --> ROPE
            ROPE --> A4 --> A5 --> A6
        end

        subgraph NORM2["NORMALIZATION · normalization/"]
            N2["RMSNorm again — stabilize before FFN"]
        end

        subgraph FFN["FEED-FORWARD · inference/layers/"]
            subgraph ACT["ACTIVATION · activation/"]
                F1["gate = SiLU(input × W_gate)<br/>silu(x) = x × sigmoid(x)"]
            end
            F2["up = input × W_up"]
            F3["out = (gate × up) × W_down"]

            subgraph QUANT["QUANTIZED KERNEL · quant/"]
                Q1["Weights stored as 4-bit integers<br/>SIMD dot product: Q4 × Q8<br/>4x less memory bandwidth"]
            end

            ACT --> F3
            F2 --> F3
            F3 --> QUANT
        end

        NORM1 --> ATTN --> NORM2 --> FFN
    end

    subgraph OUT["OUTPUT · inference/model/"]
        O1["hidden × embeddingᵀ → logits"]
        O2["logits[&quot; on&quot;] = 8.3 ← highest"]
        O3["→ predict &quot; on&quot;"]
        O1 --> O2 --> O3
    end

    Input --> EMB --> LAYER --> OUT
```

## Layer Count

Each model repeats the transformer layer block a fixed number of times:

| Model | Layers | Parameters |
|-------|--------|------------|
| GPT-2 small | 12 | 124M |
| Gemma 3 1B | 26 | 1B |
| Llama 2 7B | 32 | 7B |
| Mixtral 8x7B | 32 | 47B |

The layer count comes from `ModelConfig.n_layers`, read from the model's config at load time.

## Crate Mapping

```mermaid
graph LR
    subgraph Shared["Shared Primitives"]
        tensor["tensor/"]
        norm["normalization/"]
        act["activation/"]
        emb["embedding/"]
        hub["hub/"]
    end

    subgraph Inference["inference/"]
        layers["layers/"]
        model["model/"]
        gen["generation/"]
        quant["quant/"]
        gguf["gguf/"]
        tok["tokenizer/"]
        daemon["daemon/"]
        cli["cli/"]
    end

    tensor --> norm
    tensor --> act
    tensor --> emb
    tensor --> layers
    norm --> layers
    norm --> training
    act --> training
    emb --> model
    layers --> model
    hub --> model
    gguf --> model
    model --> gen
    tok --> gen
    model --> daemon
    gen --> daemon
    emb --> daemon

    training["training/"]
    arch["architectures/"]
    training --> arch
```

## Crate Reference

| Step | Crate | What it does |
|------|-------|-------------|
| Token → vector | `embedding/` | Row lookup from weight matrix |
| Normalize | `normalization/` | RMSNorm / LayerNorm math |
| Attention | `inference/layers/` | Q/K/V projection, scores, weighted sum |
| Position encoding | `inference/layers/` | RoPE with SIMD (AVX2/NEON) |
| Activation | `activation/` | SiLU, GELU pointwise nonlinearity |
| Quantized matmul | `inference/quant/` | SIMD dot products on 4/8-bit weights |
| KV cache | `inference/layers/` | Pre-allocated key/value buffers for decoding |
| Model assembly | `inference/model/` | Composes layers into LlmModel |
| Text generation | `inference/generation/` | Token-by-token loop, sampling, streaming |
| HTTP API | `inference/daemon/` | Serves /v1/completions and /v1/embeddings |
| Weight loading | `hub/` | Downloads from HuggingFace, loads SafeTensors |
| GGUF loading | `inference/gguf/` | Parses GGUF format model files |
| Tokenization | `inference/tokenizer/` | Text ↔ token IDs (BPE, SentencePiece, HF) |
