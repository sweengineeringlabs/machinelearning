# WeightScope — Rust Quantization Pipeline

A professional-grade, high-performance toolkit for AI model quantization and quality auditing.

---

## 🏗️ Pipeline Architecture (SEA Compliant)

| Crate | Pipeline Stage | Role |
| :--- | :--- | :--- |
| **`quant-api`** | **Foundation** | Common interfaces, traits, and error handling. |
| **`quant-io`** | **Load / Save** | Zero-copy Safetensors loading and GGUF audit support. |
| **`quant-packer`** | **Bit-Packing** | Low-level bit-shifting and 4-bit packing logic. |
| **`quant-engine`** | **Algorithm** | High-level scaling math (NF4, Int8, Q4_0). |
| **`quant-eval`** | **Evaluate** | Quality metrics: SNR, Cosine Similarity, and KL Divergence. |
| **`quant-cli`** | **App** | Unified command-line interface for model optimization. |
| **`sweetspot-finder`**| **Research** | Automated grid search for finding optimal block sizes. |

---

## 🚀 Quick Start

### 1. Find the SweetSpot for your model
```bash
cargo run -p sweetspot-finder -- --input model.safetensors --sweep "32,64,128,256"
```

### 2. Mixed-Precision Quantization (Best Quality)
```bash
cargo run -p quant-cli -- -i model.safetensors -o optimized.safetensors --format nf4 --protect-report report.json
```

### 3. Audit a GGUF model
```bash
cargo run -p quant-cli -- -i model.gguf --reference model.safetensors
```

---
*Developed with Rust and Candle.*
