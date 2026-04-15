# Guide: Configuration & CLI Usage

This guide covers how to use the `quant-cli` and the available configuration options.

---

## 1. Basic Commands
To quantize a model, use the `quant-cli` crate:
```bash
cargo run -p quant-cli -- --input <INPUT_FILE> --output <OUTPUT_FILE> --format <FORMAT>
```

## 2. Configuration Options

### **Input/Output**
- `--input` (`-i`): Path to the original `.safetensors` file.
- `--output` (`-o`): Where to save the quantized version.

### **Quantization Formats (`--format`)**
- `nf4` (**Default**): NormalFloat 4-bit. Best for LLMs like Llama and Mistral.
- `int8`: Symmetric 8-bit. Good for mobile devices and non-transformer models.
- `fp16`: Half-precision. Reduces size by 50% with almost ZERO quality loss.

### **Mixed Precision (NEW)**
- `--protect-report <JSON_PATH>`: Use a report from `sweetspot-finder` to find sensitive layers.
- `--protect-count <N>`: How many of the most sensitive layers to keep in high precision (Default: 5).
- `--protect <LAYER_NAMES>`: A comma-separated list of specific layers to skip quantization.

### **Precision Controls**
- `--block-size` (`-b`): The number of weights per scale factor. 
- `--eval`: If present, calculate quality metrics (MSE/SNR/Cosine) for every layer.

## 3. Real-World Examples

### **Advanced Mixed Precision (Recommended for Production)**
1. Find sensitive layers:
```bash
cargo run -p sweetspot-finder -- -i model.safetensors -o report.json
```
2. Quantize while protecting the top 5 fragile layers:
```bash
cargo run -p quant-cli -- -i model.safetensors -o mixed_model.safetensors --protect-report report.json --protect-count 5
```

---
*Questions? Check the [Advanced Quality Metrics](./advanced_quality_metrics.md) for more on sensitivity.*
