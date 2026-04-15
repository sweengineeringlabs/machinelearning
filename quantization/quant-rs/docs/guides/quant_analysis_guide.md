# Guide: Understanding Quantization Analysis

Welcome! If you are new to AI model optimization, this guide will help you understand how we analyze a model before and after shrinking it.

---

## 1. Why Analyze?
Quantization is the process of rounding high-precision numbers (32-bit) to low-precision ones (4-bit). Like rounding `$1.2345` to `$1.25`, you lose a little bit of accuracy. Analysis helps us measure exactly how much "intelligence" we lost.

## 2. Key Metrics
When you run our tool with the `--eval` flag, you will see three main numbers:

### **MSE (Mean Squared Error)**
- **What it is:** The average of the squares of the errors.
- **Interpretation:** **Lower is better.** An MSE of `0.0001` means the quantized model is very close to the original. An MSE of `0.1` means it is significantly distorted.

### **SNR (Signal-to-Noise Ratio)**
- **What it is:** The ratio of the original weight "signal" to the quantization "noise."
- **Interpretation:** **Higher is better.** 
    - **> 30dB:** Perfect. Human-eye/AI-logic can't tell the difference.
    - **20dB - 30dB:** Good. Standard for 4-bit models.
    - **< 15dB:** Poor. The model might start hallucinating or giving nonsensical answers.

## 3. Visualizing the Weights
Our analysis tool looks at the **Distribution** of weights. Most models follow a "Bell Curve" (Normal Distribution).
- **Outliers:** These are weights far away from zero (e.g., `50.0` when most are `0.01`).
- **Clipping:** If we round too aggressively, these outliers get "clipped" or crushed, which is the #1 cause of quality loss.

---
*Next Step: Read the [Block Size Fine-tuning Guide](./block_size_finetuning.md) to learn how to fix clipping issues.*
