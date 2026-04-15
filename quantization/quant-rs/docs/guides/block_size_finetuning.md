# Guide: Block Size Fine-tuning

This guide explains how to find the "Sweet Spot" for your model's quantization settings.

---

## 1. What is Block Size?
Imagine you are adjusting the volume on a 2-hour audio recording.
- **Global Scaling:** You set one volume for the entire 2 hours. If there's one loud scream, the whole recording becomes too quiet to hear.
- **Block-wise (e.g., 128):** You adjust the volume every few seconds. If there's a scream, you only lower the volume for that specific moment.

In our tool, `--block-size 128` means we divide the weight tensor into chunks of 128 numbers. Each chunk gets its own "Scale Factor."

## 2. The Trade-off
- **Smaller Blocks (e.g., 32):** 
    - ✅ **Higher Accuracy:** Can handle outliers perfectly.
    - ❌ **Larger File Size:** You have to save 4x more "Scale Factors" than with block size 128.
- **Larger Blocks (e.g., 256):**
    - ✅ **Smallest File Size:** Very few scale factors to save.
    - ❌ **Lower Accuracy:** One big outlier can "blind" 256 other weights.

## 3. Finding the "Sweet Spot" (Grid Search)
To find the perfect setting, engineers perform a **Grid Search**. This means running the quantization multiple times with different block sizes (e.g., 32, 64, 128, 256).

### **The Pareto Frontier**
When you plot the results (SNR vs. File Size), you get a curve. The **Sweet Spot** is the "knee" of that curve—the point where you get the most quality for the least amount of extra space.

### **How to do this automatically:**
Use our `sweetspot-finder` tool:
```bash
cargo run -p sweetspot-finder -- --input model.safetensors --sweep "32,64,128,256"
```
It will automatically recommend the best setting for your specific model!

---
*Next Step: See the [Configuration Guide](./configuration.md) for all available options.*
