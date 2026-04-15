# Guide: Advanced Quality Metrics

Beyond simple rounding errors (MSE), professional AI engineers use these advanced metrics to ensure a model retains its "logic" and "personality" after quantization.

---

## 1. Cosine Similarity (The "Logic" Alignment)
- **What it is:** Measures the *angle* between the original weight vector and the quantized one.
- **Why it matters:** In neural networks, the **direction** of a weight vector is often more important than its exact size. 
    - Imagine a compass: if you change the length of the needle (magnitude), it still points North. But if you rotate the needle by even 5 degrees (direction), you'll get lost.
    - If quantization rotates a weight vector, the neuron might start firing for the wrong reasons, changing the model's logic.
- **Goal:** **> 0.999** (Higher is better).

## 2. Layer Sensitivity Ranking (The "Mixed Precision" Strategy)
- **What it is:** A comparison of which layers suffer the most "damage" during quantization.
- **Why it matters:** Not all layers are equal. The first layer (Embedding) and the last layer (Head) are usually very sensitive.
- **The Strategy:** By identifying the **Top 5% most sensitive layers**, you can keep them in high-precision (FP16) while quantizing the other 95% to 4-bit. This gives you the speed of a small model with the brains of a large one.

## 3. Activation-Aware Analysis (The "Stress Test")
- **What it is:** Measuring the outliers in the data that *flows through* the weights.
- **Why it matters:** Weights are static, but activations change based on what the model is "thinking" about. Some weights only become important when the model encounters specific complex words or logic.
- **The Strategy:** We scale weights based on how "stressed" they get during real inference. This is the core principle behind **AWQ (Activation-aware Weight Quantization)**.

## 4. KL Divergence (The "Confusion" Metric)
- **What it is:** Measures how much the "probability distribution" of the model's predictions has shifted.
- **Why it matters:** It tells you if the quantized model is becoming "uncertain" or "confused." If the original model was 99% sure the next word is "Apple," but the quantized model is only 60% sure, KL Divergence will detect that loss of confidence even if the model still chooses the right word.

---
*These metrics turn quantization from a "guessing game" into a precise science.*
