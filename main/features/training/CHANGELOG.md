# Changelog

## 0.1.0 (2026-04-10)

- Initial release
- Tape-based reverse-mode autograd (GradientTape, BackwardOp)
- Neural network layers: Linear, Conv1d, LSTM, BatchNorm1d, LayerNorm, Dropout, Sequential
- Activations: ReLU, GELU, SiLU, Sigmoid, Tanh
- Backward ops: matmul, add, mul, relu, sigmoid, softmax, tanh
- Optimizers: SGD, Adam, AdamW with gradient clipping
- LR schedulers: StepLR, CosineAnnealingLR, WarmupCosineScheduler
- Loss functions: MSE, MAE, Huber, CrossEntropy, Quantile
