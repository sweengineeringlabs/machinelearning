# training-engine

Training engine — autograd, layers, optimizers, losses, and backward ops built on tensor-engine.

## Architecture

Built on `tensor-engine` for tensor math. Provides the training primitives that model crates (`timeseries`, `llm`) build on.

```
tensor-engine   (tensor math, SIMD)
    |
training-engine (autograd, layers, optimizers, losses)
    |
timeseries      (transformer, LSTM, TCN, N-BEATS)
```

## Components

| Module | Contents |
|--------|----------|
| `api/tape` | GradientTape, BackwardOp trait |
| `api/layer` | Layer trait |
| `api/loss` | Loss trait |
| `api/optim` | Optimizer, LRScheduler traits |
| `core/ops` | Backward ops: matmul, add, mul, relu, sigmoid, softmax, tanh |
| `core/nn` | Linear, Conv1d, LSTM, BatchNorm, LayerNorm, Dropout, Sequential, activations |
| `core/optim` | SGD, Adam, AdamW, gradient clipping, LR schedulers |
| `core/loss` | MSE, MAE, Huber, CrossEntropy, Quantile |
