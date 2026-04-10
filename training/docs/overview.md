# ML SDK Overview

Foundational ML training primitives built on `tensor-engine`.

## Autograd

Thread-local gradient tape records operations during forward pass. `tape::backward(&loss)` replays in reverse to compute gradients.

## Layers

All layers implement the `Layer` trait with `forward(&self, x: &Tensor) -> SwetsResult<Tensor>`.

## Optimizers

All optimizers implement `Optimizer` with `step(&mut self, params: &mut [Tensor])`.
