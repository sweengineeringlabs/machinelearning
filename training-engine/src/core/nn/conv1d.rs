use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// 1-D convolutional layer (FR-304, FR-111).
///
/// Applies a 1-D convolution over an input signal of shape
/// `[batch, in_channels, length]`, producing output of shape
/// `[batch, out_channels, out_length]` where
/// `out_length = (length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`.
///
/// Xavier initialization for weights, zeros for bias. Both require gradients.
pub struct Conv1d {
    weight: Tensor,   // [out_channels, in_channels, kernel_size]
    bias: Tensor,     // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    /// Creates a new Conv1d layer with stride=1, padding=0, dilation=1.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        // Xavier uniform initialization: scale = sqrt(6 / (fan_in + fan_out))
        let fan_in = in_channels * kernel_size;
        let fan_out = out_channels * kernel_size;
        let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();

        let mut weight = Tensor::randn([out_channels, in_channels, kernel_size]);
        weight = weight.mul_scalar_raw(scale);
        weight.set_requires_grad(true);

        let mut bias = Tensor::zeros([out_channels]);
        bias.set_requires_grad(true);

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
        }
    }

    /// Sets the stride and returns self (builder pattern).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Sets the padding and returns self (builder pattern).
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the dilation and returns self (builder pattern).
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn dilation(&self) -> usize {
        self.dilation
    }
}

impl Layer for Conv1d {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let input_shape = input.shape().to_vec();
        assert_eq!(
            input_shape.len(),
            3,
            "Conv1d input must be 3-D [batch, in_channels, length], got {:?}",
            input_shape
        );

        let batch = input_shape[0];
        let in_ch = input_shape[1];
        let length = input_shape[2];

        assert_eq!(
            in_ch, self.in_channels,
            "Conv1d: input channels {} != expected {}",
            in_ch, self.in_channels
        );

        let out_length = (length + 2 * self.padding
            - self.dilation * (self.kernel_size - 1)
            - 1)
            / self.stride
            + 1;

        let input_data = input.to_vec();
        let weight_data = self.weight.to_vec();
        let bias_data = self.bias.to_vec();

        let mut output_data = vec![0.0f32; batch * self.out_channels * out_length];

        // Forward convolution: element-wise implementation
        // input layout:  [batch, in_channels, length]
        // weight layout: [out_channels, in_channels, kernel_size]
        // output layout: [batch, out_channels, out_length]
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for o in 0..out_length {
                    let mut sum = bias_data[oc];

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let input_pos =
                                (o * self.stride + k * self.dilation) as isize
                                    - self.padding as isize;

                            if input_pos >= 0 && (input_pos as usize) < length {
                                let input_idx = b * in_ch * length
                                    + ic * length
                                    + input_pos as usize;
                                let weight_idx = oc * self.in_channels * self.kernel_size
                                    + ic * self.kernel_size
                                    + k;
                                sum += input_data[input_idx] * weight_data[weight_idx];
                            }
                        }
                    }

                    let output_idx =
                        b * self.out_channels * out_length + oc * out_length + o;
                    output_data[output_idx] = sum;
                }
            }
        }

        let output =
            Tensor::from_vec(output_data, vec![batch, self.out_channels, out_length])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(Conv1dBackward {
                    stride: self.stride,
                    padding: self.padding,
                    dilation: self.dilation,
                    in_channels: self.in_channels,
                    out_channels: self.out_channels,
                    kernel_size: self.kernel_size,
                    input_shape: input_shape.clone(),
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.weight.id(), self.bias.id()],
                saved_tensors: vec![input.clone(), self.weight.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Combined backward for Conv1d: handles input, weight, and bias gradients.
///
/// saved[0] = input, saved[1] = weight
/// input_ids[0] = input, input_ids[1] = weight, input_ids[2] = bias
///
/// Gradients:
/// - grad_bias = sum of grad_output over batch and spatial dims
/// - grad_weight[oc, ic, k] = sum over (b, o) of grad_output[b, oc, o] * input[b, ic, o*stride + k*dilation - padding]
/// - grad_input[b, ic, p] = sum over (oc, k) of grad_output[b, oc, o] * weight[oc, ic, k]
///   where o = (p + padding - k*dilation) / stride (when evenly divisible and in bounds)
struct Conv1dBackward {
    stride: usize,
    padding: usize,
    dilation: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    input_shape: Vec<usize>,
}

impl BackwardOp for Conv1dBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let weight = &saved[1];

        let batch = self.input_shape[0];
        let in_ch = self.input_shape[1];
        let length = self.input_shape[2];

        let grad_shape = grad_output.shape().to_vec();
        let out_length = grad_shape[2];

        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();
        let weight_data = weight.to_vec();

        // --- grad_bias: sum of grad_output over batch and spatial dimensions ---
        // grad_bias[oc] = sum over (b, o) of grad_output[b, oc, o]
        let mut grad_bias_data = vec![0.0f32; self.out_channels];
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for o in 0..out_length {
                    let idx = b * self.out_channels * out_length + oc * out_length + o;
                    grad_bias_data[oc] += grad_data[idx];
                }
            }
        }

        // --- grad_weight: transposed convolution of input with grad_output ---
        // grad_weight[oc, ic, k] = sum over (b, o) of
        //     grad_output[b, oc, o] * input[b, ic, o*stride + k*dilation - padding]
        let mut grad_weight_data =
            vec![0.0f32; self.out_channels * self.in_channels * self.kernel_size];
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for o in 0..out_length {
                    let grad_idx =
                        b * self.out_channels * out_length + oc * out_length + o;
                    let g = grad_data[grad_idx];

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let input_pos = (o * self.stride + k * self.dilation)
                                as isize
                                - self.padding as isize;

                            if input_pos >= 0 && (input_pos as usize) < length {
                                let input_idx = b * in_ch * length
                                    + ic * length
                                    + input_pos as usize;
                                let w_idx =
                                    oc * self.in_channels * self.kernel_size
                                        + ic * self.kernel_size
                                        + k;
                                grad_weight_data[w_idx] +=
                                    g * input_data[input_idx];
                            }
                        }
                    }
                }
            }
        }

        // --- grad_input: full convolution of grad_output with flipped weight ---
        // grad_input[b, ic, p] = sum over (oc, k) of
        //     grad_output[b, oc, o] * weight[oc, ic, k]
        //   where o*stride + k*dilation - padding = p
        //   i.e., o = (p + padding - k*dilation) / stride
        let mut grad_input_data = vec![0.0f32; batch * in_ch * length];
        for b in 0..batch {
            for ic in 0..self.in_channels {
                for p in 0..length {
                    let mut sum = 0.0f32;

                    for oc in 0..self.out_channels {
                        for k in 0..self.kernel_size {
                            // We need: o * stride + k * dilation - padding = p
                            // So: o * stride = p + padding - k * dilation
                            let numerator =
                                p as isize + self.padding as isize
                                    - (k * self.dilation) as isize;

                            if numerator >= 0
                                && numerator as usize % self.stride == 0
                            {
                                let o = numerator as usize / self.stride;
                                if o < out_length {
                                    let grad_idx = b * self.out_channels
                                        * out_length
                                        + oc * out_length
                                        + o;
                                    let w_idx = oc * self.in_channels
                                        * self.kernel_size
                                        + ic * self.kernel_size
                                        + k;
                                    sum +=
                                        grad_data[grad_idx] * weight_data[w_idx];
                                }
                            }
                        }
                    }

                    let input_idx = b * in_ch * length + ic * length + p;
                    grad_input_data[input_idx] = sum;
                }
            }
        }

        let grad_input =
            Tensor::from_vec(grad_input_data, self.input_shape.clone())
                .expect("conv1d grad_input");
        let grad_weight = Tensor::from_vec(
            grad_weight_data,
            vec![self.out_channels, self.in_channels, self.kernel_size],
        )
        .expect("conv1d grad_weight");
        let grad_bias = Tensor::from_vec(grad_bias_data, vec![self.out_channels])
            .expect("conv1d grad_bias");

        vec![grad_input, grad_weight, grad_bias]
    }

    fn name(&self) -> &str {
        "Conv1dBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Conv1d::new
    #[test]
    fn test_conv1d_new_creates_correct_parameter_shapes() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.in_channels(), 3);
        assert_eq!(layer.out_channels(), 8);
        assert_eq!(layer.kernel_size(), 5);
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[8, 3, 5]); // weight
        assert_eq!(params[1].shape(), &[8]);         // bias
    }

    /// @covers: Conv1d::with_stride
    #[test]
    fn test_with_stride() {
        let layer = Conv1d::new(2, 4, 3).with_stride(2);
        assert_eq!(layer.stride(), 2);
    }

    /// @covers: Conv1d::with_padding
    #[test]
    fn test_with_padding() {
        let layer = Conv1d::new(2, 4, 3).with_padding(1);
        assert_eq!(layer.padding(), 1);
    }

    /// @covers: Conv1d::with_dilation
    #[test]
    fn test_with_dilation() {
        let layer = Conv1d::new(2, 4, 3).with_dilation(2);
        assert_eq!(layer.dilation(), 2);
    }

    /// @covers: Conv1d::in_channels
    #[test]
    fn test_in_channels() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.in_channels(), 3);
    }

    /// @covers: Conv1d::out_channels
    #[test]
    fn test_out_channels() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.out_channels(), 8);
    }

    /// @covers: Conv1d::kernel_size
    #[test]
    fn test_kernel_size() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.kernel_size(), 5);
    }

    /// @covers: Conv1d::stride
    #[test]
    fn test_stride() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.stride(), 1);
    }

    /// @covers: Conv1d::padding
    #[test]
    fn test_padding() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.padding(), 0);
    }

    /// @covers: Conv1d::dilation
    #[test]
    fn test_dilation() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.dilation(), 1);
    }

    /// @covers: Conv1d::parameters
    #[test]
    fn test_parameters() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.parameters().len(), 2);
    }

    /// @covers: Conv1d::parameters_mut
    #[test]
    fn test_parameters_mut() {
        let mut layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.parameters_mut().len(), 2);
    }

    /// @covers: Conv1d::forward
    #[test]
    fn test_conv1d_forward_output_shape() {
        let mut layer = Conv1d::new(2, 4, 3);
        let input = Tensor::randn([1, 2, 10]); // batch=1, in_ch=2, length=10
        let output = layer.forward(&input).unwrap();
        // out_length = (10 + 0 - 1*(3-1) - 1) / 1 + 1 = 8
        assert_eq!(output.shape(), &[1, 4, 8]);
    }

}
