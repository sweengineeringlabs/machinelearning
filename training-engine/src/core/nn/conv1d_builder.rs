use super::conv1d::Conv1d;

/// Builder for `Conv1d` with optional stride, padding, and dilation.
pub struct Conv1dBuilder {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1dBuilder {
    /// Creates a new builder with the required channel and kernel parameters.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
        }
    }

    /// Sets the stride.
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Sets the padding.
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the dilation.
    pub fn dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Builds the `Conv1d` layer with the configured parameters.
    pub fn build(self) -> Conv1d {
        Conv1d::new(self.in_channels, self.out_channels, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_dilation(self.dilation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Conv1dBuilder::new
    #[test]
    fn test_conv1d_builder_defaults() {
        let layer = Conv1dBuilder::new(3, 8, 5).build();
        assert_eq!(layer.in_channels(), 3);
        assert_eq!(layer.out_channels(), 8);
        assert_eq!(layer.kernel_size(), 5);
        assert_eq!(layer.stride(), 1);
        assert_eq!(layer.padding(), 0);
        assert_eq!(layer.dilation(), 1);
    }

    /// @covers: Conv1dBuilder::build
    #[test]
    fn test_conv1d_builder_custom_config() {
        let layer = Conv1dBuilder::new(2, 4, 3)
            .stride(2)
            .padding(1)
            .dilation(2)
            .build();
        assert_eq!(layer.stride(), 2);
        assert_eq!(layer.padding(), 1);
        assert_eq!(layer.dilation(), 2);
    }

    /// @covers: Conv1dBuilder::stride
    #[test]
    fn test_conv1d_builder_stride_setter() {
        let layer = Conv1dBuilder::new(1, 1, 3).stride(3).build();
        assert_eq!(layer.stride(), 3);
    }

    /// @covers: Conv1dBuilder::padding
    #[test]
    fn test_conv1d_builder_padding_setter() {
        let layer = Conv1dBuilder::new(1, 1, 3).padding(2).build();
        assert_eq!(layer.padding(), 2);
    }

    /// @covers: Conv1dBuilder::dilation
    #[test]
    fn test_conv1d_builder_dilation_setter() {
        let layer = Conv1dBuilder::new(1, 1, 3).dilation(4).build();
        assert_eq!(layer.dilation(), 4);
    }
}
