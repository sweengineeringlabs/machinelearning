/// Preloaded weight data for a single LSTM layer (backward pass).
///
/// Stores the raw weight vectors and the layer input size needed
/// during backpropagation through time (BPTT).
pub(super) struct LstmWeightData {
    pub(super) w_ih: Vec<f32>,
    pub(super) w_hh: Vec<f32>,
    pub(super) layer_input_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_weight_data_stores_fields() {
        let lwd = LstmWeightData {
            w_ih: vec![1.0, 2.0, 3.0],
            w_hh: vec![4.0, 5.0],
            layer_input_size: 3,
        };
        assert_eq!(lwd.w_ih.len(), 3);
        assert_eq!(lwd.w_hh.len(), 2);
        assert_eq!(lwd.layer_input_size, 3);
    }
}
