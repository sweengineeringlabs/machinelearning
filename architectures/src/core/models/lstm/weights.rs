/// Preloaded weight and bias data for a single LSTM layer (forward pass).
///
/// Caches the raw `Vec<f32>` data extracted from weight/bias tensors
/// to avoid repeated `to_vec()` calls during the timestep loop.
pub(super) struct LstmWeights {
    pub(super) w_ih: Vec<f32>,
    pub(super) w_hh: Vec<f32>,
    pub(super) b_ih: Vec<f32>,
    pub(super) b_hh: Vec<f32>,
    pub(super) layer_input_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_weights_stores_fields() {
        let lw = LstmWeights {
            w_ih: vec![1.0, 2.0],
            w_hh: vec![3.0, 4.0],
            b_ih: vec![0.1],
            b_hh: vec![0.2],
            layer_input_size: 2,
        };
        assert_eq!(lw.w_ih.len(), 2);
        assert_eq!(lw.w_hh.len(), 2);
        assert_eq!(lw.b_ih.len(), 1);
        assert_eq!(lw.b_hh.len(), 1);
        assert_eq!(lw.layer_input_size, 2);
    }
}
