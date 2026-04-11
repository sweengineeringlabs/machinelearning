use crate::api::tape::backward_op::BackwardOp;
use crate::api::tensor::TensorId;
use crate::api::tensor::Tensor;

// --- TapeEntry ---

pub struct TapeEntry {
    pub backward_op: Box<dyn BackwardOp>,
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub saved_tensors: Vec<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubBackward;

    impl BackwardOp for StubBackward {
        fn backward(&self, _grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
            vec![]
        }
        fn name(&self) -> &str {
            "StubBackward"
        }
    }

    /// @covers: TapeEntry (struct construction)
    #[test]
    fn test_tape_entry_holds_backward_op_and_ids() {
        let t1 = Tensor::zeros(vec![2]);
        let t2 = Tensor::ones(vec![2]);
        let entry = TapeEntry {
            backward_op: Box::new(StubBackward),
            output_id: t1.id(),
            input_ids: vec![t2.id()],
            saved_tensors: vec![t2.clone()],
        };
        assert_eq!(entry.output_id, t1.id());
        assert_eq!(entry.input_ids.len(), 1);
        assert_eq!(entry.saved_tensors.len(), 1);
        assert_eq!(entry.backward_op.name(), "StubBackward");
    }
}
