/// Generic batched data loader with optional shuffling.
///
/// Works with any type implementing the `Dataset` trait.
use crate::api::dataset::Dataset;
use crate::api::tensor::Tensor;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// An iterator that yields batched `(input, target)` tensor pairs.
///
/// The last batch may be smaller than `batch_size` if the dataset length
/// is not evenly divisible.
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new `DataLoader`.
    ///
    /// If `shuffle` is true, sample indices are randomly permuted.
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }

    /// Reset the loader to the beginning, re-shuffling if configured.
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Total number of batches per epoch (ceiling division).
    pub fn num_batches(&self) -> usize {
        let len = self.indices.len();
        if len == 0 {
            return 0;
        }
        (len + self.batch_size - 1) / self.batch_size
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end];
        self.current_idx = end;

        let actual_batch_size = batch_indices.len();
        let input_shape = self.dataset.input_shape();
        let target_dim = self.dataset.target_dim();
        let input_elements: usize = input_shape.iter().product();

        let mut input_data = Vec::with_capacity(actual_batch_size * input_elements);
        let mut target_data = Vec::with_capacity(actual_batch_size * target_dim);

        for &idx in batch_indices {
            match self.dataset.get(idx) {
                Ok((input, target)) => {
                    input_data.extend_from_slice(&input.to_vec());
                    target_data.extend_from_slice(&target.to_vec());
                }
                Err(_) => continue,
            }
        }

        let mut batch_input_shape = vec![actual_batch_size];
        batch_input_shape.extend_from_slice(&input_shape);

        let input = Tensor::from_vec(input_data, batch_input_shape).ok()?;
        let target = Tensor::from_vec(target_data, vec![actual_batch_size, target_dim]).ok()?;

        Some((input, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::SwetsResult;

    struct MockDataset { n: usize }

    impl Dataset for MockDataset {
        fn len(&self) -> usize { self.n }
        fn get(&self, idx: usize) -> SwetsResult<(Tensor, Tensor)> {
            let input = Tensor::from_vec(vec![idx as f32; 3], vec![3]).unwrap();
            let target = Tensor::from_vec(vec![idx as f32], vec![1]).unwrap();
            Ok((input, target))
        }
        fn input_shape(&self) -> Vec<usize> { vec![3] }
        fn target_dim(&self) -> usize { 1 }
    }

    /// @covers: DataLoader::new
    #[test]
    fn test_dataloader_iterates_all_samples() {
        let ds = MockDataset { n: 10 };
        let loader = DataLoader::new(ds, 4, false);
        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 3); // 10/4 = 2 full + 1 partial
    }

    /// @covers: DataLoader::num_batches
    #[test]
    fn test_num_batches_ceiling_division() {
        let ds = MockDataset { n: 10 };
        let loader = DataLoader::new(ds, 4, false);
        assert_eq!(loader.num_batches(), 3);
    }

    /// @covers: DataLoader::reset
    #[test]
    fn test_reset_allows_reiteration() {
        let ds = MockDataset { n: 5 };
        let mut loader = DataLoader::new(ds, 2, false);
        let first: Vec<_> = loader.by_ref().collect();
        assert!(loader.next().is_none());
        loader.reset();
        let second: Vec<_> = loader.collect();
        assert_eq!(first.len(), second.len());
    }

    /// @covers: DataLoader::new
    #[test]
    fn test_empty_dataset_produces_no_batches() {
        let ds = MockDataset { n: 0 };
        let loader = DataLoader::new(ds, 4, false);
        assert_eq!(loader.num_batches(), 0);
        assert_eq!(loader.collect::<Vec<_>>().len(), 0);
    }
}
