/// L2-normalize a vector in place to unit length.
/// After normalization, cosine similarity equals the dot product.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize_unit_vector() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_result_has_unit_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector_unchanged() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
