//! Shape representation and utilities for tensors

use std::fmt;

/// Represents the shape of a tensor
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Create a scalar shape (0 dimensions)
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if shape is scalar
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get size at dimension (supports negative indexing)
    pub fn size(&self, dim: i64) -> Option<usize> {
        let idx = self.normalize_dim(dim)?;
        Some(self.dims[idx])
    }

    /// Normalize negative dimension index
    pub fn normalize_dim(&self, dim: i64) -> Option<usize> {
        let ndim = self.ndim() as i64;
        let normalized = if dim < 0 { dim + ndim } else { dim };
        if normalized >= 0 && normalized < ndim {
            Some(normalized as usize)
        } else {
            None
        }
    }

    /// Compute broadcast shape with another shape
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let max_dims = self.ndim().max(other.ndim());
        let mut result = Vec::with_capacity(max_dims);

        for i in 0..max_dims {
            let a = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };
            let b = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            if a == b {
                result.push(a);
            } else if a == 1 {
                result.push(b);
            } else if b == 1 {
                result.push(a);
            } else {
                return None;
            }
        }

        result.reverse();
        Some(Shape::new(result))
    }

    /// Create shape with an additional dimension
    pub fn with_dim(&self, dim: i64, size: usize) -> Option<Shape> {
        let ndim = self.ndim() as i64 + 1;
        let normalized = if dim < 0 { dim + ndim } else { dim };
        if normalized < 0 || normalized > self.ndim() as i64 {
            return None;
        }
        let idx = normalized as usize;
        let mut dims = self.dims.clone();
        dims.insert(idx, size);
        Some(Shape::new(dims))
    }

    /// Remove a dimension
    pub fn squeeze(&self, dim: i64) -> Option<Shape> {
        let idx = self.normalize_dim(dim)?;
        if self.dims[idx] != 1 {
            return None;
        }
        let mut dims = self.dims.clone();
        dims.remove(idx);
        Some(Shape::new(dims))
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: new
    #[test]
    fn test_shape_basic() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.size(0), Some(2));
        assert_eq!(shape.size(-1), Some(4));
    }

    /// @covers: broadcast_with
    #[test]
    fn test_broadcast() {
        let a = Shape::new(vec![2, 1, 4]);
        let b = Shape::new(vec![3, 4]);
        let c = a.broadcast_with(&b);
        assert_eq!(c, Some(Shape::new(vec![2, 3, 4])));
    }

    /// @covers: scalar
    #[test]
    fn test_scalar_has_zero_dims() {
        let s = Shape::scalar();
        assert!(s.is_scalar());
        assert_eq!(s.ndim(), 0);
    }

    /// @covers: numel
    #[test]
    fn test_numel_product_of_dims() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.numel(), 24);
    }

    /// @covers: size
    #[test]
    fn test_size_negative_indexing() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.size(-1), Some(4));
        assert_eq!(s.size(-3), Some(2));
        assert_eq!(s.size(-4), None);
    }

    /// @covers: with_dim
    #[test]
    fn test_with_dim_inserts_at_correct_position() {
        let s = Shape::new(vec![2, 3]);
        let expanded = s.with_dim(1, 5).unwrap();
        assert_eq!(expanded.dims(), &[2, 5, 3]);
    }

    /// @covers: squeeze
    #[test]
    fn test_squeeze_removes_size_one_dim() {
        let s = Shape::new(vec![2, 1, 3]);
        let squeezed = s.squeeze(1).unwrap();
        assert_eq!(squeezed.dims(), &[2, 3]);
    }

    /// @covers: squeeze
    #[test]
    fn test_squeeze_non_one_dim_returns_none() {
        let s = Shape::new(vec![2, 3]);
        assert!(s.squeeze(0).is_none());
    }

    /// @covers: broadcast_with
    #[test]
    fn test_broadcast_incompatible_returns_none() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![4, 5]);
        assert!(a.broadcast_with(&b).is_none());
    }

    /// @covers: fmt
    #[test]
    fn test_shape_display_format() {
        let s = Shape::new(vec![2, 3]);
        assert_eq!(format!("{}", s), "[2, 3]");
    }

    /// @covers: from
    #[test]
    fn test_shape_from_vec() {
        let s: Shape = vec![4, 5].into();
        assert_eq!(s.dims(), &[4, 5]);
    }

    /// @covers: from
    #[test]
    fn test_shape_from_array() {
        let s: Shape = [2, 3, 4].into();
        assert_eq!(s.ndim(), 3);
    }

    /// @covers: new
    #[test]
    fn test_new_stores_dimensions() {
        let s = Shape::new(vec![10, 20]);
        assert_eq!(s.dims(), &[10, 20]);
    }

    /// @covers: ndim
    #[test]
    fn test_ndim_returns_dimension_count() {
        assert_eq!(Shape::new(vec![2, 3, 4]).ndim(), 3);
        assert_eq!(Shape::scalar().ndim(), 0);
    }

    /// @covers: is_scalar
    #[test]
    fn test_is_scalar_true_for_empty_dims() {
        assert!(Shape::scalar().is_scalar());
        assert!(!Shape::new(vec![1]).is_scalar());
    }

    /// @covers: normalize_dim
    #[test]
    fn test_normalize_dim_positive_and_negative() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.normalize_dim(0), Some(0));
        assert_eq!(s.normalize_dim(-1), Some(2));
        assert_eq!(s.normalize_dim(3), None);
    }

    /// @covers: fmt
    #[test]
    fn test_fmt_debug_shows_shape_prefix() {
        let s = Shape::new(vec![2, 3]);
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("Shape"), "debug should contain 'Shape': {}", dbg);
    }

    /// @covers: as_ref
    #[test]
    fn test_as_ref_returns_dims_slice() {
        let s = Shape::new(vec![5, 6]);
        let r: &[usize] = s.as_ref();
        assert_eq!(r, &[5, 6]);
    }

    /// @covers: dims
    #[test]
    fn test_dims_returns_dimension_slice() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.dims(), &[2, 3, 4]);
    }

    /// @covers: broadcast_with
    #[test]
    fn test_broadcast_with_same_shape() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![2, 3]);
        assert_eq!(a.broadcast_with(&b), Some(Shape::new(vec![2, 3])));
    }
}
