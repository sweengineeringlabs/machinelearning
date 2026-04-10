//! KV Cache for efficient autoregressive inference.
//!
//! Pre-allocates key/value buffers for each transformer layer and provides
//! efficient update/view operations for incremental decoding.

use std::time::Instant;
use crate::api::error::{NnError, NnResult};
use tensor_engine::Tensor;

/// KV Cache storing past keys and values for each layer.
pub struct KVCache {
    past_keys: Vec<Tensor>,   // [num_slots, B, H_kv, S_max, D_head]
    past_values: Vec<Tensor>, // [num_slots, B, H_kv, S_max, D_head]
    /// Mapping from layer index to cache slot index (Gemma 4).
    /// If None, layer_idx == slot_idx.
    layer_to_slot: Option<Vec<usize>>,
    max_seq_len: usize,
    pub current_len: usize,
    /// Per-slot head dimensions (supports mixed head_dim models like Gemma 4).
    slot_head_dims: Vec<usize>,
    num_kv_heads: usize,
    /// Raw token history for models without native KV cache support.
    pub token_history: Vec<u32>,
}

impl KVCache {
    /// Create a new KV cache.
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Self {
        let slot_head_dims = vec![head_dim; num_layers];
        Self::from_slot_dims(num_layers, None, slot_head_dims, max_seq_len, num_kv_heads, 1)
    }

    /// Create a KV cache with layer sharing (Gemma 4).
    pub fn with_kv_sharing(
        num_layers: usize,
        num_slots: usize,
        layer_to_slot: Vec<usize>,
        max_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Self {
        let slot_head_dims = vec![head_dim; num_slots];
        Self::from_slot_dims(num_slots, Some(layer_to_slot), slot_head_dims, max_seq_len, num_kv_heads, 1)
    }

    /// Create a KV cache with per-slot head dimensions and optional layer sharing.
    ///
    /// Use this for models with mixed head dimensions (e.g. Gemma 4 where
    /// sliding layers use 256 and global layers use 512).
    pub fn with_per_slot_head_dims(
        layer_to_slot: Option<Vec<usize>>,
        slot_head_dims: Vec<usize>,
        max_seq_len: usize,
        num_kv_heads: usize,
    ) -> Self {
        let num_slots = slot_head_dims.len();
        Self::from_slot_dims(num_slots, layer_to_slot, slot_head_dims, max_seq_len, num_kv_heads, 1)
    }

    /// Create a KV cache with a parameterized batch size (for batched inference).
    pub fn new_batched(
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
        batch_size: usize,
    ) -> Self {
        let slot_head_dims = vec![head_dim; num_layers];
        Self::from_slot_dims(num_layers, None, slot_head_dims, max_seq_len, num_kv_heads, batch_size)
    }

    /// Internal constructor that pre-allocates buffers with per-slot head dimensions.
    fn from_slot_dims(
        num_slots: usize,
        layer_to_slot: Option<Vec<usize>>,
        slot_head_dims: Vec<usize>,
        max_seq_len: usize,
        num_kv_heads: usize,
        batch_size: usize,
    ) -> Self {
        let past_keys = slot_head_dims.iter()
            .map(|&hd| Tensor::zeros(vec![batch_size, num_kv_heads, max_seq_len, hd]))
            .collect();
        let past_values = slot_head_dims.iter()
            .map(|&hd| Tensor::zeros(vec![batch_size, num_kv_heads, max_seq_len, hd]))
            .collect();

        Self {
            past_keys,
            past_values,
            layer_to_slot,
            max_seq_len,
            current_len: 0,
            slot_head_dims,
            num_kv_heads,
            token_history: Vec::new(),
        }
    }

    /// Maximum head dimension across all slots.
    pub fn head_dim(&self) -> usize {
        self.slot_head_dims.iter().copied().max().unwrap_or(0)
    }

    /// Head dimension for a specific layer's cache slot.
    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        let slot = self.get_slot_idx(layer_idx);
        self.slot_head_dims[slot]
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Number of actual K/V storage slots in this cache.
    pub fn num_slots(&self) -> usize {
        self.past_keys.len()
    }

    /// Number of virtual layers supported by this cache.
    pub fn num_layers(&self) -> usize {
        self.layer_to_slot.as_ref().map(|m| m.len()).unwrap_or(self.past_keys.len())
    }

    pub fn get_slot_idx(&self, layer_idx: usize) -> usize {
        self.layer_to_slot.as_ref().map(|m| m[layer_idx]).unwrap_or(layer_idx)
    }

    /// Returns true if this layer is the first one to use its assigned cache slot (Gemma 4).
    /// Shared layers (which appear later in the layer_to_slot map) return false.
    pub fn is_slot_owner(&self, layer_idx: usize) -> bool {
        match self.layer_to_slot.as_ref() {
            Some(map) => {
                let slot = map[layer_idx];
                // First occurrence of this slot index in the map is the owner
                map.iter().position(|&s| s == slot) == Some(layer_idx)
            }
            None => true, // No sharing -> everyone owns their slot
        }
    }

    /// Total bytes allocated for all K/V buffers.
    pub fn memory_bytes(&self) -> usize {
        let per_tensor = |t: &Tensor| -> usize {
            t.shape().iter().product::<usize>() * std::mem::size_of::<f32>()
        };
        let k_bytes: usize = self.past_keys.iter().map(per_tensor).sum();
        let v_bytes: usize = self.past_values.iter().map(per_tensor).sum();
        k_bytes + v_bytes
    }

    /// Get a view of cached K/V for `0..len` on the sequence dimension.
    pub fn get_view(&self, layer_idx: usize, len: usize) -> NnResult<(Tensor, Tensor)> {
        let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
        let slot_idx = self.get_slot_idx(layer_idx);
        let k = self.past_keys[slot_idx].slice_sequence(0, len)?;
        let v = self.past_values[slot_idx].slice_sequence(0, len)?;
        if let Some(t) = _t {
            log::trace!("[perf] kv_cache::get_view layer={} slot={} len={} {:.3}ms",
                layer_idx, slot_idx, len, t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((k, v))
    }

    /// Write new K/V entries into the cache at `current_len`.
    pub fn update(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> NnResult<()> {
        let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
        let seq_len = key.shape()[2];
        if self.current_len + seq_len > self.max_seq_len {
            return Err(NnError::InvalidConfig(format!(
                "Sequence length exceeded: max={}, actual={}",
                self.max_seq_len,
                self.current_len + seq_len,
            )));
        }
        let slot_idx = self.get_slot_idx(layer_idx);
        self.past_keys[slot_idx].slice_assign_sequence(self.current_len, &key)?;
        self.past_values[slot_idx].slice_assign_sequence(self.current_len, &value)?;
        if let Some(t) = _t {
            log::trace!("[perf] kv_cache::update layer={} slot={} pos={} {:.3}ms",
                layer_idx, slot_idx, self.current_len, t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(())
    }

    /// Advance the current position by `step` tokens.
    pub fn advance(&mut self, step: usize) {
        self.current_len += step;
    }

    /// Snapshot the current filled prefix for later reuse.
    ///
    /// Returns a deep clone of this cache, preserving all data up to `current_len`.
    /// Use with `restore_from()` to skip redundant prefill for shared prompts.
    pub fn snapshot(&self) -> NnResult<Self> {
        self.deep_clone()
    }

    /// Restore cache state from a previously-snapshotted prefix.
    ///
    /// Copies the filled region from `snapshot` into this cache and resets
    /// `current_len` to match, so the next generation starts right after the
    /// prefix without re-running prefill.
    pub fn restore_from(&self, snapshot: &KVCache) -> NnResult<Self> {
        if snapshot.slot_head_dims != self.slot_head_dims
            || snapshot.num_kv_heads != self.num_kv_heads
            || snapshot.num_slots() != self.num_slots()
            || snapshot.num_layers() != self.num_layers()
        {
            return Err(NnError::InvalidConfig(
                "prefix snapshot dimensions do not match this cache".into(),
            ));
        }
        if snapshot.current_len > self.max_seq_len {
            return Err(NnError::InvalidConfig(format!(
                "prefix length {} exceeds max_seq_len {}",
                snapshot.current_len, self.max_seq_len,
            )));
        }
        snapshot.deep_clone()
    }

    /// Create a deep copy with independently owned tensor data (not Arc-shared).
    /// Required for beam search where each beam needs a mutable cache.
    pub fn deep_clone(&self) -> NnResult<Self> {
        let past_keys = self
            .past_keys
            .iter()
            .map(|t| {
                let bytes = t.as_raw_bytes()?.to_vec();
                Ok(Tensor::new(bytes, t.shape().to_vec(), t.dtype()))
            })
            .collect::<NnResult<Vec<_>>>()?;

        let past_values = self
            .past_values
            .iter()
            .map(|t| {
                let bytes = t.as_raw_bytes()?.to_vec();
                Ok(Tensor::new(bytes, t.shape().to_vec(), t.dtype()))
            })
            .collect::<NnResult<Vec<_>>>()?;

        Ok(Self {
            past_keys,
            past_values,
            layer_to_slot: self.layer_to_slot.clone(),
            max_seq_len: self.max_seq_len,
            current_len: self.current_len,
            slot_head_dims: self.slot_head_dims.clone(),
            num_kv_heads: self.num_kv_heads,
            token_history: self.token_history.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_basic() {
        let mut cache = KVCache::new(2, 32, 64, 4);
        assert_eq!(cache.current_len, 0);
        assert_eq!(cache.head_dim(), 64);

        // Insert [1, 4, 5, 64] K/V into layer 0
        let k = Tensor::randn(vec![1, 4, 5, 64]);
        let v = Tensor::randn(vec![1, 4, 5, 64]);
        cache.update(0, k, v).unwrap();
        cache.advance(5);
        assert_eq!(cache.current_len, 5);

        let (k_view, v_view) = cache.get_view(0, 5).unwrap();
        assert_eq!(k_view.shape()[2], 5);
        assert_eq!(v_view.shape()[2], 5);
    }

    #[test]
    fn test_kv_cache_overflow() {
        let mut cache = KVCache::new(1, 4, 8, 2);
        let k = Tensor::randn(vec![1, 2, 5, 8]);
        let v = Tensor::randn(vec![1, 2, 5, 8]);
        // seq_len=5 > max_seq_len=4 should fail
        assert!(cache.update(0, k, v).is_err());
    }

    #[test]
    fn test_kv_cache_deep_clone() {
        let mut cache = KVCache::new(1, 16, 8, 2);
        let k = Tensor::randn(vec![1, 2, 3, 8]);
        let v = Tensor::randn(vec![1, 2, 3, 8]);
        cache.update(0, k, v).unwrap();
        cache.advance(3);

        let clone = cache.deep_clone().unwrap();
        assert_eq!(clone.current_len, 3);
    }

    #[test]
    fn test_kv_sharing() {
        let num_layers = 4;
        let num_slots = 2;
        let layer_to_slot = vec![0, 1, 0, 1]; // L0, L2 use slot 0; L1, L3 use slot 1
        let mut cache = KVCache::with_kv_sharing(num_layers, num_slots, layer_to_slot, 16, 8, 2);
        
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.num_slots(), 2);
        
        // Write to L0
        let k0 = Tensor::randn(vec![1, 2, 1, 8]);
        let v0 = Tensor::randn(vec![1, 2, 1, 8]);
        cache.update(0, k0.clone(), v0.clone()).unwrap();
        
        // L2 should now have the same data
        let (k2, _v2) = cache.get_view(2, 1).unwrap();
        let d0 = k0.as_slice_f32().unwrap();
        let d2 = k2.as_slice_f32().unwrap();
        assert_eq!(d0, d2);
        
        // Write to L1 (slot 1)
        let k1 = Tensor::randn(vec![1, 2, 1, 8]);
        let v1 = Tensor::randn(vec![1, 2, 1, 8]);
        cache.update(1, k1.clone(), v1.clone()).unwrap();
        
        // L3 should have the data from L1
        let (k3, _v3) = cache.get_view(3, 1).unwrap();
        assert_eq!(k1.as_slice_f32().unwrap(), k3.as_slice_f32().unwrap());
    }

    #[test]
    fn test_kv_cache_prefix_snapshot() {
        let mut cache = KVCache::new(1, 16, 8, 2);
        let k = Tensor::randn(vec![1, 2, 5, 8]);
        let v = Tensor::randn(vec![1, 2, 5, 8]);
        cache.update(0, k, v).unwrap();
        cache.advance(5);

        // Take snapshot after prefill
        let snapshot = cache.snapshot().unwrap();
        assert_eq!(snapshot.current_len, 5);

        // Restore into a fresh cache
        let fresh = KVCache::new(1, 16, 8, 2);
        let restored = fresh.restore_from(&snapshot).unwrap();
        assert_eq!(restored.current_len, 5);

        // Verify we can continue generating from the restored cache
        let (k_view, v_view) = restored.get_view(0, 5).unwrap();
        assert_eq!(k_view.shape()[2], 5);
        assert_eq!(v_view.shape()[2], 5);
    }
}
