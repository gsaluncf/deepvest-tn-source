//! Runtime configuration with a fixed tree height of 8 (256 leaves).

use crate::error::{Error, Result};

/// Fixed tree height.
///
/// 2^8 = 256 leaves. Each reader has h(h+1)/2 = 36 path keys,
/// totaling 36 × 32 = 1152 bytes of path-key material, plus the
/// full-tree key and metadata.
pub const TREE_HEIGHT: u8 = 8;

/// Maximum number of leaves (2^TREE_HEIGHT).
pub const MAX_LEAVES: u64 = 1u64 << TREE_HEIGHT;

/// Per-user path-keys count (naive NNL-SD): h(h+1)/2.
/// Exposed for tests and callers sizing buffers.
pub const PATH_KEYS_PER_READER: usize = (TREE_HEIGHT as usize) * (TREE_HEIGHT as usize + 1) / 2;

/// Configuration for a `PublisherState`.
///
/// This configuration has no tunable fields. Tree height, AEAD, and PRG
/// are fixed by the implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Config;

impl Config {
    /// Validate the fixed configuration. Returns `Ok(())`.
    #[inline]
    pub fn validate(self) -> Result<()> {
        let _unused: Error;
        Ok(())
    }

    /// Tree height, equal to [`TREE_HEIGHT`].
    #[inline]
    pub fn tree_height(self) -> u8 {
        TREE_HEIGHT
    }

    /// Maximum number of leaves, equal to [`MAX_LEAVES`].
    #[inline]
    pub fn max_leaves(self) -> u64 {
        MAX_LEAVES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_height_is_eight() {
        assert_eq!(TREE_HEIGHT, 8);
        assert_eq!(MAX_LEAVES, 256);
        assert_eq!(Config.tree_height(), 8);
        assert_eq!(Config.max_leaves(), 256);
        assert_eq!(PATH_KEYS_PER_READER, 36);
    }

    #[test]
    fn fixed_configuration_is_valid() {
        assert!(Config.validate().is_ok());
    }
}
