//! BBG HIBE cipher core for tn-proto.
//!
//! This crate is the Rust core behind the `cipher: hibe` group option: a
//! hybrid KEM/DEM where BBG HIBE (Boneh-Boyen-Goh 2005, constant-size
//! ciphertext) wraps a 32-byte content-encryption key to an identity path
//! under an authority's master public key. AES-256-GCM seals the body.
//!
//! The scheme is provided by [`tn_bbg`], implementing BBG on BLS12-381.
//! This crate re-exports its HIBE interface.
//!
//! What the surface provides:
//! - [`Identity`]: identity paths and the pinned label→scalar mapping
//!   (`I_i = SHA-256(label) mod p`).
//! - [`setup`] / [`keygen`] / [`delegate`]: authority and delegation.
//! - [`kem_wrap`] / [`kem_unwrap`]: the CEK KEM. KEM-not-direct: the wire
//!   format is canonical AES-256-GCM output plus compressed group points —
//!   a raw GT element never leaves the process.
//! - [`seal`] / [`seal_with_aad`] / [`open`] / [`open_with_aad`]: the full
//!   group-ciphertext blob (with optional AAD marker binding).
//! - Canonical byte encodings for [`PublicParams`], [`MasterKey`],
//!   [`PrivateKey`] and [`mpk_fingerprint`].
//!
//! Direct BBG encryption over GT, [`Ciphertext`], and GT byte codecs live under
//! [`raw`] for golden-vector fixtures and advanced interop checks.
//!
//! # Security status
//!
//! `bls12_381_plus` and the `tn_bbg` scheme code are unaudited. External
//! cryptographic review is required before production use.
//!
//! # Key lifecycle
//!
//! BBG delegated keys are permanent: there is no forward revocation of an
//! admitted reader. Groups that need that use btn.

pub use tn_bbg::{
    delegate, kem_unwrap, kem_wrap, keygen, mpk_fingerprint, open, open_with_aad, seal,
    seal_with_aad, setup, Identity, MasterKey, PrivateKey, PublicParams, WRAPPED_CEK_LEN,
};

/// The scheme error type. Re-exported from [`tn_bbg`] under the public
/// `HibeError` name so existing `tn_hibe::HibeError` consumers are unchanged.
pub use tn_bbg::BbgError as HibeError;

/// Raw BBG primitives over GT.
///
/// The normal HIBE wire surface is [`kem_wrap`]/[`kem_unwrap`] and
/// [`seal`]/[`open`]. Direct GT encryption and GT byte codecs are kept here
/// for golden-vector fixtures and advanced interop checks.
pub mod raw {
    pub use tn_bbg::raw::{decrypt, encrypt, gt_from_bytes, gt_to_bytes, Ciphertext};
}
