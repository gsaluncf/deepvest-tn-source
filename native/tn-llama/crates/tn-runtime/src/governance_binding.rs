use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use subtle::ConstantTimeEq;

/// Receiver-local configuration key for the model digest permitted to
/// participate in governed processing.
pub const EXPECTED_GOVERNANCE_MODEL_SHA256_ENV: &str = "TN_LLAMA_EXPECTED_MODEL_SHA256";

/// Returns the receiver-local model-binding decision configured for this
/// process. This fact is only one input to per-transition nine-question
/// evidence.
#[must_use]
pub fn governance_model_binding_authorized_from_env(observed: &str) -> bool {
    std::env::var(EXPECTED_GOVERNANCE_MODEL_SHA256_ENV)
        .ok()
        .is_some_and(|expected| governance_model_binding_matches(observed, &expected))
}

/// Hash the model file in Rust and compare it with the receiver-local value.
pub fn governance_model_file_matches(path: impl AsRef<Path>, expected: &str) -> io::Result<bool> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let observed = format!("sha256:{}", hex::encode(hasher.finalize()));
    Ok(governance_model_binding_matches(&observed, expected))
}

/// Exact-compares two canonical SHA-256 digest tokens.
#[must_use]
pub fn governance_model_binding_matches(observed: &str, expected: &str) -> bool {
    valid_sha256_digest(observed)
        && valid_sha256_digest(expected)
        && observed.as_bytes().ct_eq(expected.as_bytes()).into()
}

fn valid_sha256_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}
