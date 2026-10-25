use std::ffi::OsString;
use std::ptr;
use std::sync::Mutex;

use tn_runtime::{
    governance_model_binding_authorized_from_env, governance_model_binding_matches,
    EXPECTED_GOVERNANCE_MODEL_SHA256_ENV,
};

const MODEL_SHA256: &str =
    "sha256:9a7bed0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
const OTHER_MODEL_SHA256: &str =
    "sha256:8a7bed0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());
const TN_STATUS_OK: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct BytesView {
    ptr: *const u8,
    len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OwnedBytes {
    ptr: *mut u8,
    len: usize,
}

extern "C" {
    fn tn_governance_model_binding_authorized(
        observed_model_sha256: BytesView,
        out_authorized: *mut u8,
        out_error: *mut OwnedBytes,
    ) -> i32;
}

#[test]
fn exact_well_formed_model_digest_is_authorized_for_governance() {
    assert!(governance_model_binding_matches(MODEL_SHA256, MODEL_SHA256));
}

#[test]
fn identical_malformed_values_are_not_authorized_for_governance() {
    assert!(!governance_model_binding_matches(
        "not-a-sha256",
        "not-a-sha256"
    ));
}

#[test]
fn different_well_formed_model_digests_are_not_authorized_for_governance() {
    assert!(!governance_model_binding_matches(
        MODEL_SHA256,
        OTHER_MODEL_SHA256
    ));
}

#[test]
fn uppercase_digest_text_is_not_silently_normalized() {
    assert!(!governance_model_binding_matches(
        &MODEL_SHA256.to_ascii_uppercase(),
        &MODEL_SHA256.to_ascii_uppercase()
    ));
}

#[test]
fn missing_expected_model_digest_environment_denies_governance_authorization() {
    let _lock = ENVIRONMENT_LOCK.lock().unwrap();
    let _environment = ExpectedDigestEnvironment::missing();

    assert!(!governance_model_binding_authorized_from_env(MODEL_SHA256));
}

#[test]
fn matching_expected_model_digest_environment_authorizes_governance() {
    let _lock = ENVIRONMENT_LOCK.lock().unwrap();
    let _environment = ExpectedDigestEnvironment::set(MODEL_SHA256);

    assert!(governance_model_binding_authorized_from_env(MODEL_SHA256));
}

#[test]
fn c_abi_returns_authorized_for_an_exact_environment_binding() {
    let _lock = ENVIRONMENT_LOCK.lock().unwrap();
    let _environment = ExpectedDigestEnvironment::set(MODEL_SHA256);
    let mut authorized = 0;
    let mut error = OwnedBytes {
        ptr: ptr::null_mut(),
        len: 0,
    };

    let status = unsafe {
        tn_governance_model_binding_authorized(
            BytesView {
                ptr: MODEL_SHA256.as_ptr(),
                len: MODEL_SHA256.len(),
            },
            &mut authorized,
            &mut error,
        )
    };

    assert_eq!(status, TN_STATUS_OK);
    assert_eq!(authorized, 1);
    assert!(error.ptr.is_null());
    assert_eq!(error.len, 0);
}

#[test]
fn c_abi_returns_denied_for_a_different_environment_binding() {
    let _lock = ENVIRONMENT_LOCK.lock().unwrap();
    let _environment = ExpectedDigestEnvironment::set(MODEL_SHA256);
    let mut authorized = 99;
    let mut error = OwnedBytes {
        ptr: ptr::null_mut(),
        len: 0,
    };

    let status = unsafe {
        tn_governance_model_binding_authorized(
            BytesView {
                ptr: OTHER_MODEL_SHA256.as_ptr(),
                len: OTHER_MODEL_SHA256.len(),
            },
            &mut authorized,
            &mut error,
        )
    };

    assert_eq!(status, TN_STATUS_OK);
    assert_eq!(authorized, 0);
    assert!(error.ptr.is_null());
    assert_eq!(error.len, 0);
}

struct ExpectedDigestEnvironment {
    previous: Option<OsString>,
}

impl ExpectedDigestEnvironment {
    fn missing() -> Self {
        let previous = std::env::var_os(EXPECTED_GOVERNANCE_MODEL_SHA256_ENV);
        std::env::remove_var(EXPECTED_GOVERNANCE_MODEL_SHA256_ENV);
        Self { previous }
    }

    fn set(value: &str) -> Self {
        let previous = std::env::var_os(EXPECTED_GOVERNANCE_MODEL_SHA256_ENV);
        std::env::set_var(EXPECTED_GOVERNANCE_MODEL_SHA256_ENV, value);
        Self { previous }
    }
}

impl Drop for ExpectedDigestEnvironment {
    fn drop(&mut self) {
        if let Some(value) = self.previous.take() {
            std::env::set_var(EXPECTED_GOVERNANCE_MODEL_SHA256_ENV, value);
        } else {
            std::env::remove_var(EXPECTED_GOVERNANCE_MODEL_SHA256_ENV);
        }
    }
}
