use std::mem;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::{self, null_mut};

use serde_json::json;

use crate::model_protocol::session_error;
use crate::{
    governance_model_binding_authorized_from_env, governance_model_file_matches, ActionOutcome,
    AdmissionStage, GovernedRuntime, NativeToolResult, ParsedToolCall, Phase, RuntimeConfig,
    Session, TnLlamaError, TurnPermit,
};

const STATUS_OK: i32 = 0;
const STATUS_ERROR: i32 = 1;
const STATUS_PANIC: i32 = 2;
const PANIC_ERROR_JSON: &[u8] =
    br#"{"code":"tn_ffi_panic","stage":"ffi","message":"panic contained at the C ABI boundary"}"#;

pub(crate) struct RuntimeHandle(GovernedRuntime);
pub(crate) struct SessionHandle(Session);
pub(crate) struct TurnHandle(Option<TurnPermit>);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BytesView {
    ptr: *const u8,
    len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct OwnedBytes {
    ptr: *mut u8,
    len: usize,
}

#[repr(C)]
pub(crate) struct NativeToolCall {
    id: BytesView,
    name: BytesView,
    normalized_arguments: BytesView,
}

#[repr(C)]
pub(crate) struct NativeToolResultInput {
    call_count: usize,
    call: *const NativeToolCall,
    content: BytesView,
    raw_model_output: BytesView,
    truncated: u8,
    stop_limit: u8,
}

impl OwnedBytes {
    const EMPTY: Self = Self {
        ptr: null_mut(),
        len: 0,
    };

    fn from_vec(bytes: Vec<u8>) -> Self {
        if bytes.is_empty() {
            return Self::EMPTY;
        }
        let mut bytes = bytes.into_boxed_slice();
        let value = Self {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
        };
        mem::forget(bytes);
        value
    }
}

#[no_mangle]
pub(crate) extern "C" fn tn_governance_model_binding_authorized(
    observed_model_sha256: BytesView,
    out_authorized: *mut u8,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || zero_u8_output(out_authorized, "out_authorized"),
        || {
            let observed = read_view(observed_model_sha256, "observed_model_sha256")?;
            let observed = std::str::from_utf8(observed)
                .map_err(|_| invalid_argument("observed_model_sha256 must contain valid UTF-8"))?;
            unsafe {
                *out_authorized = u8::from(governance_model_binding_authorized_from_env(observed));
            }
            Ok(())
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_governance_model_file_authorized(
    config_json: BytesView,
    model_path: BytesView,
    out_authorized: *mut u8,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || zero_u8_output(out_authorized, "out_authorized"),
        || {
            let config: RuntimeConfig =
                serde_json::from_slice(read_view(config_json, "config_json")?).map_err(|_| {
                    ffi_error(
                        "tn_ffi_config_invalid",
                        "runtime configuration JSON is invalid",
                    )
                })?;
            let model_path = std::str::from_utf8(read_view(model_path, "model_path")?)
                .map_err(|_| invalid_argument("model_path must contain valid UTF-8"))?;
            let expected = format!("sha256:{}", config.model_sha256);
            let authorized = governance_model_file_matches(model_path, &expected)
                .map_err(|_| ffi_error("tn_model_hash_failed", "model file could not be hashed"))?;
            unsafe {
                *out_authorized = u8::from(authorized);
            }
            Ok(())
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_runtime_create(
    config_json: BytesView,
    out_runtime: *mut *mut RuntimeHandle,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || zero_handle_output(out_runtime, "out_runtime"),
        || {
            let config_json = read_view(config_json, "config_json")?;
            let config: RuntimeConfig = serde_json::from_slice(config_json).map_err(|_| {
                ffi_error(
                    "tn_ffi_config_invalid",
                    "runtime configuration JSON is invalid",
                )
            })?;
            let runtime = RuntimeHandle(GovernedRuntime::from_config(config)?);
            unsafe {
                *out_runtime = Box::into_raw(Box::new(runtime));
            }
            Ok(())
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_session_create(
    runtime: *mut RuntimeHandle,
    request_json: BytesView,
    out_session: *mut *mut SessionHandle,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || zero_handle_output(out_session, "out_session"),
        || {
            let runtime = required_ref(runtime, "runtime")?;
            let request_json = read_view(request_json, "request_json")?;
            let session = SessionHandle(runtime.0.start_session(request_json)?);
            unsafe {
                *out_session = Box::into_raw(Box::new(session));
            }
            Ok(())
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_session_next_model_turn(
    session: *mut SessionHandle,
    out_turn: *mut *mut TurnHandle,
    out_phase: *mut u32,
    out_request_json: *mut OwnedBytes,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || zero_next_turn_outputs(out_turn, out_phase, out_request_json),
        || {
            let session = required_mut(session, "session")?;
            let model_turn = session.0.next_model_turn()?;
            let phase = phase_number(model_turn.permit.phase());
            let turn = Box::new(TurnHandle(Some(model_turn.permit)));
            let body = OwnedBytes::from_vec(model_turn.request_json);
            unsafe {
                *out_turn = Box::into_raw(turn);
                *out_phase = phase;
                *out_request_json = body;
            }
            Ok(())
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_turn_prepare(
    turn: *mut TurnHandle,
    request_json: BytesView,
    rendered_prompt: BytesView,
    runnable_manifest: BytesView,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || Ok(()),
        || {
            let turn = required_ref(turn, "turn")?;
            let request_json = read_view(request_json, "request_json")?;
            let rendered_prompt = read_view(rendered_prompt, "rendered_prompt")?;
            let runnable_manifest = read_view(runnable_manifest, "runnable_manifest")?;
            turn_permit(turn)?.prepare(request_json, rendered_prompt, runnable_manifest)
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_turn_begin(
    turn: *mut TurnHandle,
    task_id: i64,
    slot_id: i32,
    request_json: BytesView,
    rendered_prompt: BytesView,
    runnable_manifest: BytesView,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || Ok(()),
        || {
            let turn = required_ref(turn, "turn")?;
            let request_json = read_view(request_json, "request_json")?;
            let rendered_prompt = read_view(rendered_prompt, "rendered_prompt")?;
            let runnable_manifest = read_view(runnable_manifest, "runnable_manifest")?;
            turn_permit(turn)?.begin(
                task_id,
                slot_id,
                request_json,
                rendered_prompt,
                runnable_manifest,
            )
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_session_accept_model_action(
    session: *mut SessionHandle,
    turn: *mut TurnHandle,
    result: *const NativeToolResultInput,
    out_outcome: *mut u32,
    out_release_json: *mut OwnedBytes,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || zero_action_outputs(out_outcome, out_release_json),
        || {
            let session = required_mut(session, "session")?;
            let turn = required_mut(turn, "turn")?;
            let result = required_ref(result, "result")?;
            let content = read_view(result.content, "result.content")?;
            let raw_model_output = read_view(result.raw_model_output, "result.raw_model_output")?;
            let truncated = read_bool(result.truncated, "result.truncated")?;
            let stop_limit = read_bool(result.stop_limit, "result.stop_limit")?;
            let parsed_call = if result.call_count == 1 {
                let call = required_ref(result.call, "result.call")?;
                Some(ParsedToolCall {
                    id: read_view(call.id, "result.call.id")?,
                    name: read_view(call.name, "result.call.name")?,
                    parsed_arguments: read_view(
                        call.normalized_arguments,
                        "result.call.normalized_arguments",
                    )?,
                })
            } else {
                None
            };
            let permit = turn.0.take().ok_or_else(|| {
                session_error("tn_turn_consumed", "turn permit was already consumed")
            })?;
            let outcome = session.0.accept_model_action(
                permit,
                NativeToolResult {
                    call_count: result.call_count,
                    call: parsed_call,
                    content,
                    raw_model_output,
                    truncated,
                    stop_limit,
                },
            )?;
            unsafe {
                match outcome {
                    ActionOutcome::Continue => *out_outcome = 1,
                    ActionOutcome::Release(release) => {
                        *out_outcome = 2;
                        *out_release_json = OwnedBytes::from_vec(release.to_json_bytes()?);
                    }
                    ActionOutcome::GovernedRelease(release) => {
                        *out_outcome = 2;
                        *out_release_json = OwnedBytes::from_vec(release.to_json_bytes()?);
                    }
                }
            }
            Ok(())
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_session_abort(
    session: *mut SessionHandle,
    out_error: *mut OwnedBytes,
) -> i32 {
    catch_ffi(
        out_error,
        || Ok(()),
        || {
            let session = required_mut(session, "session")?;
            session.0.abort()
        },
    )
}

#[no_mangle]
pub(crate) extern "C" fn tn_bytes_free(value: *mut OwnedBytes) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let Some(value) = value.as_mut() else {
            return;
        };
        let owned = mem::replace(value, OwnedBytes::EMPTY);
        if owned.ptr.is_null() || owned.len == 0 {
            return;
        }
        let slice = ptr::slice_from_raw_parts_mut(owned.ptr, owned.len);
        drop(Box::from_raw(slice));
    }));
}

#[no_mangle]
pub(crate) extern "C" fn tn_turn_free(turn: *mut TurnHandle) {
    free_handle(turn);
}

#[no_mangle]
pub(crate) extern "C" fn tn_session_free(session: *mut SessionHandle) {
    free_handle(session);
}

#[no_mangle]
pub(crate) extern "C" fn tn_runtime_free(runtime: *mut RuntimeHandle) {
    free_handle(runtime);
}

fn catch_ffi<I, F>(out_error: *mut OwnedBytes, initialize: I, operation: F) -> i32
where
    I: FnOnce() -> Result<(), TnLlamaError>,
    F: FnOnce() -> Result<(), TnLlamaError>,
{
    let mut panic_error = OwnedBytes::EMPTY;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if !out_error.is_null() {
            unsafe {
                *out_error = OwnedBytes::EMPTY;
            }
            panic_error = OwnedBytes::from_vec(PANIC_ERROR_JSON.to_vec());
        }
        let initialized = initialize();
        if out_error.is_null() {
            return STATUS_ERROR;
        }
        let status = match initialized.and_then(|()| operation()) {
            Ok(()) => STATUS_OK,
            Err(error) => {
                publish_error(out_error, &error);
                STATUS_ERROR
            }
        };
        drop_owned_bytes(&mut panic_error);
        status
    }));
    match outcome {
        Ok(status) => status,
        Err(_) => {
            if !out_error.is_null() && !panic_error.ptr.is_null() {
                unsafe {
                    *out_error = mem::replace(&mut panic_error, OwnedBytes::EMPTY);
                }
            }
            STATUS_PANIC
        }
    }
}

fn publish_error(out_error: *mut OwnedBytes, error: &TnLlamaError) {
    let bytes = serde_json::to_vec(&json!({
        "code": error.code(),
        "stage": stage_name(error.stage()),
        "message": error.message(),
    }))
    .expect("strings always serialize to JSON");
    unsafe {
        *out_error = OwnedBytes::from_vec(bytes);
    }
}

fn drop_owned_bytes(value: &mut OwnedBytes) {
    let owned = mem::replace(value, OwnedBytes::EMPTY);
    if owned.ptr.is_null() || owned.len == 0 {
        return;
    }
    unsafe {
        let slice = ptr::slice_from_raw_parts_mut(owned.ptr, owned.len);
        drop(Box::from_raw(slice));
    }
}

const fn phase_number(phase: Phase) -> u32 {
    match phase {
        Phase::GovernanceRead => 1,
        Phase::CommitmentEcho => 2,
        Phase::SectionSelection => 3,
        Phase::GovernedAction => 4,
    }
}

const fn stage_name(stage: AdmissionStage) -> &'static str {
    match stage {
        AdmissionStage::Parse => "parse",
        AdmissionStage::RequestShape => "request_shape",
        AdmissionStage::RowShape => "row_shape",
        AdmissionStage::RowHash => "row_hash",
        AdmissionStage::Governance => "governance",
        AdmissionStage::Chain => "chain",
        AdmissionStage::Admission => "admission",
        AdmissionStage::Session => "session",
        AdmissionStage::Output => "output",
        AdmissionStage::Tool => "tool",
        AdmissionStage::Ffi => "ffi",
    }
}

fn ffi_error(code: &'static str, message: impl Into<String>) -> TnLlamaError {
    TnLlamaError::new(code, AdmissionStage::Ffi, message)
}

fn invalid_argument(message: impl Into<String>) -> TnLlamaError {
    ffi_error("tn_ffi_invalid_argument", message)
}

fn zero_handle_output<T>(output: *mut *mut T, name: &str) -> Result<(), TnLlamaError> {
    if output.is_null() {
        return Err(invalid_argument(format!("{name} is null")));
    }
    unsafe {
        *output = null_mut();
    }
    Ok(())
}

fn zero_action_outputs(
    out_outcome: *mut u32,
    out_release_json: *mut OwnedBytes,
) -> Result<(), TnLlamaError> {
    unsafe {
        if !out_outcome.is_null() {
            *out_outcome = 0;
        }
        if !out_release_json.is_null() {
            *out_release_json = OwnedBytes::EMPTY;
        }
    }
    if out_outcome.is_null() {
        return Err(invalid_argument("out_outcome is null"));
    }
    if out_release_json.is_null() {
        return Err(invalid_argument("out_release_json is null"));
    }
    Ok(())
}

fn zero_u8_output(output: *mut u8, name: &str) -> Result<(), TnLlamaError> {
    if output.is_null() {
        return Err(invalid_argument(format!("{name} is null")));
    }
    unsafe {
        *output = 0;
    }
    Ok(())
}

fn zero_next_turn_outputs(
    out_turn: *mut *mut TurnHandle,
    out_phase: *mut u32,
    out_request_json: *mut OwnedBytes,
) -> Result<(), TnLlamaError> {
    unsafe {
        if !out_turn.is_null() {
            *out_turn = null_mut();
        }
        if !out_phase.is_null() {
            *out_phase = 0;
        }
        if !out_request_json.is_null() {
            *out_request_json = OwnedBytes::EMPTY;
        }
    }
    if out_turn.is_null() {
        return Err(invalid_argument("out_turn is null"));
    }
    if out_phase.is_null() {
        return Err(invalid_argument("out_phase is null"));
    }
    if out_request_json.is_null() {
        return Err(invalid_argument("out_request_json is null"));
    }
    Ok(())
}

fn read_view<'a>(view: BytesView, name: &str) -> Result<&'a [u8], TnLlamaError> {
    if view.ptr.is_null() {
        return if view.len == 0 {
            Ok(&[])
        } else {
            Err(invalid_argument(format!(
                "{name} has a null pointer with a nonzero length"
            )))
        };
    }
    if view.len > isize::MAX as usize {
        return Err(invalid_argument(format!("{name} is too large")));
    }
    Ok(unsafe { std::slice::from_raw_parts(view.ptr, view.len) })
}

fn read_bool(value: u8, name: &str) -> Result<bool, TnLlamaError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid_argument(format!("{name} is not a boolean"))),
    }
}

fn required_ref<'a, T>(value: *const T, name: &str) -> Result<&'a T, TnLlamaError> {
    unsafe { value.as_ref() }.ok_or_else(|| invalid_argument(format!("{name} is null")))
}

fn required_mut<'a, T>(value: *mut T, name: &str) -> Result<&'a mut T, TnLlamaError> {
    unsafe { value.as_mut() }.ok_or_else(|| invalid_argument(format!("{name} is null")))
}

fn turn_permit(turn: &TurnHandle) -> Result<&TurnPermit, TnLlamaError> {
    turn.0
        .as_ref()
        .ok_or_else(|| session_error("tn_turn_consumed", "turn permit was already consumed"))
}

fn free_handle<T>(handle: *mut T) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        if !handle.is_null() {
            drop(Box::from_raw(handle));
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_wrapper_contains_panics_without_exporting_a_test_hook() {
        let mut error = OwnedBytes::EMPTY;
        let status = catch_ffi(
            &mut error,
            || Ok(()),
            || -> Result<(), TnLlamaError> { panic!("private panic payload") },
        );

        assert_eq!(status, STATUS_PANIC);
        let bytes = unsafe { std::slice::from_raw_parts(error.ptr, error.len) };
        assert_eq!(bytes, PANIC_ERROR_JSON);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(bytes).unwrap(),
            json!({
                "code": "tn_ffi_panic",
                "stage": "ffi",
                "message": "panic contained at the C ABI boundary",
            })
        );
        assert!(!bytes
            .windows(b"private panic payload".len())
            .any(|window| { window == b"private panic payload" }));
        tn_bytes_free(&mut error);
        assert!(error.ptr.is_null());
        assert_eq!(error.len, 0);
    }
}
