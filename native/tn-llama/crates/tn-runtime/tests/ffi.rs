mod governed_support;
mod support;

use std::collections::BTreeSet;
use std::fs;
use std::ptr::{self, null, null_mut};

use serde_json::{json, Value};
use tn_test_tools::{build_real_tn_fixture, RealTnFixture};

use support::{canonical_json, literal_runnable_manifest};

const TN_STATUS_OK: i32 = 0;
const TN_STATUS_ERROR: i32 = 1;
const TN_TURN_GOVERNANCE_READ: u32 = 1;

#[repr(C)]
struct RuntimeHandle {
    _private: [u8; 0],
}

#[repr(C)]
struct SessionHandle {
    _private: [u8; 0],
}

#[repr(C)]
struct TurnHandle {
    _private: [u8; 0],
}

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

#[repr(C)]
struct NativeToolCall {
    id: BytesView,
    name: BytesView,
    normalized_arguments: BytesView,
}
#[repr(C)]
struct NativeToolResultInput {
    call_count: usize,
    call: *const NativeToolCall,
    content: BytesView,
    raw_model_output: BytesView,
    truncated: u8,
    stop_limit: u8,
}

extern "C" {
    fn tn_session_accept_model_action(
        session: *mut SessionHandle,
        turn: *mut TurnHandle,
        result: *const NativeToolResultInput,
        out_outcome: *mut u32,
        out_release_json: *mut OwnedBytes,
        out_error: *mut OwnedBytes,
    ) -> i32;
    fn tn_runtime_create(
        config_json: BytesView,
        out_runtime: *mut *mut RuntimeHandle,
        out_error: *mut OwnedBytes,
    ) -> i32;
    fn tn_session_create(
        runtime: *mut RuntimeHandle,
        request_json: BytesView,
        out_session: *mut *mut SessionHandle,
        out_error: *mut OwnedBytes,
    ) -> i32;
    fn tn_session_next_model_turn(
        session: *mut SessionHandle,
        out_turn: *mut *mut TurnHandle,
        out_phase: *mut u32,
        out_request_json: *mut OwnedBytes,
        out_error: *mut OwnedBytes,
    ) -> i32;
    fn tn_turn_prepare(
        turn: *mut TurnHandle,
        request_json: BytesView,
        rendered_prompt: BytesView,
        runnable_manifest: BytesView,
        out_error: *mut OwnedBytes,
    ) -> i32;
    fn tn_turn_begin(
        turn: *mut TurnHandle,
        task_id: i64,
        slot_id: i32,
        request_json: BytesView,
        rendered_prompt: BytesView,
        runnable_manifest: BytesView,
        out_error: *mut OwnedBytes,
    ) -> i32;
    fn tn_session_abort(session: *mut SessionHandle, out_error: *mut OwnedBytes) -> i32;
    fn tn_bytes_free(value: *mut OwnedBytes);
    fn tn_turn_free(turn: *mut TurnHandle);
    fn tn_session_free(session: *mut SessionHandle);
    fn tn_runtime_free(runtime: *mut RuntimeHandle);
}

#[test]
fn standalone_ffi_releases_exact_native_wire_and_reclaims_failed_sessions() {
    standalone_ffi_release(false);
}

#[test]
fn standalone_ffi_executes_required_native_tool_and_continues_to_release() {
    standalone_ffi_release(true);
}

fn standalone_ffi_release(with_tool: bool) {
    let mut fixture = governed_support::Fixture::new();
    if with_tool {
        fixture.require_compliance();
    }
    let request = fixture.request("calculation");
    let mut runtime = null_mut();
    let mut session = null_mut();
    let mut error = empty_owned();
    unsafe {
        assert_eq!(
            tn_runtime_create(view(&fixture.runtime_json), &mut runtime, &mut error),
            TN_STATUS_OK
        );
        assert_eq!(
            tn_session_create(runtime, view(&request), &mut session, &mut error),
            TN_STATUS_OK
        );
        for phase_index in 1..=if with_tool { 5 } else { 4 } {
            let (turn, phase, body) = next_turn(session);
            assert_eq!(phase, phase_index.min(4));
            let body_value: Value = serde_json::from_slice(&body).unwrap();
            let control: Value =
                serde_json::from_str(body_value["messages"][1]["content"].as_str().unwrap())
                    .unwrap();
            let (name, args) = match phase {
                1 => (
                    "get_governance",
                    json!({"row_hash":control["row_hash"],"request_digest":control["request_digest"]}),
                ),
                2 => (
                    "continue_with_governance",
                    serde_json::from_str(
                        body_value["messages"].as_array().unwrap().last().unwrap()["content"]
                            .as_str()
                            .unwrap(),
                    )
                    .unwrap(),
                ),
                3 => ("select_sections", json!({"sections":["finance"]})),
                4 if with_tool && phase_index == 4 => (
                    "log_compliance_creation",
                    json!({"source_object_id":fixture.manifest["publications"]["calculation"]["object_id"],"action":"generate_report"}),
                ),
                4 => (
                    "emit_governed_record",
                    json!({"report_html":"<p>Maximum drawdown is 25%.</p>"}),
                ),
                _ => unreachable!(),
            };
            let manifest =
                literal_runnable_manifest(body_value["max_tokens"].as_u64().unwrap() as u32);
            assert_eq!(
                tn_turn_prepare(
                    turn,
                    view(&body),
                    view(b"rendered-v2"),
                    view(&manifest),
                    &mut error
                ),
                TN_STATUS_OK
            );
            assert_eq!(
                tn_turn_begin(
                    turn,
                    41,
                    0,
                    view(&body),
                    view(b"rendered-v2"),
                    view(&manifest),
                    &mut error
                ),
                TN_STATUS_OK
            );
            let arguments = serde_json::to_vec(&args).unwrap();
            let call_id = format!("ffi-call-{phase_index}");
            let call = NativeToolCall {
                id: view(call_id.as_bytes()),
                name: view(name.as_bytes()),
                normalized_arguments: view(&arguments),
            };
            let result = NativeToolResultInput {
                call_count: 1,
                call: &call,
                content: view(b""),
                raw_model_output: view(b"native fixture tool call"),
                truncated: 0,
                stop_limit: 0,
            };
            let mut outcome = 0;
            let mut release = empty_owned();
            assert_eq!(
                tn_session_accept_model_action(
                    session,
                    turn,
                    &result,
                    &mut outcome,
                    &mut release,
                    &mut error
                ),
                TN_STATUS_OK
            );
            if name == "emit_governed_record" {
                assert_eq!(outcome, 2);
                let released: Value = serde_json::from_slice(&copy_and_free(&mut release)).unwrap();
                assert_eq!(released["protocol"], "tn.llama.release.v2");
                let object = tn_core::governed::GovernedObject::parse(
                    released["output_object"].as_str().unwrap(),
                )
                .unwrap();
                assert_eq!(object.id(), released["output_object_id"]);
                if with_tool {
                    let tool_wires = released["tool_objects"].as_array().unwrap();
                    assert_eq!(tool_wires.len(), 1);
                    let tool =
                        tn_core::governed::GovernedObject::parse(tool_wires[0].as_str().unwrap())
                            .unwrap();
                    assert_eq!(tool.object_type(), "compliance.creation.logged");
                    let receipt: Value = serde_json::from_str(
                        body_value["messages"].as_array().unwrap().last().unwrap()["content"]
                            .as_str()
                            .unwrap(),
                    )
                    .unwrap();
                    assert_eq!(receipt["output_object_id"], tool.id());
                } else {
                    assert!(released.get("tool_objects").is_none());
                }
            } else {
                assert_eq!(outcome, 1);
                assert_empty(release);
            }
            tn_turn_free(turn);
        }
        tn_session_free(session);
        let mut malformed: Value = serde_json::from_slice(&request).unwrap();
        malformed["protocol"] = json!("tn.llama.request.v1");
        session = ptr::dangling_mut();
        assert_eq!(
            tn_session_create(
                runtime,
                view(&serde_json::to_vec(&malformed).unwrap()),
                &mut session,
                &mut error
            ),
            TN_STATUS_ERROR
        );
        assert!(session.is_null());
        assert_error_code(&mut error, "tn_request_shape");
        assert_eq!(
            tn_session_create(runtime, view(&request), &mut session, &mut error),
            TN_STATUS_OK
        );
        assert_eq!(tn_session_abort(session, &mut error), TN_STATUS_OK);
        tn_session_free(session);
        tn_runtime_free(runtime);
        assert_empty(error);
    }
}

#[test]
fn null_nonempty_views_are_ffi_errors_and_empty_buffers_can_be_freed() {
    let mut runtime = ptr::dangling_mut::<RuntimeHandle>();
    let mut error = empty_owned();
    let status = unsafe {
        tn_runtime_create(
            BytesView {
                ptr: null(),
                len: 1,
            },
            &mut runtime,
            &mut error,
        )
    };
    assert_eq!(status, TN_STATUS_ERROR);
    assert!(runtime.is_null());
    unsafe {
        assert_error(
            &mut error,
            "tn_ffi_invalid_argument",
            "ffi",
            "config_json has a null pointer with a nonzero length",
        );
    }

    runtime = ptr::dangling_mut::<RuntimeHandle>();
    let status = unsafe { tn_runtime_create(view(b"{}"), &mut runtime, null_mut()) };
    assert_eq!(status, TN_STATUS_ERROR);
    assert!(runtime.is_null());

    unsafe {
        let mut empty = empty_owned();
        tn_bytes_free(&mut empty);
        assert_empty(empty);
    }
}

#[test]
fn begin_before_prepare_is_precise_and_terminalizes_the_session() {
    let fixture = build_real_tn_fixture().unwrap();
    let (runtime, session) = unsafe { create_runtime_session(&fixture) };
    let (turn, phase, body) = unsafe { next_turn(session) };
    assert_eq!(phase, TN_TURN_GOVERNANCE_READ);
    let manifest = literal_runnable_manifest(256);
    let mut error = empty_owned();
    let status = unsafe {
        tn_turn_begin(
            turn,
            301,
            0,
            view(&body),
            view(b"begin-before-prepare"),
            view(&manifest),
            &mut error,
        )
    };
    assert_eq!(status, TN_STATUS_ERROR);
    unsafe {
        assert_error(
            &mut error,
            "tn_turn_not_prepared",
            "session",
            "turn permit must be prepared before begin",
        );
    }

    let mut later_turn = ptr::dangling_mut::<TurnHandle>();
    let mut later_phase = 99;
    let mut later_body = OwnedBytes {
        ptr: ptr::dangling_mut::<u8>(),
        len: 99,
    };
    let mut later_error = empty_owned();
    let later_status = unsafe {
        tn_session_next_model_turn(
            session,
            &mut later_turn,
            &mut later_phase,
            &mut later_body,
            &mut later_error,
        )
    };
    assert_eq!(later_status, TN_STATUS_ERROR);
    assert!(later_turn.is_null());
    assert_eq!(later_phase, 0);
    assert_empty(later_body);
    unsafe {
        assert_error_code(&mut later_error, "tn_session_state");
        tn_turn_free(turn);
        tn_session_free(session);
        tn_runtime_free(runtime);
    }
}

#[test]
fn begin_rejects_changed_body_prompt_and_manifest_bindings() {
    enum Mutation {
        Body,
        Prompt,
        Manifest,
    }

    for mutation in [Mutation::Body, Mutation::Prompt, Mutation::Manifest] {
        let fixture = build_real_tn_fixture().unwrap();
        let (runtime, session) = unsafe { create_runtime_session(&fixture) };
        let (turn, _, body) = unsafe { next_turn(session) };
        let prompt = b"prepared-binding".to_vec();
        let manifest = literal_runnable_manifest(256);
        let mut error = empty_owned();
        let prepare_status = unsafe {
            tn_turn_prepare(
                turn,
                view(&body),
                view(&prompt),
                view(&manifest),
                &mut error,
            )
        };
        assert_eq!(prepare_status, TN_STATUS_OK);
        assert_empty(error);

        let mut begun_body = body.clone();
        let mut begun_prompt = prompt.clone();
        let mut begun_manifest = manifest.clone();
        match mutation {
            Mutation::Body => begun_body.push(b' '),
            Mutation::Prompt => begun_prompt.push(b'!'),
            Mutation::Manifest => {
                let mut value: Value = serde_json::from_slice(&begun_manifest).unwrap();
                value["task"]["tokens"][0] = json!(99);
                begun_manifest = canonical_json(&value);
            }
        }
        let begin_status = unsafe {
            tn_turn_begin(
                turn,
                401,
                0,
                view(&begun_body),
                view(&begun_prompt),
                view(&begun_manifest),
                &mut error,
            )
        };
        assert_eq!(begin_status, TN_STATUS_ERROR);
        unsafe {
            assert_error_code(&mut error, "tn_turn_binding_mismatch");
            tn_turn_free(turn);
            tn_session_free(session);
            tn_runtime_free(runtime);
        }
    }
}

#[test]
fn abort_export_takes_no_reason_and_returns_an_empty_error() {
    let fixture = build_real_tn_fixture().unwrap();
    let (runtime, session) = unsafe { create_runtime_session(&fixture) };
    let mut error = empty_owned();

    assert_eq!(
        unsafe { tn_session_abort(session, &mut error) },
        TN_STATUS_OK
    );
    assert_empty(error);

    unsafe {
        tn_session_free(session);
        tn_runtime_free(runtime);
    }
}

unsafe fn create_runtime_session(
    fixture: &RealTnFixture,
) -> (*mut RuntimeHandle, *mut SessionHandle) {
    let config = fs::read(fixture.temp.path().join("runtime-config.json")).unwrap();
    let mut runtime = null_mut();
    let mut error = empty_owned();
    assert_eq!(
        tn_runtime_create(view(&config), &mut runtime, &mut error),
        TN_STATUS_OK
    );
    assert!(!runtime.is_null());
    assert_empty(error);

    let mut session = null_mut();
    assert_eq!(
        tn_session_create(
            runtime,
            view(&fixture.row1_request_json),
            &mut session,
            &mut error,
        ),
        TN_STATUS_OK
    );
    assert!(!session.is_null());
    assert_empty(error);
    (runtime, session)
}

unsafe fn next_turn(session: *mut SessionHandle) -> (*mut TurnHandle, u32, Vec<u8>) {
    let mut turn = null_mut();
    let mut phase = 0;
    let mut body = empty_owned();
    let mut error = empty_owned();
    assert_eq!(
        tn_session_next_model_turn(session, &mut turn, &mut phase, &mut body, &mut error),
        TN_STATUS_OK
    );
    assert!(!turn.is_null());
    assert_empty(error);
    let bytes = copy_and_free(&mut body);
    assert!(!bytes.is_empty());
    (turn, phase, bytes)
}

fn view(bytes: &[u8]) -> BytesView {
    BytesView {
        ptr: if bytes.is_empty() {
            ptr::null()
        } else {
            bytes.as_ptr()
        },
        len: bytes.len(),
    }
}

const fn empty_owned() -> OwnedBytes {
    OwnedBytes {
        ptr: null_mut(),
        len: 0,
    }
}

fn assert_empty(value: OwnedBytes) {
    assert!(value.ptr.is_null());
    assert_eq!(value.len, 0);
}

unsafe fn copy_and_free(value: &mut OwnedBytes) -> Vec<u8> {
    let bytes = if value.len == 0 {
        Vec::new()
    } else {
        assert!(!value.ptr.is_null());
        std::slice::from_raw_parts(value.ptr, value.len).to_vec()
    };
    tn_bytes_free(value);
    assert_empty(*value);
    bytes
}

unsafe fn assert_error(error: &mut OwnedBytes, code: &str, stage: &str, message: &str) {
    let bytes = copy_and_free(error);
    let expected = json!({"code": code, "stage": stage, "message": message});
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
    assert_eq!(bytes, serde_json::to_vec(&expected).unwrap());
}

unsafe fn assert_error_code(error: &mut OwnedBytes, code: &str) {
    let bytes = copy_and_free(error);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["code"], code);
    assert_eq!(
        value.as_object().unwrap().keys().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            &"code".to_string(),
            &"message".to_string(),
            &"stage".to_string()
        ])
    );
}
