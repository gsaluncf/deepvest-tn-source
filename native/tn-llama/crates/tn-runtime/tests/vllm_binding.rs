mod governed_support;
mod support;

use serde_json::{json, Value};
use support::{fixture_config, SessionHarness};
use tn_runtime::{ExecutionBackend, GovernedRuntime, State, VllmBackend, VllmExecutionIdentity};
use tn_test_tools::build_real_tn_fixture;

fn backend() -> VllmBackend {
    VllmBackend::new(
        "engine-test",
        &"a".repeat(64),
        &"b".repeat(64),
        &"c".repeat(64),
        "hermes",
        "98dff2a81d747d1dba01a47f939f48c3526d4206",
    )
    .unwrap()
}
fn harness() -> SessionHarness {
    let fixture = build_real_tn_fixture().unwrap();
    let runtime = GovernedRuntime::from_config_with_backend(
        fixture_config(&fixture),
        ExecutionBackend::Vllm(backend()),
    )
    .unwrap();
    SessionHarness::from_fixture_and_runtime(fixture, runtime, "held request")
}
fn manifest() -> Vec<u8> {
    let mut sampling: serde_json::Map<String, Value> = tn_runtime::SAMPLING_FIELDS
        .iter()
        .map(|key| (key.to_string(), Value::Null))
        .collect();
    sampling.extend(json!({"n":1,"max_tokens":256,"temperature":0.0,"output_kind":2,"detokenize":true,"skip_clone":false}).as_object().unwrap().clone());
    serde_json::to_vec(&json!({
        "schema":"tn.vllm.execution.v1", "engine_instance":"engine-test",
        "model_bundle_sha256":"a".repeat(64), "tokenizer_sha256":"b".repeat(64),
        "template_sha256":"c".repeat(64),"parser":"hermes",
        "request_id":"turn-internal", "external_request_id":"turn-external",
        "prompt_token_ids":[1,2,3],
        "sampling_params":sampling,
        "reasoning_ended":null,"reasoning_parser_kwargs":null,"arrival_time":1.0,"priority":0,"trace_headers":null,"cache_salt":null,"data_parallel_rank":null,"mm_features":null,"pooling_params":null,"lora_request":null,"prompt_embeds":null,"prompt_is_token_ids":null,"resumable":false,"abort_immediately":false,"session_id":null
    })).unwrap()
}
fn identity() -> VllmExecutionIdentity {
    VllmExecutionIdentity {
        engine_instance: "engine-test".into(),
        request_id: "turn-internal".into(),
        external_request_id: "turn-external".into(),
        client_index: 0,
        destination: "engine-0".into(),
        wave: 0,
    }
}
#[test]
fn vllm_executes_the_actual_native_governance_transition() {
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    turn.permit
        .begin_vllm(identity(), &turn.request_json, b"rendered", &manifest())
        .unwrap();
    h.session
        .accept_model_action(turn.permit, h.get_governance_result().native())
        .unwrap();
    assert_eq!(h.session.state(), State::CommitmentEcho);
    let next = h.session.next_model_turn().unwrap();
    assert!(String::from_utf8_lossy(&next.request_json).contains("continue_with_governance"));
    assert!(!String::from_utf8_lossy(&next.request_json).contains("held request"));
}
#[test]
fn changed_tokens_are_terminal_before_execution() {
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    let mut altered: Value = serde_json::from_slice(&manifest()).unwrap();
    altered["prompt_token_ids"] = json!([9]);
    assert!(turn
        .permit
        .begin_vllm(
            identity(),
            &turn.request_json,
            b"rendered",
            &serde_json::to_vec(&altered).unwrap()
        )
        .is_err());
    assert_eq!(h.session.state(), State::Failed);
}
#[test]
fn vllm_cannot_enter_through_llama_begin() {
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    assert!(turn
        .permit
        .begin(1, 0, &turn.request_json, b"rendered", &manifest())
        .is_err());
}
#[test]
fn abort_invalidates_prepared_permit() {
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    h.session.abort().unwrap();
    assert!(turn
        .permit
        .begin_vllm(identity(), &turn.request_json, b"rendered", &manifest())
        .is_err());
}
#[test]
fn cross_session_result_is_rejected() {
    let mut a = harness();
    let mut b = harness();
    let turn = a.session.next_model_turn().unwrap();
    let other = b.session.next_model_turn().unwrap();
    other
        .permit
        .prepare(&other.request_json, b"rendered", &manifest())
        .unwrap();
    other
        .permit
        .begin_vllm(identity(), &other.request_json, b"rendered", &manifest())
        .unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    turn.permit
        .begin_vllm(identity(), &turn.request_json, b"rendered", &manifest())
        .unwrap();
    assert!(b
        .session
        .accept_model_action(turn.permit, a.get_governance_result().native())
        .is_err());
    assert_eq!(b.session.state(), State::Failed);
}

#[test]
fn every_bound_execution_component_is_checked_again() {
    for (field, replacement) in [
        ("model_bundle_sha256", json!("d".repeat(64))),
        ("tokenizer_sha256", json!("d".repeat(64))),
        ("template_sha256", json!("d".repeat(64))),
        ("parser", json!("another-parser")),
        ("request_id", json!("different-id")),
        ("reasoning_ended", json!(true)),
    ] {
        let mut h = harness();
        let turn = h.session.next_model_turn().unwrap();
        turn.permit
            .prepare(&turn.request_json, b"rendered", &manifest())
            .unwrap();
        let mut changed: Value = serde_json::from_slice(&manifest()).unwrap();
        changed[field] = replacement;
        assert!(
            turn.permit
                .begin_vllm(
                    identity(),
                    &turn.request_json,
                    b"rendered",
                    &serde_json::to_vec(&changed).unwrap()
                )
                .is_err(),
            "{field}"
        );
        assert_eq!(h.session.state(), State::Failed);
    }
}

#[test]
fn changed_sampling_or_rendered_prompt_cannot_begin() {
    for change_prompt in [false, true] {
        let mut h = harness();
        let turn = h.session.next_model_turn().unwrap();
        turn.permit
            .prepare(&turn.request_json, b"rendered", &manifest())
            .unwrap();
        let mut changed: Value = serde_json::from_slice(&manifest()).unwrap();
        if !change_prompt {
            changed["sampling_params"]["structured_outputs"] = json!({"json_object":true});
        }
        assert!(turn
            .permit
            .begin_vllm(
                identity(),
                &turn.request_json,
                if change_prompt {
                    b"different"
                } else {
                    b"rendered"
                },
                &serde_json::to_vec(&changed).unwrap()
            )
            .is_err());
    }
}

#[test]
fn no_extra_engine_mode_or_unknown_setting_is_accepted() {
    for (field, value) in [
        ("resumable", json!(true)),
        ("lora_request", json!({"id":1})),
        ("mm_features", json!([])),
        ("cache_salt", json!("cache")),
        ("unrecognized_field", json!(true)),
    ] {
        let mut h = harness();
        let turn = h.session.next_model_turn().unwrap();
        let mut changed: Value = serde_json::from_slice(&manifest()).unwrap();
        changed[field] = value;
        assert!(
            turn.permit
                .prepare(
                    &turn.request_json,
                    b"rendered",
                    &serde_json::to_vec(&changed).unwrap()
                )
                .is_err(),
            "{field}"
        );
    }
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    let mut changed: Value = serde_json::from_slice(&manifest()).unwrap();
    changed["sampling_params"]["plugin"] = json!("anything");
    assert!(turn
        .permit
        .prepare(
            &turn.request_json,
            b"rendered",
            &serde_json::to_vec(&changed).unwrap()
        )
        .is_err());
}

#[test]
fn repeated_begin_and_mismatched_routing_are_rejected() {
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    turn.permit
        .begin_vllm(identity(), &turn.request_json, b"rendered", &manifest())
        .unwrap();
    assert!(turn
        .permit
        .begin_vllm(identity(), &turn.request_json, b"rendered", &manifest())
        .is_err());
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    let mut wrong = identity();
    wrong.request_id = "other".into();
    assert!(turn
        .permit
        .begin_vllm(wrong, &turn.request_json, b"rendered", &manifest())
        .is_err());
}

#[test]
fn llama_runtime_rejects_vllm_manifest() {
    let mut h = SessionHarness::new("llama");
    let turn = h.session.next_model_turn().unwrap();
    assert!(turn
        .permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .is_err());
}

#[test]
fn one_vllm_runtime_owns_independent_concurrent_sessions() {
    let mut fixture = governed_support::Fixture::new();
    let request = fixture.request("deepvest");
    fixture.config.max_active_sessions = 2;
    let runtime = GovernedRuntime::from_config_with_backend(
        fixture.config,
        ExecutionBackend::Vllm(backend()),
    )
    .unwrap();
    let mut first = runtime.start_session(&request).unwrap();
    let mut second = runtime.start_session(&request).unwrap();
    assert!(runtime.start_session(&request).is_err());
    let turn = second.next_model_turn().unwrap();
    first.abort().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    turn.permit
        .begin_vllm(identity(), &turn.request_json, b"rendered", &manifest())
        .unwrap();
    assert_eq!(second.state(), State::GovernanceRead);
    assert!(runtime.start_session(&request).is_ok());
    second.abort().unwrap();
}

#[test]
fn vllm_audit_records_its_real_identity_without_llama_slots() {
    let mut h = harness();
    let turn = h.session.next_model_turn().unwrap();
    turn.permit
        .prepare(&turn.request_json, b"rendered", &manifest())
        .unwrap();
    turn.permit
        .begin_vllm(identity(), &turn.request_json, b"rendered", &manifest())
        .unwrap();
    let records = std::fs::read_to_string(fixture_config(&h._fixture).audit_path).unwrap();
    let record = records
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .find(|row| row["kind"] == "turn_begin_authorized")
        .unwrap();
    assert!(record["llama_task_id"].is_null());
    assert!(record["llama_slot_id"].is_null());
    let facts: Value = serde_json::from_str(record["facts_json"].as_str().unwrap()).unwrap();
    assert_eq!(facts["execution_identity"]["backend"], "vllm");
    assert_eq!(facts["execution_identity"]["request_id"], "turn-internal");
    assert_eq!(facts["execution_identity"]["destination"], "engine-0");
}

#[test]
fn omitted_effective_fields_are_not_defaulted_by_the_binding() {
    for sampling in [false, true] {
        let mut h = harness();
        let turn = h.session.next_model_turn().unwrap();
        let mut value: Value = serde_json::from_slice(&manifest()).unwrap();
        if sampling {
            value["sampling_params"]
                .as_object_mut()
                .unwrap()
                .remove("stop");
        } else {
            value.as_object_mut().unwrap().remove("reasoning_ended");
        }
        assert!(turn
            .permit
            .prepare(
                &turn.request_json,
                b"rendered",
                &serde_json::to_vec(&value).unwrap()
            )
            .is_err());
    }
}
