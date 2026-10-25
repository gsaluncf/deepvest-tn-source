mod support;

use std::collections::BTreeSet;
use std::fs;

use serde_json::{json, Map, Value};
use tn_runtime::{ActionOutcome, GovernedRuntime, NativeToolResult, Phase, State};
use tn_test_tools::build_real_tn_fixture;

use support::{
    advance_to_echo, canonical_json, fixture_config, literal_runnable_manifest, prepare_and_begin,
    SessionHarness,
};

#[test]
fn held_user_text_is_absent_until_exact_echo_matches() {
    let mut harness = SessionHarness::new("DO-NOT-RELEASE-BEFORE-ECHO");

    let model_turn1 = harness.session.next_model_turn().unwrap();
    assert_eq!(model_turn1.permit.phase(), Phase::GovernanceRead);
    let body1: Value = serde_json::from_slice(&model_turn1.request_json).unwrap();
    assert!(
        !String::from_utf8_lossy(&model_turn1.request_json).contains("DO-NOT-RELEASE-BEFORE-ECHO")
    );
    assert!(!String::from_utf8_lossy(&model_turn1.request_json)
        .contains(harness.governance["instruction"].as_str().unwrap()));
    assert_eq!(body1["tools"].as_array().unwrap().len(), 1);
    assert_eq!(body1["tools"][0]["function"]["name"], "get_governance");
    assert_eq!(body1["max_tokens"], 256);
    assert_tool_turn_settings(&body1, 256);
    assert_eq!(
        object_keys(&body1),
        set(&[
            "cache_prompt",
            "max_tokens",
            "messages",
            "n",
            "n_cache_reuse",
            "parallel_tool_calls",
            "reasoning_effort",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
        ])
    );
    let control1: Value =
        serde_json::from_str(body1["messages"][1]["content"].as_str().unwrap()).unwrap();
    assert_eq!(
        control1,
        json!({
            "phase": "governance_read",
            "purpose": "derived_html_report",
            "request_digest": harness.request_digest,
            "row_hash": harness.row_hash,
        })
    );
    assert_eq!(
        body1["tools"][0]["function"]["parameters"],
        json!({
            "type": "object",
            "properties": {
                "request_digest": {"type": "string"},
                "row_hash": {"type": "string"}
            },
            "required": ["request_digest", "row_hash"],
            "additionalProperties": false
        })
    );
    for withheld in harness.governance.values() {
        let wire = String::from_utf8_lossy(&model_turn1.request_json);
        if let Some(withheld) = withheld.as_str() {
            assert!(!wire.contains(withheld));
        } else {
            assert!(!wire.contains(&withheld.to_string()));
        }
    }
    for withheld in harness.commitments.values() {
        assert!(!String::from_utf8_lossy(&model_turn1.request_json).contains(withheld));
    }
    let manifest1 = literal_runnable_manifest(256);
    model_turn1
        .permit
        .prepare(
            &model_turn1.request_json,
            b"rendered-governance-read",
            &manifest1,
        )
        .unwrap();
    model_turn1
        .permit
        .begin(
            11,
            0,
            &model_turn1.request_json,
            b"rendered-governance-read",
            &manifest1,
        )
        .unwrap();
    harness
        .session
        .accept_model_action(model_turn1.permit, harness.get_governance_result().native())
        .map(|outcome| assert!(matches!(outcome, ActionOutcome::Continue)))
        .unwrap();

    let model_turn2 = harness.session.next_model_turn().unwrap();
    assert_eq!(model_turn2.permit.phase(), Phase::CommitmentEcho);
    assert!(
        !String::from_utf8_lossy(&model_turn2.request_json).contains("DO-NOT-RELEASE-BEFORE-ECHO")
    );
    let body2: Value = serde_json::from_slice(&model_turn2.request_json).unwrap();
    assert_eq!(
        body2["tools"][0]["function"]["name"],
        "continue_with_governance"
    );
    assert_eq!(body2["max_tokens"], 1536);
    assert_tool_turn_settings(&body2, 1536);
    assert_eq!(object_keys(&body2), object_keys(&body1));
    assert_eq!(body2["messages"].as_array().unwrap().len(), 4);
    assert_eq!(body2["messages"][2]["role"], "assistant");
    assert_eq!(
        body2["messages"][2]["tool_calls"][0]["id"],
        "call-governance-1"
    );
    assert_eq!(body2["messages"][3]["role"], "tool");
    assert_eq!(body2["messages"][3]["tool_call_id"], "call-governance-1");
    let tool_result: Value =
        serde_json::from_str(body2["messages"][3]["content"].as_str().unwrap()).unwrap();
    assert_eq!(tool_result["governance"], json!(harness.governance));
    assert_eq!(
        tool_result["governance_commitments"],
        json!(harness.commitments)
    );
    let schema_text = body2["tools"][0]["function"]["parameters"].to_string();
    assert!(!schema_text.contains("enum"));
    assert!(!schema_text.contains("const"));
    let manifest2 = literal_runnable_manifest(1536);
    model_turn2
        .permit
        .prepare(
            &model_turn2.request_json,
            b"rendered-commitment-echo",
            &manifest2,
        )
        .unwrap();
    model_turn2
        .permit
        .begin(
            12,
            0,
            &model_turn2.request_json,
            b"rendered-commitment-echo",
            &manifest2,
        )
        .unwrap();
    harness
        .session
        .accept_model_action(
            model_turn2.permit,
            harness.commitment_echo_result().native(),
        )
        .map(|outcome| assert!(matches!(outcome, ActionOutcome::Continue)))
        .unwrap();

    let model_turn3 = harness.session.next_model_turn().unwrap();
    assert_eq!(model_turn3.permit.phase(), Phase::SectionSelection);
    let body3: Value = serde_json::from_slice(&model_turn3.request_json).unwrap();
    assert!(
        String::from_utf8_lossy(&model_turn3.request_json).contains("DO-NOT-RELEASE-BEFORE-ECHO")
    );
    assert_eq!(body3["tools"][0]["function"]["name"], "select_sections");
    assert_eq!(object_keys(&body3), object_keys(&body1));
    assert_eq!(body3["max_tokens"], 2048);
    assert_tool_turn_settings(&body3, 2048);
    assert_eq!(body3["messages"].as_array().unwrap().len(), 2);
    assert_eq!(body3["messages"][0]["role"], "system");
    assert_eq!(body3["messages"][1]["role"], "user");
}

#[test]
fn turn_one_requires_one_complete_exact_bound_native_call() {
    enum Mutation {
        EmptyId,
        InvalidId,
        WrongName,
        Malformed,
        Duplicate,
        Extra,
        WrongRow,
        WrongRequest,
        Truncated,
        StopLimit,
    }
    let cases = [
        ("empty id", Mutation::EmptyId, "tn_tool_call_id_invalid"),
        ("invalid id", Mutation::InvalidId, "tn_tool_call_id_invalid"),
        ("wrong name", Mutation::WrongName, "tn_action_not_allowed"),
        (
            "malformed",
            Mutation::Malformed,
            "tn_tool_arguments_invalid",
        ),
        (
            "duplicate",
            Mutation::Duplicate,
            "tn_tool_arguments_invalid",
        ),
        ("extra", Mutation::Extra, "tn_tool_arguments_shape"),
        ("wrong row", Mutation::WrongRow, "tn_tool_row_hash_mismatch"),
        (
            "wrong request",
            Mutation::WrongRequest,
            "tn_tool_request_digest_mismatch",
        ),
        ("truncated", Mutation::Truncated, "tn_model_turn_incomplete"),
        (
            "stop limit",
            Mutation::StopLimit,
            "tn_model_turn_incomplete",
        ),
    ];

    for (name, mutation, code) in cases {
        let mut harness = SessionHarness::new(name);
        let turn = harness.session.next_model_turn().unwrap();
        prepare_and_begin(&turn, b"turn-one", 256, 21);
        let mut result = harness.get_governance_result();
        match mutation {
            Mutation::EmptyId => result.id.clear(),
            Mutation::InvalidId => result.id = vec![0xff],
            Mutation::WrongName => result.name = b"continue_with_governance".to_vec(),
            Mutation::Malformed => result.arguments = b"{".to_vec(),
            Mutation::Duplicate => {
                result.arguments = format!(
                    r#"{{"request_digest":"{}","row_hash":"{}","row_hash":"{}"}}"#,
                    harness.request_digest, harness.row_hash, harness.row_hash
                )
                .into_bytes();
            }
            Mutation::Extra => {
                let mut value: Value = serde_json::from_slice(&result.arguments).unwrap();
                value["extra"] = json!(true);
                result.arguments = canonical_json(&value);
            }
            Mutation::WrongRow => {
                let mut value: Value = serde_json::from_slice(&result.arguments).unwrap();
                value["row_hash"] = json!("sha256:wrong");
                result.arguments = canonical_json(&value);
            }
            Mutation::WrongRequest => {
                let mut value: Value = serde_json::from_slice(&result.arguments).unwrap();
                value["request_digest"] = json!("sha256:wrong");
                result.arguments = canonical_json(&value);
            }
            Mutation::Truncated => result.truncated = true,
            Mutation::StopLimit => result.stop_limit = true,
        }

        let error = harness
            .session
            .accept_model_action(turn.permit, result.native())
            .unwrap_err();

        assert_eq!(error.code(), code, "{name}");
        assert_eq!(harness.session.state(), State::Failed, "{name}");
        assert!(harness.session.next_model_turn().is_err(), "{name}");
    }
}

#[test]
fn real_governance_open_occurs_only_after_the_exact_turn_one_proposal() {
    let mut invalid = SessionHarness::new("deferred open invalid proposal");
    let turn = invalid.session.next_model_turn().unwrap();
    prepare_and_begin(&turn, b"turn-one", 256, 81);
    fs::remove_file(
        invalid
            ._fixture
            .receiver_key_dir
            .join("tn.agents.btn.mykit"),
    )
    .unwrap();
    let mut result = invalid.get_governance_result();
    mutate_argument(&mut result.arguments, |value| {
        value["request_digest"] = json!("sha256:wrong");
    });

    let error = invalid
        .session
        .accept_model_action(turn.permit, result.native())
        .unwrap_err();
    assert_eq!(error.code(), "tn_tool_request_digest_mismatch");

    let mut exact = SessionHarness::new("deferred open exact proposal");
    let turn = exact.session.next_model_turn().unwrap();
    prepare_and_begin(&turn, b"turn-one", 256, 82);
    fs::remove_file(exact._fixture.receiver_key_dir.join("tn.agents.btn.mykit")).unwrap();
    let result = exact.get_governance_result();

    let error = exact
        .session
        .accept_model_action(turn.permit, result.native())
        .unwrap_err();
    assert_eq!(error.code(), "protected_group_open_failed");
}

#[test]
fn permits_cannot_cross_sessions_or_accept_before_begin_or_survive_abort() {
    let mut before_begin = SessionHarness::new("before begin");
    let turn = before_begin.session.next_model_turn().unwrap();
    let result = before_begin.get_governance_result();
    let error = before_begin
        .session
        .accept_model_action(turn.permit, result.native())
        .unwrap_err();
    assert_eq!(error.code(), "tn_turn_not_begun");

    let mut origin = SessionHarness::new("origin session");
    let mut target = SessionHarness::new("target session");
    let origin_turn = origin.session.next_model_turn().unwrap();
    prepare_and_begin(&origin_turn, b"origin", 256, 83);
    let _target_turn = target.session.next_model_turn().unwrap();
    let result = target.get_governance_result();
    let error = target
        .session
        .accept_model_action(origin_turn.permit, result.native())
        .unwrap_err();
    assert_eq!(error.code(), "tn_turn_session_mismatch");
    assert_eq!(target.session.state(), State::Failed);

    let mut aborted = SessionHarness::new("aborted permit");
    let turn = aborted.session.next_model_turn().unwrap();
    aborted.session.abort().unwrap();
    let manifest = literal_runnable_manifest(256);
    let error = turn
        .permit
        .prepare(&turn.request_json, b"prompt", &manifest)
        .unwrap_err();
    assert_eq!(error.code(), "tn_turn_consumed");
}

#[test]
fn echo_requires_the_complete_exact_governance_values_and_commitments() {
    for field in tn_runtime::GOVERNANCE_FIELDS {
        for mutation in [
            "missing governance",
            "changed governance",
            "missing token",
            "changed token",
        ] {
            let mut harness = SessionHarness::new(&format!("{mutation} {field}"));
            let turn = advance_to_echo(&mut harness);
            prepare_and_begin(&turn, b"echo", 1536, 22);
            let mut result = harness.commitment_echo_result();
            let mut value: Value = serde_json::from_slice(&result.arguments).unwrap();
            match mutation {
                "missing governance" => {
                    value["governance"].as_object_mut().unwrap().remove(field);
                }
                "changed governance" => value["governance"][field] = json!("changed"),
                "missing token" => {
                    value["governance_commitments"]
                        .as_object_mut()
                        .unwrap()
                        .remove(field);
                }
                "changed token" => {
                    value["governance_commitments"][field] = json!("hmac-sha256:v1:changed")
                }
                _ => unreachable!(),
            }
            result.arguments = canonical_json(&value);

            let error = harness
                .session
                .accept_model_action(turn.permit, result.native())
                .unwrap_err();

            let expected = if mutation.starts_with("missing") {
                "tn_tool_arguments_shape"
            } else {
                "tn_governance_echo_mismatch"
            };
            assert_eq!(error.code(), expected, "{mutation} {field}");
            assert_eq!(harness.session.state(), State::Failed);
        }
    }
}

#[test]
fn typed_governance_extras_drive_the_echo_schema_and_canonical_match() {
    let mut fixture = build_real_tn_fixture().unwrap();
    fixture.row1_request_json = fixture.extra_governance_field_request_json.clone();
    fixture
        .governance
        .insert("governs_sections".to_string(), json!(["precisely"]));
    fixture
        .governance
        .insert("governance_revision".to_string(), json!(2));
    let runtime = GovernedRuntime::from_config(fixture_config(&fixture)).unwrap();
    let mut harness =
        SessionHarness::from_fixture_and_runtime(fixture, runtime, "typed governance echo");
    let turn = advance_to_echo(&mut harness);
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    let parameters = &body["tools"][0]["function"]["parameters"];
    assert_eq!(
        parameters["properties"]["governance"]["properties"]["governs_sections"],
        json!({"type": "array", "items":{"type":"string"}, "minItems":1, "maxItems":1})
    );
    assert_eq!(
        parameters["properties"]["governance"]["required"],
        json!([
            "consequences",
            "do_not_use_for",
            "governance_revision",
            "governs_sections",
            "instruction",
            "on_violation_or_error",
            "policy",
            "use_for"
        ])
    );
    assert_eq!(
        parameters["properties"]["governance"]["additionalProperties"],
        false
    );
    prepare_and_begin(&turn, b"typed-governance-echo", 1536, 23);
    let result = harness.commitment_echo_result();

    let outcome = harness
        .session
        .accept_model_action(turn.permit, result.native())
        .unwrap();

    assert!(matches!(outcome, ActionOutcome::Continue));
    assert_eq!(harness.session.state(), State::SectionSelection);
}

#[test]
fn echo_rejects_malformed_duplicate_extra_stale_and_incomplete_results() {
    enum Mutation {
        Malformed,
        Duplicate,
        ExtraOuter,
        ExtraGovernance,
        ExtraToken,
        WrongRow,
        WrongRequest,
        Truncated,
        StopLimit,
    }
    let cases = [
        (Mutation::Malformed, "tn_tool_arguments_invalid"),
        (Mutation::Duplicate, "tn_tool_arguments_invalid"),
        (Mutation::ExtraOuter, "tn_tool_arguments_shape"),
        (Mutation::ExtraGovernance, "tn_tool_arguments_shape"),
        (Mutation::ExtraToken, "tn_tool_arguments_shape"),
        (Mutation::WrongRow, "tn_tool_row_hash_mismatch"),
        (Mutation::WrongRequest, "tn_tool_request_digest_mismatch"),
        (Mutation::Truncated, "tn_model_turn_incomplete"),
        (Mutation::StopLimit, "tn_model_turn_incomplete"),
    ];

    for (index, (mutation, code)) in cases.into_iter().enumerate() {
        let mut harness = SessionHarness::new(&format!("echo negative {index}"));
        let turn = advance_to_echo(&mut harness);
        prepare_and_begin(&turn, b"echo", 1536, 30 + index as i64);
        let mut result = harness.commitment_echo_result();
        match mutation {
            Mutation::Malformed => result.arguments = b"{".to_vec(),
            Mutation::Duplicate => {
                result.arguments = format!(
                    r#"{{"governance":{},"governance":{},"governance_commitments":{},"request_digest":{},"row_hash":{}}}"#,
                    serde_json::to_string(&harness.governance).unwrap(),
                    serde_json::to_string(&harness.governance).unwrap(),
                    serde_json::to_string(&harness.commitments).unwrap(),
                    serde_json::to_string(&harness.request_digest).unwrap(),
                    serde_json::to_string(&harness.row_hash).unwrap(),
                )
                .into_bytes();
            }
            Mutation::ExtraOuter => mutate_argument(&mut result.arguments, |value| {
                value["extra"] = json!(true);
            }),
            Mutation::ExtraGovernance => mutate_argument(&mut result.arguments, |value| {
                value["governance"]["extra"] = json!("x");
            }),
            Mutation::ExtraToken => mutate_argument(&mut result.arguments, |value| {
                value["governance_commitments"]["extra"] = json!("x");
            }),
            Mutation::WrongRow => mutate_argument(&mut result.arguments, |value| {
                value["row_hash"] = json!("sha256:wrong");
            }),
            Mutation::WrongRequest => mutate_argument(&mut result.arguments, |value| {
                value["request_digest"] = json!("sha256:wrong");
            }),
            Mutation::Truncated => result.truncated = true,
            Mutation::StopLimit => result.stop_limit = true,
        }

        let error = harness
            .session
            .accept_model_action(turn.permit, result.native())
            .unwrap_err();
        assert_eq!(error.code(), code, "case {index}");
        assert_eq!(harness.session.state(), State::Failed);
    }
}

#[test]
fn reordered_complete_echo_objects_are_semantically_equal() {
    let mut harness = SessionHarness::new("reordered echo");
    let turn = advance_to_echo(&mut harness);
    prepare_and_begin(&turn, b"echo", 1536, 44);
    let mut result = harness.commitment_echo_result();
    let reversed_governance = harness
        .governance
        .iter()
        .rev()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<Map<_, _>>();
    let reversed_commitments = harness
        .commitments
        .iter()
        .rev()
        .map(|(name, value)| (name.clone(), Value::String(value.clone())))
        .collect::<Map<_, _>>();
    result.arguments = format!(
        r#"{{"row_hash":{},"request_digest":{},"governance_commitments":{},"governance":{}}}"#,
        serde_json::to_string(&harness.row_hash).unwrap(),
        serde_json::to_string(&harness.request_digest).unwrap(),
        serde_json::to_string(&reversed_commitments).unwrap(),
        serde_json::to_string(&reversed_governance).unwrap(),
    )
    .into_bytes();

    harness
        .session
        .accept_model_action(turn.permit, result.native())
        .map(|outcome| assert!(matches!(outcome, ActionOutcome::Continue)))
        .unwrap();

    assert_eq!(harness.session.state(), State::SectionSelection);
    assert!(harness.session.next_model_turn().is_ok());
}

#[test]
fn permit_order_reuse_and_task_binding_fail_closed() {
    struct Case {
        name: &'static str,
        run: fn(&mut SessionHarness) -> &'static str,
        code: &'static str,
    }

    fn begin_before_prepare(harness: &mut SessionHarness) -> &'static str {
        let turn = harness.session.next_model_turn().unwrap();
        let manifest = literal_runnable_manifest(256);
        turn.permit
            .begin(1, 0, &turn.request_json, b"prompt", &manifest)
            .unwrap_err()
            .code()
    }

    fn prepare_twice(harness: &mut SessionHarness) -> &'static str {
        let turn = harness.session.next_model_turn().unwrap();
        let manifest = literal_runnable_manifest(256);
        turn.permit
            .prepare(&turn.request_json, b"prompt", &manifest)
            .unwrap();
        turn.permit
            .prepare(&turn.request_json, b"prompt", &manifest)
            .unwrap_err()
            .code()
    }

    fn begin_twice(harness: &mut SessionHarness) -> &'static str {
        let turn = harness.session.next_model_turn().unwrap();
        let manifest = literal_runnable_manifest(256);
        turn.permit
            .prepare(&turn.request_json, b"prompt", &manifest)
            .unwrap();
        turn.permit
            .begin(1, 0, &turn.request_json, b"prompt", &manifest)
            .unwrap();
        turn.permit
            .begin(1, 0, &turn.request_json, b"prompt", &manifest)
            .unwrap_err()
            .code()
    }

    fn changed_body(harness: &mut SessionHarness) -> &'static str {
        let turn = harness.session.next_model_turn().unwrap();
        let manifest = literal_runnable_manifest(256);
        let mut body = turn.request_json.clone();
        body.push(b' ');
        turn.permit
            .prepare(&body, b"prompt", &manifest)
            .unwrap_err()
            .code()
    }

    fn changed_prompt_at_begin(harness: &mut SessionHarness) -> &'static str {
        let turn = harness.session.next_model_turn().unwrap();
        let manifest = literal_runnable_manifest(256);
        turn.permit
            .prepare(&turn.request_json, b"prompt", &manifest)
            .unwrap();
        turn.permit
            .begin(1, 0, &turn.request_json, b"changed", &manifest)
            .unwrap_err()
            .code()
    }

    fn changed_manifest_at_begin(harness: &mut SessionHarness) -> &'static str {
        let turn = harness.session.next_model_turn().unwrap();
        let manifest = literal_runnable_manifest(256);
        let mut changed_value: Value = serde_json::from_slice(&manifest).unwrap();
        changed_value["task"]["tokens"][0] = json!(99);
        let changed = canonical_json(&changed_value);
        turn.permit
            .prepare(&turn.request_json, b"prompt", &manifest)
            .unwrap();
        turn.permit
            .begin(1, 0, &turn.request_json, b"prompt", &changed)
            .unwrap_err()
            .code()
    }

    for case in [
        Case {
            name: "begin before prepare",
            run: begin_before_prepare,
            code: "tn_turn_not_prepared",
        },
        Case {
            name: "prepare twice",
            run: prepare_twice,
            code: "tn_turn_already_prepared",
        },
        Case {
            name: "begin twice",
            run: begin_twice,
            code: "tn_turn_already_begun",
        },
        Case {
            name: "changed body",
            run: changed_body,
            code: "tn_turn_binding_mismatch",
        },
        Case {
            name: "changed prompt at begin",
            run: changed_prompt_at_begin,
            code: "tn_turn_binding_mismatch",
        },
        Case {
            name: "changed manifest at begin",
            run: changed_manifest_at_begin,
            code: "tn_turn_binding_mismatch",
        },
    ] {
        let mut harness = SessionHarness::new(case.name);
        assert_eq!((case.run)(&mut harness), case.code, "{}", case.name);
        assert_eq!(harness.session.state(), State::Failed, "{}", case.name);
        assert!(harness.session.next_model_turn().is_err(), "{}", case.name);
    }
}

#[test]
fn zero_or_two_native_calls_never_open_governance_or_advance() {
    for call_count in [0, 2] {
        let mut harness = SessionHarness::new("call count test");
        let turn = harness.session.next_model_turn().unwrap();
        let manifest = literal_runnable_manifest(256);
        turn.permit
            .prepare(&turn.request_json, b"prompt", &manifest)
            .unwrap();
        turn.permit
            .begin(1, 0, &turn.request_json, b"prompt", &manifest)
            .unwrap();
        let result = NativeToolResult {
            call_count,
            call: None,
            content: b"",
            raw_model_output: b"native result",
            truncated: false,
            stop_limit: false,
        };

        let error = harness
            .session
            .accept_model_action(turn.permit, result)
            .unwrap_err();

        assert_eq!(error.code(), "tn_tool_call_count");
        assert_eq!(harness.session.state(), State::Failed);
        assert!(harness.session.next_model_turn().is_err());
    }
}

fn mutate_argument(arguments: &mut Vec<u8>, mutate: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_slice(arguments).unwrap();
    mutate(&mut value);
    *arguments = canonical_json(&value);
}

fn object_keys(value: &Value) -> BTreeSet<&str> {
    value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect()
}

fn set<'a>(values: &[&'a str]) -> BTreeSet<&'a str> {
    values.iter().copied().collect()
}

fn assert_tool_turn_settings(body: &Value, max_tokens: u32) {
    assert_eq!(body["cache_prompt"], false);
    assert_eq!(body["max_tokens"], max_tokens);
    assert_eq!(body["n"], 1);
    assert_eq!(body["n_cache_reuse"], 0);
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["reasoning_effort"], "none");
    assert_eq!(body["stream"], false);
    assert_eq!(body["temperature"], 0);
    assert_eq!(body["tool_choice"], "required");
}
