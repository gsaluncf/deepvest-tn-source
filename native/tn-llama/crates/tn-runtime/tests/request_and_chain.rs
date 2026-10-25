use serde_json::{json, Map, Value};
use std::sync::{Arc, Barrier};
use std::thread;
use tn_runtime::{
    parse_and_verify_request, AcceptedRequest, AdmissionController, AdmissionStage, ChainStore,
    GovernanceKeyProvenance, GovernanceReader, RequestLimits, SourceNamespace, VerifiedRow,
};
use tn_test_tools::build_real_tn_fixture;

const OFFICIAL_CANONICAL_ROW: &str = r#"{"device_identity":"did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK","event_id":"0198d65e-9c00-7000-8000-000000000001","event_type":"tn.llama.request","level":"info","prev_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","row_hash":"sha256:568aa1ee7735405eb0d5f2f15d6567a92b424794cce0460c3241ddf19fdd0fee","sequence":1,"signature":"","timestamp":"2026-08-23T12:00:00.000000Z","tn.agents":{"ciphertext":"AQIDBA==","field_hashes":{"consequences":"hmac-sha256:v1:0000000000000000000000000000000000000000000000000000000000000011","do_not_use_for":"hmac-sha256:v1:0000000000000000000000000000000000000000000000000000000000000012","instruction":"hmac-sha256:v1:0000000000000000000000000000000000000000000000000000000000000013","on_violation_or_error":"hmac-sha256:v1:0000000000000000000000000000000000000000000000000000000000000014","policy":"hmac-sha256:v1:0000000000000000000000000000000000000000000000000000000000000015","use_for":"hmac-sha256:v1:0000000000000000000000000000000000000000000000000000000000000016"}}}"#;

#[test]
fn accepts_official_hash_valid_genesis_request() {
    let raw = fixture_bytes("official-genesis-request.json");
    let accepted = parse_and_verify_request(raw, &RequestLimits::test()).unwrap();

    assert_eq!(accepted.row.event_type(), "tn.llama.request");
    assert_eq!(accepted.messages[0].content, "Reply with exactly READY.");
    assert!(accepted.row.row_hash().starts_with("sha256:"));
    assert_eq!(accepted.row.recipient_source_json(), OFFICIAL_CANONICAL_ROW);
    assert_eq!(
        accepted.request_digest,
        "sha256:3e91536a53e249b31e6380ac8cf40f686174a054c419050c79e27c28feb138b0"
    );
}

#[test]
fn changed_stored_row_hash_is_rejected() {
    let raw = fixture_with_flipped_stored_row_hash_last_nibble();
    let error = parse_and_verify_request(&raw, &RequestLimits::test()).unwrap_err();

    assert_eq!(error.code(), "tn_row_hash_mismatch");
    assert_eq!(error.stage(), AdmissionStage::RowHash);
}

#[test]
fn changed_hash_bound_row_value_is_rejected() {
    let raw = request_mutation(|request| {
        row_object_mut(request).insert(
            "timestamp".to_string(),
            Value::String("2026-08-23T12:00:09.000000Z".to_string()),
        );
    });
    let error = parse_and_verify_request(&raw, &RequestLimits::test()).unwrap_err();

    assert_eq!(error.code(), "tn_row_hash_mismatch");
    assert_eq!(error.stage(), AdmissionStage::RowHash);
}

#[test]
fn real_signed_encrypted_groups_survive_canonical_admission_and_open_only_with_capability() {
    let fixture = build_real_tn_fixture().unwrap();
    let mut limits = RequestLimits::test();
    limits.expected_device_identity = Some(fixture.device_identity.clone());
    limits.expected_event_type = "precisely.risk.enriched".to_string();
    limits.allowed_input_groups = ["default", "precisely", "firm_data"]
        .into_iter()
        .map(str::to_string)
        .collect();
    limits.require_signature = true;

    let accepted = parse_and_verify_request(&fixture.row1_request_json, &limits).unwrap();
    assert_eq!(
        accepted.row.event_id(),
        serde_json::from_slice::<Value>(&fixture.row1_request_json).unwrap()["tn_row"]["event_id"]
    );
    assert!(accepted.row.signature_verified());
    assert_eq!(
        accepted.row.encrypted_group_manifest(),
        std::collections::BTreeMap::from([
            (
                "default".to_string(),
                vec!["request_id".to_string(), "run_id".to_string()],
            ),
            (
                "firm_data".to_string(),
                vec!["customer_account".to_string(), "report_purpose".to_string()]
            ),
            (
                "precisely".to_string(),
                vec![
                    "address".to_string(),
                    "property_id".to_string(),
                    "wildfire_risk_score".to_string()
                ]
            ),
            (
                "tn.agents".to_string(),
                vec![
                    "consequences".to_string(),
                    "do_not_use_for".to_string(),
                    "governs_sections".to_string(),
                    "instruction".to_string(),
                    "on_violation_or_error".to_string(),
                    "policy".to_string(),
                    "use_for".to_string()
                ]
            )
        ])
    );
    let wire = std::str::from_utf8(&fixture.row1_request_json).unwrap();
    for protected in ["P-98765", "123 Pine Ln", "FIRM-42", "underwriting"] {
        assert!(
            !wire.contains(protected),
            "protected value leaked: {protected}"
        );
    }

    let reader = GovernanceReader::from_paths(
        &fixture.receiver_key_dir,
        &fixture.input_index_key_path,
        GovernanceKeyProvenance::new(&fixture.ceremony_id, "tn.agents", 0),
    )
    .unwrap();
    let admitted = reader
        .open_group_and_verify(&accepted.row, "precisely")
        .unwrap();
    assert_eq!(
        serde_json::to_value(admitted.plaintext).unwrap(),
        json!({
            "property_id": "P-98765",
            "address": "123 Pine Ln, CA",
            "wildfire_risk_score": 92
        })
    );
    let mut changed: Value = serde_json::from_slice(&fixture.row1_request_json).unwrap();
    changed["tn_row"]["precisely"]["field_hashes"]["wildfire_risk_score"] =
        json!("hmac-sha256:v1:0000000000000000000000000000000000000000000000000000000000000000");
    let error =
        parse_and_verify_request(&serde_json::to_vec(&changed).unwrap(), &limits).unwrap_err();
    assert_eq!(error.code(), "tn_row_hash_mismatch");
    assert_eq!(error.stage(), AdmissionStage::RowHash);
}

#[test]
fn zero_message_array_is_rejected_at_request_shape() {
    let raw = request_mutation(|request| request["messages"] = json!([]));
    let error = parse_and_verify_request(&raw, &RequestLimits::test()).unwrap_err();

    assert_eq!(error.code(), "tn_request_shape");
    assert_eq!(error.stage(), AdmissionStage::RequestShape);
}

#[test]
fn media_shaped_content_is_rejected_at_request_shape() {
    let raw = request_mutation(|request| {
        request["messages"][0]["content"] = json!([
            {"type": "text", "text": "Reply with exactly READY."}
        ]);
    });
    let error = parse_and_verify_request(&raw, &RequestLimits::test()).unwrap_err();

    assert_eq!(error.code(), "tn_request_shape");
    assert_eq!(error.stage(), AdmissionStage::RequestShape);
}

#[test]
fn duplicate_or_client_owned_protocol_fields_are_rejected() {
    for raw in [
        duplicate_purpose_json().to_vec(),
        assistant_message_json(),
        client_tools_json(),
    ] {
        assert!(parse_and_verify_request(&raw, &RequestLimits::test()).is_err());
    }
}

#[test]
fn normative_raw_request_rejections_have_exact_code_and_stage() {
    struct Case {
        name: &'static str,
        raw: Vec<u8>,
        limits: RequestLimits,
        code: &'static str,
        stage: AdmissionStage,
    }

    let valid = fixture_bytes("official-genesis-request.json").to_vec();
    let mut body_limits = RequestLimits::test();
    body_limits.max_request_bytes = valid.len() - 1;
    let mut message_limits = RequestLimits::test();
    message_limits.max_message_bytes = "Reply with exactly READY.".len() - 1;

    let cases = vec![
        Case {
            name: "body over limit",
            raw: valid.clone(),
            limits: body_limits,
            code: "tn_request_too_large",
            stage: AdmissionStage::Parse,
        },
        Case {
            name: "message over limit",
            raw: valid,
            limits: message_limits,
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "duplicate wrapper key",
            raw: duplicate_purpose_json().to_vec(),
            limits: RequestLimits::test(),
            code: "tn_duplicate_key",
            stage: AdmissionStage::Parse,
        },
        Case {
            name: "duplicate message key",
            raw: duplicate_message_role_json().to_vec(),
            limits: RequestLimits::test(),
            code: "tn_duplicate_key",
            stage: AdmissionStage::Parse,
        },
        Case {
            name: "duplicate TN header key",
            raw: duplicate_tn_header_json().to_vec(),
            limits: RequestLimits::test(),
            code: "tn_duplicate_key",
            stage: AdmissionStage::Parse,
        },
        Case {
            name: "duplicate group-block key",
            raw: duplicate_group_member_json().to_vec(),
            limits: RequestLimits::test(),
            code: "tn_duplicate_key",
            stage: AdmissionStage::Parse,
        },
        Case {
            name: "duplicate governance commitment key",
            raw: duplicate_governance_hash_json().to_vec(),
            limits: RequestLimits::test(),
            code: "tn_duplicate_key",
            stage: AdmissionStage::Parse,
        },
        Case {
            name: "missing wrapper member",
            raw: request_mutation(|request| {
                wrapper_object_mut(request).remove("purpose");
            }),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "extra wrapper member",
            raw: request_mutation(|request| {
                wrapper_object_mut(request).insert("unexpected".into(), json!(true));
            }),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "missing stream",
            raw: request_mutation(|request| {
                wrapper_object_mut(request).remove("stream");
            }),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "stream true",
            raw: request_mutation(|request| request["stream"] = json!(true)),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "stream wrong type",
            raw: request_mutation(|request| request["stream"] = json!("false")),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "wrong protocol",
            raw: request_mutation(|request| request["protocol"] = json!("tn.llama.request.v2")),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "empty purpose",
            raw: request_mutation(|request| request["purpose"] = json!("")),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "empty content",
            raw: request_mutation(|request| request["messages"][0]["content"] = json!("")),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "two messages",
            raw: request_mutation(|request| {
                request["messages"] = json!([
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"}
                ]);
            }),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "non-user role",
            raw: assistant_message_json(),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "client tools",
            raw: client_tools_json(),
            limits: RequestLimits::test(),
            code: "tn_request_shape",
            stage: AdmissionStage::RequestShape,
        },
        Case {
            name: "missing canonical TN header",
            raw: request_mutation(|request| {
                row_object_mut(request).remove("timestamp");
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "extra canonical TN header",
            raw: request_mutation(|request| {
                row_object_mut(request).insert("unexpected_header".into(), json!("value"));
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "wrong TN header type",
            raw: request_mutation(|request| {
                row_object_mut(request).insert("sequence".into(), json!("1"));
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "missing group-block member",
            raw: request_mutation(|request| {
                governance_group_mut(request).remove("ciphertext");
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "extra group-block member",
            raw: request_mutation(|request| {
                governance_group_mut(request).insert("nonce".into(), json!("unexpected"));
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "second encrypted group",
            raw: request_mutation(|request| {
                let second = request["tn_row"]["tn.agents"].clone();
                row_object_mut(request).insert("business".into(), second);
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "wrong group ciphertext type",
            raw: request_mutation(|request| {
                governance_group_mut(request).insert("ciphertext".into(), json!(1));
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "missing governance commitment name",
            raw: request_mutation(|request| {
                governance_hashes_mut(request).remove("instruction");
            }),
            limits: RequestLimits::test(),
            code: "tn_governance_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "extra governance commitment name",
            raw: request_mutation(|request| {
                governance_hashes_mut(request).insert("unexpected".into(), json!("token"));
            }),
            limits: RequestLimits::test(),
            code: "tn_row_hash_mismatch",
            stage: AdmissionStage::RowHash,
        },
        Case {
            name: "wrong governance commitment type",
            raw: request_mutation(|request| {
                governance_hashes_mut(request).insert("instruction".into(), json!(1));
            }),
            limits: RequestLimits::test(),
            code: "tn_envelope_shape",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "configured device mismatch",
            raw: request_mutation(|request| {
                row_object_mut(request).insert("device_identity".into(), json!("did:key:zWrong"));
            }),
            limits: RequestLimits::test(),
            code: "tn_source_namespace_mismatch",
            stage: AdmissionStage::RowShape,
        },
        Case {
            name: "configured event mismatch",
            raw: request_mutation(|request| {
                row_object_mut(request).insert("event_type".into(), json!("tn.llama.other"));
            }),
            limits: RequestLimits::test(),
            code: "tn_source_namespace_mismatch",
            stage: AdmissionStage::RowShape,
        },
    ];

    for case in cases {
        let error = match parse_and_verify_request(&case.raw, &case.limits) {
            Ok(_) => panic!("{} unexpectedly passed", case.name),
            Err(error) => error,
        };
        assert_eq!(error.code(), case.code, "{} code", case.name);
        assert_eq!(error.stage(), case.stage, "{} stage", case.name);
    }
}

#[test]
fn commits_two_rows_then_rejects_an_independently_valid_stale_fork() {
    let store = ChainStore::new(SourceNamespace::test(), 1);
    let row1 = official_row("chain-row-1.json");
    let row2a = official_row("chain-row-2a.json");
    let row2b = official_row("chain-row-2b-valid-fork.json");

    store.reserve(&row1, "s1").unwrap().commit().unwrap();
    store.reserve(&row2a, "s2").unwrap().commit().unwrap();
    let error = store.reserve(&row2b, "s3").unwrap_err();

    assert_eq!(error.code(), "tn_chain_tip_mismatch");
    assert_eq!(error.stage(), AdmissionStage::Chain);
}

#[test]
fn abort_releases_but_does_not_advance_a_reservation() {
    let store = ChainStore::new(SourceNamespace::test(), 1);
    let row = official_row("chain-row-1.json");

    store.reserve(&row, "failed").unwrap().abort().unwrap();
    store.reserve(&row, "retry").unwrap().commit().unwrap();
}

#[test]
fn drop_releases_but_does_not_advance_a_reservation() {
    let store = ChainStore::new(SourceNamespace::test(), 1);
    let row = official_row("chain-row-1.json");

    drop(store.reserve(&row, "failed").unwrap());
    store.reserve(&row, "retry").unwrap().commit().unwrap();
}

#[test]
fn one_chain_key_has_at_most_one_active_reservation() {
    let store = ChainStore::new(SourceNamespace::test(), 1);
    let row = official_row("chain-row-1.json");
    let first = store.reserve(&row, "s1").unwrap();

    let error = store.reserve(&row, "s2").unwrap_err();
    assert_eq!(error.code(), "tn_chain_reservation_conflict");
    drop(first);
    assert_eq!(store.chain_namespaces(), 0);
}

#[test]
fn one_active_session_is_released_on_abort_and_drop() {
    let admission = AdmissionController::new(SourceNamespace::test(), 1, 1);
    let first = admission.admit(accepted("chain-row-1.json"), "s1").unwrap();
    assert_eq!(
        admission
            .admit(accepted("chain-row-1.json"), "s2")
            .unwrap_err()
            .code(),
        "tn_active_session_limit"
    );
    first.abort().unwrap();
    drop(admission.admit(accepted("chain-row-1.json"), "s3").unwrap());
    assert_eq!(admission.active_sessions(), 0);
    assert_eq!(admission.chain_namespaces(), 0);
}

#[test]
fn admission_commit_advances_chain_once_and_releases_session() {
    let admission = AdmissionController::new(SourceNamespace::test(), 1, 1);

    admission
        .admit(accepted("chain-row-1.json"), "s1")
        .unwrap()
        .commit()
        .unwrap();
    assert_eq!(admission.active_sessions(), 0);
    assert_eq!(admission.chain_namespaces(), 1);
    admission
        .admit(accepted("chain-row-2a.json"), "s2")
        .unwrap()
        .commit()
        .unwrap();
}

#[test]
fn concurrent_admissions_allow_exactly_one_active_session() {
    let admission = Arc::new(AdmissionController::new(SourceNamespace::test(), 1, 1));
    let start = Arc::new(Barrier::new(3));
    let finish = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();

    for session_id in ["race-1", "race-2"] {
        let admission = Arc::clone(&admission);
        let start = Arc::clone(&start);
        let finish = Arc::clone(&finish);
        workers.push(thread::spawn(move || {
            let request = accepted("chain-row-1.json");
            start.wait();
            let result = admission.admit(request, session_id);
            finish.wait();
            result.is_ok()
        }));
    }

    start.wait();
    finish.wait();
    let successes = workers
        .into_iter()
        .map(|worker| usize::from(worker.join().unwrap()))
        .sum::<usize>();

    assert_eq!(successes, 1);
    assert_eq!(admission.active_sessions(), 0);
}

#[test]
fn rejected_unconfigured_namespace_does_not_consume_capacity() {
    let admission = AdmissionController::new(
        SourceNamespace::new("did:key:zUnconfigured", "tn.llama.request"),
        1,
        1,
    );
    let request = accepted("chain-row-1.json");

    let error = admission.admit(request, "poison").unwrap_err();

    assert_eq!(error.code(), "tn_source_namespace_mismatch");
    assert_eq!(admission.active_sessions(), 0);
    assert_eq!(admission.chain_namespaces(), 0);
}

#[test]
fn zero_chain_namespace_cap_rejects_without_consuming_a_session() {
    let admission = AdmissionController::new(SourceNamespace::test(), 1, 0);

    let error = admission
        .admit(accepted("chain-row-1.json"), "no-capacity")
        .unwrap_err();

    assert_eq!(error.code(), "tn_chain_namespace_limit");
    assert_eq!(admission.active_sessions(), 0);
    assert_eq!(admission.chain_namespaces(), 0);
}

fn fixture_bytes(name: &str) -> &'static [u8] {
    match name {
        "official-genesis-request.json" => include_bytes!("fixtures/official-genesis-request.json"),
        "chain-row-1.json" => include_bytes!("fixtures/chain-row-1.json"),
        "chain-row-2a.json" => include_bytes!("fixtures/chain-row-2a.json"),
        "chain-row-2b-valid-fork.json" => {
            include_bytes!("fixtures/chain-row-2b-valid-fork.json")
        }
        _ => panic!("unknown fixture: {name}"),
    }
}

fn official_row(name: &str) -> VerifiedRow {
    accepted(name).row
}

fn accepted(name: &str) -> AcceptedRequest {
    let row: Value = serde_json::from_slice(fixture_bytes(name)).unwrap();
    let request = json!({
        "protocol": "tn.llama.request.v1",
        "tn_row": row,
        "messages": [{"role": "user", "content": "test request"}],
        "purpose": "task-1-chain-test",
        "stream": false,
    });
    let raw = serde_json::to_vec(&request).unwrap();
    parse_and_verify_request(&raw, &RequestLimits::test()).unwrap()
}

fn request_mutation(mutate: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut request: Value =
        serde_json::from_slice(fixture_bytes("official-genesis-request.json")).unwrap();
    mutate(&mut request);
    serde_json::to_vec(&request).unwrap()
}

fn wrapper_object_mut(request: &mut Value) -> &mut Map<String, Value> {
    request.as_object_mut().unwrap()
}

fn row_object_mut(request: &mut Value) -> &mut Map<String, Value> {
    request["tn_row"].as_object_mut().unwrap()
}

fn governance_group_mut(request: &mut Value) -> &mut Map<String, Value> {
    request["tn_row"]["tn.agents"].as_object_mut().unwrap()
}

fn governance_hashes_mut(request: &mut Value) -> &mut Map<String, Value> {
    request["tn_row"]["tn.agents"]["field_hashes"]
        .as_object_mut()
        .unwrap()
}

fn fixture_with_flipped_stored_row_hash_last_nibble() -> Vec<u8> {
    request_mutation(|request| {
        let hash = request["tn_row"]["row_hash"].as_str().unwrap();
        let mut changed = hash.to_string();
        let previous = changed.pop().unwrap();
        changed.push(if previous == 'f' { 'e' } else { 'f' });
        request["tn_row"]["row_hash"] = Value::String(changed);
    })
}

fn assistant_message_json() -> Vec<u8> {
    request_mutation(|request| request["messages"][0]["role"] = json!("assistant"))
}

fn client_tools_json() -> Vec<u8> {
    request_mutation(|request| {
        wrapper_object_mut(request).insert("tools".into(), json!([]));
    })
}

fn duplicate_purpose_json() -> &'static [u8] {
    br#"{"purpose":"first","purpose":"second"}"#
}

fn duplicate_message_role_json() -> &'static [u8] {
    br#"{"messages":[{"role":"user","role":"assistant","content":"x"}]}"#
}

fn duplicate_tn_header_json() -> &'static [u8] {
    br#"{"tn_row":{"event_type":"tn.llama.request","event_type":"tn.llama.other"}}"#
}

fn duplicate_group_member_json() -> &'static [u8] {
    br#"{"tn_row":{"tn.agents":{"ciphertext":"AQ==","ciphertext":"Ag=="}}}"#
}

fn duplicate_governance_hash_json() -> &'static [u8] {
    br#"{"tn_row":{"tn.agents":{"field_hashes":{"instruction":"one","instruction":"two"}}}}"#
}
