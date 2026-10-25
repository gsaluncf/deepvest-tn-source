mod support;

use std::fs;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use support::{
    advance_to_echo, advance_to_selection, canonical_json, fixture_config,
    literal_runnable_manifest, prepare_and_begin, SessionHarness,
};
use tn_runtime::{
    ActionOutcome, AuditEvent, AuditFacts, AuditRecord, DecisionAudit, GovernedRuntime, ModelTurn,
    ReportRelease, SensitiveTraceEvent, SensitiveTraceKind, SensitiveTraceSink, State,
};
use tn_test_tools::build_real_tn_fixture;

#[test]
fn runtime_accepts_group_capability_writer_authority_for_a_configured_event_type() {
    let fixture = build_real_tn_fixture().unwrap();
    let mut config = fixture_config(&fixture);
    config.expected_event_type = "configured.source.event".to_string();

    GovernedRuntime::from_config(config).unwrap();
}

#[test]
fn one_long_lived_runtime_owns_admission_and_abort_or_drop_releases_it() {
    let fixture = build_real_tn_fixture().unwrap();
    let runtime = GovernedRuntime::from_config(fixture_config(&fixture)).unwrap();

    let mut first = runtime.start_session(&fixture.row1_request_json).unwrap();
    let error = runtime
        .start_session(&fixture.row1_request_json)
        .unwrap_err();
    assert_eq!(error.code(), "tn_active_session_limit");
    first.abort().unwrap();
    assert_eq!(first.state(), State::Failed);

    let second = runtime.start_session(&fixture.row1_request_json).unwrap();
    drop(second);
    runtime.start_session(&fixture.row1_request_json).unwrap();
}

#[test]
fn a_second_turn_cannot_be_issued_while_one_is_outstanding() {
    let mut harness = SessionHarness::new("one outstanding turn");
    let _turn = harness.session.next_model_turn().unwrap();

    let error = harness.session.next_model_turn().unwrap_err();

    assert_eq!(error.code(), "tn_turn_outstanding");
    assert_eq!(harness.session.state(), State::Failed);
}

#[test]
fn concurrent_start_session_calls_admit_exactly_one_request() {
    let fixture = build_real_tn_fixture().unwrap();
    let runtime = Arc::new(GovernedRuntime::from_config(fixture_config(&fixture)).unwrap());
    let raw = Arc::new(fixture.row1_request_json.clone());
    let barrier = Arc::new(Barrier::new(3));
    let finish = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();

    for _ in 0..2 {
        let runtime = Arc::clone(&runtime);
        let raw = Arc::clone(&raw);
        let barrier = Arc::clone(&barrier);
        let finish = Arc::clone(&finish);
        workers.push(thread::spawn(move || {
            barrier.wait();
            let result = runtime.start_session(&raw);
            finish.wait();
            result.is_ok()
        }));
    }
    barrier.wait();
    finish.wait();
    let successes = workers
        .into_iter()
        .map(|worker| usize::from(worker.join().unwrap()))
        .sum::<usize>();

    assert_eq!(successes, 1);
}

#[test]
fn admission_audit_records_verified_then_reserved_and_binds_terminal_denial() {
    let fixture = build_real_tn_fixture().unwrap();
    let audit = Arc::new(RecordingAudit::passing());
    let trace = Arc::new(RecordingTrace::passing());
    let runtime = runtime_with_sinks(&fixture, Arc::clone(&audit), trace);
    let harness = SessionHarness::from_fixture_and_runtime(
        fixture,
        runtime.clone(),
        "truthful admission audit",
    );

    let admission_error = runtime.start_session(&harness.raw_request).unwrap_err();
    assert_eq!(admission_error.code(), "tn_active_session_limit");

    let events = audit.events.lock().unwrap();
    assert_eq!(events.len(), 4);
    let verified = &events[0];
    assert_eq!(verified.kind, "row_verified");
    assert_eq!(verified.state_before, "received");
    assert_eq!(verified.state_after, "row_verified");
    assert_eq!(verified.target_row_hash, harness.row_hash);
    assert_eq!(verified.request_digest, harness.request_digest);
    assert_eq!(verified.outcome, "allow");
    assert_eq!(verified.error_code, None);
    assert_eq!(
        verified.facts,
        AuditFacts {
            hashes_verified: Some(true),
            signature_verified: Some(true),
            ..AuditFacts::default()
        }
    );

    let reserved = &events[1];
    assert_eq!(reserved.kind, "chain_reserved");
    assert_eq!(reserved.state_before, "row_verified");
    assert_eq!(reserved.state_after, "governance_read");
    assert_eq!(reserved.target_row_hash, harness.row_hash);
    assert_eq!(reserved.request_digest, harness.request_digest);
    assert_eq!(reserved.outcome, "allow");
    assert_eq!(reserved.error_code, None);
    assert_eq!(reserved.facts, AuditFacts::default());

    let denied_verified = &events[2];
    assert_eq!(denied_verified.kind, "row_verified");
    assert_eq!(denied_verified.state_before, "received");
    assert_eq!(denied_verified.state_after, "row_verified");
    assert_eq!(denied_verified.target_row_hash, harness.row_hash);
    assert_eq!(denied_verified.request_digest, harness.request_digest);
    assert_eq!(
        denied_verified.facts,
        AuditFacts {
            hashes_verified: Some(true),
            signature_verified: Some(true),
            ..AuditFacts::default()
        }
    );

    let denied = &events[3];
    assert_eq!(denied.kind, "terminal_error");
    assert_eq!(denied.state_before, "row_verified");
    assert_eq!(denied.state_after, "failed");
    assert_eq!(denied.target_row_hash, harness.row_hash);
    assert_eq!(denied.request_digest, harness.request_digest);
    assert_eq!(denied.outcome, "deny");
    assert_eq!(
        denied.error_code.as_deref(),
        Some("tn_active_session_limit")
    );
    assert_eq!(denied.facts, AuditFacts::default());
}

#[test]
fn pre_verification_parse_failure_records_only_unbound_safe_error_evidence() {
    let fixture = build_real_tn_fixture().unwrap();
    let audit = Arc::new(RecordingAudit::passing());
    let trace = Arc::new(RecordingTrace::passing());
    let runtime = runtime_with_sinks(&fixture, Arc::clone(&audit), trace);

    let error = runtime.start_session(b"{").unwrap_err();

    assert_eq!(error.code(), "tn_request_json_invalid");
    let events = audit.events.lock().unwrap();
    assert_eq!(events.len(), 1);
    let denied = &events[0];
    assert_eq!(denied.kind, "terminal_error");
    assert_eq!(denied.state_before, "received");
    assert_eq!(denied.state_after, "failed");
    assert_eq!(denied.target_row_hash, "");
    assert_eq!(denied.request_digest, "");
    assert_eq!(denied.outcome, "deny");
    assert_eq!(
        denied.error_code.as_deref(),
        Some("tn_request_json_invalid")
    );
    assert_eq!(denied.facts, AuditFacts::default());
}

#[test]
fn admission_error_is_masked_only_when_its_terminal_audit_append_fails() {
    let fixture = build_real_tn_fixture().unwrap();
    let audit = Arc::new(RecordingAudit::failing_once("terminal_error"));
    let trace = Arc::new(RecordingTrace::passing());
    let runtime = runtime_with_sinks(&fixture, Arc::clone(&audit), trace);
    let first = runtime.start_session(&fixture.row1_request_json).unwrap();

    let error = runtime
        .start_session(&fixture.row1_request_json)
        .unwrap_err();

    assert_eq!(error.code(), "tn_audit_append_failed");
    assert_eq!(audit.failed_appends(), 1);
    drop(first);
}

#[test]
fn every_rejected_runnable_mode_terminalizes_before_begin_or_a_later_turn() {
    let base = literal_runnable_manifest(256);
    let mut cases = vec![
        ("schema", manifest_with(&base, &["schema"], json!("other"))),
        (
            "context shift",
            manifest_with(&base, &["effective_ctx_shift"], json!(true)),
        ),
        (
            "task type",
            manifest_with(&base, &["task", "type"], json!("embedding")),
        ),
        (
            "empty tokens",
            manifest_with(&base, &["task", "tokens"], json!([])),
        ),
        (
            "mtmd tokens",
            manifest_with(&base, &["task", "tokens_has_mtmd"], json!(true)),
        ),
        (
            "assigned slot",
            manifest_with(&base, &["task", "id_slot"], json!(0)),
        ),
        (
            "parent task",
            manifest_with(&base, &["task", "id_parent"], json!(7)),
        ),
        (
            "child task",
            manifest_with(&base, &["task", "child_count"], json!(1)),
        ),
        (
            "cli mode",
            manifest_with(&base, &["task", "cli"], json!(true)),
        ),
        (
            "cli prompt",
            manifest_with(&base, &["task", "cli_prompt_byte_length"], json!(1)),
        ),
        (
            "cli file",
            manifest_with(&base, &["task", "cli_file_count"], json!(1)),
        ),
        (
            "streaming",
            manifest_with(&base, &["params", "stream"], json!(true)),
        ),
        (
            "prompt cache",
            manifest_with(&base, &["params", "cache_prompt"], json!(true)),
        ),
        (
            "cache reuse",
            manifest_with(&base, &["params", "n_cache_reuse"], json!(1)),
        ),
        (
            "multiple completions",
            manifest_with(&base, &["params", "n_cmpl"], json!(2)),
        ),
        (
            "phase token budget",
            manifest_with(&base, &["params", "n_predict"], json!(127)),
        ),
        (
            "lora",
            manifest_with(
                &base,
                &["params", "lora"],
                json!([{"path": "adapter.gguf"}]),
            ),
        ),
        (
            "result type",
            manifest_with(&base, &["params", "res_type"], json!("tokens")),
        ),
        (
            "speculation",
            manifest_with(&base, &["speculative", "types"], json!(["draft"])),
        ),
        (
            "unknown field",
            manifest_with(&base, &["unknown"], json!(0)),
        ),
        (
            "wrong type",
            manifest_with(&base, &["effective_ctx_shift"], json!("false")),
        ),
    ];
    let base_text = String::from_utf8(base.clone()).unwrap();
    cases.push((
        "duplicate field",
        base_text
            .replacen(
                "\"schema\":",
                "\"schema\":\"tn.llama.runnable-task.v1\",\"schema\":",
                1,
            )
            .into_bytes(),
    ));

    let fixture = build_real_tn_fixture().unwrap();
    let audit = Arc::new(RecordingAudit::passing());
    let trace = Arc::new(RecordingTrace::passing());
    let runtime = runtime_with_sinks(&fixture, Arc::clone(&audit), trace);
    for (index, (name, manifest)) in cases.into_iter().enumerate() {
        let mut session = runtime.start_session(&fixture.row1_request_json).unwrap();
        let turn = session.next_model_turn().unwrap();

        let error = turn
            .permit
            .prepare(&turn.request_json, b"rendered prompt", &manifest)
            .unwrap_err();

        assert_eq!(error.code(), "tn_runnable_manifest_invalid", "{name}");
        assert_eq!(session.state(), State::Failed, "{name}");
        assert_eq!(
            session.next_model_turn().unwrap_err().code(),
            "tn_session_state",
            "{name}"
        );
        assert!(
            audit
                .events
                .lock()
                .unwrap()
                .iter()
                .all(|event| event.kind != "turn_begin_authorized"),
            "{name} at case {index}"
        );
    }
}

#[test]
fn sequential_sessions_append_to_one_runtime_evidence_without_truncating_row_one() {
    let fixture = build_real_tn_fixture().unwrap();
    let audit_path = fixture.temp.path().join("audit.ndjson");
    let trace_dir = fixture.temp.path().join("raw-trace");
    let row2 = fixture.row2_request_json.clone();
    let runtime = GovernedRuntime::from_config(fixture_config(&fixture)).unwrap();
    let mut harness =
        SessionHarness::from_fixture_and_runtime(fixture, runtime.clone(), "first sequential row");
    complete_firm_report(&mut harness, 101);
    let audit_after_row1 = fs::read(&audit_path).unwrap();
    let trace_path = fs::read_dir(&trace_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let trace_after_row1 = fs::read(&trace_path).unwrap();

    let mut row2_session = runtime.start_session(&row2).unwrap();
    let row2_turn = row2_session.next_model_turn().unwrap();
    row2_turn
        .permit
        .prepare(
            &row2_turn.request_json,
            b"second-governance",
            &literal_runnable_manifest(256),
        )
        .unwrap();

    let audit_after_row2 = fs::read(&audit_path).unwrap();
    let trace_after_row2 = fs::read(&trace_path).unwrap();
    assert!(audit_after_row2.starts_with(&audit_after_row1));
    assert!(audit_after_row2.len() > audit_after_row1.len());
    assert!(trace_after_row2.starts_with(&trace_after_row1));
    assert!(trace_after_row2.len() > trace_after_row1.len());
    assert_eq!(fs::read_dir(&trace_dir).unwrap().count(), 1);
}

#[test]
fn begin_audit_and_trace_bind_the_exact_body_prompt_manifest_and_task() {
    let fixture = build_real_tn_fixture().unwrap();
    let audit = Arc::new(RecordingAudit::passing());
    let trace = Arc::new(RecordingTrace::passing());
    let runtime = runtime_with_sinks(&fixture, Arc::clone(&audit), Arc::clone(&trace));
    let mut harness = SessionHarness::from_fixture_and_runtime(fixture, runtime, "digest binding");
    let turn = harness.session.next_model_turn().unwrap();
    let request = turn.request_json.clone();
    let manifest = literal_runnable_manifest(256);
    let prompt = b"exact rendered prompt";
    turn.permit.prepare(&request, prompt, &manifest).unwrap();
    turn.permit
        .begin(102, 7, &request, prompt, &manifest)
        .unwrap();

    let events = audit.events.lock().unwrap();
    let begun = events
        .iter()
        .find(|event| event.kind == "turn_begin_authorized")
        .unwrap();
    assert_eq!(begun.llama_task_id, Some(102));
    assert_eq!(begun.llama_slot_id, Some(7));
    assert_eq!(
        begun.facts.model_request_digest.as_ref().unwrap().as_str(),
        domain_digest(b"tn.llama.model-request.v1\0", &request)
    );
    assert_eq!(
        begun
            .facts
            .rendered_prompt_digest
            .as_ref()
            .unwrap()
            .as_str(),
        domain_digest(b"tn.llama.rendered-prompt.v1\0", prompt)
    );
    assert_eq!(
        begun.facts.runnable_task_digest.as_ref().unwrap().as_str(),
        domain_digest(b"tn.llama.runnable-task.v1\0", &manifest)
    );
    drop(events);

    let traces = trace.events.lock().unwrap();
    assert_eq!(traces.len(), 3);
    assert_eq!(traces[0].kind, SensitiveTraceKind::Ingress);
    assert_eq!(
        traces[0].request_json.as_deref(),
        Some(harness.raw_request.as_slice())
    );
    assert_eq!(traces[1].kind, SensitiveTraceKind::TurnPrepared);
    assert_eq!(traces[1].request_json.as_deref(), Some(request.as_slice()));
    assert_eq!(
        traces[1].rendered_prompt.as_deref(),
        Some(prompt.as_slice())
    );
    assert_eq!(
        traces[1].runnable_manifest.as_deref(),
        Some(manifest.as_slice())
    );
    assert_eq!(traces[2].kind, SensitiveTraceKind::TurnStarted);
    assert_eq!(traces[2].task_id, Some(102));
    assert_eq!(traces[2].slot_id, Some(7));
}

#[test]
fn sensitive_local_trace_captures_the_exact_ingress_and_closed_release_bytes() {
    let fixture = build_real_tn_fixture().unwrap();
    let audit = Arc::new(RecordingAudit::passing());
    let trace = Arc::new(RecordingTrace::passing());
    let runtime = runtime_with_sinks(&fixture, audit, Arc::clone(&trace));
    let mut harness = SessionHarness::from_fixture_and_runtime(
        fixture,
        runtime,
        "trace this exact governed request",
    );
    let ingress = harness.raw_request.clone();

    let release = complete_firm_report(&mut harness, 111);
    let release_bytes = release.to_json_bytes().unwrap();

    let traces = trace.events.lock().unwrap();
    let ingress_event = traces
        .iter()
        .find(|event| event.kind == SensitiveTraceKind::Ingress)
        .unwrap();
    assert_eq!(
        ingress_event.request_json.as_deref(),
        Some(ingress.as_slice())
    );
    let release_event = traces
        .iter()
        .find(|event| event.kind == SensitiveTraceKind::Release)
        .unwrap();
    assert_eq!(
        release_event.parsed_content.as_deref(),
        Some(release_bytes.as_slice())
    );
}

#[test]
fn audit_failures_stop_at_each_real_boundary_and_precommit_failures_abort() {
    for kind in [
        "row_verified",
        "chain_reserved",
        "turn_begin_authorized",
        "tool_call_accepted",
        "governance_opened_and_verified",
        "commitments_matched",
        "sections_selected",
        "output_emitted_and_read_back",
    ] {
        let fixture = build_real_tn_fixture().unwrap();
        let audit = Arc::new(RecordingAudit::failing_once(kind));
        let trace = Arc::new(RecordingTrace::passing());
        let runtime = runtime_with_sinks(&fixture, Arc::clone(&audit), trace);

        if kind == "row_verified" || kind == "chain_reserved" {
            let error = runtime
                .start_session(&fixture.row1_request_json)
                .unwrap_err();
            assert_eq!(error.code(), "tn_audit_append_failed", "{kind}");
            runtime.start_session(&fixture.row1_request_json).unwrap();
            continue;
        }

        let mut harness = SessionHarness::from_fixture_and_runtime(fixture, runtime.clone(), kind);
        let error = match kind {
            "turn_begin_authorized" => {
                let turn = harness.session.next_model_turn().unwrap();
                let manifest = literal_runnable_manifest(256);
                turn.permit
                    .prepare(&turn.request_json, b"prompt", &manifest)
                    .unwrap();
                turn.permit
                    .begin(91, 0, &turn.request_json, b"prompt", &manifest)
                    .unwrap_err()
            }
            "tool_call_accepted" | "governance_opened_and_verified" => {
                let turn = harness.session.next_model_turn().unwrap();
                prepare_and_begin(&turn, b"turn-one", 256, 92);
                let result = harness.get_governance_result();
                harness
                    .session
                    .accept_model_action(turn.permit, result.native())
                    .unwrap_err()
            }
            "commitments_matched" => {
                let turn = advance_to_echo(&mut harness);
                prepare_and_begin(&turn, b"echo", 1536, 93);
                let result = harness.commitment_echo_result();
                harness
                    .session
                    .accept_model_action(turn.permit, result.native())
                    .unwrap_err()
            }
            "sections_selected" => {
                let turn = advance_to_selection(&mut harness);
                prepare_and_begin(&turn, b"selection", 2048, 94);
                let result = harness.section_selection_result(&["firm_data"]);
                harness
                    .session
                    .accept_model_action(turn.permit, result.native())
                    .unwrap_err()
            }
            "output_emitted_and_read_back" => {
                let turn = advance_to_firm_emit(&mut harness, 94);
                prepare_and_begin(&turn, b"report", 2048, 95);
                let result = harness.action_result(
                    "emit-audit-failure",
                    "emit_governed_record",
                    firm_report_proposal(),
                );
                harness
                    .session
                    .accept_model_action(turn.permit, result.native())
                    .unwrap_err()
            }
            _ => unreachable!(),
        };

        assert_eq!(error.code(), "tn_audit_append_failed", "{kind}");
        assert_eq!(harness.session.state(), State::Failed, "{kind}");
        runtime.start_session(&harness.raw_request).unwrap();
        assert_eq!(audit.failed_appends(), 1, "{kind}");
    }
}

#[test]
fn post_commit_audit_failure_consumes_row_and_returns_no_release() {
    let fixture = build_real_tn_fixture().unwrap();
    let row2 = fixture.row2_request_json.clone();
    let row1 = fixture.row1_request_json.clone();
    let audit = Arc::new(RecordingAudit::failing_once("source_chain_committed"));
    let trace = Arc::new(RecordingTrace::passing());
    let runtime = runtime_with_sinks(&fixture, audit, trace);
    let mut harness =
        SessionHarness::from_fixture_and_runtime(fixture, runtime.clone(), "post commit audit");
    let turn = advance_to_firm_emit(&mut harness, 95);
    prepare_and_begin(&turn, b"report", 2048, 96);
    let result = harness.action_result(
        "emit-post-commit-audit-failure",
        "emit_governed_record",
        firm_report_proposal(),
    );

    let error = harness
        .session
        .accept_model_action(turn.permit, result.native())
        .unwrap_err();

    assert_eq!(error.code(), "tn_audit_append_failed");
    assert_eq!(harness.session.state(), State::Failed);
    assert_eq!(
        runtime.start_session(&row1).unwrap_err().code(),
        "tn_chain_tip_mismatch"
    );
    runtime.start_session(&row2).unwrap();
}

#[test]
fn sensitive_trace_failures_stop_prepare_begin_and_model_result_boundaries() {
    let cases = [
        (SensitiveTraceKind::TurnPrepared, None),
        (SensitiveTraceKind::TurnStarted, None),
        (
            SensitiveTraceKind::ModelResult,
            Some(tn_runtime::Phase::GovernanceRead),
        ),
        (
            SensitiveTraceKind::ModelResult,
            Some(tn_runtime::Phase::GovernedAction),
        ),
    ];

    for (index, (kind, phase)) in cases.into_iter().enumerate() {
        let fixture = build_real_tn_fixture().unwrap();
        let audit = Arc::new(RecordingAudit::passing());
        let trace = Arc::new(RecordingTrace::failing_once(kind, phase));
        let runtime = runtime_with_sinks(&fixture, audit, Arc::clone(&trace));
        let mut harness = SessionHarness::from_fixture_and_runtime(
            fixture,
            runtime.clone(),
            &format!("trace failure {index}"),
        );

        let error = match (kind, phase) {
            (SensitiveTraceKind::TurnPrepared, _) => {
                let turn = harness.session.next_model_turn().unwrap();
                turn.permit
                    .prepare(
                        &turn.request_json,
                        b"prompt",
                        &literal_runnable_manifest(256),
                    )
                    .unwrap_err()
            }
            (SensitiveTraceKind::TurnStarted, _) => {
                let turn = harness.session.next_model_turn().unwrap();
                let manifest = literal_runnable_manifest(256);
                turn.permit
                    .prepare(&turn.request_json, b"prompt", &manifest)
                    .unwrap();
                turn.permit
                    .begin(96, 0, &turn.request_json, b"prompt", &manifest)
                    .unwrap_err()
            }
            (SensitiveTraceKind::ModelResult, Some(tn_runtime::Phase::GovernanceRead)) => {
                let turn = harness.session.next_model_turn().unwrap();
                prepare_and_begin(&turn, b"turn-one", 256, 97);
                let result = harness.get_governance_result();
                harness
                    .session
                    .accept_model_action(turn.permit, result.native())
                    .unwrap_err()
            }
            (SensitiveTraceKind::ModelResult, Some(tn_runtime::Phase::GovernedAction)) => {
                let turn = advance_to_firm_emit(&mut harness, 98);
                prepare_and_begin(&turn, b"report", 2048, 99);
                let result = harness.action_result(
                    "emit-trace-failure",
                    "emit_governed_record",
                    firm_report_proposal(),
                );
                harness
                    .session
                    .accept_model_action(turn.permit, result.native())
                    .unwrap_err()
            }
            _ => unreachable!(),
        };

        assert_eq!(error.code(), "tn_sensitive_trace_append_failed");
        assert_eq!(harness.session.state(), State::Failed);
        runtime.start_session(&harness.raw_request).unwrap();
        assert_eq!(trace.failed_appends(), 1);
    }
}

fn advance_to_firm_emit(harness: &mut SessionHarness, task_id: i64) -> ModelTurn {
    let selection = advance_to_selection(harness);
    prepare_and_begin(&selection, b"rendered-firm-selection", 2048, task_id);
    let result = harness.section_selection_result(&["firm_data"]);
    let outcome = harness
        .session
        .accept_model_action(selection.permit, result.native())
        .unwrap();
    assert!(matches!(outcome, ActionOutcome::Continue));
    harness.session.next_model_turn().unwrap()
}

fn complete_firm_report(harness: &mut SessionHarness, task_id: i64) -> ReportRelease {
    let emission = advance_to_firm_emit(harness, task_id);
    prepare_and_begin(&emission, b"rendered-firm-report", 2048, task_id + 1);
    let result = harness.action_result(
        "emit-firm-report",
        "emit_governed_record",
        firm_report_proposal(),
    );
    let ActionOutcome::Release(release) = harness
        .session
        .accept_model_action(emission.permit, result.native())
        .unwrap()
    else {
        panic!("firm report did not produce a closed release");
    };
    release
}

fn firm_report_proposal() -> Value {
    json!({
        "event_type": "report.generated",
        "public_sections": {
            "default": {
                "report_html": "<article><p>Customer FIRM-42 underwriting report.</p></article>"
            }
        },
        "governance": {
            "instruction": "Generate only the configured derived report.",
            "use_for": "Creating the governed HTML report requested by the user.",
            "do_not_use_for": "Any ungoverned output or unrelated purpose.",
            "consequences": "Reject the report and release no output when validation fails.",
            "on_violation_or_error": "Abort the governed session and record the failing gate."
        }
    })
}

fn runtime_with_sinks(
    fixture: &tn_test_tools::RealTnFixture,
    audit: Arc<RecordingAudit>,
    trace: Arc<RecordingTrace>,
) -> GovernedRuntime {
    GovernedRuntime::from_config_with_sink_factories(
        fixture_config(fixture),
        move |_| {
            let sink: Arc<dyn DecisionAudit> = audit.clone();
            Ok::<Arc<dyn DecisionAudit>, io::Error>(sink)
        },
        move |_| {
            let sink: Arc<dyn SensitiveTraceSink> = trace.clone();
            Ok::<Arc<dyn SensitiveTraceSink>, io::Error>(sink)
        },
    )
    .unwrap()
}

struct RecordingAudit {
    events: Mutex<Vec<AuditEvent>>,
    fail_kind: Option<&'static str>,
    failures_remaining: AtomicUsize,
    failed_appends: AtomicUsize,
}

impl RecordingAudit {
    fn passing() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            fail_kind: None,
            failures_remaining: AtomicUsize::new(0),
            failed_appends: AtomicUsize::new(0),
        }
    }

    fn failing_once(kind: &'static str) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            fail_kind: Some(kind),
            failures_remaining: AtomicUsize::new(1),
            failed_appends: AtomicUsize::new(0),
        }
    }

    fn failed_appends(&self) -> usize {
        self.failed_appends.load(Ordering::SeqCst)
    }
}

impl DecisionAudit for RecordingAudit {
    fn append(&self, event: AuditEvent) -> io::Result<AuditRecord> {
        if self.fail_kind == Some(event.kind.as_str())
            && self
                .failures_remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
        {
            self.failed_appends.fetch_add(1, Ordering::SeqCst);
            return Err(io::Error::other("injected audit failure"));
        }
        self.events.lock().unwrap().push(event);
        Ok(AuditRecord {
            prev_hash: "sha256:test-prev".to_string(),
            row_hash: "sha256:test-row".to_string(),
        })
    }
}

struct RecordingTrace {
    events: Mutex<Vec<TraceSnapshot>>,
    fail_kind: Option<SensitiveTraceKind>,
    fail_phase: Option<tn_runtime::Phase>,
    failures_remaining: AtomicUsize,
    failed_appends: AtomicUsize,
}

impl RecordingTrace {
    fn passing() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            fail_kind: None,
            fail_phase: None,
            failures_remaining: AtomicUsize::new(0),
            failed_appends: AtomicUsize::new(0),
        }
    }

    fn failing_once(kind: SensitiveTraceKind, phase: Option<tn_runtime::Phase>) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            fail_kind: Some(kind),
            fail_phase: phase,
            failures_remaining: AtomicUsize::new(1),
            failed_appends: AtomicUsize::new(0),
        }
    }

    fn failed_appends(&self) -> usize {
        self.failed_appends.load(Ordering::SeqCst)
    }
}

impl SensitiveTraceSink for RecordingTrace {
    fn append(&self, event: SensitiveTraceEvent<'_>) -> io::Result<()> {
        let phase_matches = self.fail_phase.is_none() || self.fail_phase == event.phase;
        if self.fail_kind == Some(event.kind)
            && phase_matches
            && self
                .failures_remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
        {
            self.failed_appends.fetch_add(1, Ordering::SeqCst);
            return Err(io::Error::other("injected trace failure"));
        }
        self.events.lock().unwrap().push(TraceSnapshot {
            kind: event.kind,
            task_id: event.task_id,
            slot_id: event.slot_id,
            request_json: event.request_json.map(<[u8]>::to_vec),
            rendered_prompt: event.rendered_prompt.map(<[u8]>::to_vec),
            runnable_manifest: event.runnable_manifest.map(<[u8]>::to_vec),
            parsed_content: event.parsed_content.map(<[u8]>::to_vec),
        });
        Ok(())
    }
}

struct TraceSnapshot {
    kind: SensitiveTraceKind,
    task_id: Option<i64>,
    slot_id: Option<i32>,
    request_json: Option<Vec<u8>>,
    rendered_prompt: Option<Vec<u8>>,
    runnable_manifest: Option<Vec<u8>>,
    parsed_content: Option<Vec<u8>>,
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn manifest_with(base: &[u8], path: &[&str], replacement: Value) -> Vec<u8> {
    let mut manifest: Value = serde_json::from_slice(base).unwrap();
    let (field, parents) = path.split_last().unwrap();
    let mut parent = &mut manifest;
    for segment in parents {
        parent = &mut parent[*segment];
    }
    parent[*field] = replacement;
    canonical_json(&manifest)
}
