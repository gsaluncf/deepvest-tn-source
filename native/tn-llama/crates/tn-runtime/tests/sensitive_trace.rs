use std::fs;
use std::io;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde_json::Value;
use tn_runtime::{
    BuildMetadata, Phase, SensitiveTrace, SensitiveTraceEvent, SensitiveTraceKind,
    SensitiveTraceSink, SENSITIVE_TRACE_SCHEMA,
};

fn metadata() -> BuildMetadata {
    BuildMetadata {
        build_id: "build-17".to_string(),
        llama_source_revision: "llama-rev".to_string(),
        model_sha256: "sha256:model".to_string(),
        binary_sha256: "sha256:binary".to_string(),
    }
}

fn empty_event(kind: SensitiveTraceKind) -> SensitiveTraceEvent<'static> {
    SensitiveTraceEvent {
        run_id: "run-17",
        phase: None,
        kind,
        task_id: None,
        slot_id: None,
        request_json: None,
        rendered_prompt: None,
        runnable_manifest: None,
        raw_model_output: None,
        parsed_content: None,
        error_code: None,
    }
}

#[test]
fn exact_bytes_round_trip_with_fixed_metadata_and_monotonic_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let trace = SensitiveTrace::create(dir.path(), "run-17", metadata()).unwrap();
    let request = b"{\"nul\":\"\0\"}\n";
    let prompt = "governance:\nπ".as_bytes();
    let manifest = b"{\"tokens\":[0,10,255]}";
    let raw_output = [0xff, 0xfe, b'\n', 0x00];
    let parsed = "résultat\n".as_bytes();

    let mut prepared = empty_event(SensitiveTraceKind::TurnPrepared);
    prepared.phase = Some(Phase::GovernanceRead);
    prepared.task_id = Some(71);
    prepared.request_json = Some(request);
    prepared.rendered_prompt = Some(prompt);
    prepared.runnable_manifest = Some(manifest);
    trace.append(prepared).unwrap();

    let mut started = empty_event(SensitiveTraceKind::TurnStarted);
    started.phase = Some(Phase::CommitmentEcho);
    started.task_id = Some(72);
    started.slot_id = Some(3);
    trace.append(started).unwrap();

    let mut result = empty_event(SensitiveTraceKind::ModelResult);
    result.phase = Some(Phase::GovernedAction);
    result.task_id = Some(73);
    result.slot_id = Some(3);
    result.raw_model_output = Some(&raw_output);
    result.parsed_content = Some(parsed);
    trace.append(result).unwrap();

    let mut error = empty_event(SensitiveTraceKind::Error);
    error.error_code = Some("model_failed");
    trace.append(error).unwrap();

    let rows = fs::read_to_string(trace.path())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 4);
    for (sequence, row) in rows.iter().enumerate() {
        assert_eq!(row["schema"], SENSITIVE_TRACE_SCHEMA);
        assert_eq!(row["run_id"], "run-17");
        assert_eq!(row["sequence"], sequence as u64);
        assert_eq!(row["build_id"], "build-17");
        assert_eq!(row["llama_source_revision"], "llama-rev");
        assert_eq!(row["model_sha256"], "sha256:model");
        assert_eq!(row["binary_sha256"], "sha256:binary");
        assert_eq!(row["process_id"], u64::from(std::process::id()));
    }
    assert_eq!(rows[0]["phase"], 1);
    assert_eq!(rows[1]["phase"], 2);
    assert_eq!(rows[2]["phase"], 4);
    assert_eq!(rows[3]["phase"], Value::Null);
    assert_eq!(rows[0]["kind"], "turn_prepared");
    assert_eq!(rows[1]["kind"], "turn_started");
    assert_eq!(rows[2]["kind"], "model_result");
    assert_eq!(rows[3]["kind"], "error");

    assert_eq!(
        STANDARD
            .decode(rows[0]["request_json_b64"].as_str().unwrap())
            .unwrap(),
        request
    );
    assert_eq!(
        STANDARD
            .decode(rows[0]["rendered_prompt_b64"].as_str().unwrap())
            .unwrap(),
        prompt
    );
    assert_eq!(
        STANDARD
            .decode(rows[0]["runnable_manifest_b64"].as_str().unwrap())
            .unwrap(),
        manifest
    );
    assert_eq!(
        STANDARD
            .decode(rows[2]["raw_model_output_b64"].as_str().unwrap())
            .unwrap(),
        raw_output
    );
    assert_eq!(
        STANDARD
            .decode(rows[2]["parsed_content_b64"].as_str().unwrap())
            .unwrap(),
        parsed
    );
    assert_eq!(rows[3]["error_code"], "model_failed");
}

#[test]
fn trace_creation_is_exclusive_and_creation_errors_reach_the_caller() {
    let dir = tempfile::tempdir().unwrap();
    SensitiveTrace::create(dir.path(), "run-17", metadata()).unwrap();

    assert!(SensitiveTrace::create(dir.path(), "run-17", metadata()).is_err());
    assert!(SensitiveTrace::create(dir.path().join("missing"), "run-18", metadata()).is_err());
}

struct FailingSensitiveTrace;

impl SensitiveTraceSink for FailingSensitiveTrace {
    fn append(&self, _event: SensitiveTraceEvent<'_>) -> io::Result<()> {
        Err(io::Error::other("sensitive trace append failed"))
    }
}

#[test]
fn sensitive_trace_sink_is_object_safe_and_preserves_append_errors() {
    let sink: &dyn SensitiveTraceSink = &FailingSensitiveTrace;
    let error = sink
        .append(empty_event(SensitiveTraceKind::Error))
        .unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.to_string(), "sensitive trace append failed");
}
