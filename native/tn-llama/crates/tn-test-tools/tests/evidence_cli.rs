use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{json, Map, Value};
use tn_core::chain::{compute_row_hash, GroupInput, RowHashInput};
use tn_runtime::{
    AuditEvent, AuditLog, BuildMetadata, DecisionAudit, Phase, SensitiveTrace, SensitiveTraceEvent,
    SensitiveTraceKind, SensitiveTraceSink,
};

const VERDICT_SCHEMA: &str = "tn.llama.evidence-verdict.v1";

struct EvidenceFixture {
    manifest_path: PathBuf,
    audit_path: PathBuf,
    sensitive_path: PathBuf,
}

fn create_evidence_fixture(root: &Path) -> EvidenceFixture {
    let audit_path = root.join("audit.ndjson");
    let audit = AuditLog::create(&audit_path, "run-17").unwrap();
    for (before, after) in [
        ("received", "row_verified"),
        ("row_verified", "chain_reserved"),
    ] {
        let mut event = AuditEvent::transition(before, after, "sha256:input", "allow");
        event.request_digest = "sha256:request".to_string();
        audit.append(event).unwrap();
    }

    let trace = SensitiveTrace::create(
        root,
        "run-17",
        BuildMetadata {
            build_id: "build-17".to_string(),
            llama_source_revision: "llama-rev".to_string(),
            model_sha256: "sha256:model".to_string(),
            binary_sha256: "sha256:binary".to_string(),
        },
    )
    .unwrap();
    trace
        .append(SensitiveTraceEvent {
            run_id: "run-17",
            phase: Some(Phase::GovernanceRead),
            kind: SensitiveTraceKind::TurnPrepared,
            task_id: Some(71),
            slot_id: None,
            request_json: Some(b"{\"messages\":[]}"),
            rendered_prompt: Some("governance π".as_bytes()),
            runnable_manifest: Some(b"{\"tokens\":[1,2]}"),
            raw_model_output: None,
            parsed_content: None,
            error_code: None,
        })
        .unwrap();
    trace
        .append(SensitiveTraceEvent {
            run_id: "run-17",
            phase: Some(Phase::GovernanceRead),
            kind: SensitiveTraceKind::ModelResult,
            task_id: Some(71),
            slot_id: Some(3),
            request_json: None,
            rendered_prompt: None,
            runnable_manifest: None,
            raw_model_output: Some(&[0xff, b'\n', 0x00]),
            parsed_content: Some("ok π".as_bytes()),
            error_code: None,
        })
        .unwrap();
    let sensitive_path = trace.path().to_path_buf();

    let manifest_path = root.join("evidence-manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec(&json!({
            "schema": "tn.llama.evidence-manifest.v1",
            "audit_path": audit_path,
            "sensitive_trace_path": sensitive_path,
            "expected_build_id": "build-17",
            "expected_llama_source_revision": "llama-rev",
            "expected_model_sha256": "sha256:model",
            "expected_binary_sha256": "sha256:binary",
            "fixture_manifest_path": null,
            "http_results_path": null,
            "server_log_path": null
        }))
        .unwrap(),
    )
    .unwrap();

    EvidenceFixture {
        manifest_path,
        audit_path,
        sensitive_path,
    }
}

fn run_verifier(manifest_path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tn-evidence-verify"))
        .arg("verify")
        .arg("--evidence")
        .arg(manifest_path)
        .output()
        .unwrap()
}

fn verdict(output: &Output) -> Value {
    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    assert_eq!(stdout.lines().count(), 1, "stdout was {stdout:?}");
    serde_json::from_str(stdout.trim()).unwrap()
}

fn rewrite_line(path: &Path, index: usize, mutate: impl FnOnce(&mut Map<String, Value>)) {
    let source = fs::read_to_string(path).unwrap();
    let mut rows = source
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    mutate(rows[index].as_object_mut().unwrap());
    let rewritten = rows
        .iter()
        .map(|row| serde_json::to_string(row).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(path, rewritten).unwrap();
}

fn officially_rehash(row: &mut Map<String, Value>) {
    let blocks = tn_core::sealed_object::extract_group_blocks(row).unwrap();
    let groups = blocks
        .iter()
        .map(|(name, block)| {
            (
                name.clone(),
                GroupInput {
                    ciphertext: block.ciphertext.clone(),
                    field_hashes: block.field_hashes.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let public = row
        .iter()
        .filter(|(name, _)| {
            !tn_core::sealed_object::ENVELOPE_RESERVED.contains(&name.as_str())
                && !blocks.contains_key(*name)
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let string = |name: &str| row[name].as_str().unwrap();
    let row_hash = compute_row_hash(&RowHashInput {
        device_identity: string("device_identity"),
        timestamp: string("timestamp"),
        event_id: string("event_id"),
        event_type: string("event_type"),
        level: string("level"),
        prev_hash: string("prev_hash"),
        public_fields: &public,
        groups: &groups,
    });
    row.insert("row_hash".to_string(), Value::String(row_hash));
}

fn assert_rejected(output: &Output, expected_code: &str) {
    assert!(!output.status.success());
    assert_eq!(
        verdict(output),
        json!({
            "schema": VERDICT_SCHEMA,
            "valid": false,
            "audit_rows": 0,
            "sensitive_records": 0,
            "error_code": expected_code
        })
    );
}

#[test]
fn accepts_writer_produced_audit_and_sensitive_trace() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = create_evidence_fixture(dir.path());

    let output = run_verifier(&fixture.manifest_path);

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        verdict(&output),
        json!({
            "schema": VERDICT_SCHEMA,
            "valid": true,
            "audit_rows": 2,
            "sensitive_records": 2,
            "error_code": null
        })
    );
}

#[test]
fn rejects_changed_audit_public_field() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = create_evidence_fixture(dir.path());
    rewrite_line(&fixture.audit_path, 0, |row| {
        row.insert("outcome".to_string(), Value::String("deny".to_string()));
    });

    assert_rejected(&run_verifier(&fixture.manifest_path), "evidence_audit_hash");
}

#[test]
fn rejects_officially_rehashed_broken_audit_link() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = create_evidence_fixture(dir.path());
    rewrite_line(&fixture.audit_path, 1, |row| {
        row.insert(
            "prev_hash".to_string(),
            Value::String(tn_core::chain::ZERO_HASH.to_string()),
        );
        officially_rehash(row);
    });

    assert_rejected(&run_verifier(&fixture.manifest_path), "evidence_audit_link");
}

#[test]
fn rejects_invalid_sensitive_base64() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = create_evidence_fixture(dir.path());
    rewrite_line(&fixture.sensitive_path, 0, |row| {
        row.insert(
            "request_json_b64".to_string(),
            Value::String("not base64!".to_string()),
        );
    });

    assert_rejected(
        &run_verifier(&fixture.manifest_path),
        "evidence_sensitive_base64",
    );
}

#[test]
fn rejects_sensitive_schema_or_metadata_mismatch() {
    for (field, value, code) in [
        (
            "schema",
            "tn.llama.sensitive-trace.v2",
            "evidence_sensitive_schema",
        ),
        ("build_id", "different-build", "evidence_sensitive_metadata"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let fixture = create_evidence_fixture(dir.path());
        rewrite_line(&fixture.sensitive_path, 0, |row| {
            row.insert(field.to_string(), Value::String(value.to_string()));
        });

        assert_rejected(&run_verifier(&fixture.manifest_path), code);
    }
}

#[test]
fn rejects_unknown_or_forbidden_audit_facts_after_official_rehash() {
    for facts in [
        json!({"unknown": true}),
        json!({"governance_plaintext": "do not publish"}),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let fixture = create_evidence_fixture(dir.path());
        rewrite_line(&fixture.audit_path, 1, |row| {
            row.insert(
                "facts_json".to_string(),
                Value::String(serde_json::to_string(&facts).unwrap()),
            );
            officially_rehash(row);
        });

        assert_rejected(
            &run_verifier(&fixture.manifest_path),
            "evidence_audit_facts",
        );
    }
}

#[test]
fn rejects_changed_later_sensitive_process_id() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = create_evidence_fixture(dir.path());
    rewrite_line(&fixture.sensitive_path, 1, |row| {
        let changed = row["process_id"].as_u64().unwrap() + 1;
        row.insert("process_id".to_string(), Value::from(changed));
    });

    assert_rejected(
        &run_verifier(&fixture.manifest_path),
        "evidence_sensitive_metadata",
    );
}

#[test]
fn rejects_malformed_or_non_utc_sensitive_timestamp() {
    for timestamp in ["not-rfc3339", "2026-08-23T12:00:00+01:00"] {
        let dir = tempfile::tempdir().unwrap();
        let fixture = create_evidence_fixture(dir.path());
        rewrite_line(&fixture.sensitive_path, 0, |row| {
            row.insert(
                "timestamp".to_string(),
                Value::String(timestamp.to_string()),
            );
        });

        assert_rejected(
            &run_verifier(&fixture.manifest_path),
            "evidence_sensitive_timestamp",
        );
    }
}
