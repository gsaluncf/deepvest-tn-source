use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::Path;

use serde_json::{json, Map, Value};
use tn_runtime::{
    AuditDigest, AuditEvent, AuditFacts, AuditLog, AuditRecord, DecisionAudit, Phase,
    AUDIT_EVENT_TYPE, AUDIT_SCHEMA,
};

const MODEL_REQUEST_DIGEST: &str =
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";

fn read_json_lines(path: impl AsRef<Path>) -> Vec<Map<String, Value>> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .map(|value| value.as_object().unwrap().clone())
        .collect()
}

fn official_row_hash_valid(row: &Map<String, Value>) -> bool {
    let blocks = tn_core::sealed_object::extract_group_blocks(row).unwrap();
    tn_core::sealed_object::verify_sealed(row, &blocks).row_hash
}

fn transition(before: &str, after: &str) -> AuditEvent {
    let mut event = AuditEvent::transition(before, after, "sha256:input-row", "allow");
    event.request_digest = "sha256:request".to_string();
    event.facts = AuditFacts {
        phase: Some(Phase::GovernanceRead),
        turn_ordinal: Some(1),
        model_request_digest: Some(AuditDigest::new(MODEL_REQUEST_DIGEST).unwrap()),
        governance_value_count: Some(6),
        governance_commitment_count: Some(6),
        hashes_verified: Some(true),
        ..AuditFacts::default()
    };
    event
}

#[test]
fn writes_officially_verifiable_linked_audit_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.ndjson");
    let log = AuditLog::create(&path, "run-17").unwrap();

    let first = log.append(transition("received", "row_verified")).unwrap();
    let second = log
        .append(transition("row_verified", "chain_reserved"))
        .unwrap();

    assert_eq!(first.prev_hash, tn_core::chain::ZERO_HASH);
    assert_eq!(second.prev_hash, first.row_hash);
    let rows = read_json_lines(path);
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert!(official_row_hash_valid(&row));
        assert_eq!(row["event_type"], AUDIT_EVENT_TYPE);
        assert_eq!(row["audit_schema"], AUDIT_SCHEMA);
        assert_eq!(row["signature"], "");
        assert!(tn_core::sealed_object::extract_group_blocks(&row)
            .unwrap()
            .is_empty());
    }
}

#[test]
fn changed_public_audit_field_fails_official_verification() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.ndjson");
    let log = AuditLog::create(&path, "run-17").unwrap();
    log.append(transition("received", "row_verified")).unwrap();

    let mut row = read_json_lines(path).remove(0);
    row.insert("outcome".to_string(), Value::String("deny".to_string()));

    assert!(!official_row_hash_valid(&row));
}

#[test]
fn audit_has_only_the_fixed_public_schema_and_canonical_facts_string() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.ndjson");
    let log = AuditLog::create(&path, "run-17").unwrap();
    log.append(transition("received", "row_verified")).unwrap();

    let row = read_json_lines(path).remove(0);
    let actual = row.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = [
        "audit_schema",
        "device_identity",
        "error_code",
        "event_id",
        "event_type",
        "facts_json",
        "kind",
        "level",
        "llama_slot_id",
        "llama_task_id",
        "outcome",
        "prev_hash",
        "request_digest",
        "row_hash",
        "run_id",
        "sequence",
        "signature",
        "state_after",
        "state_before",
        "target_row_hash",
        "timestamp",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    let facts_json = row["facts_json"].as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(facts_json).unwrap(),
        json!({
            "content_digest": null,
            "governance_commitment_count": 6,
            "governance_value_count": 6,
            "hashes_matched": null,
            "hashes_verified": true,
            "incomplete": null,
            "model_request_digest": MODEL_REQUEST_DIGEST,
            "phase": 1,
            "rendered_prompt_digest": null,
            "runnable_task_digest": null,
            "signature_verified": null,
            "turn_ordinal": 1
        })
    );
    assert_eq!(
        tn_core::canonical::canonical_bytes(&serde_json::from_str::<Value>(facts_json).unwrap())
            .unwrap(),
        facts_json.as_bytes()
    );
    for forbidden in [
        "governance",
        "governance_plaintext",
        "key",
        "request_json_b64",
        "rendered_prompt_b64",
        "raw_user_content",
        "raw_model_output_b64",
        "model_reasoning",
    ] {
        assert!(!row.contains_key(forbidden));
    }
}

#[test]
fn digest_facts_accept_only_sha256_token_shape() {
    assert_eq!(
        AuditDigest::new(MODEL_REQUEST_DIGEST).unwrap().as_str(),
        MODEL_REQUEST_DIGEST
    );
    for forbidden_or_malformed in [
        "opened governance plaintext",
        "receipt:opaque-capability",
        r"C:\\sensitive\\trace.ndjson",
        "sha256:ABCDEF",
        "sha256:1111",
    ] {
        assert!(AuditDigest::new(forbidden_or_malformed).is_err());
    }
}

#[test]
fn ordinary_audit_creation_failure_reaches_the_caller() {
    let dir = tempfile::tempdir().unwrap();
    let missing_parent = dir.path().join("missing").join("audit.ndjson");

    assert!(AuditLog::create(missing_parent, "run-17").is_err());
}

struct FailingAudit;

impl DecisionAudit for FailingAudit {
    fn append(&self, _event: AuditEvent) -> io::Result<AuditRecord> {
        Err(io::Error::other("audit append failed"))
    }
}

#[test]
fn audit_sink_is_object_safe_and_preserves_append_errors() {
    let sink: &dyn DecisionAudit = &FailingAudit;
    let error = sink
        .append(transition("received", "row_verified"))
        .unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.to_string(), "audit append failed");
}
