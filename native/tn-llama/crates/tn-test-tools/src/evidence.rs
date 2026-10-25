use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;
use time::format_description::well_known::Rfc3339;
use time::{OffsetDateTime, UtcOffset};

pub const EVIDENCE_MANIFEST_SCHEMA: &str = "tn.llama.evidence-manifest.v1";
pub const EVIDENCE_VERDICT_SCHEMA: &str = "tn.llama.evidence-verdict.v1";

const AUDIT_SCHEMA: &str = "tn.llama.audit.v1";
const AUDIT_EVENT_TYPE: &str = "tn.llama.audit";
const SENSITIVE_TRACE_SCHEMA: &str = "tn.llama.sensitive-trace.v1";

const AUDIT_FIELDS: [&str; 21] = [
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
];

const SENSITIVE_FIELDS: [&str; 19] = [
    "binary_sha256",
    "build_id",
    "error_code",
    "kind",
    "llama_source_revision",
    "model_sha256",
    "parsed_content_b64",
    "phase",
    "process_id",
    "raw_model_output_b64",
    "rendered_prompt_b64",
    "request_json_b64",
    "run_id",
    "runnable_manifest_b64",
    "schema",
    "sequence",
    "slot_id",
    "task_id",
    "timestamp",
];

const BYTE_FIELDS: [&str; 5] = [
    "request_json_b64",
    "rendered_prompt_b64",
    "runnable_manifest_b64",
    "raw_model_output_b64",
    "parsed_content_b64",
];

const AUDIT_FACT_FIELDS: [&str; 12] = [
    "content_digest",
    "governance_commitment_count",
    "governance_value_count",
    "hashes_matched",
    "hashes_verified",
    "incomplete",
    "model_request_digest",
    "phase",
    "rendered_prompt_digest",
    "runnable_task_digest",
    "signature_verified",
    "turn_ordinal",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditFacts {
    phase: Option<u32>,
    turn_ordinal: Option<u8>,
    model_request_digest: Option<String>,
    rendered_prompt_digest: Option<String>,
    runnable_task_digest: Option<String>,
    content_digest: Option<String>,
    governance_value_count: Option<u32>,
    governance_commitment_count: Option<u32>,
    hashes_verified: Option<bool>,
    hashes_matched: Option<bool>,
    incomplete: Option<bool>,
    signature_verified: Option<bool>,
}

impl AuditFacts {
    fn valid(self) -> bool {
        let Self {
            phase,
            turn_ordinal,
            model_request_digest,
            rendered_prompt_digest,
            runnable_task_digest,
            content_digest,
            governance_value_count,
            governance_commitment_count,
            hashes_verified,
            hashes_matched,
            incomplete,
            signature_verified,
        } = self;
        let _typed_scalars = (
            turn_ordinal,
            governance_value_count,
            governance_commitment_count,
            hashes_verified,
            hashes_matched,
            incomplete,
            signature_verified,
        );
        phase.is_none_or(|value| matches!(value, 1..=3))
            && [
                model_request_digest,
                rendered_prompt_digest,
                runnable_task_digest,
                content_digest,
            ]
            .into_iter()
            .flatten()
            .all(|digest| valid_sha256_digest(&digest))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceManifest {
    pub schema: String,
    pub audit_path: PathBuf,
    pub sensitive_trace_path: PathBuf,
    pub expected_build_id: String,
    pub expected_llama_source_revision: String,
    pub expected_model_sha256: String,
    pub expected_binary_sha256: String,
    pub fixture_manifest_path: Option<PathBuf>,
    pub http_results_path: Option<PathBuf>,
    pub server_log_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvidenceCounts {
    pub audit_rows: usize,
    pub sensitive_records: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EvidenceVerdict {
    pub schema: &'static str,
    pub valid: bool,
    pub audit_rows: usize,
    pub sensitive_records: usize,
    pub error_code: Option<&'static str>,
}

impl EvidenceVerdict {
    #[must_use]
    pub const fn accepted(counts: EvidenceCounts) -> Self {
        Self {
            schema: EVIDENCE_VERDICT_SCHEMA,
            valid: true,
            audit_rows: counts.audit_rows,
            sensitive_records: counts.sensitive_records,
            error_code: None,
        }
    }

    #[must_use]
    pub const fn rejected(error: &EvidenceError) -> Self {
        Self {
            schema: EVIDENCE_VERDICT_SCHEMA,
            valid: false,
            audit_rows: 0,
            sensitive_records: 0,
            error_code: Some(error.code),
        }
    }
}

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct EvidenceError {
    code: &'static str,
    message: String,
}

impl EvidenceError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }
}

pub fn verify_evidence(manifest_path: &Path) -> Result<EvidenceCounts, EvidenceError> {
    if !manifest_path.is_absolute() {
        return Err(error(
            "evidence_manifest",
            "evidence manifest path must be absolute",
        ));
    }
    let bytes =
        fs::read(manifest_path).map_err(|source| error("evidence_manifest", source.to_string()))?;
    let manifest: EvidenceManifest = serde_json::from_slice(&bytes)
        .map_err(|source| error("evidence_manifest", source.to_string()))?;
    if manifest.schema != EVIDENCE_MANIFEST_SCHEMA {
        return Err(error(
            "evidence_manifest",
            "evidence manifest schema does not match",
        ));
    }

    let (audit_rows, run_id) = verify_audit(&manifest.audit_path)?;
    let sensitive_records = verify_sensitive(&manifest, &run_id)?;
    Ok(EvidenceCounts {
        audit_rows,
        sensitive_records,
    })
}

fn verify_audit(path: &Path) -> Result<(usize, String), EvidenceError> {
    let rows = read_json_lines(path, "evidence_audit_shape")?;
    if rows.is_empty() {
        return Err(error("evidence_audit_shape", "audit is empty"));
    }
    let mut previous_hash = tn_core::chain::ZERO_HASH.to_string();
    let mut run_id = None;

    for (index, row) in rows.iter().enumerate() {
        require_exact_keys(row, &AUDIT_FIELDS, "evidence_audit_shape")?;
        require_string(row, "device_identity", true, "evidence_audit_shape")?;
        require_string(row, "timestamp", true, "evidence_audit_shape")?;
        require_string(row, "event_id", true, "evidence_audit_shape")?;
        if require_string(row, "event_type", true, "evidence_audit_shape")? != AUDIT_EVENT_TYPE
            || require_string(row, "audit_schema", true, "evidence_audit_shape")? != AUDIT_SCHEMA
            || !require_string(row, "signature", false, "evidence_audit_shape")?.is_empty()
        {
            return Err(error(
                "evidence_audit_shape",
                "audit profile does not match",
            ));
        }
        for field in [
            "level",
            "kind",
            "state_before",
            "state_after",
            "target_row_hash",
            "request_digest",
            "outcome",
            "prev_hash",
            "row_hash",
            "run_id",
        ] {
            require_string(row, field, false, "evidence_audit_shape")?;
        }
        if row.get("sequence").and_then(Value::as_u64) != Some(index as u64)
            || !null_or_i64(row.get("llama_task_id"))
            || !null_or_i64(row.get("llama_slot_id"))
            || !null_or_string(row.get("error_code"))
        {
            return Err(error(
                "evidence_audit_shape",
                "audit field type or sequence does not match",
            ));
        }

        let facts_json = require_string(row, "facts_json", false, "evidence_audit_shape")?;
        let facts: Value = serde_json::from_str(facts_json)
            .map_err(|_| error("evidence_audit_shape", "facts_json is not JSON"))?;
        let facts_object = facts
            .as_object()
            .ok_or_else(|| error("evidence_audit_facts", "audit facts are not an object"))?;
        require_exact_keys(facts_object, &AUDIT_FACT_FIELDS, "evidence_audit_facts")?;
        let typed_facts: AuditFacts = serde_json::from_value(facts.clone())
            .map_err(|source| error("evidence_audit_facts", source.to_string()))?;
        if !typed_facts.valid() {
            return Err(error(
                "evidence_audit_facts",
                "audit facts do not match the closed value contract",
            ));
        }
        let canonical = tn_core::canonical::canonical_bytes(&facts)
            .map_err(|source| error("evidence_audit_shape", source.to_string()))?;
        if canonical.as_slice() != facts_json.as_bytes() {
            return Err(error("evidence_audit_shape", "facts_json is not canonical"));
        }

        let blocks = tn_core::sealed_object::extract_group_blocks(row)
            .map_err(|source| error("evidence_audit_shape", source.to_string()))?;
        if !blocks.is_empty() {
            return Err(error(
                "evidence_audit_shape",
                "audit must not contain encrypted groups",
            ));
        }
        if !tn_core::sealed_object::verify_sealed(row, &blocks).row_hash {
            return Err(error(
                "evidence_audit_hash",
                "audit row hash does not verify",
            ));
        }
        if require_string(row, "prev_hash", false, "evidence_audit_shape")? != previous_hash {
            return Err(error(
                "evidence_audit_link",
                "audit predecessor does not match",
            ));
        }
        previous_hash = require_string(row, "row_hash", true, "evidence_audit_shape")?.to_string();

        let row_run_id = require_string(row, "run_id", true, "evidence_audit_shape")?;
        match &run_id {
            Some(expected) if expected != row_run_id => {
                return Err(error("evidence_run_id", "audit run IDs do not match"));
            }
            None => run_id = Some(row_run_id.to_string()),
            _ => {}
        }
    }

    Ok((rows.len(), run_id.expect("nonempty audit checked")))
}

fn verify_sensitive(manifest: &EvidenceManifest, run_id: &str) -> Result<usize, EvidenceError> {
    let rows = read_json_lines(&manifest.sensitive_trace_path, "evidence_sensitive_schema")?;
    if rows.is_empty() {
        return Err(error(
            "evidence_sensitive_schema",
            "sensitive trace is empty",
        ));
    }

    let mut process_id = None;
    for (index, row) in rows.iter().enumerate() {
        require_exact_keys(row, &SENSITIVE_FIELDS, "evidence_sensitive_schema")?;
        if require_string(row, "schema", true, "evidence_sensitive_schema")?
            != SENSITIVE_TRACE_SCHEMA
        {
            return Err(error(
                "evidence_sensitive_schema",
                "sensitive trace schema does not match",
            ));
        }
        if row.get("sequence").and_then(Value::as_u64) != Some(index as u64)
            || !null_or_i64(row.get("task_id"))
            || !null_or_i64(row.get("slot_id"))
            || !null_or_string(row.get("error_code"))
        {
            return Err(error(
                "evidence_sensitive_schema",
                "sensitive trace field type or sequence does not match",
            ));
        }
        let row_process_id = row
            .get("process_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                error(
                    "evidence_sensitive_schema",
                    "sensitive process ID is not an unsigned integer",
                )
            })?;
        match process_id {
            Some(expected) if expected != row_process_id => {
                return Err(error(
                    "evidence_sensitive_metadata",
                    "sensitive process IDs do not match",
                ));
            }
            None => process_id = Some(row_process_id),
            _ => {}
        }
        let timestamp = require_string(row, "timestamp", true, "evidence_sensitive_schema")?;
        let parsed_timestamp = OffsetDateTime::parse(timestamp, &Rfc3339).map_err(|_| {
            error(
                "evidence_sensitive_timestamp",
                "sensitive timestamp is not RFC3339",
            )
        })?;
        if parsed_timestamp.offset() != UtcOffset::UTC {
            return Err(error(
                "evidence_sensitive_timestamp",
                "sensitive timestamp is not UTC",
            ));
        }
        let phase_valid = row
            .get("phase")
            .is_some_and(|phase| phase.is_null() || matches!(phase.as_u64(), Some(1..=3)));
        let kind_valid = row.get("kind").and_then(Value::as_str).is_some_and(|kind| {
            matches!(
                kind,
                "turn_prepared" | "turn_started" | "model_result" | "error"
            )
        });
        if !phase_valid || !kind_valid {
            return Err(error(
                "evidence_sensitive_schema",
                "sensitive phase or kind does not match",
            ));
        }
        if require_string(row, "run_id", true, "evidence_sensitive_schema")? != run_id {
            return Err(error(
                "evidence_run_id",
                "audit and sensitive trace run IDs do not match",
            ));
        }
        if require_string(row, "build_id", false, "evidence_sensitive_schema")?
            != manifest.expected_build_id
            || require_string(
                row,
                "llama_source_revision",
                false,
                "evidence_sensitive_schema",
            )? != manifest.expected_llama_source_revision
            || require_string(row, "model_sha256", false, "evidence_sensitive_schema")?
                != manifest.expected_model_sha256
            || require_string(row, "binary_sha256", false, "evidence_sensitive_schema")?
                != manifest.expected_binary_sha256
        {
            return Err(error(
                "evidence_sensitive_metadata",
                "sensitive trace metadata does not match the manifest",
            ));
        }

        for field in BYTE_FIELDS {
            match row.get(field) {
                Some(Value::Null) => {}
                Some(Value::String(encoded)) if STANDARD.decode(encoded).is_ok() => {}
                _ => {
                    return Err(error(
                        "evidence_sensitive_base64",
                        "sensitive byte field is not standard base64",
                    ));
                }
            }
        }
    }
    Ok(rows.len())
}

fn read_json_lines(
    path: &Path,
    code: &'static str,
) -> Result<Vec<Map<String, Value>>, EvidenceError> {
    let source = fs::read_to_string(path).map_err(|source| error(code, source.to_string()))?;
    source
        .lines()
        .map(|line| {
            let value: Value =
                serde_json::from_str(line).map_err(|source| error(code, source.to_string()))?;
            value
                .as_object()
                .cloned()
                .ok_or_else(|| error(code, "evidence line is not an object"))
        })
        .collect()
}

fn require_exact_keys<const N: usize>(
    row: &Map<String, Value>,
    fields: &[&str; N],
    code: &'static str,
) -> Result<(), EvidenceError> {
    let expected = fields.iter().copied().collect::<BTreeSet<_>>();
    let actual = row.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual == expected {
        Ok(())
    } else {
        Err(error(
            code,
            "evidence fields do not match the closed schema",
        ))
    }
}

fn require_string<'a>(
    row: &'a Map<String, Value>,
    field: &str,
    nonempty: bool,
    code: &'static str,
) -> Result<&'a str, EvidenceError> {
    row.get(field)
        .and_then(Value::as_str)
        .filter(|value| !nonempty || !value.is_empty())
        .ok_or_else(|| error(code, format!("{field} is not a valid string")))
}

fn null_or_i64(value: Option<&Value>) -> bool {
    value.is_some_and(|value| value.is_null() || value.as_i64().is_some())
}

fn null_or_string(value: Option<&Value>) -> bool {
    value.is_some_and(|value| value.is_null() || value.is_string())
}

fn valid_sha256_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn error(code: &'static str, message: impl Into<String>) -> EvidenceError {
    EvidenceError::new(code, message)
}
