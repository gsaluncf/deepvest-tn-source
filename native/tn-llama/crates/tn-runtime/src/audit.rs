use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use serde::Serialize;
use serde_json::{Map, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tn_core::chain::{compute_row_hash, RowHashInput, ZERO_HASH};
use uuid::Uuid;

pub const AUDIT_SCHEMA: &str = "tn.llama.audit.v1";
pub const AUDIT_EVENT_TYPE: &str = "tn.llama.audit";

const AUDIT_DEVICE_IDENTITY: &str = "tn.llama.integrity-only-audit";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AuditDigest(String);

impl AuditDigest {
    pub fn new(value: impl Into<String>) -> Result<Self, AuditDigestError> {
        let value = value.into();
        let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        });
        if valid {
            Ok(Self(value))
        } else {
            Err(AuditDigestError)
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("audit digest must be sha256 followed by 64 lowercase hexadecimal characters")]
pub struct AuditDigestError;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct AuditFacts {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_identity: Option<crate::ExecutionIdentity>,
    pub phase: Option<crate::model_phase::Phase>,
    pub turn_ordinal: Option<u8>,
    pub model_request_digest: Option<AuditDigest>,
    pub rendered_prompt_digest: Option<AuditDigest>,
    pub runnable_task_digest: Option<AuditDigest>,
    pub content_digest: Option<AuditDigest>,
    pub governance_value_count: Option<u32>,
    pub governance_commitment_count: Option<u32>,
    pub hashes_verified: Option<bool>,
    pub hashes_matched: Option<bool>,
    pub incomplete: Option<bool>,
    pub signature_verified: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuditEvent {
    pub kind: String,
    pub state_before: String,
    pub state_after: String,
    pub target_row_hash: String,
    pub request_digest: String,
    pub llama_task_id: Option<i64>,
    pub llama_slot_id: Option<i32>,
    pub outcome: String,
    pub error_code: Option<String>,
    pub facts: AuditFacts,
}

impl AuditEvent {
    #[must_use]
    pub fn transition(
        state_before: impl Into<String>,
        state_after: impl Into<String>,
        target_row_hash: impl Into<String>,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            kind: "transition".to_string(),
            state_before: state_before.into(),
            state_after: state_after.into(),
            target_row_hash: target_row_hash.into(),
            request_digest: String::new(),
            llama_task_id: None,
            llama_slot_id: None,
            outcome: outcome.into(),
            error_code: None,
            facts: AuditFacts::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditRecord {
    pub prev_hash: String,
    pub row_hash: String,
}

pub trait DecisionAudit: Send + Sync {
    fn append(&self, event: AuditEvent) -> io::Result<AuditRecord>;
}

pub struct AuditLog {
    run_id: String,
    state: Mutex<AuditState>,
}

struct AuditState {
    writer: BufWriter<File>,
    sequence: u64,
    prev_hash: String,
}

impl AuditLog {
    pub fn create(path: impl AsRef<Path>, run_id: impl Into<String>) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            run_id: run_id.into(),
            state: Mutex::new(AuditState {
                writer: BufWriter::new(file),
                sequence: 0,
                prev_hash: ZERO_HASH.to_string(),
            }),
        })
    }
}

impl DecisionAudit for AuditLog {
    fn append(&self, event: AuditEvent) -> io::Result<AuditRecord> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("audit writer state is unavailable"))?;
        let timestamp = OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .map_err(io::Error::other)?;
        let event_id = Uuid::now_v7().to_string();
        let facts = serde_json::to_value(&event.facts)?;
        let facts_json = String::from_utf8(
            tn_core::canonical::canonical_bytes(&facts)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        )
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

        let mut public_fields = BTreeMap::new();
        public_fields.insert(
            "audit_schema".to_string(),
            Value::String(AUDIT_SCHEMA.to_string()),
        );
        public_fields.insert("run_id".to_string(), Value::String(self.run_id.clone()));
        public_fields.insert("kind".to_string(), Value::String(event.kind));
        public_fields.insert(
            "state_before".to_string(),
            Value::String(event.state_before),
        );
        public_fields.insert("state_after".to_string(), Value::String(event.state_after));
        public_fields.insert(
            "target_row_hash".to_string(),
            Value::String(event.target_row_hash),
        );
        public_fields.insert(
            "request_digest".to_string(),
            Value::String(event.request_digest),
        );
        public_fields.insert(
            "llama_task_id".to_string(),
            event.llama_task_id.map_or(Value::Null, Value::from),
        );
        public_fields.insert(
            "llama_slot_id".to_string(),
            event.llama_slot_id.map_or(Value::Null, Value::from),
        );
        public_fields.insert("outcome".to_string(), Value::String(event.outcome));
        public_fields.insert(
            "error_code".to_string(),
            event.error_code.map_or(Value::Null, Value::String),
        );
        public_fields.insert("facts_json".to_string(), Value::String(facts_json));

        let row_hash = compute_row_hash(&RowHashInput {
            device_identity: AUDIT_DEVICE_IDENTITY,
            timestamp: &timestamp,
            event_id: &event_id,
            event_type: AUDIT_EVENT_TYPE,
            level: "info",
            prev_hash: &state.prev_hash,
            public_fields: &public_fields,
            groups: &BTreeMap::new(),
        });

        let mut row = Map::new();
        row.insert(
            "device_identity".to_string(),
            Value::String(AUDIT_DEVICE_IDENTITY.to_string()),
        );
        row.insert("timestamp".to_string(), Value::String(timestamp));
        row.insert("event_id".to_string(), Value::String(event_id));
        row.insert(
            "event_type".to_string(),
            Value::String(AUDIT_EVENT_TYPE.to_string()),
        );
        row.insert("level".to_string(), Value::String("info".to_string()));
        row.insert("sequence".to_string(), Value::from(state.sequence));
        row.insert(
            "prev_hash".to_string(),
            Value::String(state.prev_hash.clone()),
        );
        row.insert("row_hash".to_string(), Value::String(row_hash.clone()));
        row.insert("signature".to_string(), Value::String(String::new()));
        row.extend(public_fields);

        let mut line = serde_json::to_vec(&Value::Object(row))?;
        line.push(b'\n');
        state.writer.write_all(&line)?;
        state.writer.flush()?;

        let record = AuditRecord {
            prev_hash: state.prev_hash.clone(),
            row_hash: row_hash.clone(),
        };
        state.prev_hash = row_hash;
        state.sequence += 1;
        Ok(record)
    }
}
