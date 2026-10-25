use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde_json::{Map, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::config::RuntimeConfig;
use crate::model_phase::Phase;

pub const SENSITIVE_TRACE_SCHEMA: &str = "tn.llama.sensitive-trace.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildMetadata {
    pub build_id: String,
    pub llama_source_revision: String,
    pub model_sha256: String,
    pub binary_sha256: String,
}

impl From<&RuntimeConfig> for BuildMetadata {
    fn from(config: &RuntimeConfig) -> Self {
        Self {
            build_id: config.build_id.clone(),
            llama_source_revision: config.llama_source_revision.clone(),
            model_sha256: config.model_sha256.clone(),
            binary_sha256: config.binary_sha256.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SensitiveTraceKind {
    Ingress,
    TurnPrepared,
    TurnStarted,
    ModelResult,
    Error,
    Release,
}

impl SensitiveTraceKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ingress => "ingress",
            Self::TurnPrepared => "turn_prepared",
            Self::TurnStarted => "turn_started",
            Self::ModelResult => "model_result",
            Self::Error => "error",
            Self::Release => "release",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SensitiveTraceEvent<'a> {
    pub run_id: &'a str,
    pub phase: Option<Phase>,
    pub kind: SensitiveTraceKind,
    pub task_id: Option<i64>,
    pub slot_id: Option<i32>,
    pub request_json: Option<&'a [u8]>,
    pub rendered_prompt: Option<&'a [u8]>,
    pub runnable_manifest: Option<&'a [u8]>,
    pub raw_model_output: Option<&'a [u8]>,
    pub parsed_content: Option<&'a [u8]>,
    pub error_code: Option<&'a str>,
}

pub trait SensitiveTraceSink: Send + Sync {
    fn append(&self, event: SensitiveTraceEvent<'_>) -> io::Result<()>;
}

pub struct SensitiveTrace {
    path: PathBuf,
    run_id: String,
    metadata: BuildMetadata,
    process_id: u32,
    state: Mutex<SensitiveTraceState>,
}

struct SensitiveTraceState {
    file: File,
    sequence: u64,
}

impl SensitiveTrace {
    pub fn create(
        raw_trace_dir: impl AsRef<Path>,
        run_id: impl Into<String>,
        metadata: BuildMetadata,
    ) -> io::Result<Self> {
        let raw_trace_dir = raw_trace_dir.as_ref();
        if !raw_trace_dir.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sensitive trace directory must be absolute",
            ));
        }
        let run_id = run_id.into();
        let path = raw_trace_dir.join(format!("{run_id}.sensitive.ndjson"));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            path,
            run_id,
            metadata,
            process_id: std::process::id(),
            state: Mutex::new(SensitiveTraceState { file, sequence: 0 }),
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl SensitiveTraceSink for SensitiveTrace {
    fn append(&self, event: SensitiveTraceEvent<'_>) -> io::Result<()> {
        if event.run_id != self.run_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sensitive trace event run ID does not match the sink",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("sensitive trace writer state is unavailable"))?;
        let timestamp = OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .map_err(io::Error::other)?;

        let mut record = Map::new();
        record.insert(
            "schema".to_string(),
            Value::String(SENSITIVE_TRACE_SCHEMA.to_string()),
        );
        record.insert("run_id".to_string(), Value::String(self.run_id.clone()));
        record.insert("sequence".to_string(), Value::from(state.sequence));
        record.insert("timestamp".to_string(), Value::String(timestamp));
        record.insert(
            "phase".to_string(),
            event
                .phase
                .map_or(Value::Null, |phase| Value::from(u32::from(phase))),
        );
        record.insert(
            "kind".to_string(),
            Value::String(event.kind.as_str().to_string()),
        );
        record.insert(
            "task_id".to_string(),
            event.task_id.map_or(Value::Null, Value::from),
        );
        record.insert(
            "slot_id".to_string(),
            event.slot_id.map_or(Value::Null, Value::from),
        );
        insert_bytes(&mut record, "request_json_b64", event.request_json);
        insert_bytes(&mut record, "rendered_prompt_b64", event.rendered_prompt);
        insert_bytes(
            &mut record,
            "runnable_manifest_b64",
            event.runnable_manifest,
        );
        insert_bytes(&mut record, "raw_model_output_b64", event.raw_model_output);
        insert_bytes(&mut record, "parsed_content_b64", event.parsed_content);
        record.insert(
            "error_code".to_string(),
            event
                .error_code
                .map_or(Value::Null, |code| Value::String(code.to_string())),
        );
        record.insert(
            "build_id".to_string(),
            Value::String(self.metadata.build_id.clone()),
        );
        record.insert(
            "llama_source_revision".to_string(),
            Value::String(self.metadata.llama_source_revision.clone()),
        );
        record.insert(
            "model_sha256".to_string(),
            Value::String(self.metadata.model_sha256.clone()),
        );
        record.insert(
            "binary_sha256".to_string(),
            Value::String(self.metadata.binary_sha256.clone()),
        );
        record.insert("process_id".to_string(), Value::from(self.process_id));

        let mut line = serde_json::to_vec(&Value::Object(record))?;
        line.push(b'\n');
        state.file.write_all(&line)?;
        state.file.flush()?;
        state.file.sync_data()?;
        state.sequence += 1;
        Ok(())
    }
}

fn insert_bytes(record: &mut Map<String, Value>, field: &str, bytes: Option<&[u8]>) {
    record.insert(
        field.to_string(),
        bytes.map_or(Value::Null, |value| Value::String(STANDARD.encode(value))),
    );
}
