use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::contract::ReportContract;

pub const TEST_DEVICE_IDENTITY: &str = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestLimits {
    pub max_request_bytes: usize,
    pub max_message_bytes: usize,
    pub expected_device_identity: Option<String>,
    pub expected_event_type: String,
    pub allowed_input_groups: BTreeSet<String>,
    pub require_signature: bool,
    pub max_active_sessions: usize,
    pub max_chain_namespaces: usize,
}

impl RequestLimits {
    #[doc(hidden)]
    #[must_use]
    pub fn test() -> Self {
        Self {
            max_request_bytes: 1024 * 1024,
            max_message_bytes: 16 * 1024,
            expected_device_identity: Some(TEST_DEVICE_IDENTITY.to_string()),
            expected_event_type: "tn.llama.request".to_string(),
            allowed_input_groups: BTreeSet::new(),
            require_signature: false,
            max_active_sessions: 1,
            max_chain_namespaces: 1,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub governed_objects: Option<crate::governed_input::GovernedInputConfig>,
    pub version: String,
    pub writer_authority: String,
    pub expected_event_type: String,
    pub allowed_input_groups: BTreeSet<String>,
    pub report_contract: ReportContract,
    pub reader_key_dir: PathBuf,
    pub input_index_key_path: PathBuf,
    pub ceremony_id: String,
    pub governance_group: String,
    pub input_index_epoch: u64,
    pub output_tn_config_path: PathBuf,
    pub output_reader_key_dir: PathBuf,
    pub output_index_key_path: PathBuf,
    pub output_ceremony_id: String,
    pub output_governance_group: String,
    pub output_index_epoch: u64,
    pub audit_path: PathBuf,
    pub raw_trace_dir: PathBuf,
    pub build_id: String,
    pub llama_source_revision: String,
    pub model_sha256: String,
    pub binary_sha256: String,
    pub max_request_bytes: usize,
    pub max_message_bytes: usize,
    pub max_active_sessions: usize,
    pub max_chain_namespaces: usize,
    pub echo_max_tokens: u32,
    pub final_max_tokens: u32,
}

pub(crate) fn output_state_is_separate(config: &RuntimeConfig) -> bool {
    if config.ceremony_id == config.output_ceremony_id {
        return false;
    }

    let Ok(writer_config) = tn_core::config::load(&config.output_tn_config_path) else {
        return false;
    };
    let Some(output_yaml_dir) = config.output_tn_config_path.parent() else {
        return false;
    };
    let output_keystore_path =
        tn_core::handlers::spec::resolve_path(&writer_config.keystore.path, output_yaml_dir);
    let output_log_path =
        tn_core::handlers::spec::resolve_path(&writer_config.logs.path, output_yaml_dir);
    let paths = [
        &config.reader_key_dir,
        &config.input_index_key_path,
        &config.output_reader_key_dir,
        &config.output_index_key_path,
    ]
    .map(std::fs::canonicalize);
    let [Ok(source_reader), Ok(source_index), Ok(output_reader), Ok(output_index)] = paths else {
        return false;
    };
    let Some(output_keystore) = canonical_state_path(&output_keystore_path) else {
        return false;
    };
    let Some(output_log) = canonical_state_path(&output_log_path) else {
        return false;
    };

    let mut source_paths = vec![source_reader, source_index];
    if let Some(profile) = &config.governed_objects {
        let Ok(native_config) = tn_core::config::load(&profile.native_reader_config_path) else {
            return false;
        };
        let Some(directory) = profile.native_reader_config_path.parent() else {
            return false;
        };
        if native_config.ceremony.id != config.ceremony_id {
            return false;
        }
        let native_keystore =
            tn_core::handlers::spec::resolve_path(&native_config.keystore.path, directory);
        let Some(native_keystore) = canonical_state_path(&native_keystore) else {
            return false;
        };
        source_paths.push(native_keystore);
    }
    source_paths.iter().all(|source| {
        [
            output_keystore.as_path(),
            output_log.as_path(),
            output_reader.as_path(),
            output_index.as_path(),
        ]
        .iter()
        .all(|output| !paths_overlap(source, output))
    })
}

fn canonical_state_path(path: &Path) -> Option<PathBuf> {
    let mut unresolved = Vec::new();
    let mut existing = path;
    loop {
        match std::fs::canonicalize(existing) {
            Ok(mut canonical) => {
                for component in unresolved.iter().rev() {
                    canonical.push(component);
                }
                return Some(canonical);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                unresolved.push(existing.file_name()?.to_os_string());
                existing = existing.parent()?;
            }
            Err(_) => return None,
        }
    }
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}
