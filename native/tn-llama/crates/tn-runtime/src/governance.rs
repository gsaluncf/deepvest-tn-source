use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use crate::envelope::VerifiedRow;
use crate::error::{AdmissionStage, TnLlamaError};

pub(crate) const GOVERNANCE_GROUP: &str = "tn.agents";
pub const GOVERNANCE_FIELDS: [&str; 6] = [
    "consequences",
    "do_not_use_for",
    "instruction",
    "on_violation_or_error",
    "policy",
    "use_for",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GovernanceKeyProvenance {
    pub ceremony_id: String,
    pub group: String,
    pub index_epoch: u64,
}

impl GovernanceKeyProvenance {
    #[must_use]
    pub fn new(ceremony_id: impl Into<String>, group: impl Into<String>, index_epoch: u64) -> Self {
        Self {
            ceremony_id: ceremony_id.into(),
            group: group.into(),
            index_epoch,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedGovernance {
    pub plaintext: BTreeMap<String, Value>,
    pub commitments: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedGroup {
    pub name: String,
    pub plaintext: BTreeMap<String, Value>,
    pub commitments: BTreeMap<String, String>,
}

pub struct GovernanceReader {
    reader_dir: PathBuf,
    index_master_key: Zeroizing<[u8; 32]>,
    provenance: GovernanceKeyProvenance,
    index_key_digest: String,
}

impl GovernanceReader {
    pub fn from_paths(
        reader_dir: &Path,
        index_key_path: &Path,
        provenance: GovernanceKeyProvenance,
    ) -> Result<Self, TnLlamaError> {
        if provenance.ceremony_id.trim().is_empty()
            || provenance.group != GOVERNANCE_GROUP
            || provenance.index_epoch != 0
        {
            return Err(governance_error(
                "governance_provenance_invalid",
                "version 1 requires a nonempty ceremony, tn.agents, and index epoch 0",
            ));
        }
        validate_reader_directory(reader_dir)?;

        let key_bytes = Zeroizing::new(fs::read(index_key_path).map_err(|_| {
            governance_error(
                "governance_index_key_invalid",
                "governance index key could not be read",
            )
        })?);
        if key_bytes.len() != 32 {
            return Err(governance_error(
                "governance_index_key_invalid",
                "governance index key must contain exactly 32 bytes",
            ));
        }
        let mut index_master_key = Zeroizing::new([0_u8; 32]);
        index_master_key.copy_from_slice(&key_bytes);
        let index_key_digest = format!(
            "sha256:{}",
            hex::encode(Sha256::digest(index_master_key.as_slice())),
        );

        Ok(Self {
            reader_dir: reader_dir.to_path_buf(),
            index_master_key,
            provenance,
            index_key_digest,
        })
    }

    pub fn open_and_verify(&self, row: &VerifiedRow) -> Result<VerifiedGovernance, TnLlamaError> {
        let group = self.open_group_and_verify(row, GOVERNANCE_GROUP)?;
        let plaintext = parse_governance_plaintext(&Value::Object(
            group.plaintext.clone().into_iter().collect(),
        ))?;
        let policy = plaintext
            .get("policy")
            .and_then(Value::as_str)
            .expect("required policy was parsed");
        if row.encrypted_group_manifest().keys().any(|name| {
            row.aad_marker(name)
                .and_then(|marker| marker.get("policy"))
                .and_then(Value::as_str)
                != Some(policy)
        }) {
            return Err(governance_error(
                "governance_aad_policy_mismatch",
                "encrypted group AAD policy reference does not match opened governance",
            ));
        }
        Ok(VerifiedGovernance {
            plaintext,
            commitments: group.commitments,
        })
    }

    pub fn open_group_and_verify(
        &self,
        row: &VerifiedRow,
        group: &str,
    ) -> Result<VerifiedGroup, TnLlamaError> {
        let commitments = row.group_field_hashes(group).cloned().ok_or_else(|| {
            governance_error(
                "protected_group_absent",
                "requested protected group is absent from the verified row",
            )
        })?;
        let opened = tn_core::unseal_as_recipient(
            row.recipient_source_json(),
            &self.reader_dir,
            group,
            true,
        )
        .map_err(|_| {
            governance_error(
                "protected_group_open_failed",
                "protected group recipient open or canonical verification failed",
            )
        })?;
        let value = opened.plaintext.get(group).ok_or_else(|| {
            governance_error(
                "protected_group_not_opened",
                "protected group did not open with the installed BTN capability",
            )
        })?;
        let plaintext = value
            .as_object()
            .ok_or_else(|| {
                governance_error(
                    "protected_group_shape",
                    "opened protected group must be a JSON object",
                )
            })?
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        self.verify_fields(group, plaintext, commitments)
    }

    pub(crate) fn verify_native_group(
        &self,
        object: &tn_core::governed::GovernedObject,
        group: &str,
        fields: &serde_json::Map<String, Value>,
    ) -> Result<VerifiedGroup, TnLlamaError> {
        let commitments = object
            .envelope()
            .get(group)
            .and_then(|block| block.get("field_hashes"))
            .cloned()
            .ok_or_else(|| {
                governance_error("protected_group_absent", "native group lacks commitments")
            })?;
        let commitments: BTreeMap<String, String> =
            serde_json::from_value(commitments).map_err(|_| {
                governance_error("protected_group_shape", "native commitments are malformed")
            })?;
        self.verify_fields(group, fields.clone().into_iter().collect(), commitments)
    }

    fn verify_fields(
        &self,
        group: &str,
        plaintext: BTreeMap<String, Value>,
        commitments: BTreeMap<String, String>,
    ) -> Result<VerifiedGroup, TnLlamaError> {
        if plaintext.keys().ne(commitments.keys()) {
            return Err(governance_error(
                "protected_group_field_set_mismatch",
                "opened group and verified commitments have different field sets",
            ));
        }
        let group_index_key = tn_core::indexing::derive_group_index_key(
            &*self.index_master_key,
            &self.provenance.ceremony_id,
            group,
            self.provenance.index_epoch,
        )
        .map_err(|_| {
            governance_error(
                "protected_group_commitment_compute_failed",
                "official group commitment key derivation failed",
            )
        })?;
        for (field, value) in &plaintext {
            let expected =
                tn_core::indexing::index_token(&group_index_key, field, value).map_err(|_| {
                    governance_error(
                        "protected_group_commitment_compute_failed",
                        "official group commitment computation failed",
                    )
                })?;
            let stored = commitments.get(field).expect("field sets were compared");
            if expected.as_bytes().ct_eq(stored.as_bytes()).unwrap_u8() != 1 {
                return Err(governance_error(
                    "protected_group_field_commitment_mismatch",
                    "opened group does not match its verified TN commitment set",
                ));
            }
        }
        Ok(VerifiedGroup {
            name: group.to_string(),
            plaintext,
            commitments,
        })
    }

    #[must_use]
    pub fn index_key_digest(&self) -> &str {
        &self.index_key_digest
    }

    #[must_use]
    pub const fn provenance(&self) -> &GovernanceKeyProvenance {
        &self.provenance
    }
}

fn validate_reader_directory(reader_dir: &Path) -> Result<(), TnLlamaError> {
    let metadata = fs::metadata(reader_dir).map_err(|_| {
        governance_error(
            "governance_reader_material_invalid",
            "governance reader directory could not be read",
        )
    })?;
    if !metadata.is_dir() {
        return Err(governance_error(
            "governance_reader_material_invalid",
            "governance reader path is not a directory",
        ));
    }
    let mut entries = fs::read_dir(reader_dir)
        .map_err(|_| {
            governance_error(
                "governance_reader_material_invalid",
                "governance reader directory could not be enumerated",
            )
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            governance_error(
                "governance_reader_material_invalid",
                "governance reader directory could not be enumerated",
            )
        })?;
    if entries.is_empty() {
        return Err(governance_error(
            "governance_reader_material_invalid",
            "reader directory must contain BTN reader kits",
        ));
    }
    let mut has_governance = false;
    for entry in entries.drain(..) {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.ends_with(".btn.mykit")
            || !entry.file_type().is_ok_and(|file_type| file_type.is_file())
        {
            return Err(governance_error(
                "governance_reader_material_invalid",
                "reader directory may contain only BTN reader kits",
            ));
        }
        has_governance |= file_name == "tn.agents.btn.mykit";
    }
    if !has_governance {
        return Err(governance_error(
            "governance_reader_material_invalid",
            "reader directory must contain the tn.agents BTN reader kit",
        ));
    }
    Ok(())
}

fn parse_governance_plaintext(value: &Value) -> Result<BTreeMap<String, Value>, TnLlamaError> {
    let object = value.as_object().ok_or_else(governance_plaintext_shape)?;
    for field in GOVERNANCE_FIELDS {
        if object
            .get(field)
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(governance_plaintext_shape());
        }
    }
    Ok(object
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect())
}

fn governance_plaintext_shape() -> TnLlamaError {
    governance_error(
        "governance_plaintext_shape",
        "tn.agents plaintext must contain nonempty strings for all six required fields",
    )
}

fn governance_error(code: &'static str, message: &'static str) -> TnLlamaError {
    TnLlamaError::new(code, AdmissionStage::Governance, message)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::parse_governance_plaintext;

    #[test]
    fn plaintext_shape_defense_rejects_missing_empty_or_non_string_required_values() {
        for value in [
            json!({
                "consequences": "c",
                "do_not_use_for": "d",
                "instruction": "i",
                "on_violation_or_error": "o",
                "use_for": "u"
            }),
            json!({
                "consequences": "c",
                "do_not_use_for": "d",
                "instruction": "",
                "on_violation_or_error": "o",
                "policy": "p",
                "use_for": "u"
            }),
            json!({
                "consequences": "c",
                "do_not_use_for": "d",
                "instruction": 7,
                "on_violation_or_error": "o",
                "policy": "p",
                "use_for": "u"
            }),
        ] {
            let error = parse_governance_plaintext(&value).unwrap_err();
            assert_eq!(error.code(), "governance_plaintext_shape");
        }
    }
}
