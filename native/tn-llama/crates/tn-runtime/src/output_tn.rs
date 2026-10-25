use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tn_core::sealed_object::{extract_group_blocks, require_envelope_shape, verify_sealed};
use tn_core::{OnInvalid, SecureReadOptions};

use crate::config::{output_state_is_separate, RequestLimits, RuntimeConfig};
use crate::contract::{OutputPolicy, ReportContract};
use crate::envelope::{parse_and_verify_request, parse_duplicate_safe_json, VerifiedRow};
use crate::error::{AdmissionStage, TnLlamaError};
use crate::governance::{
    GovernanceKeyProvenance, GovernanceReader, VerifiedGovernance, VerifiedGroup, GOVERNANCE_FIELDS,
};
use crate::tools::ComplianceInvocation;

const DEFAULT_GROUP: &str = "default";
const GOVERNANCE_GROUP: &str = "tn.agents";
const COMPLIANCE_EVENT_TYPE: &str = "compliance.creation.logged";
const OUTPUT_GROUPS: [&str; 2] = [DEFAULT_GROUP, GOVERNANCE_GROUP];

pub struct OutputPublisher {
    runtime: tn_core::Runtime,
    reader: GovernanceReader,
}

/// Portable signed output uses its own native publisher, independently of v1 logs.
pub(crate) struct GovernedOutputPublisher {
    objects: tn_core::runtime::Objects<'static>,
    reader: tn_core::governed::GovernedReader,
    verifier: GovernanceReader,
    release_register_enabled: bool,
}

pub(crate) struct GovernedPublication {
    pub object: tn_core::governed::GovernedObject,
    pub view: tn_core::governed::GovernanceView,
    pub data: tn_core::governed::DataObject,
    pub register_error: Option<String>,
    pub register_enabled: bool,
}

impl GovernedOutputPublisher {
    pub fn from_config(config: &RuntimeConfig) -> Result<Self, TnLlamaError> {
        if !output_state_is_separate(config) {
            return Err(output_error(
                "tn_output_state_alias",
                "source and output material must be separate",
            ));
        }
        let objects =
            tn_core::runtime::Objects::open(&config.output_tn_config_path).map_err(|_| {
                output_error(
                    "tn_output_init_failed",
                    "native output context could not be opened",
                )
            })?;
        let native_config = tn_core::config::load(&config.output_tn_config_path).map_err(|_| {
            output_error(
                "tn_output_init_failed",
                "native output configuration could not be read",
            )
        })?;
        if native_config.ceremony.id != config.output_ceremony_id
            || ["default", "tn.agents"].iter().any(|name| {
                native_config.groups.get(*name).is_none_or(|group| {
                    group.policy != "private" || group.index_epoch != config.output_index_epoch
                })
            })
        {
            return Err(output_error(
                "tn_output_config_mismatch",
                "native output provenance and private groups must match",
            ));
        }
        let mut reader = tn_core::governed::GovernedReader::new();
        for group in ["default", "tn.agents"] {
            let kit = std::fs::read(
                config
                    .output_reader_key_dir
                    .join(format!("{group}.btn.mykit")),
            )
            .map_err(|_| {
                output_error(
                    "tn_output_init_failed",
                    "independent output reader capability is absent",
                )
            })?;
            let cipher =
                tn_core::cipher::btn::BtnReaderCipher::from_kit_bytes(&kit).map_err(|_| {
                    output_error(
                        "tn_output_init_failed",
                        "independent output reader capability is invalid",
                    )
                })?;
            reader = reader
                .with_group(group, std::sync::Arc::new(cipher))
                .map_err(|_| {
                    output_error(
                        "tn_output_init_failed",
                        "independent output reader could not be built",
                    )
                })?;
        }
        let verifier = GovernanceReader::from_paths(
            &config.output_reader_key_dir,
            &config.output_index_key_path,
            GovernanceKeyProvenance::new(
                &config.output_ceremony_id,
                "tn.agents",
                config.output_index_epoch,
            ),
        )?;
        Ok(Self {
            objects,
            reader,
            verifier,
            release_register_enabled: std::env::var_os("TN_OBJECT_RELEASE_REGISTER")
                .is_some_and(|value| !value.is_empty()),
        })
    }

    pub fn emit(
        &self,
        data: &tn_core::governed::DataObject,
        report_html: &str,
        object_type: &str,
        request: &crate::AcceptedGovernedRequest,
        source: &tn_core::governed::GovernanceView,
        input: &crate::governed_input::GovernedInputRuntime,
        tools: &[GovernedPublication],
    ) -> Result<GovernedPublication, TnLlamaError> {
        let mut candidate = data.clone();
        for tool in tools {
            candidate.include(&tool.data).map_err(|_| {
                output_error(
                    "tn_output_release_denied",
                    "authenticated tool result could not be included",
                )
            })?;
        }
        let parents = std::iter::once(source)
            .chain(tools.iter().map(|tool| &tool.view))
            .collect::<Vec<_>>();
        self.publish(
            candidate,
            json!({"report_html":report_html}),
            object_type,
            request,
            &parents,
            input,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn publish(
        &self,
        mut candidate: tn_core::governed::DataObject,
        content: Value,
        object_type: &str,
        request: &crate::AcceptedGovernedRequest,
        parents: &[&tn_core::governed::GovernanceView],
        input: &crate::governed_input::GovernedInputRuntime,
        attach_report_policy: bool,
    ) -> Result<GovernedPublication, TnLlamaError> {
        candidate
            .set_group("default", &content)
            .and_then(|()| candidate.retain_groups(["default"]))
            .map_err(|_| {
                output_error(
                    "tn_report_proposal_invalid",
                    "report content could not be projected",
                )
            })?;
        if let Some(policy) = input
            .report_policy
            .as_ref()
            .filter(|_| attach_report_policy)
        {
            self.objects
                .attach(&mut candidate, policy.clone(), |context| {
                    Ok(context.authority() == self.objects.did() && context.policy() == policy)
                })
                .map_err(|_| {
                    output_error(
                        "tn_output_release_denied",
                        "configured report contract attachment failed",
                    )
                })?;
        }
        let expected_policies = candidate.policies().map_err(|_| {
            output_error("tn_output_release_denied", "report contract set is invalid")
        })?;
        let expected_bindings = candidate.dataset_bindings().map_err(|_| {
            output_error(
                "tn_output_release_denied",
                "publication bindings are invalid",
            )
        })?;
        let output = self
            .objects
            .release_for(
                &mut candidate,
                object_type,
                &request.use_context,
                &request.destination,
                |ctx| {
                    Ok(ctx.use_context() == Some(&input.config.use_context)
                        && input.config.destinations.contains(ctx.destination())
                        && ctx.data().sources().len() == parents.len()
                        && parents.iter().all(|parent| {
                            ctx.data().sources().iter().any(|reference| {
                                reference
                                    .references_with_policy(parent.object(), parent.governance())
                            })
                        })
                        && ctx.data().dataset_bindings()? == expected_bindings
                        && ctx.data().policies()? == expected_policies)
                },
            )
            .map_err(|_| {
                output_error(
                    "tn_output_release_denied",
                    "native report release was refused",
                )
            })?;
        // Optional register I/O never invalidates or regenerates the signed wire.
        let register_error = candidate.register_error().map(str::to_owned);
        let output = tn_core::governed::GovernedObject::parse(output.wire()).map_err(|_| {
            output_error(
                "tn_output_readback_failed",
                "native output failed signed reread",
            )
        })?;
        if output.writer() != self.objects.did()
            || output.object_type() != object_type
            || output.group_names() != vec!["default", "tn.agents"]
        {
            return Err(output_error(
                "tn_output_readback_failed",
                "native output identity or group projection changed",
            ));
        }
        let view = self.reader.governance(&output).map_err(|_| {
            output_error(
                "tn_output_readback_failed",
                "output governance did not open",
            )
        })?;
        self.verifier
            .verify_native_group(&output, "tn.agents", view.governance().fields())?;
        let admitted = view
            .clone()
            .accept(request.use_context.clone(), ["default"], |ctx| {
                Ok(input.accepts_output(ctx, &view, parents)?
                    && ctx.policies()? == expected_policies
                    && ctx.governance().dataset_bindings()? == expected_bindings
                    && ctx.governance().fields()["release_context"]
                        == json!({
                            "application": request.use_context.application(),
                            "purpose": request.use_context.purpose(),
                            "operation": request.use_context.operation(),
                            "destination": request.destination,
                        }))
            })
            .map_err(|_| {
                output_error(
                    "tn_output_readback_failed",
                    "output inherited governance failed admission",
                )
            })?;
        let opened = self
            .reader
            .open(&admitted, ["default"])
            .map_err(|_| output_error("tn_output_readback_failed", "output report did not open"))?;
        self.verifier.verify_native_group(
            &output,
            "default",
            opened.groups()["default"]
                .as_object()
                .expect("native group object"),
        )?;
        if opened.groups()["default"] != content {
            return Err(output_error(
                "tn_output_readback_failed",
                "output report differs from accepted content",
            ));
        }
        Ok(GovernedPublication {
            object: output,
            view,
            data: tn_core::governed::DataObject::from_opened(opened).map_err(|_| {
                output_error(
                    "tn_output_readback_failed",
                    "publication could not be retained as admitted data",
                )
            })?,
            register_error,
            register_enabled: self.release_register_enabled,
        })
    }
}

#[derive(Debug)]
pub struct VerifiedOutputRow {
    pub ndjson: String,
    pub event_id: String,
    pub event_type: String,
    pub row_hash: String,
    pub default: VerifiedGroup,
    pub governance: VerifiedGovernance,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceLineage {
    pub source_device_identity: String,
    pub source_event_id: String,
    pub source_event_type: String,
    pub derived_from_row_hash: String,
    pub source_policy: String,
    pub selected_sections: Vec<String>,
    pub operation: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComplianceReceipt {
    pub event_id: String,
    pub event_type: String,
    pub row_hash: String,
    pub source_row_hash: String,
    pub action: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReportProposal {
    pub event_type: String,
    pub public_sections: ReportPublicSections,
    pub governance: ProposedOutputGovernance,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReportPublicSections {
    pub default: DefaultReportSection,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DefaultReportSection {
    pub report_html: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProposedOutputGovernance {
    pub instruction: String,
    pub use_for: String,
    pub do_not_use_for: String,
    pub consequences: String,
    pub on_violation_or_error: String,
    pub source_lineage: Option<SourceLineage>,
    pub compliance_evidence: Option<ComplianceReceipt>,
}

#[derive(Clone, Debug)]
pub struct TrustedDerivation {
    pub contract: ReportContract,
    pub request_digest: String,
    pub source_lineage: Option<SourceLineage>,
    pub compliance_evidence: Option<ComplianceReceipt>,
}

impl OutputPublisher {
    pub fn from_config(config: &RuntimeConfig) -> Result<Self, TnLlamaError> {
        if !output_state_is_separate(config) {
            return Err(output_error(
                "tn_output_state_alias",
                "TN source and output state must use separate trust boundaries",
            ));
        }
        let writer_config = tn_core::config::load(&config.output_tn_config_path).map_err(|_| {
            output_error(
                "tn_output_init_failed",
                "configured TN output publisher config could not be loaded",
            )
        })?;
        let writer_group = writer_config.groups.get(GOVERNANCE_GROUP);
        if writer_config.ceremony.id != config.output_ceremony_id
            || config.output_governance_group != GOVERNANCE_GROUP
            || writer_group.map(|group| group.index_epoch) != Some(config.output_index_epoch)
        {
            return Err(output_error(
                "tn_output_config_mismatch",
                "TN output writer and reader provenance do not match",
            ));
        }
        validate_writer_profile(&writer_config, config.output_index_epoch)?;
        validate_output_reader_kits(&config.output_reader_key_dir)?;
        let runtime = tn_core::Runtime::init(&config.output_tn_config_path).map_err(|_| {
            output_error(
                "tn_output_init_failed",
                "configured TN output publisher could not be initialized",
            )
        })?;
        let reader = GovernanceReader::from_paths(
            &config.output_reader_key_dir,
            &config.output_index_key_path,
            GovernanceKeyProvenance::new(
                config.output_ceremony_id.clone(),
                config.output_governance_group.clone(),
                config.output_index_epoch,
            ),
        )?;
        Ok(Self { runtime, reader })
    }

    pub fn emit_report(
        &self,
        proposal: &ReportProposal,
        trusted: &TrustedDerivation,
    ) -> Result<VerifiedOutputRow, TnLlamaError> {
        validate_report_proposal(proposal, trusted)?;

        let mut fields = Map::new();
        fields.insert(
            trusted.contract.output_public_section.clone(),
            serde_json::to_value(&proposal.public_sections.default).map_err(|_| {
                output_error(
                    "tn_report_proposal_invalid",
                    "report proposal public section could not be encoded",
                )
            })?,
        );
        insert_optional_governance_extra(
            &mut fields,
            "source_lineage",
            trusted.source_lineage.as_ref(),
        )?;
        insert_optional_governance_extra(
            &mut fields,
            "compliance_evidence",
            trusted.compliance_evidence.as_ref(),
        )?;

        let verified = self.emit_and_verify(&trusted.contract.output_event_type, fields)?;
        verify_reopened_report(&verified, proposal, trusted)?;
        Ok(verified)
    }

    pub(crate) fn emit_compliance(
        &self,
        invocation: &ComplianceInvocation,
    ) -> Result<ComplianceReceipt, TnLlamaError> {
        let mut fields = Map::new();
        fields.insert(
            "default".to_string(),
            json!({
                "run_id": invocation.run_id,
                "tool_name": invocation.tool_name,
                "model_tool_call_id": invocation.model_tool_call_id,
                "source_device_identity": invocation.source_device_identity,
                "source_event_id": invocation.source_event_id,
                "source_event_type": invocation.source_event_type,
                "source_row_hash": invocation.source_row_hash,
                "selected_sections": invocation.selected_sections,
                "action": invocation.action,
            }),
        );
        let verified = self.emit_and_verify(COMPLIANCE_EVENT_TYPE, fields)?;
        Ok(ComplianceReceipt {
            event_id: verified.event_id,
            event_type: verified.event_type,
            row_hash: verified.row_hash,
            source_row_hash: invocation.source_row_hash.clone(),
            action: invocation.action.clone(),
        })
    }

    fn emit_and_verify(
        &self,
        event_type: &str,
        fields: Map<String, Value>,
    ) -> Result<VerifiedOutputRow, TnLlamaError> {
        let policy_ref = output_policy_reference(&self.runtime, event_type)?;
        let aad = Map::from_iter([
            (
                "governed_by".to_string(),
                Value::String(self.runtime.did().to_string()),
            ),
            ("policy".to_string(), Value::String(policy_ref.clone())),
        ]);
        let expected_aad = expected_group_aad(&aad);
        let ndjson = self
            .runtime
            .emit_with_aad_returning_line("info", event_type, fields, None, None, Some(true), &aad)
            .map_err(|_| {
                output_error(
                    "tn_output_emit_failed",
                    "TN output publisher rejected the governed event",
                )
            })?
            .ok_or_else(|| {
                output_error(
                    "tn_output_filtered",
                    "TN output publisher filtered the governed event",
                )
            })?;
        if !ndjson.ends_with('\n') {
            return Err(output_error(
                "tn_output_shape",
                "TN output publisher did not return one newline-terminated row",
            ));
        }

        let parsed =
            parse_duplicate_safe_json(ndjson.trim_end_matches('\n').as_bytes()).map_err(|_| {
                output_error("tn_output_json_invalid", "TN output row is not strict JSON")
            })?;
        let envelope =
            require_envelope_shape(parsed.as_object().cloned().ok_or_else(|| {
                output_error("tn_output_shape", "TN output row is not an object")
            })?)
            .map_err(|_| {
                output_error("tn_output_shape", "TN output row has an invalid envelope")
            })?;
        let blocks = extract_group_blocks(&envelope).map_err(|_| {
            output_error(
                "tn_output_shape",
                "TN output row encrypted groups could not be extracted",
            )
        })?;
        if blocks.keys().map(String::as_str).collect::<BTreeSet<_>>()
            != OUTPUT_GROUPS.into_iter().collect::<BTreeSet<_>>()
        {
            return Err(output_error(
                "tn_output_shape",
                "TN output row must contain exactly encrypted default and tn.agents groups",
            ));
        }
        require_only_output_wire_fields(&envelope)?;
        let sealed = verify_sealed(&envelope, &blocks);
        if !sealed.row_hash {
            return Err(output_error(
                "tn_output_row_hash_mismatch",
                "TN output row hash did not verify canonically",
            ));
        }
        if !sealed.signature {
            return Err(output_error(
                "tn_output_signature_mismatch",
                "TN output row signature did not verify for its writer DID",
            ));
        }

        let event_id = output_header(&envelope, "event_id")?.to_string();
        let emitted_event_type = output_header(&envelope, "event_type")?.to_string();
        let row_hash = output_header(&envelope, "row_hash")?.to_string();
        let device_identity = output_header(&envelope, "device_identity")?.to_string();
        if emitted_event_type != event_type
            || device_identity != self.runtime.did()
            || envelope
                .get("signature")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Err(output_error(
                "tn_output_identity_mismatch",
                "TN output row identity did not match the requested signed event",
            ));
        }
        verify_output_aad(&envelope, &expected_aad)?;

        let reread = self
            .runtime
            .secure_read(SecureReadOptions {
                on_invalid: OnInvalid::Raise,
                ..SecureReadOptions::default()
            })
            .map_err(|_| {
                output_error(
                    "tn_output_reread_failed",
                    "TN strict reread rejected the output log",
                )
            })?;
        let last = reread.last().ok_or_else(|| {
            output_error(
                "tn_output_reread_mismatch",
                "TN strict reread returned no output event",
            )
        })?;
        if last.fields.get("event_id").and_then(Value::as_str) != Some(&event_id)
            || last.fields.get("event_type").and_then(Value::as_str)
                != Some(emitted_event_type.as_str())
            || last.fields.get("device_identity").and_then(Value::as_str)
                != Some(device_identity.as_str())
        {
            return Err(output_error(
                "tn_output_reread_mismatch",
                "TN strict reread did not end at the emitted output identity",
            ));
        }
        let raw_reread = self.runtime.read_raw_with_validity().map_err(|_| {
            output_error(
                "tn_output_reread_failed",
                "TN lossless reread rejected the output log",
            )
        })?;
        let raw_last = raw_reread.last().ok_or_else(|| {
            output_error(
                "tn_output_reread_mismatch",
                "TN lossless reread returned no output event",
            )
        })?;
        require_exact_reread(&parsed, raw_last)?;
        let verified_row = parse_verified_output_row(&parsed, &device_identity, event_type)?;
        if verified_row.event_id() != event_id || verified_row.row_hash() != row_hash {
            return Err(output_error(
                "tn_output_identity_mismatch",
                "verified TN output identity or row hash changed during reread",
            ));
        }
        let governance = self.reader.open_and_verify(&verified_row)?;
        let default = self
            .reader
            .open_group_and_verify(&verified_row, DEFAULT_GROUP)?;
        if governance.plaintext.get("policy").and_then(Value::as_str) != Some(policy_ref.as_str()) {
            return Err(output_error(
                "tn_output_policy_mismatch",
                "reopened TN output policy does not match its AAD policy reference",
            ));
        }

        Ok(VerifiedOutputRow {
            ndjson,
            event_id,
            event_type: emitted_event_type,
            row_hash,
            default,
            governance,
        })
    }
}

fn validate_writer_profile(
    writer: &tn_core::config::Config,
    expected_epoch: u64,
) -> Result<(), TnLlamaError> {
    if !writer.ceremony.sign || !writer.ceremony.chain || !writer.public_fields.is_empty() {
        return Err(output_error(
            "tn_output_config_mismatch",
            "TN output writer must sign, chain, and expose no application fields",
        ));
    }

    for group in OUTPUT_GROUPS {
        let spec = writer.groups.get(group).ok_or_else(|| {
            output_error(
                "tn_output_config_mismatch",
                "TN output writer omitted a required encrypted group",
            )
        })?;
        if spec.policy != "private"
            || spec.cipher != "btn"
            || spec.index_epoch != expected_epoch
            || !spec.aad.is_empty()
        {
            return Err(output_error(
                "tn_output_config_mismatch",
                "TN output groups must use private BTN, the configured epoch, and no group-specific AAD defaults",
            ));
        }
    }

    let routes = writer.field_to_groups().map_err(|_| {
        output_error(
            "tn_output_config_mismatch",
            "TN output field routing is invalid",
        )
    })?;
    for field in GOVERNANCE_FIELDS
        .into_iter()
        .chain(["source_lineage", "compliance_evidence"])
    {
        if routes
            .get(field)
            .is_none_or(|groups| groups.as_slice() != [GOVERNANCE_GROUP])
        {
            return Err(output_error(
                "tn_output_config_mismatch",
                "TN output governance fields must route to tn.agents",
            ));
        }
    }
    if routes
        .get(DEFAULT_GROUP)
        .is_some_and(|groups| groups.as_slice() != [DEFAULT_GROUP])
    {
        return Err(output_error(
            "tn_output_config_mismatch",
            "TN output default payload must route only to the default group",
        ));
    }
    Ok(())
}

fn validate_output_reader_kits(reader_dir: &std::path::Path) -> Result<(), TnLlamaError> {
    for group in OUTPUT_GROUPS {
        let path = reader_dir.join(format!("{group}.btn.mykit"));
        if !path.metadata().is_ok_and(|metadata| metadata.is_file()) {
            return Err(output_error(
                "tn_output_reader_material_invalid",
                "TN output recipient must hold default and tn.agents BTN kits",
            ));
        }
    }
    Ok(())
}

fn output_policy_reference(
    runtime: &tn_core::Runtime,
    event_type: &str,
) -> Result<String, TnLlamaError> {
    let template = runtime
        .agent_policy_doc()
        .and_then(|document| document.templates.get(event_type))
        .ok_or_else(|| {
            output_error(
                "tn_output_policy_missing",
                "TN output policy has no template for the requested event",
            )
        })?;
    Ok(format!(
        "{}#{}@{}#{}",
        template.path, template.event_type, template.version, template.content_hash
    ))
}

fn expected_group_aad(per_emit: &Map<String, Value>) -> BTreeMap<String, Map<String, Value>> {
    OUTPUT_GROUPS
        .into_iter()
        .map(|group| (group.to_string(), per_emit.clone()))
        .collect()
}

fn require_only_output_wire_fields(envelope: &Map<String, Value>) -> Result<(), TnLlamaError> {
    for (name, value) in envelope {
        if tn_core::sealed_object::ENVELOPE_RESERVED.contains(&name.as_str())
            || name == DEFAULT_GROUP
            || name == GOVERNANCE_GROUP
            || name == "tn_aad"
        {
            continue;
        }
        if name == "run_id" && value.as_str().is_some_and(|run_id| !run_id.is_empty()) {
            continue;
        }
        return Err(output_error(
            "tn_output_shape",
            "TN output row exposed an unexpected public field",
        ));
    }
    Ok(())
}

fn verify_output_aad(
    envelope: &Map<String, Value>,
    expected: &BTreeMap<String, Map<String, Value>>,
) -> Result<(), TnLlamaError> {
    let raw = envelope
        .get("tn_aad")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            output_error(
                "tn_output_aad_mismatch",
                "TN output row omitted its canonical AAD echo",
            )
        })?;
    let parsed = parse_duplicate_safe_json(raw.as_bytes()).map_err(|_| {
        output_error(
            "tn_output_aad_mismatch",
            "TN output row AAD echo is not strict JSON",
        )
    })?;
    let canonical = tn_core::canonical::canonical_bytes(&parsed).map_err(|_| {
        output_error(
            "tn_output_aad_mismatch",
            "TN output row AAD echo could not be canonicalized",
        )
    })?;
    let expected_value = Value::Object(
        expected
            .iter()
            .map(|(group, marker)| (group.clone(), Value::Object(marker.clone())))
            .collect(),
    );
    if canonical.as_slice() != raw.as_bytes() || parsed != expected_value {
        return Err(output_error(
            "tn_output_aad_mismatch",
            "TN output row AAD bindings do not match the emitted transition",
        ));
    }
    Ok(())
}

fn parse_verified_output_row(
    parsed: &Value,
    device_identity: &str,
    event_type: &str,
) -> Result<VerifiedRow, TnLlamaError> {
    let request = serde_json::to_vec(&json!({
        "protocol": "tn.llama.request.v1",
        "tn_row": parsed,
        "messages": [{"role": "user", "content": "verify governed output"}],
        "purpose": "verify_governed_output",
        "stream": false
    }))
    .map_err(|_| output_error("tn_output_shape", "TN output verification wrapper failed"))?;
    let verified = parse_and_verify_request(
        &request,
        &RequestLimits {
            max_request_bytes: request.len().saturating_add(1),
            max_message_bytes: 1024,
            expected_device_identity: Some(device_identity.to_string()),
            expected_event_type: event_type.to_string(),
            allowed_input_groups: BTreeSet::from([DEFAULT_GROUP.to_string()]),
            require_signature: true,
            max_active_sessions: 1,
            max_chain_namespaces: 1,
        },
    )?;
    Ok(verified.row)
}

fn validate_report_proposal(
    proposal: &ReportProposal,
    trusted: &TrustedDerivation,
) -> Result<(), TnLlamaError> {
    let valid_identity = proposal.event_type == trusted.contract.output_event_type
        && trusted.contract.output_public_section == "default"
        && !trusted.request_digest.is_empty()
        && !proposal.public_sections.default.report_html.is_empty();
    let valid_policy = proposed_policy(&proposal.governance) == trusted.contract.output_policy;
    let valid_extras = proposal.governance.source_lineage == trusted.source_lineage
        && proposal.governance.compliance_evidence == trusted.compliance_evidence;
    let valid_lineage_order = trusted
        .source_lineage
        .as_ref()
        .is_none_or(|lineage| is_strictly_sorted_unique(&lineage.selected_sections));
    if valid_identity && valid_policy && valid_extras && valid_lineage_order {
        Ok(())
    } else {
        Err(output_error(
            "tn_report_proposal_invalid",
            "report proposal does not match the server-owned output contract and derivation",
        ))
    }
}

fn proposed_policy(governance: &ProposedOutputGovernance) -> OutputPolicy {
    OutputPolicy {
        instruction: governance.instruction.clone(),
        use_for: governance.use_for.clone(),
        do_not_use_for: governance.do_not_use_for.clone(),
        consequences: governance.consequences.clone(),
        on_violation_or_error: governance.on_violation_or_error.clone(),
    }
}

fn verify_reopened_report(
    verified: &VerifiedOutputRow,
    proposal: &ReportProposal,
    trusted: &TrustedDerivation,
) -> Result<(), TnLlamaError> {
    let section = trusted.contract.output_public_section.as_str();
    let default_keys = verified
        .default
        .plaintext
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected_default_keys = BTreeSet::from([section, "run_id"]);
    let expected_default =
        serde_json::to_value(&proposal.public_sections.default).map_err(|_| {
            output_error(
                "tn_output_payload_mismatch",
                "expected report payload could not be encoded for reread comparison",
            )
        })?;
    if verified.default.name != DEFAULT_GROUP
        || default_keys != expected_default_keys
        || verified.default.plaintext.get(section) != Some(&expected_default)
        || verified
            .default
            .plaintext
            .get("run_id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    {
        return Err(output_error(
            "tn_output_payload_mismatch",
            "reopened TN output payload does not exactly match the accepted report proposal",
        ));
    }

    let policy = verified
        .governance
        .plaintext
        .get("policy")
        .cloned()
        .ok_or_else(|| {
            output_error(
                "tn_output_policy_mismatch",
                "reopened TN output governance omitted its computed policy",
            )
        })?;
    let output_policy = &trusted.contract.output_policy;
    let policy_fields = [
        ("instruction", output_policy.instruction.as_str()),
        ("use_for", output_policy.use_for.as_str()),
        ("do_not_use_for", output_policy.do_not_use_for.as_str()),
        ("consequences", output_policy.consequences.as_str()),
        (
            "on_violation_or_error",
            output_policy.on_violation_or_error.as_str(),
        ),
    ];
    if policy_fields.iter().any(|(name, expected)| {
        verified
            .governance
            .plaintext
            .get(*name)
            .and_then(Value::as_str)
            != Some(*expected)
    }) {
        return Err(output_error(
            "tn_output_policy_mismatch",
            "reopened TN output policy does not match the configured report contract",
        ));
    }
    let mut expected_governance = BTreeMap::from([
        (
            "instruction".to_string(),
            Value::String(output_policy.instruction.clone()),
        ),
        (
            "use_for".to_string(),
            Value::String(output_policy.use_for.clone()),
        ),
        (
            "do_not_use_for".to_string(),
            Value::String(output_policy.do_not_use_for.clone()),
        ),
        (
            "consequences".to_string(),
            Value::String(output_policy.consequences.clone()),
        ),
        (
            "on_violation_or_error".to_string(),
            Value::String(output_policy.on_violation_or_error.clone()),
        ),
        ("policy".to_string(), policy),
    ]);
    insert_expected_governance(
        &mut expected_governance,
        "source_lineage",
        trusted.source_lineage.as_ref(),
    )?;
    insert_expected_governance(
        &mut expected_governance,
        "compliance_evidence",
        trusted.compliance_evidence.as_ref(),
    )?;
    if verified.governance.plaintext != expected_governance {
        return Err(output_error(
            "tn_output_governance_mismatch",
            "reopened TN output governance does not exactly match server-held derivation facts",
        ));
    }
    Ok(())
}

fn insert_expected_governance<T: Serialize>(
    expected: &mut BTreeMap<String, Value>,
    name: &'static str,
    value: Option<&T>,
) -> Result<(), TnLlamaError> {
    if let Some(value) = value {
        expected.insert(
            name.to_string(),
            serde_json::to_value(value).map_err(|_| {
                output_error(
                    "tn_output_governance_mismatch",
                    "trusted output governance could not be encoded for reread comparison",
                )
            })?,
        );
    }
    Ok(())
}

fn insert_optional_governance_extra<T: Serialize>(
    fields: &mut Map<String, Value>,
    name: &'static str,
    value: Option<&T>,
) -> Result<(), TnLlamaError> {
    if let Some(value) = value {
        fields.insert(
            name.to_string(),
            serde_json::to_value(value).map_err(|_| {
                output_error(
                    "tn_output_governance_invalid",
                    "trusted output governance could not be encoded",
                )
            })?,
        );
    }
    Ok(())
}

fn is_strictly_sorted_unique(values: &[String]) -> bool {
    let unique = values.iter().collect::<HashSet<_>>().len() == values.len();
    unique && values.windows(2).all(|pair| pair[0] < pair[1])
}

fn output_header<'a>(
    envelope: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a str, TnLlamaError> {
    envelope
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| output_error("tn_output_shape", "TN output row omitted a string header"))
}

fn require_exact_reread(
    emitted: &Value,
    reread: &(tn_core::ReadEntry, tn_core::ValidFlags),
) -> Result<(), TnLlamaError> {
    if !reread.1.signature || !reread.1.row_hash || !reread.1.chain {
        return Err(output_error(
            "tn_output_reread_mismatch",
            "TN lossless reread did not verify output signature, hash, and chain",
        ));
    }
    if reread.0.envelope != *emitted {
        return Err(output_error(
            "tn_output_reread_mismatch",
            "TN lossless reread did not return the exact emitted row",
        ));
    }
    Ok(())
}

fn output_error(code: &'static str, message: &'static str) -> TnLlamaError {
    TnLlamaError::new(code, AdmissionStage::Output, message)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use tn_core::{ReadEntry, ValidFlags};

    use super::require_exact_reread;

    #[test]
    fn strict_reread_rejects_a_different_row_with_the_same_public_identity() {
        let returned = json!({
            "device_identity": "did:key:zOutput",
            "event_id": "evt-same",
            "event_type": "report.generated",
            "row_hash": "hash-returned",
            "default": {"report_html": "returned"}
        });
        let stored = json!({
            "device_identity": "did:key:zOutput",
            "event_id": "evt-same",
            "event_type": "report.generated",
            "row_hash": "hash-different",
            "default": {"report_html": "different"}
        });
        let reread = (
            ReadEntry {
                envelope: stored,
                plaintext_per_group: BTreeMap::new(),
            },
            ValidFlags {
                signature: false,
                row_hash: true,
                chain: true,
                writer_authenticated: false,
                writer_authorized: false,
                reasons: Vec::new(),
            },
        );

        let error = require_exact_reread(&returned, &reread)
            .expect_err("a different stored row must not satisfy strict reread evidence");

        assert_eq!(error.code(), "tn_output_reread_mismatch");
    }
}
