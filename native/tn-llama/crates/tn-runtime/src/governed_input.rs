//! Exact standalone TN publications, admitted under receiver-held authority.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;
use tn_core::governed::{
    AdmissionContext, DataObject, DatasetCatalog, DatasetEdition, GovernanceView, GovernedObject,
    GovernedReader, LineageVerifier, PolicyDag, PolicyRevision, UseContext,
};

use crate::config::RequestLimits;
use crate::envelope::{parse_duplicate_safe_json, UserMessage};
use crate::governance::{GovernanceReader, VerifiedGovernance};
use crate::model_protocol::{digest_bytes, session_error};
use crate::TnLlamaError;

pub const GOVERNED_REQUEST_PROTOCOL: &str = "tn.llama.request.v2";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractPermission {
    pub revision_id: String,
    pub scope: String,
    pub allowed_uses: BTreeSet<UseContext>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GovernedInputConfig {
    #[serde(default)]
    pub report_contract: Option<tn_core::governed::ContractBinding>,
    pub native_reader_config_path: PathBuf,
    pub metadata_store_path: PathBuf,
    pub use_context: UseContext,
    pub destinations: BTreeSet<String>,
    pub trusted_writers: BTreeSet<String>,
    pub authority_writers: BTreeSet<String>,
    pub contracts: Vec<ContractPermission>,
}

#[derive(Debug)]
pub struct AcceptedGovernedRequest {
    pub object: GovernedObject,
    pub use_context: UseContext,
    pub destination: String,
    pub messages: Vec<UserMessage>,
    pub request_digest: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wrapper {
    protocol: String,
    tn_object: String,
    #[serde(rename = "use")]
    use_context: UseContext,
    destination: String,
    messages: Vec<Message>,
    stream: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}

pub fn parse_governed_object_request(
    raw: &[u8],
    limits: &RequestLimits,
    config: &GovernedInputConfig,
) -> Result<AcceptedGovernedRequest, TnLlamaError> {
    if raw.len() > limits.max_request_bytes {
        return Err(refused("tn_request_too_large"));
    }
    let value = parse_duplicate_safe_json(raw).map_err(|_| refused("tn_request_shape"))?;
    let wrapper: Wrapper =
        serde_json::from_value(value).map_err(|_| refused("tn_request_shape"))?;
    if wrapper.protocol != GOVERNED_REQUEST_PROTOCOL
        || wrapper.stream
        || wrapper.messages.len() != 1
        || wrapper.messages[0].role != "user"
        || wrapper.messages[0].content.trim().is_empty()
        || wrapper.messages[0].content.len() > limits.max_message_bytes
    {
        return Err(refused("tn_request_shape"));
    }
    if wrapper.use_context != config.use_context
        || !config.destinations.contains(&wrapper.destination)
    {
        return Err(refused("tn_governed_use_denied"));
    }
    let object = GovernedObject::parse(&wrapper.tn_object)
        .map_err(|_| refused("tn_governed_object_invalid"))?;
    if object.object_type() != limits.expected_event_type
        || !config.trusted_writers.contains(object.writer())
        || object
            .group_names()
            .iter()
            .any(|name| *name != "tn.agents" && !limits.allowed_input_groups.contains(*name))
    {
        return Err(refused("tn_governed_source_denied"));
    }
    let message = wrapper.messages.into_iter().next().expect("one message");
    let digest = tn_core::canonical::canonical_bytes(&json!({
        "object_id":object.id(), "use":wrapper.use_context, "destination":wrapper.destination,
        "messages":[{"role":"user","content":message.content}]
    }))
    .map_err(|_| refused("tn_request_shape"))?;
    Ok(AcceptedGovernedRequest {
        object,
        use_context: wrapper.use_context,
        destination: wrapper.destination,
        messages: vec![UserMessage {
            role: "user".into(),
            content: message.content,
        }],
        request_digest: digest_bytes(b"tn.llama.request.v2\0", &digest),
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataStore {
    policy_revisions: Vec<String>,
    dataset_editions: Vec<String>,
    publications: Vec<String>,
}

pub(crate) struct GovernedInputRuntime {
    pub config: GovernedInputConfig,
    reader: GovernedReader,
    pub verifier: Arc<GovernanceReader>,
    pub report_policy: Option<tn_core::governed::Governance>,
    dag: PolicyDag,
    catalog: DatasetCatalog,
    retained: BTreeMap<String, GovernanceView>,
}

impl GovernedInputRuntime {
    pub fn load(
        config: GovernedInputConfig,
        verifier: Arc<GovernanceReader>,
    ) -> Result<Self, TnLlamaError> {
        if !config.native_reader_config_path.is_absolute()
            || !config.metadata_store_path.is_absolute()
            || config.trusted_writers.is_empty()
            || config.authority_writers.is_empty()
            || config.destinations.is_empty()
            || config.destinations.iter().any(|s| s.trim().is_empty())
            || config.contracts.is_empty()
        {
            return Err(refused("tn_governed_config_invalid"));
        }
        let reader = tn_core::runtime::Objects::open(&config.native_reader_config_path)
            .and_then(|objects| objects.reader())
            .map_err(|_| refused("tn_governed_reader_invalid"))?;
        let raw = std::fs::read(&config.metadata_store_path)
            .map_err(|_| refused("tn_governed_store_invalid"))?;
        let metadata: MetadataStore = serde_json::from_value(
            parse_duplicate_safe_json(&raw).map_err(|_| refused("tn_governed_store_invalid"))?,
        )
        .map_err(|_| refused("tn_governed_store_invalid"))?;
        let mut dag = PolicyDag::new();
        let mut catalog = DatasetCatalog::new();
        let open_metadata = |wire: &str, group: &str| -> Result<_, TnLlamaError> {
            let object =
                GovernedObject::parse(wire).map_err(|_| refused("tn_governed_metadata_invalid"))?;
            if !config.authority_writers.contains(object.writer()) {
                return Err(refused("tn_governed_authority_denied"));
            }
            let view = reader
                .governance(&object)
                .map_err(|_| refused("tn_governed_metadata_invalid"))?;
            verifier.verify_native_group(&object, "tn.agents", view.governance().fields())?;
            let admitted = view
                .accept(
                    UseContext::new("tn-llama.catalog", "administration", "inspect")
                        .expect("constant use"),
                    [group],
                    |ctx| {
                        Ok(ctx
                            .policies()?
                            .iter()
                            .all(|policy| config.authority_writers.contains(policy.governed_by())))
                    },
                )
                .map_err(|_| refused("tn_governed_authority_denied"))?;
            let opened = reader
                .open(&admitted, [group])
                .map_err(|_| refused("tn_governed_metadata_invalid"))?;
            verifier.verify_native_group(
                &object,
                group,
                opened.groups()[group]
                    .as_object()
                    .expect("native group object"),
            )?;
            Ok(opened)
        };
        for wire in &metadata.policy_revisions {
            let revision = PolicyRevision::from_opened(&open_metadata(wire, "policy_revision")?)
                .map_err(|_| refused("tn_governed_metadata_invalid"))?;
            dag.admit(revision, |revision, _| {
                Ok(config.authority_writers.contains(revision.writer())
                    && config
                        .authority_writers
                        .contains(revision.governance().governed_by()))
            })
            .map_err(|_| refused("tn_governed_authority_denied"))?;
        }
        let mut permitted = BTreeSet::new();
        for permission in &config.contracts {
            if !permitted.insert(permission.revision_id.clone())
                || permission.allowed_uses.is_empty()
                || dag
                    .select(&permission.revision_id, &permission.scope, |_| Ok(true))
                    .is_err()
            {
                return Err(refused("tn_governed_config_invalid"));
            }
        }
        for wire in &metadata.dataset_editions {
            let edition = DatasetEdition::from_opened(&open_metadata(wire, "dataset_edition")?)
                .map_err(|_| refused("tn_governed_metadata_invalid"))?;
            catalog
                .admit(edition, &dag, |item| {
                    Ok(config.authority_writers.contains(item.writer()))
                })
                .map_err(|_| refused("tn_governed_authority_denied"))?;
        }
        let mut retained = BTreeMap::new();
        for wire in &metadata.publications {
            let object =
                GovernedObject::parse(wire).map_err(|_| refused("tn_governed_metadata_invalid"))?;
            if !config.trusted_writers.contains(object.writer()) {
                return Err(refused("tn_governed_source_denied"));
            }
            let view = reader
                .governance(&object)
                .map_err(|_| refused("tn_governed_metadata_invalid"))?;
            verifier.verify_native_group(&object, "tn.agents", view.governance().fields())?;
            if retained.insert(object.id().to_owned(), view).is_some() {
                return Err(refused("tn_governed_store_invalid"));
            }
        }
        let report_policy = config
            .report_contract
            .as_ref()
            .map(|binding| {
                if !config.contracts.iter().any(|permission| {
                    permission.revision_id == binding.revision_id()
                        && permission.scope == binding.scope()
                        && permission.allowed_uses.contains(&config.use_context)
                }) {
                    return Err(refused("tn_governed_config_invalid"));
                }
                dag.select(binding.revision_id(), binding.scope(), |_| Ok(true))
                    .map_err(|_| refused("tn_governed_config_invalid"))
            })
            .transpose()?;
        Ok(Self {
            config,
            reader,
            verifier,
            report_policy,
            dag,
            catalog,
            retained,
        })
    }

    fn accepts(
        &self,
        context: &AdmissionContext<'_>,
        view: &GovernanceView,
    ) -> tn_core::Result<bool> {
        if context.use_context() != Some(&self.config.use_context)
            || !self
                .config
                .trusted_writers
                .contains(context.object().writer())
        {
            return Ok(false);
        }
        if !self.accepts_contracts(context)? {
            return Ok(false);
        }
        LineageVerifier::default().verify(view, &self.catalog, &self.dag, |id| {
            self.retained
                .get(id)
                .cloned()
                .ok_or_else(|| tn_core::Error::InvalidConfig("missing retained ancestor".into()))
        })?;
        Ok(true)
    }

    fn accepts_contracts(&self, context: &AdmissionContext<'_>) -> tn_core::Result<bool> {
        if context.use_context() != Some(&self.config.use_context) {
            return Ok(false);
        }
        for policy in context.policies()? {
            let Some(permission) = self
                .config
                .contracts
                .iter()
                .find(|p| Some(p.revision_id.as_str()) == policy.revision_id())
            else {
                return Ok(false);
            };
            if !permission.allowed_uses.contains(&self.config.use_context) {
                return Ok(false);
            }
            self.dag.resolve(&policy, &permission.scope)?;
        }
        if !self.catalog.accepts(context, &self.dag)? {
            return Ok(false);
        }
        Ok(true)
    }

    pub fn accepts_output(
        &self,
        context: &AdmissionContext<'_>,
        view: &GovernanceView,
        parents: &[&GovernanceView],
    ) -> tn_core::Result<bool> {
        if !self.accepts_contracts(context)? {
            return Ok(false);
        }
        let sources = context.sources()?;
        if sources.len() != parents.len()
            || !parents.iter().all(|parent| {
                sources.iter().any(|reference| {
                    reference.references_with_policy(parent.object(), parent.governance())
                })
            })
        {
            return Ok(false);
        }
        LineageVerifier::default().verify(view, &self.catalog, &self.dag, |id| {
            if let Some(parent) = parents.iter().find(|parent| id == parent.object().id()) {
                return Ok((*parent).clone());
            }
            self.retained
                .get(id)
                .cloned()
                .ok_or_else(|| tn_core::Error::InvalidConfig("missing retained ancestor".into()))
        })?;
        Ok(true)
    }

    pub fn governance(
        &self,
        request: &AcceptedGovernedRequest,
        groups: &BTreeSet<String>,
    ) -> Result<(GovernanceView, VerifiedGovernance), TnLlamaError> {
        let view = self
            .reader
            .governance(&request.object)
            .map_err(|_| refused("tn_governed_governance_invalid"))?;
        let checked = self.verifier.verify_native_group(
            &request.object,
            "tn.agents",
            view.governance().fields(),
        )?;
        // Strict group-bound acceptance before any contract content reaches inference.
        view.clone()
            .accept(request.use_context.clone(), groups, |ctx| {
                self.accepts(ctx, &view)
            })
            .map_err(|_| refused("tn_governed_use_denied"))?;
        Ok((
            view,
            VerifiedGovernance {
                plaintext: checked.plaintext,
                commitments: checked.commitments,
            },
        ))
    }

    pub fn open(
        &self,
        request: &AcceptedGovernedRequest,
        view: &GovernanceView,
        groups: &BTreeSet<String>,
    ) -> Result<DataObject, TnLlamaError> {
        let admitted = view
            .clone()
            .accept(request.use_context.clone(), groups, |ctx| {
                self.accepts(ctx, view)
            })
            .map_err(|_| refused("tn_governed_use_denied"))?;
        let opened = self
            .reader
            .open(&admitted, groups)
            .map_err(|_| refused("tn_governed_group_open_failed"))?;
        for (name, value) in opened.groups() {
            self.verifier.verify_native_group(
                &request.object,
                name,
                value.as_object().expect("native group object"),
            )?;
        }
        DataObject::from_opened(opened).map_err(|_| refused("tn_governed_group_open_failed"))
    }
}

pub(crate) fn refused(code: &'static str) -> TnLlamaError {
    session_error(
        code,
        "governed object request does not satisfy the receiver's configured profile",
    )
}
