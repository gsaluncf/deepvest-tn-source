use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::envelope::parse_duplicate_safe_json;
use crate::error::{AdmissionStage, TnLlamaError};
use crate::output_tn::{ComplianceReceipt, OutputPublisher};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisteredCall {
    pub id: String,
    pub name: String,
    pub arguments: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolContext {
    pub run_id: String,
    pub source_device_identity: String,
    pub source_event_id: String,
    pub source_event_type: String,
    pub source_row_hash: String,
    pub selected_sections: BTreeSet<String>,
    pub action: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A dispatcher-validated compliance invocation.
///
/// Downstream callers cannot construct one or bypass the registered handler:
///
/// ```compile_fail
/// use tn_runtime::{ComplianceInvocation, OutputPublisher};
///
/// fn bypass(publisher: &OutputPublisher) {
///     let invocation = ComplianceInvocation {
///         run_id: "untrusted-run".to_string(),
///         tool_name: "unregistered".to_string(),
///         model_tool_call_id: "untrusted-call".to_string(),
///         source_device_identity: "did:key:untrusted".to_string(),
///         source_event_id: "untrusted-event".to_string(),
///         source_event_type: "untrusted.type".to_string(),
///         source_row_hash: "sha256:untrusted".to_string(),
///         selected_sections: vec!["untrusted".to_string()],
///         action: "untrusted-action".to_string(),
///     };
///     let _ = publisher.emit_compliance(&invocation);
/// }
/// ```
pub struct ComplianceInvocation {
    pub(crate) run_id: String,
    pub(crate) tool_name: String,
    pub(crate) model_tool_call_id: String,
    pub(crate) source_device_identity: String,
    pub(crate) source_event_id: String,
    pub(crate) source_event_type: String,
    pub(crate) source_row_hash: String,
    pub(crate) selected_sections: Vec<String>,
    pub(crate) action: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolReceipt {
    Compliance(ComplianceReceipt),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisteredToolSpec {
    name: String,
    description: String,
    parameters: Value,
}

impl RegisteredToolSpec {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    #[must_use]
    pub fn parameters(&self) -> &Value {
        &self.parameters
    }

    pub(crate) fn model_tool(&self, source_event_id: &str, action: &str) -> Value {
        let mut parameters = self.parameters.clone();
        parameters["properties"]["source_event_id"]["const"] =
            Value::String(source_event_id.to_string());
        parameters["properties"]["action"]["const"] = Value::String(action.to_string());
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            }
        })
    }
}

impl RegisteredToolSpec {
    pub(crate) fn native_model_tool(&self, source_object_id: &str, action: &str) -> Value {
        let mut parameters = self.parameters.clone();
        parameters["properties"]["source_object_id"]["const"] = json!(source_object_id);
        parameters["properties"]["action"]["const"] = json!(action);
        parameters["properties"]["source_object_id"]["description"] = json!(format!(
            "Current admitted source object. Copy this exact value: {source_object_id}"
        ));
        parameters["properties"]["action"]["description"] = json!(format!(
            "Runtime-authorized action. Copy this exact value: {action}"
        ));
        json!({"type":"function", "function":{"name":self.name, "description":self.description, "parameters":parameters}})
    }
}

pub struct ToolDispatcher {
    handlers: BTreeMap<String, Arc<dyn RegisteredToolHandler>>,
    specs: BTreeMap<String, RegisteredToolSpec>,
    native_publisher: Option<Arc<crate::output_tn::GovernedOutputPublisher>>,
}

/// Session-owned context: no caller or model can construct an accepted native invocation.
pub(crate) struct NativeToolContext<'a> {
    pub run_id: &'a str,
    pub data: &'a tn_core::governed::DataObject,
    pub request: &'a crate::AcceptedGovernedRequest,
    pub source: &'a tn_core::governed::GovernanceView,
    pub input: &'a crate::governed_input::GovernedInputRuntime,
    pub selected_sections: &'a BTreeSet<String>,
    pub action: &'a str,
}

pub(crate) struct NativeToolReceipt {
    pub publication: crate::output_tn::GovernedPublication,
    pub receipt: Value,
}

trait RegisteredToolHandler: Send + Sync {
    fn execute(
        &self,
        call: &RegisteredCall,
        context: &ToolContext,
    ) -> Result<ToolReceipt, TnLlamaError>;
}

struct ComplianceCreationTool {
    publisher: Arc<OutputPublisher>,
}

impl ToolDispatcher {
    pub fn new(
        publisher: Arc<OutputPublisher>,
        configured_name: impl Into<String>,
    ) -> Result<Self, TnLlamaError> {
        let configured_name = configured_name.into();
        if configured_name.trim().is_empty() {
            return Err(tool_error(
                "tn_tool_registration_invalid",
                "configured governed tool name must be nonempty",
            ));
        }
        let handler: Arc<dyn RegisteredToolHandler> =
            Arc::new(ComplianceCreationTool { publisher });
        let spec = RegisteredToolSpec {
            name: configured_name.clone(),
            description:
                "Log creation of a governed derived report and return its canonical TN receipt."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "source_event_id": {"type": "string"},
                    "action": {"type": "string"}
                },
                "required": ["source_event_id", "action"],
                "additionalProperties": false
            }),
        };
        Ok(Self {
            handlers: BTreeMap::from([(configured_name.clone(), handler)]),
            specs: BTreeMap::from([(configured_name, spec)]),
            native_publisher: None,
        })
    }

    pub(crate) fn empty() -> Self {
        Self {
            handlers: BTreeMap::new(),
            specs: BTreeMap::new(),
            native_publisher: None,
        }
    }

    pub(crate) fn native(
        publisher: Arc<crate::output_tn::GovernedOutputPublisher>,
        configured_name: &str,
    ) -> Result<Self, TnLlamaError> {
        if configured_name.trim().is_empty() || configured_name == "emit_governed_record" {
            return Err(tool_error(
                "tn_tool_registration_invalid",
                "protected tool name conflicts with the governed model protocol",
            ));
        }
        let spec = RegisteredToolSpec {
            name: configured_name.into(),
            description: "Record governed report creation through the native TN publisher and return its authenticated receipt.".into(),
            parameters: json!({"type":"object", "properties":{
                "source_object_id":{"type":"string"}, "action":{"type":"string"}
            }, "required":["source_object_id","action"], "additionalProperties":false}),
        };
        Ok(Self {
            handlers: BTreeMap::new(),
            specs: BTreeMap::from([(configured_name.into(), spec)]),
            native_publisher: Some(publisher),
        })
    }

    pub(crate) fn execute_native(
        &self,
        call: &RegisteredCall,
        context: &NativeToolContext<'_>,
    ) -> Result<NativeToolReceipt, TnLlamaError> {
        let publisher = self
            .native_publisher
            .as_ref()
            .filter(|_| self.specs.contains_key(&call.name))
            .ok_or_else(|| {
                tool_error(
                    "tn_tool_not_registered",
                    "native protected tool is not registered",
                )
            })?;
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Arguments {
            source_object_id: String,
            action: String,
        }
        let value = parse_duplicate_safe_json(&call.arguments).map_err(|_| arguments_error())?;
        let arguments: Arguments = serde_json::from_value(value).map_err(|_| arguments_error())?;
        if call.id.trim().is_empty()
            || arguments.source_object_id != context.request.object.id()
            || arguments.action != context.action
            || context.action != context.request.use_context.operation()
            || context.selected_sections.is_empty()
        {
            return Err(arguments_error());
        }
        let content = json!({
            "run_id":context.run_id, "request_digest":context.request.request_digest,
            "tool_name":call.name, "model_tool_call_id":call.id,
            "source_object_id":context.request.object.id(),
            "source_writer":context.request.object.writer(),
            "source_type":context.request.object.object_type(),
            "selected_sections":context.selected_sections, "action":context.action,
        });
        let publication = publisher.publish(
            context.data.clone(),
            content.clone(),
            "compliance.creation.logged",
            context.request,
            &[context.source],
            context.input,
            false,
        )?;
        let mut receipt = content;
        receipt["output_object_id"] = json!(publication.object.id());
        receipt["output_object_type"] = json!(publication.object.object_type());
        Ok(NativeToolReceipt {
            publication,
            receipt,
        })
    }

    #[must_use]
    pub fn specification(&self, name: &str) -> Option<&RegisteredToolSpec> {
        self.specs.get(name)
    }

    pub fn specifications(&self) -> impl Iterator<Item = &RegisteredToolSpec> {
        self.specs.values()
    }

    pub fn execute(
        &self,
        call: &RegisteredCall,
        context: &ToolContext,
    ) -> Result<ToolReceipt, TnLlamaError> {
        self.handlers
            .get(&call.name)
            .ok_or_else(|| {
                tool_error(
                    "tn_tool_not_registered",
                    "model requested an unregistered governed tool",
                )
            })?
            .execute(call, context)
    }
}

impl RegisteredToolHandler for ComplianceCreationTool {
    fn execute(
        &self,
        call: &RegisteredCall,
        context: &ToolContext,
    ) -> Result<ToolReceipt, TnLlamaError> {
        let invocation = ComplianceInvocation::validate(call, context)?;
        self.publisher
            .emit_compliance(&invocation)
            .map(ToolReceipt::Compliance)
    }
}

impl ComplianceInvocation {
    pub(crate) fn validate(
        call: &RegisteredCall,
        context: &ToolContext,
    ) -> Result<Self, TnLlamaError> {
        let value = parse_duplicate_safe_json(&call.arguments).map_err(|_| arguments_error())?;
        let arguments: ComplianceArguments =
            serde_json::from_value(value).map_err(|_| arguments_error())?;
        let context_strings = [
            context.run_id.as_str(),
            context.source_device_identity.as_str(),
            context.source_event_id.as_str(),
            context.source_event_type.as_str(),
            context.source_row_hash.as_str(),
            context.action.as_str(),
        ];
        if call.id.trim().is_empty()
            || call.name.trim().is_empty()
            || context_strings.iter().any(|value| value.trim().is_empty())
            || arguments.source_event_id != context.source_event_id
            || arguments.action != context.action
        {
            return Err(arguments_error());
        }

        Ok(Self {
            run_id: context.run_id.clone(),
            tool_name: call.name.clone(),
            model_tool_call_id: call.id.clone(),
            source_device_identity: context.source_device_identity.clone(),
            source_event_id: context.source_event_id.clone(),
            source_event_type: context.source_event_type.clone(),
            source_row_hash: context.source_row_hash.clone(),
            selected_sections: context.selected_sections.iter().cloned().collect(),
            action: context.action.clone(),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComplianceArguments {
    source_event_id: String,
    action: String,
}

fn arguments_error() -> TnLlamaError {
    tool_error(
        "tn_tool_arguments_invalid",
        "governed tool arguments do not exactly match the trusted source and action",
    )
}

fn tool_error(code: &'static str, message: &'static str) -> TnLlamaError {
    TnLlamaError::new(code, AdmissionStage::Tool, message)
}
