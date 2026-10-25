use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::contract::OutputPolicy;
use crate::error::{AdmissionStage, TnLlamaError};
use crate::model_phase::Phase;
use crate::output_tn::{ComplianceReceipt, SourceLineage};

pub const MODEL_REQUEST_DIGEST_DOMAIN: &[u8] = b"tn.llama.model-request.v1\0";
pub const RENDERED_PROMPT_DIGEST_DOMAIN: &[u8] = b"tn.llama.rendered-prompt.v1\0";
pub const RUNNABLE_TASK_DIGEST_DOMAIN: &[u8] = b"tn.llama.runnable-task.v1\0";
pub const RUNNABLE_TASK_SCHEMA: &str = "tn.llama.runnable-task.v1";
pub(crate) const GOVERNANCE_READ_MAX_TOKENS: u32 = 256;

pub(crate) const SYSTEM_CONTENT: &str = "This is a server-governed action interaction. Call exactly one available tool and return no prose. Copy server-provided trusted fields exactly. Generate only the requested output fields. The server validates, executes, emits, and releases every action.";

pub struct ParsedToolCall<'a> {
    pub id: &'a [u8],
    pub name: &'a [u8],
    pub parsed_arguments: &'a [u8],
}

pub struct NativeToolResult<'a> {
    pub call_count: usize,
    pub call: Option<ParsedToolCall<'a>>,
    pub content: &'a [u8],
    pub raw_model_output: &'a [u8],
    pub truncated: bool,
    pub stop_limit: bool,
}

pub(crate) fn governance_read_body(
    purpose: &str,
    row_hash: &str,
    request_digest: &str,
) -> Result<Vec<u8>, TnLlamaError> {
    let control = canonical_text(&json!({
        "phase": "governance_read",
        "purpose": purpose,
        "request_digest": request_digest,
        "row_hash": row_hash,
    }))?;
    canonical_value(&json!({
        "cache_prompt": false,
        "max_tokens": GOVERNANCE_READ_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": control}
        ],
        "n": 1,
        "n_cache_reuse": 0,
        "parallel_tool_calls": false,
        "reasoning_effort": "none",
        "stream": false,
        "temperature": 0,
        "tool_choice": "required",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_governance",
                "description": "Request the governance bound to this admitted row and request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "request_digest": {"type": "string"},
                        "row_hash": {"type": "string"}
                    },
                    "required": ["request_digest", "row_hash"],
                    "additionalProperties": false
                }
            }
        }]
    }))
}

#[derive(Clone, Copy)]
pub(crate) struct AcceptedCall<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub arguments: &'a str,
}

pub(crate) fn commitment_echo_body(
    purpose: &str,
    row_hash: &str,
    request_digest: &str,
    governance: &BTreeMap<String, Value>,
    commitments: &BTreeMap<String, String>,
    first_call: AcceptedCall<'_>,
    max_tokens: u32,
) -> Result<Vec<u8>, TnLlamaError> {
    let control = canonical_text(&json!({
        "phase": "governance_read",
        "purpose": purpose,
        "request_digest": request_digest,
        "row_hash": row_hash,
    }))?;
    let result = canonical_text(&json!({
        "governance": governance,
        "governance_commitments": commitments,
        "request_digest": request_digest,
        "row_hash": row_hash,
    }))?;
    canonical_value(&json!({
        "cache_prompt": false,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": control},
            assistant_call_message(first_call),
            {"role": "tool", "tool_call_id": first_call.id, "content": result}
        ],
        "n": 1,
        "n_cache_reuse": 0,
        "parallel_tool_calls": false,
        "reasoning_effort": "none",
        "stream": false,
        "temperature": 0,
        "tool_choice": "required",
        "tools": [{
            "type": "function",
            "function": {
                "name": "continue_with_governance",
                "description": "Continue only by echoing the complete governance and commitment set exactly.",
                "parameters": echo_parameters_schema(governance)
            }
        }]
    }))
}

pub(crate) fn section_selection_body(
    user_content: &str,
    available_sections: &BTreeMap<String, Vec<String>>,
    max_tokens: u32,
) -> Result<Vec<u8>, TnLlamaError> {
    let selection = canonical_text(&json!({
        "phase": "section_selection",
        "user_request": user_content,
        "available_sections": available_sections,
    }))?;
    let section_names = available_sections.keys().cloned().collect::<Vec<_>>();
    canonical_value(&json!({
        "cache_prompt": false,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": selection}
        ],
        "n": 1,
        "n_cache_reuse": 0,
        "parallel_tool_calls": false,
        "reasoning_effort": "none",
        "stream": false,
        "temperature": 0,
        "tool_choice": "required",
        "tools": [{
            "type": "function",
            "function": {
                "name": "select_sections",
                "description": "Select every available input section needed to answer the held user request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sections": {
                            "type": "array",
                            "items": {"type": "string", "enum": section_names},
                            "minItems": 1,
                            "uniqueItems": true
                        }
                    },
                    "required": ["sections"],
                    "additionalProperties": false
                }
            }
        }]
    }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn governed_object_action_body(
    user_content: &str,
    selected: &BTreeMap<String, Value>,
    governance: &BTreeMap<String, Value>,
    report_policy: Option<&tn_core::governed::Governance>,
    attributions: &[String],
    source_object_id: &str,
    action: &str,
    tools: Vec<Value>,
    tool_exchanges: &[ToolExchange<'_>],
    max_tokens: u32,
) -> Result<Vec<u8>, TnLlamaError> {
    let trusted_source_binding = json!({"source_object_id":source_object_id,"action":action});
    let required_tool_calls = tools
        .iter()
        .filter(|tool| tool["function"]["name"] != "emit_governed_record")
        .map(|tool| json!({"name":tool["function"]["name"],"arguments":trusted_source_binding}))
        .collect::<Vec<_>>();
    let mut messages = vec![
        json!({"role":"system","content":SYSTEM_CONTENT}),
        json!({"role":"user","content":canonical_text(&json!({
        "phase":"governed_action", "user_request":user_content, "selected_values":selected,
        "applicable_source_governance":governance, "required_attributions":attributions,
        "trusted_source_binding":trusted_source_binding,
        "required_tool_calls":required_tool_calls,
        "additional_report_contract": report_policy.map(|policy| json!({"governed_by":policy.governed_by(),"fields":policy.fields()})),
        "output_instruction":"Execute each required_tool_calls entry using its name and arguments copied exactly. trusted_source_binding identifies the current admitted input and authorized action; IDs in source_lineage and calculation input fields identify earlier ancestors. When emission is offered, write report_html only and use the authenticated tool results when relevant. The runtime retains and signs every inherited contract, dataset binding, exact input source, and accepted tool publication."
    }))?}),
    ];
    for exchange in tool_exchanges {
        messages.push(assistant_call_message(exchange.call));
        messages.push(
            json!({"role":"tool", "tool_call_id":exchange.call.id, "content":exchange.result}),
        );
    }
    canonical_value(&json!({
        "cache_prompt":false, "max_tokens":max_tokens, "n":1, "n_cache_reuse":0,
        "parallel_tool_calls":false, "reasoning_effort":"none", "stream":false, "temperature":0, "tool_choice":"required",
        "messages":messages,
        "tools":tools
    }))
}

pub(crate) fn native_report_tool() -> Value {
    json!({"type":"function","function":{"name":"emit_governed_record",
        "description":"Submit report content for native governed release.",
        "parameters":{"type":"object","properties":{"report_html":{"type":"string","minLength":1}},
            "required":["report_html"],"additionalProperties":false}}})
}

pub(crate) struct ToolExchange<'a> {
    pub call: AcceptedCall<'a>,
    pub result: &'a str,
}

pub(crate) struct GovernedActionBodyInput<'a> {
    pub user_content: &'a str,
    pub available_sections: &'a BTreeMap<String, Vec<String>>,
    pub selected_values: &'a BTreeMap<String, Value>,
    pub selected_sections: &'a [String],
    pub applicable_governance: Option<&'a BTreeMap<String, Value>>,
    pub output_event_type: &'a str,
    pub output_public_section: &'a str,
    pub operation: &'a str,
    pub output_policy: &'a OutputPolicy,
    pub required_attributions: &'a [String],
    pub source_lineage: Option<&'a SourceLineage>,
    pub selection_call: AcceptedCall<'a>,
    pub tool_exchanges: &'a [ToolExchange<'a>],
    pub tools: Vec<Value>,
    pub max_tokens: u32,
}

pub(crate) fn governed_action_body(
    input: GovernedActionBodyInput<'_>,
) -> Result<Vec<u8>, TnLlamaError> {
    let selection_request = canonical_text(&json!({
        "phase": "section_selection",
        "user_request": input.user_content,
        "available_sections": input.available_sections,
    }))?;
    let selection_result = canonical_text(&json!({
        "accepted": true,
        "selected_sections": input.selected_sections,
    }))?;
    let mut governed_context = Map::new();
    governed_context.insert("phase".to_string(), json!("governed_action"));
    governed_context.insert("user_request".to_string(), json!(input.user_content));
    governed_context.insert("selected_values".to_string(), json!(input.selected_values));
    governed_context.insert("operation".to_string(), json!(input.operation));
    governed_context.insert(
        "output_contract".to_string(),
        json!({
            "event_type": input.output_event_type,
            "public_section": input.output_public_section,
            "policy": {
                "instruction": input.output_policy.instruction,
                "use_for": input.output_policy.use_for,
                "do_not_use_for": input.output_policy.do_not_use_for,
                "consequences": input.output_policy.consequences,
                "on_violation_or_error": input.output_policy.on_violation_or_error,
            },
            "required_attributions": input.required_attributions,
        }),
    );
    if let Some(governance) = input.applicable_governance {
        governed_context.insert(
            "applicable_source_governance".to_string(),
            json!(governance),
        );
    }
    if let Some(lineage) = input.source_lineage {
        governed_context.insert("required_source_lineage".to_string(), json!(lineage));
    }
    let governed_context = canonical_text(&Value::Object(governed_context))?;

    let mut messages = vec![
        json!({"role": "system", "content": SYSTEM_CONTENT}),
        json!({"role": "user", "content": selection_request}),
        assistant_call_message(input.selection_call),
        json!({
            "role": "tool",
            "tool_call_id": input.selection_call.id,
            "content": selection_result
        }),
        json!({"role": "user", "content": governed_context}),
    ];
    for exchange in input.tool_exchanges {
        messages.push(assistant_call_message(exchange.call));
        messages.push(json!({
            "role": "tool",
            "tool_call_id": exchange.call.id,
            "content": exchange.result,
        }));
    }

    canonical_value(&json!({
        "cache_prompt": false,
        "max_tokens": input.max_tokens,
        "messages": messages,
        "n": 1,
        "n_cache_reuse": 0,
        "parallel_tool_calls": false,
        "reasoning_effort": "none",
        "stream": false,
        "temperature": 0,
        "tool_choice": "required",
        "tools": input.tools,
    }))
}

pub(crate) fn emit_governed_record_tool(
    output_event_type: &str,
    output_public_section: &str,
    output_policy: &OutputPolicy,
    source_lineage: Option<&SourceLineage>,
    compliance_evidence: Option<&ComplianceReceipt>,
) -> Value {
    let mut governance_properties = Map::from_iter([
        (
            "instruction".to_string(),
            json!({"type": "string", "const": &output_policy.instruction}),
        ),
        (
            "use_for".to_string(),
            json!({"type": "string", "const": &output_policy.use_for}),
        ),
        (
            "do_not_use_for".to_string(),
            json!({"type": "string", "const": &output_policy.do_not_use_for}),
        ),
        (
            "consequences".to_string(),
            json!({"type": "string", "const": &output_policy.consequences}),
        ),
        (
            "on_violation_or_error".to_string(),
            json!({"type": "string", "const": &output_policy.on_violation_or_error}),
        ),
    ]);
    let mut governance_required = vec![
        "instruction",
        "use_for",
        "do_not_use_for",
        "consequences",
        "on_violation_or_error",
    ];
    if let Some(lineage) = source_lineage {
        governance_properties.insert("source_lineage".to_string(), source_lineage_schema(lineage));
        governance_required.push("source_lineage");
    }
    if let Some(evidence) = compliance_evidence {
        governance_properties.insert(
            "compliance_evidence".to_string(),
            compliance_evidence_schema(evidence),
        );
        governance_required.push("compliance_evidence");
    }
    let public_section_properties = Map::from_iter([(
        output_public_section.to_string(),
        json!({
            "type": "object",
            "properties": {"report_html": {"type": "string"}},
            "required": ["report_html"],
            "additionalProperties": false
        }),
    )]);
    json!({
        "type": "function",
        "function": {
            "name": "emit_governed_record",
            "description": "Propose the configured governed record. The server owns all TN headers, policy references, commitments, hashes, and emission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {"type": "string", "const": output_event_type},
                    "public_sections": {
                        "type": "object",
                        "properties": public_section_properties,
                        "required": [output_public_section],
                        "additionalProperties": false
                    },
                    "governance": {
                        "type": "object",
                        "properties": governance_properties,
                        "required": governance_required,
                        "additionalProperties": false
                    }
                },
                "required": ["event_type", "public_sections", "governance"],
                "additionalProperties": false
            }
        }
    })
}

fn source_lineage_schema(lineage: &SourceLineage) -> Value {
    json!({
        "type": "object",
        "properties": {
            "source_device_identity": {"type": "string", "const": &lineage.source_device_identity},
            "source_event_id": {"type": "string", "const": &lineage.source_event_id},
            "source_event_type": {"type": "string", "const": &lineage.source_event_type},
            "derived_from_row_hash": {"type": "string", "const": &lineage.derived_from_row_hash},
            "source_policy": {"type": "string", "const": &lineage.source_policy},
            "selected_sections": {
                "type": "array",
                "items": {"type": "string"},
                "const": &lineage.selected_sections
            },
            "operation": {"type": "string", "const": &lineage.operation}
        },
        "required": [
            "source_device_identity",
            "source_event_id",
            "source_event_type",
            "derived_from_row_hash",
            "source_policy",
            "selected_sections",
            "operation"
        ],
        "additionalProperties": false,
    })
}

fn compliance_evidence_schema(evidence: &ComplianceReceipt) -> Value {
    json!({
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "const": &evidence.event_id},
            "event_type": {"type": "string", "const": &evidence.event_type},
            "row_hash": {"type": "string", "const": &evidence.row_hash},
            "source_row_hash": {"type": "string", "const": &evidence.source_row_hash},
            "action": {"type": "string", "const": &evidence.action}
        },
        "required": ["event_id", "event_type", "row_hash", "source_row_hash", "action"],
        "additionalProperties": false,
    })
}

fn assistant_call_message(call: AcceptedCall<'_>) -> Value {
    json!({
        "role": "assistant",
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments}
        }]
    })
}

fn echo_parameters_schema(governance: &BTreeMap<String, Value>) -> Value {
    let mut governance_properties = Map::new();
    let mut commitment_properties = Map::new();
    for (field, value) in governance {
        governance_properties.insert(field.clone(), value_type_schema(value));
        commitment_properties.insert(field.clone(), json!({"type": "string"}));
    }
    let fields = governance.keys().cloned().collect::<Vec<_>>();
    json!({
        "type": "object",
        "properties": {
            "governance": {
                "type": "object",
                "properties": governance_properties,
                "required": &fields,
                "additionalProperties": false
            },
            "governance_commitments": {
                "type": "object",
                "properties": commitment_properties,
                "required": &fields,
                "additionalProperties": false
            },
            "request_digest": {"type": "string"},
            "row_hash": {"type": "string"}
        },
        "required": ["governance", "governance_commitments", "request_digest", "row_hash"],
        "additionalProperties": false
    })
}

fn value_type_schema(value: &Value) -> Value {
    // The echo copies structure as well as scalar values. Describe nested
    // governance records without embedding their values in the decoder grammar.
    // Rust's canonical comparison remains the authority for the actual contents.
    if let Value::Object(fields) = value {
        let properties = fields
            .iter()
            .map(|(key, value)| (key.clone(), value_type_schema(value)))
            .collect::<Map<_, _>>();
        let required = fields.keys().cloned().collect::<Vec<_>>();
        return json!({"type":"object","properties":properties,"required":required,"additionalProperties":false});
    }
    if let Value::Array(values) = value {
        let mut shapes = Vec::new();
        for value in values {
            let shape = value_type_schema(value);
            if !shapes.contains(&shape) {
                shapes.push(shape);
            }
        }
        let items = match shapes.len() {
            0 => json!({}),
            1 => shapes.remove(0),
            _ => json!({"anyOf":shapes}),
        };
        return json!({"type":"array","items":items,"minItems":values.len(),"maxItems":values.len()});
    }
    let value_type = match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) | Value::Object(_) => unreachable!("containers handled above"),
    };
    json!({"type": value_type})
}

pub(crate) fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

pub(crate) fn model_request_digest(body: &[u8]) -> String {
    digest_bytes(MODEL_REQUEST_DIGEST_DOMAIN, body)
}

pub(crate) fn rendered_prompt_digest(prompt: &[u8]) -> String {
    digest_bytes(RENDERED_PROMPT_DIGEST_DOMAIN, prompt)
}

pub(crate) fn parse_and_digest_manifest(
    raw: &[u8],
    phase: Phase,
    expected_n_predict: u32,
) -> Result<String, TnLlamaError> {
    let value =
        crate::envelope::parse_duplicate_safe_json(raw).map_err(|_| runnable_manifest_error())?;
    let manifest: RunnableManifest =
        serde_json::from_value(value.clone()).map_err(|_| runnable_manifest_error())?;
    manifest.validate(phase, expected_n_predict)?;
    let canonical = tn_core::canonical::canonical_bytes(&value).map_err(|_| {
        session_error(
            "tn_runnable_manifest_invalid",
            "runnable manifest cannot be canonically serialized",
        )
    })?;
    Ok(digest_bytes(RUNNABLE_TASK_DIGEST_DOMAIN, &canonical))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnableManifest {
    schema: String,
    effective_ctx_shift: bool,
    task: RunnableTask,
    params: RunnableParams,
    sampling_native: SamplingNative,
    speculative: Speculative,
    chat_parser: ChatParser,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnableTask {
    #[serde(rename = "type")]
    task_type: String,
    tokens: Vec<i64>,
    tokens_has_mtmd: bool,
    id_slot: i64,
    index: u64,
    id_parent: i64,
    child_count: u64,
    cli: bool,
    cli_prompt_byte_length: u64,
    cli_file_count: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnableParams {
    stream: bool,
    cache_prompt: bool,
    n_cache_reuse: i64,
    n_cmpl: i64,
    n_predict: i64,
    n_keep: i64,
    n_discard: i64,
    n_indent: i64,
    t_max_predict_ms: i64,
    lora: Vec<Value>,
    stop: Vec<String>,
    response_fields: Vec<String>,
    post_sampling_probs: bool,
    res_type: String,
    upstream_params_json: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingNative {
    n_prev: i64,
    logit_bias_eog: Vec<Value>,
    grammar_type: String,
    sampling_generation_prompt: String,
    reasoning_budget_tokens: i64,
    reasoning_budget_start: Vec<i64>,
    reasoning_budget_end: Vec<Vec<i64>>,
    reasoning_budget_forced: Vec<i64>,
    reasoning_budget_message: String,
    reasoning_control: bool,
    backend_sampling: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Speculative {
    types: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatParser {
    format: String,
    reasoning_format: String,
    reasoning_in_content: bool,
    generation_prompt: String,
    parse_tool_calls: bool,
    is_continuation: bool,
    echo: bool,
    debug: bool,
}

impl RunnableManifest {
    fn validate(&self, _phase: Phase, expected_n_predict: u32) -> Result<(), TnLlamaError> {
        let _recorded_but_not_restricted = (
            self.task.index,
            self.params.n_keep,
            self.params.n_discard,
            self.params.n_indent,
            self.params.t_max_predict_ms,
            &self.params.stop,
            &self.params.response_fields,
            self.params.post_sampling_probs,
            &self.params.upstream_params_json,
            self.sampling_native.n_prev,
            &self.sampling_native.logit_bias_eog,
            &self.sampling_native.grammar_type,
            &self.sampling_native.sampling_generation_prompt,
            self.sampling_native.reasoning_budget_tokens,
            &self.sampling_native.reasoning_budget_start,
            &self.sampling_native.reasoning_budget_end,
            &self.sampling_native.reasoning_budget_forced,
            &self.sampling_native.reasoning_budget_message,
            self.sampling_native.reasoning_control,
            self.sampling_native.backend_sampling,
            &self.chat_parser.format,
            &self.chat_parser.reasoning_format,
            self.chat_parser.reasoning_in_content,
            &self.chat_parser.generation_prompt,
            self.chat_parser.parse_tool_calls,
            self.chat_parser.is_continuation,
            self.chat_parser.echo,
            self.chat_parser.debug,
        );
        let valid = self.schema == RUNNABLE_TASK_SCHEMA
            && !self.effective_ctx_shift
            && self.task.task_type == "completion"
            && !self.task.tokens.is_empty()
            && !self.task.tokens_has_mtmd
            && self.task.id_slot == -1
            && self.task.id_parent == -1
            && self.task.child_count == 0
            && !self.task.cli
            && self.task.cli_prompt_byte_length == 0
            && self.task.cli_file_count == 0
            && !self.params.stream
            && !self.params.cache_prompt
            && self.params.n_cache_reuse == 0
            && self.params.n_cmpl == 1
            && self.params.n_predict == i64::from(expected_n_predict)
            && self.params.lora.is_empty()
            && self.params.res_type == "oai_chat"
            && self.speculative.types == ["none"];
        if valid {
            Ok(())
        } else {
            Err(runnable_manifest_error())
        }
    }
}

fn canonical_value(value: &Value) -> Result<Vec<u8>, TnLlamaError> {
    tn_core::canonical::canonical_bytes(value).map_err(|_| {
        session_error(
            "tn_session_state",
            "model request could not be canonically serialized",
        )
    })
}

fn canonical_text(value: &Value) -> Result<String, TnLlamaError> {
    String::from_utf8(canonical_value(value)?).map_err(|_| {
        session_error(
            "tn_session_state",
            "canonical model control data is not UTF-8",
        )
    })
}

fn runnable_manifest_error() -> TnLlamaError {
    session_error(
        "tn_runnable_manifest_invalid",
        "runnable manifest does not match the closed version-1 task projection",
    )
}

pub(crate) fn session_error(code: &'static str, message: &'static str) -> TnLlamaError {
    TnLlamaError::new(code, AdmissionStage::Session, message)
}
