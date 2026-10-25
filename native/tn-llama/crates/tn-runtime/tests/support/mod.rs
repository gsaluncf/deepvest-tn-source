#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;

use serde_json::{json, Value};
use tn_runtime::{
    parse_and_verify_request, ActionOutcome, GovernedRuntime, ModelTurn, NativeToolResult,
    ParsedToolCall, RequestLimits, RuntimeConfig, Session,
};
use tn_test_tools::{build_real_tn_fixture, RealTnFixture};

pub struct SessionHarness {
    pub _fixture: RealTnFixture,
    pub runtime: GovernedRuntime,
    pub session: Session,
    pub raw_request: Vec<u8>,
    pub row_hash: String,
    pub request_digest: String,
    pub governance: BTreeMap<String, Value>,
    pub commitments: BTreeMap<String, String>,
}

impl SessionHarness {
    pub fn new(user_content: &str) -> Self {
        let fixture = build_real_tn_fixture().unwrap();
        let runtime = GovernedRuntime::from_config(fixture_config(&fixture)).unwrap();
        Self::from_fixture_and_runtime(fixture, runtime, user_content)
    }

    pub fn from_fixture_and_runtime(
        fixture: RealTnFixture,
        runtime: GovernedRuntime,
        user_content: &str,
    ) -> Self {
        let mut request: Value = serde_json::from_slice(&fixture.row1_request_json).unwrap();
        request["messages"][0]["content"] = Value::String(user_content.to_string());
        let raw_request = serde_json::to_vec(&request).unwrap();
        let config = fixture_config(&fixture);
        let accepted = parse_and_verify_request(
            &raw_request,
            &RequestLimits {
                max_request_bytes: config.max_request_bytes,
                max_message_bytes: config.max_message_bytes,
                expected_device_identity: Some(fixture.device_identity.clone()),
                expected_event_type: config.expected_event_type.clone(),
                allowed_input_groups: config.allowed_input_groups.clone(),
                require_signature: true,
                max_active_sessions: config.max_active_sessions,
                max_chain_namespaces: config.max_chain_namespaces,
            },
        )
        .unwrap();
        let row_hash = accepted.row.row_hash().to_string();
        let request_digest = accepted.request_digest;
        let commitments = request["tn_row"]["tn.agents"]["field_hashes"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| (name.clone(), value.as_str().unwrap().to_string()))
            .collect();
        let governance = fixture.governance.clone();
        let session = runtime.start_session(&raw_request).unwrap();
        Self {
            _fixture: fixture,
            runtime,
            session,
            raw_request,
            row_hash,
            request_digest,
            governance,
            commitments,
        }
    }

    pub fn get_governance_result(&self) -> OwnedToolResult {
        OwnedToolResult {
            id: b"call-governance-1".to_vec(),
            name: b"get_governance".to_vec(),
            arguments: canonical_json(&json!({
                "request_digest": self.request_digest,
                "row_hash": self.row_hash,
            })),
            raw: br#"<tool_call>{"name":"get_governance"}</tool_call>"#.to_vec(),
            content: Vec::new(),
            call_count: 1,
            truncated: false,
            stop_limit: false,
        }
    }

    pub fn commitment_echo_result(&self) -> OwnedToolResult {
        OwnedToolResult {
            id: b"call-echo-2".to_vec(),
            name: b"continue_with_governance".to_vec(),
            arguments: canonical_json(&json!({
                "governance": self.governance,
                "governance_commitments": self.commitments,
                "request_digest": self.request_digest,
                "row_hash": self.row_hash,
            })),
            raw: br#"<tool_call>{"name":"continue_with_governance"}</tool_call>"#.to_vec(),
            content: Vec::new(),
            call_count: 1,
            truncated: false,
            stop_limit: false,
        }
    }

    pub fn section_selection_result(&self, sections: &[&str]) -> OwnedToolResult {
        OwnedToolResult {
            id: b"call-selection-3".to_vec(),
            name: b"select_sections".to_vec(),
            arguments: canonical_json(&json!({"sections": sections})),
            raw: br#"<tool_call>{"name":"select_sections"}</tool_call>"#.to_vec(),
            content: Vec::new(),
            call_count: 1,
            truncated: false,
            stop_limit: false,
        }
    }

    pub fn action_result(&self, id: &str, name: &str, arguments: Value) -> OwnedToolResult {
        OwnedToolResult {
            id: id.as_bytes().to_vec(),
            name: name.as_bytes().to_vec(),
            arguments: canonical_json(&arguments),
            raw: format!(r#"<tool_call>{{"name":"{name}"}}</tool_call>"#).into_bytes(),
            content: Vec::new(),
            call_count: 1,
            truncated: false,
            stop_limit: false,
        }
    }
}

pub struct OwnedToolResult {
    pub id: Vec<u8>,
    pub name: Vec<u8>,
    pub arguments: Vec<u8>,
    pub raw: Vec<u8>,
    pub content: Vec<u8>,
    pub call_count: usize,
    pub truncated: bool,
    pub stop_limit: bool,
}

impl OwnedToolResult {
    pub fn native(&self) -> NativeToolResult<'_> {
        NativeToolResult {
            call_count: self.call_count,
            call: (self.call_count == 1).then_some(ParsedToolCall {
                id: &self.id,
                name: &self.name,
                parsed_arguments: &self.arguments,
            }),
            content: &self.content,
            raw_model_output: &self.raw,
            truncated: self.truncated,
            stop_limit: self.stop_limit,
        }
    }
}

pub fn fixture_config(fixture: &RealTnFixture) -> RuntimeConfig {
    serde_json::from_slice(&fs::read(fixture.temp.path().join("runtime-config.json")).unwrap())
        .unwrap()
}

pub fn literal_runnable_manifest(n_predict: u32) -> Vec<u8> {
    canonical_json(&json!({
        "schema": "tn.llama.runnable-task.v1",
        "effective_ctx_shift": false,
        "task": {
            "type": "completion",
            "tokens": [1, 2, -3],
            "tokens_has_mtmd": false,
            "id_slot": -1,
            "index": 0,
            "id_parent": -1,
            "child_count": 0,
            "cli": false,
            "cli_prompt_byte_length": 0,
            "cli_file_count": 0
        },
        "params": {
            "stream": false,
            "cache_prompt": false,
            "n_cache_reuse": 0,
            "n_cmpl": 1,
            "n_predict": n_predict,
            "n_keep": 0,
            "n_discard": 0,
            "n_indent": 0,
            "t_max_predict_ms": -1,
            "lora": [],
            "stop": [],
            "response_fields": [],
            "post_sampling_probs": false,
            "res_type": "oai_chat",
            "upstream_params_json": {}
        },
        "sampling_native": {
            "n_prev": 64,
            "logit_bias_eog": [],
            "grammar_type": "tool_calls",
            "sampling_generation_prompt": "",
            "reasoning_budget_tokens": -1,
            "reasoning_budget_start": [],
            "reasoning_budget_end": [],
            "reasoning_budget_forced": [],
            "reasoning_budget_message": "",
            "reasoning_control": false,
            "backend_sampling": false
        },
        "speculative": {"types": ["none"]},
        "chat_parser": {
            "format": "content_only",
            "reasoning_format": "none",
            "reasoning_in_content": false,
            "generation_prompt": "",
            "parse_tool_calls": true,
            "is_continuation": false,
            "echo": false,
            "debug": false
        }
    }))
}

pub fn canonical_json(value: &Value) -> Vec<u8> {
    tn_core::canonical::canonical_bytes(value).unwrap()
}

pub fn prepare_and_begin(turn: &ModelTurn, prompt: &[u8], n_predict: u32, task_id: i64) {
    let manifest = literal_runnable_manifest(n_predict);
    turn.permit
        .prepare(&turn.request_json, prompt, &manifest)
        .unwrap();
    turn.permit
        .begin(task_id, 0, &turn.request_json, prompt, &manifest)
        .unwrap();
}

pub fn advance_to_echo(harness: &mut SessionHarness) -> ModelTurn {
    let turn = harness.session.next_model_turn().unwrap();
    prepare_and_begin(&turn, b"rendered-governance-read", 256, 11);
    let result = harness.get_governance_result();
    harness
        .session
        .accept_model_action(turn.permit, result.native())
        .map(|outcome| assert!(matches!(outcome, ActionOutcome::Continue)))
        .unwrap();
    harness.session.next_model_turn().unwrap()
}

pub fn advance_to_selection(harness: &mut SessionHarness) -> ModelTurn {
    let turn = advance_to_echo(harness);
    prepare_and_begin(&turn, b"rendered-commitment-echo", 1536, 12);
    let result = harness.commitment_echo_result();
    harness
        .session
        .accept_model_action(turn.permit, result.native())
        .map(|outcome| assert!(matches!(outcome, ActionOutcome::Continue)))
        .unwrap();
    harness.session.next_model_turn().unwrap()
}
