//! Backend-selected execution observations. These are internal runtime records,
//! not TN envelope formats or application protocol verbs.
use crate::model_protocol::session_error;
use crate::{Phase, TnLlamaError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const VLLM_REVISION: &str = "98dff2a81d747d1dba01a47f939f48c3526d4206";

#[derive(Clone, Debug, Default)]
pub enum ExecutionBackend {
    #[default]
    Llama,
    Vllm(VllmBackend),
}

#[derive(Clone, Debug)]
pub struct VllmBackend {
    engine_instance: String,
    model_bundle_sha256: String,
    tokenizer_sha256: String,
    template_sha256: String,
    parser: String,
}
impl VllmBackend {
    pub fn new(
        engine_instance: &str,
        model_bundle_sha256: &str,
        tokenizer_sha256: &str,
        template_sha256: &str,
        parser: &str,
        revision: &str,
    ) -> Result<Self, TnLlamaError> {
        if !identifier(engine_instance)
            || !identifier(parser)
            || revision != VLLM_REVISION
            || ![model_bundle_sha256, tokenizer_sha256, template_sha256]
                .iter()
                .all(|s| {
                    s.len() == 64
                        && s.bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
        {
            return Err(invalid());
        }
        Ok(Self {
            engine_instance: engine_instance.into(),
            model_bundle_sha256: model_bundle_sha256.into(),
            tokenizer_sha256: tokenizer_sha256.into(),
            template_sha256: template_sha256.into(),
            parser: parser.into(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VllmExecutionIdentity {
    pub engine_instance: String,
    pub request_id: String,
    pub external_request_id: String,
    pub client_index: u32,
    pub destination: String,
    pub wave: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum ExecutionIdentity {
    Llama { task_id: i64, slot_id: i32 },
    Vllm(VllmExecutionIdentity),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VllmManifest {
    schema: String,
    engine_instance: String,
    model_bundle_sha256: String,
    tokenizer_sha256: String,
    template_sha256: String,
    parser: String,
    request_id: String,
    external_request_id: String,
    prompt_token_ids: Vec<u32>,
    sampling_params: Map<String, Value>,
    reasoning_ended: Option<bool>,
    reasoning_parser_kwargs: Value,
    arrival_time: f64,
    priority: i32,
    trace_headers: Value,
    cache_salt: Value,
    data_parallel_rank: Value,
    mm_features: Value,
    pooling_params: Value,
    lora_request: Value,
    prompt_embeds: Value,
    prompt_is_token_ids: Value,
    resumable: bool,
    abort_immediately: bool,
    session_id: Value,
}

fn identifier(s: &str) -> bool {
    !s.is_empty() && s.len() <= 512 && !s.chars().any(char::is_control)
}
fn invalid() -> TnLlamaError {
    session_error(
        "tn_vllm_manifest_invalid",
        "execution does not match the configured vLLM text backend",
    )
}

impl ExecutionBackend {
    pub(crate) fn digest(
        &self,
        raw: &[u8],
        phase: Phase,
        max_tokens: u32,
    ) -> Result<String, TnLlamaError> {
        match self {
            Self::Llama => crate::model_protocol::parse_and_digest_manifest(raw, phase, max_tokens),
            Self::Vllm(config) => {
                let value =
                    crate::envelope::parse_duplicate_safe_json(raw).map_err(|_| invalid())?;
                let object = value.as_object().ok_or_else(invalid)?;
                if object.len() != MANIFEST_FIELDS.len()
                    || MANIFEST_FIELDS.iter().any(|key| !object.contains_key(*key))
                {
                    return Err(invalid());
                }
                let manifest: VllmManifest =
                    serde_json::from_value(value.clone()).map_err(|_| invalid())?;
                manifest.validate(config, max_tokens)?;
                let bytes = tn_core::canonical::canonical_bytes(&value).map_err(|_| invalid())?;
                let mut hash = Sha256::new();
                hash.update(b"tn.vllm.execution.v1\0");
                hash.update(bytes);
                Ok(format!("sha256:{}", hex::encode(hash.finalize())))
            }
        }
    }
    pub(crate) fn validate_identity(
        &self,
        identity: &ExecutionIdentity,
        raw: &[u8],
    ) -> Result<(), TnLlamaError> {
        match (self, identity) {
            (Self::Llama, ExecutionIdentity::Llama { .. }) => Ok(()),
            (Self::Vllm(config), ExecutionIdentity::Vllm(id)) => {
                let manifest: VllmManifest = serde_json::from_slice(raw).map_err(|_| invalid())?;
                if id.engine_instance != config.engine_instance
                    || id.request_id != manifest.request_id
                    || id.external_request_id != manifest.external_request_id
                    || id.client_index != 0
                    || id.wave != 0
                    || !identifier(&id.destination)
                {
                    return Err(invalid());
                }
                Ok(())
            }
            _ => Err(session_error(
                "tn_execution_backend_mismatch",
                "execution identity does not match the runtime backend",
            )),
        }
    }
}

impl VllmManifest {
    fn validate(&self, c: &VllmBackend, max_tokens: u32) -> Result<(), TnLlamaError> {
        let _bound_reasoning = (&self.reasoning_ended, &self.reasoning_parser_kwargs);
        if self.schema != "tn.vllm.execution.v1"
            || self.engine_instance != c.engine_instance
            || self.model_bundle_sha256 != c.model_bundle_sha256
            || self.tokenizer_sha256 != c.tokenizer_sha256
            || self.template_sha256 != c.template_sha256
            || self.parser != c.parser
            || !identifier(&self.request_id)
            || !identifier(&self.external_request_id)
            || self.prompt_token_ids.is_empty()
            || !self.arrival_time.is_finite()
            || self.arrival_time < 0.0
            || self.priority != 0
            || self.resumable
            || self.abort_immediately
            || [
                &self.trace_headers,
                &self.cache_salt,
                &self.data_parallel_rank,
                &self.mm_features,
                &self.pooling_params,
                &self.lora_request,
                &self.prompt_embeds,
                &self.prompt_is_token_ids,
                &self.session_id,
            ]
            .iter()
            .any(|v| !v.is_null())
        {
            return Err(invalid());
        }
        let p = &self.sampling_params;
        if p.len() != SAMPLING_FIELDS.len()
            || p.get("n") != Some(&Value::from(1))
            || p.get("max_tokens") != Some(&Value::from(max_tokens))
            || p.get("temperature").and_then(Value::as_f64) != Some(0.0)
            || p.get("output_kind") != Some(&Value::from(2))
            || p.get("detokenize") != Some(&Value::Bool(true))
            || p.get("skip_clone") != Some(&Value::Bool(false))
            || p.keys().any(|key| !SAMPLING_FIELDS.contains(&key.as_str()))
        {
            return Err(invalid());
        }
        // Runtime-loaded extensions or additional output channels are not part of this backend.
        for key in [
            "extra_args",
            "logprobs",
            "prompt_logprobs",
            "logprob_token_ids",
            "trace_decode_token_ids",
        ] {
            if p.get(key).is_some_and(|v| !v.is_null()) {
                return Err(invalid());
            }
        }
        Ok(())
    }
}

// Every v0.29.0 SamplingParams field is either bound here or explicitly refused above.
// The frontend checks __struct_fields__ against this list on startup.
pub const SAMPLING_FIELDS: &[&str] = &[
    "n",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "max_tokens",
    "min_tokens",
    "logprobs",
    "prompt_logprobs",
    "logprob_token_ids",
    "flat_logprobs",
    "detokenize",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "include_stop_str_in_output",
    "output_kind",
    "stream_interval",
    "skip_clone",
    "output_text_buffer_length",
    "_eos_token_id",
    "_all_stop_token_ids",
    "structured_outputs",
    "logit_bias",
    "allowed_token_ids",
    "extra_args",
    "bad_words",
    "_bad_words_token_ids",
    "skip_reading_prefix_cache",
    "thinking_token_budget",
    "repetition_detection",
    "routed_experts_prompt_start",
    "trace_decode_token_ids",
];

const MANIFEST_FIELDS: &[&str] = &[
    "schema",
    "engine_instance",
    "model_bundle_sha256",
    "tokenizer_sha256",
    "template_sha256",
    "parser",
    "request_id",
    "external_request_id",
    "prompt_token_ids",
    "sampling_params",
    "reasoning_ended",
    "reasoning_parser_kwargs",
    "arrival_time",
    "priority",
    "trace_headers",
    "cache_salt",
    "data_parallel_rank",
    "mm_features",
    "pooling_params",
    "lora_request",
    "prompt_embeds",
    "prompt_is_token_ids",
    "resumable",
    "abort_immediately",
    "session_id",
];
