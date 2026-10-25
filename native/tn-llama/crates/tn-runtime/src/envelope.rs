use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tn_core::sealed_object::{extract_group_blocks, require_envelope_shape, verify_sealed};

use crate::config::RequestLimits;
use crate::error::{AdmissionStage, TnLlamaError};
use crate::governance::{GOVERNANCE_FIELDS, GOVERNANCE_GROUP};

pub const REQUEST_PROTOCOL: &str = "tn.llama.request.v1";
pub const REQUEST_DIGEST_DOMAIN: &[u8] = b"tn.llama.request-context.v1\0";

const WRAPPER_FIELDS: [&str; 5] = ["messages", "protocol", "purpose", "stream", "tn_row"];
const MESSAGE_FIELDS: [&str; 2] = ["content", "role"];
const GROUP_FIELDS: [&str; 2] = ["ciphertext", "field_hashes"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UserMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct VerifiedRow {
    envelope: Map<String, Value>,
    recipient_source_json: String,
    device_identity: String,
    event_id: String,
    event_type: String,
    sequence: u64,
    prev_hash: String,
    row_hash: String,
    group_field_hashes: BTreeMap<String, BTreeMap<String, String>>,
    aad_markers: BTreeMap<String, BTreeMap<String, Value>>,
    signature_verified: bool,
}

impl VerifiedRow {
    #[must_use]
    pub fn recipient_source_json(&self) -> &str {
        &self.recipient_source_json
    }

    #[must_use]
    pub fn device_identity(&self) -> &str {
        &self.device_identity
    }

    #[must_use]
    pub fn event_id(&self) -> &str {
        &self.event_id
    }

    #[must_use]
    pub fn event_type(&self) -> &str {
        &self.event_type
    }

    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    #[must_use]
    pub fn prev_hash(&self) -> &str {
        &self.prev_hash
    }

    #[must_use]
    pub fn row_hash(&self) -> &str {
        &self.row_hash
    }

    #[must_use]
    pub fn governance_field_hashes(&self) -> &BTreeMap<String, String> {
        self.group_field_hashes
            .get(GOVERNANCE_GROUP)
            .expect("validated row always has tn.agents commitments")
    }

    #[must_use]
    pub fn group_field_hashes(&self, group: &str) -> Option<&BTreeMap<String, String>> {
        self.group_field_hashes.get(group)
    }

    #[must_use]
    pub fn encrypted_group_manifest(&self) -> BTreeMap<String, Vec<String>> {
        self.group_field_hashes
            .iter()
            .map(|(group, hashes)| (group.clone(), hashes.keys().cloned().collect()))
            .collect()
    }

    #[must_use]
    pub fn aad_marker(&self, group: &str) -> Option<&BTreeMap<String, Value>> {
        self.aad_markers.get(group)
    }

    #[must_use]
    pub const fn signature_verified(&self) -> bool {
        self.signature_verified
    }

    #[must_use]
    pub fn public_aad(&self) -> Option<&str> {
        self.envelope.get("tn_aad").and_then(Value::as_str)
    }
}

#[derive(Clone, Debug)]
pub struct AcceptedRequest {
    pub row: VerifiedRow,
    pub messages: Vec<UserMessage>,
    pub purpose: String,
    pub request_digest: String,
}

pub fn parse_and_verify_request(
    raw: &[u8],
    limits: &RequestLimits,
) -> Result<AcceptedRequest, TnLlamaError> {
    if raw.len() > limits.max_request_bytes {
        return Err(error(
            "tn_request_too_large",
            AdmissionStage::Parse,
            "request body exceeds the configured byte limit",
        ));
    }

    let request = parse_duplicate_safe_json(raw).map_err(|source| {
        let message = source.to_string();
        if message.contains("duplicate key") {
            error("tn_duplicate_key", AdmissionStage::Parse, message)
        } else {
            error("tn_request_json_invalid", AdmissionStage::Parse, message)
        }
    })?;

    let wrapper = request.as_object().ok_or_else(request_shape)?;
    require_exact_keys(wrapper, &WRAPPER_FIELDS).map_err(|_| request_shape())?;

    if wrapper.get("protocol").and_then(Value::as_str) != Some(REQUEST_PROTOCOL)
        || wrapper.get("stream").and_then(Value::as_bool) != Some(false)
    {
        return Err(request_shape());
    }

    let purpose = nonempty_string(wrapper.get("purpose")).ok_or_else(request_shape)?;
    let message_values = wrapper
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| messages.len() == 1)
        .ok_or_else(request_shape)?;
    let message = message_values[0].as_object().ok_or_else(request_shape)?;
    require_exact_keys(message, &MESSAGE_FIELDS).map_err(|_| request_shape())?;
    if message.get("role").and_then(Value::as_str) != Some("user") {
        return Err(request_shape());
    }
    let content = nonempty_string(message.get("content")).ok_or_else(request_shape)?;
    if content.len() > limits.max_message_bytes {
        return Err(request_shape());
    }
    let messages = vec![UserMessage {
        role: "user".to_string(),
        content: content.to_string(),
    }];

    let row_value = wrapper.get("tn_row").cloned().ok_or_else(request_shape)?;
    let envelope = row_value.as_object().cloned().ok_or_else(row_shape)?;
    let device_identity = header_string(&envelope, "device_identity")?.to_string();
    let event_type = header_string(&envelope, "event_type")?.to_string();
    if limits
        .expected_device_identity
        .as_deref()
        .is_some_and(|expected| device_identity != expected)
        || event_type != limits.expected_event_type
    {
        return Err(error(
            "tn_source_namespace_mismatch",
            AdmissionStage::RowShape,
            "row is outside the configured source namespace",
        ));
    }
    let validated = validate_envelope_shape(
        envelope.clone(),
        &limits.allowed_input_groups,
        limits.require_signature,
    )?;

    let recipient_source_json = String::from_utf8(
        tn_core::canonical::canonical_bytes(&row_value).map_err(|source| {
            error(
                "tn_canonicalization_failed",
                AdmissionStage::RowShape,
                source.to_string(),
            )
        })?,
    )
    .map_err(|source| {
        error(
            "tn_canonicalization_failed",
            AdmissionStage::RowShape,
            source.to_string(),
        )
    })?;

    let row_hash = header_string(&envelope, "row_hash")?.to_string();
    let digest_input = json!({
        "messages": message_values,
        "purpose": purpose,
        "row_hash": &row_hash,
    });
    let canonical_context =
        tn_core::canonical::canonical_bytes(&digest_input).map_err(|source| {
            error(
                "tn_canonicalization_failed",
                AdmissionStage::RequestShape,
                source.to_string(),
            )
        })?;
    let mut request_hasher = Sha256::new();
    request_hasher.update(REQUEST_DIGEST_DOMAIN);
    request_hasher.update(canonical_context);
    let request_digest = format!("sha256:{}", hex::encode(request_hasher.finalize()));

    Ok(AcceptedRequest {
        row: VerifiedRow {
            envelope,
            recipient_source_json,
            device_identity,
            event_id: header_string(
                wrapper["tn_row"].as_object().ok_or_else(row_shape)?,
                "event_id",
            )?
            .to_string(),
            event_type,
            sequence: wrapper["tn_row"]["sequence"]
                .as_u64()
                .ok_or_else(row_shape)?,
            prev_hash: header_string(
                wrapper["tn_row"].as_object().ok_or_else(row_shape)?,
                "prev_hash",
            )?
            .to_string(),
            row_hash,
            group_field_hashes: validated.group_field_hashes,
            aad_markers: validated.aad_markers,
            signature_verified: validated.signature_verified,
        },
        messages,
        purpose: purpose.to_string(),
        request_digest,
    })
}

struct ValidatedEnvelope {
    group_field_hashes: BTreeMap<String, BTreeMap<String, String>>,
    aad_markers: BTreeMap<String, BTreeMap<String, Value>>,
    signature_verified: bool,
}

fn validate_envelope_shape(
    envelope: Map<String, Value>,
    allowed_input_groups: &BTreeSet<String>,
    require_signature: bool,
) -> Result<ValidatedEnvelope, TnLlamaError> {
    for key in tn_core::sealed_object::ENVELOPE_RESERVED {
        if !envelope.contains_key(key) {
            return Err(row_shape());
        }
    }

    for key in [
        "device_identity",
        "timestamp",
        "event_id",
        "event_type",
        "level",
        "prev_hash",
        "row_hash",
    ] {
        if nonempty_string(envelope.get(key)).is_none() {
            return Err(row_shape());
        }
    }
    if envelope.get("sequence").and_then(Value::as_u64).is_none()
        || envelope.get("signature").and_then(Value::as_str).is_none()
    {
        return Err(row_shape());
    }
    if require_signature
        && envelope
            .get("signature")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    {
        return Err(signature_mismatch());
    }

    let mut found_governance = false;
    let mut encrypted_groups = BTreeMap::new();
    for (name, value) in &envelope {
        if tn_core::sealed_object::ENVELOPE_RESERVED.contains(&name.as_str()) {
            continue;
        }
        if name == "tn_aad" {
            continue;
        }
        if name != GOVERNANCE_GROUP && !allowed_input_groups.contains(name) {
            return Err(row_shape());
        }
        let block = value.as_object().ok_or_else(row_shape)?;
        require_exact_keys(block, &GROUP_FIELDS).map_err(|_| row_shape())?;
        if block
            .get("ciphertext")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
            || !block.get("field_hashes").is_some_and(Value::is_object)
        {
            return Err(row_shape());
        }
        validate_string_map(block.get("field_hashes")).map_err(|_| row_shape())?;
        if name == GOVERNANCE_GROUP {
            validate_governance_hashes(block)?;
            found_governance = true;
        }
        encrypted_groups.insert(name.clone(), ());
    }
    if !found_governance || (!allowed_input_groups.is_empty() && encrypted_groups.len() < 2) {
        return Err(governance_shape());
    }
    let aad_markers = match envelope.get("tn_aad") {
        Some(value) => validate_public_aad(value, &encrypted_groups)?,
        None if !require_signature => BTreeMap::new(),
        None => return Err(aad_shape()),
    };
    let upstream_envelope = require_envelope_shape(envelope).map_err(|source| {
        error(
            "tn_envelope_shape",
            AdmissionStage::RowShape,
            source.to_string(),
        )
    })?;
    let blocks = extract_group_blocks(&upstream_envelope).map_err(|source| {
        error(
            "tn_envelope_shape",
            AdmissionStage::RowShape,
            source.to_string(),
        )
    })?;
    if blocks.len() != encrypted_groups.len()
        || !blocks.contains_key(GOVERNANCE_GROUP)
        || blocks
            .keys()
            .any(|group| !encrypted_groups.contains_key(group))
    {
        return Err(row_shape());
    }
    let valid = verify_sealed(&upstream_envelope, &blocks);
    if !valid.row_hash {
        return Err(row_hash_mismatch());
    }
    if require_signature && !valid.signature {
        return Err(signature_mismatch());
    }
    let group_field_hashes = blocks
        .into_iter()
        .map(|(group, block)| (group, block.field_hashes))
        .collect();
    Ok(ValidatedEnvelope {
        group_field_hashes,
        aad_markers,
        signature_verified: valid.signature,
    })
}

fn validate_public_aad(
    value: &Value,
    encrypted_groups: &BTreeMap<String, ()>,
) -> Result<BTreeMap<String, BTreeMap<String, Value>>, TnLlamaError> {
    let raw = value.as_str().ok_or_else(aad_shape)?;
    let parsed = parse_duplicate_safe_json(raw.as_bytes()).map_err(|_| aad_shape())?;
    let groups = parsed
        .as_object()
        .filter(|groups| !groups.is_empty())
        .ok_or_else(aad_shape)?;
    if groups.len() != encrypted_groups.len() {
        return Err(aad_shape());
    }
    let mut validated = BTreeMap::new();
    let mut expected_binding: Option<(&str, &str)> = None;
    for (group, marker) in groups {
        if !encrypted_groups.contains_key(group) {
            return Err(aad_shape());
        }
        let marker = marker
            .as_object()
            .filter(|marker| !marker.is_empty())
            .ok_or_else(aad_shape)?;
        if marker.len() != 2 || !marker.values().all(is_json_scalar) {
            return Err(aad_shape());
        }
        let required_string = |name: &str| {
            marker
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
        };
        let governed_by = required_string("governed_by").ok_or_else(aad_binding_mismatch)?;
        let policy = required_string("policy").ok_or_else(aad_binding_mismatch)?;
        if tn_core::trust::parse_ed25519_did_key(governed_by).is_err()
            || !is_content_addressed_policy_reference(policy)
            || expected_binding.is_some_and(|expected| expected != (governed_by, policy))
        {
            return Err(aad_binding_mismatch());
        }
        expected_binding.get_or_insert((governed_by, policy));
        validated.insert(
            group.clone(),
            marker
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        );
    }
    let canonical = tn_core::canonical::canonical_bytes(&parsed).map_err(|_| aad_shape())?;
    if canonical.as_slice() != raw.as_bytes() {
        return Err(aad_shape());
    }
    Ok(validated)
}

fn is_content_addressed_policy_reference(value: &str) -> bool {
    let Some((identity, digest)) = value.rsplit_once("#sha256:") else {
        return false;
    };
    !identity.is_empty()
        && digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_json_scalar(value: &Value) -> bool {
    value.is_null() || value.is_boolean() || value.is_number() || value.is_string()
}

fn validate_governance_hashes(block: &Map<String, Value>) -> Result<(), TnLlamaError> {
    let hashes = block
        .get("field_hashes")
        .and_then(Value::as_object)
        .ok_or_else(governance_shape)?;
    if !GOVERNANCE_FIELDS
        .iter()
        .all(|field| hashes.contains_key(*field))
    {
        return Err(governance_shape());
    }
    validate_string_map(block.get("field_hashes")).map_err(|_| governance_shape())
}

fn validate_string_map(value: Option<&Value>) -> Result<(), ()> {
    let values = value.and_then(Value::as_object).ok_or(())?;
    if values.values().all(Value::is_string) {
        Ok(())
    } else {
        Err(())
    }
}

fn header_string<'a>(
    envelope: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a str, TnLlamaError> {
    nonempty_string(envelope.get(name)).ok_or_else(row_shape)
}

fn nonempty_string(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn require_exact_keys<const N: usize>(
    object: &Map<String, Value>,
    expected: &[&str; N],
) -> Result<(), ()> {
    if object.len() != expected.len() || !expected.iter().all(|key| object.contains_key(*key)) {
        return Err(());
    }
    Ok(())
}

fn error(code: &'static str, stage: AdmissionStage, message: impl Into<String>) -> TnLlamaError {
    TnLlamaError::new(code, stage, message)
}

fn request_shape() -> TnLlamaError {
    error(
        "tn_request_shape",
        AdmissionStage::RequestShape,
        "request does not match tn.llama.request.v1",
    )
}

fn row_shape() -> TnLlamaError {
    error(
        "tn_envelope_shape",
        AdmissionStage::RowShape,
        "TN envelope does not match the encrypted governed request profile",
    )
}

fn governance_shape() -> TnLlamaError {
    error(
        "tn_governance_shape",
        AdmissionStage::RowShape,
        "tn.agents commitments must contain the six required governance fields",
    )
}

fn row_hash_mismatch() -> TnLlamaError {
    error(
        "tn_row_hash_mismatch",
        AdmissionStage::RowHash,
        "canonical TN row hash does not match the stored row hash",
    )
}

fn signature_mismatch() -> TnLlamaError {
    error(
        "tn_signature_mismatch",
        AdmissionStage::RowHash,
        "TN row signature does not verify for its self-certifying writer DID",
    )
}

fn aad_shape() -> TnLlamaError {
    error(
        "tn_aad_shape",
        AdmissionStage::RowShape,
        "tn_aad must be canonical duplicate-free JSON keyed to present encrypted groups",
    )
}

fn aad_binding_mismatch() -> TnLlamaError {
    error(
        "tn_aad_binding_mismatch",
        AdmissionStage::RowShape,
        "TN group AAD must carry one identical governing DID and content-addressed policy reference",
    )
}

pub(crate) fn parse_duplicate_safe_json(raw: &[u8]) -> Result<Value, serde_json::Error> {
    serde_json::from_slice(raw).map(|StrictValue(value)| value)
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .map(StrictValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_string())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(StrictValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut seen = HashSet::new();
        while let Some(key) = object.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate key {key:?}")));
            }
            let StrictValue(value) = object.next_value()?;
            values.insert(key, value);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}
