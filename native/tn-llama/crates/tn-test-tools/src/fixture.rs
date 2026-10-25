use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tn_core::chain::{compute_row_hash, GroupInput, RowHashInput};
use tn_core::sealed_object::{extract_group_blocks, ENVELOPE_RESERVED};
use tn_core::signing::signature_b64;
use tn_core::{DeviceKey, Runtime};

use crate::FixtureError;

const EVENT_TYPE: &str = "precisely.risk.enriched";
const GOVERNANCE_GROUP: &str = "tn.agents";
const PROTECTED_GROUPS: [&str; 4] = ["default", "firm_data", "precisely", GOVERNANCE_GROUP];
const PROPERTY_REPORT_PROMPT: &str = "Generate a concise HTML wildfire-risk report.";
const POLICY_TEXT: &str = "# TN Agents Policy\n\
version: 1\n\
schema: tn-agents-policy@v1\n\
\n\
## precisely.risk.enriched\n\
\n\
### instruction\n\
This record contains proprietary Precisely Wildfire Risk data in the encrypted `precisely` section.\n\
\n\
### use_for\n\
Use the `precisely` section for property reporting, underwriting, and risk assessment. Any derived report that uses this section must carry a human-readable statement attributing its data source to Precisely.\n\
\n\
### do_not_use_for\n\
Do not use the `precisely` section for unattributed public disclosure or strip its source lineage.\n\
\n\
### consequences\n\
Any output derived from the `precisely` section must carry this governance and exact source lineage.\n\
\n\
### on_violation_or_error\n\
Before emitting a derived report that uses the `precisely` section, call the configured compliance logging tool. If attribution or lineage cannot be preserved, stop and release nothing.\n\
";
const OUTPUT_POLICY_TEXT: &str = include_str!("../../../demo/policies/output-agents.txt");
const GOVERNANCE_FIELDS: [&str; 7] = [
    "instruction",
    "use_for",
    "do_not_use_for",
    "consequences",
    "on_violation_or_error",
    "policy",
    "governs_sections",
];
const FIXTURE_LLAMA_REVISION: &str = "3f545beccee69d9975f466ec7e45fd9aacd8ba90";
const FIXTURE_MODEL_SHA256: &str =
    "9a7bed0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
const FIXTURE_BINARY_SHA256: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildInputs {
    pub llama_source_revision: String,
    pub model_sha256: String,
    pub binary_sha256: String,
}

impl BuildInputs {
    #[must_use]
    pub fn new(
        llama_source_revision: impl Into<String>,
        model_sha256: impl Into<String>,
        binary_sha256: impl Into<String>,
    ) -> Self {
        Self {
            llama_source_revision: llama_source_revision.into(),
            model_sha256: model_sha256.into(),
            binary_sha256: binary_sha256.into(),
        }
    }

    fn validate(&self) -> Result<(), FixtureError> {
        require_lower_hex("llama source revision", &self.llama_source_revision, 40)?;
        require_lower_hex("model SHA-256", &self.model_sha256, 64)?;
        require_lower_hex("binary SHA-256", &self.binary_sha256, 64)
    }
}

pub struct RealTnFixture {
    pub temp: TempDir,
    pub receiver_key_dir: PathBuf,
    pub input_index_key_path: PathBuf,
    pub ceremony_id: String,
    pub input_index_epoch: u64,
    pub device_identity: String,
    pub governance: BTreeMap<String, Value>,
    pub aad_marker: Value,
    pub aad_bound_request_json: Vec<u8>,
    pub plaintext_report_request_json: Vec<u8>,
    pub row1_request_json: Vec<u8>,
    pub row2_request_json: Vec<u8>,
    pub stale_fork_request_json: Vec<u8>,
    pub signature_mutation_request_json: Vec<u8>,
    pub commitment_mutation_request_json: Vec<u8>,
    pub aad_mutation_request_json: Vec<u8>,
    pub aad_policy_mismatch_request_json: Vec<u8>,
    pub ciphertext_mutation_request_json: Vec<u8>,
    pub missing_governance_field_request_json: Vec<u8>,
    pub extra_governance_field_request_json: Vec<u8>,
    pub extra_governance_commitment_mutation_request_json: Vec<u8>,
}

pub fn build_real_tn_fixture() -> Result<RealTnFixture, FixtureError> {
    let temp = tempfile::Builder::new()
        .prefix("tn-llama-fixture-")
        .tempdir()?;
    let inputs = BuildInputs::new(
        FIXTURE_LLAMA_REVISION,
        FIXTURE_MODEL_SHA256,
        FIXTURE_BINARY_SHA256,
    );
    let generated = generate_fixture()?;
    write_fixture(temp.path(), &inputs, &generated)?;

    Ok(RealTnFixture {
        receiver_key_dir: temp.path().join("receiver"),
        input_index_key_path: temp.path().join("input.index.key"),
        ceremony_id: generated.ceremony_id,
        input_index_epoch: generated.input_index_epoch,
        device_identity: generated.device_identity,
        governance: generated.governance,
        aad_marker: Value::Object(generated.aad_marker.clone()),
        aad_bound_request_json: generated.aad_bound_request_json,
        plaintext_report_request_json: generated.plaintext_report_request_json,
        row1_request_json: generated.row1_request_json,
        row2_request_json: generated.row2_request_json,
        stale_fork_request_json: generated.stale_fork_request_json,
        signature_mutation_request_json: generated.signature_mutation_request_json,
        commitment_mutation_request_json: generated.commitment_mutation_request_json,
        aad_mutation_request_json: generated.aad_mutation_request_json,
        aad_policy_mismatch_request_json: generated.aad_policy_mismatch_request_json,
        ciphertext_mutation_request_json: generated.ciphertext_mutation_request_json,
        missing_governance_field_request_json: generated.missing_governance_field_request_json,
        extra_governance_field_request_json: generated.extra_governance_field_request_json,
        extra_governance_commitment_mutation_request_json: generated
            .extra_governance_commitment_mutation_request_json,
        temp,
    })
}

pub fn create_fixture_at(out: &Path, build: &BuildInputs) -> Result<PathBuf, FixtureError> {
    build.validate()?;
    prepare_output_directory(out)?;
    let generated = generate_fixture()?;
    write_fixture(out, build, &generated)
}

struct GeneratedFixture {
    receiver_kits: BTreeMap<String, Vec<u8>>,
    input_index_key: [u8; 32],
    ceremony_id: String,
    input_index_epoch: u64,
    device_identity: String,
    governance: BTreeMap<String, Value>,
    aad_marker: Map<String, Value>,
    aad_bound_request_json: Vec<u8>,
    plaintext_report_request_json: Vec<u8>,
    plaintext_source_row1_ndjson: Vec<u8>,
    plaintext_source_row2_ndjson: Vec<u8>,
    row1_request_json: Vec<u8>,
    row2_request_json: Vec<u8>,
    stale_fork_request_json: Vec<u8>,
    signature_mutation_request_json: Vec<u8>,
    commitment_mutation_request_json: Vec<u8>,
    aad_mutation_request_json: Vec<u8>,
    aad_policy_mismatch_request_json: Vec<u8>,
    ciphertext_mutation_request_json: Vec<u8>,
    missing_governance_field_request_json: Vec<u8>,
    extra_governance_field_request_json: Vec<u8>,
    extra_governance_commitment_mutation_request_json: Vec<u8>,
}

struct EmittedRows {
    receiver_kits: BTreeMap<String, Vec<u8>>,
    input_index_key: [u8; 32],
    ceremony_id: String,
    input_index_epoch: u64,
    device_identity: String,
    rows: Vec<Map<String, Value>>,
    aad_marker: Map<String, Value>,
}

fn generate_fixture() -> Result<GeneratedFixture, FixtureError> {
    let device = DeviceKey::generate();
    let governance_authority = DeviceKey::generate();
    let governed_by = governance_authority.did().to_string();
    let device_seed = device.private_bytes();
    let base = emit_rows(
        &device_seed,
        &governed_by,
        &GOVERNANCE_FIELDS,
        source_fields(),
        2,
        b"base",
        None,
    )?;
    let missing_fields = [
        "instruction",
        "use_for",
        "do_not_use_for",
        "consequences",
        "on_violation_or_error",
        "governs_sections",
    ];
    let missing = emit_rows(
        &device_seed,
        &governed_by,
        &missing_fields,
        source_fields(),
        1,
        b"missing",
        None,
    )?;
    let mut extra_fields = GOVERNANCE_FIELDS.to_vec();
    extra_fields.push("governance_revision");
    let mut extra_event_fields = source_fields();
    extra_event_fields.insert("governance_revision".to_string(), json!(2));
    let extra = emit_rows(
        &device_seed,
        &governed_by,
        &extra_fields,
        extra_event_fields,
        1,
        b"base",
        None,
    )?;
    let aad_policy_mismatch = emit_rows(
        &device_seed,
        &governed_by,
        &GOVERNANCE_FIELDS,
        source_fields(),
        1,
        b"base",
        Some(".tn/config/agents.md#other.policy@1#sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    )?;

    let row1 = base.rows[0].clone();
    let row2 = base.rows[1].clone();
    let governance = open_source_governance(&row1, &base)?;
    let mut plaintext_source_row1 = plaintext_variant(&row1, &governance)?;
    official_plaintext_rehash_and_sign(&mut plaintext_source_row1, &device)?;
    let mut plaintext_source_row2 = plaintext_variant(&row2, &governance)?;
    plaintext_source_row2.insert(
        "prev_hash".to_string(),
        Value::String(header(&plaintext_source_row1, "row_hash")?.to_string()),
    );
    official_plaintext_rehash_and_sign(&mut plaintext_source_row2, &device)?;

    let mut stale_fork = row2.clone();
    stale_fork.insert(
        "event_id".to_string(),
        Value::String(changed_identifier(header(&row2, "event_id")?)),
    );
    official_rehash_and_sign(&mut stale_fork, &device)?;

    let mut signature_mutation = row1.clone();
    let signature = header(&signature_mutation, "signature")?;
    let mut signature_bytes = URL_SAFE_NO_PAD
        .decode(signature)
        .map_err(|_| FixtureError::Invariant("TN emitted invalid signature base64"))?;
    let first = signature_bytes
        .first_mut()
        .ok_or(FixtureError::Invariant("TN emitted empty signature"))?;
    *first ^= 0x01;
    signature_mutation.insert(
        "signature".to_string(),
        Value::String(URL_SAFE_NO_PAD.encode(signature_bytes)),
    );

    let mut commitment_mutation = row1.clone();
    let token = commitment_mutation["precisely"]["field_hashes"]["address"]
        .as_str()
        .ok_or(FixtureError::Invariant("address token is not a string"))?;
    commitment_mutation["precisely"]["field_hashes"]["address"] =
        Value::String(flip_terminal_hex(token)?);
    official_rehash_and_sign(&mut commitment_mutation, &device)?;

    let mut aad_mutation = row1.clone();
    let aad_text = aad_mutation["tn_aad"]
        .as_str()
        .ok_or(FixtureError::Invariant("real row omitted tn_aad"))?;
    let mut aad_value: Value = serde_json::from_str(aad_text)?;
    let changed_governance_authority = DeviceKey::generate().did().to_string();
    for group in PROTECTED_GROUPS {
        aad_value[group]["governed_by"] = json!(changed_governance_authority);
    }
    let canonical_aad = tn_core::canonical::canonical_bytes(&aad_value)?;
    aad_mutation["tn_aad"] = Value::String(
        String::from_utf8(canonical_aad)
            .map_err(|_| FixtureError::Invariant("canonical AAD is not UTF-8"))?,
    );
    official_rehash_and_sign(&mut aad_mutation, &device)?;

    let mut extra_commitment_mutation = extra.rows[0].clone();
    let token = extra_commitment_mutation["tn.agents"]["field_hashes"]["governance_revision"]
        .as_str()
        .ok_or(FixtureError::Invariant(
            "governance_revision token is not a string",
        ))?;
    extra_commitment_mutation["tn.agents"]["field_hashes"]["governance_revision"] =
        Value::String(flip_terminal_hex(token)?);
    official_rehash_and_sign(&mut extra_commitment_mutation, &device)?;

    let mut ciphertext_mutation = row1.clone();
    let ciphertext = ciphertext_mutation["precisely"]["ciphertext"]
        .as_str()
        .ok_or(FixtureError::Invariant(
            "precisely ciphertext is not a string",
        ))?;
    let mut ciphertext_bytes = STANDARD
        .decode(ciphertext)
        .map_err(|_| FixtureError::Invariant("TN emitted invalid base64 ciphertext"))?;
    let first = ciphertext_bytes
        .first_mut()
        .ok_or(FixtureError::Invariant("TN emitted empty ciphertext"))?;
    *first ^= 0x01;
    ciphertext_mutation["precisely"]["ciphertext"] =
        Value::String(STANDARD.encode(ciphertext_bytes));
    official_rehash_and_sign(&mut ciphertext_mutation, &device)?;

    Ok(GeneratedFixture {
        receiver_kits: base.receiver_kits,
        input_index_key: base.input_index_key,
        ceremony_id: base.ceremony_id,
        input_index_epoch: base.input_index_epoch,
        device_identity: base.device_identity,
        governance,
        aad_marker: base.aad_marker,
        aad_bound_request_json: wrap_request(&row1)?,
        plaintext_report_request_json: wrap_request_with_prompt(
            &plaintext_source_row1,
            PROPERTY_REPORT_PROMPT,
        )?,
        plaintext_source_row1_ndjson: ndjson_row(&plaintext_source_row1)?,
        plaintext_source_row2_ndjson: ndjson_row(&plaintext_source_row2)?,
        row1_request_json: wrap_request(&row1)?,
        row2_request_json: wrap_request(&row2)?,
        stale_fork_request_json: wrap_request(&stale_fork)?,
        signature_mutation_request_json: wrap_request(&signature_mutation)?,
        commitment_mutation_request_json: wrap_request(&commitment_mutation)?,
        aad_mutation_request_json: wrap_request(&aad_mutation)?,
        aad_policy_mismatch_request_json: wrap_request(&aad_policy_mismatch.rows[0])?,
        ciphertext_mutation_request_json: wrap_request(&ciphertext_mutation)?,
        missing_governance_field_request_json: wrap_request(&missing.rows[0])?,
        extra_governance_field_request_json: wrap_request(&extra.rows[0])?,
        extra_governance_commitment_mutation_request_json: wrap_request(
            &extra_commitment_mutation,
        )?,
    })
}

fn source_fields() -> Map<String, Value> {
    json!({
        "request_id": "demo-001",
        "property_id": "P-98765",
        "address": "123 Pine Ln, CA",
        "wildfire_risk_score": 92,
        "customer_account": "FIRM-42",
        "report_purpose": "underwriting",
        "governs_sections": ["precisely"]
    })
    .as_object()
    .expect("source fields are an object")
    .clone()
}

fn emit_rows(
    device_seed: &[u8; 32],
    governed_by: &str,
    routed_fields: &[&str],
    event_fields: Map<String, Value>,
    count: usize,
    seed_domain: &[u8],
    aad_policy_override: Option<&str>,
) -> Result<EmittedRows, FixtureError> {
    let source = tempfile::Builder::new()
        .prefix("tn-llama-source-")
        .tempdir()?;
    let keys = source.path().join(".tn").join("keys");
    let config_dir = source.path().join(".tn").join("config");
    fs::create_dir_all(&keys)?;
    fs::create_dir_all(&config_dir)?;

    let device = DeviceKey::from_private_bytes(device_seed)?;
    fs::write(keys.join("local.private"), device.private_bytes())?;
    let master = derive_fixture_secret(device_seed, b"index-master");
    fs::write(keys.join("index_master.key"), master)?;

    let mut receiver_kits = BTreeMap::new();
    for group in PROTECTED_GROUPS {
        let mut state = tn_btn::PublisherState::setup_with_seed(
            tn_btn::Config,
            derive_fixture_secret(
                device_seed,
                &[group.as_bytes(), b"-btn-", seed_domain].concat(),
            ),
        )?;
        let kit = state.mint()?.to_bytes();
        fs::write(keys.join(format!("{group}.btn.state")), state.to_bytes())?;
        fs::write(keys.join(format!("{group}.btn.mykit")), &kit)?;
        receiver_kits.insert(group.to_string(), kit);
    }
    fs::write(config_dir.join("agents.md"), POLICY_TEXT)?;

    let ceremony_id = format!(
        "cer_tn_llama_{}",
        &hex::encode(Sha256::digest(device_seed))[..16]
    );
    let routed = routed_fields.join(", ");
    let yaml = format!(
        "ceremony: {{id: {ceremony_id}, mode: local, cipher: btn, sign: true, chain: true, admin_log_location: main_log}}\n\
         logs: {{path: ./.tn/logs/tn.ndjson}}\n\
         keystore: {{path: ./.tn/keys}}\n\
         device: {{device_identity: \"{}\"}}\n\
         public_fields: []\n\
         default_policy: private\n\
         groups:\n\
         \x20 default:\n\
         \x20   policy: private\n\
         \x20   cipher: btn\n\
         \x20   recipients:\n\
         \x20     - {{recipient_identity: \"{}\"}}\n\
         \x20   index_epoch: 0\n\
         \x20   fields: [request_id]\n\
         \x20 firm_data:\n\
         \x20   policy: private\n\
         \x20   cipher: btn\n\
         \x20   recipients:\n\
         \x20     - {{recipient_identity: \"{}\"}}\n\
         \x20   index_epoch: 0\n\
         \x20   fields: [customer_account, report_purpose]\n\
         \x20 precisely:\n\
         \x20   policy: private\n\
         \x20   cipher: btn\n\
         \x20   recipients:\n\
         \x20     - {{recipient_identity: \"{}\"}}\n\
         \x20   index_epoch: 0\n\
         \x20   fields: [address, property_id, wildfire_risk_score]\n\
         \x20 \"tn.agents\":\n\
         \x20   policy: private\n\
         \x20   cipher: btn\n\
         \x20   recipients:\n\
         \x20     - {{recipient_identity: \"{}\"}}\n\
         \x20   index_epoch: 0\n\
         \x20   fields: [{routed}]\n\
         fields: {{}}\n\
         llm_classifier: {{enabled: false, provider: \"\", model: \"\"}}\n\
         handlers: [{{kind: file, rotate_on_init: false}}]\n",
        device.did(),
        device.did(),
        device.did(),
        device.did(),
        device.did(),
    );
    let yaml_path = source.path().join("tn.yaml");
    fs::write(&yaml_path, yaml)?;

    let config = tn_core::config::load(&yaml_path)?;
    for group in PROTECTED_GROUPS {
        let spec = config.groups.get(group).ok_or(FixtureError::Invariant(
            "fixture config omitted protected group",
        ))?;
        if spec.index_epoch != 0 {
            return Err(FixtureError::Invariant(
                "fixture protected group index epoch was not zero",
            ));
        }
    }
    let runtime = Runtime::init(&yaml_path)?;
    let policy = runtime
        .agent_policy_doc()
        .and_then(|document| document.templates.get(EVENT_TYPE))
        .ok_or(FixtureError::Invariant(
            "fixture runtime omitted source policy template",
        ))?;
    let generated_policy_ref = format!(
        "{}#{}@{}#{}",
        policy.path, policy.event_type, policy.version, policy.content_hash
    );
    let policy_ref = aad_policy_override.unwrap_or(&generated_policy_ref);
    let aad_overlay = Map::from_iter([
        (
            "governed_by".to_string(),
            Value::String(governed_by.to_string()),
        ),
        ("policy".to_string(), Value::String(policy_ref.to_string())),
    ]);
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let line = runtime
            .emit_with_aad_returning_line(
                "info",
                EVENT_TYPE,
                event_fields.clone(),
                None,
                None,
                Some(true),
                &aad_overlay,
            )?
            .ok_or(FixtureError::Invariant("fixture event was filtered"))?;
        let row: Value = serde_json::from_str(&line)?;
        rows.push(
            row.as_object()
                .cloned()
                .ok_or(FixtureError::Invariant("TN emit was not an object"))?,
        );
    }
    let aad_marker = rows
        .first()
        .and_then(|row| row.get("tn_aad"))
        .and_then(Value::as_str)
        .ok_or(FixtureError::Invariant("fixture row omitted tn_aad"))?;
    let aad_marker = serde_json::from_str::<Value>(aad_marker)?
        .as_object()
        .cloned()
        .ok_or(FixtureError::Invariant("fixture tn_aad was not an object"))?;

    Ok(EmittedRows {
        receiver_kits,
        input_index_key: master,
        ceremony_id,
        input_index_epoch: 0,
        device_identity: device.did().to_string(),
        rows,
        aad_marker,
    })
}

fn open_source_governance(
    row: &Map<String, Value>,
    source: &EmittedRows,
) -> Result<BTreeMap<String, Value>, FixtureError> {
    let reader_temp = tempfile::Builder::new()
        .prefix("tn-llama-source-reader-")
        .tempdir()?;
    fs::write(
        reader_temp.path().join("tn.agents.btn.mykit"),
        source
            .receiver_kits
            .get(GOVERNANCE_GROUP)
            .ok_or(FixtureError::Invariant(
                "fixture omitted governance reader kit",
            ))?,
    )?;
    let source_json = String::from_utf8(tn_core::canonical::canonical_bytes(&Value::Object(
        row.clone(),
    ))?)
    .map_err(|_| FixtureError::Invariant("canonical row is not UTF-8"))?;
    let opened =
        tn_core::unseal_as_recipient(&source_json, reader_temp.path(), GOVERNANCE_GROUP, true)?;
    let plaintext = opened
        .plaintext
        .get(GOVERNANCE_GROUP)
        .and_then(Value::as_object)
        .ok_or(FixtureError::Invariant("source could not reopen tn.agents"))?;
    if plaintext.len() != GOVERNANCE_FIELDS.len() {
        return Err(FixtureError::Invariant(
            "source governance did not contain the complete fixture field set",
        ));
    }
    Ok(plaintext
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect())
}

fn plaintext_variant(
    source: &Map<String, Value>,
    governance: &BTreeMap<String, Value>,
) -> Result<Map<String, Value>, FixtureError> {
    let mut row = source
        .iter()
        .filter(|(name, _)| ENVELOPE_RESERVED.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<Map<_, _>>();
    row.remove("tn_aad");
    row.insert("default".to_string(), json!({"request_id": "demo-001"}));
    row.insert(
        "precisely".to_string(),
        json!({
            "property_id": "P-98765",
            "address": "123 Pine Ln, CA",
            "wildfire_risk_score": 92
        }),
    );
    row.insert(
        "firm_data".to_string(),
        json!({
            "customer_account": "FIRM-42",
            "report_purpose": "underwriting"
        }),
    );
    row.insert(
        GOVERNANCE_GROUP.to_string(),
        Value::Object(governance.clone().into_iter().collect()),
    );
    if row.get("row_hash").and_then(Value::as_str).is_none() {
        return Err(FixtureError::Invariant(
            "plaintext variant source omitted row hash",
        ));
    }
    Ok(row)
}

fn official_rehash_and_sign(
    row: &mut Map<String, Value>,
    device: &DeviceKey,
) -> Result<(), FixtureError> {
    official_rehash(row)?;
    sign_current_row_hash(row, device)
}

fn official_plaintext_rehash_and_sign(
    row: &mut Map<String, Value>,
    device: &DeviceKey,
) -> Result<(), FixtureError> {
    official_plaintext_rehash(row)?;
    sign_current_row_hash(row, device)
}

fn sign_current_row_hash(
    row: &mut Map<String, Value>,
    device: &DeviceKey,
) -> Result<(), FixtureError> {
    let signature = signature_b64(&device.sign(header(row, "row_hash")?.as_bytes()));
    row.insert("signature".to_string(), Value::String(signature));
    Ok(())
}

fn official_rehash(row: &mut Map<String, Value>) -> Result<(), FixtureError> {
    let blocks = extract_group_blocks(row)?;
    let groups = blocks
        .iter()
        .map(|(name, block)| {
            (
                name.clone(),
                GroupInput {
                    ciphertext: block.ciphertext.clone(),
                    field_hashes: block.field_hashes.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let group_names = blocks.keys().cloned().collect::<BTreeSet<_>>();
    let public_fields = row
        .iter()
        .filter(|(name, _)| {
            !ENVELOPE_RESERVED.contains(&name.as_str()) && !group_names.contains(*name)
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let row_hash = compute_row_hash(&RowHashInput {
        device_identity: header(row, "device_identity")?,
        timestamp: header(row, "timestamp")?,
        event_id: header(row, "event_id")?,
        event_type: header(row, "event_type")?,
        level: header(row, "level")?,
        prev_hash: header(row, "prev_hash")?,
        public_fields: &public_fields,
        groups: &groups,
    });
    row.insert("row_hash".to_string(), Value::String(row_hash));
    Ok(())
}

fn official_plaintext_rehash(row: &mut Map<String, Value>) -> Result<(), FixtureError> {
    let public_fields = row
        .iter()
        .filter(|(name, _)| !ENVELOPE_RESERVED.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let row_hash = compute_row_hash(&RowHashInput {
        device_identity: header(row, "device_identity")?,
        timestamp: header(row, "timestamp")?,
        event_id: header(row, "event_id")?,
        event_type: header(row, "event_type")?,
        level: header(row, "level")?,
        prev_hash: header(row, "prev_hash")?,
        public_fields: &public_fields,
        groups: &BTreeMap::new(),
    });
    row.insert("row_hash".to_string(), Value::String(row_hash));
    Ok(())
}

fn ndjson_row(row: &Map<String, Value>) -> Result<Vec<u8>, FixtureError> {
    let mut line = serde_json::to_vec(row)?;
    line.push(b'\n');
    Ok(line)
}

fn wrap_request(row: &Map<String, Value>) -> Result<Vec<u8>, FixtureError> {
    wrap_request_with_prompt(row, "Reply with exactly READY.")
}

fn wrap_request_with_prompt(
    row: &Map<String, Value>,
    prompt: &str,
) -> Result<Vec<u8>, FixtureError> {
    Ok(serde_json::to_vec(&json!({
        "protocol": "tn.llama.request.v1",
        "tn_row": row,
        "messages": [{"role": "user", "content": prompt}],
        "purpose": "derived_html_report",
        "stream": false,
    }))?)
}

fn write_fixture(
    out: &Path,
    build: &BuildInputs,
    generated: &GeneratedFixture,
) -> Result<PathBuf, FixtureError> {
    fs::create_dir_all(out)?;
    let out = out.canonicalize()?;
    let receiver_key_dir = out.join("receiver");
    let requests_dir = out.join("requests");
    let sources_dir = out.join("sources");
    let raw_trace_dir = out.join("raw-trace");
    let output = prepare_output_ceremony(&out)?;
    fs::create_dir(&receiver_key_dir)?;
    fs::create_dir(&requests_dir)?;
    fs::create_dir(&sources_dir)?;
    fs::create_dir(&raw_trace_dir)?;

    let input_index_key_path = out.join("input.index.key");
    let row1_request_path = requests_dir.join("row1-request.json");
    let plaintext_report_request_path = requests_dir.join("plaintext-report-request.json");
    let plaintext_source_row1_path = sources_dir.join("plaintext-source-row1.ndjson");
    let plaintext_source_row2_path = sources_dir.join("plaintext-source-row2.ndjson");
    let row2_request_path = requests_dir.join("row2-request.json");
    let stale_fork_request_path = requests_dir.join("stale-fork-request.json");
    let signature_mutation_request_path = requests_dir.join("signature-mutation-request.json");
    let commitment_mutation_request_path = requests_dir.join("commitment-mutation-request.json");
    let aad_mutation_request_path = requests_dir.join("aad-mutation-request.json");
    let ciphertext_mutation_request_path = requests_dir.join("ciphertext-mutation-request.json");
    let missing_governance_field_request_path =
        requests_dir.join("missing-governance-field-request.json");
    let extra_governance_field_request_path =
        requests_dir.join("extra-governance-field-request.json");
    let extra_governance_commitment_mutation_request_path =
        requests_dir.join("extra-governance-commitment-mutation-request.json");
    let expected_governance_path = out.join("expected-governance.json");
    let runtime_config_path = out.join("runtime-config.json");
    let audit_path = out.join("audit.ndjson");

    for (group, kit) in &generated.receiver_kits {
        fs::write(receiver_key_dir.join(format!("{group}.btn.mykit")), kit)?;
    }
    fs::write(&input_index_key_path, generated.input_index_key)?;
    fs::write(&row1_request_path, &generated.row1_request_json)?;
    fs::write(
        &plaintext_report_request_path,
        &generated.plaintext_report_request_json,
    )?;
    fs::write(
        &plaintext_source_row1_path,
        &generated.plaintext_source_row1_ndjson,
    )?;
    fs::write(
        &plaintext_source_row2_path,
        &generated.plaintext_source_row2_ndjson,
    )?;
    fs::write(&row2_request_path, &generated.row2_request_json)?;
    fs::write(&stale_fork_request_path, &generated.stale_fork_request_json)?;
    fs::write(
        &signature_mutation_request_path,
        &generated.signature_mutation_request_json,
    )?;
    fs::write(
        &commitment_mutation_request_path,
        &generated.commitment_mutation_request_json,
    )?;
    fs::write(
        &aad_mutation_request_path,
        &generated.aad_mutation_request_json,
    )?;
    fs::write(
        &ciphertext_mutation_request_path,
        &generated.ciphertext_mutation_request_json,
    )?;
    fs::write(
        &missing_governance_field_request_path,
        &generated.missing_governance_field_request_json,
    )?;
    fs::write(
        &extra_governance_field_request_path,
        &generated.extra_governance_field_request_json,
    )?;
    fs::write(
        &extra_governance_commitment_mutation_request_path,
        &generated.extra_governance_commitment_mutation_request_json,
    )?;
    fs::write(
        &expected_governance_path,
        serde_json::to_vec(&generated.governance)?,
    )?;
    fs::write(&audit_path, [])?;

    let runtime_config = RuntimeConfigFile {
        version: "tn.llama.runtime.v2",
        writer_authority: "group_capability",
        expected_event_type: EVENT_TYPE,
        allowed_input_groups: &["default", "firm_data", "precisely"],
        report_contract: ReportContractFile {
            allowed_input_sections: &["default", "firm_data", "precisely"],
            output_event_type: "report.generated",
            output_public_section: "default",
            operation: "derived_html_report",
            section_rules: BTreeMap::from([(
                "precisely",
                SectionRuleFile {
                    propagate_source_governance: true,
                    required_attribution: Some(
                        "Source: Risk assessment derived from Precisely Wildfire Risk Data.",
                    ),
                    required_tool: Some("log_compliance_creation"),
                },
            )]),
            output_policy: OutputPolicyFile {
                instruction: "Generate only the configured derived report.",
                use_for: "Creating the governed HTML report requested by the user.",
                do_not_use_for: "Any ungoverned output or unrelated purpose.",
                consequences: "Reject the report and release no output when validation fails.",
                on_violation_or_error: "Abort the governed session and record the failing gate.",
            },
        },
        reader_key_dir: &receiver_key_dir,
        input_index_key_path: &input_index_key_path,
        ceremony_id: &generated.ceremony_id,
        governance_group: GOVERNANCE_GROUP,
        input_index_epoch: generated.input_index_epoch,
        output_tn_config_path: &output.config_path,
        output_reader_key_dir: &output.reader_key_dir,
        output_index_key_path: &output.index_key_path,
        output_ceremony_id: &output.ceremony_id,
        output_governance_group: GOVERNANCE_GROUP,
        output_index_epoch: output.index_epoch,
        audit_path: &audit_path,
        raw_trace_dir: &raw_trace_dir,
        build_id: "tn-fixture",
        llama_source_revision: &build.llama_source_revision,
        model_sha256: &build.model_sha256,
        binary_sha256: &build.binary_sha256,
        max_request_bytes: 1024 * 1024,
        max_message_bytes: 16 * 1024,
        max_active_sessions: 1,
        max_chain_namespaces: 8,
        echo_max_tokens: 1536,
        final_max_tokens: 2048,
    };
    fs::write(&runtime_config_path, serde_json::to_vec(&runtime_config)?)?;

    let manifest_path = out.join("manifest.json");
    let manifest = FixtureManifest {
        schema: "tn.llama.fixture.v1",
        runtime_config_path: &runtime_config_path,
        plaintext_report_request_path: &plaintext_report_request_path,
        plaintext_source_row1_path: &plaintext_source_row1_path,
        plaintext_source_row2_path: &plaintext_source_row2_path,
        row1_request_path: &row1_request_path,
        row2_request_path: &row2_request_path,
        stale_fork_request_path: &stale_fork_request_path,
        signature_mutation_request_path: &signature_mutation_request_path,
        expected_governance_path: &expected_governance_path,
        commitment_mutation_request_path: &commitment_mutation_request_path,
        aad_mutation_request_path: &aad_mutation_request_path,
        ciphertext_mutation_request_path: &ciphertext_mutation_request_path,
        missing_governance_field_request_path: &missing_governance_field_request_path,
        extra_governance_field_request_path: &extra_governance_field_request_path,
        extra_governance_commitment_mutation_request_path:
            &extra_governance_commitment_mutation_request_path,
        audit_path: &audit_path,
        raw_trace_dir: &raw_trace_dir,
    };
    fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
    Ok(manifest_path)
}

#[derive(Serialize)]
struct FixtureManifest<'a> {
    schema: &'static str,
    runtime_config_path: &'a Path,
    plaintext_report_request_path: &'a Path,
    plaintext_source_row1_path: &'a Path,
    plaintext_source_row2_path: &'a Path,
    row1_request_path: &'a Path,
    row2_request_path: &'a Path,
    stale_fork_request_path: &'a Path,
    signature_mutation_request_path: &'a Path,
    expected_governance_path: &'a Path,
    commitment_mutation_request_path: &'a Path,
    aad_mutation_request_path: &'a Path,
    ciphertext_mutation_request_path: &'a Path,
    missing_governance_field_request_path: &'a Path,
    extra_governance_field_request_path: &'a Path,
    extra_governance_commitment_mutation_request_path: &'a Path,
    audit_path: &'a Path,
    raw_trace_dir: &'a Path,
}

#[derive(Serialize)]
struct RuntimeConfigFile<'a> {
    version: &'static str,
    writer_authority: &'static str,
    expected_event_type: &'static str,
    allowed_input_groups: &'static [&'static str],
    report_contract: ReportContractFile<'a>,
    reader_key_dir: &'a Path,
    input_index_key_path: &'a Path,
    ceremony_id: &'a str,
    governance_group: &'static str,
    input_index_epoch: u64,
    output_tn_config_path: &'a Path,
    output_reader_key_dir: &'a Path,
    output_index_key_path: &'a Path,
    output_ceremony_id: &'a str,
    output_governance_group: &'static str,
    output_index_epoch: u64,
    audit_path: &'a Path,
    raw_trace_dir: &'a Path,
    build_id: &'static str,
    llama_source_revision: &'a str,
    model_sha256: &'a str,
    binary_sha256: &'a str,
    max_request_bytes: usize,
    max_message_bytes: usize,
    max_active_sessions: usize,
    max_chain_namespaces: usize,
    echo_max_tokens: u32,
    final_max_tokens: u32,
}

#[derive(Serialize)]
struct ReportContractFile<'a> {
    allowed_input_sections: &'static [&'static str],
    output_event_type: &'static str,
    output_public_section: &'static str,
    operation: &'static str,
    section_rules: BTreeMap<&'static str, SectionRuleFile<'a>>,
    output_policy: OutputPolicyFile,
}

#[derive(Serialize)]
struct SectionRuleFile<'a> {
    propagate_source_governance: bool,
    required_attribution: Option<&'a str>,
    required_tool: Option<&'a str>,
}

#[derive(Serialize)]
struct OutputPolicyFile {
    instruction: &'static str,
    use_for: &'static str,
    do_not_use_for: &'static str,
    consequences: &'static str,
    on_violation_or_error: &'static str,
}

struct OutputCeremony {
    config_path: PathBuf,
    reader_key_dir: PathBuf,
    index_key_path: PathBuf,
    ceremony_id: String,
    index_epoch: u64,
}

fn prepare_output_ceremony(out: &Path) -> Result<OutputCeremony, FixtureError> {
    let root = out.join("output-writer");
    let keys = root.join(".tn").join("keys");
    let config_dir = root.join(".tn").join("config");
    let logs = root.join(".tn").join("logs");
    let reader_key_dir = out.join("output-reader");
    fs::create_dir_all(&keys)?;
    fs::create_dir_all(&config_dir)?;
    fs::create_dir_all(&logs)?;
    fs::create_dir_all(&reader_key_dir)?;

    let device = DeviceKey::generate();
    let device_seed = device.private_bytes();
    fs::write(keys.join("local.private"), device_seed)?;
    let master = derive_fixture_secret(&device_seed, b"output-index-master");
    fs::write(keys.join("index_master.key"), master)?;

    let mut default_state = tn_btn::PublisherState::setup_with_seed(
        tn_btn::Config,
        derive_fixture_secret(&device_seed, b"output-default-btn"),
    )?;
    let default_kit = default_state.mint()?;
    fs::write(keys.join("default.btn.state"), default_state.to_bytes())?;
    fs::write(keys.join("default.btn.mykit"), default_kit.to_bytes())?;
    fs::write(
        reader_key_dir.join("default.btn.mykit"),
        default_kit.to_bytes(),
    )?;

    let mut agents_state = tn_btn::PublisherState::setup_with_seed(
        tn_btn::Config,
        derive_fixture_secret(&device_seed, b"output-agents-btn"),
    )?;
    let agents_kit = agents_state.mint()?;
    fs::write(keys.join("tn.agents.btn.state"), agents_state.to_bytes())?;
    fs::write(keys.join("tn.agents.btn.mykit"), agents_kit.to_bytes())?;
    fs::write(
        reader_key_dir.join("tn.agents.btn.mykit"),
        agents_kit.to_bytes(),
    )?;
    fs::write(config_dir.join("agents.md"), OUTPUT_POLICY_TEXT)?;

    let ceremony_id = format!(
        "cer_tn_llama_output_{}",
        &hex::encode(Sha256::digest(device_seed))[..16]
    );
    let yaml = format!(
        "ceremony: {{id: {ceremony_id}, mode: local, cipher: btn, sign: true, chain: true, admin_log_location: main_log}}\n\
         logs: {{path: ./.tn/logs/tn.ndjson}}\n\
         keystore: {{path: ./.tn/keys}}\n\
         device: {{device_identity: \"{}\"}}\n\
         public_fields: []\n\
         default_policy: private\n\
         groups:\n\
         \x20 default:\n\
         \x20   policy: private\n\
         \x20   cipher: btn\n\
         \x20   recipients:\n\
         \x20     - {{recipient_identity: \"{}\"}}\n\
         \x20   index_epoch: 0\n\
         \x20   fields: [default]\n\
         \x20 \"tn.agents\":\n\
         \x20   policy: private\n\
         \x20   cipher: btn\n\
         \x20   recipients:\n\
         \x20     - {{recipient_identity: \"{}\"}}\n\
         \x20   index_epoch: 0\n\
         \x20   fields: [instruction, use_for, do_not_use_for, consequences, on_violation_or_error, policy, source_lineage, compliance_evidence]\n\
         fields: {{}}\n\
         llm_classifier: {{enabled: false, provider: \"\", model: \"\"}}\n\
         handlers: [{{kind: file, rotate_on_init: false}}]\n",
        device.did(),
        device.did(),
        device.did(),
    );
    let config_path = root.join("tn.yaml");
    fs::write(&config_path, yaml)?;
    let config = tn_core::config::load(&config_path)?;
    for group in ["default", GOVERNANCE_GROUP] {
        let spec = config.groups.get(group).ok_or(FixtureError::Invariant(
            "output config omitted protected group",
        ))?;
        if spec.index_epoch != 0 {
            return Err(FixtureError::Invariant(
                "output protected group index epoch was not zero",
            ));
        }
    }
    let index_key_path = out.join("output.index.key");
    fs::write(&index_key_path, master)?;

    Ok(OutputCeremony {
        config_path,
        reader_key_dir,
        index_key_path,
        ceremony_id,
        index_epoch: 0,
    })
}

fn prepare_output_directory(out: &Path) -> Result<(), FixtureError> {
    if !out.is_absolute() {
        return Err(FixtureError::OutputNotAbsolute);
    }
    match fs::symlink_metadata(out) {
        Ok(metadata) => {
            if is_reparse_target(&metadata) {
                return Err(FixtureError::OutputReparseTarget);
            }
            if !metadata.is_dir() {
                return Err(FixtureError::OutputNotDirectory);
            }
            if fs::read_dir(out)?.next().transpose()?.is_some() {
                return Err(FixtureError::OutputNotEmpty);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(out)?;
        }
        Err(error) => return Err(error.into()),
    }
    let metadata = fs::symlink_metadata(out)?;
    if is_reparse_target(&metadata) {
        return Err(FixtureError::OutputReparseTarget);
    }
    Ok(())
}

#[cfg(windows)]
fn is_reparse_target(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    metadata.file_type().is_symlink() || metadata.file_attributes() & 0x400 != 0
}

#[cfg(not(windows))]
fn is_reparse_target(metadata: &fs::Metadata) -> bool {
    metadata.file_type().is_symlink()
}

fn derive_fixture_secret(seed: &[u8; 32], domain: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"tn.llama.fixture-secret.v1\0");
    hasher.update(domain);
    hasher.update(seed);
    hasher.finalize().into()
}

fn require_lower_hex(name: &str, value: &str, expected_len: usize) -> Result<(), FixtureError> {
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FixtureError::ArgumentInvalid(format!(
            "{name} must be exactly {expected_len} lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn header<'a>(row: &'a Map<String, Value>, name: &'static str) -> Result<&'a str, FixtureError> {
    row.get(name)
        .and_then(Value::as_str)
        .ok_or(FixtureError::Invariant("TN emitted a non-string header"))
}

fn changed_identifier(value: &str) -> String {
    let mut changed = value.to_string();
    if let Some(last) = changed.pop() {
        changed.push(if last == 'f' { 'e' } else { 'f' });
    }
    changed
}

fn flip_terminal_hex(value: &str) -> Result<String, FixtureError> {
    let mut changed = value.to_string();
    let last = changed
        .pop()
        .ok_or(FixtureError::Invariant("empty commitment token"))?;
    if !last.is_ascii_hexdigit() {
        return Err(FixtureError::Invariant(
            "commitment token did not end in hexadecimal",
        ));
    }
    changed.push(if last == 'f' { 'e' } else { 'f' });
    Ok(changed)
}
