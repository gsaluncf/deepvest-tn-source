#![allow(dead_code)]
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};
use std::fs;
use tn_runtime::{ActionOutcome, GovernedRuntime, ModelTurn, RuntimeConfig, Session, TnLlamaError};

pub struct Fixture {
    pub legacy: tn_test_tools::RealTnFixture,
    pub manifest: Value,
    pub config: RuntimeConfig,
    pub runtime_json: Vec<u8>,
}
impl Fixture {
    pub fn require_compliance(&mut self) {
        self.config.report_contract.section_rules.insert(
            "finance".into(),
            tn_runtime::SectionRule {
                propagate_source_governance: true,
                required_attribution: None,
                required_tool: Some("log_compliance_creation".into()),
            },
        );
        let mut config: Value = serde_json::from_slice(&self.runtime_json).unwrap();
        config["report_contract"]["section_rules"] = json!({"finance":{
            "propagate_source_governance":true, "required_attribution":null,
            "required_tool":"log_compliance_creation"
        }});
        self.runtime_json = serde_json::to_vec(&config).unwrap();
    }
    pub fn new() -> Self {
        let legacy = tn_test_tools::build_real_tn_fixture().unwrap();
        let manifest: Value =
            serde_json::from_str(include_str!("../fixtures/governed/m1-manifest.json")).unwrap();
        let input = legacy.temp.path().join("native-input");
        let keys = input.join("keys");
        let reader = input.join("reader");
        fs::create_dir_all(&keys).unwrap();
        fs::create_dir_all(&reader).unwrap();
        fs::write(
            keys.join("local.private"),
            hex::decode(manifest["private_seed_hex"].as_str().unwrap()).unwrap(),
        )
        .unwrap();
        fs::write(
            keys.join("index_master.key"),
            hex::decode(manifest["index_master_hex"].as_str().unwrap()).unwrap(),
        )
        .unwrap();
        for (name, group) in manifest["groups"].as_object().unwrap() {
            let kit = STANDARD
                .decode(group["reader_kit_base64"].as_str().unwrap())
                .unwrap();
            fs::write(keys.join(format!("{name}.btn.mykit")), &kit).unwrap();
            fs::write(reader.join(format!("{name}.btn.mykit")), &kit).unwrap();
            fs::write(
                keys.join(format!("{name}.btn.state")),
                STANDARD
                    .decode(group["publisher_state_base64"].as_str().unwrap())
                    .unwrap(),
            )
            .unwrap();
        }
        let native = json!({"ceremony":{"id":"cer_governed_m1_test_fixture","cipher":"btn","mode":"local"},
            "keystore":{"path":"keys"},"device":{"device_identity":manifest["writer"]},
            "groups":manifest["groups"].as_object().unwrap().keys().map(|name| (name.clone(),json!({"cipher":"btn","policy":"private","index_epoch":0}))).collect::<serde_json::Map<_,_>>()});
        fs::write(input.join("tn.yaml"), native.to_string()).unwrap();
        let publications = &manifest["publications"];
        let metadata = json!({
            "policy_revisions":[publications["policy_earlier"]["wire"],publications["policy_later"]["wire"]],
            "dataset_editions":[publications["edition_earlier"]["wire"],publications["edition_later"]["wire"]],
            "publications":[publications["source_earlier"]["wire"],publications["source_later"]["wire"],publications["joined"]["wire"]]
        });
        fs::write(input.join("metadata.json"), metadata.to_string()).unwrap();
        let mut config = super::support::fixture_config(&legacy);
        config.expected_event_type = "deepvest.calculation".into();
        config.allowed_input_groups = ["finance".into(), "audit".into()].into();
        config.report_contract.allowed_input_sections = ["finance".into()].into();
        config.report_contract.section_rules.clear();
        config.report_contract.output_event_type = "portfolio.report".into();
        config.report_contract.operation = "generate_report".into();
        config.reader_key_dir = reader;
        config.input_index_key_path = keys.join("index_master.key");
        config.ceremony_id = "cer_governed_m1_test_fixture".into();
        let profile = json!({
            "native_reader_config_path":input.join("tn.yaml"), "metadata_store_path":input.join("metadata.json"),
            "use_context":manifest["uses"]["model"], "destinations":["internal-client"],
            "trusted_writers":[manifest["writer"]], "authority_writers":[manifest["writer"]],
            "contracts":(["policy_earlier","policy_later"].iter().map(|name| json!({
                "revision_id":publications[*name]["object_id"],"scope":"finance.wealth_path","allowed_uses":[manifest["uses"]["model"]]
            })).collect::<Vec<_>>())
        });
        config.governed_objects = Some(serde_json::from_value(profile.clone()).unwrap());
        let mut runtime_json: Value = serde_json::from_slice(
            &fs::read(legacy.temp.path().join("runtime-config.json")).unwrap(),
        )
        .unwrap();
        runtime_json["governed_objects"] = profile;
        runtime_json["expected_event_type"] = json!(config.expected_event_type);
        runtime_json["allowed_input_groups"] = json!(config.allowed_input_groups);
        runtime_json["reader_key_dir"] = json!(config.reader_key_dir);
        runtime_json["input_index_key_path"] = json!(config.input_index_key_path);
        runtime_json["ceremony_id"] = json!(config.ceremony_id);
        runtime_json["report_contract"]["allowed_input_sections"] = json!(["finance"]);
        runtime_json["report_contract"]["section_rules"] = json!({});
        runtime_json["report_contract"]["output_event_type"] =
            json!(config.report_contract.output_event_type);
        runtime_json["report_contract"]["operation"] = json!(config.report_contract.operation);
        Self {
            legacy,
            manifest,
            config,
            runtime_json: serde_json::to_vec(&runtime_json).unwrap(),
        }
    }
    pub fn request(&self, publication: &str) -> Vec<u8> {
        serde_json::to_vec(&json!({"protocol":"tn.llama.request.v2", "tn_object":self.manifest["publications"][publication]["wire"],
            "use":self.manifest["uses"]["model"],"destination":"internal-client", "messages":[{"role":"user","content":"Explain the maximum drawdown."}],"stream":false})).unwrap()
    }
    pub fn runtime(self) -> (tn_test_tools::RealTnFixture, GovernedRuntime) {
        (
            self.legacy,
            GovernedRuntime::from_config(self.config).unwrap(),
        )
    }
}

pub fn accept(
    session: &mut Session,
    turn: ModelTurn,
    name: &str,
    arguments: Value,
) -> Result<ActionOutcome, TnLlamaError> {
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    super::support::prepare_and_begin(
        &turn,
        b"native-fixture-rendered",
        body["max_tokens"].as_u64().unwrap() as u32,
        11,
    );
    let result = super::support::OwnedToolResult {
        id: format!("native-{name}").into_bytes(),
        name: name.as_bytes().to_vec(),
        arguments: serde_json::to_vec(&arguments).unwrap(),
        raw: b"native fixture tool result".to_vec(),
        content: vec![],
        call_count: 1,
        truncated: false,
        stop_limit: false,
    };
    session.accept_model_action(turn.permit, result.native())
}
pub fn get_governance(session: &mut Session) -> Result<(), TnLlamaError> {
    let turn = session.next_model_turn()?;
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    assert!(!String::from_utf8_lossy(&turn.request_json).contains("wealth_path"));
    let control: Value =
        serde_json::from_str(body["messages"][1]["content"].as_str().unwrap()).unwrap();
    accept(
        session,
        turn,
        "get_governance",
        json!({"row_hash":control["row_hash"],"request_digest":control["request_digest"]}),
    )?;
    Ok(())
}
pub fn echo(session: &mut Session) -> Result<(), TnLlamaError> {
    let turn = session.next_model_turn()?;
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    let args: Value = serde_json::from_str(
        body["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    accept(session, turn, "continue_with_governance", args)?;
    Ok(())
}
pub fn select(session: &mut Session, groups: &[&str]) -> Result<(), TnLlamaError> {
    let turn = session.next_model_turn()?;
    accept(session, turn, "select_sections", json!({"sections":groups}))?;
    Ok(())
}
