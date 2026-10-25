mod support;

use std::collections::BTreeSet;
use std::fs;
use std::sync::Arc;

use serde_json::{json, Value};
use tn_core::sealed_object::{extract_group_blocks, require_envelope_shape, verify_sealed};
use tn_runtime::{OutputPublisher, RegisteredCall, ToolContext, ToolDispatcher, ToolReceipt};

use support::fixture_config;

#[test]
fn registry_exposes_the_configured_name_and_closed_model_schema() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let dispatcher = ToolDispatcher::new(
        Arc::new(OutputPublisher::from_config(&config).unwrap()),
        "configured-compliance",
    )
    .unwrap();

    let specs = dispatcher.specifications().collect::<Vec<_>>();
    assert_eq!(specs.len(), 1);
    let spec = specs[0];
    assert_eq!(spec.name(), "configured-compliance");
    assert_eq!(
        spec.description(),
        "Log creation of a governed derived report and return its canonical TN receipt."
    );
    assert_eq!(
        spec.parameters(),
        &json!({
            "type": "object",
            "properties": {
                "source_event_id": {"type": "string"},
                "action": {"type": "string"}
            },
            "required": ["source_event_id", "action"],
            "additionalProperties": false
        })
    );
    assert_eq!(
        dispatcher.specification("configured-compliance").unwrap(),
        spec
    );
    assert!(dispatcher.specification("other").is_none());
}

#[test]
fn configured_compliance_handler_emits_bound_row_and_exact_receipt() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let publisher = Arc::new(OutputPublisher::from_config(&config).unwrap());
    let dispatcher = ToolDispatcher::new(publisher, "log_compliance_creation").unwrap();
    let source_row: Value = serde_json::from_slice(&fixture.row1_request_json).unwrap();
    let source = &source_row["tn_row"];
    let context = ToolContext {
        run_id: "run-019f".to_string(),
        source_device_identity: source["device_identity"].as_str().unwrap().to_string(),
        source_event_id: source["event_id"].as_str().unwrap().to_string(),
        source_event_type: source["event_type"].as_str().unwrap().to_string(),
        source_row_hash: source["row_hash"].as_str().unwrap().to_string(),
        selected_sections: BTreeSet::from(["precisely".to_string(), "default".to_string()]),
        action: "derived_html_report".to_string(),
    };
    let call = RegisteredCall {
        id: "model-call-7".to_string(),
        name: "log_compliance_creation".to_string(),
        arguments: serde_json::to_vec(&json!({
            "source_event_id": context.source_event_id,
            "action": "derived_html_report"
        }))
        .unwrap(),
    };

    let ToolReceipt::Compliance(receipt) = dispatcher.execute(&call, &context).unwrap();

    let receipt_json = serde_json::to_value(&receipt).unwrap();
    assert_eq!(
        receipt_json
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "action".to_string(),
            "event_id".to_string(),
            "event_type".to_string(),
            "row_hash".to_string(),
            "source_row_hash".to_string(),
        ])
    );
    assert_eq!(receipt.event_type, "compliance.creation.logged");
    assert_eq!(receipt.source_row_hash, context.source_row_hash);
    assert_eq!(receipt.action, "derived_html_report");

    let log_path = config
        .output_tn_config_path
        .parent()
        .unwrap()
        .join(".tn")
        .join("logs")
        .join("tn.ndjson");
    let log = fs::read_to_string(log_path).unwrap();
    let line = log.lines().last().unwrap();
    let parsed: Value = serde_json::from_str(line).unwrap();
    assert_eq!(parsed["event_id"], receipt.event_id);
    assert_eq!(parsed["row_hash"], receipt.row_hash);
    assert!(parsed["default"].get("source_event_id").is_none());
    let opened_default =
        tn_core::unseal_as_recipient(line, &config.output_reader_key_dir, "default", true)
            .unwrap()
            .plaintext;
    let payload = &opened_default["default"]["default"];
    assert_eq!(payload["source_event_id"], context.source_event_id);
    assert_eq!(payload["source_row_hash"], context.source_row_hash);
    assert_eq!(payload["model_tool_call_id"], "model-call-7");
    assert_eq!(payload["run_id"], "run-019f");
    assert_eq!(payload["tool_name"], "log_compliance_creation");
    assert_eq!(
        payload["selected_sections"],
        json!(["default", "precisely"])
    );
    assert_eq!(payload["action"], "derived_html_report");
    let envelope = require_envelope_shape(parsed.as_object().unwrap().clone()).unwrap();
    let blocks = extract_group_blocks(&envelope).unwrap();
    assert!(verify_sealed(&envelope, &blocks).row_hash);
    let opened =
        tn_core::unseal_as_recipient(line, &config.output_reader_key_dir, "tn.agents", true)
            .unwrap();
    let policy = opened.plaintext["tn.agents"]["policy"].as_str().unwrap();
    assert!(policy.contains("#compliance.creation.logged@1#sha256:"));
    let aad_raw = parsed["tn_aad"].as_str().unwrap();
    let aad: Value = serde_json::from_str(aad_raw).unwrap();
    assert_eq!(
        tn_core::canonical::canonical_bytes(&aad).unwrap(),
        aad_raw.as_bytes()
    );
    let marker = json!({
        "governed_by": parsed["device_identity"],
        "policy": policy,
    });
    assert_eq!(aad["default"], marker);
    assert_eq!(aad["tn.agents"], marker);
}

#[test]
fn dispatcher_rejects_unregistered_names_and_nonexact_arguments_without_emitting() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let dispatcher = ToolDispatcher::new(
        Arc::new(OutputPublisher::from_config(&config).unwrap()),
        "configured_compliance_tool",
    )
    .unwrap();
    let context = ToolContext {
        run_id: "run-019f".to_string(),
        source_device_identity: fixture.device_identity.clone(),
        source_event_id: "source-1".to_string(),
        source_event_type: "precisely.risk.enriched".to_string(),
        source_row_hash: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_string(),
        selected_sections: BTreeSet::from(["firm_data".to_string()]),
        action: "derived_html_report".to_string(),
    };
    let arguments = br#"{"source_event_id":"source-1","action":"derived_html_report"}"#.to_vec();

    let unregistered = dispatcher
        .execute(
            &RegisteredCall {
                id: "call-1".to_string(),
                name: "log_compliance_creation".to_string(),
                arguments: arguments.clone(),
            },
            &context,
        )
        .unwrap_err();
    assert_eq!(unregistered.code(), "tn_tool_not_registered");

    for invalid in [
        br#"{"source_event_id":"other","action":"derived_html_report"}"#.to_vec(),
        br#"{"source_event_id":"source-1","action":"other"}"#.to_vec(),
        br#"{"source_event_id":"source-1","action":"derived_html_report","extra":true}"#
            .to_vec(),
        br#"{"source_event_id":"source-1","source_event_id":"source-1","action":"derived_html_report"}"#
            .to_vec(),
    ] {
        let error = dispatcher
            .execute(
                &RegisteredCall {
                    id: "call-2".to_string(),
                    name: "configured_compliance_tool".to_string(),
                    arguments: invalid,
                },
                &context,
            )
            .unwrap_err();
        assert_eq!(error.code(), "tn_tool_arguments_invalid");
    }
}
