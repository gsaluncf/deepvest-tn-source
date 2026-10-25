mod support;

use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value};
use tn_core::sealed_object::{extract_group_blocks, require_envelope_shape, verify_sealed};
use tn_runtime::{
    parse_and_verify_request, ComplianceReceipt, OutputPublisher, RegisteredCall, ReportProposal,
    RequestLimits, SourceLineage, ToolContext, ToolDispatcher, ToolReceipt, TrustedDerivation,
};

use support::fixture_config;

const SOURCE_POLICY: &str = "fixture-policy#precisely.risk.enriched@1#sha256:source";
const REQUEST_DIGEST: &str =
    "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const REPORT_HTML: &str = "<article><p>Score 92.</p><p>Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></article>";

#[test]
fn report_emit_is_signed_encrypted_aad_bound_and_strictly_reopened() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let source = parse_and_verify_request(
        &fixture.row1_request_json,
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
    let source_hash = source.row.row_hash().to_string();
    let lineage = SourceLineage {
        source_device_identity: source.row.device_identity().to_string(),
        source_event_id: source.row.event_id().to_string(),
        source_event_type: source.row.event_type().to_string(),
        derived_from_row_hash: source_hash.clone(),
        source_policy: SOURCE_POLICY.to_string(),
        selected_sections: vec!["precisely".to_string()],
        operation: "derived_html_report".to_string(),
    };
    let evidence = ComplianceReceipt {
        event_id: "019f-compliance".to_string(),
        event_type: "compliance.creation.logged".to_string(),
        row_hash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        source_row_hash: source_hash.clone(),
        action: "derived_html_report".to_string(),
    };
    let proposal: ReportProposal = serde_json::from_value(json!({
        "event_type": "report.generated",
        "public_sections": {
            "default": {"report_html": REPORT_HTML}
        },
        "governance": {
            "instruction": "Generate only the configured derived report.",
            "use_for": "Creating the governed HTML report requested by the user.",
            "do_not_use_for": "Any ungoverned output or unrelated purpose.",
            "consequences": "Reject the report and release no output when validation fails.",
            "on_violation_or_error": "Abort the governed session and record the failing gate.",
            "source_lineage": lineage,
            "compliance_evidence": evidence
        }
    }))
    .unwrap();
    let trusted = TrustedDerivation {
        contract: config.report_contract.clone(),
        request_digest: source.request_digest.clone(),
        source_lineage: Some(lineage.clone()),
        compliance_evidence: Some(evidence.clone()),
    };
    let publisher = OutputPublisher::from_config(&config).unwrap();

    let verified = publisher.emit_report(&proposal, &trusted).unwrap();

    assert!(verified.ndjson.ends_with('\n'));
    assert_eq!(verified.event_type, "report.generated");
    let parsed: Value = serde_json::from_str(verified.ndjson.trim_end()).unwrap();
    let envelope = require_envelope_shape(parsed.as_object().unwrap().clone()).unwrap();
    let blocks = extract_group_blocks(&envelope).unwrap();
    assert_eq!(
        blocks.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from(["default".to_string(), "tn.agents".to_string()])
    );
    let sealed = verify_sealed(&envelope, &blocks);
    assert!(sealed.row_hash);
    assert!(sealed.signature);
    assert!(parsed["signature"]
        .as_str()
        .is_some_and(|value| !value.is_empty()));
    assert!(parsed["default"].get("ciphertext").is_some());
    assert!(parsed["tn.agents"].get("ciphertext").is_some());
    assert!(!verified.ndjson.contains(REPORT_HTML));
    assert!(!verified
        .ndjson
        .contains("Generate only the configured derived report."));

    let aad_raw = parsed["tn_aad"].as_str().expect("canonical AAD echo");
    let aad: Value = serde_json::from_str(aad_raw).unwrap();
    assert_eq!(
        tn_core::canonical::canonical_bytes(&aad).unwrap(),
        aad_raw.as_bytes()
    );
    let output_did = parsed["device_identity"].as_str().unwrap();
    let policy_ref = verified.governance.plaintext["policy"].as_str().unwrap();
    for group in ["default", "tn.agents"] {
        assert_eq!(
            aad[group],
            json!({
                "governed_by": output_did,
                "policy": policy_ref,
            })
        );
    }

    let mut tampered = parsed.clone();
    let mut tampered_aad = aad.clone();
    tampered_aad["default"]["policy"] = json!("policy:swapped#sha256:bad");
    tampered["tn_aad"] = Value::String(
        String::from_utf8(tn_core::canonical::canonical_bytes(&tampered_aad).unwrap()).unwrap(),
    );
    let tampered_wire =
        String::from_utf8(tn_core::canonical::canonical_bytes(&tampered).unwrap()).unwrap();
    let tampered_open = tn_core::unseal_as_recipient(
        &tampered_wire,
        &config.output_reader_key_dir,
        "default",
        false,
    )
    .unwrap();
    assert!(!tampered_open.plaintext.contains_key("default"));

    let opened = tn_core::unseal_as_recipient(
        verified.ndjson.trim_end(),
        &config.output_reader_key_dir,
        "tn.agents",
        true,
    )
    .unwrap()
    .plaintext;
    assert_eq!(
        opened["tn.agents"]["source_lineage"]["derived_from_row_hash"],
        source_hash
    );
    assert_eq!(
        opened["tn.agents"]["compliance_evidence"]["event_id"],
        "019f-compliance"
    );
    assert!(opened["tn.agents"]["policy"]
        .as_str()
        .unwrap()
        .contains("#report.generated@1#sha256:"));
    let committed = verified
        .governance
        .commitments
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let opened_fields = opened["tn.agents"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(committed, opened_fields);
    assert_eq!(
        verified.governance.plaintext["source_lineage"],
        serde_json::to_value(&lineage).unwrap()
    );
    assert_eq!(
        verified.governance.plaintext["compliance_evidence"],
        serde_json::to_value(&evidence).unwrap()
    );
    assert_eq!(
        verified
            .default
            .plaintext
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["default".to_string(), "run_id".to_string()])
    );
    assert_eq!(
        verified.default.plaintext["default"]["report_html"],
        REPORT_HTML
    );

    let opened_default = tn_core::unseal_as_recipient(
        verified.ndjson.trim_end(),
        &config.output_reader_key_dir,
        "default",
        true,
    )
    .unwrap()
    .plaintext;
    assert_eq!(
        opened_default["default"]["default"]["report_html"],
        REPORT_HTML
    );

    let master = std::fs::read(&config.output_index_key_path).unwrap();
    let writer_config = tn_core::config::load(&config.output_tn_config_path).unwrap();
    for (group, plaintext) in [
        ("default", &opened_default["default"]),
        ("tn.agents", &opened["tn.agents"]),
    ] {
        let group_key = tn_core::indexing::derive_group_index_key(
            &master,
            &writer_config.ceremony.id,
            group,
            writer_config.groups[group].index_epoch,
        )
        .unwrap();
        let plaintext = plaintext.as_object().unwrap();
        let hashes = blocks[group].field_hashes.clone();
        assert_eq!(
            plaintext.keys().cloned().collect::<BTreeSet<_>>(),
            hashes.keys().cloned().collect::<BTreeSet<_>>()
        );
        for (field, value) in plaintext {
            assert_eq!(
                tn_core::indexing::index_token(&group_key, field, value).unwrap(),
                hashes[field]
            );
        }
    }
}

#[test]
fn report_proposal_is_closed_and_rejects_server_owned_fields() {
    let base = json!({
        "event_type": "report.generated",
        "public_sections": {"default": {"report_html": "<p>ok</p>"}},
        "governance": {
            "instruction": "i",
            "use_for": "u",
            "do_not_use_for": "d",
            "consequences": "c",
            "on_violation_or_error": "o"
        }
    });
    assert!(serde_json::from_value::<ReportProposal>(base.clone()).is_ok());

    for changed in [
        with_member(&base, "row_hash", json!("sha256:forged")),
        with_member(&base, "policy", json!("policy/forged")),
        with_member(&base, "extra", json!(true)),
        json!({
            "event_type": "report.generated",
            "public_sections": {
                "default": {"report_html": "<p>ok</p>"},
                "precisely": {"wildfire_risk_score": 92}
            },
            "governance": base["governance"]
        }),
    ] {
        assert!(serde_json::from_value::<ReportProposal>(changed).is_err());
    }

    for server_owned in ["policy", "row_hash", "field_hashes", "signature"] {
        let mut changed = base.clone();
        changed["governance"][server_owned] = json!("forged");
        assert!(serde_json::from_value::<ReportProposal>(changed).is_err());
    }
}

#[test]
fn report_emission_rejects_policy_lineage_and_evidence_not_owned_by_the_server() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let publisher = OutputPublisher::from_config(&config).unwrap();
    let trusted = TrustedDerivation {
        contract: config.report_contract.clone(),
        request_digest: REQUEST_DIGEST.to_string(),
        source_lineage: None,
        compliance_evidence: None,
    };
    let base = exact_report_proposal("<p>ok</p>", None, None);

    let mut wrong_event = base.clone();
    wrong_event.event_type = "report.other".to_string();
    assert_eq!(
        publisher
            .emit_report(&wrong_event, &trusted)
            .unwrap_err()
            .code(),
        "tn_report_proposal_invalid"
    );

    let mut wrong_policy = base.clone();
    wrong_policy.governance.instruction = "Forged policy".to_string();
    assert_eq!(
        publisher
            .emit_report(&wrong_policy, &trusted)
            .unwrap_err()
            .code(),
        "tn_report_proposal_invalid"
    );

    let mut invented_lineage = base.clone();
    invented_lineage.governance.source_lineage = Some(literal_lineage(
        &fixture.device_identity,
        "source-1",
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ));
    assert_eq!(
        publisher
            .emit_report(&invented_lineage, &trusted)
            .unwrap_err()
            .code(),
        "tn_report_proposal_invalid"
    );

    let mut invented_evidence = base;
    invented_evidence.governance.compliance_evidence = Some(ComplianceReceipt {
        event_id: "compliance-1".to_string(),
        event_type: "compliance.creation.logged".to_string(),
        row_hash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        source_row_hash: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_string(),
        action: "derived_html_report".to_string(),
    });
    assert_eq!(
        publisher
            .emit_report(&invented_evidence, &trusted)
            .unwrap_err()
            .code(),
        "tn_report_proposal_invalid"
    );
}

#[test]
fn reopened_report_policy_must_equal_the_configured_output_contract() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let policy_path = config
        .output_tn_config_path
        .parent()
        .unwrap()
        .join(".tn")
        .join("config")
        .join("agents.md");
    let changed_policy = include_str!("../../../demo/policies/output-agents.txt").replace(
        "Generate only the configured derived report.",
        "Generate a different report policy.",
    );
    std::fs::write(policy_path, changed_policy).unwrap();
    let publisher = OutputPublisher::from_config(&config).unwrap();
    let trusted = TrustedDerivation {
        contract: config.report_contract.clone(),
        request_digest: REQUEST_DIGEST.to_string(),
        source_lineage: None,
        compliance_evidence: None,
    };

    let error = publisher
        .emit_report(
            &exact_report_proposal("<p>policy drift</p>", None, None),
            &trusted,
        )
        .unwrap_err();

    assert_eq!(error.code(), "tn_output_policy_mismatch");
}

#[test]
fn output_publisher_rejects_writer_and_reader_provenance_mismatch_at_init() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let mut config = fixture_config(&fixture);
    config.output_ceremony_id = "cer_wrong_output".to_string();

    let error = match OutputPublisher::from_config(&config) {
        Ok(_) => panic!("mismatched output provenance was accepted"),
        Err(error) => error,
    };

    assert_eq!(error.code(), "tn_output_config_mismatch");
}

#[test]
fn output_publisher_rejects_unsigned_public_unbound_or_cross_routed_output_profiles() {
    for mutation in [
        "unsigned",
        "public_default",
        "unexpected_group_aad",
        "governance_routed_to_default",
    ] {
        let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
        let config = fixture_config(&fixture);
        let mut writer = tn_core::config::load(&config.output_tn_config_path).unwrap();
        match mutation {
            "unsigned" => writer.ceremony.sign = false,
            "public_default" => writer.public_fields.push("default".to_string()),
            "unexpected_group_aad" => {
                writer
                    .groups
                    .get_mut("default")
                    .unwrap()
                    .aad
                    .insert("group".to_string(), json!("default"));
            }
            "governance_routed_to_default" => writer
                .groups
                .get_mut("default")
                .unwrap()
                .fields
                .push("instruction".to_string()),
            _ => unreachable!(),
        }
        tn_core::config::save(&writer, &config.output_tn_config_path).unwrap();

        let error = match OutputPublisher::from_config(&config) {
            Ok(_) => panic!("{mutation} output profile was accepted"),
            Err(error) => error,
        };

        assert_eq!(error.code(), "tn_output_config_mismatch", "{mutation}");
    }
}

#[test]
fn output_publisher_rejects_source_and_output_state_aliases() {
    for alias in ["ceremony", "reader", "index"] {
        let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
        let mut config = fixture_config(&fixture);
        match alias {
            "ceremony" => config.output_ceremony_id = config.ceremony_id.clone(),
            "reader" => config.output_reader_key_dir = config.reader_key_dir.clone(),
            "index" => {
                config.output_index_key_path = config.input_index_key_path.clone();
            }
            _ => unreachable!(),
        }

        let error = match OutputPublisher::from_config(&config) {
            Ok(_) => panic!("{alias} source/output alias was accepted"),
            Err(error) => error,
        };

        assert_eq!(error.code(), "tn_output_state_alias", "{alias}");
    }
}

#[test]
fn output_publisher_rejects_effective_keystore_alias_of_source_reader() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let mut writer_config = tn_core::config::load(&config.output_tn_config_path).unwrap();
    writer_config.keystore.path = config.reader_key_dir.to_string_lossy().into_owned();
    tn_core::config::save(&writer_config, &config.output_tn_config_path).unwrap();

    let error = match OutputPublisher::from_config(&config) {
        Ok(_) => panic!("effective output keystore alias was accepted"),
        Err(error) => error,
    };

    assert_eq!(error.code(), "tn_output_state_alias");
}

#[test]
fn output_publisher_rejects_effective_log_alias_of_source_index() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let mut writer_config = tn_core::config::load(&config.output_tn_config_path).unwrap();
    writer_config.logs.path = config.input_index_key_path.to_string_lossy().into_owned();
    tn_core::config::save(&writer_config, &config.output_tn_config_path).unwrap();

    let error = match OutputPublisher::from_config(&config) {
        Ok(_) => panic!("effective output log alias was accepted"),
        Err(error) => error,
    };

    assert_eq!(error.code(), "tn_output_state_alias");
}

#[test]
fn report_and_compliance_rows_advance_independent_per_event_type_chains() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let config = fixture_config(&fixture);
    let publisher = Arc::new(OutputPublisher::from_config(&config).unwrap());
    let dispatcher = ToolDispatcher::new(Arc::clone(&publisher), "configured-compliance").unwrap();
    let trusted = TrustedDerivation {
        contract: config.report_contract.clone(),
        request_digest: REQUEST_DIGEST.to_string(),
        source_lineage: None,
        compliance_evidence: None,
    };
    let proposal = exact_report_proposal("<p>chain</p>", None, None);
    let context = ToolContext {
        run_id: "run-chain".to_string(),
        source_device_identity: fixture.device_identity.clone(),
        source_event_id: "source-chain".to_string(),
        source_event_type: "precisely.risk.enriched".to_string(),
        source_row_hash: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_string(),
        selected_sections: BTreeSet::from(["default".to_string(), "precisely".to_string()]),
        action: "derived_html_report".to_string(),
    };
    let call = RegisteredCall {
        id: "call-chain".to_string(),
        name: "configured-compliance".to_string(),
        arguments: br#"{"source_event_id":"source-chain","action":"derived_html_report"}"#.to_vec(),
    };

    let report_one = publisher.emit_report(&proposal, &trusted).unwrap();
    let ToolReceipt::Compliance(compliance_one) = dispatcher.execute(&call, &context).unwrap();
    let report_two = publisher.emit_report(&proposal, &trusted).unwrap();
    let ToolReceipt::Compliance(compliance_two) = dispatcher.execute(&call, &context).unwrap();

    let output_log = std::fs::read_to_string(
        config
            .output_tn_config_path
            .parent()
            .unwrap()
            .join(".tn")
            .join("logs")
            .join("tn.ndjson"),
    )
    .unwrap();
    let rows = output_log
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|row| {
            matches!(
                row["event_type"].as_str(),
                Some("report.generated" | "compliance.creation.logged")
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 4);
    let reports = rows
        .iter()
        .filter(|row| row["event_type"] == "report.generated")
        .collect::<Vec<_>>();
    let compliance = rows
        .iter()
        .filter(|row| row["event_type"] == "compliance.creation.logged")
        .collect::<Vec<_>>();
    assert_eq!(reports[0]["row_hash"], report_one.row_hash);
    assert_eq!(reports[1]["row_hash"], report_two.row_hash);
    assert_eq!(reports[1]["prev_hash"], report_one.row_hash);
    assert_eq!(
        reports[1]["sequence"].as_u64(),
        reports[0]["sequence"].as_u64().map(|value| value + 1)
    );
    assert_eq!(compliance[0]["row_hash"], compliance_one.row_hash);
    assert_eq!(compliance[1]["row_hash"], compliance_two.row_hash);
    assert_eq!(compliance[1]["prev_hash"], compliance_one.row_hash);
    assert_eq!(
        compliance[1]["sequence"].as_u64(),
        compliance[0]["sequence"].as_u64().map(|value| value + 1)
    );
    assert_ne!(reports[1]["prev_hash"], compliance[0]["row_hash"]);
    assert_ne!(compliance[1]["prev_hash"], reports[0]["row_hash"]);
}

fn exact_report_proposal(
    html: &str,
    lineage: Option<SourceLineage>,
    evidence: Option<ComplianceReceipt>,
) -> ReportProposal {
    serde_json::from_value(json!({
        "event_type": "report.generated",
        "public_sections": {"default": {"report_html": html}},
        "governance": {
            "instruction": "Generate only the configured derived report.",
            "use_for": "Creating the governed HTML report requested by the user.",
            "do_not_use_for": "Any ungoverned output or unrelated purpose.",
            "consequences": "Reject the report and release no output when validation fails.",
            "on_violation_or_error": "Abort the governed session and record the failing gate.",
            "source_lineage": lineage,
            "compliance_evidence": evidence
        }
    }))
    .unwrap()
}

fn literal_lineage(device: &str, event_id: &str, row_hash: &str) -> SourceLineage {
    SourceLineage {
        source_device_identity: device.to_string(),
        source_event_id: event_id.to_string(),
        source_event_type: "precisely.risk.enriched".to_string(),
        derived_from_row_hash: row_hash.to_string(),
        source_policy: SOURCE_POLICY.to_string(),
        selected_sections: vec!["precisely".to_string()],
        operation: "derived_html_report".to_string(),
    }
}

fn with_member(base: &Value, name: &str, value: Value) -> Value {
    let mut changed = base.clone();
    changed
        .as_object_mut()
        .unwrap()
        .insert(name.to_string(), value);
    changed
}
