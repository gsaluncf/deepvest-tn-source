use std::collections::BTreeSet;
use std::fs;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tn_runtime::{
    parse_and_verify_request, AdmissionStage, GovernanceKeyProvenance, GovernanceReader,
    RequestLimits, RuntimeConfig, TnLlamaError, GOVERNANCE_FIELDS,
};
use tn_test_tools::build_real_tn_fixture;

#[test]
fn opens_only_tn_agents_and_verifies_the_six_required_fields() {
    let fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.row1_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();
    let reader = fixture_reader(&fixture);

    let verified = reader.open_and_verify(&request.row).unwrap();

    assert_eq!(
        serde_json::to_value(&verified.plaintext).unwrap(),
        serde_json::to_value(&fixture.governance).unwrap()
    );
    assert_eq!(verified.commitments.len(), 7);
    let names = verified
        .commitments
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert!(GOVERNANCE_FIELDS.iter().all(|field| names.contains(*field)));
    assert!(names.contains("governs_sections"));
    assert_eq!(
        verified.plaintext["instruction"],
        "This record contains proprietary Precisely Wildfire Risk data in the encrypted `precisely` section."
    );
    assert_eq!(
        verified.plaintext["use_for"],
        "Use the `precisely` section for property reporting, underwriting, and risk assessment. Any derived report that uses this section must carry a human-readable statement attributing its data source to Precisely."
    );
    assert_eq!(
        verified.plaintext["do_not_use_for"],
        "Do not use the `precisely` section for unattributed public disclosure or strip its source lineage."
    );
    assert_eq!(
        verified.plaintext["consequences"],
        "Any output derived from the `precisely` section must carry this governance and exact source lineage."
    );
    assert_eq!(
        verified.plaintext["on_violation_or_error"],
        "Before emitting a derived report that uses the `precisely` section, call the configured compliance logging tool. If attribution or lineage cannot be preserved, stop and release nothing."
    );
    let plaintext = serde_json::to_value(&verified.plaintext).unwrap();
    assert!(plaintext["policy"]
        .as_str()
        .unwrap()
        .starts_with(".tn/config/agents.md#precisely.risk.enriched@1#sha256:"));
}

#[test]
fn plaintext_governance_is_rejected_by_the_encrypted_runtime_profile() {
    let fixture = build_real_tn_fixture().unwrap();
    let limits = request_limits(&fixture.device_identity);

    let error = parse_and_verify_request(&fixture.plaintext_report_request_json, &limits)
        .expect_err("plaintext governance must never enter the encrypted endpoint");
    assert_eq!(error.code(), "tn_envelope_shape");
    assert_eq!(error.stage(), AdmissionStage::RowShape);
}

#[test]
fn opened_typed_governance_extra_survives_and_its_commitment_is_verified() {
    let fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.extra_governance_field_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();

    let verified = fixture_reader(&fixture)
        .open_and_verify(&request.row)
        .unwrap();

    let plaintext = serde_json::to_value(&verified.plaintext).unwrap();
    assert_eq!(plaintext["governance_revision"], json!(2));
    assert_eq!(
        verified.plaintext.keys().cloned().collect::<BTreeSet<_>>(),
        verified
            .commitments
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
    );
}

#[test]
fn officially_rehashed_typed_extra_commitment_mutation_fails_after_open() {
    let fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.extra_governance_commitment_mutation_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();

    let error = fixture_reader(&fixture)
        .open_and_verify(&request.row)
        .unwrap_err();

    assert_eq!(error.code(), "protected_group_field_commitment_mismatch");
    assert_eq!(error.stage(), AdmissionStage::Governance);
}

#[test]
fn positive_real_aad_row_passes_admission_with_the_literal_canonical_marker() {
    let fixture = build_real_tn_fixture().unwrap();

    let request = parse_and_verify_request(
        &fixture.aad_bound_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();

    let expected = serde_json::to_string(&fixture.aad_marker).unwrap();
    assert_eq!(request.row.public_aad(), Some(expected.as_str()));
    let marker = fixture.aad_marker["tn.agents"].as_object().unwrap();
    assert_eq!(
        marker.keys().map(String::as_str).collect::<Vec<_>>(),
        ["governed_by", "policy"]
    );
    assert_ne!(
        marker["governed_by"].as_str().unwrap(),
        request.row.device_identity()
    );

    let governance = fixture_reader(&fixture)
        .open_and_verify(&request.row)
        .unwrap();
    assert_eq!(marker["policy"], governance.plaintext["policy"]);
}

#[test]
fn encrypted_groups_must_share_one_governance_marker() {
    let fixture = build_real_tn_fixture().unwrap();
    let mut request: Value = serde_json::from_slice(&fixture.row1_request_json).unwrap();
    let mut aad = fixture.aad_marker.clone();
    aad["tn.agents"]["governed_by"] = json!(fixture.device_identity.clone());
    request["tn_row"]["tn_aad"] = Value::String(
        String::from_utf8(tn_core::canonical::canonical_bytes(&aad).unwrap()).unwrap(),
    );

    let error = parse_and_verify_request(
        &serde_json::to_vec(&request).unwrap(),
        &request_limits(&fixture.device_identity),
    )
    .unwrap_err();

    assert_eq!(error.code(), "tn_aad_binding_mismatch");
    assert_eq!(error.stage(), AdmissionStage::RowShape);
}

#[test]
fn valid_outer_hash_with_changed_commitment_fails_after_open() {
    let fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.commitment_mutation_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();

    let error = fixture_reader(&fixture)
        .open_group_and_verify(&request.row, "precisely")
        .unwrap_err();

    assert_eq!(error.code(), "protected_group_field_commitment_mismatch");
    assert_eq!(error.stage(), AdmissionStage::Governance);
}

#[test]
fn officially_rehashed_ciphertext_tamper_fails_after_outer_verification() {
    let fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.ciphertext_mutation_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();

    let error = fixture_reader(&fixture)
        .open_group_and_verify(&request.row, "precisely")
        .unwrap_err();

    assert_eq!(error.code(), "protected_group_not_opened");
    assert_eq!(error.stage(), AdmissionStage::Governance);
}

#[test]
fn officially_rehashed_aad_tamper_fails_after_outer_verification() {
    let fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.aad_mutation_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();

    let error = fixture_reader(&fixture)
        .open_and_verify(&request.row)
        .unwrap_err();

    assert!(matches!(
        error.code(),
        "protected_group_open_failed" | "protected_group_not_opened"
    ));
    assert_eq!(error.stage(), AdmissionStage::Governance);
}

#[test]
fn authenticated_aad_policy_must_match_the_opened_governance_policy() {
    let fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.aad_policy_mismatch_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();

    let error = fixture_reader(&fixture)
        .open_and_verify(&request.row)
        .unwrap_err();

    assert_eq!(error.code(), "governance_aad_policy_mismatch");
    assert_eq!(error.stage(), AdmissionStage::Governance);
}

#[test]
fn a_real_wrong_btn_reader_cannot_open_governance() {
    let fixture = build_real_tn_fixture().unwrap();
    let wrong_fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.row1_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();
    let wrong_reader = GovernanceReader::from_paths(
        &wrong_fixture.receiver_key_dir,
        &fixture.input_index_key_path,
        provenance(&fixture),
    )
    .unwrap();

    let error = wrong_reader.open_and_verify(&request.row).unwrap_err();

    assert!(matches!(
        error.code(),
        "protected_group_open_failed" | "protected_group_not_opened"
    ));
    assert_eq!(error.stage(), AdmissionStage::Governance);
}

#[test]
fn a_wrong_derived_index_key_cannot_validate_opened_governance() {
    let fixture = build_real_tn_fixture().unwrap();
    let wrong_fixture = build_real_tn_fixture().unwrap();
    let request = parse_and_verify_request(
        &fixture.row1_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap();
    let wrong_key_reader = GovernanceReader::from_paths(
        &fixture.receiver_key_dir,
        &wrong_fixture.input_index_key_path,
        provenance(&fixture),
    )
    .unwrap();

    let error = wrong_key_reader.open_and_verify(&request.row).unwrap_err();

    assert_eq!(error.code(), "protected_group_field_commitment_mismatch");
    assert_eq!(error.stage(), AdmissionStage::Governance);
}

#[test]
fn real_emit_with_a_missing_required_tn_agents_field_fails_before_open() {
    let fixture = build_real_tn_fixture().unwrap();
    let error = parse_and_verify_request(
        &fixture.missing_governance_field_request_json,
        &request_limits(&fixture.device_identity),
    )
    .unwrap_err();
    assert_eq!(error.code(), "tn_governance_shape");
    assert_eq!(error.stage(), AdmissionStage::RowShape);
}

#[test]
fn optional_tn_aad_rejects_noncanonical_duplicate_malformed_or_misscoped_content() {
    let fixture = build_real_tn_fixture().unwrap();
    let cases = [
        ("wrong public type", json!({"tn.agents": {"case": "A-17"}})),
        ("malformed inner JSON", Value::String("{".to_string())),
        (
            "duplicate inner key",
            Value::String(r#"{"tn.agents":{"case":"first","case":"second"}}"#.to_string()),
        ),
        (
            "noncanonical inner JSON",
            Value::String(r#"{ "tn.agents": {"case":"A-17"}}"#.to_string()),
        ),
        (
            "marker for absent group",
            Value::String(r#"{"other":{"case":"A-17"}}"#.to_string()),
        ),
        (
            "empty marker",
            Value::String(r#"{"tn.agents":{}}"#.to_string()),
        ),
        (
            "non-scalar marker value",
            Value::String(r#"{"tn.agents":{"case":{"nested":true}}}"#.to_string()),
        ),
    ];

    for (name, aad) in cases {
        let raw = mutate_public_aad(&fixture.row1_request_json, aad);
        let error = parse_and_verify_request(&raw, &request_limits(&fixture.device_identity))
            .expect_err(name);
        assert_eq!(error.code(), "tn_aad_shape", "{name}");
        assert_eq!(error.stage(), AdmissionStage::RowShape, "{name}");
    }
}

#[test]
fn reader_material_and_version_one_provenance_are_closed() {
    let fixture = build_real_tn_fixture().unwrap();

    for (name, provenance) in [
        (
            "empty ceremony",
            GovernanceKeyProvenance::new("", "tn.agents", 0),
        ),
        (
            "wrong group",
            GovernanceKeyProvenance::new(&fixture.ceremony_id, "default", 0),
        ),
        (
            "rotated epoch",
            GovernanceKeyProvenance::new(&fixture.ceremony_id, "tn.agents", 1),
        ),
    ] {
        let error = expect_reader_error(GovernanceReader::from_paths(
            &fixture.receiver_key_dir,
            &fixture.input_index_key_path,
            provenance,
        ));
        assert_eq!(error.code(), "governance_provenance_invalid", "{name}");
    }

    let short_key = fixture.temp.path().join("short-index-key.bin");
    fs::write(&short_key, [0x42_u8; 31]).unwrap();
    let error = expect_reader_error(GovernanceReader::from_paths(
        &fixture.receiver_key_dir,
        &short_key,
        provenance(&fixture),
    ));
    assert_eq!(error.code(), "governance_index_key_invalid");

    fs::write(
        fixture.receiver_key_dir.join("unexpected.key"),
        b"not allowed",
    )
    .unwrap();
    let error = expect_reader_error(GovernanceReader::from_paths(
        &fixture.receiver_key_dir,
        &fixture.input_index_key_path,
        provenance(&fixture),
    ));
    assert_eq!(error.code(), "governance_reader_material_invalid");
}

#[test]
fn reader_exposes_only_the_group_key_digest_and_configured_provenance() {
    let fixture = build_real_tn_fixture().unwrap();
    let reader = fixture_reader(&fixture);
    let key = fs::read(&fixture.input_index_key_path).unwrap();
    let expected = format!("sha256:{}", hex::encode(Sha256::digest(key)));

    assert_eq!(reader.index_key_digest(), expected);
    assert_eq!(reader.provenance().ceremony_id, fixture.ceremony_id);
    assert_eq!(reader.provenance().group, "tn.agents");
    assert_eq!(reader.provenance().index_epoch, 0);
}

#[test]
fn generated_runtime_config_deserializes_with_the_exact_governance_group() {
    let fixture = build_real_tn_fixture().unwrap();
    let config: RuntimeConfig =
        serde_json::from_slice(&fs::read(fixture.temp.path().join("runtime-config.json")).unwrap())
            .unwrap();

    assert_eq!(config.writer_authority, "group_capability");
    assert_eq!(config.expected_event_type, "precisely.risk.enriched");
    assert_eq!(
        config.allowed_input_groups,
        ["default", "firm_data", "precisely"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
    assert_eq!(
        config.report_contract.allowed_input_sections,
        ["default", "firm_data", "precisely"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
    assert_eq!(config.report_contract.output_event_type, "report.generated");
    assert_eq!(config.report_contract.output_public_section, "default");
    assert_eq!(config.report_contract.operation, "derived_html_report");
    let precisely_rule = &config.report_contract.section_rules["precisely"];
    assert!(precisely_rule.propagate_source_governance);
    assert_eq!(
        precisely_rule.required_attribution.as_deref(),
        Some("Source: Risk assessment derived from Precisely Wildfire Risk Data.")
    );
    assert_eq!(
        precisely_rule.required_tool.as_deref(),
        Some("log_compliance_creation")
    );
    assert_eq!(config.ceremony_id, fixture.ceremony_id);
    assert_eq!(config.governance_group, "tn.agents");
    assert_eq!(config.input_index_epoch, 0);
    assert_eq!(
        config.reader_key_dir.canonicalize().unwrap(),
        fixture.receiver_key_dir.canonicalize().unwrap()
    );
    assert_eq!(
        config.input_index_key_path.canonicalize().unwrap(),
        fixture.input_index_key_path.canonicalize().unwrap()
    );
}

fn fixture_reader(fixture: &tn_test_tools::RealTnFixture) -> GovernanceReader {
    GovernanceReader::from_paths(
        &fixture.receiver_key_dir,
        &fixture.input_index_key_path,
        provenance(fixture),
    )
    .unwrap()
}

fn provenance(fixture: &tn_test_tools::RealTnFixture) -> GovernanceKeyProvenance {
    GovernanceKeyProvenance::new(&fixture.ceremony_id, "tn.agents", fixture.input_index_epoch)
}

fn request_limits(device_identity: &str) -> RequestLimits {
    RequestLimits {
        max_request_bytes: 1024 * 1024,
        max_message_bytes: 16 * 1024,
        expected_device_identity: Some(device_identity.to_string()),
        expected_event_type: "precisely.risk.enriched".to_string(),
        allowed_input_groups: ["default", "precisely", "firm_data"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        require_signature: true,
        max_active_sessions: 1,
        max_chain_namespaces: 1,
    }
}

fn mutate_public_aad(raw: &[u8], aad: Value) -> Vec<u8> {
    let mut request: Value = serde_json::from_slice(raw).unwrap();
    request["tn_row"]["tn_aad"] = aad;
    serde_json::to_vec(&request).unwrap()
}

fn expect_reader_error(result: Result<GovernanceReader, TnLlamaError>) -> TnLlamaError {
    match result {
        Ok(_) => panic!("reader construction unexpectedly succeeded"),
        Err(error) => error,
    }
}
