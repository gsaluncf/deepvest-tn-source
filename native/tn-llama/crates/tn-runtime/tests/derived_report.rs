mod support;

use serde_json::{json, Value};
use tn_core::sealed_object::{extract_group_blocks, require_envelope_shape, verify_sealed};
use tn_runtime::{ActionOutcome, Phase, RELEASE_PROTOCOL};

use support::{advance_to_selection, prepare_and_begin, SessionHarness};

const USER_REQUEST: &str = "Generate a property risk card from the Precisely section.";
const ATTRIBUTION: &str = "Source: Risk assessment derived from Precisely Wildfire Risk Data.";
const SELECTED_HTML: &str = "<article><p>Wildfire Risk Score: <span style=\"color: red\">92</span></p><p>Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></article>";
const FIRM_HTML: &str = "<article><p>Customer FIRM-42 underwriting report.</p></article>";

#[test]
fn model_action_phases_have_the_exact_native_numbers() {
    assert_eq!(u32::from(Phase::GovernanceRead), 1);
    assert_eq!(u32::from(Phase::CommitmentEcho), 2);
    assert_eq!(u32::from(Phase::SectionSelection), 3);
    assert_eq!(u32::from(Phase::GovernedAction), 4);
}

#[test]
fn exact_echo_leads_to_a_value_free_section_selection_turn() {
    let mut harness = SessionHarness::new(USER_REQUEST);

    let turn = advance_to_selection(&mut harness);
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();

    assert_eq!(body["tools"][0]["function"]["name"], "select_sections");
    assert_eq!(body["tool_choice"], "required");
    assert_eq!(body["parallel_tool_calls"], false);

    let messages = body["messages"].as_array().unwrap();
    assert_eq!(
        messages[0]["content"],
        "This is a server-governed action interaction. Call exactly one available tool and return no prose. Copy server-provided trusted fields exactly. Generate only the requested output fields. The server validates, executes, emits, and releases every action."
    );
    let selection_request = messages.last().unwrap()["content"].as_str().unwrap();
    let selection_request: Value = serde_json::from_str(selection_request).unwrap();
    assert_eq!(selection_request["user_request"], USER_REQUEST);
    assert_eq!(
        selection_request["available_sections"],
        json!({
            "default": ["request_id", "run_id"],
            "firm_data": ["customer_account", "report_purpose"],
            "precisely": ["address", "property_id", "wildfire_risk_score"]
        })
    );

    let exact_body = String::from_utf8(turn.request_json).unwrap();
    for forbidden in [
        "92",
        "123 Pine Ln",
        "P-98765",
        "FIRM-42",
        "proprietary Precisely",
        "governance_commitments",
    ] {
        assert!(!exact_body.contains(forbidden), "leaked {forbidden}");
    }
}

#[test]
fn selected_values_and_next_action_are_derived_from_the_exact_selection() {
    for (selected, required_action, included, excluded) in [
        (
            "precisely",
            "log_compliance_creation",
            &["123 Pine Ln, CA", "P-98765", "92"][..],
            &["FIRM-42"][..],
        ),
        (
            "firm_data",
            "emit_governed_record",
            &["FIRM-42", "underwriting"][..],
            &["123 Pine Ln, CA", "P-98765", "92"][..],
        ),
    ] {
        let mut harness = SessionHarness::new(USER_REQUEST);
        let turn = advance_to_selection(&mut harness);
        prepare_and_begin(&turn, b"rendered-section-selection", 2048, 13);
        let result = harness.section_selection_result(&[selected]);

        let outcome = harness
            .session
            .accept_model_action(turn.permit, result.native())
            .unwrap();
        assert!(matches!(outcome, tn_runtime::ActionOutcome::Continue));

        let action_turn = harness.session.next_model_turn().unwrap();
        assert_eq!(action_turn.permit.phase(), Phase::GovernedAction);
        let body: Value = serde_json::from_slice(&action_turn.request_json).unwrap();
        assert_eq!(body["tools"][0]["function"]["name"], required_action);
        let exact_body = String::from_utf8(action_turn.request_json).unwrap();
        for expected in included {
            assert!(exact_body.contains(expected), "missing selected {expected}");
        }
        for forbidden in excluded {
            assert!(
                !exact_body.contains(forbidden),
                "leaked unselected {forbidden}"
            );
        }
    }
}

#[test]
fn compliance_tool_schema_pins_the_trusted_source_and_action() {
    let mut harness = SessionHarness::new(USER_REQUEST);
    let selection_turn = advance_to_selection(&mut harness);
    prepare_and_begin(&selection_turn, b"rendered-constant-selection", 2048, 29);
    let selection = harness.section_selection_result(&["precisely"]);
    harness
        .session
        .accept_model_action(selection_turn.permit, selection.native())
        .unwrap();

    let action_turn = harness.session.next_model_turn().unwrap();
    let body: Value = serde_json::from_slice(&action_turn.request_json).unwrap();
    let properties = &body["tools"][0]["function"]["parameters"]["properties"];
    assert_eq!(
        properties["source_event_id"]["const"],
        source_event_id(&harness)
    );
    assert_eq!(properties["action"]["const"], "derived_html_report");
}

#[test]
fn emission_tool_schema_pins_server_owned_derivation_values() {
    let (harness, emit_turn, proposal) = selected_ready_to_emit();
    let body: Value = serde_json::from_slice(&emit_turn.request_json).unwrap();
    let parameters = &body["tools"][0]["function"]["parameters"];
    let governance = &parameters["properties"]["governance"]["properties"];

    assert_eq!(
        parameters["properties"]["event_type"]["const"],
        "report.generated"
    );
    assert_eq!(
        governance["instruction"]["const"],
        "Generate only the configured derived report."
    );
    assert_eq!(
        governance["source_lineage"]["properties"]["source_event_id"]["const"],
        source_event_id(&harness)
    );
    assert_eq!(
        governance["source_lineage"]["properties"]["derived_from_row_hash"]["const"],
        harness.row_hash
    );
    assert_eq!(
        governance["compliance_evidence"]["properties"]["event_id"]["const"],
        proposal["governance"]["compliance_evidence"]["event_id"]
    );
    assert_eq!(
        governance["compliance_evidence"]["properties"]["action"]["const"],
        "derived_html_report"
    );
}

#[test]
fn attribution_gate_checks_the_human_visible_html_text() {
    let (mut harness, emit_turn, mut proposal) = selected_ready_to_emit();
    proposal["public_sections"]["default"]["report_html"] = json!(
        "<article><p>Wildfire Risk Score: 92</p><p aria-hidden=\"false\" style=\"display: block\"><strong>Source:</strong> Risk assessment derived from Precisely Wildfire Risk Data.</p></article>"
    );
    prepare_and_begin(&emit_turn, b"rendered-visible-attribution", 2048, 30);
    let emission = harness.action_result("visible-attribution", "emit_governed_record", proposal);

    assert!(matches!(
        harness
            .session
            .accept_model_action(emit_turn.permit, emission.native())
            .unwrap(),
        ActionOutcome::Release(_)
    ));
}

#[test]
fn raw_text_tag_like_strings_do_not_hide_visible_attribution() {
    let html = "<article><script>const marker = \"<template>\";</script><p>Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></article>";
    let (mut harness, emit_turn, mut proposal) = selected_ready_to_emit();
    proposal["public_sections"]["default"]["report_html"] = json!(html);
    prepare_and_begin(&emit_turn, b"rendered-raw-text-attribution", 2048, 30);
    let emission = harness.action_result(
        "raw-text-attribution-script",
        "emit_governed_record",
        proposal,
    );

    assert!(matches!(
        harness
            .session
            .accept_model_action(emit_turn.permit, emission.native())
            .unwrap(),
        ActionOutcome::Release(_)
    ));
}

#[test]
fn hidden_attribution_in_non_rendered_html_is_rejected_without_output_emission() {
    for (case, html) in [
        (
            "script",
            "<article><p>Wildfire Risk Score: 92</p><script type=\"application/json\">Source: Risk assessment derived from Precisely Wildfire Risk Data.</script></article>",
        ),
        (
            "style",
            "<article><p>Wildfire Risk Score: 92</p><style>Source: Risk assessment derived from Precisely Wildfire Risk Data.</style></article>",
        ),
        (
            "template",
            "<article><p>Wildfire Risk Score: 92</p><template>Source: Risk assessment derived from Precisely Wildfire Risk Data.</template></article>",
        ),
        (
            "script_self_closing_form",
            "<article><p>Wildfire Risk Score: 92</p><script/>Source: Risk assessment derived from Precisely Wildfire Risk Data.</script></article>",
        ),
        (
            "nested_template",
            "<article><p>Wildfire Risk Score: 92</p><template><template>hidden</template>Source: Risk assessment derived from Precisely Wildfire Risk Data.</template></article>",
        ),
        (
            "hidden_attribute",
            "<article><p>Wildfire Risk Score: 92</p><p hidden>Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></article>",
        ),
        (
            "quoted_greater_than_before_hidden_attribute",
            "<article><p>Wildfire Risk Score: 92</p><p title=\">\" hidden>Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></article>",
        ),
        (
            "aria_hidden_ancestor",
            "<article><p>Wildfire Risk Score: 92</p><aside aria-hidden=\"true\"><p>Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></aside></article>",
        ),
        (
            "hidden_same_tag_ancestor",
            "<article><p>Wildfire Risk Score: 92</p><div hidden><div>nested</div>Source: Risk assessment derived from Precisely Wildfire Risk Data.</div></article>",
        ),
        (
            "inline_display_none",
            "<article><p>Wildfire Risk Score: 92</p><p style=\"display: none\">Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></article>",
        ),
        (
            "inline_visibility_hidden",
            "<article><p>Wildfire Risk Score: 92</p><div style='visibility: hidden'><p>Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></div></article>",
        ),
        (
            "embedded_stylesheet",
            "<article><style>.source { display: none; }</style><p class=\"source\">Source: Risk assessment derived from Precisely Wildfire Risk Data.</p></article>",
        ),
    ] {
        let (mut harness, emit_turn, mut proposal) = selected_ready_to_emit();
        let config = support::fixture_config(&harness._fixture);
        let output_log = config
            .output_tn_config_path
            .parent()
            .unwrap()
            .join(".tn")
            .join("logs")
            .join("tn.ndjson");
        let output_before = std::fs::read(&output_log).unwrap();
        proposal["public_sections"]["default"]["report_html"] = json!(html);
        prepare_and_begin(&emit_turn, b"rendered-hidden-attribution", 2048, 30);
        let emission =
            harness.action_result(&format!("hidden-attribution-{case}"), "emit_governed_record", proposal);

        let error = harness
            .session
            .accept_model_action(emit_turn.permit, emission.native())
            .unwrap_err();

        assert_eq!(error.code(), "tn_required_attribution_missing", "{case}");
        assert_eq!(harness.session.state(), tn_runtime::State::Failed, "{case}");
        assert_eq!(std::fs::read(output_log).unwrap(), output_before, "{case}");
    }
}

#[test]
fn section_selection_rejects_empty_duplicate_unknown_and_public_metadata_names() {
    for (name, sections) in [
        ("empty", vec![]),
        ("duplicate", vec!["precisely", "precisely"]),
        ("unknown", vec!["unknown"]),
        ("public_metadata", vec!["run_id"]),
    ] {
        let mut harness = SessionHarness::new(USER_REQUEST);
        let turn = advance_to_selection(&mut harness);
        prepare_and_begin(&turn, b"rendered-invalid-selection", 2048, 31);
        let result = harness.section_selection_result(&sections);

        let error = harness
            .session
            .accept_model_action(turn.permit, result.native())
            .unwrap_err();

        assert_eq!(error.code(), "tn_section_selection_invalid", "{name}");
        assert_eq!(harness.session.state(), tn_runtime::State::Failed, "{name}");
    }
}

#[test]
fn wrong_action_is_terminal_in_every_state() {
    for state in [
        "governance_read",
        "commitment_echo",
        "section_selection",
        "governed_action",
    ] {
        let mut harness = SessionHarness::new(USER_REQUEST);
        let turn = match state {
            "governance_read" => harness.session.next_model_turn().unwrap(),
            "commitment_echo" => support::advance_to_echo(&mut harness),
            "section_selection" => advance_to_selection(&mut harness),
            "governed_action" => {
                let selection_turn = advance_to_selection(&mut harness);
                prepare_and_begin(
                    &selection_turn,
                    b"rendered-selection-before-wrong-action",
                    2048,
                    32,
                );
                let selection = harness.section_selection_result(&["firm_data"]);
                harness
                    .session
                    .accept_model_action(selection_turn.permit, selection.native())
                    .unwrap();
                harness.session.next_model_turn().unwrap()
            }
            _ => unreachable!(),
        };
        let n_predict = match state {
            "governance_read" => 256,
            "commitment_echo" => 1536,
            "section_selection" | "governed_action" => 2048,
            _ => unreachable!(),
        };
        prepare_and_begin(&turn, b"rendered-wrong-action", n_predict, 33);
        let result = harness.action_result("wrong-call", "unregistered_action", json!({}));

        let error = harness
            .session
            .accept_model_action(turn.permit, result.native())
            .unwrap_err();

        assert_eq!(error.code(), "tn_action_not_allowed", "{state}");
        assert_eq!(
            harness.session.state(),
            tn_runtime::State::Failed,
            "{state}"
        );
    }
}

#[test]
fn incomplete_multiple_call_and_prose_action_results_fail_terminally() {
    for (name, expected_code) in [
        ("incomplete", "tn_model_turn_incomplete"),
        ("multiple", "tn_tool_call_count"),
        ("prose", "tn_model_prose_not_allowed"),
    ] {
        let mut harness = SessionHarness::new(USER_REQUEST);
        let turn = harness.session.next_model_turn().unwrap();
        prepare_and_begin(&turn, b"rendered-invalid-native-result", 256, 34);
        let mut result = harness.get_governance_result();
        match name {
            "incomplete" => result.stop_limit = true,
            "multiple" => result.call_count = 2,
            "prose" => result.content = b"extra prose".to_vec(),
            _ => unreachable!(),
        }

        let error = harness
            .session
            .accept_model_action(turn.permit, result.native())
            .unwrap_err();

        assert_eq!(error.code(), expected_code, "{name}");
        assert_eq!(harness.session.state(), tn_runtime::State::Failed, "{name}");
    }
}

#[test]
fn selected_section_tools_are_mandatory_scoped_and_single_use() {
    let mut firm = SessionHarness::new("Generate the firm report.");
    let firm_selection = advance_to_selection(&mut firm);
    prepare_and_begin(&firm_selection, b"rendered-firm-selection", 2048, 41);
    let select_firm = firm.section_selection_result(&["firm_data"]);
    firm.session
        .accept_model_action(firm_selection.permit, select_firm.native())
        .unwrap();
    let firm_action = firm.session.next_model_turn().unwrap();
    prepare_and_begin(&firm_action, b"rendered-illegal-compliance", 2048, 42);
    let illegal_compliance = firm.action_result(
        "illegal-compliance",
        "log_compliance_creation",
        json!({"source_event_id": source_event_id(&firm), "action": "derived_html_report"}),
    );
    assert_eq!(
        firm.session
            .accept_model_action(firm_action.permit, illegal_compliance.native())
            .unwrap_err()
            .code(),
        "tn_action_not_allowed"
    );

    let mut missing = SessionHarness::new(USER_REQUEST);
    let missing_selection = advance_to_selection(&mut missing);
    prepare_and_begin(
        &missing_selection,
        b"rendered-precisely-selection",
        2048,
        43,
    );
    let select_precisely = missing.section_selection_result(&["precisely"]);
    missing
        .session
        .accept_model_action(missing_selection.permit, select_precisely.native())
        .unwrap();
    let premature_emit = missing.session.next_model_turn().unwrap();
    prepare_and_begin(&premature_emit, b"rendered-premature-emission", 2048, 44);
    let emission = missing.action_result(
        "premature-emission",
        "emit_governed_record",
        report_proposal(SELECTED_HTML, None, None),
    );
    assert_eq!(
        missing
            .session
            .accept_model_action(premature_emit.permit, emission.native())
            .unwrap_err()
            .code(),
        "tn_required_tool_missing"
    );

    let (mut duplicate, emit_turn, _) = selected_ready_to_emit();
    prepare_and_begin(&emit_turn, b"rendered-duplicate-compliance", 2048, 45);
    let repeated = duplicate.action_result(
        "duplicate-compliance",
        "log_compliance_creation",
        json!({
            "source_event_id": source_event_id(&duplicate),
            "action": "derived_html_report"
        }),
    );
    assert_eq!(
        duplicate
            .session
            .accept_model_action(emit_turn.permit, repeated.native())
            .unwrap_err()
            .code(),
        "tn_tool_already_executed"
    );
}

#[test]
fn terminal_proposal_rejects_every_substituted_trusted_fact_and_server_owned_member() {
    for case in [
        "source_device_identity",
        "source_event_id",
        "source_event_type",
        "source_row_hash",
        "source_policy",
        "selected_section",
        "operation",
        "receipt_event_id",
        "receipt_event_type",
        "receipt_row_hash",
        "receipt_source_row_hash",
        "receipt_action",
        "output_event_type",
        "unauthorized_output_section",
        "trusted_header",
        "computed_policy",
        "field_commitments",
        "unknown_proposal_member",
        "unknown_governance_member",
        "missing_attribution",
    ] {
        let (mut harness, emit_turn, mut proposal) = selected_ready_to_emit();
        match case {
            "source_device_identity" => {
                proposal["governance"]["source_lineage"]["source_device_identity"] =
                    json!("did:key:zForged")
            }
            "source_event_id" => {
                proposal["governance"]["source_lineage"]["source_event_id"] = json!("forged")
            }
            "source_event_type" => {
                proposal["governance"]["source_lineage"]["source_event_type"] =
                    json!("source.other")
            }
            "source_row_hash" => {
                proposal["governance"]["source_lineage"]["derived_from_row_hash"] =
                    json!("sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            }
            "source_policy" => {
                proposal["governance"]["source_lineage"]["source_policy"] = json!("forged")
            }
            "selected_section" => {
                proposal["governance"]["source_lineage"]["selected_sections"] = json!(["firm_data"])
            }
            "operation" => proposal["governance"]["source_lineage"]["operation"] = json!("other"),
            "receipt_event_id" => {
                proposal["governance"]["compliance_evidence"]["event_id"] = json!("forged")
            }
            "receipt_event_type" => {
                proposal["governance"]["compliance_evidence"]["event_type"] = json!("other")
            }
            "receipt_row_hash" => {
                proposal["governance"]["compliance_evidence"]["row_hash"] =
                    json!("sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            }
            "receipt_source_row_hash" => {
                proposal["governance"]["compliance_evidence"]["source_row_hash"] =
                    json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
            }
            "receipt_action" => {
                proposal["governance"]["compliance_evidence"]["action"] = json!("other")
            }
            "output_event_type" => proposal["event_type"] = json!("report.other"),
            "unauthorized_output_section" => {
                proposal["public_sections"]["precisely"] = json!({"wildfire_risk_score": 92})
            }
            "trusted_header" => proposal["row_hash"] = json!(harness.row_hash),
            "computed_policy" => proposal["governance"]["policy"] = json!("forged"),
            "field_commitments" => {
                proposal["governance"]["field_hashes"] = json!({"instruction": "forged"})
            }
            "unknown_proposal_member" => proposal["extra"] = json!(true),
            "unknown_governance_member" => proposal["governance"]["extra"] = json!(true),
            "missing_attribution" => {
                proposal["public_sections"]["default"]["report_html"] =
                    json!("<article><p>Score 92.</p></article>")
            }
            _ => unreachable!(),
        }
        prepare_and_begin(&emit_turn, b"rendered-forged-emission", 2048, 61);
        let result =
            harness.action_result(&format!("forged-{case}"), "emit_governed_record", proposal);

        let error = harness
            .session
            .accept_model_action(emit_turn.permit, result.native())
            .unwrap_err();

        let expected = if case == "missing_attribution" {
            "tn_required_attribution_missing"
        } else {
            "tn_report_proposal_invalid"
        };
        assert_eq!(error.code(), expected, "{case}");
        assert_eq!(harness.session.state(), tn_runtime::State::Failed, "{case}");
        let mut retry = harness.runtime.start_session(&harness.raw_request).unwrap();
        retry.abort().unwrap();
    }
}

#[test]
fn firm_only_proposal_rejects_precisely_attribution_without_committing_source() {
    for (case, html) in [
        ("visible", format!("{FIRM_HTML}<p>{ATTRIBUTION}</p>")),
        (
            "script",
            format!("{FIRM_HTML}<script type=\"application/json\">{ATTRIBUTION}</script>"),
        ),
        ("hidden", format!("{FIRM_HTML}<p hidden>{ATTRIBUTION}</p>")),
    ] {
        let mut harness = SessionHarness::new("Generate an underwriting account report.");
        let selection_turn = advance_to_selection(&mut harness);
        prepare_and_begin(&selection_turn, b"rendered-firm-selection", 2048, 68);
        let selection = harness.section_selection_result(&["firm_data"]);
        harness
            .session
            .accept_model_action(selection_turn.permit, selection.native())
            .unwrap();
        let emit_turn = harness.session.next_model_turn().unwrap();
        prepare_and_begin(&emit_turn, b"rendered-unselected-attribution", 2048, 69);
        let emission = harness.action_result(
            &format!("firm-with-precisely-attribution-{case}"),
            "emit_governed_record",
            report_proposal(&html, None, None),
        );

        let error = harness
            .session
            .accept_model_action(emit_turn.permit, emission.native())
            .unwrap_err();

        assert_eq!(error.code(), "tn_unselected_attribution_present", "{case}");
        assert_eq!(harness.session.state(), tn_runtime::State::Failed, "{case}");
        let mut retry = harness.runtime.start_session(&harness.raw_request).unwrap();
        retry.abort().unwrap();
    }
}

#[test]
fn output_verification_failure_returns_no_release_and_does_not_commit_source_chain() {
    let (mut harness, emit_turn, proposal) = selected_ready_to_emit();
    let config = support::fixture_config(&harness._fixture);
    let output_log = config
        .output_tn_config_path
        .parent()
        .unwrap()
        .join(".tn")
        .join("logs")
        .join("tn.ndjson");
    use std::io::Write as _;
    writeln!(
        std::fs::OpenOptions::new()
            .append(true)
            .open(output_log)
            .unwrap(),
        "not-json"
    )
    .unwrap();
    prepare_and_begin(&emit_turn, b"rendered-output-failure", 2048, 71);
    let result = harness.action_result("output-failure", "emit_governed_record", proposal);

    let error = harness
        .session
        .accept_model_action(emit_turn.permit, result.native())
        .unwrap_err();

    assert_eq!(error.code(), "tn_output_reread_failed");
    assert_eq!(harness.session.state(), tn_runtime::State::Failed);
    let mut retry = harness.runtime.start_session(&harness.raw_request).unwrap();
    retry.abort().unwrap();
}

#[test]
fn precisely_path_executes_real_tool_and_releases_the_exact_verified_tn_row() {
    let mut harness = SessionHarness::new(USER_REQUEST);
    let selection_turn = advance_to_selection(&mut harness);
    prepare_and_begin(&selection_turn, b"rendered-section-selection", 2048, 13);
    let selection = harness.section_selection_result(&["precisely"]);
    assert!(matches!(
        harness
            .session
            .accept_model_action(selection_turn.permit, selection.native())
            .unwrap(),
        ActionOutcome::Continue
    ));

    let source: Value = serde_json::from_slice(&harness.raw_request).unwrap();
    let source_event_id = source["tn_row"]["event_id"].as_str().unwrap();
    let compliance_turn = harness.session.next_model_turn().unwrap();
    prepare_and_begin(&compliance_turn, b"rendered-compliance", 2048, 14);
    let compliance = harness.action_result(
        "call-compliance-4",
        "log_compliance_creation",
        json!({
            "source_event_id": source_event_id,
            "action": "derived_html_report"
        }),
    );
    assert!(matches!(
        harness
            .session
            .accept_model_action(compliance_turn.permit, compliance.native())
            .unwrap(),
        ActionOutcome::Continue
    ));

    let emit_turn = harness.session.next_model_turn().unwrap();
    assert_eq!(emit_turn.permit.phase(), Phase::GovernedAction);
    let emit_body: Value = serde_json::from_slice(&emit_turn.request_json).unwrap();
    assert_eq!(
        emit_body["tools"][0]["function"]["name"],
        "emit_governed_record"
    );
    let tool_messages = emit_body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect::<Vec<_>>();
    let receipt: Value =
        serde_json::from_str(tool_messages.last().unwrap()["content"].as_str().unwrap()).unwrap();
    assert_eq!(
        receipt
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        [
            "action",
            "event_id",
            "event_type",
            "row_hash",
            "source_row_hash"
        ]
    );
    assert_eq!(receipt["source_row_hash"], harness.row_hash);
    let lineage = json!({
        "source_device_identity": source["tn_row"]["device_identity"],
        "source_event_id": source_event_id,
        "source_event_type": "precisely.risk.enriched",
        "derived_from_row_hash": harness.row_hash,
        "source_policy": harness.governance["policy"],
        "selected_sections": ["precisely"],
        "operation": "derived_html_report"
    });
    prepare_and_begin(&emit_turn, b"rendered-report-emission", 2048, 15);
    let emission = harness.action_result(
        "call-emit-5",
        "emit_governed_record",
        report_proposal(SELECTED_HTML, Some(lineage.clone()), Some(receipt.clone())),
    );

    let ActionOutcome::Release(release) = harness
        .session
        .accept_model_action(emit_turn.permit, emission.native())
        .unwrap()
    else {
        panic!("terminal emission did not release");
    };

    assert_eq!(release.protocol, RELEASE_PROTOCOL);
    assert_eq!(release.source_event_id, source_event_id);
    assert_eq!(release.source_row_hash, harness.row_hash);
    let release_json = serde_json::to_value(&release).unwrap();
    assert_eq!(
        release_json
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        [
            "protocol",
            "run_id",
            "source_event_id",
            "source_row_hash",
            "output_event_id",
            "output_row_hash",
            "output_ndjson",
            "trace"
        ]
    );
    assert!(release.output_ndjson.ends_with('\n'));
    assert_eq!(release.output_ndjson.lines().count(), 1);
    let output: Value = serde_json::from_str(release.output_ndjson.trim_end()).unwrap();
    assert_eq!(output["event_id"], release.output_event_id);
    assert_eq!(output["row_hash"], release.output_row_hash);
    assert!(output["default"].get("report_html").is_none());
    assert!(!release.output_ndjson.contains(SELECTED_HTML));
    let opened_default = tn_core::unseal_as_recipient(
        release.output_ndjson.trim_end(),
        &harness._fixture.temp.path().join("output-reader"),
        "default",
        true,
    )
    .unwrap()
    .plaintext;
    assert_eq!(
        opened_default["default"]["default"]["report_html"],
        SELECTED_HTML
    );
    let envelope = require_envelope_shape(output.as_object().unwrap().clone()).unwrap();
    let blocks = extract_group_blocks(&envelope).unwrap();
    assert!(verify_sealed(&envelope, &blocks).row_hash);
    assert_eq!(
        release
            .trace
            .iter()
            .map(|event| event.event.as_str())
            .collect::<Vec<_>>(),
        [
            "admission",
            "governance_opened",
            "governance_echo_matched",
            "sections_selected",
            "tool_executed",
            "receipt_inserted",
            "output_emitted",
            "output_readback",
            "source_chain_committed",
            "release"
        ]
    );
    assert_eq!(
        release
            .trace
            .iter()
            .map(|event| event.ordinal)
            .collect::<Vec<_>>(),
        (1..=10).collect::<Vec<_>>()
    );
    assert!(release.trace[1..4]
        .iter()
        .all(|event| event.task_id.is_some() && event.slot_id.is_some()));
    let public_trace = serde_json::to_string(&release.trace).unwrap();
    for forbidden in [
        "123 Pine Ln",
        "P-98765",
        "FIRM-42",
        "wildfire_risk_score",
        "This record contains proprietary",
        "derived_html_report",
        "source_row_hash",
        "ciphertext",
        "field_hashes",
        "<tool_call>",
        "token",
        "btn.state",
    ] {
        assert!(
            !public_trace.contains(forbidden),
            "trace leaked {forbidden}"
        );
    }
    assert_eq!(harness.session.state(), tn_runtime::State::Complete);
    assert_eq!(
        harness.session.next_model_turn().unwrap_err().code(),
        "tn_session_state"
    );
    assert_eq!(
        harness.session.abort().unwrap_err().code(),
        "tn_session_state"
    );
}

#[test]
fn firm_only_path_emits_without_precisely_governance_tool_or_attribution() {
    let mut harness = SessionHarness::new("Generate an underwriting account report.");
    let selection_turn = advance_to_selection(&mut harness);
    prepare_and_begin(
        &selection_turn,
        b"rendered-section-selection-firm",
        2048,
        23,
    );
    let selection = harness.section_selection_result(&["firm_data"]);
    harness
        .session
        .accept_model_action(selection_turn.permit, selection.native())
        .unwrap();

    let emit_turn = harness.session.next_model_turn().unwrap();
    let body = String::from_utf8(emit_turn.request_json.clone()).unwrap();
    assert!(body.contains("FIRM-42"));
    assert!(!body.contains("123 Pine Ln"));
    assert!(!body.contains(ATTRIBUTION));
    assert!(!body.contains("log_compliance_creation"));
    assert!(!body.contains("required_source_lineage"));
    prepare_and_begin(&emit_turn, b"rendered-firm-emission", 2048, 24);
    let emission = harness.action_result(
        "call-emit-firm-4",
        "emit_governed_record",
        report_proposal(FIRM_HTML, None, None),
    );

    let ActionOutcome::Release(release) = harness
        .session
        .accept_model_action(emit_turn.permit, emission.native())
        .unwrap()
    else {
        panic!("firm-only emission did not release");
    };

    let output: Value = serde_json::from_str(release.output_ndjson.trim_end()).unwrap();
    assert!(output["default"].get("report_html").is_none());
    assert!(!release.output_ndjson.contains(FIRM_HTML));
    let opened_default = tn_core::unseal_as_recipient(
        release.output_ndjson.trim_end(),
        &harness._fixture.temp.path().join("output-reader"),
        "default",
        true,
    )
    .unwrap()
    .plaintext;
    assert_eq!(
        opened_default["default"]["default"]["report_html"],
        FIRM_HTML
    );
    let opened = tn_core::unseal_as_recipient(
        release.output_ndjson.trim_end(),
        &harness._fixture.temp.path().join("output-reader"),
        "tn.agents",
        true,
    )
    .unwrap()
    .plaintext;
    assert!(opened["tn.agents"].get("source_lineage").is_none());
    assert!(opened["tn.agents"].get("compliance_evidence").is_none());
    assert!(!release
        .trace
        .iter()
        .any(|event| event.event == "tool_executed"));
}

fn report_proposal(html: &str, lineage: Option<Value>, evidence: Option<Value>) -> Value {
    let mut governance = json!({
        "instruction": "Generate only the configured derived report.",
        "use_for": "Creating the governed HTML report requested by the user.",
        "do_not_use_for": "Any ungoverned output or unrelated purpose.",
        "consequences": "Reject the report and release no output when validation fails.",
        "on_violation_or_error": "Abort the governed session and record the failing gate."
    });
    if let Some(lineage) = lineage {
        governance["source_lineage"] = lineage;
    }
    if let Some(evidence) = evidence {
        governance["compliance_evidence"] = evidence;
    }
    json!({
        "event_type": "report.generated",
        "public_sections": {"default": {"report_html": html}},
        "governance": governance
    })
}

fn source_event_id(harness: &SessionHarness) -> String {
    let source: Value = serde_json::from_slice(&harness.raw_request).unwrap();
    source["tn_row"]["event_id"].as_str().unwrap().to_string()
}

fn selected_ready_to_emit() -> (SessionHarness, tn_runtime::ModelTurn, Value) {
    let mut harness = SessionHarness::new(USER_REQUEST);
    let selection_turn = advance_to_selection(&mut harness);
    prepare_and_begin(&selection_turn, b"rendered-selected-helper", 2048, 51);
    let selection = harness.section_selection_result(&["precisely"]);
    harness
        .session
        .accept_model_action(selection_turn.permit, selection.native())
        .unwrap();
    let compliance_turn = harness.session.next_model_turn().unwrap();
    prepare_and_begin(&compliance_turn, b"rendered-compliance-helper", 2048, 52);
    let compliance = harness.action_result(
        "helper-compliance",
        "log_compliance_creation",
        json!({
            "source_event_id": source_event_id(&harness),
            "action": "derived_html_report"
        }),
    );
    harness
        .session
        .accept_model_action(compliance_turn.permit, compliance.native())
        .unwrap();
    let emit_turn = harness.session.next_model_turn().unwrap();
    let body: Value = serde_json::from_slice(&emit_turn.request_json).unwrap();
    let receipt: Value = serde_json::from_str(
        body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .rfind(|message| message["role"] == "tool")
            .unwrap()["content"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    let source: Value = serde_json::from_slice(&harness.raw_request).unwrap();
    let lineage = json!({
        "source_device_identity": source["tn_row"]["device_identity"],
        "source_event_id": source["tn_row"]["event_id"],
        "source_event_type": source["tn_row"]["event_type"],
        "derived_from_row_hash": harness.row_hash,
        "source_policy": harness.governance["policy"],
        "selected_sections": ["precisely"],
        "operation": "derived_html_report"
    });
    let proposal = report_proposal(SELECTED_HTML, Some(lineage), Some(receipt));
    (harness, emit_turn, proposal)
}
