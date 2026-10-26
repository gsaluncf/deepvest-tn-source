mod governed_support;
mod support;
use governed_support::*;
use serde_json::{json, Value};
use tn_runtime::{ActionOutcome, State};

fn fixture() -> Fixture {
    let mut fixture = Fixture::new();
    fixture.require_compliance();
    fixture
}

fn args(request: &[u8]) -> Value {
    let request: Value = serde_json::from_slice(request).unwrap();
    let object =
        tn_core::governed::GovernedObject::parse(request["tn_object"].as_str().unwrap()).unwrap();
    json!({"source_object_id":object.id(),"action":"generate_report"})
}

#[test]
fn native_action_request_exposes_current_arguments_separately_from_ancestor_ids() {
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..2 {
        let fixture = fixture();
        let profile = fixture.config.governed_objects.as_ref().unwrap();
        let source = tn_core::runtime::Objects::open(&profile.native_reader_config_path).unwrap();
        let original_wire = fixture.manifest["publications"]["calculation"]["wire"]
            .as_str()
            .unwrap();
        let original_id = fixture.manifest["publications"]["calculation"]["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let mut data = source
            .receive_for(
                original_wire,
                &profile.use_context,
                ["finance"],
                None,
                |_| Ok(true),
            )
            .unwrap();
        let current = source
            .release_for(
                &mut data,
                &fixture.config.expected_event_type,
                &profile.use_context,
                "internal-client",
                |_| Ok(true),
            )
            .unwrap();
        let mut metadata: Value =
            serde_json::from_slice(&std::fs::read(&profile.metadata_store_path).unwrap()).unwrap();
        metadata["publications"]
            .as_array_mut()
            .unwrap()
            .push(json!(original_wire));
        std::fs::write(&profile.metadata_store_path, metadata.to_string()).unwrap();
        let mut request: Value = serde_json::from_slice(&fixture.request("calculation")).unwrap();
        request["tn_object"] = json!(current.wire());
        let request = serde_json::to_vec(&request).unwrap();
        assert!(
            seen.insert(current.id().to_owned()),
            "fresh native publication identity"
        );
        assert_ne!(current.id(), original_id);
        let (_keys, runtime) = fixture.runtime();
        let mut session = runtime.start_session(&request).unwrap();
        get_governance(&mut session).unwrap();
        echo(&mut session).unwrap();
        select(&mut session, &["finance"]).unwrap();
        let turn = session.next_model_turn().unwrap();
        let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
        let context: Value =
            serde_json::from_str(body["messages"][1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(context["trusted_source_binding"], args(&request));
        assert_eq!(
            context["required_tool_calls"],
            json!([{"name":"log_compliance_creation", "arguments":args(&request)}])
        );
        assert_eq!(
            context["applicable_source_governance"]["source_lineage"][0]["object_id"],
            original_id
        );
        let properties = &body["tools"][0]["function"]["parameters"]["properties"];
        assert!(properties["source_object_id"]["description"]
            .as_str()
            .unwrap()
            .contains(current.id()));
        assert!(properties["action"]["description"]
            .as_str()
            .unwrap()
            .contains("generate_report"));
        session.abort().unwrap();
    }
}

#[test]
fn native_tool_executes_once_and_signed_result_reenters_before_report() {
    let fixture = fixture();
    let request = fixture.request("calculation");
    let output_config = fixture.config.output_tn_config_path.clone();
    let use_context = fixture
        .config
        .governed_objects
        .as_ref()
        .unwrap()
        .use_context
        .clone();
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    get_governance(&mut session).unwrap();
    echo(&mut session).unwrap();
    select(&mut session, &["finance"]).unwrap();
    let turn = session.next_model_turn().unwrap();
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    assert_eq!(
        body["tools"][0]["function"]["name"],
        "log_compliance_creation"
    );
    assert_eq!(
        body["tools"][0]["function"]["parameters"]["properties"]["source_object_id"]["const"],
        args(&request)["source_object_id"]
    );
    assert!(matches!(
        accept(
            &mut session,
            turn,
            "log_compliance_creation",
            args(&request)
        )
        .unwrap(),
        ActionOutcome::Continue
    ));
    let turn = session.next_model_turn().unwrap();
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    assert_eq!(body["tools"][0]["function"]["name"], "emit_governed_record");
    let receipt: Value = serde_json::from_str(
        body["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        receipt["source_object_id"],
        args(&request)["source_object_id"]
    );
    assert_eq!(receipt["action"], "generate_report");
    assert_eq!(receipt["selected_sections"], json!(["finance"]));
    let ActionOutcome::GovernedRelease(release) = accept(
        &mut session,
        turn,
        "emit_governed_record",
        json!({"report_html":"<p>25% drawdown. Compliance creation recorded.</p>"}),
    )
    .unwrap() else {
        panic!("native release")
    };
    let release_json: Value = serde_json::from_slice(&release.to_json_bytes().unwrap()).unwrap();
    let wires = release_json["tool_objects"].as_array().unwrap();
    assert_eq!(wires.len(), 1);
    let object = tn_core::governed::GovernedObject::parse(wires[0].as_str().unwrap()).unwrap();
    assert_eq!(object.id(), receipt["output_object_id"]);
    assert_eq!(object.object_type(), "compliance.creation.logged");
    let objects = tn_core::runtime::Objects::open(&output_config).unwrap();
    let tool_data = objects
        .receive_for(object.wire(), &use_context, ["default"], None, |ctx| {
            assert_eq!(ctx.policies()?.len(), 2);
            assert_eq!(ctx.governance().dataset_bindings()?.len(), 2);
            assert_eq!(ctx.sources()?.len(), 1);
            assert_eq!(ctx.sources()?[0].object_id(), release.source_object_id);
            Ok(true)
        })
        .unwrap();
    let fields = tool_data.group("default").unwrap();
    for key in [
        "run_id",
        "request_digest",
        "source_object_id",
        "action",
        "selected_sections",
        "tool_name",
        "model_tool_call_id",
    ] {
        assert_eq!(fields[key], receipt[key], "scoped receipt {key}");
    }
    let report = objects
        .receive_for(
            &release.output_object,
            &use_context,
            ["default"],
            None,
            |ctx| {
                assert_eq!(ctx.policies()?.len(), 2);
                assert_eq!(ctx.governance().dataset_bindings()?.len(), 2);
                let sources = ctx.sources()?;
                assert_eq!(sources.len(), 2);
                assert!(sources
                    .iter()
                    .any(|source| source.object_id() == release.source_object_id));
                assert!(sources
                    .iter()
                    .any(|source| source.object_id() == object.id()));
                Ok(true)
            },
        )
        .unwrap();
    assert_eq!(report.group("default").unwrap().len(), 1);
    assert_eq!(
        release
            .trace
            .iter()
            .filter(|event| event.event == "tool_executed")
            .count(),
        1
    );
    assert_eq!(session.state(), State::Complete);
}

#[test]
fn native_tool_missing_wrong_arguments_repeated_and_forged_receipt_are_terminal() {
    for case in [
        "missing",
        "wrong_source",
        "wrong_action",
        "unknown",
        "extra_receipt",
        "repeated",
        "forged_report_receipt",
    ] {
        let fixture = fixture();
        let request = fixture.request("calculation");
        let (_keys, runtime) = fixture.runtime();
        let mut session = runtime.start_session(&request).unwrap();
        get_governance(&mut session).unwrap();
        echo(&mut session).unwrap();
        select(&mut session, &["finance"]).unwrap();
        let mut arguments = args(&request);
        let mut name = "log_compliance_creation";
        let expected = match case {
            "missing" => {
                name = "emit_governed_record";
                arguments = json!({"report_html":"<p>25%</p>"});
                "tn_required_tool_missing"
            }
            "wrong_source" => {
                arguments["source_object_id"] = json!("sha256:wrong");
                "tn_tool_arguments_invalid"
            }
            "wrong_action" => {
                arguments["action"] = json!("other");
                "tn_tool_arguments_invalid"
            }
            "unknown" => {
                name = "shell";
                "tn_action_not_allowed"
            }
            "extra_receipt" => {
                arguments["receipt"] = json!({"done":true});
                "tn_tool_arguments_invalid"
            }
            "repeated" | "forged_report_receipt" => {
                let turn = session.next_model_turn().unwrap();
                accept(&mut session, turn, name, arguments.clone()).unwrap();
                if case == "repeated" {
                    "tn_tool_already_executed"
                } else {
                    name = "emit_governed_record";
                    arguments =
                        json!({"report_html":"<p>25%</p>","compliance_evidence":{"done":true}});
                    "tn_report_proposal_invalid"
                }
            }
            _ => unreachable!(),
        };
        let turn = session.next_model_turn().unwrap();
        assert_eq!(
            accept(&mut session, turn, name, arguments)
                .unwrap_err()
                .code(),
            expected,
            "{case}"
        );
        assert_eq!(session.state(), State::Failed);
        assert!(runtime.start_session(&request).is_ok());
    }
}

#[test]
fn native_tool_readback_failure_never_reinserts_a_receipt_or_releases_report() {
    let fixture = fixture();
    let request = fixture.request("calculation");
    std::fs::copy(
        fixture.config.reader_key_dir.join("tn.agents.btn.mykit"),
        fixture
            .config
            .output_reader_key_dir
            .join("tn.agents.btn.mykit"),
    )
    .unwrap();
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    get_governance(&mut session).unwrap();
    echo(&mut session).unwrap();
    select(&mut session, &["finance"]).unwrap();
    let turn = session.next_model_turn().unwrap();
    assert_eq!(
        accept(
            &mut session,
            turn,
            "log_compliance_creation",
            args(&request)
        )
        .unwrap_err()
        .code(),
        "tn_output_readback_failed"
    );
    assert_eq!(session.state(), State::Failed);
    assert!(session.next_model_turn().is_err());
    assert!(runtime.start_session(&request).is_ok());
}
