mod governed_support;
mod support;
use governed_support::*;
use serde_json::{json, Value};
use tn_runtime::{GovernedRuntime, State};

#[test]
fn missing_configured_section_terminalizes_and_releases_capacity() {
    let mut fixture = Fixture::new();
    let request = fixture.request("deepvest");
    fixture
        .config
        .allowed_input_groups
        .insert("not_present".into());
    fixture.config.report_contract.allowed_input_sections = ["not_present".into()].into();
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    assert_eq!(
        get_governance(&mut session).unwrap_err().code(),
        "tn_section_selection_invalid"
    );
    assert_eq!(session.state(), State::Failed);
    assert!(runtime.start_session(&request).is_ok());
}

#[test]
fn absent_business_capability_cannot_reach_value_bearing_turn() {
    let fixture = Fixture::new();
    let request = fixture.request("deepvest");
    let path = &fixture
        .config
        .governed_objects
        .as_ref()
        .unwrap()
        .native_reader_config_path;
    let mut config: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    config["groups"].as_object_mut().unwrap().remove("finance");
    std::fs::write(path, config.to_string()).unwrap();
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    get_governance(&mut session).unwrap();
    echo(&mut session).unwrap();
    assert_eq!(
        select(&mut session, &["finance"]).unwrap_err().code(),
        "tn_governed_group_open_failed"
    );
    assert_eq!(session.state(), State::Failed);
    assert!(session.next_model_turn().is_err());
}

#[test]
fn metadata_extra_contract_requires_its_own_accepted_authority() {
    use tn_core::governed::{Governance, UseContext};
    let mut fixture = Fixture::new();
    let profile = fixture.config.governed_objects.as_mut().unwrap();
    let objects = tn_core::runtime::Objects::open(&profile.native_reader_config_path).unwrap();
    let foreign = tn_core::DeviceKey::generate();
    let text="---\nversion: 1\nschema: tn-agents-policy@v1\n---\n## report\n### instruction\nReport.\n### use_for\nResearch.\n### do_not_use_for\nDisclosure.\n### consequences\nReview.\n### on_violation_or_error\nRefuse.\n";
    let policy =
        Governance::from_markdown(foreign.did(), text, "foreign-policy", "report").unwrap();
    let mut data = objects
        .receive_for(
            fixture.manifest["publications"]["policy_earlier"]["wire"]
                .as_str()
                .unwrap(),
            &UseContext::new("fixture", "administration", "inspect").unwrap(),
            ["policy_revision"],
            None,
            |_| Ok(true),
        )
        .unwrap();
    objects.attach(&mut data, policy, |_| Ok(true)).unwrap();
    let changed = objects
        .release(
            &mut data,
            "tn.policy.revision",
            "administration",
            "catalog",
            |_| Ok(true),
        )
        .unwrap();
    profile.contracts.truncate(1);
    profile.contracts[0].revision_id = changed.id().to_owned();
    std::fs::write(
        &profile.metadata_store_path,
        json!({"policy_revisions":[changed.wire()],"dataset_editions":[],"publications":[]})
            .to_string(),
    )
    .unwrap();
    let result = GovernedRuntime::from_config(fixture.config);
    assert_eq!(
        result
            .err()
            .expect("foreign governing authority denied")
            .code(),
        "tn_governed_authority_denied"
    );
}

#[test]
fn all_contracts_are_accepted_before_echo_or_business_exposure() {
    let fixture = Fixture::new();
    let request = fixture.request("deepvest");
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    get_governance(&mut session).unwrap();
    echo(&mut session).unwrap();
    select(&mut session, &["finance"]).unwrap();
    let turn = session.next_model_turn().unwrap();
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    let context: Value =
        serde_json::from_str(body["messages"][1]["content"].as_str().unwrap()).unwrap();
    assert_eq!(
        context["selected_values"]["finance"]["calculation"]["result"],
        json!(0.25)
    );
    assert_eq!(
        context["applicable_source_governance"]["dataset_bindings"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(context["selected_values"].get("audit").is_none());
    assert!(runtime.start_session(&request).is_err());
    session.abort().unwrap();
    assert!(runtime.start_session(&request).is_ok());
}

#[test]
fn either_contract_can_refuse_and_capacity_is_released() {
    for denied in 0..2 {
        let mut fixture = Fixture::new();
        let request = fixture.request("deepvest");
        fixture.config.governed_objects.as_mut().unwrap().contracts[denied].allowed_uses =
            [tn_core::governed::UseContext::new(
                "another-app",
                "portfolio_analysis",
                "generate_report",
            )
            .unwrap()]
            .into();
        let (_keys, runtime) = fixture.runtime();
        let mut session = runtime.start_session(&request).unwrap();
        assert_eq!(
            get_governance(&mut session).unwrap_err().code(),
            "tn_governed_use_denied"
        );
        assert_eq!(session.state(), State::Failed);
        assert!(session.next_model_turn().is_err());
        assert!(runtime.start_session(&request).is_ok());
    }
}

#[test]
fn lost_contract_binding_or_ancestor_refuses_before_echo() {
    for publication in ["lost_contract", "lost_binding", "unreached_origins"] {
        let mut fixture = Fixture::new();
        let request = fixture.request(publication);
        fixture.config.expected_event_type = fixture.manifest["publications"][publication]
            ["object_type"]
            .as_str()
            .unwrap()
            .into();
        let (_keys, runtime) = fixture.runtime();
        let mut session = runtime.start_session(&request).unwrap();
        assert_eq!(
            get_governance(&mut session).unwrap_err().code(),
            "tn_governed_use_denied"
        );
        assert_eq!(session.state(), State::Failed);
    }
}

#[test]
fn altered_echo_or_unapproved_group_cannot_open_business_data() {
    for bad_echo in [true, false] {
        let fixture = Fixture::new();
        let request = fixture.request("deepvest");
        let (_keys, runtime) = fixture.runtime();
        let mut session = runtime.start_session(&request).unwrap();
        get_governance(&mut session).unwrap();
        if bad_echo {
            let turn = session.next_model_turn().unwrap();
            let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
            let mut args: Value = serde_json::from_str(
                body["messages"].as_array().unwrap().last().unwrap()["content"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            args["governance"]["dataset_bindings"] = json!([]);
            assert_eq!(
                accept(&mut session, turn, "continue_with_governance", args)
                    .unwrap_err()
                    .code(),
                "tn_governance_echo_mismatch"
            );
        } else {
            echo(&mut session).unwrap();
            assert!(select(&mut session, &["audit"]).is_err());
        }
        assert_eq!(session.state(), State::Failed);
    }
}

#[test]
fn metadata_substitution_and_wrong_authority_fail_startup() {
    for wrong_authority in [true, false] {
        let mut fixture = Fixture::new();
        if wrong_authority {
            fixture
                .config
                .governed_objects
                .as_mut()
                .unwrap()
                .authority_writers = ["did:key:wrong".into()].into();
        } else {
            let path = &fixture
                .config
                .governed_objects
                .as_ref()
                .unwrap()
                .metadata_store_path;
            let mut store: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            store["dataset_editions"][1] =
                fixture.manifest["publications"]["edition_earlier"]["wire"].clone();
            std::fs::write(path, store.to_string()).unwrap();
        }
        assert!(GovernedRuntime::from_config(fixture.config).is_err());
    }
}
