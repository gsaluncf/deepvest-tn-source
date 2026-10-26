mod governed_support;
mod support;
use governed_support::*;
use serde_json::json;
use tn_runtime::ActionOutcome;

#[test]
fn independent_output_reader_failure_releases_no_wire() {
    let fixture = Fixture::new();
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
            "emit_governed_record",
            json!({"report_html":"<p>25%.</p>"})
        )
        .unwrap_err()
        .code(),
        "tn_output_readback_failed"
    );
    assert!(runtime.start_session(&request).is_ok());
}

#[test]
fn v2_reuses_visible_and_unselected_attribution_gates() {
    for (selected, html, code) in [
        (
            true,
            "<script>Required credit</script>",
            "tn_required_attribution_missing",
        ),
        (
            true,
            "<p hidden>Required credit</p>",
            "tn_required_attribution_missing",
        ),
        (
            false,
            "<p>Required credit</p>",
            "tn_unselected_attribution_present",
        ),
    ] {
        let mut fixture = Fixture::new();
        let request = fixture.request("calculation");
        let section = if selected { "finance" } else { "audit" };
        fixture
            .config
            .report_contract
            .allowed_input_sections
            .insert(section.into());
        fixture.config.report_contract.section_rules.insert(
            section.into(),
            tn_runtime::SectionRule {
                propagate_source_governance: true,
                required_attribution: Some("Required credit".into()),
                required_tool: None,
            },
        );
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
                "emit_governed_record",
                json!({"report_html":html})
            )
            .unwrap_err()
            .code(),
            code
        );
    }
}

#[test]
fn configured_report_revision_is_attached_without_replacing_inherited_policies() {
    use tn_core::governed::{ContractBinding, Governance, PolicyRevisionDraft};
    let mut fixture = Fixture::new();
    let request = fixture.request("calculation");
    let output_path = fixture.config.output_tn_config_path.clone();
    let profile = fixture.config.governed_objects.as_mut().unwrap();
    let source = tn_core::runtime::Objects::open(&profile.native_reader_config_path).unwrap();
    let text="---\nversion: 1\nschema: tn-agents-policy@v1\n---\n## report\n### instruction\nExplain approved results.\n### use_for\nPortfolio research.\n### do_not_use_for\nIndividual disclosure.\n### consequences\nReview.\n### on_violation_or_error\nRefuse.\n";
    let administration =
        Governance::from_markdown(source.did(), text, "report-admin", "report").unwrap();
    let revision = source
        .seal(
            PolicyRevisionDraft::from_markdown(
                source.did(),
                text,
                "report-policy",
                "report",
                "report",
            )
            .unwrap()
            .into_draft(administration)
            .unwrap(),
        )
        .unwrap();
    let mut store: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&profile.metadata_store_path).unwrap()).unwrap();
    store["policy_revisions"]
        .as_array_mut()
        .unwrap()
        .push(json!(revision.wire()));
    std::fs::write(&profile.metadata_store_path, store.to_string()).unwrap();
    profile.contracts.push(tn_runtime::ContractPermission {
        revision_id: revision.id().into(),
        scope: "report".into(),
        allowed_uses: [profile.use_context.clone()].into(),
    });
    profile.report_contract = Some(ContractBinding::new(revision.id(), "report").unwrap());
    let model_use = profile.use_context.clone();
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    get_governance(&mut session).unwrap();
    echo(&mut session).unwrap();
    select(&mut session, &["finance"]).unwrap();
    let turn = session.next_model_turn().unwrap();
    let ActionOutcome::GovernedRelease(release) = accept(
        &mut session,
        turn,
        "emit_governed_record",
        json!({"report_html":"<p>25% drawdown.</p>"}),
    )
    .unwrap() else {
        panic!("v2 release")
    };
    let output = tn_core::runtime::Objects::open(&output_path).unwrap();
    let data = output
        .receive_for(
            &release.output_object,
            &model_use,
            ["default"],
            None,
            |_| Ok(true),
        )
        .unwrap();
    assert_eq!(data.policies().unwrap().len(), 3);
    assert!(data
        .policies()
        .unwrap()
        .iter()
        .any(|policy| policy.revision_id() == Some(revision.id())));
    assert_eq!(data.dataset_bindings().unwrap().len(), 2);
}

#[test]
fn native_report_retains_both_contracts_bindings_and_exact_parent() {
    let fixture = Fixture::new();
    let request = fixture.request("calculation");
    let source = tn_core::governed::GovernedObject::parse(
        fixture.manifest["publications"]["calculation"]["wire"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    let output_path = fixture.config.output_tn_config_path.clone();
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
    let outcome = accept(
        &mut session,
        turn,
        "emit_governed_record",
        json!({"report_html":"<p>Maximum drawdown: 25%.</p>"}),
    )
    .unwrap();
    let ActionOutcome::GovernedRelease(release) = outcome else {
        panic!("v2 release required")
    };
    assert_eq!(release.protocol, "tn.llama.release.v2");
    assert_eq!(release.source_object_id, source.id());
    let object = tn_core::governed::GovernedObject::parse(&release.output_object).unwrap();
    assert_eq!(release.output_object_id, object.id());
    assert_eq!(object.group_names(), vec!["default", "tn.agents"]);
    assert_ne!(object.writer(), source.writer());
    let output = tn_core::runtime::Objects::open(&output_path).unwrap();
    let data = output
        .receive_for(object.wire(), &use_context, ["default"], None, |ctx| {
            assert_eq!(ctx.policies()?.len(), 2);
            assert_eq!(ctx.governance().dataset_bindings()?.len(), 2);
            assert_eq!(ctx.sources()?.len(), 1);
            assert_eq!(ctx.sources()?[0].object_id(), source.id());
            assert_eq!(
                ctx.governance().fields()["release_context"]["destination"],
                "internal-client"
            );
            Ok(true)
        })
        .unwrap();
    assert_eq!(
        data.group("default").unwrap()["report_html"],
        "<p>Maximum drawdown: 25%.</p>"
    );
    assert!(data.hidden_groups().is_empty());
    assert!(release
        .trace
        .iter()
        .all(|event| event.event != "source_chain_committed"));
    assert!(runtime.start_session(&request).is_ok());
}

#[test]
fn model_cannot_supply_governance_or_lineage() {
    let fixture = Fixture::new();
    let request = fixture.request("calculation");
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
            "emit_governed_record",
            json!({"report_html":"<p>Report.</p>","governance":{}})
        )
        .unwrap_err()
        .code(),
        "tn_report_proposal_invalid"
    );
    assert!(runtime.start_session(&request).is_ok());
}
