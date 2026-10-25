mod governed_support;
mod support;

use governed_support::*;
use serde_json::{json, Value};
use tn_runtime::{ActionOutcome, GovernedRuntime};

#[test]
fn runtime_register_outcomes_are_isolated_and_preserve_the_signed_release() {
    for mode in ["enabled", "disabled", "unwritable"] {
        let directory = tempfile::tempdir().unwrap();
        let release = directory.path().join("release.ndjson");
        let creation = directory.path().join("creation.ndjson");
        if mode == "unwritable" {
            std::fs::create_dir(&release).unwrap();
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "register_child", "--nocapture"])
            .env("TN_LLAMA_REGISTER_TEST_MODE", mode)
            .env("TN_LLAMA_REGISTER_TEST_ROOT", directory.path())
            .env_remove("TN_OBJECT_CREATION_REGISTER")
            .env_remove("TN_OBJECT_RELEASE_REGISTER");
        if mode != "disabled" {
            command.env("TN_OBJECT_CREATION_REGISTER", &creation);
            command.env("TN_OBJECT_RELEASE_REGISTER", &release);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn register_child() {
    let Ok(mode) = std::env::var("TN_LLAMA_REGISTER_TEST_MODE") else {
        return;
    };
    let directory =
        std::path::PathBuf::from(std::env::var_os("TN_LLAMA_REGISTER_TEST_ROOT").unwrap());
    let fixture = Fixture::new();
    let request = fixture.request("deepvest");
    let audit = fixture.config.audit_path.clone();
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
    let ActionOutcome::GovernedRelease(release) = accept(
        &mut session,
        turn,
        "emit_governed_record",
        json!({"report_html":"<p>25%.</p>"}),
    )
    .unwrap() else {
        panic!("release")
    };
    let object = tn_core::governed::GovernedObject::parse(&release.output_object).unwrap();
    assert_eq!(object.id(), release.output_object_id);
    assert!(
        !directory.join("creation.ndjson").exists(),
        "receive/release does not originate an object"
    );
    let events: Vec<Value> = std::fs::read_to_string(audit)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let event = events
        .iter()
        .find(|event| event["kind"] == "object_release_register")
        .expect("native operational outcome must be audited");
    assert_eq!(event["target_row_hash"], release.output_object_id);
    if mode == "unwritable" {
        assert_eq!(event["outcome"], "error");
        assert_eq!(event["error_code"], "tn_object_register_failed");
    } else {
        assert_eq!(
            event["outcome"],
            if mode == "enabled" {
                "recorded"
            } else {
                "disabled"
            }
        );
        assert!(event["error_code"].is_null());
    }
    if mode == "enabled" {
        let rows = std::fs::read_to_string(directory.join("release.ndjson")).unwrap();
        let rows: Vec<Value> = rows
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            rows.len(),
            1,
            "one release, never regenerated for registration"
        );
        assert_eq!(rows[0]["object_id"], release.output_object_id);
        let verified = tn_core::sealed_object::verify_sealed(
            rows[0].as_object().unwrap(),
            &std::collections::BTreeMap::new(),
        );
        assert!(verified.signature && verified.row_hash);
        let objects = tn_core::runtime::Objects::open(&output_config).unwrap();
        let data = objects
            .receive_for(object.wire(), &use_context, ["default"], None, |_| Ok(true))
            .unwrap();
        let created = objects
            .create_obj(
                "register.test",
                data.policies().unwrap()[0].clone(),
                "default",
                json!({"synthetic":true}),
            )
            .unwrap();
        assert!(created.register_error().is_none());
        let created_rows = std::fs::read_to_string(directory.join("creation.ndjson")).unwrap();
        assert_eq!(created_rows.lines().count(), 1);
        let row: Value = serde_json::from_str(created_rows.lines().next().unwrap()).unwrap();
        assert_eq!(row["object_id"], created.snapshot().unwrap().id());
        assert_eq!(row["event_type"], "tn.object.created");
        let verified = tn_core::sealed_object::verify_sealed(
            row.as_object().unwrap(),
            &std::collections::BTreeMap::new(),
        );
        assert!(verified.signature && verified.row_hash);
    } else if mode == "disabled" {
        assert!(!directory.join("release.ndjson").exists());
    }
}

#[test]
fn v2_rejects_false_source_governance_compatibility_assertion() {
    let mut fixture = Fixture::new();
    fixture.config.report_contract.section_rules.insert(
        "finance".into(),
        tn_runtime::SectionRule {
            propagate_source_governance: false,
            required_attribution: None,
            required_tool: None,
        },
    );
    assert!(GovernedRuntime::from_config(fixture.config).is_err());
}

#[test]
fn legacy_rejects_nondefault_public_section_before_session_start() {
    let fixture = tn_test_tools::build_real_tn_fixture().unwrap();
    let mut config = support::fixture_config(&fixture);
    config.report_contract.output_public_section = "custom".into();
    assert!(GovernedRuntime::from_config(config).is_err());
}
