mod governed_support;
mod support;
use governed_support::{accept, get_governance, Fixture};
use serde_json::{json, Value};
use tn_runtime::{ActionOutcome, State};

#[test]
fn structured_governance_echo_describes_nested_records_without_scalar_answers() {
    let fixture = Fixture::new();
    let request = fixture.request("deepvest");
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    get_governance(&mut session).unwrap();
    let turn = session.next_model_turn().unwrap();
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    if let Some(path) = std::env::var_os("TN_ECHO_SCHEMA_INSPECTION") {
        std::fs::write(path, &turn.request_json).unwrap();
    }
    let schema = &body["tools"][0]["function"]["parameters"];
    let governance = &schema["properties"]["governance"]["properties"];
    assert_eq!(governance["dataset_bindings"]["items"]["type"], "object");
    assert_eq!(
        governance["dataset_bindings"]["items"]["additionalProperties"],
        false
    );
    assert_eq!(
        governance["dataset_bindings"]["items"]["properties"]["contracts"]["items"]["type"],
        "object"
    );
    assert_eq!(
        governance["release_context"]["properties"]["operation"],
        json!({"type":"string"})
    );
    assert_eq!(governance["source_lineage"]["items"]["type"], "object");
    let expected: Value =
        serde_json::from_str(body["messages"][3]["content"].as_str().unwrap()).unwrap();
    assert_eq!(
        governance["dataset_bindings"]["minItems"],
        json!(expected["governance"]["dataset_bindings"]
            .as_array()
            .unwrap()
            .len())
    );
    assert!(!schema.to_string().contains("\"const\""));
    assert!(!schema.to_string().contains("\"enum\""));
    assert!(matches!(
        accept(&mut session, turn, "continue_with_governance", expected).unwrap(),
        ActionOutcome::Continue
    ));
    assert_eq!(session.state(), State::SectionSelection);
}

#[test]
fn nested_scalar_substitution_still_fails_the_native_exact_comparison() {
    let fixture = Fixture::new();
    let request = fixture.request("deepvest");
    let (_keys, runtime) = fixture.runtime();
    let mut session = runtime.start_session(&request).unwrap();
    get_governance(&mut session).unwrap();
    let turn = session.next_model_turn().unwrap();
    let body: Value = serde_json::from_slice(&turn.request_json).unwrap();
    let mut expected: Value =
        serde_json::from_str(body["messages"][3]["content"].as_str().unwrap()).unwrap();
    expected["governance"]["release_context"]["operation"] = json!("substituted");
    assert!(accept(&mut session, turn, "continue_with_governance", expected).is_err());
    assert_eq!(session.state(), State::Failed);
}
