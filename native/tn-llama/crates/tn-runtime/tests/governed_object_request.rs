use serde_json::{json, Value};
use tn_runtime::{
    parse_and_verify_request, parse_governed_object_request, GovernedInputConfig, RequestLimits,
};

fn fixture() -> (Value, GovernedInputConfig, RequestLimits) {
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/governed/manifest.json")).unwrap();
    let config = serde_json::from_value(json!({
        "native_reader_config_path":"unused", "metadata_store_path":"unused",
        "use_context":manifest["uses"]["model"], "destinations":["internal-client"],
        "trusted_writers":[manifest["writer"]], "authority_writers":[manifest["writer"]], "contracts":[]
    })).unwrap();
    let mut limits = RequestLimits::test();
    limits.expected_event_type = "calculation.result".into();
    limits.allowed_input_groups = ["finance".into(), "audit".into()].into();
    let request = json!({"protocol":"tn.llama.request.v2", "tn_object":manifest["publications"]["calculation"]["wire"],
        "use":manifest["uses"]["model"],"destination":"internal-client","messages":[{"role":"user","content":"Explain the result."}],"stream":false});
    (request, config, limits)
}

#[test]
fn exact_wire_and_native_identity_survive_transport() {
    let (request, config, limits) = fixture();
    let accepted =
        parse_governed_object_request(&serde_json::to_vec(&request).unwrap(), &limits, &config)
            .unwrap();
    assert_eq!(
        accepted.object.wire(),
        request["tn_object"].as_str().unwrap()
    );
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/governed/manifest.json")).unwrap();
    assert_eq!(
        accepted.object.id(),
        manifest["publications"]["calculation"]["object_id"]
            .as_str()
            .unwrap()
    );
    assert!(parse_and_verify_request(&serde_json::to_vec(&request).unwrap(), &limits).is_err());
}

#[test]
fn closed_profile_refuses_use_changes_types_and_tampering() {
    let (request, config, limits) = fixture();
    for (pointer, value) in [
        ("/use/application", json!("claimed-app")),
        ("/use/purpose", json!("advertising")),
        ("/use/operation", json!("redistribute")),
        ("/destination", json!("public")),
        ("/stream", json!(true)),
        ("/protocol", json!("tn.llama.request.v1")),
        ("/tn_object", json!({})),
        ("/messages/0/role", json!("system")),
    ] {
        let mut bad = request.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            parse_governed_object_request(&serde_json::to_vec(&bad).unwrap(), &limits, &config)
                .is_err(),
            "{pointer}"
        );
    }
    let mut extra = request.clone();
    extra["temperature"] = json!(1);
    assert!(
        parse_governed_object_request(&serde_json::to_vec(&extra).unwrap(), &limits, &config)
            .is_err()
    );
    let raw = serde_json::to_string(&request).unwrap();
    let duplicate = format!("{{\"stream\":false,{}", &raw[1..]);
    assert!(parse_governed_object_request(duplicate.as_bytes(), &limits, &config).is_err());
    let mut changed = request.clone();
    let mut object: Value = serde_json::from_str(changed["tn_object"].as_str().unwrap()).unwrap();
    object["event_type"] = json!("calculation.forged");
    changed["tn_object"] = json!(object.to_string());
    assert!(parse_governed_object_request(
        &serde_json::to_vec(&changed).unwrap(),
        &limits,
        &config
    )
    .is_err());
}
