use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Map, Value};
use tn_core::chain::{compute_row_hash, RowHashInput, ZERO_HASH};
use tn_core::sealed_object::{
    extract_group_blocks, require_envelope_shape, verify_sealed, ENVELOPE_RESERVED,
};

const LLAMA_REVISION: &str = "3f545beccee69d9975f466ec7e45fd9aacd8ba90";
const MODEL_SHA256: &str = "9a7bed0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
const BINARY_SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn external_cli_creates_a_closed_real_tn_fixture_and_prints_only_the_manifest_path() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fixture");

    let output = create_fixture(&out);

    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    assert!(output.stderr.is_empty());
    let expected_manifest = out.canonicalize().unwrap().join("manifest.json");
    assert_eq!(
        text(&output.stdout),
        format!("{}\n", expected_manifest.display())
    );

    let manifest: Value = serde_json::from_slice(&fs::read(&expected_manifest).unwrap()).unwrap();
    let manifest = manifest.as_object().unwrap();
    assert_eq!(manifest["schema"], "tn.llama.fixture.v1");
    assert_eq!(
        manifest.keys().cloned().collect::<BTreeSet<_>>(),
        [
            "aad_mutation_request_path",
            "audit_path",
            "ciphertext_mutation_request_path",
            "commitment_mutation_request_path",
            "expected_governance_path",
            "extra_governance_commitment_mutation_request_path",
            "extra_governance_field_request_path",
            "missing_governance_field_request_path",
            "plaintext_report_request_path",
            "plaintext_source_row1_path",
            "plaintext_source_row2_path",
            "raw_trace_dir",
            "row1_request_path",
            "row2_request_path",
            "runtime_config_path",
            "schema",
            "signature_mutation_request_path",
            "stale_fork_request_path",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    );

    let canonical_out = out.canonicalize().unwrap();
    for (name, value) in manifest {
        if name == "schema" {
            continue;
        }
        let path = PathBuf::from(value.as_str().unwrap());
        assert!(path.is_absolute(), "{name} is not absolute");
        let comparable = if path.exists() {
            path.canonicalize().unwrap()
        } else {
            path.clone()
        };
        assert!(
            comparable.starts_with(&canonical_out),
            "{name} escaped output"
        );
    }

    assert_eq!(
        fs::read(manifest_path(manifest, "audit_path")).unwrap(),
        b""
    );
    assert_eq!(
        fs::read_dir(manifest_path(manifest, "raw_trace_dir"))
            .unwrap()
            .count(),
        0
    );
    assert_output_writer_material_is_confined(&canonical_out);
}

#[test]
fn source_rows_are_signed_and_encrypt_every_protected_group_with_canonical_aad() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fixture");
    let output = create_fixture(&out);
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let manifest = manifest.as_object().unwrap();

    let row1 = request_row(manifest_path(manifest, "row1_request_path"));
    let row2 = request_row(manifest_path(manifest, "row2_request_path"));
    let protected_groups = ["default", "firm_data", "precisely", "tn.agents"];

    for (name, row) in [("row1", &row1), ("row2", &row2)] {
        let envelope = require_envelope_shape(row.clone()).unwrap();
        let blocks = extract_group_blocks(&envelope).unwrap();
        let verification = verify_sealed(&envelope, &blocks);
        assert!(verification.row_hash, "{name} row hash");
        assert!(verification.signature, "{name} signature");
        assert!(row["signature"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert_eq!(
            blocks.keys().map(String::as_str).collect::<Vec<_>>(),
            protected_groups,
            "{name} encrypted groups"
        );

        for group in protected_groups {
            assert!(
                row[group]["ciphertext"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "{name} {group} ciphertext"
            );
            assert!(
                row[group]["field_hashes"].is_object(),
                "{name} {group} commitments"
            );
        }
        for plaintext in [
            "demo-001",
            "P-98765",
            "123 Pine Ln, CA",
            "FIRM-42",
            "underwriting",
            "Precisely Wildfire Risk",
        ] {
            assert!(
                !serde_json::to_string(row).unwrap().contains(plaintext),
                "{name} leaked {plaintext:?}"
            );
        }

        let aad_text = row["tn_aad"].as_str().expect("tn_aad string");
        let aad: Value = serde_json::from_str(aad_text).unwrap();
        let canonical = tn_core::canonical::canonical_bytes(&aad).unwrap();
        assert_eq!(aad_text.as_bytes(), canonical, "{name} canonical AAD echo");
        let governance_marker = aad["tn.agents"].clone();
        assert_eq!(
            governance_marker
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["governed_by", "policy"]
        );
        tn_core::trust::parse_ed25519_did_key(governance_marker["governed_by"].as_str().unwrap())
            .expect("governing DID must be canonical");
        assert_ne!(
            governance_marker["governed_by"], row["device_identity"],
            "{name} governing authority must not be conflated with the row signer"
        );
        assert!(
            governance_marker["policy"]
                .as_str()
                .is_some_and(|value| value
                    .starts_with(".tn/config/agents.md#precisely.risk.enriched@1#sha256:")),
            "{name} content-addressed governance policy"
        );
        for group in protected_groups {
            assert_eq!(
                aad[group], governance_marker,
                "{name} {group} must carry the same governance marker"
            );
        }
    }

    assert_eq!(row1["prev_hash"], ZERO_HASH);
    assert_eq!(row1["sequence"], 1);
    assert_eq!(row2["prev_hash"], row1["row_hash"]);
    assert_eq!(row2["sequence"], 2);
}

#[test]
fn installed_group_kits_strictly_open_each_group_and_match_complete_commitment_names() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fixture");
    let output = create_fixture(&out);
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let manifest = manifest.as_object().unwrap();
    let config: Value =
        serde_json::from_slice(&fs::read(manifest_path(manifest, "runtime_config_path")).unwrap())
            .unwrap();
    let reader = PathBuf::from(config["reader_key_dir"].as_str().unwrap());
    let row = request_row(manifest_path(manifest, "row1_request_path"));

    let default = open_group(&row, &reader, "default");
    assert_eq!(default["request_id"], "demo-001");
    assert!(default["run_id"]
        .as_str()
        .is_some_and(|value| !value.is_empty()));
    let precisely = open_group(&row, &reader, "precisely");
    assert_eq!(
        precisely,
        serde_json::json!({
            "address": "123 Pine Ln, CA",
            "property_id": "P-98765",
            "wildfire_risk_score": 92
        })
    );
    let firm = open_group(&row, &reader, "firm_data");
    assert_eq!(
        firm,
        serde_json::json!({
            "customer_account": "FIRM-42",
            "report_purpose": "underwriting"
        })
    );
    let governance = open_group(&row, &reader, "tn.agents");
    assert_eq!(
        governance["governs_sections"],
        serde_json::json!(["precisely"])
    );
    assert!(governance["policy"].as_str().is_some_and(
        |value| value.starts_with(".tn/config/agents.md#precisely.risk.enriched@1#sha256:")
    ));

    for (group, opened) in [
        ("default", default),
        ("firm_data", firm),
        ("precisely", precisely),
        ("tn.agents", governance),
    ] {
        let committed_names = row[group]["field_hashes"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let opened_names = opened
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        assert_eq!(committed_names, opened_names, "{group} commitment coverage");
    }
}

#[test]
fn fixture_rows_are_real_outer_valid_tn_outputs_with_exact_negative_mutations() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fixture");
    let output = create_fixture(&out);
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let manifest = manifest.as_object().unwrap();

    let row1 = request_row(manifest_path(manifest, "row1_request_path"));
    let row2 = request_row(manifest_path(manifest, "row2_request_path"));
    let stale = request_row(manifest_path(manifest, "stale_fork_request_path"));
    let signature = request_row(manifest_path(manifest, "signature_mutation_request_path"));
    let commitment = request_row(manifest_path(manifest, "commitment_mutation_request_path"));
    let aad = request_row(manifest_path(manifest, "aad_mutation_request_path"));
    let ciphertext = request_row(manifest_path(manifest, "ciphertext_mutation_request_path"));
    let missing = request_row(manifest_path(
        manifest,
        "missing_governance_field_request_path",
    ));
    let extra = request_row(manifest_path(
        manifest,
        "extra_governance_field_request_path",
    ));
    let extra_commitment = request_row(manifest_path(
        manifest,
        "extra_governance_commitment_mutation_request_path",
    ));
    let plaintext_request: Value = serde_json::from_slice(
        &fs::read(manifest_path(manifest, "plaintext_report_request_path")).unwrap(),
    )
    .unwrap();
    let plaintext = plaintext_request["tn_row"].as_object().unwrap();

    for (name, row) in [
        ("row1", &row1),
        ("row2", &row2),
        ("stale fork", &stale),
        ("commitment mutation", &commitment),
        ("AAD mutation", &aad),
        ("ciphertext mutation", &ciphertext),
        ("missing governance field", &missing),
        ("extra governance field", &extra),
        ("extra governance commitment mutation", &extra_commitment),
    ] {
        assert_outer_sealed_valid(name, row);
        assert_eq!(row["event_type"], "precisely.risk.enriched", "{name}");
    }

    let envelope = require_envelope_shape(signature.clone()).unwrap();
    let blocks = extract_group_blocks(&envelope).unwrap();
    let verification = verify_sealed(&envelope, &blocks);
    assert!(verification.row_hash, "signature mutation row hash");
    assert!(
        !verification.signature,
        "signature mutation unexpectedly verified"
    );

    assert_eq!(row1["prev_hash"], ZERO_HASH);
    assert_eq!(row1["sequence"], 1);
    assert_eq!(row2["prev_hash"], row1["row_hash"]);
    assert_eq!(row2["sequence"], 2);
    assert_eq!(stale["prev_hash"], row1["row_hash"]);
    assert_only_paths_changed(&row1, &signature, &["/signature"]);

    assert_only_paths_changed(&row2, &stale, &["/event_id", "/row_hash", "/signature"]);
    assert_ne!(row2["event_id"], stale["event_id"]);

    assert_only_paths_changed(
        &row1,
        &commitment,
        &["/row_hash", "/signature", "/precisely/field_hashes/address"],
    );
    assert_ne!(
        row1["precisely"]["field_hashes"]["address"],
        commitment["precisely"]["field_hashes"]["address"]
    );

    assert_only_paths_changed(&row1, &aad, &["/row_hash", "/signature", "/tn_aad"]);
    let aad_marker: Value = serde_json::from_str(aad["tn_aad"].as_str().unwrap()).unwrap();
    let source_aad: Value = serde_json::from_str(row1["tn_aad"].as_str().unwrap()).unwrap();
    assert_ne!(
        aad_marker["tn.agents"]["governed_by"],
        source_aad["tn.agents"]["governed_by"]
    );
    for group in ["default", "firm_data", "precisely", "tn.agents"] {
        assert_eq!(
            aad_marker[group]["governed_by"],
            aad_marker["tn.agents"]["governed_by"]
        );
    }

    assert_only_paths_changed(
        &row1,
        &ciphertext,
        &["/row_hash", "/signature", "/precisely/ciphertext"],
    );
    assert_ne!(
        row1["precisely"]["ciphertext"],
        ciphertext["precisely"]["ciphertext"]
    );

    assert!(missing["tn.agents"]["field_hashes"].get("policy").is_none());
    assert!(missing["default"]["field_hashes"]["policy"].is_string());
    assert!(extra["tn.agents"]["field_hashes"]["governance_revision"].is_string());
    assert_only_paths_changed(
        &extra,
        &extra_commitment,
        &[
            "/row_hash",
            "/signature",
            "/tn.agents/field_hashes/governance_revision",
        ],
    );

    assert_eq!(
        plaintext_request["messages"][0]["content"],
        "Generate a concise HTML wildfire-risk report."
    );
    assert!(plaintext["signature"]
        .as_str()
        .is_some_and(|value| !value.is_empty()));
    assert!(plaintext["tn.agents"].get("ciphertext").is_none());
    assert!(plaintext["tn.agents"].get("field_hashes").is_none());
    for field in [
        "instruction",
        "use_for",
        "do_not_use_for",
        "consequences",
        "on_violation_or_error",
        "policy",
    ] {
        assert!(
            plaintext["tn.agents"][field]
                .as_str()
                .is_some_and(|value| !value.is_empty()),
            "missing plaintext governance field {field}"
        );
    }
    assert_eq!(
        plaintext["tn.agents"]["governs_sections"],
        serde_json::json!(["precisely"])
    );
    assert_plaintext_row_hash_valid(plaintext);
}

#[test]
fn fixture_emits_two_sequential_plaintext_source_rows_with_official_hashes() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fixture");
    let output = create_fixture(&out);
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let manifest = manifest.as_object().unwrap();

    let row1 = plaintext_source_row(manifest_path(manifest, "plaintext_source_row1_path"));
    let row2 = plaintext_source_row(manifest_path(manifest, "plaintext_source_row2_path"));

    for (name, row) in [("row 1", &row1), ("row 2", &row2)] {
        assert_eq!(row["event_type"], "precisely.risk.enriched", "{name}");
        assert!(
            row["tn.agents"].is_object(),
            "{name} governance is not plaintext"
        );
        assert!(row["tn.agents"].get("ciphertext").is_none(), "{name}");
        assert!(row["tn.agents"].get("field_hashes").is_none(), "{name}");
        assert_eq!(
            row["tn.agents"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            [
                "consequences",
                "do_not_use_for",
                "governs_sections",
                "instruction",
                "on_violation_or_error",
                "policy",
                "use_for",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            "{name} governance fields"
        );
        assert!(row["signature"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert_plaintext_row_hash_valid(row);
    }

    assert_eq!(row1["prev_hash"], ZERO_HASH);
    assert_eq!(row2["prev_hash"], row1["row_hash"]);
}

#[test]
fn fixture_runtime_config_uses_group_capability_authority_and_exact_governance_provenance() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fixture");
    let output = create_fixture(&out);
    assert!(output.status.success(), "stderr: {}", text(&output.stderr));
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let manifest = manifest.as_object().unwrap();
    let row1 = request_row(manifest_path(manifest, "row1_request_path"));
    let config: Value =
        serde_json::from_slice(&fs::read(manifest_path(manifest, "runtime_config_path")).unwrap())
            .unwrap();

    assert_eq!(config["version"], "tn.llama.runtime.v2");
    assert_eq!(config["writer_authority"], "group_capability");
    assert!(config.get("expected_device_identity").is_none());
    assert!(row1["device_identity"]
        .as_str()
        .is_some_and(|value| value.starts_with("did:key:")));
    assert_eq!(config["expected_event_type"], "precisely.risk.enriched");
    assert_eq!(
        config["allowed_input_groups"],
        serde_json::json!(["default", "firm_data", "precisely"])
    );
    assert_eq!(
        config["report_contract"]["allowed_input_sections"],
        serde_json::json!(["default", "firm_data", "precisely"])
    );
    assert_eq!(
        config["report_contract"]["output_event_type"],
        "report.generated"
    );
    assert_eq!(
        config["report_contract"]["output_public_section"],
        "default"
    );
    assert_eq!(
        config["report_contract"]["operation"],
        "derived_html_report"
    );
    assert_eq!(
        config["report_contract"]["section_rules"]["precisely"]["required_attribution"],
        "Source: Risk assessment derived from Precisely Wildfire Risk Data."
    );
    assert_eq!(
        config["report_contract"]["section_rules"]["precisely"]["required_tool"],
        "log_compliance_creation"
    );
    assert_eq!(config["governance_group"], "tn.agents");
    assert_eq!(config["input_index_epoch"], 0);
    assert_eq!(config["output_governance_group"], "tn.agents");
    assert_eq!(config["output_index_epoch"], 0);
    assert_eq!(config["max_chain_namespaces"], 8);
    let canonical_out = out.canonicalize().unwrap();
    for field in [
        "reader_key_dir",
        "input_index_key_path",
        "output_tn_config_path",
        "output_reader_key_dir",
        "output_index_key_path",
    ] {
        let path = PathBuf::from(config[field].as_str().unwrap());
        assert!(path.is_absolute(), "{field} is not absolute");
        assert!(path.exists(), "{field} does not exist");
        assert!(path.canonicalize().unwrap().starts_with(&canonical_out));
    }
    assert_eq!(config["llama_source_revision"], LLAMA_REVISION);
    assert_eq!(config["model_sha256"], MODEL_SHA256);
    assert_eq!(config["binary_sha256"], BINARY_SHA256);

    let reader = PathBuf::from(config["reader_key_dir"].as_str().unwrap());
    for group in ["default", "firm_data", "precisely", "tn.agents"] {
        assert!(reader.join(format!("{group}.btn.mykit")).is_file());
    }
    let output_reader = PathBuf::from(config["output_reader_key_dir"].as_str().unwrap());
    for group in ["default", "tn.agents"] {
        assert!(output_reader.join(format!("{group}.btn.mykit")).is_file());
    }

    let expected: Value = serde_json::from_slice(
        &fs::read(manifest_path(manifest, "expected_governance_path")).unwrap(),
    )
    .unwrap();
    assert_eq!(expected.as_object().unwrap().len(), 7);
    assert_eq!(
        expected["instruction"],
        "This record contains proprietary Precisely Wildfire Risk data in the encrypted `precisely` section."
    );
    assert_eq!(
        expected["use_for"],
        "Use the `precisely` section for property reporting, underwriting, and risk assessment. Any derived report that uses this section must carry a human-readable statement attributing its data source to Precisely."
    );
    assert_eq!(
        expected["do_not_use_for"],
        "Do not use the `precisely` section for unattributed public disclosure or strip its source lineage."
    );
    assert_eq!(
        expected["consequences"],
        "Any output derived from the `precisely` section must carry this governance and exact source lineage."
    );
    assert_eq!(
        expected["on_violation_or_error"],
        "Before emitting a derived report that uses the `precisely` section, call the configured compliance logging tool. If attribution or lineage cannot be preserved, stop and release nothing."
    );
    assert!(expected["policy"]
        .as_str()
        .unwrap()
        .starts_with(".tn/config/agents.md#precisely.risk.enriched@1#sha256:"));
    assert_eq!(
        expected["governs_sections"],
        serde_json::json!(["precisely"])
    );
}

#[test]
fn cli_rejects_invalid_arguments_and_unsafe_output_targets_without_a_manifest() {
    let temp = tempfile::tempdir().unwrap();

    let relative = Command::new(binary())
        .args([
            "create",
            "--out",
            "relative-fixture",
            "--llama-source-revision",
            LLAMA_REVISION,
            "--model-sha256",
            MODEL_SHA256,
            "--binary-sha256",
            BINARY_SHA256,
        ])
        .output()
        .unwrap();
    assert_failed_with(&relative, "fixture_output_not_absolute");

    let nonempty = temp.path().join("nonempty");
    fs::create_dir(&nonempty).unwrap();
    fs::write(nonempty.join("keep.txt"), b"keep").unwrap();
    let output = create_fixture(&nonempty);
    assert_failed_with(&output, "fixture_output_not_empty");
    assert_eq!(fs::read(nonempty.join("keep.txt")).unwrap(), b"keep");
    assert!(!nonempty.join("manifest.json").exists());

    let bad_hash = temp.path().join("bad-hash");
    let output = Command::new(binary())
        .args([
            "create",
            "--out",
            bad_hash.to_str().unwrap(),
            "--llama-source-revision",
            LLAMA_REVISION,
            "--model-sha256",
            "not-a-sha256",
            "--binary-sha256",
            BINARY_SHA256,
        ])
        .output()
        .unwrap();
    assert_failed_with(&output, "fixture_argument_invalid");
    assert!(!bad_hash.join("manifest.json").exists());

    let real = temp.path().join("real-target");
    fs::create_dir(&real).unwrap();
    let link = temp.path().join("linked-target");
    create_directory_symlink(&real, &link);
    let output = create_fixture(&link);
    assert_failed_with(&output, "fixture_output_reparse_target");
    assert!(!real.join("manifest.json").exists());
}

fn create_fixture(out: &Path) -> Output {
    Command::new(binary())
        .args([
            "create",
            "--out",
            out.to_str().unwrap(),
            "--llama-source-revision",
            LLAMA_REVISION,
            "--model-sha256",
            MODEL_SHA256,
            "--binary-sha256",
            BINARY_SHA256,
        ])
        .output()
        .unwrap()
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tn-fixture")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn assert_failed_with(output: &Output, code: &str) {
    assert!(!output.status.success(), "command unexpectedly succeeded");
    assert!(output.stdout.is_empty(), "failed command wrote stdout");
    assert!(
        text(&output.stderr).contains(code),
        "stderr: {}",
        text(&output.stderr)
    );
}

fn manifest_path(manifest: &Map<String, Value>, name: &str) -> PathBuf {
    PathBuf::from(manifest[name].as_str().unwrap())
}

fn request_row(path: PathBuf) -> Map<String, Value> {
    let request: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(
        request
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        ["messages", "protocol", "purpose", "stream", "tn_row"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
    assert_eq!(request["protocol"], "tn.llama.request.v1");
    assert_eq!(
        request["messages"],
        serde_json::json!([
            {"role": "user", "content": "Reply with exactly READY."}
        ])
    );
    assert_eq!(request["stream"], false);
    request["tn_row"].as_object().unwrap().clone()
}

fn plaintext_source_row(path: PathBuf) -> Map<String, Value> {
    let bytes = fs::read(path).unwrap();
    assert!(
        bytes.ends_with(b"\n"),
        "plaintext source is not newline-terminated"
    );
    let line = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
    assert!(
        !line.contains('\n'),
        "plaintext source contains more than one NDJSON row"
    );
    serde_json::from_str::<Value>(line)
        .unwrap()
        .as_object()
        .unwrap()
        .clone()
}

fn assert_outer_sealed_valid(name: &str, row: &Map<String, Value>) {
    let envelope = require_envelope_shape(row.clone()).unwrap();
    let blocks = extract_group_blocks(&envelope).unwrap();
    let verification = verify_sealed(&envelope, &blocks);
    assert!(verification.row_hash, "{name} row hash");
    assert!(verification.signature, "{name} signature");
}

fn open_group(row: &Map<String, Value>, reader: &Path, group: &str) -> Value {
    let wire = String::from_utf8(
        tn_core::canonical::canonical_bytes(&Value::Object(row.clone())).unwrap(),
    )
    .unwrap();
    let opened = tn_core::unseal_as_recipient(&wire, reader, group, true).unwrap();
    assert!(opened.valid.row_hash, "{group} row hash");
    assert!(opened.valid.signature, "{group} signature");
    opened.plaintext[group].clone()
}

fn assert_plaintext_row_hash_valid(row: &Map<String, Value>) {
    let public_fields = row
        .iter()
        .filter(|(name, _)| !ENVELOPE_RESERVED.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let expected = compute_row_hash(&RowHashInput {
        device_identity: row["device_identity"].as_str().unwrap(),
        timestamp: row["timestamp"].as_str().unwrap(),
        event_id: row["event_id"].as_str().unwrap(),
        event_type: row["event_type"].as_str().unwrap(),
        level: row["level"].as_str().unwrap(),
        prev_hash: row["prev_hash"].as_str().unwrap(),
        public_fields: &public_fields,
        groups: &BTreeMap::new(),
    });
    assert_eq!(row["row_hash"], expected);
}

fn assert_only_paths_changed(
    left: &Map<String, Value>,
    right: &Map<String, Value>,
    paths: &[&str],
) {
    let mut normalized = Value::Object(right.clone());
    let original = Value::Object(left.clone());
    for path in paths {
        set_path(&mut normalized, path, value_at(&original, path).clone());
    }
    assert_eq!(normalized, original, "mutation changed more than {paths:?}");
}

fn value_at<'a>(value: &'a Value, path: &str) -> &'a Value {
    value.pointer(path).unwrap()
}

fn set_path(value: &mut Value, path: &str, replacement: Value) {
    *value.pointer_mut(path).unwrap() = replacement;
}

fn assert_output_writer_material_is_confined(root: &Path) {
    let mut paths = Vec::new();
    collect_file_paths(root, &mut paths);
    for name in ["input.index.key", "output.index.key"] {
        assert!(
            paths.iter().any(|path| path.ends_with(name)),
            "missing {name}"
        );
    }
    for group in ["default", "firm_data", "precisely", "tn.agents"] {
        assert!(
            root.join("receiver")
                .join(format!("{group}.btn.mykit"))
                .is_file(),
            "missing input reader kit for {group}"
        );
    }
    for group in ["default", "tn.agents"] {
        assert!(
            root.join("output-reader")
                .join(format!("{group}.btn.mykit"))
                .is_file(),
            "missing output reader kit for {group}"
        );
    }
    let output_keys = root.join("output-writer").join(".tn").join("keys");
    for forbidden in [
        "local.private",
        "index_master.key",
        "default.btn.state",
        "tn.agents.btn.state",
    ] {
        let matches = paths
            .iter()
            .filter(|path| path.file_name().is_some_and(|name| name == forbidden))
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "unexpected {forbidden} count");
        assert!(
            matches[0].starts_with(&output_keys),
            "{forbidden} escaped output writer state"
        );
    }
}

fn collect_file_paths(path: &Path, paths: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            collect_file_paths(&entry.path(), paths);
        } else {
            paths.push(entry.path());
        }
    }
}

#[cfg(unix)]
fn create_directory_symlink(target: &Path, link: &Path) {
    std::os::unix::fs::symlink(target, link).unwrap();
}

#[cfg(windows)]
fn create_directory_symlink(target: &Path, link: &Path) {
    std::os::windows::fs::symlink_dir(target, link).unwrap();
}
