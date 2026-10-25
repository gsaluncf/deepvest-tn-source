//! Independent native recipient verification of a bridge fixture release.
use serde_json::Value;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: vllm_verify FIXTURE_DIRECTORY RELEASE_JSON".into());
    }
    let config: tn_runtime::RuntimeConfig = serde_json::from_slice(&std::fs::read(
        std::path::Path::new(&args[0]).join("runtime-config-v2.json"),
    )?)?;
    let release: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let wire = release["output_object"]
        .as_str()
        .ok_or("missing output_object")?;
    let object = tn_core::governed::GovernedObject::parse(wire)?;
    if release["output_object_id"] != object.id() {
        return Err("output identity mismatch".into());
    }
    let reader = tn_core::runtime::Objects::open(&config.output_tn_config_path)?;
    let data = reader.receive_for(
        wire,
        &config
            .governed_objects
            .ok_or("v2 config required")?
            .use_context,
        ["default"],
        None,
        |_| Ok(true),
    )?;
    if data.policies()?.len() != 2 || data.dataset_bindings()?.len() != 2 {
        return Err("inherited governance missing".into());
    }
    println!(
        "{}",
        serde_json::json!({"verified":true,"output_object_id":object.id(),"policies":data.policies()?.len(),"dataset_bindings":data.dataset_bindings()?.len()})
    );
    Ok(())
}
