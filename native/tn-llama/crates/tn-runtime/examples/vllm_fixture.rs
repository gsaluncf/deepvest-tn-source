//! Materialize the standalone governed fixture for cross-process tests.
//! All keys in this fixture are test material. Never configure a deployed service with them.
#[path = "../tests/governed_support/mod.rs"]
mod governed_support;
#[path = "../tests/support/mod.rs"]
mod support;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = governed_support::Fixture::new();
    let root = fixture.legacy.temp.path();
    std::fs::write(root.join("runtime-config-v2.json"), &fixture.runtime_json)?;
    std::fs::write(root.join("request-v2.json"), fixture.request("deepvest"))?;
    std::fs::write(
        root.join("governed-manifest.json"),
        serde_json::to_vec_pretty(&fixture.manifest)?,
    )?;
    let path = fixture.legacy.temp.keep();
    println!("{}", path.display());
    Ok(())
}
