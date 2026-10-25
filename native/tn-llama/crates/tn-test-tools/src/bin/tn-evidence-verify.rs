use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tn_test_tools::{verify_evidence, EvidenceVerdict};

#[derive(Debug, Parser)]
#[command(name = "tn-evidence-verify")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Verify {
        #[arg(long)]
        evidence: PathBuf,
    },
}

fn main() -> ExitCode {
    let Cli { command } = Cli::parse();
    let (verdict, exit_code) = match command {
        Command::Verify { evidence } => match verify_evidence(&evidence) {
            Ok(counts) => (EvidenceVerdict::accepted(counts), ExitCode::SUCCESS),
            Err(error) => (EvidenceVerdict::rejected(&error), ExitCode::FAILURE),
        },
    };
    println!(
        "{}",
        serde_json::to_string(&verdict).expect("closed verdict serializes")
    );
    exit_code
}
