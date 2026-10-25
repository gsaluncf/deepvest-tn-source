use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tn_test_tools::{create_fixture_at, BuildInputs, FixtureError};

#[derive(Debug, Parser)]
#[command(name = "tn-fixture", disable_help_subcommand = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Create {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        llama_source_revision: String,
        #[arg(long)]
        model_sha256: String,
        #[arg(long)]
        binary_sha256: String,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(manifest) => {
            println!("{}", manifest.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<PathBuf, FixtureError> {
    let cli = Cli::try_parse().map_err(|error| FixtureError::ArgumentInvalid(error.to_string()))?;
    match cli.command {
        Command::Create {
            out,
            llama_source_revision,
            model_sha256,
            binary_sha256,
        } => create_fixture_at(
            &out,
            &BuildInputs::new(llama_source_revision, model_sha256, binary_sha256),
        ),
    }
}
