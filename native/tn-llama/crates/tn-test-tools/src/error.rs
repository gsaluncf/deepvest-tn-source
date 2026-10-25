use std::io;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FixtureError {
    #[error("fixture_argument_invalid: {0}")]
    ArgumentInvalid(String),
    #[error("fixture_output_not_absolute: output directory must be an absolute path")]
    OutputNotAbsolute,
    #[error("fixture_output_not_directory: output target must be a directory")]
    OutputNotDirectory,
    #[error("fixture_output_not_empty: output directory must be empty")]
    OutputNotEmpty,
    #[error("fixture_output_reparse_target: output target must not be a symlink or reparse point")]
    OutputReparseTarget,
    #[error("fixture_io_error: {0}")]
    Io(#[from] io::Error),
    #[error("fixture_tn_error: {0}")]
    Tn(#[from] tn_core::Error),
    #[error("fixture_btn_error: {0}")]
    Btn(#[from] tn_btn::Error),
    #[error("fixture_json_error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("fixture_invariant_failed: {0}")]
    Invariant(&'static str),
}
