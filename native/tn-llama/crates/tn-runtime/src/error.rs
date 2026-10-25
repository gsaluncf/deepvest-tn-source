use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionStage {
    Parse,
    RequestShape,
    RowShape,
    RowHash,
    Governance,
    Chain,
    Admission,
    Session,
    Output,
    Tool,
    Ffi,
}

#[derive(Debug, Error)]
#[error("{code}: {message}")]
pub struct TnLlamaError {
    code: &'static str,
    stage: AdmissionStage,
    message: String,
}

impl TnLlamaError {
    pub(crate) fn new(
        code: &'static str,
        stage: AdmissionStage,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            stage,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    #[must_use]
    pub const fn stage(&self) -> AdmissionStage {
        self.stage
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}
