//! TN-governed llama runtime boundary.

mod admission;
mod audit;
mod chain_store;
mod client_trace;
mod config;
mod contract;
mod envelope;
mod error;
mod execution_binding;
mod ffi;
mod governance;
mod governance_binding;
mod governed_input;
mod model_phase;
mod model_protocol;
mod output_tn;
mod runtime;
mod sensitive_trace;
mod session;
mod tools;

pub use admission::{AdmissionController, AdmittedRequest};
pub use audit::{
    AuditDigest, AuditDigestError, AuditEvent, AuditFacts, AuditLog, AuditRecord, DecisionAudit,
    AUDIT_EVENT_TYPE, AUDIT_SCHEMA,
};
pub use chain_store::{ChainReservation, ChainStore, SourceNamespace};
pub use client_trace::{
    ActionOutcome, ClientTraceEvent, GovernedReportRelease, ReportRelease, RELEASE_PROTOCOL,
};
pub use config::{RequestLimits, RuntimeConfig};
pub use contract::{OutputPolicy, ReportContract, SectionRule};
pub use envelope::{
    parse_and_verify_request, AcceptedRequest, UserMessage, VerifiedRow, REQUEST_DIGEST_DOMAIN,
    REQUEST_PROTOCOL,
};
pub use error::{AdmissionStage, TnLlamaError};
pub use governance::{
    GovernanceKeyProvenance, GovernanceReader, VerifiedGovernance, VerifiedGroup, GOVERNANCE_FIELDS,
};
pub use governance_binding::{
    governance_model_binding_authorized_from_env, governance_model_binding_matches,
    governance_model_file_matches, EXPECTED_GOVERNANCE_MODEL_SHA256_ENV,
};
pub use governed_input::{
    parse_governed_object_request, AcceptedGovernedRequest, ContractPermission,
    GovernedInputConfig, GOVERNED_REQUEST_PROTOCOL,
};
pub use model_phase::Phase;
pub use model_protocol::{
    NativeToolResult, ParsedToolCall, MODEL_REQUEST_DIGEST_DOMAIN, RENDERED_PROMPT_DIGEST_DOMAIN,
    RUNNABLE_TASK_DIGEST_DOMAIN, RUNNABLE_TASK_SCHEMA,
};
pub use output_tn::{
    ComplianceReceipt, OutputPublisher, ReportProposal, SourceLineage, TrustedDerivation,
    VerifiedOutputRow,
};
pub use runtime::GovernedRuntime;
pub use sensitive_trace::{
    BuildMetadata, SensitiveTrace, SensitiveTraceEvent, SensitiveTraceKind, SensitiveTraceSink,
    SENSITIVE_TRACE_SCHEMA,
};
pub use session::{ModelTurn, Session, State, TurnPermit};
pub use tools::{
    ComplianceInvocation, RegisteredCall, RegisteredToolSpec, ToolContext, ToolDispatcher,
    ToolReceipt,
};

pub use execution_binding::{
    ExecutionBackend, ExecutionIdentity, VllmBackend, VllmExecutionIdentity, SAMPLING_FIELDS,
    VLLM_REVISION,
};
