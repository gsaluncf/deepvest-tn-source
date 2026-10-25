//! Real TN fixture support for integration and external-process tests.

mod error;
mod evidence;
mod fixture;

pub use error::FixtureError;
pub use evidence::{
    verify_evidence, EvidenceCounts, EvidenceError, EvidenceManifest, EvidenceVerdict,
    EVIDENCE_MANIFEST_SCHEMA, EVIDENCE_VERDICT_SCHEMA,
};
pub use fixture::{build_real_tn_fixture, create_fixture_at, BuildInputs, RealTnFixture};
