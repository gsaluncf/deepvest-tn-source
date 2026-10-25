use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReportContract {
    pub allowed_input_sections: BTreeSet<String>,
    pub output_event_type: String,
    pub output_public_section: String,
    pub operation: String,
    pub section_rules: BTreeMap<String, SectionRule>,
    pub output_policy: OutputPolicy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SectionRule {
    pub propagate_source_governance: bool,
    pub required_attribution: Option<String>,
    pub required_tool: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutputPolicy {
    pub instruction: String,
    pub use_for: String,
    pub do_not_use_for: String,
    pub consequences: String,
    pub on_violation_or_error: String,
}
