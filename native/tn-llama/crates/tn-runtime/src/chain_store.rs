use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use tn_core::chain::ZERO_HASH;

use crate::config::TEST_DEVICE_IDENTITY;
use crate::envelope::VerifiedRow;
use crate::error::{AdmissionStage, TnLlamaError};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceNamespace {
    pub device_identity: String,
    pub event_type: String,
}

impl SourceNamespace {
    #[must_use]
    pub fn new(device_identity: impl Into<String>, event_type: impl Into<String>) -> Self {
        Self {
            device_identity: device_identity.into(),
            event_type: event_type.into(),
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn test() -> Self {
        Self::new(TEST_DEVICE_IDENTITY, "tn.llama.request")
    }
}

impl From<&VerifiedRow> for SourceNamespace {
    fn from(row: &VerifiedRow) -> Self {
        Self::new(row.device_identity(), row.event_type())
    }
}

#[derive(Clone, Debug)]
pub struct ChainStore {
    state: Arc<SharedAdmissionState>,
}

impl ChainStore {
    #[must_use]
    pub fn new(configured_namespace: SourceNamespace, max_chain_namespaces: usize) -> Self {
        Self {
            state: Arc::new(SharedAdmissionState::new(
                configured_namespace,
                usize::MAX,
                max_chain_namespaces,
            )),
        }
    }

    pub fn reserve(
        &self,
        row: &VerifiedRow,
        session_id: &str,
    ) -> Result<ChainReservation, TnLlamaError> {
        reserve(&self.state, row, session_id, false)
    }

    #[must_use]
    pub fn chain_namespaces(&self) -> usize {
        self.state.chain_namespaces()
    }
}

#[derive(Debug)]
pub struct ChainReservation {
    state: Arc<SharedAdmissionState>,
    namespace: SourceNamespace,
    session_id: String,
    row_hash: String,
    finalized: bool,
}

impl ChainReservation {
    pub fn commit(mut self) -> Result<(), TnLlamaError> {
        self.finish(true)
    }

    pub fn abort(mut self) -> Result<(), TnLlamaError> {
        self.finish(false)
    }

    fn finish(&mut self, commit: bool) -> Result<(), TnLlamaError> {
        if self.finalized {
            return Err(chain_error(
                "tn_chain_reservation_finalized",
                "chain reservation was already finalized",
            ));
        }

        let mut state = self.state.lock()?;
        let Some(record) = state.reservations.get(&self.namespace) else {
            return Err(chain_error(
                "tn_chain_reservation_lost",
                "chain reservation is not active",
            ));
        };
        if record.session_id != self.session_id || record.row_hash != self.row_hash {
            return Err(chain_error(
                "tn_chain_reservation_lost",
                "chain reservation binding changed",
            ));
        }

        let record = state.reservations.remove(&self.namespace).ok_or_else(|| {
            chain_error(
                "tn_chain_reservation_lost",
                "chain reservation disappeared during finalization",
            )
        })?;
        if commit {
            state
                .committed_tips
                .insert(self.namespace.clone(), self.row_hash.clone());
        }
        if record.owns_active_session {
            state.active_sessions.remove(&self.session_id);
        }
        self.finalized = true;
        Ok(())
    }
}

impl Drop for ChainReservation {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        let mut state = lock_recover(&self.state.inner);
        let matches = state
            .reservations
            .get(&self.namespace)
            .is_some_and(|record| {
                record.session_id == self.session_id && record.row_hash == self.row_hash
            });
        if !matches {
            return;
        }
        if let Some(record) = state.reservations.remove(&self.namespace) {
            if record.owns_active_session {
                state.active_sessions.remove(&self.session_id);
            }
        }
        self.finalized = true;
    }
}

#[derive(Debug)]
pub(crate) struct SharedAdmissionState {
    inner: Mutex<AdmissionState>,
}

/// Standalone objects acquire concurrency capacity without inventing a chain tip.
#[derive(Debug)]
pub(crate) struct ActiveSessionReservation {
    state: Arc<SharedAdmissionState>,
    session_id: String,
}
impl Drop for ActiveSessionReservation {
    fn drop(&mut self) {
        lock_recover(&self.state.inner)
            .active_sessions
            .remove(&self.session_id);
    }
}

pub(crate) fn reserve_active(
    shared: &Arc<SharedAdmissionState>,
    session_id: &str,
) -> Result<ActiveSessionReservation, TnLlamaError> {
    let mut state = shared.lock()?;
    if session_id.is_empty()
        || state.active_sessions.contains(session_id)
        || state.active_sessions.len() >= state.max_active_sessions
    {
        return Err(TnLlamaError::new(
            "tn_active_session_limit",
            AdmissionStage::Admission,
            "active session limit reached or session identity already reserved",
        ));
    }
    state.active_sessions.insert(session_id.to_owned());
    Ok(ActiveSessionReservation {
        state: Arc::clone(shared),
        session_id: session_id.to_owned(),
    })
}

impl SharedAdmissionState {
    pub(crate) fn new(
        configured_namespace: SourceNamespace,
        max_active_sessions: usize,
        max_chain_namespaces: usize,
    ) -> Self {
        Self {
            inner: Mutex::new(AdmissionState {
                configured_namespace,
                max_active_sessions,
                max_chain_namespaces,
                committed_tips: BTreeMap::new(),
                reservations: BTreeMap::new(),
                active_sessions: HashSet::new(),
            }),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, AdmissionState>, TnLlamaError> {
        self.inner.lock().map_err(|_| {
            TnLlamaError::new(
                "tn_admission_state_poisoned",
                AdmissionStage::Admission,
                "admission state lock is poisoned",
            )
        })
    }

    pub(crate) fn active_sessions(&self) -> usize {
        lock_recover(&self.inner).active_sessions.len()
    }

    pub(crate) fn chain_namespaces(&self) -> usize {
        namespace_count(&lock_recover(&self.inner))
    }
}

#[derive(Debug)]
struct AdmissionState {
    configured_namespace: SourceNamespace,
    max_active_sessions: usize,
    max_chain_namespaces: usize,
    committed_tips: BTreeMap<SourceNamespace, String>,
    reservations: BTreeMap<SourceNamespace, ReservationRecord>,
    active_sessions: HashSet<String>,
}

#[derive(Debug)]
struct ReservationRecord {
    session_id: String,
    row_hash: String,
    owns_active_session: bool,
}

pub(crate) fn reserve(
    shared: &Arc<SharedAdmissionState>,
    row: &VerifiedRow,
    session_id: &str,
    acquire_active_session: bool,
) -> Result<ChainReservation, TnLlamaError> {
    let namespace = SourceNamespace::from(row);
    let mut state = shared.lock()?;

    let namespace_allowed = namespace.event_type == state.configured_namespace.event_type
        && (state.configured_namespace.device_identity == "*"
            || namespace.device_identity == state.configured_namespace.device_identity);
    if !namespace_allowed {
        return Err(TnLlamaError::new(
            "tn_source_namespace_mismatch",
            AdmissionStage::Admission,
            "row is outside the configured source namespace",
        ));
    }
    if session_id.is_empty() {
        return Err(TnLlamaError::new(
            "tn_session_id_invalid",
            AdmissionStage::Admission,
            "session ID must not be empty",
        ));
    }
    if acquire_active_session && state.active_sessions.len() >= state.max_active_sessions {
        return Err(TnLlamaError::new(
            "tn_active_session_limit",
            AdmissionStage::Admission,
            "active session limit reached",
        ));
    }

    let namespace_exists = state.committed_tips.contains_key(&namespace)
        || state.reservations.contains_key(&namespace);
    if !namespace_exists && namespace_count(&state) >= state.max_chain_namespaces {
        return Err(TnLlamaError::new(
            "tn_chain_namespace_limit",
            AdmissionStage::Admission,
            "chain namespace limit reached",
        ));
    }

    let expected_tip = state
        .committed_tips
        .get(&namespace)
        .map_or(ZERO_HASH, String::as_str);
    if row.prev_hash() != expected_tip {
        return Err(chain_error(
            "tn_chain_tip_mismatch",
            "row predecessor does not match the receiver-held committed tip",
        ));
    }
    if state.reservations.contains_key(&namespace) {
        return Err(chain_error(
            "tn_chain_reservation_conflict",
            "source namespace already has an active reservation",
        ));
    }
    if acquire_active_session && state.active_sessions.contains(session_id) {
        return Err(TnLlamaError::new(
            "tn_active_session_conflict",
            AdmissionStage::Admission,
            "session ID already owns an active lease",
        ));
    }

    if acquire_active_session {
        state.active_sessions.insert(session_id.to_string());
    }
    state.reservations.insert(
        namespace.clone(),
        ReservationRecord {
            session_id: session_id.to_string(),
            row_hash: row.row_hash().to_string(),
            owns_active_session: acquire_active_session,
        },
    );
    drop(state);

    Ok(ChainReservation {
        state: Arc::clone(shared),
        namespace,
        session_id: session_id.to_string(),
        row_hash: row.row_hash().to_string(),
        finalized: false,
    })
}

fn namespace_count(state: &AdmissionState) -> usize {
    state
        .committed_tips
        .keys()
        .chain(state.reservations.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .len()
}

fn lock_recover(lock: &Mutex<AdmissionState>) -> MutexGuard<'_, AdmissionState> {
    lock.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn chain_error(code: &'static str, message: &'static str) -> TnLlamaError {
    TnLlamaError::new(code, AdmissionStage::Chain, message)
}
