use std::sync::Arc;

use crate::chain_store::{
    reserve, reserve_active, ActiveSessionReservation, ChainReservation, SharedAdmissionState,
    SourceNamespace,
};
use crate::envelope::AcceptedRequest;
use crate::error::{AdmissionStage, TnLlamaError};

#[derive(Clone, Debug)]
pub struct AdmissionController {
    state: Arc<SharedAdmissionState>,
}

impl AdmissionController {
    pub(crate) fn admit_governed(
        &self,
        request: crate::AcceptedGovernedRequest,
        session_id: &str,
    ) -> Result<AdmittedGovernedRequest, TnLlamaError> {
        Ok(AdmittedGovernedRequest {
            request,
            _reservation: reserve_active(&self.state, session_id)?,
        })
    }
    #[must_use]
    pub fn new(
        configured_namespace: SourceNamespace,
        max_active_sessions: usize,
        max_chain_namespaces: usize,
    ) -> Self {
        Self {
            state: Arc::new(SharedAdmissionState::new(
                configured_namespace,
                max_active_sessions,
                max_chain_namespaces,
            )),
        }
    }

    pub fn admit(
        &self,
        request: AcceptedRequest,
        session_id: &str,
    ) -> Result<AdmittedRequest, TnLlamaError> {
        let reservation = reserve(&self.state, &request.row, session_id, true)?;
        Ok(AdmittedRequest {
            request,
            reservation: Some(reservation),
        })
    }

    #[must_use]
    pub fn active_sessions(&self) -> usize {
        self.state.active_sessions()
    }

    #[must_use]
    pub fn chain_namespaces(&self) -> usize {
        self.state.chain_namespaces()
    }
}

pub(crate) struct AdmittedGovernedRequest {
    pub request: crate::AcceptedGovernedRequest,
    _reservation: ActiveSessionReservation,
}

pub(crate) enum AdmittedInput {
    LegacyChained(AdmittedRequest),
    GovernedObject(AdmittedGovernedRequest),
}
impl AdmittedInput {
    pub fn abort(self) -> Result<(), TnLlamaError> {
        match self {
            Self::LegacyChained(request) => request.abort(),
            Self::GovernedObject(_) => Ok(()),
        }
    }
    pub fn commit(self) -> Result<(), TnLlamaError> {
        match self {
            Self::LegacyChained(request) => request.commit(),
            Self::GovernedObject(_) => Ok(()),
        }
    }
}

#[derive(Debug)]
pub struct AdmittedRequest {
    pub request: AcceptedRequest,
    reservation: Option<ChainReservation>,
}

impl AdmittedRequest {
    pub fn commit(mut self) -> Result<(), TnLlamaError> {
        self.take_reservation()?.commit()
    }

    pub fn abort(mut self) -> Result<(), TnLlamaError> {
        self.take_reservation()?.abort()
    }

    fn take_reservation(&mut self) -> Result<ChainReservation, TnLlamaError> {
        self.reservation.take().ok_or_else(|| {
            TnLlamaError::new(
                "tn_admission_finalized",
                AdmissionStage::Admission,
                "admitted request no longer owns a reservation",
            )
        })
    }
}
