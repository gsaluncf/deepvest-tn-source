"""Native application-route configuration for already enrolled TN sessions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import tn


def bind_workflow(
    session: tn.Session,
    *,
    application: str,
    receive_use: tn.UseContext,
    release_use: tn.UseContext,
    input_type: str,
    output_type: str,
    groups: Sequence[str],
    to: str,
    decide_receive: Callable[[tn.AdmissionContext], bool],
    decide_release: Callable[[tn.ReleaseContext], bool],
) -> tn.Workflow:
    """Register exact native routes and bind their live decisions to one application."""
    if receive_use.application != application or release_use.application != application:
        raise ValueError("workflow uses must match the configured application")
    session.configure_receive(
        use=receive_use,
        groups=groups,
        object_type=input_type,
        decide=decide_receive,
    )
    session.configure_release(
        use=release_use,
        to=to,
        object_type=output_type,
        decide=decide_release,
    )
    return session.workflow(receive=receive_use.purpose, release=release_use.purpose)
