"""Signed, chained TN envelopes for governed releases.

DeskLedger wraps one tn-proto ceremony (profile ``transaction``) so every
pipeline boundary can emit a signed, chained, encrypted envelope with one
call. When an ``agents.md`` policy is supplied, the SDK includes its five
governance fields plus a content-hashed ``policy`` reference into every
covered event, and routes them into the envelope's encrypted ``tn.agents``
group. The ``tn`` package is imported lazily so the rest of tn_polars
works without it.
"""
import json
import os
from pathlib import Path

AGENTS_MD_RELATIVE = Path(".tn") / "config" / "agents.md"


class DeskLedger:
    """One tn-proto ceremony shared by every stage of a governed pipeline."""

    def __init__(self, project_dir, ceremony, ceremony_dir):
        self.project_dir = Path(project_dir)
        self.ceremony = ceremony
        self.ceremony_dir = Path(ceremony_dir)

    @classmethod
    def open(cls, project_dir, *, ceremony="desk", agents_md=None, identity_dir=None):
        project_dir = Path(project_dir).resolve()
        project_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TN_NO_LINK"] = "1"
        if identity_dir is not None:
            os.environ["TN_IDENTITY_DIR"] = str(Path(identity_dir).resolve())
        ceremony_dir = project_dir / ".tn" / ceremony
        if agents_md is not None:
            policy_path = ceremony_dir / AGENTS_MD_RELATIVE
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.write_text(agents_md, encoding="utf-8")
        cls._init_ceremony(project_dir, ceremony)
        return cls(project_dir, ceremony, ceremony_dir)

    @staticmethod
    def _init_ceremony(project_dir, ceremony):
        import tn
        previous = os.getcwd()
        os.chdir(project_dir)
        try:
            tn.init(ceremony, profile="transaction")
        finally:
            os.chdir(previous)

    def log(self, event_type, **fields):
        import tn
        tn.log(event_type, **fields)

    def entries(self):
        import tn
        decoded = []
        for entry in tn.read(all_runs=True):
            body = entry if isinstance(entry, dict) else dict(entry.__dict__)
            decoded.append(body)
        return decoded

    def raw_envelopes(self):
        log_path = self.ceremony_dir / "logs" / "default.ndjson"
        if not log_path.exists():
            return []
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines if line]

    def directive(self, event_type):
        """Return the agents.md ``on_violation_or_error`` directive, or None."""
        from tn import _agents_policy
        policy_path = self.ceremony_dir / AGENTS_MD_RELATIVE
        if not policy_path.exists():
            return None
        text = policy_path.read_text(encoding="utf-8")
        document = _agents_policy.parse_policy_text(text, str(AGENTS_MD_RELATIVE))
        template = document.templates.get(event_type)
        return template.on_violation_or_error if template else None
