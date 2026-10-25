import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class PolicyDenied(Exception):
    pass


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str
    decision_id: str


@dataclass(frozen=True)
class ReleaseArtifact:
    path: object
    manifest_path: object


@dataclass(frozen=True)
class GovernedRelease:
    path: Path
    manifest: dict
    payload: bytes = b""


def read_governed_release(path, *, cipher=None, current_policy=None):
    """Verify a release before a later governed loop consumes it.

    ``current_policy`` enables the Continuation-stage skew check: a manifest
    compiled under any other policy fails closed. ``cipher`` unseals sealed
    artifacts; the returned ``payload`` is the plaintext bytes either way.
    """
    path = Path(path)
    manifest_path = Path(str(path) + ".gov.json")
    if not manifest_path.exists():
        raise PolicyDenied(f"missing governance manifest for {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != manifest.get("file_sha256"):
        raise PolicyDenied(f"release file hash differs from its governance manifest: {path.name}")
    if not manifest.get("tn_governance", {}).get("policy"):
        raise PolicyDenied("release manifest has no TN governance body")
    _check_policy_skew(manifest, current_policy, path)
    return GovernedRelease(path=path, manifest=manifest,
                           payload=_unseal(manifest, body, cipher, path))


def _check_policy_skew(manifest, current_policy, path):
    if current_policy is None:
        return
    from .policy import policy_sha256
    if manifest.get("compiled_policy_sha256") != policy_sha256(current_policy):
        raise PolicyDenied(
            f"release {path.name} was compiled under a deprecated policy; "
            "failing closed until the release is re-issued")


def _unseal(manifest, body, cipher, path):
    sealed = manifest.get("sealed")
    if not sealed:
        return body
    if cipher is None:
        raise PolicyDenied(
            f"release {path.name} is sealed under a governance-bound key; "
            "a keyed reader is required")
    return cipher.open(body, sealed["marker"])
