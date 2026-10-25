"""Unity Catalog as the registry: resolve a feed name to its governance bundle.

The design doc's two-liner is ``prepare(internal_registry.get("bbg.enrich"))``.
Here the internal registry is a Unity Catalog server. A governed feed is a UC
volume holding the compiled vendor reading, the TN governance body, the
declared profile, and the OPA artifacts. A release is registered back as a UC
external table over the governed Delta path, so the catalog knows both ends.
"""

import json
import hashlib
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .policy import CompiledPolicy
from .release import PolicyDenied

API = "/api/2.1/unity-catalog"

ENTRY_FILES = ("compiled_policy.json", "tn_governance.json", "profile.json",
               "policy.wasm", "data.json")


@dataclass(frozen=True)
class RegistryEntry:
    """Everything prepare() needs, resolved from one catalog name."""

    name: str
    policy: CompiledPolicy
    tn_governance: dict
    dtypes: dict | None
    label_map: dict | None
    wasm_path: Path
    data_path: Path
    revision_id: str | None = None
    verified_artifacts: object | None = None
    compiled_policy_bytes: bytes | None = None
    profile_bytes: bytes | None = None

    def evaluator(self):
        from .opa import OpaWasmEvaluator
        if self.verified_artifacts is not None:
            if self.revision_id != self.verified_artifacts.policy_revision:
                raise PolicyDenied("evaluator revision differs from its accepted artifact binding")
            compiled, profile = self.compiled_policy_bytes, self.profile_bytes
            wasm, data = self.wasm_path.read_bytes(), self.data_path.read_bytes()
            _verify_artifact_bytes(self.verified_artifacts, compiled, profile, wasm, data)
            profile_value = json.loads(profile)
            if (self.policy.to_dict() != json.loads(compiled)
                    or self.dtypes != profile_value.get("dtypes")
                    or self.label_map != profile_value.get("labels")):
                raise PolicyDenied("prepared evaluator metadata changed after artifact verification")
            return OpaWasmEvaluator.from_bytes(wasm, data)
        return OpaWasmEvaluator(self.wasm_path, self.data_path)


class UnityCatalogRegistry:
    """A thin client for the Unity Catalog REST API, OSS or hosted."""

    def __init__(self, base_url, catalog, schema, *, token=None):
        self.base_url = base_url.rstrip("/")
        self.catalog = catalog
        self.schema = schema
        self.token = token

    def _request(self, method, path, payload=None):
        request = urllib.request.Request(
            self.base_url + API + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            method=method, headers=self._headers(payload))
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            raise PolicyDenied(
                f"Unity Catalog refused {method} {path}: HTTP {error.code}. "
                f"The catalog is the authority for governed feeds, so this "
                f"halts the operation. Server said: {detail}") from error

    def _headers(self, payload):
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def volume_name(self, name):
        return name.replace(".", "_")

    def get(self, name):
        """Resolve a feed name to its governance bundle via its UC volume."""
        full = f"{self.catalog}.{self.schema}.{self.volume_name(name)}"
        volume = self._request("GET", f"/volumes/{full}")
        root = _local_path(volume.get("storage_location", ""), full)
        return _load_entry(name, root, described_as=f"volume {full}")

    def get_edition(self, dataset, edition, record_id):
        full = f"{self.catalog}.{self.schema}.tn_edition_{_digest(record_id)}"
        volume = self._request("GET", f"/volumes/{full}")
        return _load_edition(dataset, edition, record_id,
                             _local_path(volume.get("storage_location", ""), full))

    def register_edition(self, dataset, edition, record_id, bundle_dir):
        root = Path(bundle_dir).resolve()
        _load_edition(dataset, edition, record_id, root)
        name = f"tn_edition_{_digest(record_id)}"
        self._ensure("/catalogs", {"name": self.catalog})
        self._ensure("/schemas", {"name": self.schema, "catalog_name": self.catalog})
        self._ensure("/volumes", {
            "catalog_name": self.catalog, "schema_name": self.schema,
            "name": name, "volume_type": "EXTERNAL", "storage_location": root.as_uri(),
        })
        full = f"{self.catalog}.{self.schema}.{name}"
        actual = self._request("GET", f"/volumes/{full}")
        if _local_path(actual.get("storage_location", ""), full).resolve() != root:
            raise PolicyDenied("immutable edition volume already points to another bundle")
        self.get_edition(dataset, edition, record_id)
        return full

    def register_feed(self, name, bundle_dir):
        """Publish a governed feed bundle as a UC external volume (setup step)."""
        self._ensure("/catalogs", {"name": self.catalog})
        self._ensure("/schemas", {"name": self.schema, "catalog_name": self.catalog})
        self._ensure("/volumes", {
            "catalog_name": self.catalog, "schema_name": self.schema,
            "name": self.volume_name(name), "volume_type": "EXTERNAL",
            "storage_location": Path(bundle_dir).resolve().as_uri(),
        })

    def register_release(self, name, delta_path, columns):
        """Register a governed Delta release back into the catalog."""
        table = {
            "name": self.volume_name(name), "catalog_name": self.catalog,
            "schema_name": self.schema, "table_type": "EXTERNAL",
            "data_source_format": "DELTA",
            "storage_location": Path(delta_path).resolve().as_uri(),
            "columns": [
                {"name": column, "type_name": type_name, "type_text": type_name.lower(),
                 "type_json": json.dumps({
                     "name": column, "type": type_name.lower(),
                     "nullable": True, "metadata": {}}),
                 "position": position, "nullable": True}
                for position, (column, type_name) in enumerate(columns.items())
            ],
        }
        self._ensure("/tables", table)
        return f"{self.catalog}.{self.schema}.{self.volume_name(name)}"

    def _ensure(self, path, payload):
        try:
            self._request("POST", path, payload)
        except PolicyDenied as denial:
            if "409" not in str(denial) and "ALREADY_EXISTS" not in str(denial):
                raise


def _load_entry(name, root, *, described_as):
    missing = [item for item in ENTRY_FILES if not (root / item).exists()]
    if missing:
        raise PolicyDenied(
            f"governed feed {name} resolved to {described_as}, but its "
            f"bundle is incomplete: missing {', '.join(missing)}. "
            "The feed cannot be prepared until data governance publishes "
            "a complete bundle.")
    profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    return RegistryEntry(
        name=name,
        policy=CompiledPolicy.from_dict(
            json.loads((root / "compiled_policy.json").read_text(encoding="utf-8"))),
        tn_governance=json.loads(
            (root / "tn_governance.json").read_text(encoding="utf-8")),
        dtypes=profile.get("dtypes"),
        label_map=profile.get("labels"),
        wasm_path=root / "policy.wasm",
        data_path=root / "data.json",
    )


class LocalRegistry:
    """A file-backed registry with the same contract as UnityCatalogRegistry.

    For laptops and CI. A feeds.json in the root maps a feed name to its
    bundle directory; releases.json records what went out and where. Swapping
    this for the Unity Catalog adapter changes one constructor and nothing
    else, which is the point of the interface.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _index(self, filename):
        path = self.root / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _write_index(self, filename, index):
        (self.root / filename).write_text(json.dumps(index, indent=2),
                                          encoding="utf-8")

    def get(self, name):
        feeds = self._index("feeds.json")
        if name not in feeds:
            raise PolicyDenied(
                f"governed feed {name} is not registered in {self.root}. "
                "Feeds available: " + (", ".join(sorted(feeds)) or "none")
                + ". Platform ops registers feeds with register_feed().")
        return _load_entry(name, Path(feeds[name]), described_as=feeds[name])

    def register_feed(self, name, bundle_dir):
        feeds = self._index("feeds.json")
        feeds[name] = str(Path(bundle_dir).resolve())
        self._write_index("feeds.json", feeds)

    def get_edition(self, dataset, edition, record_id):
        path = self.root / "editions" / f"{_digest(record_id)}.json"
        value = json.loads(path.read_bytes())
        if (value.get("dataset"), value.get("edition"), value.get("record_id")) != (
                dataset, edition, record_id):
            raise PolicyDenied("immutable edition index does not match the requested identity")
        return _load_edition(dataset, edition, record_id, Path(value["bundle_dir"]))

    def register_edition(self, dataset, edition, record_id, bundle_dir):
        root = Path(bundle_dir).resolve()
        _load_edition(dataset, edition, record_id, root)
        value = {"dataset": dataset, "edition": edition,
                 "record_id": record_id, "bundle_dir": str(root)}
        _immutable_write(self.root / "editions" / f"{_digest(record_id)}.json",
                         json.dumps(value, sort_keys=True).encode("utf-8"))
        return record_id

    def register_release(self, name, delta_path, columns):
        releases = self._index("releases.json")
        releases[name] = {"path": str(Path(delta_path).resolve()),
                          "format": "DELTA", "columns": dict(columns)}
        self._write_index("releases.json", releases)
        return name


def _local_path(storage_location, full_name):
    if not storage_location:
        raise PolicyDenied(
            f"volume {full_name} has no storage location; the catalog entry "
            "exists but points nowhere, so the bundle cannot be read.")
    parsed = urllib.parse.urlparse(storage_location)
    if parsed.scheme in ("", "file"):
        return Path(urllib.request.url2pathname(parsed.path) if parsed.scheme
                    else storage_location)
    raise PolicyDenied(
        f"volume {full_name} stores its bundle at {storage_location}; only "
        "local and file:// locations are supported by this adapter today.")


@dataclass(frozen=True)
class EvaluatorArtifactPaths:
    compiled_policy_path: Path
    profile_path: Path
    wasm_path: Path
    data_path: Path


@dataclass(frozen=True)
class EditionRegistryEntry:
    """Untrusted transport bytes and locations, requiring native metadata admission."""

    dataset: str
    edition: str
    record_id: str
    edition_wire: bytes
    revision_wires: tuple[tuple[str, bytes], ...]
    evaluator_paths: object


_ARTIFACT_FILES = ("compiled_policy.json", "profile.json", "policy.wasm", "data.json")


def _digest(identity):
    if not isinstance(identity, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
        raise PolicyDenied("edition and revision identities require canonical sha256 digests")
    return identity[7:]


def _immutable_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".edition-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise PolicyDenied(f"immutable edition bundle differs: {path.name}") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_edition_bundle(root, *, dataset, edition, record_id, edition_wire,
                         revision_wires, evaluators):
    """Retain signed metadata and artifacts; this transport manifest grants no authority."""
    _digest(record_id)
    root = Path(root)
    revisions = list(revision_wires)
    revision_ids = [identity for identity, _wire in revisions]
    if len(set(revision_ids)) != len(revision_ids):
        raise PolicyDenied("edition bundle must list each revision once")
    _immutable_write(root / "edition.tn", edition_wire)
    for identity, wire in revisions:
        _immutable_write(root / "revisions" / f"{_digest(identity)}.tn", wire)
    for identity, artifacts in evaluators.items():
        if set(artifacts) != set(_ARTIFACT_FILES):
            raise PolicyDenied("each evaluator needs compiled policy, profile, wasm and data")
        directory = root / "evaluators" / _digest(identity)
        for filename in _ARTIFACT_FILES:
            _immutable_write(directory / filename, artifacts[filename])
    manifest = {"schema": "tn.dataset.bundle.v1", "dataset": dataset, "edition": edition,
                "record_id": record_id, "revisions": revision_ids,
                "evaluators": list(evaluators)}
    _immutable_write(root / "bundle.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))
    return root


def _load_edition(dataset, edition, record_id, root):
    _digest(record_id)
    try:
        manifest = json.loads((root / "bundle.json").read_bytes())
        if (manifest.get("schema"), manifest.get("dataset"), manifest.get("edition"),
                manifest.get("record_id")) != (
                "tn.dataset.bundle.v1", dataset, edition, record_id):
            raise PolicyDenied("immutable edition bundle identity does not match the request")
        revisions = manifest["revisions"]
        evaluators = manifest["evaluators"]
        if (not isinstance(revisions, list) or not isinstance(evaluators, list)
                or len(revisions) != len(set(revisions))
                or len(evaluators) != len(set(evaluators))):
            raise PolicyDenied("edition transport lists must contain unique identities")
        paths = {}
        for identity in evaluators:
            directory = root / "evaluators" / _digest(identity)
            files = [directory / name for name in _ARTIFACT_FILES]
            if not all(path.is_file() for path in files):
                raise PolicyDenied("edition evaluator artifacts are incomplete")
            paths[identity] = EvaluatorArtifactPaths(*files)
        return EditionRegistryEntry(
            dataset, edition, record_id, (root / "edition.tn").read_bytes(),
            tuple((identity, (root / "revisions" / f"{_digest(identity)}.tn").read_bytes())
                  for identity in revisions), MappingProxyType(paths),
        )
    except FileNotFoundError as error:
        raise PolicyDenied(f"edition bundle is incomplete: missing {error.filename}") from error


def _verify_artifact_bytes(expected, compiled, profile, wasm, data):
    for name, raw, digest in (
        ("compiled policy", compiled, expected.compiled_policy_sha256),
        ("profile", profile, expected.profile_sha256),
        ("wasm", wasm, expected.wasm_sha256),
        ("data", data, expected.data_sha256),
    ):
        if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != digest:
            raise PolicyDenied(f"{name} artifact digest differs from its signed edition")


def verified_edition_entry(name, governance, revision_id, expected, paths):
    """Prepare from the exact artifact bytes bound by an accepted native edition."""
    from tn.governed import EvaluatorArtifactSet
    if not isinstance(expected, EvaluatorArtifactSet) or expected.policy_revision != revision_id:
        raise PolicyDenied("evaluator needs its native per-revision artifact binding")
    compiled, profile = paths.compiled_policy_path.read_bytes(), paths.profile_path.read_bytes()
    wasm, data = paths.wasm_path.read_bytes(), paths.data_path.read_bytes()
    _verify_artifact_bytes(expected, compiled, profile, wasm, data)
    profile_value = json.loads(profile)
    return RegistryEntry(
        name, CompiledPolicy.from_dict(json.loads(compiled)), dict(governance),
        profile_value.get("dtypes"), profile_value.get("labels"), paths.wasm_path, paths.data_path,
        revision_id, expected, compiled, profile,
    )
