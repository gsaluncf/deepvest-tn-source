import json

import pytest
import tn

from deepvest_tn.publications import PublicationStore

POLICY = """---
version: 1
schema: tn-agents-policy@v1
---
## portfolio.feed.data.v1
### instruction
Prepare approved portfolio analysis.
### use_for
Portfolio analysis.
### do_not_use_for
Individual disclosure.
### consequences
Contract review.
### on_violation_or_error
Refuse release.
"""


def publication():
    with tn.Session(POLICY, groups=["finance"]) as session:
        data = session.create_obj(
            {"rows": [{"amount": 42}]}, session.policy("portfolio.feed.data.v1"),
            object_type="portfolio.feed.data.v1", group="finance",
        )
        return data.release(
            use=tn.UseContext("source", "portfolio_analysis", "publish"),
            to="analytics", decide=lambda context: True,
        )


def test_retained_publication_is_exact_and_survives_store_restart(tmp_path):
    source = publication()
    store = PublicationStore(tmp_path)
    assert store.put(source) == source.id
    assert store.put(source) == source.id
    assert store.get(source.id) == source.forward()
    assert PublicationStore(tmp_path).get(source.id) == source.forward()


def test_same_native_identity_cannot_replace_retained_wire_representation(tmp_path):
    source = publication()
    reformatted = tn.GovernedObject.parse(json.dumps(source.envelope, indent=2))
    assert reformatted.id == source.id
    assert reformatted.wire != source.wire
    store = PublicationStore(tmp_path)
    store.put(source)
    with pytest.raises(ValueError, match="immutable"):
        store.put(reformatted)
    assert store.get(source.id) == source.forward()


def test_missing_publication_and_unsafe_identity_fail_closed(tmp_path):
    store = PublicationStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("sha256:" + "a" * 64)
    for identity in ["../outside", "sha256:" + "A" * 64, "sha256:short"]:
        with pytest.raises(ValueError):
            store.get(identity)


def test_store_detects_content_and_route_index_substitution(tmp_path):
    first, second = publication(), publication()
    store = PublicationStore(tmp_path)
    store.put(first)
    store.put(second)
    first_path = tmp_path / "objects" / f"{first.id.removeprefix('sha256:')}.tn"
    first_path.write_bytes(second.forward())
    with pytest.raises(ValueError, match="identity|digest"):
        store.get(first.id)
    first_path.write_bytes(first.forward())
    index = tmp_path / "index" / f"{first.id.removeprefix('sha256:')}.json"
    value = json.loads(index.read_text())
    value["object_id"] = second.id
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="index"):
        store.get(first.id)
