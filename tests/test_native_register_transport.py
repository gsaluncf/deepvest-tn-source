"""Optional service registers identify their own publications and leave transport exact."""

import json

import pytest
import tn

from deepvest_tn.native_reference import POLICY
from deepvest_tn.publications import PublicationStore


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_publisher_and_source_keep_their_own_registers(tmp_path, monkeypatch, enabled):
    def configure(role):
        for action in ["CREATION", "RELEASE"]:
            name = f"TN_OBJECT_{action}_REGISTER"
            if enabled:
                monkeypatch.setenv(name, str(tmp_path / f"{role}-{action}.ndjson"))
            else:
                monkeypatch.delenv(name, raising=False)

    configure("source")
    with tn.Session(POLICY, groups=["finance"]) as source:
        configure("publisher")
        with tn.Session(POLICY, groups=["finance"]) as publisher:
            initial = source.create_obj(
                {"amount": 42},
                source.policy("portfolio.feed.data.v1"),
                object_type="portfolio.feed.data.v1",
                group="finance",
            ).snapshot
            use = tn.UseContext("analytics", "portfolio_analysis", "release_result")
            data = source.receive(initial, use=use, groups=["finance"], decide=lambda _: True)
            data.data["amount"] = 84
            output = publisher.release(data, use=use, to="reporting", decide=lambda _: True)
            assert data.register_error is None
            store = PublicationStore(tmp_path / "publications")
            store.put(output)
            assert store.get(output.id) == output.wire.encode()
            assert PublicationStore(store.root).get(output.id) == output.wire.encode()
            if enabled:
                for role, action, object, writer in [
                    ("source", "CREATION", initial, source.did),
                    ("publisher", "RELEASE", output, publisher.did),
                ]:
                    row = json.loads((tmp_path / f"{role}-{action}.ndjson").read_bytes())
                    assert row["device_identity"] == writer
                    assert row["object_id"] == object.id and row["signature"]
                assert not (tmp_path / "source-RELEASE.ndjson").exists()
                assert not (tmp_path / "publisher-CREATION.ndjson").exists()
            else:
                assert not list(tmp_path.glob("*.ndjson"))
