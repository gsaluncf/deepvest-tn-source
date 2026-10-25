from importlib.metadata import version


def test_existing_workflow_imports_are_available_at_pinned_versions() -> None:
    assert version("tn-proto") == "2026.9.10b2.dev2"
    assert version("polars") == "1.39.3"

    import tn
    from tn_llama_client.clients import GovernedLlamaClient, TnOutputReader
    from tn_llama_client.protocol import EncryptedReportRelease
    from tn_llama_client.transcript import HttpTranscript
    from tn_polars.builder import GovernedPolars
    from tn_polars.registry import UnityCatalogRegistry
    from tn_polars.tn_envelopes import DeskLedger

    assert tn.__version__ == "2026.9.10b2.dev2"
    assert GovernedPolars
    assert UnityCatalogRegistry
    assert DeskLedger
    assert GovernedLlamaClient
    assert TnOutputReader
    assert EncryptedReportRelease
    assert HttpTranscript
