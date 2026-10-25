"""The application uses the built Rust SDK through its Python bindings."""

import tn


def test_governed_public_types_are_the_native_implementation():
    from tn._native import governed

    for name in (
        "Session", "DataObject", "GovernedObject", "UseContext", "PolicyDag",
        "PolicyRevision", "DatasetCatalog", "DatasetEdition", "DatasetSelection",
        "LineageVerifier",
    ):
        assert getattr(tn, name) is getattr(governed, name)


def test_current_python_workflow_surface_is_available():
    from tn._native import governed
    from tn.providers import Providers

    assert tn.Workflow is governed.Workflow
    assert callable(Providers.session)
    for name in ("create", "workflow"):
        assert callable(getattr(tn.Session, name))
    for name in ("get", "set", "select", "inspect"):
        assert callable(getattr(tn.DataObject, name))


def test_use_context_is_validated_by_rust():
    import pytest

    context = tn.UseContext("deepvest", "portfolio_analysis", "calculate")
    assert context.application == "deepvest"
    assert context.purpose == "portfolio_analysis"
    assert context.operation == "calculate"
    with pytest.raises(ValueError):
        tn.UseContext("", "portfolio_analysis", "calculate")
