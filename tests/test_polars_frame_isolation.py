"""Native child frames retain their state when a constructor's input is mutated."""

import polars as pl
import tn

from deepvest_tn.unity import resolve_finance_editions
from governed_edition_fixtures import BUSINESS_USE, signed_edition
from test_publications import POLICY
from test_unity_editions import register, request, trust
from tn_polars.frame import GovernedLazyFrame
from tn_polars.registry import LocalRegistry
from tn_polars.tn_boundary import PolicySet, TnBoundary, root_name


def test_retained_constructor_handle_cannot_mutate_native_siblings(tmp_path):
    fixture = signed_edition(tmp_path / "source")
    with fixture.session as session:
        registry = LocalRegistry(tmp_path / "registry")
        register(registry, fixture)
        resolved, = resolve_finance_editions(
            [request(fixture)], registry=registry, catalog_session=session,
            policy_dag=tn.PolicyDag(), trust=trust(session.did),
        )
        policies = PolicySet.for_selection(resolved.entries, resolved.selection)
        sources = {"amount": {root_name(fixture.source.id, "amount")}}

        def accept(context):
            policies.validate_contracts(context)
            policies.decide("load", "internal", sources, set(), BUSINESS_USE)
            return context.writer == session.did

        retained = session.receive(fixture.source, use=BUSINESS_USE, groups=["finance"],
                                   selection=resolved.selection, decide=accept)
        original_policy = retained.policies[0]
        original_bindings = retained.dataset_bindings
        original_sources = {source.object_id for source in retained.sources}
        boundary = TnBoundary(retained, policies, BUSINESS_USE, "finance", "rows")
        entry = next(iter(policies._entries.values()))
        frame = GovernedLazyFrame(
            pl.DataFrame(retained.get("rows", group="finance")).lazy(),
            entry.policy, {}, entry.evaluator, sources=sources, tn_boundary=boundary,
        )
        first, second = frame.select("amount"), frame.select("amount")

        unrelated = session.create({"rows": [{"amount": 99}]}, original_policy,
                                   object_type="portfolio.feed.data.v1", group="finance")
        unrelated_publication = unrelated.release(use=BUSINESS_USE, to="internal", decide=lambda _: True)
        unrelated = session.receive(unrelated_publication, use=BUSINESS_USE,
                                    groups=["finance"], decide=lambda _: True)
        retained.include(unrelated)
        additional = tn.Governance.from_markdown(
            session.did,
            POLICY.replace("portfolio.feed.data.v1", "portfolio.additional.v1"),
            "additional.md", "portfolio.additional.v1",
        )
        retained.attach(additional, decide=lambda _: True)
        retained.set("rows", [{"amount": 999}], group="finance")
        assert len(retained.policies) == 2
        assert unrelated_publication.id in {source.object_id for source in retained.sources}

        released_ids = []
        for child in (first, second):
            independent = child.tn_object
            assert independent.get("rows", group="finance") == [{"amount": 42}]
            assert len(independent.policies) == 1
            assert independent.policies[0].matches_contract(original_policy)
            assert independent.dataset_bindings == original_bindings
            assert {source.object_id for source in independent.sources} == original_sources

            publication = child.release_tn(
                use=BUSINESS_USE, to="internal",
                decide=lambda context: context.writer == session.did and len(context.policies) == 1,
            )
            governance = session.governance(publication).governance
            assert len(governance.policies) == 1
            assert governance.policies[0].matches_contract(original_policy)
            assert governance.dataset_bindings == original_bindings
            assert {source.object_id for source in governance.sources} == original_sources
            result = session.receive(publication, use=BUSINESS_USE, groups=["finance"],
                                     decide=lambda _: True)
            assert result.get("rows", group="finance") == [{"amount": 42}]
            released_ids.append(publication.id)
        assert released_ids[0] != released_ids[1]
