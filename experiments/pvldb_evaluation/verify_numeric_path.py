"""Live SQL/Unity/Polars numeric cases and targeted refusal probes (no model)."""

import argparse
from contextlib import redirect_stdout
from decimal import Decimal
import json
from pathlib import Path
from unittest.mock import patch

from .frames import DataframeContext, DataframeGovernance
from .numeric_profile import numeric_profile, build_frame, check_fixed_plan_range
from .run import run_case


def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for mode in ('primitive', 'explicit'):
        for kind in ('int', 'float', 'Decimal'):
            label = f'{mode}-{kind}'
            with (root / f'{label}.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
                result = run_case(root / label, mode, rows=3, numeric_type=kind)
            assert result['completed'] and result['verified'], result
            facts = result['dataframe'][0]
            assert facts['numeric_profile']['kind'] == kind
            assert facts['input_schema']['value'] == facts['output_schema']['amount']
            outcomes.append({'case': label, 'completed': True, 'verified': True,
                             'input_schema': facts['input_schema'], 'output_schema': facts['output_schema'],
                             'output': facts['output'],
                             'opa_decisions_replayed': result['dataframe_verification']['opa_decisions_replayed']})

    # A real, signed Decimal source that does not fit its declared scale.
    with (root / 'lossy-scale.log').open('w') as log, redirect_stdout(log):
        result = run_case(root / 'lossy-scale', 'primitive', rows=2,
                          numeric_type='Decimal', decimal_scale=2)
    assert not result['completed'] and result['error']['stage'] == 'dataframe'
    assert result['error']['type'] == 'ValueError'
    assert not result.get('dataframe')
    outcomes.append({'case': 'lossy-scale', 'rejected': True, 'error': result['error']})

    # Change profile bytes after the authority computed their signed artifact hash.
    original_artifacts = DataframeGovernance._evaluator_artifacts
    original_open = DataframeContext._open_values
    opens = []

    def tamper(self, *args):
        artifacts, files = original_artifacts(self, *args)
        files = dict(files)
        files['profile.json'] = b'{"dtypes":{"name":"String","value":"Float64"}}'
        return artifacts, files

    def observe_open(self, request, values):
        if request.operation == 'dataframe_calculate':
            opens.append(True)
        return original_open(self, request, values)

    with patch.object(DataframeGovernance, '_evaluator_artifacts', tamper), \
            patch.object(DataframeContext, '_open_values', observe_open), \
            (root / 'tampered-profile.log').open('w') as log, redirect_stdout(log):
        result = run_case(root / 'tampered-profile', 'primitive', rows=2)
    assert not result['completed'] and not opens
    assert result['error']['stage'] == 'query'
    outcomes.append({'case': 'tampered-profile', 'rejected': True,
                     'dataframe_payload_opens': len(opens), 'error': result['error']})

    # Signed assignment/profile inconsistency: a valid signature is not sufficient.
    def conflicting(self, policy, application, profile):
        return original_artifacts(self, policy, application, numeric_profile('float'))

    opens.clear()
    with patch.object(DataframeGovernance, '_evaluator_artifacts', conflicting), \
            patch.object(DataframeContext, '_open_values', observe_open), \
            (root / 'conflicting-profile.log').open('w') as log, redirect_stdout(log):
        result = run_case(root / 'conflicting-profile', 'primitive', rows=2)
    assert not result['completed'] and not opens
    assert result['error']['stage'] == 'dataframe' and result['error']['type'] == 'EvidenceError'
    outcomes.append({'case': 'conflicting-profile', 'rejected': True,
                     'dataframe_payload_opens': len(opens), 'error': result['error']})

    # Numeric storage boundaries: native Polars must not silently discard digits.
    cases = [
        ('precision', Decimal('1234.50'), numeric_profile('Decimal', 5, 2)),
        ('scale', Decimal('1.125'), numeric_profile('Decimal', 38, 2)),
        ('tiny', Decimal('1e-999999999'), numeric_profile('Decimal', 38, 2)),
        ('negative-zero', Decimal('-0.00'), numeric_profile('Decimal', 38, 2)),
        ('integer-range', 2**63, numeric_profile('int')),
    ]
    for label, value, profile in cases:
        try:
            build_frame([{'name': 'A', 'value': value}], profile)
        except (ValueError, OverflowError):
            outcomes.append({'case': label, 'rejected': True})
        else:
            raise AssertionError(label)
    for value, profile in [(2**63-1, numeric_profile('int')),
                           (Decimal('99.99'), numeric_profile('Decimal', 4, 2))]:
        frame = build_frame([{'name': 'A', 'value': value}], profile)
        try:
            check_fixed_plan_range(frame, profile)
        except OverflowError:
            outcomes.append({'case': profile['kind'] + '-arithmetic-overflow', 'rejected': True})
        else:
            raise AssertionError('overflow was not rejected')
    report = {'cases': outcomes, 'real_sql_unity_polars_opa': True, 'model_invoked': False}
    (root / 'numeric-path-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_directory), indent=2))
