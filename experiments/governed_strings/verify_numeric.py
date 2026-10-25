"""Run numeric integration checks with actual native encryption and signed evidence.

Run: python -m experiments.governed_strings.verify_numeric
This exercises local endpoints; it does not invoke Unity or a model service.
"""

from decimal import Decimal, localcontext, FloatOperation
import json
import argparse
import hashlib
import importlib.metadata
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import tn
from tn.governed import NotEntitled

from deepvest_tn.publications import open_publication_store
from . import Context, EvidenceRecorder, Assessment, Request, TNNumeric, OperationDenied
from .evidence import verify_tape, EvidenceError
from .numeric_codec import encode
from .provision import provision


def run(directory, *, storage="sqlite"):
    configs = provision(Path(directory) / 'sessions')
    store = open_publication_store(Path(directory) / 'publications', backend=storage)
    checks = []
    with store, tn.Session.from_config(str(configs['worker'])) as session:
        policy = session.policy('experiment.administration')
        recorder = EvidenceRecorder(session, store, session.policy('experiment.evidence'))
        context = Context(session, recorder, lambda r: Assessment(True, 'local numeric verification'),
                          application='numeric', purpose='verification', actor='worker',
                          metadata_accept=lambda c: c.writer == session.did)

        def source(raw):
            request = Request(context.application, context.purpose, 'select', (), (), (policy,), ())
            call = uuid4().hex
            decision = context._decision(request, call)
            return context.originate(raw, policy, source={'column': 'value'}, request=request,
                                     decision_id=decision, call_id=call)

        def same(value, expected):
            assert isinstance(value, TNNumeric)
            assert value.dtype is type(expected)
            actual = value.reveal(context, 'verification')
            assert type(actual) is type(expected) and encode(actual) == encode(expected)
            # A second handle must hydrate from the exact signed type and records.
            received = context.hydrate(value.reference, value.decision_id, value.result_id,
                                       kind=type(expected).__name__)
            assert encode(received.reveal(context, 'verification')) == encode(expected)

        with context:
            for raw in [0, -(2**150), 2.5, -0.0, float('inf'), float('nan'),
                        Decimal('1.2300'), Decimal('-0.00'), Decimal('sNaN42'),
                        Decimal('Infinity'), 2+3j]:
                same(source(raw), raw)
            checks.append('lossless types, scale, large integers, signed zero, infinity and NaNs')
            for left, right in [(7, 2), (7, 2.5), (2.5, 7),
                                (Decimal('7.00'), 2), (7, Decimal('2.50'))]:
                a, b = source(left), source(right)
                for operation in [lambda a,b:a+b, lambda a,b:a-b, lambda a,b:a*b,
                                  lambda a,b:a/b, lambda a,b:a//b, lambda a,b:a%b]:
                    same(operation(a,b), operation(left,right))
                    same(operation(a,right), operation(left,right))
                    same(operation(left,b), operation(left,right))
            same(source(1) + 2**150, 1 + 2**150)
            checks.append('governed, literal and reflected arithmetic with normal type promotion')
            d = source(Decimal('1.2350'))
            same(d.quantize(Decimal('.01')), Decimal('1.24'))
            same(d.quantize(source(Decimal('.01'))), Decimal('1.24'))
            same(round(d, 2), Decimal('1.24'))
            same(round(d), 1)
            same(math.floor(d), 1)
            same(math.ceil(d), 2)
            with localcontext() as decimal_context:
                decimal_context.prec = 5
                same(source(Decimal(1)) / 7, Decimal(1) / 7)
                same(source(Decimal(2)).sqrt(), Decimal(2).sqrt())
            checks.append('Decimal precision, rounding, quantize and result types')
            try:
                d + 0.5
            except TypeError:
                assert recorder.read(recorder.head).body['kind'] == 'Failure'
            else:
                raise AssertionError('Decimal plus float must raise TypeError')
            with localcontext() as dc:
                dc.traps[FloatOperation] = True
                try:
                    d < 0.5
                except FloatOperation:
                    pass
                else:
                    raise AssertionError('Decimal ordering must respect FloatOperation trap')
            for raw in [1, 1.0, Decimal('1.00')]:
                value = source(raw)
                assert value == 1 and hash(value) == hash(1)
            nan = source(float('nan'))
            assert not (nan == nan) and bool(nan)
            try:
                source(2+3j) < source(2+3j)
            except TypeError:
                pass
            else:
                raise AssertionError('complex ordering must fail even for equal numbers')
            checks.append('cross-type equality/hash, NaN semantics and native exceptions')
            a,b = source(7), source(3)
            q,r = divmod(a,b)
            same(q, 2)
            same(r, 1)
            same(pow(a, b, source(5)), 3)
            same(source(-1) ** 0.5, (-1) ** 0.5)
            checks.append('multi-result arithmetic and complex power results')
            a = a.attest('Original source numeric claim', context)
            combined = a + d
            _, binding, claims = context._metadata(combined)
            assert binding['parents'] == [a.reference.to_dict(), d.reference.to_dict()]
            assert len(claims) == 1 and claims[0].subjects == (source_subject := context._metadata(a)[2][0].subjects)
            assert source_subject != (combined.reference,)
            same(combined.copy(), Decimal('8.2350'))
            checks.append('both parents and original claim subjects retained across mixed arithmetic and copy')
            try:
                context.hydrate(d.reference, d.decision_id, d.result_id, kind='float')
            except EvidenceError:
                pass
            else:
                raise AssertionError('hydration must reject a substituted numeric type')
            original_open = context._open_values
            opened = []
            def track(*args):
                opened.append(True)
                return original_open(*args)
            context._open_values = track
            context.evaluator = lambda request: Assessment(False, 'refusal probe')
            try:
                a + 1
            except OperationDenied:
                assert not opened
            else:
                raise AssertionError('refused operation ran')
            context.evaluator = lambda request: Assessment(True, 'missing obligation', ('unavailable',))
            try:
                a + 1
            except OperationDenied:
                assert not opened
            else:
                raise AssertionError('operation ran without its prerequisite')
            context._open_values = original_open
            context.evaluator = lambda request: Assessment(True, 'local numeric verification')
            checks.append('refusal and unmet obligation stop before payload opening')
        try:
            a + 1
        except OperationDenied:
            pass
        else:
            raise AssertionError('content operation ran without a context')
        with tn.Session.from_config(str(configs['metadata_only'])) as reader:
            other_recorder = EvidenceRecorder(reader, store, recorder.policy)
            other = Context(reader, other_recorder, lambda r: Assessment(True, 'permit without key'),
                            application='numeric', purpose='verification', actor='metadata_only',
                            metadata_accept=lambda c: c.writer == session.did)
            handle = other.hydrate(d.reference, d.decision_id, d.result_id, kind='Decimal')
            with other:
                try:
                    handle + 1
                except NotEntitled:
                    assert other_recorder.read(other_recorder.head).body['kind'] == 'Failure'
                else:
                    raise AssertionError('operation succeeded without a values key')
            other_recorder.export(Path(directory) / 'metadata-reader-records')
            checks.append('permitted numeric computation still requires a decryption key')
        records = verify_tape(recorder.records(), session=session, expected_writer=session.did,
                              expected_context=recorder.context_id, expected_head=recorder.head,
                              policy=recorder.policy)
        checks.append('independent signature and record-chain verification')
        checkpoint = recorder.export(Path(directory) / 'worker-records')
        return {'checks': checks, 'signed_records_verified': len(records), 'checkpoint': checkpoint,
                'storage': storage,
                'tn_version': importlib.metadata.version('tn-proto'),
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob('*.py')},
                'native_crypto': True, 'unity_or_model_invoked': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path)
    parser.add_argument('--storage', choices=('files', 'sqlite'), default='sqlite',
                        help='Publication storage (default: sqlite; files retains separate-file storage)')
    args = parser.parse_args()
    if args.run_directory:
        result = run(args.run_directory, storage=args.storage)
        (args.run_directory / 'numeric-verification.json').write_text(json.dumps(result, indent=2))
    else:
        with TemporaryDirectory(prefix='tn-numeric-') as folder:
            result = run(folder, storage=args.storage)
    print(json.dumps(result, indent=2))
