"""Lossless numeric payloads inside the native encrypted TN publication."""

from decimal import Decimal
import math
import struct

NUMERIC_TYPES = (int, float, Decimal, complex)
KINDS = {t.__name__: t for t in NUMERIC_TYPES}


def encode(value):
    if type(value) is str:
        return value
    if type(value) is int:
        return str(value)  # preserves the integer wire representation
    if type(value) is float:
        return struct.pack('>d', value).hex()
    if type(value) is Decimal:
        return str(value)  # retains scale, signed zero, infinities and NaN payloads
    if type(value) is complex:
        return struct.pack('>dd', value.real, value.imag).hex()
    raise TypeError(f'unsupported primitive type: {type(value).__name__}')


def decode(payload, kind):
    if type(payload) is not str:
        raise TypeError('primitive payload must be encoded text')
    if kind == 'str':
        return payload
    if kind == 'int':
        return int(payload)
    if kind == 'float':
        return struct.unpack('>d', bytes.fromhex(payload))[0]
    if kind == 'Decimal':
        return Decimal(payload)
    if kind == 'complex':
        return complex(*struct.unpack('>dd', bytes.fromhex(payload)))
    raise TypeError(f'unsupported numeric kind: {kind}')


def record_value(value):
    """Make numeric parameters exact and JSON-safe for signed evidence."""
    if type(value) in (float, Decimal, complex) or (type(value) is int and not -(2**63) <= value < 2**63):
        return {'numeric_type': type(value).__name__, 'encoding': encode(value)}
    if isinstance(value, (tuple, list)):
        return [record_value(item) for item in value]
    if isinstance(value, dict):
        return {key: record_value(item) for key, item in value.items()}
    return value


def equality_key(value):
    """Equal finite numbers share a key across types; NaNs need runtime comparison."""
    if type(value) is complex:
        if value.imag == 0:
            return equality_key(value.real)
        if math.isnan(value.real) or math.isnan(value.imag):
            return None
        return ('complex', equality_key(value.real), equality_key(value.imag))
    if type(value) is Decimal:
        if value.is_nan():
            return None
        if value.is_infinite():
            return ('infinity', '-' if value.is_signed() else '+')
    if type(value) is float:
        if math.isnan(value):
            return None
        if math.isinf(value):
            return ('infinity', '-' if value < 0 else '+')
    numerator, denominator = value.as_integer_ratio()
    return ('ratio', str(numerator), str(denominator))
