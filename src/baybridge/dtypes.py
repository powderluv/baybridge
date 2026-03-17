from __future__ import annotations

import struct
from typing import Any


_WIDTH_TABLE = {
    "i1": 1,
    "i8": 8,
    "i32": 32,
    "i64": 64,
    "fp8": 8,
    "bf8": 8,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "index": 64,
}

_FLOAT_DTYPES = {"f16", "bf16", "f32"}
_INTEGER_DTYPES = {"i1", "i8", "i32", "i64", "index"}
_STORAGE_ONLY_DTYPES = {"fp8", "bf8"}
_ALIAS_TABLE = {
    "bool": "i1",
    "torch.bool": "i1",
    "int": "index",
    "torch.int": "index",
    "int8": "i8",
    "torch.int8": "i8",
    "int32": "i32",
    "torch.int32": "i32",
    "int64": "i64",
    "torch.int64": "i64",
    "float": "f32",
    "float16": "f16",
    "torch.float16": "f16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "float8_e4m3fnuz": "fp8",
    "torch.float8_e4m3fnuz": "fp8",
    "float8_e5m2fnuz": "bf8",
    "torch.float8_e5m2fnuz": "bf8",
    "float32": "f32",
    "torch.float32": "f32",
    "torch.float": "f32",
}


class ElementType(str):
    @property
    def width(self) -> int:
        try:
            return _WIDTH_TABLE[str(self)]
        except KeyError as exc:
            raise ValueError(f"baybridge element type '{self}' does not define a bit width") from exc


def element_type(dtype: str) -> ElementType:
    return ElementType(dtype)


def resolve_element_type_name(value: Any) -> str:
    if isinstance(value, ElementType):
        return normalize_dtype_name(str(value))
    if isinstance(value, str):
        return normalize_dtype_name(value)
    dtype = getattr(value, "__baybridge_dtype__", None)
    if isinstance(dtype, str):
        return normalize_dtype_name(dtype)
    dtype = getattr(value, "dtype", None)
    if isinstance(dtype, str):
        return normalize_dtype_name(dtype)
    raise TypeError("element type values must be baybridge dtype strings or scalar constructors")


def normalize_dtype_name(dtype: str) -> str:
    normalized = _ALIAS_TABLE.get(dtype, dtype)
    if normalized not in _WIDTH_TABLE:
        raise ValueError(f"unsupported baybridge dtype '{dtype}'")
    return normalized


def is_float_dtype(dtype: str) -> bool:
    return normalize_dtype_name(dtype) in _FLOAT_DTYPES


def is_integer_dtype(dtype: str) -> bool:
    return normalize_dtype_name(dtype) in _INTEGER_DTYPES


def is_storage_only_dtype(dtype: str) -> bool:
    return normalize_dtype_name(dtype) in _STORAGE_ONLY_DTYPES


def _f32_to_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _pack_fp8_scalar(value: float) -> int:
    bias = 8
    max_stored_exp = 15
    max_magnitude = 240.0
    bits = _f32_to_bits(float(value))
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    if exp == 0xFF:
        return 0x80
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return 0x00
    if magnitude > max_magnitude:
        return (sign << 7) | 0x7F

    f32_exp = exp - 127
    fp8_exp = f32_exp + bias
    if fp8_exp <= 0:
        shift = 1 - fp8_exp
        full_mantissa = (1 << 23) | frac
        total_shift = 20 + shift
        if total_shift >= 24:
            return 0x00
        rounded = (full_mantissa + (1 << (total_shift - 1))) >> total_shift
        if rounded > 7:
            rounded = 7
        if rounded == 0:
            return 0x00
        return (sign << 7) | rounded
    if fp8_exp > max_stored_exp:
        return (sign << 7) | 0x7F

    round_bit = 1 << 19
    guard_bit = frac & round_bit
    remainder = frac & (round_bit - 1)
    mantissa = frac >> 20
    if guard_bit and (remainder > 0 or (mantissa & 1)):
        mantissa += 1
        if mantissa > 7:
            mantissa = 0
            fp8_exp += 1
            if fp8_exp > max_stored_exp:
                return (sign << 7) | 0x7F
    return (sign << 7) | (fp8_exp << 3) | mantissa


def _pack_bf8_scalar(value: float) -> int:
    bias = 16
    max_stored_exp = 31
    max_magnitude = 57344.0
    bits = _f32_to_bits(float(value))
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    if exp == 0xFF:
        return 0x80
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return 0x00
    if magnitude > max_magnitude:
        return (sign << 7) | 0x7F

    f32_exp = exp - 127
    bf8_exp = f32_exp + bias
    if bf8_exp <= 0:
        shift = 1 - bf8_exp
        full_mantissa = (1 << 23) | frac
        total_shift = 21 + shift
        if total_shift >= 24:
            return 0x00
        rounded = (full_mantissa + (1 << (total_shift - 1))) >> total_shift
        if rounded > 3:
            rounded = 3
        if rounded == 0:
            return 0x00
        return (sign << 7) | rounded
    if bf8_exp > max_stored_exp:
        return (sign << 7) | 0x7F

    round_bit = 1 << 20
    guard_bit = frac & round_bit
    remainder = frac & (round_bit - 1)
    mantissa = frac >> 21
    if guard_bit and (remainder > 0 or (mantissa & 1)):
        mantissa += 1
        if mantissa > 3:
            mantissa = 0
            bf8_exp += 1
            if bf8_exp > max_stored_exp:
                return (sign << 7) | 0x7F
    return (sign << 7) | (bf8_exp << 2) | mantissa


def _unpack_fp8_scalar(raw: int) -> float:
    raw = int(raw) & 0xFF
    if raw == 0x80:
        return float("nan")
    sign = -1.0 if (raw & 0x80) else 1.0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    if exponent == 0:
        return sign * ((mantissa / 8.0) * (2.0**-7))
    return sign * ((1.0 + mantissa / 8.0) * (2.0 ** (exponent - 8)))


def _unpack_bf8_scalar(raw: int) -> float:
    raw = int(raw) & 0xFF
    if raw == 0x80:
        return float("nan")
    sign = -1.0 if (raw & 0x80) else 1.0
    exponent = (raw >> 2) & 0x1F
    mantissa = raw & 0x3
    if exponent == 0:
        return sign * ((mantissa / 4.0) * (2.0**-15))
    return sign * ((1.0 + mantissa / 4.0) * (2.0 ** (exponent - 16)))


def pack_storage_only_scalar(value: Any, dtype: str) -> int:
    dtype = normalize_dtype_name(dtype)
    if dtype == "fp8":
        return _pack_fp8_scalar(float(value))
    if dtype == "bf8":
        return _pack_bf8_scalar(float(value))
    raise ValueError(f"baybridge dtype '{dtype}' is not a storage-only float8 format")


def unpack_storage_only_scalar(value: Any, dtype: str) -> float:
    dtype = normalize_dtype_name(dtype)
    if dtype == "fp8":
        return _unpack_fp8_scalar(int(value))
    if dtype == "bf8":
        return _unpack_bf8_scalar(int(value))
    raise ValueError(f"baybridge dtype '{dtype}' is not a storage-only float8 format")


def promote_scalar_dtype(lhs: str, rhs: str) -> str:
    lhs = normalize_dtype_name(lhs)
    rhs = normalize_dtype_name(rhs)
    if lhs in _STORAGE_ONLY_DTYPES or rhs in _STORAGE_ONLY_DTYPES:
        raise TypeError(
            f"baybridge dtype '{lhs if lhs in _STORAGE_ONLY_DTYPES else rhs}' is storage-only and does not support scalar arithmetic"
        )
    if lhs == rhs:
        return lhs
    if is_float_dtype(lhs) or is_float_dtype(rhs):
        return "f32"
    if lhs == "index" or rhs == "index":
        return "index"
    return lhs if _WIDTH_TABLE[lhs] >= _WIDTH_TABLE[rhs] else rhs


def cast_scalar_value(value: Any, dtype: str) -> bool | int | float:
    dtype = normalize_dtype_name(dtype)
    if dtype == "i1":
        return bool(value)
    if dtype in _STORAGE_ONLY_DTYPES:
        raw = int(value)
        width = _WIDTH_TABLE[dtype]
        return raw % (1 << width)
    if is_float_dtype(dtype):
        return float(value)
    raw = int(value)
    if dtype == "index":
        return raw
    width = _WIDTH_TABLE[dtype]
    wrapped = raw % (1 << width)
    sign_bit = 1 << (width - 1)
    if wrapped >= sign_bit:
        wrapped -= 1 << width
    return wrapped


def dtype_constructor_name(dtype: str) -> str:
    table = {
        "i1": "Boolean",
        "i8": "Int8",
        "i32": "Int32",
        "i64": "Int64",
        "fp8": "Float8E4M3FNUZ",
        "bf8": "BFloat8E5M2FNUZ",
        "f16": "Float16",
        "bf16": "BFloat16",
        "f32": "Float32",
        "index": "Int",
    }
    return table[normalize_dtype_name(dtype)]
