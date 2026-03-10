from __future__ import annotations

from typing import Any


_WIDTH_TABLE = {
    "i1": 1,
    "i8": 8,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "index": 64,
}

_FLOAT_DTYPES = {"f16", "bf16", "f32"}
_INTEGER_DTYPES = {"i1", "i8", "i32", "i64", "index"}
_ALIAS_TABLE = {
    "bool": "i1",
    "int": "index",
    "float": "f32",
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


def normalize_dtype_name(dtype: str) -> str:
    normalized = _ALIAS_TABLE.get(dtype, dtype)
    if normalized not in _WIDTH_TABLE:
        raise ValueError(f"unsupported baybridge dtype '{dtype}'")
    return normalized


def is_float_dtype(dtype: str) -> bool:
    return normalize_dtype_name(dtype) in _FLOAT_DTYPES


def is_integer_dtype(dtype: str) -> bool:
    return normalize_dtype_name(dtype) in _INTEGER_DTYPES


def promote_scalar_dtype(lhs: str, rhs: str) -> str:
    lhs = normalize_dtype_name(lhs)
    rhs = normalize_dtype_name(rhs)
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
        "f16": "Float16",
        "bf16": "BFloat16",
        "f32": "Float32",
        "index": "Int",
    }
    return table[normalize_dtype_name(dtype)]
