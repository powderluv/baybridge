from __future__ import annotations

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
