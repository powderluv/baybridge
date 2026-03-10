from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, get_type_hints

from .dtypes import element_type
from .runtime import Pointer


@dataclass(frozen=True)
class MemRangeType:
    element_type: str
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("MemRange count must be > 0")


class MemRange:
    def __class_getitem__(cls, params: Any) -> MemRangeType:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("MemRange[...] expects (element_type, count)")
        dtype, count = params
        return MemRangeType(str(dtype), int(count))


@dataclass(frozen=True)
class AlignType:
    base: Any
    alignment: int

    def __post_init__(self) -> None:
        if self.alignment <= 0:
            raise ValueError("Align alignment must be > 0")


class Align:
    def __class_getitem__(cls, params: Any) -> AlignType:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Align[...] expects (base_type, alignment)")
        base, alignment = params
        return AlignType(base=base, alignment=int(alignment))


@dataclass(frozen=True)
class StructField:
    name: str
    annotation: Any
    kind: str
    dtype: str | None
    count: int | None
    struct_type: type | None
    offset_bytes: int
    size_bytes: int
    alignment_bytes: int


@dataclass(frozen=True)
class StructSpec:
    name: str
    fields: tuple[StructField, ...]
    size_bytes: int
    alignment_bytes: int


class StructFieldProxy:
    @property
    def dtype(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class MemRangeProxy(StructFieldProxy):
    pointer: Pointer
    count: int

    @property
    def dtype(self) -> str:
        return self.pointer.value_type

    def data_ptr(self) -> Pointer:
        return self.pointer

    def get_tensor(self, layout: Any):
        from .frontend import make_tensor

        shape = tuple(layout.shape)
        if prod(shape) != self.count:
            raise ValueError(f"layout shape {shape} does not match MemRange element count {self.count}")
        return make_tensor(self.pointer, layout)

    def __repr__(self) -> str:
        return f"MemRange(dtype={self.dtype}, count={self.count})"


class StructInstance:
    def __init__(self, name: str, fields: dict[str, Any]):
        self._name = name
        for field_name, value in fields.items():
            setattr(self, field_name, value)

    def __repr__(self) -> str:
        field_names = ", ".join(sorted(key for key in self.__dict__ if not key.startswith("_")))
        return f"{self._name}({field_names})"


class _StructNamespace:
    MemRange = MemRange
    Align = Align

    def __call__(self, cls: type) -> type:
        spec = build_struct_spec(cls)
        setattr(cls, "__baybridge_struct__", spec)
        return cls


struct = _StructNamespace()


def is_struct_type(value: Any) -> bool:
    return hasattr(value, "__baybridge_struct__")


def get_struct_spec(value: Any) -> StructSpec:
    spec = getattr(value, "__baybridge_struct__", None)
    if spec is None:
        raise TypeError(f"{value!r} is not a baybridge @struct type")
    return spec


def build_struct_spec(cls: type) -> StructSpec:
    try:
        annotations = get_type_hints(cls, include_extras=False)
    except Exception:
        annotations = dict(getattr(cls, "__annotations__", {}))
    if not annotations:
        raise TypeError("baybridge @struct classes require at least one annotated field")

    fields: list[StructField] = []
    offset = 0
    struct_alignment = 1
    for name, annotation in annotations.items():
        field = _resolve_field(name, annotation, offset)
        fields.append(field)
        offset = field.offset_bytes + field.size_bytes
        struct_alignment = max(struct_alignment, field.alignment_bytes)
    total_size = _align_up(offset, struct_alignment)
    return StructSpec(
        name=cls.__name__,
        fields=tuple(fields),
        size_bytes=total_size,
        alignment_bytes=struct_alignment,
    )


def _resolve_field(name: str, annotation: Any, current_offset: int) -> StructField:
    alignment_override: int | None = None
    normalized = annotation
    if isinstance(annotation, AlignType):
        alignment_override = annotation.alignment
        normalized = annotation.base

    if isinstance(normalized, MemRangeType):
        size_bytes = normalized.count * _dtype_size(normalized.element_type)
        natural_alignment = _dtype_size(normalized.element_type)
        kind = "memrange"
        dtype = normalized.element_type
        count = normalized.count
        struct_type = None
    elif is_struct_type(normalized):
        nested = get_struct_spec(normalized)
        size_bytes = nested.size_bytes
        natural_alignment = nested.alignment_bytes
        kind = "struct"
        dtype = None
        count = None
        struct_type = normalized
    else:
        dtype = str(normalized)
        size_bytes = _dtype_size(dtype)
        natural_alignment = _dtype_size(dtype)
        kind = "scalar"
        count = None
        struct_type = None

    alignment = max(natural_alignment, alignment_override or 1)
    offset = _align_up(current_offset, alignment)
    return StructField(
        name=name,
        annotation=annotation,
        kind=kind,
        dtype=dtype,
        count=count,
        struct_type=struct_type,
        offset_bytes=offset,
        size_bytes=size_bytes,
        alignment_bytes=alignment,
    )


def _dtype_size(dtype: str) -> int:
    return (element_type(str(dtype)).width + 7) // 8


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
