from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Any

from .runtime import LaunchConfig


def _canonical_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return tuple(stride)


def _flatten_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (tuple, list)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_ints(item))
        return tuple(flattened)
    raise TypeError("layout shapes and strides must contain only integers or nested tuples/lists of integers")


def _compact_coords(linear_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 1:
        return (linear_index,)
    remaining = linear_index
    coords: list[int] = []
    for axis, dim in enumerate(shape):
        if axis == len(shape) - 1:
            coords.append(remaining)
            break
        coords.append(remaining % dim)
        remaining //= dim
    return tuple(coords)


@dataclass(frozen=True)
class Layout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    swizzle: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _flatten_ints(self.shape))
        object.__setattr__(self, "stride", _flatten_ints(self.stride))
        if len(self.shape) != len(self.stride):
            raise ValueError("layout shape and stride must have the same rank")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError("layout shape dimensions must be > 0")
        if any(step <= 0 for step in self.stride):
            raise ValueError("layout strides must be > 0")

    def __call__(self, coord: int | tuple[int, ...]) -> int:
        if isinstance(coord, int):
            indices = _compact_coords(coord, self.shape)
        else:
            indices = _flatten_ints(coord)
        if len(indices) != len(self.shape):
            raise ValueError(f"layout expects {len(self.shape)} coordinates, got {len(indices)}")
        return sum(index * stride for index, stride in zip(indices, self.stride))

    def __repr__(self) -> str:
        return f"{self.shape}:{self.stride}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "swizzle": self.swizzle,
        }

    @classmethod
    def row_major(cls, shape: tuple[int, ...], swizzle: str | None = None) -> "Layout":
        return cls(shape=shape, stride=_canonical_stride(shape), swizzle=swizzle)

    @classmethod
    def ordered(
        cls,
        shape: tuple[int, ...],
        order: tuple[int, ...],
        swizzle: str | None = None,
    ) -> "Layout":
        if len(shape) != len(order):
            raise ValueError("layout order must have the same rank as shape")
        if tuple(sorted(order)) != tuple(range(len(shape))):
            raise ValueError("layout order must be a permutation of the shape axes")
        stride = [0] * len(shape)
        running = 1
        for axis in order:
            stride[axis] = running
            running *= shape[axis]
        return cls(shape=shape, stride=tuple(stride), swizzle=swizzle)


class AddressSpace(str, Enum):
    GLOBAL = "global"
    SHARED = "shared"
    REGISTER = "register"
    LOCAL = "local"


def normalize_address_space(value: AddressSpace | str) -> AddressSpace:
    if isinstance(value, AddressSpace):
        return value
    try:
        return AddressSpace(value)
    except ValueError as exc:
        supported = ", ".join(space.value for space in AddressSpace)
        raise ValueError(f"unsupported address_space '{value}', expected one of: {supported}") from exc


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    layout: Layout | None = None
    address_space: AddressSpace | str = AddressSpace.GLOBAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "address_space", normalize_address_space(self.address_space))

    def resolved_layout(self) -> Layout:
        return self.layout or Layout.row_major(self.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.resolved_layout().to_dict(),
            "address_space": self.address_space.value,
        }


@dataclass(frozen=True)
class ScalarSpec:
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype}


@dataclass(frozen=True)
class KernelArgument:
    name: str
    kind: str
    spec: TensorSpec | ScalarSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "spec": self.spec.to_dict(),
        }


@dataclass(frozen=True)
class Operation:
    op: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": self.attrs,
        }


@dataclass(frozen=True)
class PortableKernelIR:
    name: str
    arguments: tuple[KernelArgument, ...]
    operations: tuple[Operation, ...]
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "operations": [operation.to_dict() for operation in self.operations],
            "launch": self.launch.to_dict(),
            "metadata": self.metadata,
        }
