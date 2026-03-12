from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterator, Sequence

from .dtypes import (
    cast_scalar_value,
    dtype_constructor_name,
    element_type,
    is_float_dtype,
    normalize_dtype_name,
    promote_scalar_dtype,
)


@dataclass(frozen=True)
class LaunchConfig:
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    cluster: tuple[int, int, int] = (1, 1, 1)
    shared_mem_bytes: int = 0
    cooperative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", _normalize_dim3(self.grid, name="grid"))
        object.__setattr__(self, "block", _normalize_dim3(self.block, name="block"))
        object.__setattr__(self, "cluster", _normalize_dim3(self.cluster, name="cluster"))
        if self.shared_mem_bytes < 0:
            raise ValueError("shared_mem_bytes must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        launch = {
            "grid": list(self.grid),
            "block": list(self.block),
            "cluster": list(self.cluster),
            "shared_mem_bytes": self.shared_mem_bytes,
        }
        if self.cooperative:
            launch["cooperative"] = True
        return launch


@dataclass(frozen=True)
class TensorHandle:
    capsule: Any
    shape: tuple[int, ...]
    dtype: str
    device_type: int
    device_id: int
    source: Any | None = None
    stride: tuple[int, ...] | None = None
    assumed_align: int | None = None
    dynamic_layout_leading_dim: int | None = None
    raw_address: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(self, "dtype", normalize_dtype_name(str(self.dtype)))
        if self.stride is not None:
            object.__setattr__(self, "stride", tuple(int(dim) for dim in self.stride))

    def mark_layout_dynamic(self, leading_dim: int | None = None) -> "TensorHandle":
        if leading_dim is not None and (leading_dim < 0 or leading_dim >= len(self.shape)):
            raise ValueError(f"leading_dim must be within [0, {len(self.shape)})")
        return replace(self, dynamic_layout_leading_dim=leading_dim)

    def data_ptr(self) -> int:
        if self.raw_address is not None:
            return int(self.raw_address)
        if self.source is not None and hasattr(self.source, "data_ptr"):
            return int(self.source.data_ptr())
        return 0

    def __dlpack__(self, stream: Any | None = None):
        if self.source is not None and hasattr(self.source, "__dlpack__"):
            try:
                if stream is not None:
                    return self.source.__dlpack__(stream=stream)
            except TypeError:
                pass
            return self.source.__dlpack__()
        if self.capsule is None:
            raise TypeError("tensor handle does not carry a DLPack capsule")
        return self.capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        if self.source is not None and hasattr(self.source, "__dlpack_device__"):
            return tuple(self.source.__dlpack_device__())
        return int(self.device_type), int(self.device_id)

    def to_runtime_tensor(self) -> "RuntimeTensor":
        if self.source is not None and hasattr(self.source, "tolist"):
            return tensor(self.source.tolist(), dtype=self.dtype)
        raise TypeError("tensor handle cannot be converted to a baybridge runtime tensor")

    def copy_from_runtime_tensor(self, value: "RuntimeTensor") -> None:
        if self.source is None:
            return
        if value.shape != self.shape:
            raise ValueError(f"runtime tensor shape mismatch for copy-back: expected {self.shape}, got {value.shape}")
        if hasattr(self.source, "copy_from_runtime_tensor"):
            self.source.copy_from_runtime_tensor(value)
            return
        try:
            for index in _iter_indices(value.shape):
                target_index: int | tuple[int, ...]
                if len(index) == 1:
                    target_index = index[0]
                else:
                    target_index = index
                self.source[target_index] = value[index]
            return
        except Exception:
            pass
        if isinstance(self.source, list):
            self.source[:] = value.tolist()
            return
        raise TypeError("tensor handle source does not support baybridge runtime copy-back")


@dataclass(frozen=True)
class Pointer:
    value_type: str
    tensor: Any | None = None
    raw_address: int | None = None
    address_space: Any = "global"
    assumed_align: int | None = None

    def __c_pointers__(self):
        return (self.raw_address or 0,)

    def __get_mlir_types__(self):
        return ("ptr",)

    def __extract_mlir_values__(self):
        return (self.tensor,) if self.tensor is not None else (self.raw_address or 0,)

    def __new_from_mlir_values__(self, values):
        value = values[0] if isinstance(values, (list, tuple)) else values
        if isinstance(value, int):
            return Pointer(
                value_type=self.value_type,
                raw_address=value,
                address_space=self.address_space,
                assumed_align=self.assumed_align,
            )
        return Pointer(
            value_type=self.value_type,
            tensor=value,
            address_space=self.address_space,
            assumed_align=self.assumed_align,
        )

    @property
    def dtype(self) -> str:
        return self.value_type

    @property
    def _pointer(self) -> int | None:
        if self.raw_address is not None:
            return int(self.raw_address)
        if self.tensor is not None and hasattr(self.tensor, "data_ptr"):
            data_ptr = self.tensor.data_ptr()
            if isinstance(data_ptr, Pointer):
                return data_ptr.raw_address
            if data_ptr is not None:
                return int(data_ptr)
        return None

    @property
    def memspace(self) -> Any:
        return self.address_space

    def data_ptr(self) -> "Pointer":
        return self


@dataclass(frozen=True)
class RuntimeScalar:
    value: bool | int | float
    dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", normalize_dtype_name(self.dtype))
        object.__setattr__(self, "value", cast_scalar_value(self.value, self.dtype))

    @property
    def element_type(self):
        return element_type(self.dtype)

    def to(self, target: Any) -> "RuntimeScalar | Any":
        return make_scalar(_unwrap_scalar_host_value(self), dtype=_resolve_scalar_target_dtype(target))

    def _binary(self, other: Any, op, *, reverse: bool = False, result_dtype: str | None = None) -> "RuntimeScalar":
        rhs = _coerce_runtime_scalar(other, preferred_dtype=self.dtype)
        lhs = rhs if reverse else self
        rhs = self if reverse else rhs
        dtype = normalize_dtype_name(result_dtype or promote_scalar_dtype(lhs.dtype, rhs.dtype))
        lhs_value = cast_scalar_value(lhs.value, dtype)
        rhs_value = cast_scalar_value(rhs.value, dtype)
        return RuntimeScalar(op(lhs_value, rhs_value), dtype)

    def _compare(self, other: Any, op, *, reverse: bool = False) -> "RuntimeScalar":
        rhs = _coerce_runtime_scalar(other, preferred_dtype=self.dtype)
        lhs = rhs if reverse else self
        rhs = self if reverse else rhs
        dtype = promote_scalar_dtype(lhs.dtype, rhs.dtype)
        lhs_value = cast_scalar_value(lhs.value, dtype)
        rhs_value = cast_scalar_value(rhs.value, dtype)
        return RuntimeScalar(op(lhs_value, rhs_value), "i1")

    def __add__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs + rhs)

    def __radd__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs + rhs, reverse=True)

    def __sub__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs - rhs)

    def __rsub__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs - rhs, reverse=True)

    def __mul__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs * rhs)

    def __rmul__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs * rhs, reverse=True)

    def __truediv__(self, other: Any) -> "RuntimeScalar":
        rhs = _coerce_runtime_scalar(other, preferred_dtype=self.dtype)
        dtype = promote_scalar_dtype(self.dtype, rhs.dtype)
        lhs_value = cast_scalar_value(self.value, dtype)
        rhs_value = cast_scalar_value(rhs.value, dtype)
        if is_float_dtype(dtype):
            return RuntimeScalar(lhs_value / rhs_value, dtype)
        return RuntimeScalar(int(lhs_value / rhs_value), dtype)

    def __rtruediv__(self, other: Any) -> "RuntimeScalar":
        lhs = _coerce_runtime_scalar(other, preferred_dtype=self.dtype)
        dtype = promote_scalar_dtype(lhs.dtype, self.dtype)
        lhs_value = cast_scalar_value(lhs.value, dtype)
        rhs_value = cast_scalar_value(self.value, dtype)
        if is_float_dtype(dtype):
            return RuntimeScalar(lhs_value / rhs_value, dtype)
        return RuntimeScalar(int(lhs_value / rhs_value), dtype)

    def __and__(self, other: Any) -> "RuntimeScalar":
        if is_float_dtype(self.dtype):
            raise TypeError("bitwise and is only supported for integer baybridge scalars")
        return self._binary(other, lambda lhs, rhs: lhs & rhs)

    def __rand__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs & rhs, reverse=True)

    def __or__(self, other: Any) -> "RuntimeScalar":
        if is_float_dtype(self.dtype):
            raise TypeError("bitwise or is only supported for integer baybridge scalars")
        return self._binary(other, lambda lhs, rhs: lhs | rhs)

    def __ror__(self, other: Any) -> "RuntimeScalar":
        return self._binary(other, lambda lhs, rhs: lhs | rhs, reverse=True)

    def __neg__(self) -> "RuntimeScalar":
        return RuntimeScalar(-_unwrap_scalar_host_value(self), self.dtype)

    def __invert__(self) -> "RuntimeScalar":
        if is_float_dtype(self.dtype):
            raise TypeError("bitwise invert is only supported for integer baybridge scalars")
        return RuntimeScalar(~int(_unwrap_scalar_host_value(self)), self.dtype)

    def __lt__(self, other: Any) -> "RuntimeScalar":
        return self._compare(other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other: Any) -> "RuntimeScalar":
        return self._compare(other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other: Any) -> "RuntimeScalar":
        return self._compare(other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other: Any) -> "RuntimeScalar":
        return self._compare(other, lambda lhs, rhs: lhs >= rhs)

    def __eq__(self, other: object) -> "RuntimeScalar":  # type: ignore[override]
        return self._compare(other, lambda lhs, rhs: lhs == rhs)

    def __ne__(self, other: object) -> "RuntimeScalar":  # type: ignore[override]
        return self._compare(other, lambda lhs, rhs: lhs != rhs)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{dtype_constructor_name(self.dtype)}({self.value!r})"


class ReductionOp(str, Enum):
    ADD = "add"
    MUL = "mul"
    MAX = "max"
    MIN = "min"


@dataclass(frozen=True)
class ExecutionState:
    launch: LaunchConfig
    block_idx: tuple[int, int, int]
    thread_idx: tuple[int, int, int]

    @property
    def block_dim(self) -> tuple[int, int, int]:
        return self.launch.block

    @property
    def grid_dim(self) -> tuple[int, int, int]:
        return self.launch.grid

    @property
    def cluster_dim(self) -> tuple[int, int, int]:
        return self.launch.cluster

    @property
    def cluster_idx(self) -> tuple[int, int, int]:
        return tuple(block // cluster for block, cluster in zip(self.block_idx, self.cluster_dim))

    @property
    def block_idx_in_cluster(self) -> tuple[int, int, int]:
        return tuple(block % cluster for block, cluster in zip(self.block_idx, self.cluster_dim))

    @property
    def cluster_size(self) -> int:
        x, y, z = self.cluster_dim
        return x * y * z

    @property
    def block_rank_in_cluster(self) -> int:
        x, y, z = self.block_idx_in_cluster
        dim_x, dim_y, _ = self.cluster_dim
        return x + (dim_x * (y + (dim_y * z)))


_current_execution: ContextVar[ExecutionState | None] = ContextVar("baybridge_execution", default=None)


class RuntimeTensor:
    def __init__(
        self,
        storage: list[Any],
        shape: tuple[int, ...],
        *,
        dtype: str,
        stride: tuple[int, ...] | None = None,
        offset: int = 0,
    ):
        self._storage = storage
        self.shape = shape
        self.dtype = dtype
        self.stride = stride or _canonical_stride(shape)
        self.offset = offset

    @classmethod
    def from_data(cls, data: Any, *, dtype: str | None = None) -> "RuntimeTensor":
        shape = _infer_shape(data)
        flat: list[Any] = []
        _flatten_nested(data, flat)
        inferred_dtype = dtype or _infer_tensor_dtype(flat)
        return cls(flat, shape, dtype=inferred_dtype)

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, dtype: str = "f32") -> "RuntimeTensor":
        size = 1
        for dim in shape:
            size *= dim
        return cls([0 for _ in range(size)], shape, dtype=dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def element_type(self) -> str:
        return element_type(self.dtype)

    @property
    def type(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, stride={self.stride})"

    @property
    def layout(self):
        from .ir import Layout

        return Layout(shape=self.shape, stride=self.stride)

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def tolist(self) -> Any:
        return _unflatten_nested(self._storage, self.shape, self.stride, self.offset)

    def view(self, *, shape: tuple[int, ...], offset_elements: tuple[int, ...]) -> "RuntimeTensor":
        if len(shape) != len(self.shape):
            raise ValueError("view shape rank must match tensor rank")
        linear_offset = self.offset
        for axis, index in enumerate(offset_elements):
            linear_offset += index * self.stride[axis]
        return RuntimeTensor(self._storage, shape, dtype=self.dtype, stride=self.stride, offset=linear_offset)

    def broadcast_to(self, shape: tuple[int, ...]) -> "RuntimeTensor":
        target_shape = tuple(int(dim) for dim in shape)
        target_stride = _broadcast_stride(self.shape, self.stride, target_shape)
        return RuntimeTensor(
            self._storage,
            target_shape,
            dtype=self.dtype,
            stride=target_stride,
            offset=self.offset,
        )

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, tuple) and any(item is None for item in index):
            from .frontend import slice_

            return slice_(self, index)
        normalized = _normalize_runtime_index(index, self.ndim)
        linear = self.offset + sum(item * step for item, step in zip(normalized, self.stride))
        return self._storage[linear]

    def __setitem__(self, index: Any, value: Any) -> None:
        normalized = _normalize_runtime_index(index, self.ndim)
        linear = self.offset + sum(item * step for item, step in zip(normalized, self.stride))
        self._storage[linear] = _unwrap_scalar_host_value(value)

    def load(self) -> "RuntimeTensor":
        return RuntimeTensor(
            [self[index] for index in _iter_indices(self.shape)],
            self.shape,
            dtype=self.dtype,
        )

    def store(self, value: Any) -> None:
        if isinstance(value, RuntimeTensor):
            if value.shape != self.shape:
                raise ValueError(f"tensor shapes must match, got {self.shape} and {value.shape}")
            for index in _iter_indices(self.shape):
                self[index] = value[index]
            return
        for index in _iter_indices(self.shape):
            self[index] = value

    def fill(self, value: Any) -> None:
        self.store(value)

    def reduce(
        self,
        op: ReductionOp | str,
        init_value: Any,
        *,
        reduction_profile: Any = 0,
    ) -> "RuntimeTensor | Any":
        reduction = _normalize_reduction_op(op)
        reduce_axes, keep_axes = _normalize_reduction_profile(reduction_profile, self.ndim)
        if not keep_axes:
            accumulator = init_value
            for index in _iter_indices(self.shape):
                accumulator = _apply_reduction(reduction, accumulator, self[index])
            return accumulator
        result_shape = tuple(self.shape[axis] for axis in keep_axes)
        result = full(result_shape, init_value, dtype=self.dtype)
        for index in _iter_indices(self.shape):
            out_index = tuple(index[axis] for axis in keep_axes)
            result[out_index] = _apply_reduction(reduction, result[out_index], self[index])
        return result

    def _binary_op(self, other: Any, op, *, reverse: bool = False, dtype: str | None = None) -> "RuntimeTensor":
        if isinstance(other, RuntimeTensor):
            result_shape = _broadcast_shape(self.shape, other.shape)
            lhs_tensor = self.broadcast_to(result_shape) if self.shape != result_shape else self
            rhs_tensor = other.broadcast_to(result_shape) if other.shape != result_shape else other
            flat = []
            for index in _iter_indices(result_shape):
                lhs = lhs_tensor[index]
                rhs = rhs_tensor[index]
                flat.append(op(rhs, lhs) if reverse else op(lhs, rhs))
            result_dtype = dtype or _binary_dtype(self.dtype, other.dtype)
            return RuntimeTensor(flat, result_shape, dtype=result_dtype)
        other_value = _unwrap_scalar_host_value(other)
        other_dtype = other.dtype if isinstance(other, RuntimeScalar) else type(other).__name__
        flat = []
        for index in _iter_indices(self.shape):
            lhs = self[index]
            flat.append(op(other_value, lhs) if reverse else op(lhs, other_value))
        result_dtype = dtype or _binary_dtype(self.dtype, other_dtype)
        return RuntimeTensor(flat, self.shape, dtype=result_dtype)

    def _compare_op(self, other: Any, op, *, reverse: bool = False) -> "RuntimeTensor":
        return self._binary_op(other, op, reverse=reverse, dtype="i1")

    def __add__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs + rhs)

    def __radd__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs + rhs, reverse=True)

    def __sub__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs - rhs)

    def __rsub__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs - rhs, reverse=True)

    def __mul__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs * rhs)

    def __rmul__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs * rhs, reverse=True)

    def __truediv__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs / rhs)

    def __rtruediv__(self, other: Any) -> "RuntimeTensor":
        return self._binary_op(other, lambda lhs, rhs: lhs / rhs, reverse=True)

    def __lt__(self, other: Any) -> "RuntimeTensor":
        return self._compare_op(other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other: Any) -> "RuntimeTensor":
        return self._compare_op(other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other: Any) -> "RuntimeTensor":
        return self._compare_op(other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other: Any) -> "RuntimeTensor":
        return self._compare_op(other, lambda lhs, rhs: lhs >= rhs)

    def __eq__(self, other: object) -> "RuntimeTensor":  # type: ignore[override]
        return self._compare_op(other, lambda lhs, rhs: lhs == rhs)

    def __ne__(self, other: object) -> "RuntimeTensor":  # type: ignore[override]
        return self._compare_op(other, lambda lhs, rhs: lhs != rhs)

    def __repr__(self) -> str:
        return f"RuntimeTensor(shape={self.shape}, dtype={self.dtype})"


Tensor = RuntimeTensor
Shape = tuple[int, ...]


@contextmanager
def execution_context(state: ExecutionState) -> Iterator[ExecutionState]:
    token = _current_execution.set(state)
    try:
        yield state
    finally:
        _current_execution.reset(token)


def current_execution() -> ExecutionState | None:
    return _current_execution.get()


def tensor(data: Any, *, dtype: str | None = None) -> RuntimeTensor:
    return RuntimeTensor.from_data(data, dtype=dtype)


def zeros(shape: tuple[int, ...], *, dtype: str = "f32") -> RuntimeTensor:
    return RuntimeTensor.zeros(shape, dtype=dtype)


def clone(value: RuntimeTensor) -> RuntimeTensor:
    return RuntimeTensor(list(value._storage), value.shape, dtype=value.dtype, stride=value.stride, offset=value.offset)


def full(shape: tuple[int, ...], fill_value: Any, *, dtype: str = "f32") -> RuntimeTensor:
    size = 1
    for dim in shape:
        size *= dim
    return RuntimeTensor([_unwrap_scalar_host_value(fill_value) for _ in range(size)], shape, dtype=dtype)


def from_dlpack(value: Any, assumed_align: int | None = None) -> TensorHandle:
    if not hasattr(value, "__dlpack__") or not hasattr(value, "__dlpack_device__"):
        raise TypeError("from_dlpack expects an object implementing __dlpack__ and __dlpack_device__")
    device_type, device_id = value.__dlpack_device__()
    capsule = value.__dlpack__()
    shape = tuple(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", "unknown"))
    stride_value = getattr(value, "stride", None)
    if callable(stride_value):
        stride_value = stride_value()
    stride = tuple(int(dim) for dim in stride_value) if stride_value is not None else None
    raw_address = int(value.data_ptr()) if hasattr(value, "data_ptr") else None
    return TensorHandle(
        capsule=capsule,
        shape=shape,
        dtype=dtype,
        device_type=device_type,
        device_id=device_id,
        source=value,
        stride=stride,
        assumed_align=assumed_align,
        raw_address=raw_address,
    )


def make_ptr(
    value_type: str,
    value: Any,
    address_space: Any = "global",
    *,
    assumed_align: int | None = None,
) -> Pointer:
    if isinstance(value, int):
        return Pointer(
            value_type=str(value_type),
            raw_address=value,
            address_space=address_space,
            assumed_align=assumed_align,
        )
    normalized = normalize_runtime_argument(value)
    if isinstance(normalized, TensorHandle):
        return Pointer(
            value_type=str(value_type),
            tensor=normalized,
            raw_address=normalized.data_ptr() or normalized.raw_address,
            address_space=address_space,
            assumed_align=assumed_align or normalized.assumed_align,
        )
    if not isinstance(normalized, RuntimeTensor):
        raise TypeError("make_ptr expects a RuntimeTensor, tensor handle, nested tensor data, or an integer address")
    return Pointer(
        value_type=str(value_type),
        tensor=normalized,
        address_space=address_space,
        assumed_align=assumed_align,
    )


def infer_runtime_spec(value: Any) -> tuple[str, tuple[int, ...] | None]:
    if isinstance(value, RuntimeScalar):
        return value.dtype, None
    if isinstance(value, RuntimeTensor):
        return value.dtype, value.shape
    if isinstance(value, TensorHandle):
        return value.dtype, value.shape
    if isinstance(value, Pointer) and isinstance(value.tensor, RuntimeTensor):
        return "python_object", None
    if _looks_like_nested_tensor(value):
        runtime_tensor = tensor(value)
        return runtime_tensor.dtype, runtime_tensor.shape
    if isinstance(value, bool):
        return "i1", None
    if isinstance(value, int):
        return "index", None
    if isinstance(value, float):
        return "f32", None
    return "python_object", None


def normalize_runtime_argument(value: Any) -> Any:
    if isinstance(value, RuntimeScalar):
        return value
    if isinstance(value, RuntimeTensor):
        return value
    if isinstance(value, TensorHandle):
        return value
    if isinstance(value, Pointer):
        if value.tensor is None:
            return value
        normalized_tensor = normalize_runtime_argument(value.tensor)
        return Pointer(
            value_type=value.value_type,
            tensor=normalized_tensor,
            raw_address=value.raw_address,
            address_space=value.address_space,
            assumed_align=value.assumed_align,
        )
    if hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__"):
        return from_dlpack(value)
    if _looks_like_nested_tensor(value):
        return tensor(value)
    if isinstance(value, list):
        return [normalize_runtime_argument(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_runtime_argument(item) for item in value)
    if isinstance(value, dict):
        return {key: normalize_runtime_argument(item) for key, item in value.items()}
    return value


def materialize_runtime_argument(value: Any) -> Any:
    if isinstance(value, TensorHandle):
        return value.to_runtime_tensor()
    if isinstance(value, Pointer):
        if value.tensor is None:
            return value
        return Pointer(
            value_type=value.value_type,
            tensor=materialize_runtime_argument(value.tensor),
            raw_address=value.raw_address,
            address_space=value.address_space,
            assumed_align=value.assumed_align,
        )
    if isinstance(value, list):
        return [materialize_runtime_argument(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_runtime_argument(item) for item in value)
    if isinstance(value, dict):
        return {key: materialize_runtime_argument(item) for key, item in value.items()}
    return value


def sync_runtime_argument(original: Any, materialized: Any) -> None:
    if isinstance(original, TensorHandle) and isinstance(materialized, RuntimeTensor):
        original.copy_from_runtime_tensor(materialized)
        return
    if isinstance(original, Pointer) and isinstance(materialized, Pointer):
        if original.tensor is not None and materialized.tensor is not None:
            sync_runtime_argument(original.tensor, materialized.tensor)
        return
    if isinstance(original, list) and isinstance(materialized, list):
        for original_item, materialized_item in zip(original, materialized):
            sync_runtime_argument(original_item, materialized_item)
        return
    if isinstance(original, tuple) and isinstance(materialized, tuple):
        for original_item, materialized_item in zip(original, materialized):
            sync_runtime_argument(original_item, materialized_item)
        return
    if isinstance(original, dict) and isinstance(materialized, dict):
        for key, original_item in original.items():
            if key in materialized:
                sync_runtime_argument(original_item, materialized[key])


@contextmanager
def materialized_runtime_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
    materialized_args = tuple(materialize_runtime_argument(arg) for arg in args)
    materialized_kwargs = {name: materialize_runtime_argument(value) for name, value in kwargs.items()}
    try:
        yield materialized_args, materialized_kwargs
    finally:
        for original, materialized in zip(args, materialized_args):
            sync_runtime_argument(original, materialized)
        for name, original in kwargs.items():
            if name in materialized_kwargs:
                sync_runtime_argument(original, materialized_kwargs[name])


def _normalize_dim3(value: tuple[int, int, int], *, name: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly 3 dimensions")
    normalized = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in normalized):
        raise ValueError(f"{name} dimensions must be > 0")
    return normalized


def _canonical_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return tuple(stride)


def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    rank = max(len(lhs), len(rhs))
    lhs_aligned = (1,) * (rank - len(lhs)) + lhs
    rhs_aligned = (1,) * (rank - len(rhs)) + rhs
    result: list[int] = []
    for lhs_dim, rhs_dim in zip(lhs_aligned, rhs_aligned):
        if lhs_dim == rhs_dim:
            result.append(lhs_dim)
            continue
        if lhs_dim == 1:
            result.append(rhs_dim)
            continue
        if rhs_dim == 1:
            result.append(lhs_dim)
            continue
        raise ValueError(f"tensor shapes are not broadcast-compatible: {lhs} and {rhs}")
    return tuple(result)


def _broadcast_stride(
    source_shape: tuple[int, ...],
    source_stride: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if len(target_shape) < len(source_shape):
        raise ValueError(f"cannot broadcast rank-{len(source_shape)} tensor to rank-{len(target_shape)} shape {target_shape}")
    padded_shape = (1,) * (len(target_shape) - len(source_shape)) + source_shape
    padded_stride = (0,) * (len(target_shape) - len(source_stride)) + source_stride
    result: list[int] = []
    for source_dim, target_dim, stride in zip(padded_shape, target_shape, padded_stride):
        if source_dim == target_dim:
            result.append(stride)
            continue
        if source_dim == 1 and target_dim >= 1:
            result.append(0)
            continue
        raise ValueError(f"cannot broadcast shape {source_shape} to {target_shape}")
    return tuple(result)


def _looks_like_nested_tensor(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    head = value[0]
    if isinstance(head, (RuntimeTensor, dict)):
        return False
    if isinstance(head, list):
        return all(_looks_like_nested_tensor(item) for item in value)
    return all(isinstance(item, (bool, int, float, RuntimeScalar)) for item in value)


def _infer_shape(data: Any) -> tuple[int, ...]:
    if not isinstance(data, (list, tuple)):
        return ()
    if not data:
        raise ValueError("tensor data must be a non-empty nested list")
    child_shape = _infer_shape(data[0])
    for item in data[1:]:
        if _infer_shape(item) != child_shape:
            raise ValueError("tensor data must have a regular rectangular shape")
    return (len(data),) + child_shape


def _flatten_nested(data: Any, flat: list[Any]) -> None:
    if isinstance(data, (list, tuple)):
        for item in data:
            _flatten_nested(item, flat)
        return
    flat.append(_unwrap_scalar_host_value(data))


def _infer_tensor_dtype(flat: Sequence[Any]) -> str:
    if not flat:
        return "unknown"
    if any(isinstance(item, RuntimeScalar) and item.dtype.startswith("f") for item in flat):
        return "f32"
    if any(isinstance(item, RuntimeScalar) and item.dtype == "i1" for item in flat):
        return "i1"
    if any(isinstance(item, RuntimeScalar) for item in flat):
        return next(item.dtype for item in flat if isinstance(item, RuntimeScalar))
    if any(isinstance(item, float) for item in flat):
        return "f32"
    if any(isinstance(item, bool) for item in flat):
        return "i1"
    if any(isinstance(item, int) for item in flat):
        return "index"
    return type(flat[0]).__name__


def _normalize_runtime_index(index: Any, rank: int) -> tuple[int, ...]:
    if isinstance(index, tuple):
        normalized = index
    else:
        normalized = (index,)
    if len(normalized) != rank:
        raise ValueError(f"expected {rank} indices, got {len(normalized)}")
    result: list[int] = []
    for item in normalized:
        if isinstance(item, RuntimeScalar):
            item = int(item)
        if not isinstance(item, int):
            raise TypeError("runtime tensor indices must be integers")
        result.append(item)
    return tuple(result)


def _normalize_reduction_op(value: ReductionOp | str) -> ReductionOp:
    if isinstance(value, ReductionOp):
        return value
    try:
        return ReductionOp(str(value))
    except ValueError as exc:
        supported = ", ".join(item.value for item in ReductionOp)
        raise ValueError(f"unsupported reduction op '{value}', expected one of: {supported}") from exc


def _normalize_reduction_profile(value: Any, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if value == 0:
        return tuple(range(rank)), ()
    if not isinstance(value, (tuple, list)):
        raise TypeError("reduction_profile must be 0 or a tuple/list describing kept axes with None")
    if len(value) != rank:
        raise ValueError(f"reduction_profile must have length {rank} for tensor rank {rank}")
    reduce_axes: list[int] = []
    keep_axes: list[int] = []
    for axis, item in enumerate(value):
        if item is None:
            keep_axes.append(axis)
        else:
            reduce_axes.append(axis)
    return tuple(reduce_axes), tuple(keep_axes)


def _apply_reduction(op: ReductionOp, lhs: Any, rhs: Any) -> Any:
    if op is ReductionOp.ADD:
        return lhs + rhs
    if op is ReductionOp.MUL:
        return lhs * rhs
    if op is ReductionOp.MAX:
        return lhs if lhs >= rhs else rhs
    if op is ReductionOp.MIN:
        return lhs if lhs <= rhs else rhs
    raise ValueError(f"unsupported reduction op '{op}'")


def _iter_indices(shape: tuple[int, ...]):
    if not shape:
        yield ()
        return
    from itertools import product

    yield from product(*(range(dim) for dim in shape))


def _binary_dtype(lhs: str, rhs: str) -> str:
    if lhs == "python_object":
        return normalize_dtype_name(rhs)
    if rhs == "python_object":
        return normalize_dtype_name(lhs)
    return promote_scalar_dtype(lhs, rhs)


def _coerce_runtime_scalar(value: Any, *, preferred_dtype: str | None = None) -> RuntimeScalar:
    if isinstance(value, RuntimeScalar):
        return value
    dtype = normalize_dtype_name(preferred_dtype or _infer_scalar_dtype(value))
    return RuntimeScalar(value, dtype)


def _unwrap_scalar_host_value(value: Any) -> Any:
    if isinstance(value, RuntimeScalar):
        return value.value
    return value


def _resolve_scalar_target_dtype(target: Any) -> str:
    if isinstance(target, RuntimeScalar):
        return target.dtype
    if isinstance(target, str):
        return normalize_dtype_name(target)
    dtype = getattr(target, "__baybridge_dtype__", None)
    if isinstance(dtype, str):
        return normalize_dtype_name(dtype)
    raise TypeError("scalar conversion target must be a baybridge scalar constructor or dtype string")


def _infer_scalar_dtype(value: Any) -> str:
    if isinstance(value, RuntimeScalar):
        return value.dtype
    if isinstance(value, bool):
        return "i1"
    if isinstance(value, int):
        return "index"
    if isinstance(value, float):
        return "f32"
    raise TypeError(f"unsupported baybridge scalar value type: {type(value).__name__}")


def make_scalar(value: Any, *, dtype: str) -> RuntimeScalar | Any:
    dtype = normalize_dtype_name(dtype)
    from .diagnostics import CompilationError
    from .tracing import ScalarValue, require_builder

    try:
        builder = require_builder()
    except CompilationError:
        return RuntimeScalar(_unwrap_scalar_host_value(value), dtype)
    if isinstance(value, ScalarValue):
        return value.to(dtype)
    if isinstance(value, RuntimeScalar):
        return builder.constant(value.value, dtype=dtype)
    return builder.constant(cast_scalar_value(value, dtype), dtype=dtype)


def _scalar_constructor(name: str, dtype: str):
    def constructor(value: Any = 0) -> RuntimeScalar | Any:
        return make_scalar(value, dtype=dtype)

    constructor.__name__ = name
    constructor.__baybridge_dtype__ = dtype
    return constructor


Boolean = _scalar_constructor("Boolean", "i1")
Int8 = _scalar_constructor("Int8", "i8")
Int32 = _scalar_constructor("Int32", "i32")
Int64 = _scalar_constructor("Int64", "i64")
Int = _scalar_constructor("Int", "index")
Float16 = _scalar_constructor("Float16", "f16")
BFloat16 = _scalar_constructor("BFloat16", "bf16")
Float32 = _scalar_constructor("Float32", "f32")


def _unflatten_nested(storage: list[Any], shape: tuple[int, ...], stride: tuple[int, ...], offset: int) -> Any:
    if not shape:
        return storage[offset]
    if len(shape) == 1:
        return [storage[offset + i * stride[0]] for i in range(shape[0])]
    head = shape[0]
    tail_shape = shape[1:]
    tail_stride = stride[1:]
    return [
        _unflatten_nested(storage, tail_shape, tail_stride, offset + i * stride[0])
        for i in range(head)
    ]
