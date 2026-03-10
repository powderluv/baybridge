from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Sequence

from .dtypes import element_type


@dataclass(frozen=True)
class LaunchConfig:
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    shared_mem_bytes: int = 0
    cooperative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", _normalize_dim3(self.grid, name="grid"))
        object.__setattr__(self, "block", _normalize_dim3(self.block, name="block"))
        if self.shared_mem_bytes < 0:
            raise ValueError("shared_mem_bytes must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        launch = {
            "grid": list(self.grid),
            "block": list(self.block),
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

    def data_ptr(self) -> "Pointer":
        return self


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
        self._storage[linear] = value

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
            if self.shape != other.shape:
                raise ValueError(f"tensor shapes must match, got {self.shape} and {other.shape}")
            flat = []
            for index in _iter_indices(self.shape):
                lhs = self[index]
                rhs = other[index]
                flat.append(op(rhs, lhs) if reverse else op(lhs, rhs))
            result_dtype = dtype or _binary_dtype(self.dtype, other.dtype)
            return RuntimeTensor(flat, self.shape, dtype=result_dtype)
        flat = []
        for index in _iter_indices(self.shape):
            lhs = self[index]
            flat.append(op(other, lhs) if reverse else op(lhs, other))
        result_dtype = dtype or _binary_dtype(self.dtype, type(other).__name__)
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
    return RuntimeTensor([fill_value for _ in range(size)], shape, dtype=dtype)


def from_dlpack(value: Any) -> TensorHandle:
    if not hasattr(value, "__dlpack__") or not hasattr(value, "__dlpack_device__"):
        raise TypeError("from_dlpack expects an object implementing __dlpack__ and __dlpack_device__")
    device_type, device_id = value.__dlpack_device__()
    capsule = value.__dlpack__()
    shape = tuple(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", "unknown"))
    return TensorHandle(
        capsule=capsule,
        shape=shape,
        dtype=dtype,
        device_type=device_type,
        device_id=device_id,
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
    if not isinstance(normalized, RuntimeTensor):
        raise TypeError("make_ptr expects a RuntimeTensor, nested tensor data, or an integer address")
    return Pointer(
        value_type=str(value_type),
        tensor=normalized,
        address_space=address_space,
        assumed_align=assumed_align,
    )


def infer_runtime_spec(value: Any) -> tuple[str, tuple[int, ...] | None]:
    if isinstance(value, RuntimeTensor):
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
    if isinstance(value, RuntimeTensor):
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
    if _looks_like_nested_tensor(value):
        return tensor(value)
    if isinstance(value, list):
        return [normalize_runtime_argument(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_runtime_argument(item) for item in value)
    if isinstance(value, dict):
        return {key: normalize_runtime_argument(item) for key, item in value.items()}
    return value


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


def _looks_like_nested_tensor(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    head = value[0]
    if isinstance(head, (RuntimeTensor, dict)):
        return False
    if isinstance(head, list):
        return all(_looks_like_nested_tensor(item) for item in value)
    return all(isinstance(item, (bool, int, float)) for item in value)


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
    flat.append(data)


def _infer_tensor_dtype(flat: Sequence[Any]) -> str:
    if not flat:
        return "unknown"
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
    if lhs == "i1" and rhs == "i1":
        return "i1"
    if lhs.startswith("f") or rhs.startswith("f") or rhs in {"float", "f32", "f16", "bf16"}:
        return "f32"
    return lhs if lhs != "python_object" else "index"


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
