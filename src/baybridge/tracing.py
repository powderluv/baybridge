from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from math import prod
from typing import Iterator

from .dtypes import element_type, is_float_dtype, normalize_dtype_name, promote_scalar_dtype
from .diagnostics import CompilationError
from .ir import AddressSpace, KernelArgument, Operation, PortableKernelIR, ScalarSpec, TensorSpec
from .runtime import LaunchConfig, ReductionOp, RuntimeScalar, _normalize_reduction_op, _normalize_reduction_profile

_current_builder: ContextVar["IRBuilder | None"] = ContextVar("baybridge_builder", default=None)


@dataclass(frozen=True)
class ScalarValue:
    name: str
    spec: ScalarSpec

    def __bool__(self) -> bool:
        raise CompilationError(
            "dynamic Python control flow on traced baybridge scalars is not supported; use predication or runtime execution"
        )

    def _binary(self, op: str, other: "ScalarValue | RuntimeScalar | bool | int | float", *, reverse: bool = False) -> "ScalarValue":
        builder = require_builder()
        rhs = _coerce_scalar(other)
        lhs = rhs if reverse else self
        rhs = self if reverse else rhs
        result_dtype = _binary_dtype(lhs.spec.dtype, rhs.spec.dtype)
        return builder.emit_scalar(
            op,
            lhs,
            rhs,
            spec=ScalarSpec(dtype=result_dtype),
            name_hint=op,
        )

    def __add__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._binary("add", other)

    def __radd__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._binary("add", other, reverse=True)

    def __sub__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._binary("sub", other)

    def __rsub__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._binary("sub", other, reverse=True)

    def __mul__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._binary("mul", other)

    def __rmul__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._binary("mul", other, reverse=True)

    def __truediv__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        builder = require_builder()
        rhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        result_dtype = _binary_dtype(self.spec.dtype, rhs.spec.dtype)
        return builder.emit_scalar(
            "div",
            self,
            rhs,
            spec=ScalarSpec(dtype=result_dtype),
            name_hint="div",
        )

    def __rtruediv__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        builder = require_builder()
        lhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        result_dtype = _binary_dtype(lhs.spec.dtype, self.spec.dtype)
        return builder.emit_scalar(
            "div",
            lhs,
            self,
            spec=ScalarSpec(dtype=result_dtype),
            name_hint="div",
        )

    def __floordiv__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        builder = require_builder()
        rhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        if is_float_dtype(self.spec.dtype) or is_float_dtype(rhs.spec.dtype):
            raise TypeError("floordiv is only supported for integer/index scalars")
        return builder.emit_scalar(
            "floordiv",
            self,
            rhs,
            spec=ScalarSpec(dtype="index"),
            name_hint="floordiv",
        )

    def __rfloordiv__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        builder = require_builder()
        lhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        if is_float_dtype(self.spec.dtype) or is_float_dtype(lhs.spec.dtype):
            raise TypeError("floordiv is only supported for integer/index scalars")
        return builder.emit_scalar(
            "floordiv",
            lhs,
            self,
            spec=ScalarSpec(dtype="index"),
            name_hint="floordiv",
        )

    def __mod__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        builder = require_builder()
        rhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        if is_float_dtype(self.spec.dtype) or is_float_dtype(rhs.spec.dtype):
            raise TypeError("mod is only supported for integer/index scalars")
        return builder.emit_scalar(
            "mod",
            self,
            rhs,
            spec=ScalarSpec(dtype="index"),
            name_hint="mod",
        )

    def __rmod__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        builder = require_builder()
        lhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        if is_float_dtype(self.spec.dtype) or is_float_dtype(lhs.spec.dtype):
            raise TypeError("mod is only supported for integer/index scalars")
        return builder.emit_scalar(
            "mod",
            lhs,
            self,
            spec=ScalarSpec(dtype="index"),
            name_hint="mod",
        )

    def _compare(
        self,
        op: str,
        other: "ScalarValue | RuntimeScalar | bool | int | float",
        *,
        reverse: bool = False,
    ) -> "ScalarValue":
        builder = require_builder()
        rhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        lhs = rhs if reverse else self
        rhs = self if reverse else rhs
        return builder.emit_scalar(
            op,
            lhs,
            rhs,
            spec=ScalarSpec(dtype="i1"),
            name_hint=op,
        )

    def __lt__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._compare("cmp_lt", other)

    def __le__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._compare("cmp_le", other)

    def __gt__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._compare("cmp_gt", other)

    def __ge__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._compare("cmp_ge", other)

    def __eq__(self, other: object) -> "ScalarValue":  # type: ignore[override]
        if not isinstance(other, (ScalarValue, RuntimeScalar, bool, int, float)):
            raise TypeError("comparisons are only supported against baybridge scalar values")
        return self._compare("cmp_eq", other)

    def __ne__(self, other: object) -> "ScalarValue":  # type: ignore[override]
        if not isinstance(other, (ScalarValue, RuntimeScalar, bool, int, float)):
            raise TypeError("comparisons are only supported against baybridge scalar values")
        return self._compare("cmp_ne", other)

    def _bitwise(self, op: str, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        rhs = _coerce_scalar(other, preferred_dtype=self.spec.dtype)
        if self.spec.dtype == "i1" and rhs.spec.dtype == "i1":
            return require_builder().emit_scalar(
                op,
                self,
                rhs,
                spec=ScalarSpec(dtype="i1"),
                name_hint=op,
            )
        if is_float_dtype(self.spec.dtype) or is_float_dtype(rhs.spec.dtype):
            raise TypeError(f"{op} requires integer baybridge scalars")
        result_dtype = _binary_dtype(self.spec.dtype, rhs.spec.dtype)
        return require_builder().emit_scalar(
            "bitand" if op == "and" else "bitor",
            self,
            rhs,
            spec=ScalarSpec(dtype=result_dtype),
            name_hint="bit" + op,
        )

    def __and__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._bitwise("and", other)

    def __or__(self, other: "ScalarValue | RuntimeScalar | bool | int | float") -> "ScalarValue":
        return self._bitwise("or", other)

    def __neg__(self) -> "ScalarValue":
        return require_builder().emit_scalar(
            "neg",
            self,
            spec=ScalarSpec(dtype=self.spec.dtype),
            name_hint="neg",
        )

    def __invert__(self) -> "ScalarValue":
        if is_float_dtype(self.spec.dtype):
            raise TypeError("bitwise invert is only supported for integer baybridge scalars")
        return require_builder().emit_scalar(
            "bitnot",
            self,
            spec=ScalarSpec(dtype=self.spec.dtype),
            name_hint="bitnot",
        )

    def to(self, target: object) -> "ScalarValue":
        target_dtype = _resolve_scalar_target_dtype(target)
        if self.spec.dtype == target_dtype:
            return self
        return require_builder().emit_scalar(
            "cast",
            self,
            spec=ScalarSpec(dtype=target_dtype),
            attrs={"source_dtype": self.spec.dtype, "target_dtype": target_dtype},
            name_hint="cast",
        )


@dataclass(frozen=True)
class TensorValue:
    name: str
    spec: TensorSpec

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def ndim(self) -> int:
        return len(self.spec.shape)

    @property
    def element_type(self) -> str:
        return element_type(self.spec.dtype)

    @property
    def type(self) -> str:
        return f"Tensor(shape={self.spec.shape}, dtype={self.spec.dtype})"

    @property
    def layout(self):
        return self.spec.resolved_layout()

    def _register_tensor(self, *, name_hint: str) -> "TensorValue":
        builder = require_builder()
        builder._temp_index += 1
        return builder.make_tensor(
            f"{name_hint}_{builder._temp_index}",
            TensorSpec(
                shape=self.spec.shape,
                dtype=self.spec.dtype,
                address_space=AddressSpace.REGISTER,
            ),
        )

    def _tensor_binary(self, op: str, other: "TensorValue") -> "TensorValue":
        if not isinstance(other, TensorValue):
            raise TypeError(f"{op} requires baybridge tensor operands")
        if self.spec.shape != other.spec.shape:
            raise ValueError(f"tensor shapes must match, got {self.spec.shape} and {other.spec.shape}")
        if self.spec.dtype != other.spec.dtype:
            raise ValueError(f"tensor dtypes must match, got {self.spec.dtype} and {other.spec.dtype}")
        return require_builder().emit_tensor(
            op,
            self,
            other,
            spec=TensorSpec(
                shape=self.spec.shape,
                dtype=self.spec.dtype,
                address_space=AddressSpace.REGISTER,
            ),
            name_hint=op,
        )

    def __getitem__(self, index: ScalarValue | int | tuple[ScalarValue | int, ...]) -> ScalarValue:
        if isinstance(index, tuple) and any(item is None for item in index):
            from .frontend import slice_

            return slice_(self, index)
        from .frontend import load

        return load(self, index)

    def __setitem__(self, index: ScalarValue | int | tuple[ScalarValue | int, ...], value: ScalarValue) -> None:
        from .frontend import store

        store(value, self, index)

    def load(self) -> "TensorValue":
        loaded = self._register_tensor(name_hint=f"{self.name}_load")
        require_builder().emit_void("copy", self, loaded)
        return loaded

    def store(self, value: "TensorValue | ScalarValue") -> None:
        if not isinstance(value, TensorValue):
            raise CompilationError("whole-tensor store requires a baybridge tensor value while tracing")
        if self.spec.shape != value.spec.shape:
            raise ValueError(f"tensor shapes must match, got {self.spec.shape} and {value.spec.shape}")
        if self.spec.dtype != value.spec.dtype:
            raise ValueError(f"tensor dtypes must match, got {self.spec.dtype} and {value.spec.dtype}")
        require_builder().emit_void("copy", value, self)

    def fill(self, value: "ScalarValue | int | float") -> None:
        builder = require_builder()
        if isinstance(value, ScalarValue):
            fill_value = value
        else:
            fill_value = builder.constant(value, dtype=self.spec.dtype)
        if fill_value.spec.dtype != self.spec.dtype:
            raise ValueError(f"fill expects dtype {self.spec.dtype}, got {fill_value.spec.dtype}")
        builder.emit_void("fill", self, fill_value)

    def reduce(
        self,
        op: ReductionOp | str,
        init_value: "ScalarValue | int | float",
        *,
        reduction_profile: object = 0,
    ) -> "TensorValue | ScalarValue":
        reduction = _normalize_reduction_op(op)
        reduce_axes, keep_axes = _normalize_reduction_profile(reduction_profile, self.ndim)
        del reduce_axes
        builder = require_builder()
        if isinstance(init_value, ScalarValue):
            init_scalar = init_value
        else:
            init_scalar = builder.constant(init_value, dtype=self.spec.dtype)
        if init_scalar.spec.dtype != self.spec.dtype:
            raise ValueError(f"reduce expects init value dtype {self.spec.dtype}, got {init_scalar.spec.dtype}")
        op_name = f"reduce_{reduction.value}"
        if not keep_axes:
            return builder.emit_scalar(
                op_name,
                self,
                init_scalar,
                spec=ScalarSpec(dtype=self.spec.dtype),
                attrs={"reduction_profile": list(reduction_profile) if isinstance(reduction_profile, (tuple, list)) else reduction_profile},
                name_hint=op_name,
            )
        return builder.emit_tensor(
            op_name,
            self,
            init_scalar,
            spec=TensorSpec(
                shape=tuple(self.spec.shape[axis] for axis in keep_axes),
                dtype=self.spec.dtype,
                address_space=AddressSpace.REGISTER,
            ),
            attrs={"reduction_profile": list(reduction_profile) if isinstance(reduction_profile, (tuple, list)) else reduction_profile},
            name_hint=op_name,
        )

    def __add__(self, other: "TensorValue") -> "TensorValue":
        return self._tensor_binary("tensor_add", other)

    def __sub__(self, other: "TensorValue") -> "TensorValue":
        return self._tensor_binary("tensor_sub", other)

    def __mul__(self, other: "TensorValue") -> "TensorValue":
        return self._tensor_binary("tensor_mul", other)


class IRBuilder:
    def __init__(self, kernel_name: str, *, launch: LaunchConfig | None = None):
        self.kernel_name = kernel_name
        self.launch = launch or LaunchConfig()
        self.arguments: list[KernelArgument] = []
        self.operations: list[Operation] = []
        self.metadata: dict[str, object] = {
            "dialect": "baybridge.portable",
            "stage": "portable_ir",
            "launch": self.launch.to_dict(),
        }
        self._temp_index = 0
        self._dynamic_shared_mem_bytes = 0

    def bind_argument(self, name: str, spec: TensorSpec | ScalarSpec) -> TensorValue | ScalarValue:
        kind = "tensor" if isinstance(spec, TensorSpec) else "scalar"
        self.arguments.append(KernelArgument(name=name, kind=kind, spec=spec))
        if isinstance(spec, TensorSpec):
            return TensorValue(name=name, spec=spec)
        return ScalarValue(name=name, spec=spec)

    def make_tensor(
        self,
        name: str,
        spec: TensorSpec,
        *,
        dynamic_shared: bool = False,
        byte_alignment: int | None = None,
        byte_offset: int | None = None,
    ) -> TensorValue:
        attrs = {
            "shape": list(spec.shape),
            "dtype": spec.dtype,
            "address_space": spec.address_space.value,
            "layout": spec.resolved_layout().to_dict(),
        }
        if dynamic_shared:
            if spec.address_space is not AddressSpace.SHARED:
                raise ValueError("dynamic shared tensors must use baybridge.AddressSpace.SHARED")
            alignment = max(1, int(byte_alignment or self._dtype_size(spec.dtype)))
            size_bytes = prod(spec.shape) * self._dtype_size(spec.dtype)
            offset = self.reserve_dynamic_shared(
                size_bytes,
                byte_alignment=alignment,
                byte_offset=byte_offset,
            )
            attrs["dynamic_shared"] = True
            attrs["byte_offset"] = offset
            attrs["byte_alignment"] = alignment
        self.operations.append(
            Operation(
                op="make_tensor",
                outputs=(name,),
                attrs=attrs,
            )
        )
        return TensorValue(name=name, spec=spec)

    def reserve_dynamic_shared(
        self,
        size_bytes: int,
        *,
        byte_alignment: int | None = None,
        byte_offset: int | None = None,
    ) -> int:
        if size_bytes < 0:
            raise ValueError("dynamic shared reservation size must be >= 0")
        alignment = max(1, int(byte_alignment or 1))
        if byte_offset is None:
            offset = self._align_up(self._dynamic_shared_mem_bytes, alignment)
        else:
            offset = int(byte_offset)
            if offset < 0:
                raise ValueError("dynamic shared reservation offset must be >= 0")
            if offset % alignment != 0:
                raise ValueError("dynamic shared reservation offset must respect the requested alignment")
        self._dynamic_shared_mem_bytes = max(self._dynamic_shared_mem_bytes, offset + size_bytes)
        return offset

    def emit_scalar(
        self,
        op: str,
        *inputs: TensorValue | ScalarValue,
        spec: ScalarSpec,
        attrs: dict[str, object] | None = None,
        name_hint: str = "scalar",
    ) -> ScalarValue:
        self._temp_index += 1
        name = f"{name_hint}_{self._temp_index}"
        merged_attrs = {"result": {"kind": "scalar", **spec.to_dict()}}
        if attrs:
            merged_attrs.update(attrs)
        self.operations.append(
            Operation(
                op=op,
                inputs=tuple(value.name for value in inputs),
                outputs=(name,),
                attrs=merged_attrs,
            )
        )
        return ScalarValue(name=name, spec=spec)

    def constant(self, value: RuntimeScalar | bool | int | float, *, dtype: str | None = None) -> ScalarValue:
        if isinstance(value, RuntimeScalar):
            if dtype is None:
                dtype = value.dtype
            value = value.value
        if dtype is None:
            if isinstance(value, bool):
                dtype = "i1"
            elif isinstance(value, int):
                dtype = "index"
            elif isinstance(value, float):
                dtype = "f32"
            else:
                raise TypeError(f"unsupported scalar literal type: {type(value).__name__}")
        return self.emit_scalar(
            "constant",
            spec=ScalarSpec(dtype=dtype),
            attrs={"value": value},
            name_hint="cst",
        )

    def emit_void(self, op: str, *inputs: TensorValue | ScalarValue, attrs: dict[str, object] | None = None) -> None:
        self.operations.append(
            Operation(
                op=op,
                inputs=tuple(value.name for value in inputs),
                attrs=dict(attrs or {}),
            )
        )

    def emit_tensor(
        self,
        op: str,
        *inputs: TensorValue | ScalarValue,
        spec: TensorSpec,
        attrs: dict[str, object] | None = None,
        name_hint: str = "tmp",
    ) -> TensorValue:
        self._temp_index += 1
        name = f"{name_hint}_{self._temp_index}"
        merged_attrs = {"result": {"kind": "tensor", **spec.to_dict()}}
        if attrs:
            merged_attrs.update(attrs)
        self.operations.append(
            Operation(
                op=op,
                inputs=tuple(value.name for value in inputs),
                outputs=(name,),
                attrs=merged_attrs,
            )
        )
        return TensorValue(name=name, spec=spec)

    def finalize(self) -> PortableKernelIR:
        launch = self.launch
        if self._dynamic_shared_mem_bytes > launch.shared_mem_bytes:
            launch = replace(launch, shared_mem_bytes=self._dynamic_shared_mem_bytes)
        metadata = dict(self.metadata)
        metadata["launch"] = launch.to_dict()
        if self._dynamic_shared_mem_bytes:
            metadata["inferred_shared_mem_bytes"] = self._dynamic_shared_mem_bytes
        return PortableKernelIR(
            name=self.kernel_name,
            arguments=tuple(self.arguments),
            operations=tuple(self.operations),
            launch=launch,
            metadata=metadata,
        )

    def _dtype_size(self, dtype: str) -> int:
        return (element_type(dtype).width + 7) // 8

    def _align_up(self, value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment


@contextmanager
def tracing(builder: IRBuilder) -> Iterator[IRBuilder]:
    token = _current_builder.set(builder)
    try:
        yield builder
    finally:
        _current_builder.reset(token)


def require_builder() -> IRBuilder:
    builder = _current_builder.get()
    if builder is None:
        raise CompilationError("baybridge operations can only be used while tracing a @kernel or @jit function")
    return builder


def _coerce_scalar(
    value: ScalarValue | RuntimeScalar | bool | int | float,
    preferred_dtype: str | None = None,
) -> ScalarValue:
    if isinstance(value, ScalarValue):
        return value
    if isinstance(value, RuntimeScalar):
        return require_builder().constant(value.value, dtype=preferred_dtype or value.dtype)
    return require_builder().constant(value, dtype=preferred_dtype)


def _binary_dtype(lhs: str, rhs: str) -> str:
    return promote_scalar_dtype(lhs, rhs)


def _resolve_scalar_target_dtype(target: object) -> str:
    if isinstance(target, RuntimeScalar):
        return target.dtype
    if isinstance(target, str):
        return normalize_dtype_name(target)
    dtype = getattr(target, "__baybridge_dtype__", None)
    if isinstance(dtype, str):
        return normalize_dtype_name(dtype)
    raise TypeError("scalar conversion target must be a baybridge scalar constructor or dtype string")
