from __future__ import annotations

import builtins as _builtins
import math as _math
from dataclasses import dataclass
from typing import Any, Callable

from .ir import AddressSpace, ScalarSpec, TensorSpec
from .runtime import RuntimeScalar, RuntimeTensor
from .tracing import ScalarValue, TensorValue, _coerce_scalar, require_builder


@dataclass(frozen=True)
class _MathNamespace:
    def round(self, value: Any):
        return _tensor_unary(value, "math_round", _round)

    def floor(self, value: Any):
        return _tensor_unary(value, "math_floor", _math.floor)

    def ceil(self, value: Any):
        return _tensor_unary(value, "math_ceil", _math.ceil)

    def trunc(self, value: Any):
        return _tensor_unary(value, "math_trunc", _math.trunc)

    def acos(self, value: Any):
        return _tensor_unary(value, "math_acos", _math.acos)

    def asin(self, value: Any):
        return _tensor_unary(value, "math_asin", _math.asin)

    def atan(self, value: Any):
        return _tensor_unary(value, "math_atan", _math.atan)

    def sqrt(self, value: Any):
        return _tensor_unary(value, "math_sqrt", _math.sqrt)

    def rsqrt(self, value: Any):
        return _tensor_unary(value, "math_rsqrt", _rsqrt)

    def exp(self, value: Any):
        return _tensor_unary(value, "math_exp", _math.exp)

    def sin(self, value: Any):
        return _tensor_unary(value, "math_sin", _math.sin)

    def cos(self, value: Any):
        return _tensor_unary(value, "math_cos", _math.cos)

    def exp2(self, value: Any):
        return _tensor_unary(value, "math_exp2", _exp2)

    def log(self, value: Any):
        return _tensor_unary(value, "math_log", _math.log)

    def log2(self, value: Any):
        return _tensor_unary(value, "math_log2", _math.log2)

    def log10(self, value: Any):
        return _tensor_unary(value, "math_log10", _math.log10)

    def erf(self, value: Any):
        return _tensor_unary(value, "math_erf", _math.erf)

    def atan2(self, lhs: Any, rhs: Any):
        return _tensor_binary(lhs, rhs, "math_atan2", _math.atan2)

    def maximum(self, lhs: Any, rhs: Any):
        return _tensor_extrema(lhs, rhs, "max", "tensor_max", _maximum)

    def minimum(self, lhs: Any, rhs: Any):
        return _tensor_extrema(lhs, rhs, "min", "tensor_min", _minimum)


math = _MathNamespace()


def acos(value: Any):
    return math.acos(value)


def round(value: Any):
    return math.round(value)


def floor(value: Any):
    return math.floor(value)


def ceil(value: Any):
    return math.ceil(value)


def trunc(value: Any):
    return math.trunc(value)


def asin(value: Any):
    return math.asin(value)


def atan(value: Any):
    return math.atan(value)


def sqrt(value: Any):
    return math.sqrt(value)


def rsqrt(value: Any):
    return math.rsqrt(value)


def exp(value: Any):
    return math.exp(value)


def sin(value: Any):
    return math.sin(value)


def cos(value: Any):
    return math.cos(value)


def exp2(value: Any):
    return math.exp2(value)


def log(value: Any):
    return math.log(value)


def log2(value: Any):
    return math.log2(value)


def log10(value: Any):
    return math.log10(value)


def erf(value: Any):
    return math.erf(value)


def atan2(lhs: Any, rhs: Any):
    return math.atan2(lhs, rhs)


def maximum(lhs: Any, rhs: Any):
    return math.maximum(lhs, rhs)


def minimum(lhs: Any, rhs: Any):
    return math.minimum(lhs, rhs)


def _exp2(value: float) -> float:
    exp2 = getattr(_math, "exp2", None)
    if exp2 is not None:
        return exp2(value)
    return 2.0 ** value


def _rsqrt(value: float) -> float:
    return 1.0 / _math.sqrt(value)


def _round(value: float) -> float:
    return float(_builtins.round(value))


def _maximum(lhs: bool | int | float, rhs: bool | int | float) -> bool | int | float:
    return lhs if lhs >= rhs else rhs


def _minimum(lhs: bool | int | float, rhs: bool | int | float) -> bool | int | float:
    return lhs if lhs <= rhs else rhs


def _tensor_unary(value: Any, op_name: str, fn: Callable[[float], float | int]):
    if isinstance(value, RuntimeScalar):
        if not value.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires a floating-point scalar")
        return RuntimeScalar(fn(float(value)), dtype=value.dtype)
    if isinstance(value, ScalarValue):
        if not value.spec.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires a floating-point scalar")
        return require_builder().emit_scalar(
            op_name,
            value,
            spec=ScalarSpec(dtype=value.spec.dtype),
            name_hint=op_name,
        )
    if isinstance(value, RuntimeTensor):
        if not value.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires a floating-point tensor")
        return RuntimeTensor([fn(float(value[index])) for index in _iter_indices(value.shape)], value.shape, dtype=value.dtype)
    if isinstance(value, TensorValue):
        if not value.spec.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires a floating-point tensor")
        return require_builder().emit_tensor(
            op_name,
            value,
            spec=TensorSpec(
                shape=value.spec.shape,
                dtype=value.spec.dtype,
                address_space=AddressSpace.REGISTER,
            ),
            name_hint=op_name,
        )
    return fn(float(value))


def _tensor_binary(lhs: Any, rhs: Any, op_name: str, fn: Callable[[float, float], float]):
    if isinstance(lhs, RuntimeScalar) and isinstance(rhs, RuntimeScalar):
        if not lhs.dtype.startswith("f") or not rhs.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires floating-point inputs")
        return RuntimeScalar(fn(float(lhs), float(rhs)), dtype=lhs.dtype)
    if isinstance(lhs, ScalarValue) and isinstance(rhs, ScalarValue):
        if not lhs.spec.dtype.startswith("f") or not rhs.spec.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires floating-point inputs")
        if lhs.spec.dtype != rhs.spec.dtype:
            raise TypeError(f"{op_name} requires matching scalar dtypes")
        return require_builder().emit_scalar(
            op_name,
            lhs,
            rhs,
            spec=ScalarSpec(dtype=lhs.spec.dtype),
            name_hint=op_name,
        )
    if isinstance(lhs, (RuntimeTensor, TensorValue)) or isinstance(rhs, (RuntimeTensor, TensorValue)):
        return _tensor_binary_tensor(lhs, rhs, op_name, fn)
    lhs_scalar = RuntimeScalar(lhs, dtype="f32") if not isinstance(lhs, RuntimeScalar) else lhs
    rhs_scalar = RuntimeScalar(rhs, dtype="f32") if not isinstance(rhs, RuntimeScalar) else rhs
    return RuntimeScalar(fn(float(lhs_scalar), float(rhs_scalar)), dtype=lhs_scalar.dtype)


def _tensor_binary_tensor(lhs: Any, rhs: Any, op_name: str, fn: Callable[[float, float], float]):
    if isinstance(lhs, RuntimeTensor) or isinstance(rhs, RuntimeTensor):
        lhs_tensor = lhs if isinstance(lhs, RuntimeTensor) else None
        rhs_tensor = rhs if isinstance(rhs, RuntimeTensor) else None
        if lhs_tensor is not None and rhs_tensor is not None:
            if not lhs_tensor.dtype.startswith("f") or not rhs_tensor.dtype.startswith("f"):
                raise TypeError(f"{op_name} requires floating-point tensors")
            result_shape = _broadcast_shape(lhs_tensor.shape, rhs_tensor.shape)
            lhs_view = lhs_tensor.broadcast_to(result_shape) if lhs_tensor.shape != result_shape else lhs_tensor
            rhs_view = rhs_tensor.broadcast_to(result_shape) if rhs_tensor.shape != result_shape else rhs_tensor
            return RuntimeTensor(
                [fn(float(lhs_view[index]), float(rhs_view[index])) for index in _iter_indices(result_shape)],
                result_shape,
                dtype=lhs_view.dtype,
            )
        tensor = lhs_tensor or rhs_tensor
        assert tensor is not None
        scalar = rhs if lhs_tensor is not None else lhs
        if not tensor.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires floating-point tensors")
        scalar_value = float(scalar.value if isinstance(scalar, RuntimeScalar) else scalar)
        if lhs_tensor is not None:
            return RuntimeTensor([fn(float(lhs_tensor[index]), scalar_value) for index in _iter_indices(lhs_tensor.shape)], lhs_tensor.shape, dtype=lhs_tensor.dtype)
        return RuntimeTensor([fn(scalar_value, float(rhs_tensor[index])) for index in _iter_indices(rhs_tensor.shape)], rhs_tensor.shape, dtype=rhs_tensor.dtype)  # type: ignore[union-attr]

    builder = require_builder()
    lhs_tensor = lhs if isinstance(lhs, TensorValue) else None
    rhs_tensor = rhs if isinstance(rhs, TensorValue) else None
    if lhs_tensor is not None and rhs_tensor is not None:
        if not lhs_tensor.spec.dtype.startswith("f") or not rhs_tensor.spec.dtype.startswith("f"):
            raise TypeError(f"{op_name} requires floating-point tensors")
        if lhs_tensor.spec.dtype != rhs_tensor.spec.dtype:
            raise TypeError(f"{op_name} requires matching tensor dtypes")
        result_shape = _broadcast_shape(lhs_tensor.spec.shape, rhs_tensor.spec.shape)
        lhs_value = lhs_tensor.broadcast_to(result_shape) if lhs_tensor.spec.shape != result_shape else lhs_tensor
        rhs_value = rhs_tensor.broadcast_to(result_shape) if rhs_tensor.spec.shape != result_shape else rhs_tensor
        return builder.emit_tensor(
            op_name,
            lhs_value,
            rhs_value,
            spec=TensorSpec(
                shape=result_shape,
                dtype=lhs_value.spec.dtype,
                address_space=AddressSpace.REGISTER,
            ),
            name_hint=op_name,
        )
    tensor = lhs_tensor or rhs_tensor
    assert tensor is not None
    scalar = rhs if lhs_tensor is not None else lhs
    scalar_value = scalar if isinstance(scalar, ScalarValue) else builder.constant(scalar, dtype=tensor.spec.dtype)
    if scalar_value.spec.dtype != tensor.spec.dtype:
        raise TypeError(f"{op_name} requires matching tensor/scalar dtypes")
    if lhs_tensor is not None:
        return builder.emit_tensor(
            op_name,
            tensor,
            scalar_value,
            spec=TensorSpec(shape=tensor.spec.shape, dtype=tensor.spec.dtype, address_space=AddressSpace.REGISTER),
            name_hint=op_name,
        )
    return builder.emit_tensor(
        op_name,
        scalar_value,
        tensor,
        spec=TensorSpec(shape=tensor.spec.shape, dtype=tensor.spec.dtype, address_space=AddressSpace.REGISTER),
        name_hint=op_name,
    )


def _tensor_extrema(
    lhs: Any,
    rhs: Any,
    scalar_op_name: str,
    tensor_op_name: str,
    fn: Callable[[bool | int | float, bool | int | float], bool | int | float],
):
    if isinstance(lhs, (RuntimeTensor, TensorValue)) or isinstance(rhs, (RuntimeTensor, TensorValue)):
        return _tensor_extrema_tensor(lhs, rhs, tensor_op_name, fn)
    if isinstance(lhs, ScalarValue) or isinstance(rhs, ScalarValue):
        lhs_scalar = lhs if isinstance(lhs, ScalarValue) else _coerce_scalar(lhs, preferred_dtype=rhs.spec.dtype if isinstance(rhs, ScalarValue) else None)
        rhs_scalar = rhs if isinstance(rhs, ScalarValue) else _coerce_scalar(rhs, preferred_dtype=lhs_scalar.spec.dtype)
        _require_extrema_dtype(lhs_scalar.spec.dtype, scalar_op_name, kind="scalar")
        if lhs_scalar.spec.dtype != rhs_scalar.spec.dtype:
            raise TypeError(f"{scalar_op_name} requires matching scalar dtypes")
        return require_builder().emit_scalar(
            scalar_op_name,
            lhs_scalar,
            rhs_scalar,
            spec=ScalarSpec(dtype=lhs_scalar.spec.dtype),
            name_hint=scalar_op_name,
        )
    if isinstance(lhs, RuntimeScalar) or isinstance(rhs, RuntimeScalar):
        lhs_dtype = lhs.dtype if isinstance(lhs, RuntimeScalar) else rhs.dtype if isinstance(rhs, RuntimeScalar) else "f32"
        rhs_dtype = rhs.dtype if isinstance(rhs, RuntimeScalar) else lhs_dtype
        lhs_scalar = lhs if isinstance(lhs, RuntimeScalar) else RuntimeScalar(lhs, dtype=lhs_dtype)
        rhs_scalar = rhs if isinstance(rhs, RuntimeScalar) else RuntimeScalar(rhs, dtype=rhs_dtype)
        _require_extrema_dtype(lhs_scalar.dtype, scalar_op_name, kind="scalar")
        if lhs_scalar.dtype != rhs_scalar.dtype:
            raise TypeError(f"{scalar_op_name} requires matching scalar dtypes")
        return RuntimeScalar(fn(lhs_scalar.value, rhs_scalar.value), dtype=lhs_scalar.dtype)
    return fn(lhs, rhs)


def _tensor_extrema_tensor(
    lhs: Any,
    rhs: Any,
    tensor_op_name: str,
    fn: Callable[[bool | int | float, bool | int | float], bool | int | float],
):
    if isinstance(lhs, RuntimeTensor) or isinstance(rhs, RuntimeTensor):
        lhs_tensor = lhs if isinstance(lhs, RuntimeTensor) else None
        rhs_tensor = rhs if isinstance(rhs, RuntimeTensor) else None
        if lhs_tensor is not None and rhs_tensor is not None:
            _require_extrema_dtype(lhs_tensor.dtype, tensor_op_name, kind="tensor")
            if lhs_tensor.dtype != rhs_tensor.dtype:
                raise TypeError(f"{tensor_op_name} requires matching tensor dtypes")
            result_shape = _broadcast_shape(lhs_tensor.shape, rhs_tensor.shape)
            lhs_view = lhs_tensor.broadcast_to(result_shape) if lhs_tensor.shape != result_shape else lhs_tensor
            rhs_view = rhs_tensor.broadcast_to(result_shape) if rhs_tensor.shape != result_shape else rhs_tensor
            return RuntimeTensor(
                [fn(lhs_view[index], rhs_view[index]) for index in _iter_indices(result_shape)],
                result_shape,
                dtype=lhs_view.dtype,
            )
        tensor = lhs_tensor or rhs_tensor
        assert tensor is not None
        _require_extrema_dtype(tensor.dtype, tensor_op_name, kind="tensor")
        scalar = rhs if lhs_tensor is not None else lhs
        scalar_value = scalar if isinstance(scalar, RuntimeScalar) else RuntimeScalar(scalar, dtype=tensor.dtype)
        if scalar_value.dtype != tensor.dtype:
            raise TypeError(f"{tensor_op_name} requires matching tensor/scalar dtypes")
        if lhs_tensor is not None:
            return RuntimeTensor(
                [fn(lhs_tensor[index], scalar_value.value) for index in _iter_indices(lhs_tensor.shape)],
                lhs_tensor.shape,
                dtype=lhs_tensor.dtype,
            )
        return RuntimeTensor(
            [fn(scalar_value.value, rhs_tensor[index]) for index in _iter_indices(rhs_tensor.shape)],
            rhs_tensor.shape,
            dtype=rhs_tensor.dtype,
        )  # type: ignore[union-attr]

    builder = require_builder()
    lhs_tensor = lhs if isinstance(lhs, TensorValue) else None
    rhs_tensor = rhs if isinstance(rhs, TensorValue) else None
    if lhs_tensor is not None and rhs_tensor is not None:
        _require_extrema_dtype(lhs_tensor.spec.dtype, tensor_op_name, kind="tensor")
        if lhs_tensor.spec.dtype != rhs_tensor.spec.dtype:
            raise TypeError(f"{tensor_op_name} requires matching tensor dtypes")
        result_shape = _broadcast_shape(lhs_tensor.spec.shape, rhs_tensor.spec.shape)
        lhs_value = lhs_tensor.broadcast_to(result_shape) if lhs_tensor.spec.shape != result_shape else lhs_tensor
        rhs_value = rhs_tensor.broadcast_to(result_shape) if rhs_tensor.spec.shape != result_shape else rhs_tensor
        return builder.emit_tensor(
            tensor_op_name,
            lhs_value,
            rhs_value,
            spec=TensorSpec(
                shape=result_shape,
                dtype=lhs_value.spec.dtype,
                address_space=AddressSpace.REGISTER,
            ),
            name_hint=tensor_op_name,
        )
    tensor = lhs_tensor or rhs_tensor
    assert tensor is not None
    _require_extrema_dtype(tensor.spec.dtype, tensor_op_name, kind="tensor")
    scalar = rhs if lhs_tensor is not None else lhs
    scalar_value = scalar if isinstance(scalar, ScalarValue) else builder.constant(scalar, dtype=tensor.spec.dtype)
    if scalar_value.spec.dtype != tensor.spec.dtype:
        raise TypeError(f"{tensor_op_name} requires matching tensor/scalar dtypes")
    if lhs_tensor is not None:
        return builder.emit_tensor(
            tensor_op_name,
            lhs_tensor,
            scalar_value,
            spec=TensorSpec(shape=lhs_tensor.spec.shape, dtype=lhs_tensor.spec.dtype, address_space=AddressSpace.REGISTER),
            name_hint=tensor_op_name,
        )
    return builder.emit_tensor(
        tensor_op_name,
        scalar_value,
        rhs_tensor,
        spec=TensorSpec(shape=rhs_tensor.spec.shape, dtype=rhs_tensor.spec.dtype, address_space=AddressSpace.REGISTER),
        name_hint=tensor_op_name,
    )


def _require_extrema_dtype(dtype: str, op_name: str, *, kind: str) -> None:
    if dtype not in {"f32", "i32"}:
        raise TypeError(f"{op_name} requires {kind} dtype f32 or i32")


def _iter_indices(shape: tuple[int, ...]):
    if not shape:
        yield ()
        return
    from itertools import product

    yield from product(*(range(dim) for dim in shape))


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
