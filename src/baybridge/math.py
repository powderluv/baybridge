from __future__ import annotations

import math as _math
from dataclasses import dataclass
from typing import Any, Callable

from .ir import AddressSpace, ScalarSpec, TensorSpec
from .runtime import RuntimeScalar, RuntimeTensor
from .tracing import ScalarValue, TensorValue, require_builder


@dataclass(frozen=True)
class _MathNamespace:
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


math = _MathNamespace()


def _exp2(value: float) -> float:
    exp2 = getattr(_math, "exp2", None)
    if exp2 is not None:
        return exp2(value)
    return 2.0 ** value


def _rsqrt(value: float) -> float:
    return 1.0 / _math.sqrt(value)


def _tensor_unary(value: Any, op_name: str, fn: Callable[[float], float]):
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
