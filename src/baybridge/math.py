from __future__ import annotations

import math as _math
from dataclasses import dataclass
from typing import Any, Callable

from .ir import AddressSpace, TensorSpec
from .runtime import RuntimeTensor
from .tracing import TensorValue, require_builder


@dataclass(frozen=True)
class _MathNamespace:
    def sqrt(self, value: Any):
        return _tensor_unary(value, "math_sqrt", _math.sqrt)

    def sin(self, value: Any):
        return _tensor_unary(value, "math_sin", _math.sin)

    def exp2(self, value: Any):
        return _tensor_unary(value, "math_exp2", _exp2)


math = _MathNamespace()


def _exp2(value: float) -> float:
    exp2 = getattr(_math, "exp2", None)
    if exp2 is not None:
        return exp2(value)
    return 2.0 ** value


def _tensor_unary(value: Any, op_name: str, fn: Callable[[float], float]):
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


def _iter_indices(shape: tuple[int, ...]):
    if not shape:
        yield ()
        return
    from itertools import product

    yield from product(*(range(dim) for dim in shape))
