from __future__ import annotations

import builtins
from itertools import product
import math as py_math
from math import prod
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .dtypes import element_type, is_storage_only_dtype, resolve_element_type_name
from .diagnostics import CompilationError, UnsupportedOperationError
from .ir import AddressSpace, Layout, ScalarSpec, TensorSpec, normalize_address_space
from .mfma import MFMADescriptor, resolve_mfma_descriptor
from .runtime import (
    ExecutionState,
    LaunchConfig,
    Pointer,
    ReductionOp,
    RuntimeScalar,
    TensorHandle,
    RuntimeTensor,
    current_execution,
    execution_context,
    full as runtime_full,
    materialized_runtime_call,
    normalize_runtime_argument,
    zeros,
)
from .structs import MemRangeProxy, StructInstance, get_struct_spec, is_struct_type, struct
from .tracing import ScalarValue, TensorValue, require_builder
from .views import (
    IdentityTensor,
    LocalCoordinateTensor,
    LocalCoordinateTensorValue,
    IdentityTileTensor,
    IdentityTileTensorValue,
    ThreadFragmentTensorValue,
    ThreadFragmentView,
    ThreadValueComposedCoordinateView,
    ThreadValueComposedTensorView,
    ThreadValueComposedView,
    ThreadValueLayout,
    TiledTensorView,
)


def _execution(state: ExecutionState):
    return execution_context(state)


@dataclass(frozen=True)
class CopyUniversalOp:
    pass


@dataclass(frozen=True)
class MmaUniversalOp:
    accumulator_dtype: str
    wave_size: int = 64
    tile: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "accumulator_dtype", resolve_element_type_name(self.accumulator_dtype))


@dataclass(frozen=True)
class CpAsyncCopyG2SOp:
    cache_mode: str | None = None


@dataclass(frozen=True)
class CpAsyncCopyBulkTensorTileG2SOp:
    cta_group: Any = None


class _CpAsyncReductionOp:
    ADD = "add"
    INC = "inc"
    DEC = "dec"
    MIN = "min"
    MAX = "max"
    AND = "and"
    OR = "or"
    XOR = "xor"


def _normalize_cpasync_reduction_kind(value: Any) -> str:
    normalized = getattr(value, "value", value)
    text = str(normalized).lower()
    supported = {
        _CpAsyncReductionOp.ADD,
        _CpAsyncReductionOp.INC,
        _CpAsyncReductionOp.DEC,
        _CpAsyncReductionOp.MIN,
        _CpAsyncReductionOp.MAX,
        _CpAsyncReductionOp.AND,
        _CpAsyncReductionOp.OR,
        _CpAsyncReductionOp.XOR,
    }
    if text not in supported:
        raise ValueError(f"unsupported cpasync reduction op '{value}'")
    return text


@dataclass(frozen=True)
class CpAsyncCopyBulkTensorTileG2SMulticastOp:
    cta_group: Any = None


@dataclass(frozen=True)
class CpAsyncCopyBulkTensorTileS2GOp:
    cta_group: Any = None


@dataclass(frozen=True)
class CpAsyncCopyReduceBulkTensorTileS2GOp:
    reduction_kind: Any = _CpAsyncReductionOp.ADD
    cta_group: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reduction_kind", _normalize_cpasync_reduction_kind(self.reduction_kind))


@dataclass(frozen=True)
class CpAsyncCopyDsmemStoreOp:
    pass


@dataclass(frozen=True)
class Tcgen05SmemDesc:
    src: Pointer
    layout: Layout
    major: str
    next_src: Pointer | None = None
    swizzle: str = "SWIZZLE_NONE"


@dataclass(frozen=True)
class WarpLdMatrix8x8x16bOp:
    transpose: bool = False
    num_matrices: int = 1

    def __post_init__(self) -> None:
        if self.num_matrices <= 0:
            raise ValueError("LdMatrix8x8x16bOp requires num_matrices > 0")


@dataclass(frozen=True)
class WarpMmaF16BF16Op:
    dtype: Any
    accumulator_dtype: Any
    tile: tuple[int, int, int]
    wave_size: int = 64
    lane_shape: tuple[int, int] = (4, 8)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", resolve_element_type_name(self.dtype))
        object.__setattr__(self, "accumulator_dtype", resolve_element_type_name(self.accumulator_dtype))

    @property
    def operand_dtype(self) -> str:
        return str(self.dtype)


@dataclass(frozen=True)
class WarpgroupMmaF16BF16Op:
    dtype: Any
    accumulator_dtype: Any
    tile: tuple[int, int, int]
    wave_size: int = 128
    lane_shape: tuple[int, int] = (4, 8)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", resolve_element_type_name(self.dtype))
        object.__setattr__(self, "accumulator_dtype", resolve_element_type_name(self.accumulator_dtype))

    @property
    def operand_dtype(self) -> str:
        return str(self.dtype)


@dataclass(frozen=True)
class WarpgroupMmaTF32Op:
    accumulator_dtype: Any
    tile: tuple[int, int, int]
    wave_size: int = 128
    lane_shape: tuple[int, int] = (4, 8)
    dtype: str = "f32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "accumulator_dtype", resolve_element_type_name(self.accumulator_dtype))

    @property
    def operand_dtype(self) -> str:
        return str(self.dtype)


@dataclass(frozen=True)
class Tcgen05Ld32x32bOp:
    repetition: Any = None
    pack: Any = None


@dataclass(frozen=True)
class Tcgen05Ld16x128bOp:
    repetition: Any = None
    pack: Any = None


@dataclass(frozen=True)
class Tcgen05Ld16x64bOp:
    repetition: Any = None
    pack: Any = None


@dataclass(frozen=True)
class Tcgen05Ld16x256bOp:
    repetition: Any = None
    pack: Any = None


@dataclass(frozen=True)
class Tcgen05Ld16x32bx2Op:
    repetition: Any = None
    pack: Any = None


@dataclass(frozen=True)
class Tcgen05St16x64bOp:
    repetition: Any = None
    unpack: Any = None


@dataclass(frozen=True)
class Tcgen05St16x128bOp:
    repetition: Any = None
    unpack: Any = None


@dataclass(frozen=True)
class Tcgen05St16x256bOp:
    repetition: Any = None
    unpack: Any = None


@dataclass(frozen=True)
class Tcgen05St16x32bx2Op:
    repetition: Any = None
    unpack: Any = None


@dataclass(frozen=True)
class Tcgen05St32x32bOp:
    repetition: Any = None
    unpack: Any = None


@dataclass(frozen=True)
class Tcgen05MmaF16BF16Op:
    dtype: Any
    accumulator_dtype: Any
    tile: tuple[int, int, int]
    cta_group: Any = None
    operand_source: Any = None
    operand_major_mode_a: Any = None
    operand_major_mode_b: Any = None
    wave_size: int = 64
    lane_shape: tuple[int, int] = (4, 8)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", resolve_element_type_name(self.dtype))
        object.__setattr__(self, "accumulator_dtype", resolve_element_type_name(self.accumulator_dtype))

    @property
    def operand_dtype(self) -> str:
        return str(self.dtype)


@dataclass(frozen=True)
class Tcgen05MmaTF32Op:
    accumulator_dtype: Any
    tile: tuple[int, int, int]
    cta_group: Any = None
    operand_source: Any = None
    operand_major_mode_a: Any = None
    operand_major_mode_b: Any = None
    wave_size: int = 64
    lane_shape: tuple[int, int] = (4, 8)

    def __post_init__(self) -> None:
        object.__setattr__(self, "accumulator_dtype", resolve_element_type_name(self.accumulator_dtype))

    @property
    def dtype(self) -> str:
        return "f32"

    @property
    def operand_dtype(self) -> str:
        return "f32"


@dataclass(frozen=True)
class Tcgen05MmaI8Op:
    dtype: Any
    accumulator_dtype: Any
    tile: tuple[int, int, int]
    cta_group: Any = None
    operand_source: Any = None
    operand_major_mode_a: Any = None
    operand_major_mode_b: Any = None
    wave_size: int = 64
    lane_shape: tuple[int, int] = (4, 8)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", resolve_element_type_name(self.dtype))
        object.__setattr__(self, "accumulator_dtype", resolve_element_type_name(self.accumulator_dtype))

    @property
    def operand_dtype(self) -> str:
        return str(self.dtype)


@dataclass(frozen=True)
class CompatibleMmaDescriptor:
    tile: tuple[int, int, int]
    operand_dtype: str
    accumulator_dtype: str
    wave_size: int = 64
    lane_shape: tuple[int, int] = (4, 16)
    variant_name: str = "compatible_mma"
    llvm_intrinsic: str | None = None

    def operand_shape(self, role: str) -> tuple[int, int]:
        m, n, k = self.tile
        table = {
            "a": (m, k),
            "b": (k, n),
            "acc": (m, n),
        }
        try:
            return table[role]
        except KeyError as exc:
            raise ValueError(f"unsupported MMA fragment role '{role}'") from exc


class _ConfigurableAtom:
    runtime_state: dict[str, Any]

    def _modifier_key(self, modifier: Any) -> str:
        if modifier is None:
            return "value"
        if isinstance(modifier, str):
            return modifier
        return getattr(modifier, "name", str(modifier))

    def get(self, modifier: Any = None, default: Any = None) -> Any:
        key = self._modifier_key(modifier)
        if key == "value" and hasattr(self, "op"):
            return getattr(self, "op")
        return self.runtime_state.get(key, default)

    def set(self, modifier: Any, value: Any) -> "_ConfigurableAtom":
        self.runtime_state[self._modifier_key(modifier)] = value
        return self

    def with_(self, **updates: Any):
        state = dict(self.runtime_state)
        state.update(updates)
        return replace(self, runtime_state=state)


@dataclass(frozen=True)
class CopyAtom(_ConfigurableAtom):
    op: Any
    value_type: str
    num_bits_per_copy: int | None = None
    runtime_state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_type", resolve_element_type_name(self.value_type))
        if self.num_bits_per_copy is not None and self.num_bits_per_copy <= 0:
            raise ValueError("copy atoms require num_bits_per_copy > 0 when specified")

    @property
    def vector_bytes(self) -> int | None:
        if self.num_bits_per_copy is None:
            return None
        if self.num_bits_per_copy % 8 != 0:
            raise ValueError("copy atoms require num_bits_per_copy to be byte aligned")
        return self.num_bits_per_copy // 8

    @property
    def type(self) -> str:
        return self.value_type


@dataclass(frozen=True)
class TiledCopyTV(_ConfigurableAtom):
    atom: CopyAtom
    thread_layout: Layout
    value_layout: Layout
    runtime_state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def tv_layout(self) -> ThreadValueLayout:
        return ThreadValueLayout(self.thread_layout, self.value_layout)

    @property
    def layout_tv(self) -> ThreadValueLayout:
        return self.tv_layout

    @property
    def thr_layout(self) -> Layout:
        return self.thread_layout

    @property
    def val_layout(self) -> Layout:
        return self.value_layout

    @property
    def tiler(self) -> tuple[int, ...]:
        return self.tv_layout.tile_shape

    @property
    def num_threads(self) -> int:
        return prod(self.thread_layout.shape)

    @property
    def num_values(self) -> int:
        return prod(self.value_layout.shape)

    @property
    def type(self) -> str:
        return self.atom.value_type

    def get_slice(self, thread_index: ScalarValue | int) -> "ThreadCopySlice":
        return ThreadCopySlice(self, thread_index)

    def get_thread_slice(self, thread_index: ScalarValue | int) -> "ThreadCopySlice":
        return self.get_slice(thread_index)

    def partition_shape_S(self, shape: Any | None = None) -> tuple[int, ...]:
        del shape
        return self.tiler

    def partition_shape_D(self, shape: Any | None = None) -> tuple[int, ...]:
        del shape
        return self.tiler

    def partition_S(self, tensor: Any, thread_index: ScalarValue | int = 0) -> Any:
        return self.get_slice(thread_index).partition_S(tensor)

    def partition_D(self, tensor: Any, thread_index: ScalarValue | int = 0) -> Any:
        return self.get_slice(thread_index).partition_D(tensor)


@dataclass(frozen=True)
class ThreadCopySlice:
    tiled_copy: TiledCopyTV
    thread_index: ScalarValue | int

    def _partition(self, tensor: Any) -> Any:
        return composition(tensor, self.tiled_copy.tv_layout)[(self.thread_index, None)]

    def partition_S(self, tensor: Any) -> Any:
        return self._partition(tensor)

    def partition_D(self, tensor: Any) -> Any:
        return self._partition(tensor)

    def partition_shape_S(self, shape: Any | None = None) -> tuple[int, ...]:
        del shape
        return self.tiled_copy.partition_shape_S()

    def partition_shape_D(self, shape: Any | None = None) -> tuple[int, ...]:
        del shape
        return self.tiled_copy.partition_shape_D()

    def retile(self, value: Any) -> Any:
        return value

    def retile_S(self, value: Any) -> Any:
        return self.retile(value)

    def retile_D(self, value: Any) -> Any:
        return self.retile(value)

    def partition_fragment_S(self, value: Any) -> Any:
        return self.retile(value)

    def partition_fragment_D(self, value: Any) -> Any:
        return self.retile(value)


TiledCopy = TiledCopyTV


@dataclass(frozen=True)
class TiledMma(_ConfigurableAtom):
    descriptor: MFMADescriptor | CompatibleMmaDescriptor
    axes: tuple[str, str] = ("x", "y")
    atom_layout: Layout | None = None
    permutation_mnk: Any = None
    runtime_state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def tile(self) -> tuple[int, int, int]:
        return self.descriptor.tile

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.descriptor.tile

    @property
    def shape_mnk(self) -> tuple[int, int, int]:
        return self.descriptor.tile

    @property
    def type(self) -> str:
        return self.descriptor.accumulator_dtype

    def get_slice(self, thread_index: ScalarValue | int) -> "TiledMmaSlice":
        return TiledMmaSlice(self, thread_index)

    def get_thread_slice(self, thread_index: ScalarValue | int) -> "TiledMmaSlice":
        return self.get_slice(thread_index)

    def partition_shape_A(self, shape: Any | None = None) -> tuple[int, int]:
        del shape
        return self.descriptor.operand_shape("a")

    def partition_shape_B(self, shape: Any | None = None) -> tuple[int, int]:
        del shape
        return self.descriptor.operand_shape("b")

    def make_fragment_A(self, tensor: TensorValue | RuntimeTensor) -> TensorValue | RuntimeTensor:
        dtype = tensor.spec.dtype if isinstance(tensor, TensorValue) else tensor.dtype
        return make_rmem_tensor(self.descriptor.operand_shape("a"), dtype)

    def make_fragment_B(self, tensor: TensorValue | RuntimeTensor) -> TensorValue | RuntimeTensor:
        dtype = tensor.spec.dtype if isinstance(tensor, TensorValue) else tensor.dtype
        return make_rmem_tensor(self.descriptor.operand_shape("b"), dtype)

    def partition_shape_C(self, shape: tuple[int, int]) -> tuple[int, int]:
        del shape
        return self.descriptor.operand_shape("acc")

    def partition_A(self, tensor: TensorValue | RuntimeTensor, thread_index: ScalarValue | int = 0):
        return self.get_slice(thread_index).partition_A(tensor)

    def partition_B(self, tensor: TensorValue | RuntimeTensor, thread_index: ScalarValue | int = 0):
        return self.get_slice(thread_index).partition_B(tensor)

    def partition_C(self, tensor: TensorValue | RuntimeTensor, thread_index: ScalarValue | int = 0):
        return self.get_slice(thread_index).partition_C(tensor)

    def make_fragment_C(self, shape: Any) -> TensorValue | RuntimeTensor:
        if hasattr(shape, "shape"):
            fragment_shape = tuple(getattr(shape, "shape"))
        elif isinstance(shape, (tuple, list)):
            fragment_shape = tuple(int(dim) for dim in shape)
        else:
            raise TypeError("make_fragment_C expects a shape tuple/list or a tensor-like value with a shape")
        return make_rmem_tensor(fragment_shape, self.descriptor.accumulator_dtype)

    def make_fragment_ACC(self, shape: Any) -> TensorValue | RuntimeTensor:
        return self.make_fragment_C(shape)

    def set(self, field: Any, value: Any) -> "TiledMma":
        return super().set(field, value)


@dataclass(frozen=True)
class TiledMmaSlice:
    tiled_mma: TiledMma
    thread_index: ScalarValue | int

    def partition_A(self, tensor: TensorValue) -> TensorValue:
        _ = self.thread_index
        try:
            return make_fragment_a(
                tensor,
                tile=self.tiled_mma.tile,
                axes=self.tiled_mma.axes,
                wave_size=self.tiled_mma.descriptor.wave_size,
                accumulator_dtype=self.tiled_mma.descriptor.accumulator_dtype,
            )
        except ValueError:
            return partition_wave(
                tensor,
                self.tiled_mma.descriptor.operand_shape("a"),
                axes=self.tiled_mma.axes,
                wave_size=self.tiled_mma.descriptor.wave_size,
            )

    def partition_B(self, tensor: TensorValue) -> TensorValue:
        _ = self.thread_index
        try:
            return make_fragment_b(
                tensor,
                tile=self.tiled_mma.tile,
                axes=self.tiled_mma.axes,
                wave_size=self.tiled_mma.descriptor.wave_size,
                accumulator_dtype=self.tiled_mma.descriptor.accumulator_dtype,
            )
        except ValueError:
            return partition_wave(
                tensor,
                self.tiled_mma.descriptor.operand_shape("b"),
                axes=self.tiled_mma.axes,
                wave_size=self.tiled_mma.descriptor.wave_size,
            )

    def partition_C(self, tensor: TensorValue) -> TensorValue:
        _ = self.thread_index
        return partition_wave(
            tensor,
            self.tiled_mma.descriptor.operand_shape("acc"),
            axes=self.tiled_mma.axes,
            wave_size=self.tiled_mma.descriptor.wave_size,
        )

    def partition_shape_A(self, shape: Any | None = None) -> tuple[int, int]:
        return self.tiled_mma.partition_shape_A(shape)

    def partition_shape_B(self, shape: Any | None = None) -> tuple[int, int]:
        return self.tiled_mma.partition_shape_B(shape)

    def partition_shape_C(self, shape: Any | None = None) -> tuple[int, int]:
        if shape is None:
            resolved_shape = self.tiled_mma.descriptor.operand_shape("acc")
        elif hasattr(shape, "shape"):
            resolved_shape = tuple(getattr(shape, "shape"))
        else:
            resolved_shape = tuple(shape)
        return self.tiled_mma.partition_shape_C(resolved_shape)

    def make_fragment_A(self, tensor: TensorValue | RuntimeTensor) -> TensorValue | RuntimeTensor:
        return self.tiled_mma.make_fragment_A(tensor)

    def make_fragment_B(self, tensor: TensorValue | RuntimeTensor) -> TensorValue | RuntimeTensor:
        return self.tiled_mma.make_fragment_B(tensor)

    def make_fragment_C(self, shape: Any) -> TensorValue | RuntimeTensor:
        return self.tiled_mma.make_fragment_C(shape)

    def make_fragment_ACC(self, shape: Any) -> TensorValue | RuntimeTensor:
        return self.make_fragment_C(shape)

    def partition_fragment_A(self, value: Any) -> Any:
        return value

    def partition_fragment_B(self, value: Any) -> Any:
        return value

    def partition_fragment_C(self, value: Any) -> Any:
        return value


@dataclass(frozen=True)
class ComposedLayout:
    inner: Any
    offset: int | tuple[int, ...]
    outer: Any

    @property
    def shape(self) -> tuple[int, ...]:
        return self.outer.shape

    def __call__(self, coord: int | tuple[int, ...]) -> Any:
        outer_value = self.outer(coord)
        adjusted = _add_layout_offset(outer_value, self.offset)
        if isinstance(self.inner, Layout):
            return self.inner(adjusted)
        if callable(self.inner):
            return self.inner(adjusted)
        raise TypeError("composed layout inner must be callable or a baybridge.Layout")

    def __repr__(self) -> str:
        return f"ComposedLayout(offset={self.offset}, outer={self.outer})"


@dataclass(frozen=True)
class IdentityLayout:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = _flatten_dims(self.shape)
        if not normalized:
            raise ValueError("identity layouts require a non-empty shape")
        if any(dim <= 0 for dim in normalized):
            raise ValueError("identity layout dimensions must be > 0")
        object.__setattr__(self, "shape", normalized)

    def __call__(self, coord: int | tuple[int, ...]) -> Any:
        if isinstance(coord, int):
            if len(self.shape) == 1:
                return coord
            remaining = coord
            coords: list[int] = []
            for axis in range(len(self.shape) - 1):
                stride = prod(self.shape[axis + 1 :])
                coords.append(remaining // stride)
                remaining %= stride
            coords.append(remaining)
            return tuple(coords)
        return coord

    def __repr__(self) -> str:
        return f"IdentityLayout(shape={self.shape})"


@dataclass(frozen=True)
class Swizzle:
    bits: int
    base: int
    shift: int

    def __post_init__(self) -> None:
        if self.bits < 0 or self.base < 0 or self.shift < 0:
            raise ValueError("swizzle parameters must be >= 0")

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def __call__(self, coord: int | tuple[int, ...]) -> int | tuple[int, ...]:
        # Baybridge currently treats swizzles as layout metadata for the AMD path.
        return coord

    def __repr__(self) -> str:
        return f"Swizzle(bits={self.bits}, base={self.base}, shift={self.shift})"


@dataclass(frozen=True)
class HierarchicalLayout:
    shape: Any
    stride: Any
    swizzle: str | None = None

    def __call__(self, coord: int | tuple[int, ...]) -> int:
        flat_shape = _flatten_dims(self.shape)
        if isinstance(coord, int):
            indices = _compact_flat_coord(coord, flat_shape)
        else:
            indices = _flatten_dims(coord)
        flat_stride = _flatten_dims(self.stride)
        if len(indices) != len(flat_stride):
            raise ValueError(f"layout expects {len(flat_stride)} coordinates, got {len(indices)}")
        return sum(index * stride for index, stride in zip(indices, flat_stride))

    def __repr__(self) -> str:
        return f"{self.shape}:{self.stride}"


@dataclass(frozen=True)
class KernelDefinition:
    fn: Callable[..., Any]
    kind: str
    launch: LaunchConfig

    @property
    def __name__(self) -> str:
        return self.fn.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        normalized_args = tuple(normalize_runtime_argument(arg) for arg in args)
        normalized_kwargs = {name: normalize_runtime_argument(value) for name, value in kwargs.items()}
        if self.kind == "kernel":
            return KernelLaunch(self, normalized_args, normalized_kwargs)
        return self.fn(*normalized_args, **normalized_kwargs)

    def set_name_prefix(self, prefix: str) -> None:
        del prefix


@dataclass(frozen=True)
class KernelLaunch:
    definition: KernelDefinition
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def launch(
        self,
        *,
        grid: tuple[int, int, int] | list[int],
        block: tuple[int, int, int] | list[int],
        cluster: tuple[int, int, int] | list[int] = (1, 1, 1),
        shared_mem_bytes: int = 0,
        cooperative: bool = False,
        stream: Any | None = None,
    ) -> None:
        del stream
        launch = LaunchConfig(
            grid=tuple(grid),
            block=tuple(block),
            cluster=tuple(cluster),
            shared_mem_bytes=shared_mem_bytes,
            cooperative=cooperative,
        )
        try:
            require_builder()
        except CompilationError:
            pass
        else:
            raise TraceKernelLaunch(self.definition, self.args, self.kwargs, launch)
        with materialized_runtime_call(self.args, self.kwargs) as (call_args, call_kwargs):
            for block_z in range(launch.grid[2]):
                for block_y in range(launch.grid[1]):
                    for block_x in range(launch.grid[0]):
                        block_idx_state = (block_x, block_y, block_z)
                        for thread_z in range(launch.block[2]):
                            for thread_y in range(launch.block[1]):
                                for thread_x in range(launch.block[0]):
                                    state = ExecutionState(
                                        launch=launch,
                                        block_idx=block_idx_state,
                                        thread_idx=(thread_x, thread_y, thread_z),
                                    )
                                    with _execution(state):
                                        self.definition.fn(*call_args, **call_kwargs)

    def smem_usage(self) -> int:
        from .compiler import _trace

        traced = _trace(self.definition, self.args, self.kwargs)
        return traced.launch.shared_mem_bytes


class TraceKernelLaunch(CompilationError):
    def __init__(
        self,
        definition: KernelDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        launch: LaunchConfig,
    ):
        super().__init__("kernel launch tracing is being handled by baybridge compile")
        self.definition = definition
        self.args = args
        self.kwargs = kwargs
        self.launch = launch


def kernel(
    fn: Callable[..., Any] | None = None,
    *,
    launch: LaunchConfig | None = None,
) -> KernelDefinition | Callable[[Callable[..., Any]], KernelDefinition]:
    selected_launch = launch or LaunchConfig()
    if fn is None:
        return lambda inner: KernelDefinition(fn=inner, kind="kernel", launch=selected_launch)
    return KernelDefinition(fn=fn, kind="kernel", launch=selected_launch)


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    launch: LaunchConfig | None = None,
) -> KernelDefinition | Callable[[Callable[..., Any]], KernelDefinition]:
    selected_launch = launch or LaunchConfig()
    if fn is None:
        return lambda inner: KernelDefinition(fn=inner, kind="jit", launch=selected_launch)
    return KernelDefinition(fn=fn, kind="jit", launch=selected_launch)


def make_layout(
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    *,
    swizzle: str | None = None,
) -> Layout:
    if stride is None:
        return Layout.row_major(shape, swizzle=swizzle)
    return Layout(shape=shape, stride=stride, swizzle=swizzle)


def make_ordered_layout(
    shape: tuple[int, ...],
    order: tuple[int, ...],
    *,
    swizzle: str | None = None,
) -> Layout:
    return Layout.ordered(shape=shape, order=order, swizzle=swizzle)


def make_identity_layout(shape: tuple[int, ...] | int) -> IdentityLayout:
    if isinstance(shape, int):
        return IdentityLayout((shape,))
    return IdentityLayout(shape)


def _add_layout_offset(value: Any, offset: int | tuple[int, ...]) -> Any:
    if isinstance(offset, int):
        return value + offset
    if isinstance(value, tuple):
        if len(value) != len(offset):
            raise ValueError("composed layout tuple offsets must match the coordinate rank")
        return tuple(item + delta for item, delta in zip(value, offset))
    if len(offset) != 1:
        raise ValueError("scalar composed layout values require a scalar offset or a 1-tuple offset")
    return value + offset[0]


def make_composed_layout(inner: Any, offset: int | tuple[int, ...], outer: Any) -> ComposedLayout:
    if not hasattr(outer, "shape") or not callable(outer):
        raise TypeError("make_composed_layout expects a callable outer layout with a shape attribute")
    return ComposedLayout(inner=inner, offset=offset, outer=outer)


def make_swizzle(bits: int, base: int, shift: int) -> Swizzle:
    return Swizzle(int(bits), int(base), int(shift))


def tile_to_shape(
    layout_atom: Layout | ComposedLayout,
    shape: tuple[int, ...] | list[int],
    order: tuple[int, ...] | list[int],
) -> Layout | ComposedLayout:
    target_shape = tuple(int(dim) for dim in shape)
    target_order = tuple(int(axis) for axis in order)
    outer = Layout.ordered(target_shape, target_order)
    if isinstance(layout_atom, ComposedLayout):
        return ComposedLayout(inner=layout_atom.inner, offset=layout_atom.offset, outer=outer)
    if isinstance(layout_atom, Layout):
        return Layout.ordered(target_shape, target_order, swizzle=layout_atom.swizzle)
    raise TypeError("tile_to_shape expects a baybridge.Layout or baybridge.ComposedLayout")


def _layout_size(layout: Layout) -> int:
    return prod(layout.shape) if layout.shape else 1


def _layout_cosize(layout: Layout) -> int:
    if not layout.shape:
        return 1
    return 1 + sum((dim - 1) * stride for dim, stride in zip(layout.shape, layout.stride))


def _group_shape_stride(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    start: int,
    end: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if start < 0 or end < 0 or start >= end or end > len(shape):
        raise ValueError(f"group_modes expects 0 <= start < end <= rank, got start={start}, end={end}, rank={len(shape)}")
    grouped_shape = shape[:start] + (prod(shape[start:end]),) + shape[end:]
    grouped_stride = stride[:start] + (min(stride[start:end]),) + stride[end:]
    return grouped_shape, grouped_stride


def _compact_flat_coord(linear_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    coords: list[int] = []
    remaining = linear_index
    for axis in range(len(shape) - 1):
        stride = prod(shape[axis + 1 :])
        coords.append(remaining // stride)
        remaining %= stride
    coords.append(remaining)
    return tuple(coords)


def _normalize_shape_tree(value: Any) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_shape_tree(item) for item in value)
    raise TypeError("layout trees must contain integers or nested tuples/lists of integers")


def _as_layout(tiler: Any) -> Layout:
    if isinstance(tiler, Layout):
        return tiler
    if isinstance(tiler, int):
        return make_layout(tiler)
    if isinstance(tiler, (tuple, list)):
        shapes: list[int] = []
        strides: list[int] = []
        for item in tiler:
            if item is None:
                continue
            if isinstance(item, Layout):
                shapes.extend(item.shape)
                strides.extend(item.stride)
                continue
            if isinstance(item, int):
                shapes.append(item)
                strides.append(1)
                continue
            raise TypeError("tiler entries must be baybridge.Layout, integers, or None")
        return Layout(shape=tuple(shapes), stride=tuple(strides))
    raise TypeError("tiler must be a baybridge.Layout, integer, or tuple/list of those values")


def _expand_tiler_entries(tiler: Any, *, rank: int) -> tuple[int | None, ...]:
    if isinstance(tiler, Layout):
        dims = tiler.shape
    elif isinstance(tiler, int):
        dims = (tiler,)
    elif isinstance(tiler, (tuple, list)):
        dims_list: list[int | None] = []
        for item in tiler:
            if item is None:
                dims_list.append(None)
                continue
            if isinstance(item, Layout):
                dims_list.extend(item.shape)
                continue
            if isinstance(item, int):
                dims_list.append(item)
                continue
            raise TypeError("tiler values must be baybridge.Layout, integers, or None")
        dims = tuple(dims_list)
    else:
        raise TypeError("tiler must be a baybridge.Layout, integer, or tuple/list of those values")
    if len(dims) != rank:
        raise ValueError(f"tiler rank {len(dims)} must match layout rank {rank}")
    if any(dim is not None and dim <= 0 for dim in dims):
        raise ValueError("tiler dimensions must be > 0 when specified")
    return tuple(dims)


def depth(value: Any) -> int:
    if hasattr(value, "shape") and not isinstance(value, Layout):
        return depth(getattr(value, "shape"))
    if isinstance(value, Layout):
        return 1
    if isinstance(value, int):
        return 0
    if isinstance(value, (tuple, list)):
        if not value:
            return 1
        if all(isinstance(item, int) for item in value):
            return 1
        return 1 + max(depth(item) for item in value)
    raise TypeError("depth expects a baybridge.Layout or a shape-like tuple/list")


def ceil_div(lhs: Any, rhs: Any) -> Any:
    if isinstance(lhs, (tuple, list)) or isinstance(rhs, (tuple, list)):
        if not isinstance(lhs, (tuple, list)) or not isinstance(rhs, (tuple, list)):
            raise TypeError("ceil_div expects both operands to be scalars or both to be tuples/lists")
        if len(lhs) != len(rhs):
            raise ValueError("ceil_div tuple operands must have the same length")
        return tuple(ceil_div(lhs_item, rhs_item) for lhs_item, rhs_item in zip(lhs, rhs))
    if isinstance(lhs, ScalarValue) or isinstance(rhs, ScalarValue):
        lhs_value = lhs if isinstance(lhs, ScalarValue) else require_builder().constant(lhs)
        rhs_value = rhs if isinstance(rhs, ScalarValue) else require_builder().constant(rhs, dtype=lhs_value.spec.dtype)
        return (lhs_value + rhs_value - 1) // rhs_value
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def cosize(value: Any) -> int:
    if isinstance(value, Layout):
        return _layout_cosize(value)
    if isinstance(value, HierarchicalLayout):
        flat_shape = _flatten_dims(value.shape)
        flat_stride = _flatten_dims(value.stride)
        if not flat_shape:
            return 1
        return 1 + sum((dim - 1) * stride for dim, stride in zip(flat_shape, flat_stride))
    if isinstance(value, ComposedLayout):
        if isinstance(value.outer, Layout):
            return _layout_cosize(value.outer)
        if hasattr(value.outer, "shape"):
            return prod(_flatten_dims(value.outer.shape))
    if hasattr(value, "layout") and isinstance(getattr(value, "layout"), Layout):
        return _layout_cosize(getattr(value, "layout"))
    if hasattr(value, "shape"):
        return prod(_flatten_dims(getattr(value, "shape")))
    raise TypeError("cosize expects a baybridge layout or shape-carrying value")


def size_in_bytes(dtype: Any, value: Any) -> int:
    if hasattr(dtype, "width"):
        width_bits = int(dtype.width)
    else:
        resolved_dtype = getattr(dtype, "__baybridge_dtype__", dtype)
        width_bits = element_type(str(resolved_dtype)).width
    byte_width = (width_bits + 7) // 8
    if isinstance(value, Layout):
        element_count = cosize(value)
    elif hasattr(value, "layout") and isinstance(getattr(value, "layout"), Layout):
        element_count = cosize(getattr(value, "layout"))
    elif hasattr(value, "shape"):
        element_count = prod(_flatten_dims(getattr(value, "shape")))
    elif isinstance(value, (tuple, list)):
        element_count = prod(_flatten_dims(value))
    else:
        raise TypeError("size_in_bytes expects a layout, tensor-like value, or shape tuple/list")
    return byte_width * element_count


def group_modes(value: Any, start: int, end: int) -> Any:
    if isinstance(value, Layout):
        grouped_shape, grouped_stride = _group_shape_stride(value.shape, value.stride, start, end)
        return Layout(shape=grouped_shape, stride=grouped_stride, swizzle=value.swizzle)
    if isinstance(value, RuntimeTensor):
        grouped_shape, grouped_stride = _group_shape_stride(value.shape, value.stride, start, end)
        return RuntimeTensor(
            value._storage,
            grouped_shape,
            dtype=value.dtype,
            stride=grouped_stride,
            offset=value.offset,
        )
    if isinstance(value, TensorValue):
        source_layout = value.spec.resolved_layout()
        grouped_shape, grouped_stride = _group_shape_stride(source_layout.shape, source_layout.stride, start, end)
        grouped_spec = TensorSpec(
            shape=grouped_shape,
            dtype=value.spec.dtype,
            layout=Layout(shape=grouped_shape, stride=grouped_stride, swizzle=source_layout.swizzle),
            address_space=value.spec.address_space,
        )
        return require_builder().emit_tensor(
            "group_modes",
            value,
            spec=grouped_spec,
            attrs={"start": start, "end": end},
            name_hint=f"{value.name}_group",
        )
    if hasattr(value, "shape") and hasattr(value, "layout"):
        return group_modes(value.layout, start, end)
    raise TypeError("group_modes expects a baybridge tensor or layout")


def _local_tile_shape_stride(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    tiler: tuple[int, ...],
    coord: tuple[ScalarValue | int | None, ...],
    proj: tuple[int | None, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[ScalarValue | int, ...], tuple[int, ...], tuple[int | None, ...], tuple[int, ...]]:
    if len(tiler) != len(coord) or len(coord) != len(proj):
        raise ValueError("local_tile expects tiler, coord, and proj to have the same rank")
    selected_axes = tuple(axis for axis, item in enumerate(proj) if item is not None)
    if len(shape) != len(selected_axes):
        raise ValueError(
            f"local_tile expects tensor rank {len(shape)} to match the number of projected tiler axes {len(selected_axes)}"
        )
    out_shape: list[int] = []
    out_stride: list[int] = []
    base_offsets: list[ScalarValue | int] = []
    tile_axes: list[int] = []
    rest_axes: list[int | None] = []
    tile_sizes: list[int] = []
    for tensor_axis, tiler_axis in enumerate(selected_axes):
        tile_size = int(tiler[tiler_axis])
        if tile_size <= 0:
            raise ValueError("local_tile tiler dimensions must be > 0")
        axis_coord = coord[tiler_axis]
        tile_axes.append(len(out_shape))
        out_shape.append(tile_size)
        out_stride.append(stride[tensor_axis])
        tile_sizes.append(tile_size)
        if axis_coord is None:
            rest_axes.append(len(out_shape))
            out_shape.append(ceil_div(shape[tensor_axis], tile_size))
            out_stride.append(tile_size * stride[tensor_axis])
            base_offsets.append(0)
            continue
        base_offsets.append(axis_coord * tile_size)
        rest_axes.append(None)
    return (
        tuple(out_shape),
        tuple(out_stride),
        tuple(base_offsets),
        tuple(tile_axes),
        tuple(rest_axes),
        tuple(tile_sizes),
    )


def local_tile(
    tensor: TensorValue | RuntimeTensor | IdentityTensor,
    tiler: tuple[int, ...],
    coord: tuple[ScalarValue | int | None, ...],
    *,
    proj: tuple[int | None, ...],
) -> Any:
    if isinstance(tensor, IdentityTensor):
        canonical_stride = []
        running = 1
        for axis in range(len(tensor.shape) - 1, -1, -1):
            canonical_stride.insert(0, running)
            running *= tensor.shape[axis]
        out_shape, _, base_offsets, tile_axes, rest_axes, tile_sizes = _local_tile_shape_stride(
            tensor.shape,
            tuple(canonical_stride),
            tiler,
            coord,
            proj,
        )
        if any(isinstance(offset, ScalarValue) for offset in base_offsets):
            return LocalCoordinateTensorValue(out_shape, base_offsets, tile_sizes, tile_axes, rest_axes)
        return LocalCoordinateTensor(out_shape, tuple(int(offset) for offset in base_offsets), tile_sizes, tile_axes, rest_axes)
    if isinstance(tensor, RuntimeTensor):
        out_shape, out_stride, base_offsets, _, _, _ = _local_tile_shape_stride(
            tensor.shape,
            tensor.stride,
            tiler,
            coord,
            proj,
        )
        if any(isinstance(offset, ScalarValue) for offset in base_offsets):
            raise TypeError("runtime local_tile currently requires integer coordinates or None")
        linear_offset = tensor.offset + sum(int(offset) * step for offset, step in zip(base_offsets, tensor.stride))
        return RuntimeTensor(
            tensor._storage,
            out_shape,
            dtype=tensor.dtype,
            stride=out_stride,
            offset=linear_offset,
        )
    source_layout = tensor.spec.resolved_layout()
    out_shape, out_stride, base_offsets, _, _, tile_sizes = _local_tile_shape_stride(
        tensor.spec.shape,
        source_layout.stride,
        tiler,
        coord,
        proj,
    )
    builder = require_builder()
    fixed_coords = []
    fixed_axes = []
    selected_axes = [axis for axis, item in enumerate(proj) if item is not None]
    for tensor_axis, (tiler_axis, offset) in enumerate(zip(selected_axes, base_offsets)):
        if coord[tiler_axis] is None:
            continue
        fixed_axes.append(tensor_axis)
        if isinstance(offset, ScalarValue):
            fixed_coords.append(offset)
        else:
            fixed_coords.append(builder.constant(offset))
    local_spec = TensorSpec(
        shape=out_shape,
        dtype=tensor.spec.dtype,
        layout=Layout(shape=out_shape, stride=out_stride, swizzle=source_layout.swizzle),
        address_space=tensor.spec.address_space,
    )
    return builder.emit_tensor(
        "local_tile",
        tensor,
        *fixed_coords,
        spec=local_spec,
        attrs={
            "tiler": list(tiler),
            "proj": [item for item in proj],
            "fixed_axes": fixed_axes,
            "tile_sizes": list(tile_sizes),
        },
        name_hint=f"{tensor.name}_tile",
    )


def coalesce(layout: Layout, *, target_profile: tuple[int, ...] | None = None) -> Layout:
    if not isinstance(layout, Layout):
        raise TypeError("coalesce expects a baybridge.Layout")
    if target_profile is not None and len(target_profile) != len(layout.shape):
        raise ValueError("target_profile must have the same rank as the layout")
    merged_shape: list[int] = []
    merged_stride: list[int] = []
    for dim, stride in zip(layout.shape, layout.stride):
        if dim == 1:
            if merged_shape:
                merged_shape[-1] *= dim
                merged_stride[-1] = min(merged_stride[-1], stride)
            else:
                merged_shape.append(dim)
                merged_stride.append(stride)
            continue
        if merged_shape:
            prev_dim = merged_shape[-1]
            prev_stride = merged_stride[-1]
            contiguous = prev_stride == stride * dim or stride == prev_stride * prev_dim
            if contiguous:
                merged_shape[-1] *= dim
                merged_stride[-1] = min(prev_stride, stride)
                continue
        merged_shape.append(dim)
        merged_stride.append(stride)
    if not merged_shape:
        merged_shape = [1]
        merged_stride = [1]
    return Layout(shape=tuple(merged_shape), stride=tuple(merged_stride), swizzle=layout.swizzle)


def _normalize_tiler_dims(tiler: Any, *, rank: int) -> tuple[int, ...]:
    if isinstance(tiler, Layout):
        dims = tiler.shape
    elif isinstance(tiler, int):
        dims = (tiler,)
    elif isinstance(tiler, (tuple, list)):
        dims_list: list[int] = []
        for item in tiler:
            if isinstance(item, Layout):
                dims_list.extend(item.shape)
            elif isinstance(item, int):
                dims_list.append(item)
            else:
                raise TypeError("tiler values must be baybridge.Layout or integers")
        dims = tuple(dims_list)
    else:
        raise TypeError("tiler must be a baybridge.Layout, integer, or tuple/list of those values")
    if len(dims) != rank:
        raise ValueError(f"tiler rank {len(dims)} must match layout rank {rank}")
    if any(dim <= 0 for dim in dims):
        raise ValueError("tiler dimensions must be > 0")
    return dims


def flat_divide(layout: Layout, *, tiler: Any) -> Layout:
    if not isinstance(layout, Layout):
        raise TypeError("flat_divide expects a baybridge.Layout")
    tile_entries = _expand_tiler_entries(tiler, rank=len(layout.shape))
    tile = tuple(1 if step is None else step for step in tile_entries)
    rest = tuple((dim + step - 1) // step for dim, step in zip(layout.shape, tile))
    return Layout(
        shape=tile + rest,
        stride=layout.stride + tuple(step * stride for step, stride in zip(tile, layout.stride)),
        swizzle=layout.swizzle,
    )


def logical_divide(layout: Layout, *, tiler: Any) -> HierarchicalLayout:
    if not isinstance(layout, Layout):
        raise TypeError("logical_divide expects a baybridge.Layout")
    tile_entries = _expand_tiler_entries(tiler, rank=len(layout.shape))
    shape_parts: list[Any] = []
    stride_parts: list[Any] = []
    for dim, stride, tile in zip(layout.shape, layout.stride, tile_entries):
        if tile is None:
            shape_parts.append(dim)
            stride_parts.append(stride)
            continue
        rest = (dim + tile - 1) // tile
        shape_parts.append((tile, rest))
        stride_parts.append((stride, tile * stride))
    return HierarchicalLayout(
        shape=_normalize_shape_tree(tuple(shape_parts)),
        stride=_normalize_shape_tree(tuple(stride_parts)),
        swizzle=layout.swizzle,
    )


def tiled_divide(layout: Layout, *, tiler: Any) -> HierarchicalLayout:
    if not isinstance(layout, Layout):
        raise TypeError("tiled_divide expects a baybridge.Layout")
    tile_entries = _expand_tiler_entries(tiler, rank=len(layout.shape))
    tile_shape = tuple(tile for tile in tile_entries if tile is not None)
    tile_stride = tuple(stride for stride, tile in zip(layout.stride, tile_entries) if tile is not None)
    rest_shape = tuple(
        dim if tile is None else (dim + tile - 1) // tile
        for dim, tile in zip(layout.shape, tile_entries)
    )
    rest_stride = tuple(
        stride if tile is None else tile * stride
        for stride, tile in zip(layout.stride, tile_entries)
    )
    shape_parts: tuple[Any, ...]
    stride_parts: tuple[Any, ...]
    if tile_shape:
        shape_parts = (_normalize_shape_tree(tile_shape),) + tuple(rest_shape)
        stride_parts = (_normalize_shape_tree(tile_stride),) + tuple(rest_stride)
    else:
        shape_parts = tuple(rest_shape)
        stride_parts = tuple(rest_stride)
    return HierarchicalLayout(shape=shape_parts, stride=stride_parts, swizzle=layout.swizzle)


def blocked_product(layout: Layout, *, tiler: Layout | tuple[int, ...] | int) -> Layout:
    if not isinstance(layout, Layout):
        raise TypeError("blocked_product expects a baybridge.Layout")
    if isinstance(tiler, Layout):
        tile_layout = tiler
    else:
        tile_layout = Layout.row_major(_normalize_tiler_dims(tiler, rank=len(_flatten_dims(tiler))))
    tile_size = _layout_size(tile_layout)
    return Layout(
        shape=layout.shape + tile_layout.shape,
        stride=tuple(stride * tile_size for stride in layout.stride) + tile_layout.stride,
        swizzle=layout.swizzle or tile_layout.swizzle,
    )


def raked_product(layout: Layout, *, tiler: Layout | tuple[int, ...] | int) -> Layout:
    blocked = blocked_product(layout, tiler=tiler)
    if isinstance(tiler, Layout):
        tile_layout = tiler
    else:
        tile_layout = Layout.row_major(_normalize_tiler_dims(tiler, rank=len(_flatten_dims(tiler))))
    if len(layout.shape) != len(tile_layout.shape):
        return blocked
    interleaved_shape: list[int] = []
    interleaved_stride: list[int] = []
    tile_size = _layout_size(tile_layout)
    for index, (shape_dim, stride_dim, tile_dim, tile_stride) in enumerate(
        zip(layout.shape, layout.stride, tile_layout.shape, tile_layout.stride)
    ):
        del index
        interleaved_shape.extend([shape_dim, tile_dim])
        interleaved_stride.extend([stride_dim * tile_size, tile_stride])
    return Layout(shape=tuple(interleaved_shape), stride=tuple(interleaved_stride), swizzle=blocked.swizzle)


def logical_product(layout: Layout, *, tiler: Any) -> HierarchicalLayout:
    if not isinstance(layout, Layout):
        raise TypeError("logical_product expects a baybridge.Layout")
    tile_layout = _as_layout(tiler)
    replication_stride = _layout_cosize(layout)
    return HierarchicalLayout(
        shape=(_normalize_shape_tree(layout.shape), _normalize_shape_tree(tile_layout.shape)),
        stride=(
            _normalize_shape_tree(layout.stride),
            _normalize_shape_tree(tuple(stride * replication_stride for stride in tile_layout.stride)),
        ),
        swizzle=layout.swizzle or tile_layout.swizzle,
    )


def zipped_product(layout: Layout, *, tiler: Any) -> HierarchicalLayout:
    return logical_product(layout, tiler=tiler)


def tiled_product(layout: Layout, *, tiler: Any) -> HierarchicalLayout:
    if not isinstance(layout, Layout):
        raise TypeError("tiled_product expects a baybridge.Layout")
    tile_layout = _as_layout(tiler)
    replication_stride = _layout_cosize(layout)
    return HierarchicalLayout(
        shape=(_normalize_shape_tree(layout.shape),) + tuple(tile_layout.shape),
        stride=(_normalize_shape_tree(layout.stride),)
        + tuple(stride * replication_stride for stride in tile_layout.stride),
        swizzle=layout.swizzle or tile_layout.swizzle,
    )


def flat_product(layout: Layout, *, tiler: Any) -> Layout:
    if not isinstance(layout, Layout):
        raise TypeError("flat_product expects a baybridge.Layout")
    tile_layout = _as_layout(tiler)
    replication_stride = _layout_cosize(layout)
    return Layout(
        shape=layout.shape + tile_layout.shape,
        stride=layout.stride + tuple(stride * replication_stride for stride in tile_layout.stride),
        swizzle=layout.swizzle or tile_layout.swizzle,
    )


def recast_layout(dst_bits: int, src_bits: int, layout: Layout) -> Layout:
    if dst_bits <= 0 or src_bits <= 0:
        raise ValueError("recast_layout bit widths must be > 0")
    contiguous_axes = [axis for axis, stride in enumerate(layout.stride) if stride == 1]
    if len(contiguous_axes) != 1:
        raise ValueError("recast_layout currently requires exactly one contiguous axis with stride 1")
    axis = contiguous_axes[0]
    scaled_extent = layout.shape[axis] * src_bits
    if scaled_extent % dst_bits != 0:
        raise ValueError("recast_layout requires the contiguous axis extent to be divisible by the target bit width")
    scale = src_bits / dst_bits
    new_shape = list(layout.shape)
    new_shape[axis] = scaled_extent // dst_bits
    new_stride: list[int] = []
    for index, stride in enumerate(layout.stride):
        if index == axis:
            new_stride.append(1)
            continue
        scaled_stride = stride * scale
        if int(scaled_stride) != scaled_stride:
            raise ValueError("recast_layout produced a non-integral stride")
        new_stride.append(int(scaled_stride))
    return Layout(shape=tuple(new_shape), stride=tuple(new_stride), swizzle=layout.swizzle)


def make_tensor(
    name: str | Pointer,
    shape: tuple[int, ...] | Layout | None = None,
    dtype: str | None = None,
    *,
    layout: Layout | None = None,
    address_space: AddressSpace | str = AddressSpace.REGISTER,
) -> TensorValue | RuntimeTensor:
    if isinstance(name, Pointer):
        target_layout = shape if isinstance(shape, Layout) else layout
        if not isinstance(target_layout, Layout):
            raise TypeError("make_tensor(ptr, ...) expects a baybridge.Layout as the second argument or layout=")
        if name.tensor is None:
            raise UnsupportedOperationError("baybridge pointer tensors currently require a tensor-backed Pointer")
        target_space = normalize_address_space(name.address_space)
        if isinstance(name.tensor, TensorValue):
            spec = TensorSpec(
                shape=target_layout.shape,
                dtype=str(name.value_type),
                layout=target_layout,
                address_space=target_space,
            )
            return require_builder().emit_tensor(
                "pointer_tensor",
                name.tensor,
                spec=spec,
                attrs={
                    "assumed_align": name.assumed_align,
                    "address_space": target_space.value,
                },
                name_hint=f"{name.tensor.name}_ptr",
            )
        runtime_tensor = _runtime_tensor(name.tensor)
        needed = 1
        for dim in target_layout.shape:
            needed *= dim
        if len(runtime_tensor._storage) < needed:
            raise ValueError("pointer-backed tensor layout exceeds the underlying storage")
        return RuntimeTensor(
            runtime_tensor._storage,
            target_layout.shape,
            dtype=str(name.value_type),
            stride=target_layout.stride,
            offset=runtime_tensor.offset,
        )
    if not isinstance(name, str):
        raise TypeError("make_tensor expects either a tensor name or a baybridge Pointer")
    if shape is None or dtype is None:
        raise TypeError("make_tensor(name, ...) requires shape=... and dtype=...")
    spec = TensorSpec(shape=shape, dtype=dtype, layout=layout, address_space=address_space)
    try:
        builder = require_builder()
        existing_names = {argument.name for argument in builder.arguments}
        for operation in builder.operations:
            existing_names.update(operation.outputs)
        unique_name = name
        if unique_name in existing_names:
            builder._temp_index += 1
            unique_name = f"{name}_{builder._temp_index}"
        return builder.make_tensor(unique_name, spec)
    except CompilationError:
        del name
        del layout
        del address_space
        return zeros(shape, dtype=dtype)


def recast_ptr(value: Pointer, *, dtype: str) -> Pointer:
    if not isinstance(value, Pointer):
        raise TypeError("recast_ptr expects a baybridge.Pointer")
    return Pointer(
        value_type=str(dtype),
        tensor=value.tensor,
        raw_address=value.raw_address,
        address_space=value.address_space,
        assumed_align=value.assumed_align,
    )


@dataclass
class SmemAllocator:
    allocation_index: int = 0

    def allocate(
        self,
        item: Any,
        *,
        byte_alignment: int = 1,
    ) -> Pointer | StructInstance:
        if byte_alignment <= 0:
            raise ValueError("byte_alignment must be > 0")
        if isinstance(item, int):
            if item <= 0:
                raise ValueError("raw shared-memory allocations must be > 0 bytes")
            return self._allocate_pointer(
                dtype="i8",
                count=item,
                alignment=byte_alignment,
                name_hint="smem_bytes",
            )
        if is_struct_type(item):
            return self._allocate_struct(item, byte_alignment=byte_alignment)
        dtype = str(item)
        return self._allocate_pointer(
            dtype=dtype,
            count=1,
            alignment=max(byte_alignment, self._dtype_size(dtype)),
            name_hint="smem_scalar",
        )

    def allocate_tensor(
        self,
        *,
        element_type: str,
        layout: Layout,
        byte_alignment: int = 1,
        swizzle: str | None = None,
    ) -> TensorValue | RuntimeTensor:
        if not isinstance(layout, Layout):
            raise TypeError("SmemAllocator.allocate_tensor expects a baybridge.Layout")
        if byte_alignment <= 0:
            raise ValueError("byte_alignment must be > 0")
        target_layout = layout if swizzle is None else Layout(shape=layout.shape, stride=layout.stride, swizzle=swizzle)
        return self._allocate_shared_tensor(
            dtype=str(element_type),
            layout=target_layout,
            alignment=byte_alignment,
            name_hint="smem_alloc",
        )

    def allocate_array(
        self,
        *,
        element_type: str,
        num_elems: int,
        byte_alignment: int = 1,
    ) -> TensorValue | RuntimeTensor:
        if num_elems <= 0:
            raise ValueError("num_elems must be > 0")
        return self.allocate_tensor(
            element_type=element_type,
            layout=Layout.row_major((num_elems,)),
            byte_alignment=byte_alignment,
        )

    def _allocate_struct(
        self,
        struct_type: type,
        *,
        byte_alignment: int,
        base_offset: int | None = None,
        reserve_region: bool = True,
        name_prefix: str | None = None,
    ) -> StructInstance:
        spec = get_struct_spec(struct_type)
        builder = self._maybe_builder()
        region_offset = base_offset
        if builder is not None and reserve_region:
            region_offset = builder.reserve_dynamic_shared(
                spec.size_bytes,
                byte_alignment=max(spec.alignment_bytes, byte_alignment),
                byte_offset=base_offset,
            )
        prefix = name_prefix or self._next_name("smem_struct")
        fields: dict[str, Any] = {}
        for field in spec.fields:
            field_offset = region_offset + field.offset_bytes if region_offset is not None else None
            if field.kind == "struct":
                assert field.struct_type is not None
                fields[field.name] = self._allocate_struct(
                    field.struct_type,
                    byte_alignment=field.alignment_bytes,
                    base_offset=field_offset,
                    reserve_region=False,
                    name_prefix=f"{prefix}_{field.name}",
                )
                continue
            if field.kind == "memrange":
                assert field.dtype is not None
                assert field.count is not None
                backing = self._allocate_shared_tensor(
                    dtype=field.dtype,
                    layout=Layout.row_major((field.count,)),
                    alignment=field.alignment_bytes,
                    byte_offset=field_offset,
                    name_hint=f"{prefix}_{field.name}",
                )
                fields[field.name] = MemRangeProxy(
                    pointer=Pointer(
                        value_type=field.dtype,
                        tensor=backing,
                        address_space=AddressSpace.SHARED,
                        assumed_align=field.alignment_bytes,
                    ),
                    count=field.count,
                )
                continue
            assert field.dtype is not None
            fields[field.name] = self._allocate_pointer(
                dtype=field.dtype,
                count=1,
                alignment=field.alignment_bytes,
                byte_offset=field_offset,
                name_hint=f"{prefix}_{field.name}",
            )
        return StructInstance(spec.name, fields)

    def _allocate_pointer(
        self,
        *,
        dtype: str,
        count: int,
        alignment: int,
        byte_offset: int | None = None,
        name_hint: str,
    ) -> Pointer:
        backing = self._allocate_shared_tensor(
            dtype=dtype,
            layout=Layout.row_major((count,)),
            alignment=alignment,
            byte_offset=byte_offset,
            name_hint=name_hint,
        )
        return Pointer(
            value_type=dtype,
            tensor=backing,
            address_space=AddressSpace.SHARED,
            assumed_align=alignment,
        )

    def _allocate_shared_tensor(
        self,
        *,
        dtype: str,
        layout: Layout,
        alignment: int,
        byte_offset: int | None = None,
        name_hint: str,
    ) -> TensorValue | RuntimeTensor:
        spec = TensorSpec(
            shape=layout.shape,
            dtype=dtype,
            layout=layout,
            address_space=AddressSpace.SHARED,
        )
        builder = self._maybe_builder()
        if builder is not None:
            return builder.make_tensor(
                self._next_name(name_hint),
                spec,
                dynamic_shared=True,
                byte_alignment=alignment,
                byte_offset=byte_offset,
            )
        return zeros(layout.shape, dtype=dtype)

    def _maybe_builder(self):
        try:
            return require_builder()
        except CompilationError:
            return None

    def _next_name(self, prefix: str) -> str:
        self.allocation_index += 1
        return f"{prefix}_{self.allocation_index}"

    def _dtype_size(self, dtype: str) -> int:
        from .dtypes import element_type

        return (element_type(dtype).width + 7) // 8


@dataclass(frozen=True)
class NamedBarrier:
    barrier_id: int
    num_threads: int

    def __post_init__(self) -> None:
        if self.barrier_id < 0:
            raise ValueError("barrier_id must be >= 0")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be > 0")

    def arrive(self) -> None:
        barrier(kind="block", barrier_id=self.barrier_id, num_threads=self.num_threads, scope="named")

    def wait(self) -> None:
        barrier(kind="block", barrier_id=self.barrier_id, num_threads=self.num_threads, scope="named")

    def arrive_and_wait(self) -> None:
        barrier(kind="block", barrier_id=self.barrier_id, num_threads=self.num_threads, scope="named")

    def sync(self) -> None:
        self.arrive_and_wait()


@dataclass(frozen=True)
class Mbarrier:
    index: int

    def init(self, arrival_count: int | None = None) -> None:
        barrier(kind="block", barrier_id=self.index, scope="mbarrier", action="init", arrival_count=arrival_count)

    def init_fence(self, arrival_count: int | None = None) -> None:
        barrier(kind="block", barrier_id=self.index, scope="mbarrier", action="init_fence", arrival_count=arrival_count)

    def arrive(self) -> None:
        barrier(kind="block", barrier_id=self.index, scope="mbarrier")

    def expect_tx(self, bytes: int) -> None:
        barrier(kind="block", barrier_id=self.index, scope="mbarrier", action="expect_tx", bytes=int(bytes))

    def arrive_and_expect_tx(self, bytes: int) -> None:
        barrier(
            kind="block",
            barrier_id=self.index,
            scope="mbarrier",
            action="arrive_and_expect_tx",
            bytes=int(bytes),
        )

    def wait(self) -> None:
        barrier(kind="block", barrier_id=self.index, scope="mbarrier")

    def arrive_and_wait(self) -> None:
        barrier(kind="block", barrier_id=self.index, scope="mbarrier")

    def try_wait(self, phase: int | None = None) -> bool | ScalarValue:
        del phase
        return True

    def conditional_try_wait(self, predicate: ScalarValue | bool, phase: int | None = None) -> bool | ScalarValue:
        del phase
        return predicate

    def test_wait(self, phase: int | None = None) -> bool | ScalarValue:
        del phase
        return True

    def invalidate(self) -> None:
        barrier(kind="block", barrier_id=self.index, scope="mbarrier", action="invalidate")

    def data_ptr(self) -> Pointer:
        return Pointer(value_type="i64", raw_address=0, address_space=AddressSpace.SHARED, assumed_align=8)


@dataclass(frozen=True)
class MbarrierArray:
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("MbarrierArray count must be > 0")

    def __getitem__(self, index: int) -> Mbarrier:
        if not 0 <= index < self.count:
            raise IndexError(index)
        return Mbarrier(index=index)

    def data_ptr(self) -> Pointer:
        return Pointer(value_type="i64", raw_address=0, address_space=AddressSpace.SHARED, assumed_align=8)


@dataclass
class TmemAllocator:
    backing: Any | None = None
    barrier_for_retrieve: NamedBarrier | None = None
    allocation_index: int = 0

    def allocate(self, item: int | tuple[int, ...], *, dtype: str = "i32") -> TensorValue | RuntimeTensor:
        if isinstance(item, int):
            shape = (int(item),)
        else:
            shape = tuple(int(dim) for dim in item)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("TmemAllocator.allocate expects a positive shape")
        self.allocation_index += 1
        return make_tensor(
            f"tmem_{self.allocation_index}",
            shape,
            dtype,
            address_space=AddressSpace.REGISTER,
        )


@dataclass(frozen=True)
class TmaStoreFence:
    def arrive(self) -> None:
        barrier(kind="block", scope="tma_store_fence")

    def wait(self) -> None:
        barrier(kind="block", scope="tma_store_fence")

    def arrive_and_wait(self) -> None:
        barrier(kind="block", scope="tma_store_fence")

    def sync(self) -> None:
        self.arrive_and_wait()


def _normalize_axis(axis: str) -> str:
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of: x, y, z")
    return axis


def _default_axes(rank: int) -> tuple[str, ...]:
    table = {
        1: ("x",),
        2: ("y", "x"),
        3: ("z", "y", "x"),
    }
    if rank not in table:
        raise ValueError("default execution axes are only defined for ranks 1, 2, and 3")
    return table[rank]


def _resolve_axes(rank: int, axes: tuple[str, ...] | None) -> tuple[str, ...]:
    resolved = axes or _default_axes(rank)
    if len(resolved) != rank:
        raise ValueError(f"axes must have length {rank} for tensor rank {rank}")
    return tuple(_normalize_axis(axis) for axis in resolved)


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _runtime_state() -> ExecutionState:
    state = current_execution()
    if state is None:
        raise CompilationError("baybridge operations can only be used while tracing or during a launched kernel execution")
    return state


def _runtime_tensor(value: TensorValue | RuntimeTensor | TensorHandle) -> RuntimeTensor:
    if isinstance(value, RuntimeTensor):
        return value
    if isinstance(value, TensorHandle):
        return value.to_runtime_tensor()
    raise CompilationError("runtime tensor operations require launched baybridge tensors")


def _normalize_runtime_indices(
    index: ScalarValue | int | tuple[ScalarValue | int, ...],
    *,
    rank: int,
) -> tuple[int, ...]:
    if isinstance(index, tuple):
        raw_indices = index
    else:
        raw_indices = (index,)
    if len(raw_indices) != rank:
        raise ValueError(f"expected {rank} indices, got {len(raw_indices)}")
    normalized: list[int] = []
    for item in raw_indices:
        if isinstance(item, ScalarValue):
            raise CompilationError("runtime tensor indexing cannot use traced ScalarValue operands")
        if not isinstance(item, int):
            raise TypeError("runtime tensor indices must be integers")
        normalized.append(item)
    return tuple(normalized)


def _iter_indices(shape: tuple[int, ...]):
    if not shape:
        yield ()
        return
    yield from product(*(range(dim) for dim in shape))


def _copy_runtime_tensor(src: RuntimeTensor, dst: RuntimeTensor) -> None:
    if src.shape != dst.shape:
        raise ValueError(f"copy requires matching shapes, got {src.shape} and {dst.shape}")
    for index in _iter_indices(src.shape):
        dst[index] = src[index]


def _apply_copy_reduction(lhs: Any, rhs: Any, reduction_kind: str) -> Any:
    if reduction_kind == _CpAsyncReductionOp.ADD:
        return lhs + rhs
    if reduction_kind == _CpAsyncReductionOp.MAX:
        return lhs if lhs >= rhs else rhs
    if reduction_kind == _CpAsyncReductionOp.MIN:
        return lhs if lhs <= rhs else rhs
    if reduction_kind == _CpAsyncReductionOp.AND:
        return lhs & rhs
    if reduction_kind == _CpAsyncReductionOp.OR:
        return lhs | rhs
    if reduction_kind == _CpAsyncReductionOp.XOR:
        return lhs ^ rhs
    raise UnsupportedOperationError(
        f"cpasync TMA-reduce lowering currently supports add/max/min/and/or/xor, got '{reduction_kind}'"
    )


def _copy_reduce_runtime_tensor(src: RuntimeTensor, dst: RuntimeTensor, reduction_kind: str) -> None:
    if src.shape != dst.shape:
        raise ValueError(f"copy_reduce requires matching shapes, got {src.shape} and {dst.shape}")
    for index in _iter_indices(src.shape):
        dst[index] = _apply_copy_reduction(dst[index], src[index], reduction_kind)


def _runtime_printf_value(value: Any) -> Any:
    if isinstance(value, RuntimeTensor):
        return value.tolist()
    if isinstance(value, RuntimeScalar):
        return value.value
    return value


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, (TensorValue, RuntimeTensor, Layout)):
        return tuple(value.shape)
    if isinstance(value, (TiledTensorView, IdentityTensor)):
        raise TypeError("shape tuple extraction expects a flat tensor/layout shape; use size(..., mode=...) or select(...)")
    if isinstance(value, tuple) and all(isinstance(item, int) for item in value):
        return value
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return tuple(value)
    raise TypeError("size expects a tensor, layout, or shape tuple/list")


def _shape_value(value: Any) -> Any:
    if isinstance(
        value,
        (
            TensorValue,
            RuntimeTensor,
            Layout,
            TiledTensorView,
            IdentityTensor,
            ThreadValueLayout,
            ThreadValueComposedView,
            ThreadValueComposedTensorView,
            ThreadValueComposedCoordinateView,
        ),
    ):
        return value.shape
    if hasattr(value, "shape"):
        return value.shape
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError("expected a tensor, layout, or shape-like value")


def _flatten_dims(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (tuple, list)):
        dims: list[int] = []
        for item in value:
            dims.extend(_flatten_dims(item))
        return tuple(dims)
    raise TypeError("shape values must contain only integers or nested tuples of integers")


def program_id(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    try:
        return require_builder().emit_scalar(
            "program_id",
            spec=ScalarSpec(dtype="index"),
            attrs={"axis": normalized_axis},
            name_hint=f"program_id_{normalized_axis}",
        )
    except CompilationError:
        return _runtime_state().block_idx[_axis_index(normalized_axis)]


def block_idx(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    try:
        return require_builder().emit_scalar(
            "block_idx",
            spec=ScalarSpec(dtype="index"),
            attrs={"axis": normalized_axis},
            name_hint=f"block_idx_{normalized_axis}",
        )
    except CompilationError:
        return _runtime_state().block_idx[_axis_index(normalized_axis)]


def thread_idx(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    try:
        return require_builder().emit_scalar(
            "thread_idx",
            spec=ScalarSpec(dtype="index"),
            attrs={"axis": normalized_axis},
            name_hint=f"thread_idx_{normalized_axis}",
        )
    except CompilationError:
        return _runtime_state().thread_idx[_axis_index(normalized_axis)]


def block_dim(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    try:
        return require_builder().emit_scalar(
            "block_dim",
            spec=ScalarSpec(dtype="index"),
            attrs={"axis": normalized_axis},
            name_hint=f"block_dim_{normalized_axis}",
        )
    except CompilationError:
        return _runtime_state().block_dim[_axis_index(normalized_axis)]


def grid_dim(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    try:
        return require_builder().emit_scalar(
            "grid_dim",
            spec=ScalarSpec(dtype="index"),
            attrs={"axis": normalized_axis},
            name_hint=f"grid_dim_{normalized_axis}",
        )
    except CompilationError:
        return _runtime_state().grid_dim[_axis_index(normalized_axis)]


def lane_id() -> ScalarValue | int:
    try:
        return require_builder().emit_scalar(
            "lane_id",
            spec=ScalarSpec(dtype="index"),
            name_hint="lane_id",
        )
    except CompilationError:
        return _runtime_state().thread_idx[0] % 64


def warp_idx(*, wave_size: int = 64) -> ScalarValue | int:
    if wave_size <= 0:
        raise ValueError("wave_size must be > 0")
    return thread_idx("x") // wave_size


def wave_id(*, axis: str = "x", wave_size: int = 64) -> ScalarValue:
    normalized_axis = _normalize_axis(axis)
    if wave_size <= 0:
        raise ValueError("wave_size must be > 0")
    return thread_idx(normalized_axis) // wave_size


def cluster_dim(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    try:
        builder = require_builder()
        return builder.constant(builder.launch.cluster[_axis_index(normalized_axis)], dtype="index")
    except CompilationError:
        return _runtime_state().cluster_dim[_axis_index(normalized_axis)]


def cluster_idx(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    return block_idx(normalized_axis) // cluster_dim(normalized_axis)


def cluster_size() -> ScalarValue | int:
    try:
        builder = require_builder()
        return builder.constant(prod(builder.launch.cluster), dtype="index")
    except CompilationError:
        return _runtime_state().cluster_size


def block_idx_in_cluster(axis: str = "x") -> ScalarValue | int:
    normalized_axis = _normalize_axis(axis)
    return block_idx(normalized_axis) % cluster_dim(normalized_axis)


def block_rank_in_cluster() -> ScalarValue | int:
    x = block_idx_in_cluster("x")
    y = block_idx_in_cluster("y")
    z = block_idx_in_cluster("z")
    dim_x = cluster_dim("x")
    dim_y = cluster_dim("y")
    return x + (dim_x * (y + (dim_y * z)))


def block_in_cluster_idx() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
    return (block_idx_in_cluster("x"), block_idx_in_cluster("y"), block_idx_in_cluster("z"))


def block_in_cluster_dim() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
    return (cluster_dim("x"), cluster_dim("y"), cluster_dim("z"))


def sync_warp(*, loc: Any | None = None, ip: Any | None = None) -> None:
    del loc
    del ip
    barrier(kind="warp")


def elect_one() -> ScalarValue | bool:
    return lane_id() == 0


def lane_idx() -> ScalarValue | int:
    return lane_id()


def get_dyn_smem_size() -> ScalarValue | int:
    try:
        builder = require_builder()
        return builder.constant(max(builder.launch.shared_mem_bytes, builder._dynamic_shared_mem_bytes), dtype="index")
    except CompilationError:
        return _runtime_state().launch.shared_mem_bytes


def make_warp_uniform(value: Any) -> Any:
    return value


def any_(value: Any) -> ScalarValue | bool:
    if isinstance(value, TensorValue):
        if value.spec.dtype != "i1":
            raise TypeError("any_ requires an i1 tensor")
        return value.reduce(ReductionOp.MAX, 0)
    if isinstance(value, RuntimeTensor):
        if value.dtype != "i1":
            raise TypeError("any_ requires an i1 tensor")
        return bool(value.reduce(ReductionOp.MAX, 0))
    if isinstance(value, ScalarValue):
        if value.spec.dtype != "i1":
            raise TypeError("any_ requires an i1 scalar")
        return value
    if isinstance(value, RuntimeScalar):
        if value.dtype != "i1":
            raise TypeError("any_ requires an i1 scalar")
        return bool(value)
    if isinstance(value, (tuple, list)):
        result = None
        for item in value:
            reduced = any_(item)
            result = reduced if result is None else (result | reduced)
        if result is None:
            return False
        return result
    return builtins.bool(value)


def all_(value: Any) -> ScalarValue | bool:
    if isinstance(value, TensorValue):
        if value.spec.dtype != "i1":
            raise TypeError("all_ requires an i1 tensor")
        return value.reduce(ReductionOp.MIN, 1)
    if isinstance(value, RuntimeTensor):
        if value.dtype != "i1":
            raise TypeError("all_ requires an i1 tensor")
        return bool(value.reduce(ReductionOp.MIN, 1))
    if isinstance(value, ScalarValue):
        if value.spec.dtype != "i1":
            raise TypeError("all_ requires an i1 scalar")
        return value
    if isinstance(value, RuntimeScalar):
        if value.dtype != "i1":
            raise TypeError("all_ requires an i1 scalar")
        return bool(value)
    if isinstance(value, (tuple, list)):
        result = None
        for item in value:
            reduced = all_(item)
            result = reduced if result is None else (result & reduced)
        if result is None:
            return True
        return result
    return builtins.bool(value)


def lane_coords(shape: tuple[int, ...]) -> tuple[ScalarValue, ...]:
    if not shape:
        raise ValueError("lane_coords requires a non-empty shape")
    if any(dim <= 0 for dim in shape):
        raise ValueError("lane_coords shape dimensions must be > 0")
    remaining = lane_id()
    coords: list[ScalarValue] = []
    for axis in range(len(shape) - 1):
        stride = prod(shape[axis + 1 :])
        coords.append(remaining // stride)
        remaining = remaining % stride
    coords.append(remaining)
    return tuple(coords)


def size(value: Any, *, mode: list[int] | tuple[int, ...] | None = None) -> int:
    shape = _shape_value(value)
    if mode is None:
        selected = shape
    else:
        selected_items = tuple(shape[axis] for axis in mode)
        selected = selected_items[0] if len(selected_items) == 1 else selected_items
    dims = _flatten_dims(selected)
    return prod(dims) if dims else 1


def _static_loop_value(value: ScalarValue | RuntimeScalar | int, *, name: str) -> int:
    if isinstance(value, RuntimeScalar):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, ScalarValue):
        builder = require_builder()
        operations_by_output = {
            output: operation
            for operation in builder.operations
            for output in operation.outputs
        }

        def resolve(scalar_name: str) -> int:
            operation = operations_by_output.get(scalar_name)
            if operation is None:
                raise CompilationError(
                    f"{name} only supports statically resolvable traced bounds; value '{scalar_name}' is dynamic"
                )
            if operation.op in {"constant", "tensor_dim"}:
                return int(operation.attrs["value"])
            if operation.op == "add":
                return resolve(operation.inputs[0]) + resolve(operation.inputs[1])
            if operation.op == "sub":
                return resolve(operation.inputs[0]) - resolve(operation.inputs[1])
            if operation.op == "mul":
                return resolve(operation.inputs[0]) * resolve(operation.inputs[1])
            if operation.op in {"div", "floordiv"}:
                return resolve(operation.inputs[0]) // resolve(operation.inputs[1])
            if operation.op == "mod":
                return resolve(operation.inputs[0]) % resolve(operation.inputs[1])
            if operation.op == "neg":
                return -resolve(operation.inputs[0])
            if operation.op == "cast":
                return resolve(operation.inputs[0])
            raise CompilationError(
                f"{name} only supports statically resolvable traced bounds; op '{operation.op}' is not static"
            )

        return resolve(value.name)
    raise TypeError(f"{name} expects integer or statically-known baybridge scalar bounds")


def _normalize_loop_range_args(args: tuple[Any, ...], *, name: str) -> tuple[int, int, int]:
    if not 1 <= len(args) <= 3:
        raise TypeError(f"{name} expects 1 to 3 positional arguments")
    if len(args) == 1:
        start = 0
        stop = _static_loop_value(args[0], name=name)
        step = 1
    elif len(args) == 2:
        start = _static_loop_value(args[0], name=name)
        stop = _static_loop_value(args[1], name=name)
        step = 1
    else:
        start = _static_loop_value(args[0], name=name)
        stop = _static_loop_value(args[1], name=name)
        step = _static_loop_value(args[2], name=name)
    if step == 0:
        raise ValueError(f"{name} step must not be zero")
    return start, stop, step


def _record_loop_hint(
    kind: str,
    *,
    start: int,
    stop: int,
    step: int,
    prefetch_stages: int | None,
    unroll_full: bool | None,
) -> None:
    try:
        builder = require_builder()
    except CompilationError:
        return
    hints = builder.metadata.setdefault("loop_hints", [])
    if isinstance(hints, list):
        hints.append(
            {
                "kind": kind,
                "start": start,
                "stop": stop,
                "step": step,
                "prefetch_stages": prefetch_stages,
                "unroll_full": unroll_full,
            }
        )


def dsl_range(
    *args: Any,
    prefetch_stages: int | None = None,
    unroll_full: bool | None = None,
    loc: Any | None = None,
    ip: Any | None = None,
):
    del loc
    del ip
    if prefetch_stages is not None and prefetch_stages < 0:
        raise ValueError("prefetch_stages must be >= 0")
    start, stop, step = _normalize_loop_range_args(args, name="range")
    _record_loop_hint(
        "range",
        start=start,
        stop=stop,
        step=step,
        prefetch_stages=prefetch_stages,
        unroll_full=unroll_full,
    )
    return builtins.range(start, stop, step)


def dsl_range_constexpr(*args: Any, loc: Any | None = None, ip: Any | None = None):
    del loc
    del ip
    start, stop, step = _normalize_loop_range_args(args, name="range_constexpr")
    _record_loop_hint(
        "range_constexpr",
        start=start,
        stop=stop,
        step=step,
        prefetch_stages=None,
        unroll_full=True,
    )
    return builtins.range(start, stop, step)


def product_each(value: Any) -> tuple[int, int, int]:
    shape = _shape_value(value)
    dims = _flatten_dims(shape)
    if len(dims) > 3:
        raise ValueError("product_each only supports up to 3 dimensions")
    return tuple(dims) + (1,) * (3 - len(dims))


def select(value: Any, *, mode: list[int] | tuple[int, ...]) -> Any:
    if not mode:
        raise ValueError("mode must be non-empty")
    if isinstance(value, Layout):
        return Layout(
            shape=tuple(value.shape[axis] for axis in mode),
            stride=tuple(value.stride[axis] for axis in mode),
            swizzle=value.swizzle,
        )
    shape = _shape_value(value)
    selected = tuple(shape[axis] for axis in mode)
    return selected


def make_layout_tv(thr_layout: Layout, val_layout: Layout) -> tuple[tuple[int, ...], ThreadValueLayout]:
    if len(thr_layout.shape) != len(val_layout.shape):
        raise ValueError("thread and value layouts must have the same rank")
    tiler = tuple(lhs * rhs for lhs, rhs in zip(thr_layout.shape, val_layout.shape))
    return tiler, ThreadValueLayout(thr_layout, val_layout)


def make_identity_tensor(shape: tuple[int, ...]) -> IdentityTensor:
    if not shape:
        raise ValueError("identity tensors require a non-empty shape")
    if any(dim <= 0 for dim in shape):
        raise ValueError("identity tensor dimensions must be > 0")
    return IdentityTensor(shape=shape)


def zipped_divide(
    value: TensorValue | RuntimeTensor | IdentityTensor | Layout,
    tiler: Any,
) -> TiledTensorView | HierarchicalLayout:
    if isinstance(value, Layout):
        tile_entries = _expand_tiler_entries(tiler, rank=len(value.shape))
        tile_shape = tuple(tile for tile in tile_entries if tile is not None)
        tile_stride = tuple(stride for stride, tile in zip(value.stride, tile_entries) if tile is not None)
        rest_shape = tuple(
            dim if tile is None else (dim + tile - 1) // tile
            for dim, tile in zip(value.shape, tile_entries)
        )
        rest_stride = tuple(
            stride if tile is None else tile * stride
            for stride, tile in zip(value.stride, tile_entries)
        )
        return HierarchicalLayout(
            shape=(
                _normalize_shape_tree(tile_shape),
                _normalize_shape_tree(rest_shape),
            ),
            stride=(
                _normalize_shape_tree(tile_stride),
                _normalize_shape_tree(rest_stride),
            ),
            swizzle=value.swizzle,
        )
    return TiledTensorView.create(value, tiler)


def _permute_axes(source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(source_shape) != len(target_shape):
        raise ValueError("layout remap rank must match the tiled tensor rest rank")
    used = [False] * len(source_shape)
    permutation: list[int] = []
    for dim in target_shape:
        match = None
        for axis, source_dim in enumerate(source_shape):
            if not used[axis] and source_dim == dim:
                match = axis
                break
        if match is None:
            raise ValueError("layout remap shape must be a permutation of the tiled tensor rest shape")
        used[match] = True
        permutation.append(match)
    return tuple(permutation)


def composition(value: Any, mapping: Any) -> Any:
    if isinstance(value, RuntimeTensor) and isinstance(mapping, ThreadValueLayout):
        return ThreadValueComposedView(value, mapping)
    if isinstance(value, (IdentityTileTensorValue, LocalCoordinateTensorValue)) and isinstance(mapping, ThreadValueLayout):
        return ThreadValueComposedCoordinateView(value, mapping)
    if isinstance(value, TensorValue) and isinstance(mapping, ThreadValueLayout):
        return ThreadValueComposedTensorView(value, mapping)
    if mapping is not None and isinstance(mapping, ThreadValueLayout):
        from .views import IdentityTileTensor

        if isinstance(value, (RuntimeTensor, IdentityTileTensor, LocalCoordinateTensor)):
            return ThreadValueComposedView(value, mapping)
    if isinstance(value, TiledTensorView) and isinstance(mapping, tuple) and len(mapping) == 2:
        tile_mapping, rest_mapping = mapping
        if tile_mapping is not None:
            raise UnsupportedOperationError(
                "baybridge.composition for tiled tensors currently only supports remapping the rest space"
            )
        if not isinstance(rest_mapping, Layout):
            raise TypeError("tiled tensor rest remaps require a baybridge.Layout")
        return value.remap_rest(_permute_axes(value.shape[1], rest_mapping.shape))
    raise UnsupportedOperationError(
        "baybridge.composition is only implemented for tiled tensor remaps of the form composition(t, (None, layout))"
    )


def elem_less(lhs: Any, rhs: Any) -> ScalarValue | bool:
    if isinstance(lhs, (tuple, list)) or isinstance(rhs, (tuple, list)):
        if not isinstance(lhs, (tuple, list)) or not isinstance(rhs, (tuple, list)):
            raise TypeError("elem_less requires both operands to be scalars or both to be tuples/lists")
        if len(lhs) != len(rhs):
            raise ValueError("elem_less tuple operands must have the same length")
        result = None
        for lhs_item, rhs_item in zip(lhs, rhs):
            comparison = elem_less(lhs_item, rhs_item)
            result = comparison if result is None else result & comparison
        if result is None:
            raise ValueError("elem_less requires non-empty tuple operands")
        return result
    if isinstance(lhs, ScalarValue) or isinstance(rhs, ScalarValue):
        lhs_scalar = lhs if isinstance(lhs, ScalarValue) else require_builder().constant(lhs)
        rhs_scalar = rhs if isinstance(rhs, ScalarValue) else require_builder().constant(rhs, dtype=lhs_scalar.spec.dtype)
        return lhs_scalar < rhs_scalar
    return lhs < rhs


def where(
    predicate: ScalarValue | TensorValue | RuntimeTensor | bool,
    true_value: TensorValue | RuntimeTensor | ScalarValue | int | float,
    false_value: TensorValue | RuntimeTensor | ScalarValue | int | float,
    *,
    loc: Any | None = None,
    ip: Any | None = None,
) -> TensorValue | RuntimeTensor | ScalarValue | int | float:
    del loc
    del ip
    if isinstance(predicate, ScalarValue):
        if predicate.spec.dtype != "i1":
            raise ValueError("where requires an i1 predicate")
        builder = require_builder()
        if isinstance(true_value, ScalarValue):
            true_scalar = true_value
        else:
            preferred_dtype = false_value.spec.dtype if isinstance(false_value, ScalarValue) else None
            true_scalar = builder.constant(true_value, dtype=preferred_dtype)
        if isinstance(false_value, ScalarValue):
            false_scalar = false_value
        else:
            false_scalar = builder.constant(false_value, dtype=true_scalar.spec.dtype)
        if true_scalar.spec.dtype != false_scalar.spec.dtype:
            raise ValueError("where requires the true and false branches to have the same dtype")
        return builder.emit_scalar(
            "select",
            predicate,
            true_scalar,
            false_scalar,
            spec=ScalarSpec(dtype=true_scalar.spec.dtype),
            name_hint="select",
        )
    if isinstance(predicate, TensorValue):
        if predicate.spec.dtype != "i1":
            raise ValueError("where requires an i1 predicate tensor")
        builder = require_builder()

        def _coerce_tensor_branch(
            value: TensorValue | ScalarValue | int | float,
            *,
            required_dtype: str | None,
        ) -> tuple[TensorValue | ScalarValue, str, bool]:
            if isinstance(value, TensorValue):
                tensor = value
                if tensor.spec.shape != predicate.spec.shape:
                    tensor = tensor.broadcast_to(predicate.spec.shape)
                if required_dtype is not None and tensor.spec.dtype != required_dtype:
                    raise ValueError(f"where tensor branch dtype must match {required_dtype}, got {tensor.spec.dtype}")
                return tensor, tensor.spec.dtype, True
            if required_dtype is None:
                if isinstance(value, ScalarValue):
                    scalar = value
                else:
                    scalar = builder.constant(value)
            else:
                if isinstance(value, ScalarValue):
                    scalar = value
                else:
                    scalar = builder.constant(value, dtype=required_dtype)
            if required_dtype is not None and scalar.spec.dtype != required_dtype:
                raise ValueError(f"where scalar branch dtype must match {required_dtype}, got {scalar.spec.dtype}")
            return scalar, scalar.spec.dtype, False

        branch_dtype: str | None = None
        if isinstance(true_value, TensorValue):
            branch_dtype = true_value.spec.dtype
        elif isinstance(false_value, TensorValue):
            branch_dtype = false_value.spec.dtype
        true_branch, true_dtype, true_is_tensor = _coerce_tensor_branch(true_value, required_dtype=branch_dtype)
        false_branch, false_dtype, false_is_tensor = _coerce_tensor_branch(false_value, required_dtype=true_dtype)
        if true_dtype != false_dtype:
            raise ValueError("where requires the true and false branches to have the same dtype")
        if not true_is_tensor and not false_is_tensor:
            raise TypeError("where with a traced tensor predicate requires at least one tensor branch")
        return builder.emit_tensor(
            "tensor_select",
            predicate,
            true_branch,
            false_branch,
            spec=TensorSpec(
                shape=predicate.spec.shape,
                dtype=true_dtype,
                address_space=AddressSpace.REGISTER,
            ),
            name_hint="tensor_select",
        )
    if isinstance(predicate, RuntimeTensor):
        if predicate.dtype != "i1":
            raise ValueError("where requires an i1 predicate tensor")
        true_tensor = true_value if isinstance(true_value, RuntimeTensor) else full(predicate.shape, true_value)
        false_tensor = false_value if isinstance(false_value, RuntimeTensor) else full(predicate.shape, false_value)
        if true_tensor.shape != predicate.shape or false_tensor.shape != predicate.shape:
            raise ValueError("where tensor branches must match the predicate tensor shape")
        result_dtype = true_tensor.dtype
        return RuntimeTensor(
            [
                true_tensor[index] if predicate[index] else false_tensor[index]
                for index in _iter_indices(predicate.shape)
            ],
            predicate.shape,
            dtype=result_dtype,
        )
    return true_value if predicate else false_value


def _resolve_layout_input(value: Layout | tuple[int, ...]) -> tuple[tuple[int, ...], Layout | None]:
    if isinstance(value, Layout):
        return value.shape, value
    if isinstance(value, tuple) and all(isinstance(item, int) for item in value):
        return value, None
    raise TypeError("expected a baybridge.Layout or a shape tuple")


def make_rmem_tensor(shape_or_layout: Layout | tuple[int, ...], dtype: str) -> TensorValue | RuntimeTensor:
    shape, layout = _resolve_layout_input(shape_or_layout)
    return make_tensor("rmem", shape, dtype, layout=layout, address_space=AddressSpace.REGISTER)


def make_fragment(shape_or_layout: Layout | tuple[int, ...], dtype: str) -> TensorValue | RuntimeTensor:
    return make_rmem_tensor(shape_or_layout, dtype)


def make_fragment_like(
    value: Any,
    dtype: str | None = None,
) -> TensorValue | RuntimeTensor:
    shape = tuple(value.shape)
    layout = value.layout if isinstance(getattr(value, "layout", None), Layout) else None
    element = dtype or value.element_type
    return make_tensor(
        "fragment",
        shape,
        element,
        layout=layout,
        address_space=AddressSpace.REGISTER,
    )


def make_rmem_tensor_like(
    value: Any,
    dtype: str | None = None,
) -> TensorValue | RuntimeTensor:
    return make_fragment_like(value, dtype=dtype)


def print_tensor(value: Any, verbose: bool = False) -> None:
    if isinstance(value, RuntimeTensor):
        if verbose:
            for index in _iter_indices(value.shape):
                print(f"{index}= {value[index]}")
            return
        print(value.tolist())
        return
    print(value)


def slice_(
    tensor: TensorValue | RuntimeTensor,
    coord: tuple[ScalarValue | int | None, ...],
) -> TensorValue | RuntimeTensor:
    if isinstance(tensor, RuntimeTensor):
        if len(coord) != tensor.ndim:
            raise ValueError(f"slice_ expects {tensor.ndim} coordinates for tensor rank {tensor.ndim}")
        kept_axes = [axis for axis, item in enumerate(coord) if item is None]
        if not kept_axes:
            raise ValueError("slice_ requires at least one None axis to keep")
        offset = tensor.offset
        for axis, item in enumerate(coord):
            if item is None:
                continue
            if not isinstance(item, int):
                raise TypeError("runtime slice_ coordinates must be integers or None")
            if item < 0 or item >= tensor.shape[axis]:
                raise ValueError(f"slice_ coordinate {item} is out of range for axis {axis} with extent {tensor.shape[axis]}")
            offset += item * tensor.stride[axis]
        return RuntimeTensor(
            tensor._storage,
            tuple(tensor.shape[axis] for axis in kept_axes),
            dtype=tensor.dtype,
            stride=tuple(tensor.stride[axis] for axis in kept_axes),
            offset=offset,
        )
    if len(coord) != len(tensor.spec.shape):
        raise ValueError(f"slice_ expects {len(tensor.spec.shape)} coordinates for tensor rank {len(tensor.spec.shape)}")
    kept_axes = [axis for axis, item in enumerate(coord) if item is None]
    if not kept_axes:
        raise ValueError("slice_ requires at least one None axis to keep")
    fixed_indices = []
    builder = require_builder()
    for axis, item in enumerate(coord):
        if item is None:
            continue
        if isinstance(item, ScalarValue):
            fixed_indices.append(item)
            continue
        if not isinstance(item, int):
            raise TypeError("traced slice_ coordinates must be baybridge scalars, integers, or None")
        if item < 0 or item >= tensor.spec.shape[axis]:
            raise ValueError(f"slice_ coordinate {item} is out of range for axis {axis} with extent {tensor.spec.shape[axis]}")
        fixed_indices.append(builder.constant(item))
    source_layout = tensor.spec.resolved_layout()
    sliced_spec = TensorSpec(
        shape=tuple(tensor.spec.shape[axis] for axis in kept_axes),
        dtype=tensor.spec.dtype,
        layout=Layout(
            shape=tuple(source_layout.shape[axis] for axis in kept_axes),
            stride=tuple(source_layout.stride[axis] for axis in kept_axes),
            swizzle=source_layout.swizzle,
        ),
        address_space=tensor.spec.address_space,
    )
    return builder.emit_tensor(
        "slice",
        tensor,
        *fixed_indices,
        spec=sliced_spec,
        attrs={"fixed_axes": [axis for axis, item in enumerate(coord) if item is not None], "kept_axes": kept_axes},
        name_hint=f"{tensor.name}_slice",
    )


def repeat_like(value: Any, like: Any) -> Any:
    if isinstance(like, (tuple, list)):
        return type(like)(repeat_like(value, item) for item in like)
    return value


def repeat_as_tuple(value: Any, n: int) -> tuple[Any, ...]:
    n = int(n)
    if n < 1:
        raise ValueError("repeat_as_tuple expects n >= 1")
    return tuple(value for _ in range(n))


def repeat(value: Any, n: int) -> Any:
    repeated = repeat_as_tuple(value, n)
    if n == 1:
        return repeated[0]
    return repeated


def tuple_cat(*tuples: Any) -> tuple[Any, ...]:
    flattened: list[Any] = []
    for value in tuples:
        if isinstance(value, tuple):
            flattened.extend(value)
        else:
            flattened.append(value)
    return tuple(flattened)


def transform_apply(*args: Any, f: Callable[..., Any], g: Callable[..., Any]) -> Any:
    if not args:
        raise TypeError("transform_apply expects at least one positional argument")
    if all(isinstance(value, tuple) for value in args):
        lengths = {len(value) for value in args}
        if len(lengths) != 1:
            raise ValueError("transform_apply tuple arguments must have the same length")
        transformed = [f(*(value[index] for value in args)) for index in range(len(args[0]))]
        return g(*transformed)
    transformed = [f(*args)]
    return g(*transformed)


def full(
    shape: tuple[int, ...],
    fill_value: Any,
    *,
    dtype: str = "f32",
) -> TensorValue | RuntimeTensor:
    try:
        builder = require_builder()
    except CompilationError:
        return runtime_full(shape, fill_value, dtype=dtype)
    tensor = builder.make_tensor(
        f"full_{builder._temp_index + 1}",
        TensorSpec(
            shape=tuple(int(dim) for dim in shape),
            dtype=str(dtype),
            address_space=AddressSpace.REGISTER,
        ),
    )
    tensor.fill(fill_value)
    return tensor


def empty_like(value: TensorValue | RuntimeTensor, dtype: str | None = None) -> TensorValue | RuntimeTensor:
    shape = tuple(value.shape)
    layout = value.layout if isinstance(getattr(value, "layout", None), Layout) else None
    if isinstance(value, RuntimeTensor):
        element_dtype = dtype or value.dtype
    else:
        element_dtype = dtype or value.spec.dtype
    return make_tensor(
        "empty_like",
        shape,
        element_dtype,
        layout=layout,
        address_space=AddressSpace.REGISTER,
    )


def full_like(value: TensorValue | RuntimeTensor, fill_value: Any, dtype: str | None = None) -> TensorValue | RuntimeTensor:
    target = empty_like(value, dtype=dtype)
    target.fill(fill_value)
    return target


def zeros_like(value: TensorValue | RuntimeTensor, dtype: str | None = None) -> TensorValue | RuntimeTensor:
    target_dtype = dtype or (value.dtype if isinstance(value, RuntimeTensor) else value.spec.dtype)
    return full_like(value, 0, dtype=target_dtype)


def ones_like(value: TensorValue | RuntimeTensor, dtype: str | None = None) -> TensorValue | RuntimeTensor:
    target_dtype = dtype or (value.dtype if isinstance(value, RuntimeTensor) else value.spec.dtype)
    return full_like(value, 1, dtype=target_dtype)


def assume(value: Any, **kwargs: Any) -> Any:
    del kwargs
    return value


def dim(tensor: TensorValue | RuntimeTensor, axis: int) -> ScalarValue | int:
    if isinstance(tensor, RuntimeTensor):
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(f"axis {axis} is out of range for tensor rank {tensor.ndim}")
        return tensor.shape[axis]
    try:
        if axis < 0 or axis >= len(tensor.spec.shape):
            raise ValueError(f"axis {axis} is out of range for tensor rank {len(tensor.spec.shape)}")
        return require_builder().emit_scalar(
            "tensor_dim",
            tensor,
            spec=ScalarSpec(dtype="index"),
            attrs={"axis": axis, "value": tensor.spec.shape[axis]},
            name_hint=f"{tensor.name}_dim_{axis}",
        )
    except CompilationError:
        runtime_tensor = _runtime_tensor(tensor)
        return runtime_tensor.shape[axis]


def _normalize_indices(index: ScalarValue | int | tuple[ScalarValue | int, ...]) -> tuple[ScalarValue, ...]:
    builder = require_builder()
    if isinstance(index, tuple):
        raw_indices = index
    else:
        raw_indices = (index,)
    normalized: list[ScalarValue] = []
    for item in raw_indices:
        if isinstance(item, ScalarValue):
            normalized.append(item)
        elif isinstance(item, int):
            normalized.append(builder.constant(item))
        else:
            raise TypeError("indices must be ScalarValue, int, or tuples of those values")
    return tuple(normalized)


def _normalize_offsets(
    offset: tuple[ScalarValue | int, ...],
    *,
    shape: tuple[int, ...],
    tile: tuple[int, ...],
) -> tuple[ScalarValue, ...]:
    builder = require_builder()
    if len(offset) != len(shape):
        raise ValueError(f"offset expects {len(shape)} values for tensor rank {len(shape)}")
    normalized: list[ScalarValue] = []
    for axis, item in enumerate(offset):
        if isinstance(item, ScalarValue):
            normalized.append(item)
            continue
        if not isinstance(item, int):
            raise TypeError("partition offsets must be ScalarValue or int")
        if item < 0:
            raise ValueError("partition offsets must be >= 0")
        if item + tile[axis] > shape[axis]:
            raise ValueError("static partition offsets plus tile dimensions must stay within the source tensor shape")
        normalized.append(builder.constant(item))
    return tuple(normalized)


def _normalize_predicate(predicate: ScalarValue | None) -> ScalarValue | None:
    if predicate is None:
        return None
    if predicate.spec.dtype != "i1":
        raise ValueError("predicate values must have dtype i1")
    return predicate


def load(
    tensor: TensorValue | RuntimeTensor,
    index: ScalarValue | int | tuple[ScalarValue | int, ...],
    *,
    predicate: ScalarValue | None = None,
    else_value: ScalarValue | int | float | None = None,
) -> ScalarValue | int | float:
    try:
        indices = _normalize_indices(index)
        if len(indices) != len(tensor.spec.shape):
            raise ValueError(f"load expects {len(tensor.spec.shape)} indices for tensor rank {len(tensor.spec.shape)}")
        normalized_predicate = _normalize_predicate(predicate)
        builder = require_builder()
        if normalized_predicate is None:
            return builder.emit_scalar(
                "load",
                tensor,
                *indices,
                spec=ScalarSpec(dtype=tensor.spec.dtype),
                attrs={"rank": len(indices)},
                name_hint=f"{tensor.name}_val",
            )
        fallback = else_value if else_value is not None else 0
        if isinstance(fallback, ScalarValue):
            fallback_value = fallback
        else:
            fallback_value = builder.constant(fallback, dtype=tensor.spec.dtype)
        if fallback_value.spec.dtype != tensor.spec.dtype:
            raise ValueError(
                f"else_value dtype mismatch: value has {fallback_value.spec.dtype}, tensor expects {tensor.spec.dtype}"
            )
        return builder.emit_scalar(
            "masked_load",
            tensor,
            *indices,
            normalized_predicate,
            fallback_value,
            spec=ScalarSpec(dtype=tensor.spec.dtype),
            attrs={"rank": len(indices)},
            name_hint=f"{tensor.name}_val",
        )
    except CompilationError:
        runtime_tensor = _runtime_tensor(tensor)
        normalized_indices = _normalize_runtime_indices(index, rank=runtime_tensor.ndim)
        if predicate is not None and not predicate:
            return else_value if else_value is not None else 0
        return runtime_tensor[normalized_indices]


def store(
    value: ScalarValue | int | float,
    tensor: TensorValue | RuntimeTensor,
    index: ScalarValue | int | tuple[ScalarValue | int, ...],
    *,
    predicate: ScalarValue | None = None,
) -> None:
    try:
        indices = _normalize_indices(index)
        if len(indices) != len(tensor.spec.shape):
            raise ValueError(f"store expects {len(tensor.spec.shape)} indices for tensor rank {len(tensor.spec.shape)}")
        if not isinstance(value, ScalarValue):
            raise TypeError("traced store values must be baybridge scalar expressions")
        if value.spec.dtype != tensor.spec.dtype:
            raise ValueError(f"store dtype mismatch: value has {value.spec.dtype}, tensor expects {tensor.spec.dtype}")
        normalized_predicate = _normalize_predicate(predicate)
        builder = require_builder()
        if normalized_predicate is None:
            builder.emit_void(
                "store",
                value,
                tensor,
                *indices,
                attrs={"rank": len(indices)},
            )
            return
        builder.emit_void(
            "masked_store",
            value,
            tensor,
            *indices,
            normalized_predicate,
            attrs={"rank": len(indices)},
        )
    except CompilationError:
        runtime_tensor = _runtime_tensor(tensor)
        normalized_indices = _normalize_runtime_indices(index, rank=runtime_tensor.ndim)
        if predicate is not None and not predicate:
            return
        runtime_tensor[normalized_indices] = value


def _layout_shape(layout: Any) -> tuple[int, ...]:
    if isinstance(layout, Layout):
        return tuple(int(dim) for dim in layout.shape)
    if hasattr(layout, "shape"):
        shape = getattr(layout, "shape")
        if isinstance(shape, tuple):
            return tuple(int(dim) for dim in shape)
        if isinstance(shape, list):
            return tuple(int(dim) for dim in shape)
    if isinstance(layout, tuple):
        return tuple(int(dim) for dim in layout)
    raise TypeError("tiled MMA layout inputs must provide a shape")


def _infer_universal_mma_tile(atom_layout: Any, permutation_mnk: Any) -> tuple[int, int, int]:
    if isinstance(permutation_mnk, tuple) and len(permutation_mnk) == 3:
        resolved: list[int] = []
        for item in permutation_mnk:
            if item is None:
                resolved.append(1)
            elif isinstance(item, int):
                resolved.append(int(item))
            else:
                shape = _layout_shape(item)
                resolved.append(prod(shape) if shape else 1)
        return tuple(resolved)  # type: ignore[return-value]
    if atom_layout is not None:
        shape = _layout_shape(atom_layout)
        if len(shape) >= 2:
            m = int(shape[0])
            n = int(shape[1])
            k = int(shape[2]) if len(shape) >= 3 else 1
            return (m, n, k)
    return (1, 1, 1)


def _compatible_mma_descriptor(
    *,
    tile: tuple[int, int, int],
    operand_dtype: str,
    accumulator_dtype: str,
    wave_size: int = 64,
    lane_shape: tuple[int, int] = (4, 16),
    variant_name: str = "compatible_mma",
) -> CompatibleMmaDescriptor:
    return CompatibleMmaDescriptor(
        tile=tuple(int(dim) for dim in tile),
        operand_dtype=resolve_element_type_name(operand_dtype),
        accumulator_dtype=resolve_element_type_name(accumulator_dtype),
        wave_size=int(wave_size),
        lane_shape=tuple(int(dim) for dim in lane_shape),
        variant_name=variant_name,
    )


def _resolve_descriptor_or_compatible(
    *,
    tile: tuple[int, int, int],
    operand_dtype: str,
    accumulator_dtype: str,
    wave_size: int = 64,
    lane_shape: tuple[int, int] = (4, 16),
    variant_name: str = "compatible_mma",
) -> MFMADescriptor | CompatibleMmaDescriptor:
    operand_dtype = resolve_element_type_name(operand_dtype)
    accumulator_dtype = resolve_element_type_name(accumulator_dtype)
    try:
        return resolve_mfma_descriptor(tuple(tile), operand_dtype, accumulator_dtype, wave_size=wave_size)
    except ValueError:
        return _compatible_mma_descriptor(
            tile=tile,
            operand_dtype=operand_dtype,
            accumulator_dtype=accumulator_dtype,
            wave_size=wave_size,
            lane_shape=lane_shape,
            variant_name=variant_name,
        )


def _is_cpasync_copy_op(op: Any) -> bool:
    return isinstance(
        op,
        (
            CpAsyncCopyG2SOp,
            CpAsyncCopyBulkTensorTileG2SOp,
            CpAsyncCopyBulkTensorTileG2SMulticastOp,
        ),
    )


def _tensor_element_type_name(tensor: Any) -> str:
    if isinstance(tensor, TensorValue):
        return tensor.spec.dtype
    if isinstance(tensor, RuntimeTensor):
        return tensor.dtype
    value = getattr(tensor, "element_type", None)
    if value is not None:
        return resolve_element_type_name(value)
    raise TypeError("tensor inputs must expose an element type")


def make_copy_atom(op: Any, value_type: Any, *, num_bits_per_copy: int | None = None) -> CopyAtom:
    resolved_type = resolve_element_type_name(value_type)
    if num_bits_per_copy is None:
        num_bits_per_copy = element_type(resolved_type).width
    return CopyAtom(op=op, value_type=resolved_type, num_bits_per_copy=num_bits_per_copy)


def make_atom(op: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(op, (MmaUniversalOp, WarpMmaF16BF16Op, Tcgen05MmaF16BF16Op, Tcgen05MmaI8Op)) or (
        hasattr(op, "tile") and hasattr(op, "dtype")
    ):
        return make_mma_atom(op, *args, **kwargs)
    if args:
        value_type = args[0]
        return make_copy_atom(op, value_type, **kwargs)
    value_type = kwargs.pop("value_type", None)
    if value_type is None:
        raise TypeError("make_atom expects either an MMA-like op or a copy op plus value_type")
    return make_copy_atom(op, value_type, **kwargs)


def make_mma_atom(
    op: Any,
    atom_layout: Any = None,
    *,
    permutation_mnk: Any = None,
) -> MFMADescriptor | CompatibleMmaDescriptor:
    normalized_atom_layout = atom_layout if isinstance(atom_layout, Layout) or atom_layout is None else make_layout(atom_layout)
    return _resolve_tiled_mma_descriptor(op, atom_layout=normalized_atom_layout, permutation_mnk=permutation_mnk)


def make_tiled_copy_tv(atom: CopyAtom, thr_layout: Layout, val_layout: Layout) -> TiledCopyTV:
    if not isinstance(atom, CopyAtom):
        raise TypeError("make_tiled_copy_tv expects a baybridge CopyAtom")
    return TiledCopyTV(atom=atom, thread_layout=thr_layout, value_layout=val_layout)


def make_tiled_copy(atom: CopyAtom, thr_layout: Layout, val_layout: Layout) -> TiledCopyTV:
    return make_tiled_copy_tv(atom, thr_layout, val_layout)


def make_cotiled_copy(atom: CopyAtom, thr_layout: Layout, val_layout: Layout) -> TiledCopyTV:
    return make_tiled_copy_tv(atom, thr_layout, val_layout)


def _resolve_tiled_mma_descriptor(
    op: Any,
    atom_layout: Any = None,
    permutation_mnk: Any = None,
) -> MFMADescriptor | CompatibleMmaDescriptor:
    if isinstance(op, MFMADescriptor):
        return op
    if isinstance(op, CompatibleMmaDescriptor):
        return op
    if isinstance(op, MmaUniversalOp):
        tile = tuple(op.tile) if op.tile is not None else _infer_universal_mma_tile(atom_layout, permutation_mnk)
        return _compatible_mma_descriptor(
            tile=tile,
            operand_dtype=op.accumulator_dtype,
            accumulator_dtype=op.accumulator_dtype,
            wave_size=op.wave_size,
            lane_shape=(4, 4),
            variant_name="universal_mma",
        )
    if isinstance(op, WarpMmaF16BF16Op):
        return _resolve_descriptor_or_compatible(
            tile=tuple(op.tile),
            operand_dtype=op.operand_dtype,
            accumulator_dtype=op.accumulator_dtype,
            wave_size=op.wave_size,
            lane_shape=op.lane_shape,
            variant_name="warp_mma_f16bf16",
        )
    if isinstance(op, Tcgen05MmaF16BF16Op):
        return _resolve_descriptor_or_compatible(
            tile=tuple(op.tile),
            operand_dtype=op.operand_dtype,
            accumulator_dtype=op.accumulator_dtype,
            wave_size=op.wave_size,
            lane_shape=op.lane_shape,
            variant_name="tcgen05_mma_f16bf16",
        )
    if isinstance(op, Tcgen05MmaTF32Op):
        return _resolve_descriptor_or_compatible(
            tile=tuple(op.tile),
            operand_dtype=op.operand_dtype,
            accumulator_dtype=op.accumulator_dtype,
            wave_size=op.wave_size,
            lane_shape=op.lane_shape,
            variant_name="tcgen05_mma_tf32",
        )
    if isinstance(op, Tcgen05MmaI8Op):
        return _resolve_descriptor_or_compatible(
            tile=tuple(op.tile),
            operand_dtype=op.operand_dtype,
            accumulator_dtype=op.accumulator_dtype,
            wave_size=op.wave_size,
            lane_shape=op.lane_shape,
            variant_name="tcgen05_mma_i8",
        )
    if hasattr(op, "variant_name") and hasattr(op, "tile") and hasattr(op, "operand_dtype") and hasattr(op, "accumulator_dtype"):
        wave_size = int(getattr(op, "wave_size", 64))
        lane_shape = tuple(getattr(op, "lane_shape", (4, 16)))
        return _resolve_descriptor_or_compatible(
            tile=tuple(op.tile),
            operand_dtype=str(op.operand_dtype),
            accumulator_dtype=str(op.accumulator_dtype),
            wave_size=wave_size,
            lane_shape=lane_shape,
            variant_name=str(getattr(op, "variant_name", "compatible_mma")),
        )
    if hasattr(op, "tile") and hasattr(op, "dtype"):
        operand_dtype = resolve_element_type_name(op.dtype)
        accumulator_dtype = resolve_element_type_name(getattr(op, "accumulator_dtype", operand_dtype))
        wave_size = int(getattr(op, "wave_size", 64))
        lane_shape = tuple(getattr(op, "lane_shape", (4, 16)))
        return _resolve_descriptor_or_compatible(
            tile=tuple(op.tile),
            operand_dtype=operand_dtype,
            accumulator_dtype=accumulator_dtype,
            wave_size=wave_size,
            lane_shape=lane_shape,
            variant_name=str(getattr(op, "variant_name", "compatible_mma")),
        )
    raise TypeError("make_tiled_mma expects an MFMA descriptor or a descriptor-like object with tile/dtype metadata")


def make_tiled_mma(
    op: Any,
    atom_layout: Any = None,
    *,
    permutation_mnk: Any = None,
    axes: tuple[str, str] = ("x", "y"),
) -> TiledMma:
    normalized_atom_layout = atom_layout if isinstance(atom_layout, Layout) or atom_layout is None else make_layout(atom_layout)
    descriptor = _resolve_tiled_mma_descriptor(op, atom_layout=normalized_atom_layout, permutation_mnk=permutation_mnk)
    return TiledMma(
        descriptor=descriptor,
        axes=axes,
        atom_layout=normalized_atom_layout,
        permutation_mnk=permutation_mnk,
    )


def _make_tiled_copy_from_mma(atom: CopyAtom, tiled_mma: TiledMma, role: str) -> TiledCopyTV:
    operand_shape = tiled_mma.descriptor.operand_shape(role)
    lane_shape = tiled_mma.descriptor.lane_shape
    if len(operand_shape) != 2 or len(lane_shape) != 2:
        raise ValueError("tiled copy helpers expect two-dimensional MFMA operand shapes")
    thread_shape = (
        min(lane_shape[0], operand_shape[0]),
        min(lane_shape[1], operand_shape[1]),
    )
    thread_layout = Layout.ordered(thread_shape, order=(1, 0))
    value_layout = Layout.ordered(
        (
            ceil_div(operand_shape[0], thread_shape[0]),
            ceil_div(operand_shape[1], thread_shape[1]),
        ),
        order=(1, 0),
    )
    return make_tiled_copy_tv(atom, thread_layout, value_layout)


def make_tiled_copy_A(atom: CopyAtom, tiled_mma: TiledMma) -> TiledCopyTV:
    return _make_tiled_copy_from_mma(atom, tiled_mma, "a")


def make_tiled_copy_B(atom: CopyAtom, tiled_mma: TiledMma) -> TiledCopyTV:
    return _make_tiled_copy_from_mma(atom, tiled_mma, "b")


def make_tiled_copy_C(atom: CopyAtom, tiled_mma: TiledMma) -> TiledCopyTV:
    return _make_tiled_copy_from_mma(atom, tiled_mma, "acc")


def make_tiled_copy_S(atom: CopyAtom, tiled_mma: TiledMma) -> TiledCopyTV:
    return make_tiled_copy_A(atom, tiled_mma)


def make_tiled_copy_D(atom: CopyAtom, tiled_mma: TiledMma) -> TiledCopyTV:
    return make_tiled_copy_C(atom, tiled_mma)


def make_tiled_copy_C_atom(op: Any, value_type: Any, *, num_bits_per_copy: int | None = None) -> CopyAtom:
    return make_copy_atom(op, value_type, num_bits_per_copy=num_bits_per_copy)


def _normalize_predicate_fragment(
    predicate: TensorValue | RuntimeTensor | None,
    *,
    shape: tuple[int, ...],
) -> TensorValue | RuntimeTensor | None:
    if predicate is None:
        return None
    if isinstance(predicate, TensorValue):
        if predicate.spec.dtype != "i1":
            raise ValueError("copy predicates must have dtype i1")
        if predicate.spec.shape != shape:
            raise ValueError(f"copy predicates must have shape {shape}, got {predicate.spec.shape}")
        return predicate
    if isinstance(predicate, RuntimeTensor):
        if predicate.dtype != "i1":
            raise ValueError("copy predicates must have dtype i1")
        if predicate.shape != shape:
            raise ValueError(f"copy predicates must have shape {shape}, got {predicate.shape}")
        return predicate
    raise TypeError("copy predicates must be baybridge tensors")


def _copy_atom_attrs(copy_atom: CopyAtom | None) -> dict[str, Any]:
    if copy_atom is None:
        return {}
    attrs: dict[str, Any] = {}
    op = copy_atom.op
    if isinstance(op, CpAsyncCopyBulkTensorTileS2GOp):
        attrs["copy_variant"] = "cpasync_tma_s2g"
        attrs["cta_group"] = op.cta_group
    elif isinstance(op, CpAsyncCopyReduceBulkTensorTileS2GOp):
        attrs["copy_variant"] = "cpasync_tma_s2g_reduce"
        attrs["cta_group"] = op.cta_group
        attrs["reduction"] = op.reduction_kind
    elif isinstance(op, CpAsyncCopyBulkTensorTileG2SMulticastOp):
        attrs["copy_variant"] = "cpasync_tma_g2s_multicast"
        attrs["cta_group"] = op.cta_group
        attrs["num_multicast"] = copy_atom.get("num_multicast", 1)
    elif isinstance(op, CpAsyncCopyBulkTensorTileG2SOp):
        attrs["copy_variant"] = "cpasync_tma_g2s"
        attrs["cta_group"] = op.cta_group
    elif isinstance(op, CpAsyncCopyDsmemStoreOp):
        attrs["copy_variant"] = "cpasync_dsmem_store"
    return attrs


def copy(
    *args: Any,
    vector_bytes: int | None = None,
    pred: TensorValue | RuntimeTensor | None = None,
    tma_bar_ptr: Any | None = None,
) -> None:
    copy_atom = None
    if len(args) == 2:
        src, dst = args
    elif len(args) == 3 and isinstance(args[0], CopyAtom):
        copy_atom, src, dst = args
    else:
        raise TypeError("copy expects (src, dst) or (copy_atom, src, dst)")

    if isinstance(src, ThreadFragmentTensorValue) and isinstance(dst, TensorValue):
        predicate = _normalize_predicate_fragment(pred, shape=src.shape)
        dst.store(src.load(predicate=predicate, else_value=0))
        return
    if isinstance(src, ThreadFragmentView) and isinstance(dst, RuntimeTensor):
        predicate = _normalize_predicate_fragment(pred, shape=src.shape)
        dst.store(src.load(predicate=predicate, else_value=0))
        return
    if isinstance(dst, ThreadFragmentTensorValue) and isinstance(src, TensorValue):
        predicate = _normalize_predicate_fragment(pred, shape=dst.shape)
        dst.store(src, predicate=predicate if isinstance(predicate, TensorValue) else None)
        return
    if isinstance(dst, ThreadFragmentView) and isinstance(src, RuntimeTensor):
        predicate = _normalize_predicate_fragment(pred, shape=dst.shape)
        dst.store(src, predicate=predicate if isinstance(predicate, RuntimeTensor) else None)
        return

    if copy_atom is not None and vector_bytes is None:
        vector_bytes = copy_atom.vector_bytes

    if pred is not None:
        raise UnsupportedOperationError("predicated copy is only implemented for baybridge thread fragments today")

    copy_attrs = {"vector_bytes": vector_bytes, **_copy_atom_attrs(copy_atom)}

    if copy_atom is not None and _is_cpasync_copy_op(copy_atom.op):
        del tma_bar_ptr
        copy_async(src, dst, vector_bytes=vector_bytes)
        return

    if copy_atom is not None and isinstance(copy_atom.op, CpAsyncCopyReduceBulkTensorTileS2GOp):
        reduction_kind = copy_atom.op.reduction_kind
        try:
            require_builder().emit_void("copy_reduce", src, dst, attrs=copy_attrs)
        except CompilationError:
            del tma_bar_ptr
            _copy_reduce_runtime_tensor(_runtime_tensor(src), _runtime_tensor(dst), reduction_kind)
        return

    try:
        require_builder().emit_void("copy", src, dst, attrs=copy_attrs)
    except CompilationError:
        del vector_bytes
        del tma_bar_ptr
        _copy_runtime_tensor(_runtime_tensor(src), _runtime_tensor(dst))


def basic_copy(src: Any, dst: Any) -> None:
    copy(src, dst)


def basic_copy_if(predicate: ScalarValue | RuntimeScalar | bool, src: Any, dst: Any) -> None:
    if isinstance(predicate, ScalarValue):
        raise UnsupportedOperationError(
            "basic_copy_if with a traced predicate is not implemented yet; use predicated stores or runtime execution"
        )
    if isinstance(predicate, RuntimeScalar):
        should_copy = bool(predicate)
    else:
        should_copy = bool(predicate)
    if should_copy:
        copy(src, dst)


def autovec_copy(src: Any, dst: Any, *, vector_bytes: int | None = None) -> None:
    copy(src, dst, vector_bytes=vector_bytes)


def prefetch(value: Any) -> None:
    del value


def copy_async(
    src: TensorValue | RuntimeTensor,
    dst: TensorValue | RuntimeTensor,
    *,
    vector_bytes: int | None = None,
    stages: int = 1,
) -> None:
    if stages <= 0:
        raise ValueError("stages must be > 0")
    if isinstance(src, RuntimeTensor) or isinstance(dst, RuntimeTensor):
        copy(src, dst, vector_bytes=vector_bytes)
        return
    try:
        if src.spec.address_space.value != "global":
            raise ValueError("copy_async currently requires the source tensor to be in global memory")
        if dst.spec.address_space.value != "shared":
            raise ValueError("copy_async currently requires the destination tensor to be in shared memory")
        require_builder().emit_void(
            "copy_async",
            src,
            dst,
            attrs={"vector_bytes": vector_bytes, "stages": stages},
        )
    except CompilationError:
        copy(src, dst, vector_bytes=vector_bytes)


def commit_group(*, group: str = "default") -> None:
    try:
        require_builder().emit_void("commit_group", attrs={"group": group})
    except CompilationError:
        del group


def wait_group(*, count: int = 0, group: str = "default") -> None:
    if count < 0:
        raise ValueError("count must be >= 0")
    try:
        require_builder().emit_void("wait_group", attrs={"count": count, "group": group})
    except CompilationError:
        del count
        del group


def barrier(*, kind: str = "block", **attrs: Any) -> None:
    if kind not in {"block", "grid", "warp"}:
        raise ValueError("barrier kind must be 'block', 'grid', or 'warp'")
    try:
        payload = {"kind": kind}
        payload.update(attrs)
        require_builder().emit_void("barrier", attrs=payload)
    except CompilationError:
        if kind == "grid":
            raise UnsupportedOperationError(
                "baybridge grid-wide barriers require a compiled cooperative launch"
            ) from None
        del kind
        del attrs


def domain_offset(
    offset: tuple[ScalarValue | int, ...],
    tensor: TensorValue | RuntimeTensor | IdentityTileTensor | IdentityTileTensorValue,
) -> TensorValue | RuntimeTensor | IdentityTileTensor | IdentityTileTensorValue:
    if isinstance(tensor, RuntimeTensor):
        if len(offset) != tensor.ndim:
            raise ValueError(f"domain_offset expects {tensor.ndim} offsets for tensor rank {tensor.ndim}")
        linear_offset = tensor.offset
        for axis, item in enumerate(offset):
            if not isinstance(item, int):
                raise TypeError("runtime domain_offset currently requires integer offsets")
            linear_offset += item * tensor.stride[axis]
        return RuntimeTensor(
            tensor._storage,
            tensor.shape,
            dtype=tensor.dtype,
            stride=tensor.stride,
            offset=linear_offset,
        )
    if isinstance(tensor, IdentityTileTensor):
        if len(offset) != tensor.ndim:
            raise ValueError(f"domain_offset expects {tensor.ndim} offsets for tensor rank {tensor.ndim}")
        if any(not isinstance(item, int) for item in offset):
            raise TypeError("runtime coordinate domain_offset currently requires integer offsets")
        return IdentityTileTensor(tensor.shape, tuple(base + delta for base, delta in zip(tensor.offset, offset)))
    if isinstance(tensor, IdentityTileTensorValue):
        if len(offset) != tensor.ndim:
            raise ValueError(f"domain_offset expects {tensor.ndim} offsets for tensor rank {tensor.ndim}")
        return IdentityTileTensorValue(tensor.shape, tuple(base + delta for base, delta in zip(tensor.offset, offset)))
    if len(offset) != len(tensor.spec.shape):
        raise ValueError(f"domain_offset expects {len(tensor.spec.shape)} offsets for tensor rank {len(tensor.spec.shape)}")
    builder = require_builder()
    normalized = []
    for item in offset:
        if isinstance(item, ScalarValue):
            normalized.append(item)
        elif isinstance(item, int):
            normalized.append(builder.constant(item))
        else:
            raise TypeError("traced domain_offset expects baybridge scalars or integers")
    return builder.emit_tensor(
        "domain_offset",
        tensor,
        *normalized,
        spec=tensor.spec,
        attrs={"offset_rank": len(normalized)},
        name_hint=f"{tensor.name}_offset",
    )


def partition(
    tensor: TensorValue | RuntimeTensor,
    tile: tuple[int, ...],
    *,
    offset: tuple[ScalarValue | int, ...] | None = None,
    policy: str = "blocked",
) -> TensorValue | RuntimeTensor:
    if isinstance(tensor, RuntimeTensor):
        if len(tile) != tensor.ndim:
            raise ValueError(f"partition expects a tile rank of {tensor.ndim}")
        if any(step <= 0 for step in tile):
            raise ValueError("partition tile dimensions must be > 0")
        if any(step > dim for step, dim in zip(tile, tensor.shape)):
            raise ValueError("partition tile dimensions cannot exceed the source tensor shape")
        raw_offset = offset or tuple(0 for _ in tile)
        if len(raw_offset) != tensor.ndim:
            raise ValueError(f"offset expects {tensor.ndim} values for tensor rank {tensor.ndim}")
        normalized_offset: list[int] = []
        for axis, item in enumerate(raw_offset):
            if not isinstance(item, int):
                raise TypeError("runtime partition offsets must be integers")
            if item < 0:
                raise ValueError("partition offsets must be >= 0")
            if item + tile[axis] > tensor.shape[axis]:
                raise ValueError("static partition offsets plus tile dimensions must stay within the source tensor shape")
            normalized_offset.append(item)
        del policy
        return tensor.view(shape=tile, offset_elements=tuple(normalized_offset))
    try:
        if len(tile) != len(tensor.spec.shape):
            raise ValueError(f"partition expects a tile rank of {len(tensor.spec.shape)}")
        if any(step <= 0 for step in tile):
            raise ValueError("partition tile dimensions must be > 0")
        if any(step > dim for step, dim in zip(tile, tensor.spec.shape)):
            raise ValueError("partition tile dimensions cannot exceed the source tensor shape")
        offsets = _normalize_offsets(offset, shape=tensor.spec.shape, tile=tile) if offset is not None else ()
        source_layout = tensor.spec.resolved_layout()
        tiled_layout = Layout(shape=tile, stride=source_layout.stride, swizzle=source_layout.swizzle)
        tiled_spec = TensorSpec(
            shape=tile,
            dtype=tensor.spec.dtype,
            layout=tiled_layout,
            address_space=tensor.spec.address_space,
        )
        return require_builder().emit_tensor(
            "partition",
            tensor,
            *offsets,
            spec=tiled_spec,
            attrs={"tile": list(tile), "policy": policy},
            name_hint=f"{tensor.name}_part",
        )
    except CompilationError:
        runtime_tensor = _runtime_tensor(tensor)
        return runtime_tensor


def partition_program(
    tensor: TensorValue,
    tile: tuple[int, ...],
    *,
    axes: tuple[str, ...] | None = None,
    policy: str = "blocked",
) -> TensorValue:
    resolved_axes = _resolve_axes(len(tile), axes)
    offsets = tuple(program_id(axis) * step for axis, step in zip(resolved_axes, tile))
    return partition(tensor, tile, offset=offsets, policy=policy)


def partition_thread(
    tensor: TensorValue,
    tile: tuple[int, ...],
    *,
    axes: tuple[str, ...] | None = None,
    policy: str = "blocked",
) -> TensorValue:
    resolved_axes = _resolve_axes(len(tile), axes)
    offsets = tuple(thread_idx(axis) * step for axis, step in zip(resolved_axes, tile))
    return partition(tensor, tile, offset=offsets, policy=policy)


def partition_wave(
    tensor: TensorValue,
    tile: tuple[int, ...],
    *,
    axes: tuple[str, ...] | None = None,
    wave_size: int = 64,
    policy: str = "blocked",
) -> TensorValue:
    resolved_axes = _resolve_axes(len(tile), axes)
    offsets = tuple(wave_id(axis=axis, wave_size=wave_size) * step for axis, step in zip(resolved_axes, tile))
    return partition(tensor, tile, offset=offsets, policy=policy)


def local_partition(
    tensor: TensorValue | RuntimeTensor | IdentityTensor | TiledTensorView,
    tiler: ThreadValueLayout | TiledCopyTV | Layout | tuple[int, ...] | tuple[tuple[int, ...], ThreadValueLayout],
    index: ScalarValue | int | tuple[ScalarValue | int, ...],
    proj: Any = 1,
) -> Any:
    if isinstance(tiler, TiledCopyTV):
        tile_shape = tiler.tiler
        tv_layout = tiler.tv_layout
    elif isinstance(tiler, ThreadValueLayout):
        tile_shape = tiler.tile_shape
        tv_layout = tiler
    elif (
        isinstance(tiler, tuple)
        and len(tiler) == 2
        and isinstance(tiler[0], tuple)
        and isinstance(tiler[1], ThreadValueLayout)
    ):
        tile_shape = tuple(int(dim) for dim in tiler[0])
        tv_layout = tiler[1]
    elif isinstance(tiler, Layout):
        tile_shape = tiler.shape
        tv_layout = None
    elif isinstance(tiler, tuple):
        tile_shape = tuple(int(dim) for dim in tiler)
        tv_layout = None
    else:
        raise TypeError("local_partition expects a baybridge tiler shape/layout or thread-value partitioner")
    if proj not in (1, None):
        raise UnsupportedOperationError("baybridge.local_partition currently supports proj=1 only")
    tiled = tensor if isinstance(tensor, TiledTensorView) else zipped_divide(tensor, tile_shape)
    if tv_layout is not None:
        target = tiled[(None, tuple(0 for _ in tiled.shape[1]))]
        return composition(target, tv_layout)[(index, None)]
    return tiled[(None, index)]


def gemm(
    a: TensorValue | RuntimeTensor,
    b: TensorValue | RuntimeTensor,
    c: TensorValue | RuntimeTensor,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> Any:
    if isinstance(a, RuntimeTensor) and isinstance(b, RuntimeTensor) and isinstance(c, RuntimeTensor):
        if any(is_storage_only_dtype(tensor.dtype) for tensor in (a, b, c)):
            raise TypeError(
                "runtime baybridge.gemm does not support storage-only operand dtypes; compile the kernel for execution"
            )
        if len(a.shape) != 2 or len(b.shape) != 2 or len(c.shape) != 2:
            raise ValueError("runtime baybridge.gemm currently requires rank-2 tensors")
        if transpose_a:
            k, m = a.shape
        else:
            m, k = a.shape
        if transpose_b:
            n, kb = b.shape
        else:
            kb, n = b.shape
        if kb != k or c.shape != (m, n):
            raise ValueError(f"gemm shape mismatch: {a.shape} x {b.shape} -> {c.shape}")
        for row in range(m):
            for col in range(n):
                acc = 0.0 if c.dtype.startswith("f") else 0
                for kk in range(k):
                    lhs = a[kk, row] if transpose_a else a[row, kk]
                    rhs = b[col, kk] if transpose_b else b[kk, col]
                    acc += lhs * rhs
                c[row, col] = acc
        return c
    if not isinstance(a, TensorValue) or not isinstance(b, TensorValue) or not isinstance(c, TensorValue):
        raise TypeError("baybridge.gemm expects all arguments to be runtime tensors or all to be traced tensors")
    if len(a.spec.shape) != 2 or len(b.spec.shape) != 2 or len(c.spec.shape) != 2:
        raise ValueError("traced baybridge.gemm currently requires rank-2 tensors")
    if transpose_a:
        k, m = a.spec.shape
    else:
        m, k = a.spec.shape
    if transpose_b:
        n, kb = b.spec.shape
    else:
        kb, n = b.spec.shape
    if kb != k or c.spec.shape != (m, n):
        raise ValueError(f"gemm shape mismatch: {a.spec.shape} x {b.spec.shape} -> {c.spec.shape}")
    return mma(
        a,
        b,
        c,
        tile=(m, n, k),
        accumulator_dtype=c.spec.dtype,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


def _runtime_index(coords: tuple[int, ...]) -> int | tuple[int, ...]:
    return coords[0] if len(coords) == 1 else coords


def _runtime_prefix_indices(shape: tuple[int, ...]):
    if not shape:
        yield ()
        return
    yield from product(*(range(dim) for dim in shape))


def layernorm(
    x: TensorValue | RuntimeTensor,
    residual: TensorValue | RuntimeTensor,
    out: TensorValue | RuntimeTensor,
    out_resid: TensorValue | RuntimeTensor,
    weight: TensorValue | RuntimeTensor,
    bias: TensorValue | RuntimeTensor,
    *,
    epsilon: float = 1e-5,
) -> Any:
    if all(isinstance(value, RuntimeTensor) for value in (x, residual, out, out_resid, weight, bias)):
        if x.shape != residual.shape or x.shape != out.shape or x.shape != out_resid.shape:
            raise ValueError("baybridge.layernorm requires x, residual, out, and out_resid to have matching shapes")
        if len(x.shape) < 1:
            raise ValueError("baybridge.layernorm requires rank >= 1 tensors")
        hidden = x.shape[-1]
        if weight.shape not in {(hidden,), x.shape}:
            raise ValueError(
                f"baybridge.layernorm expects weight shape {(hidden,)} or {x.shape}, got {weight.shape}"
            )
        if bias.shape not in {(hidden,), x.shape}:
            raise ValueError(
                f"baybridge.layernorm expects bias shape {(hidden,)} or {x.shape}, got {bias.shape}"
            )
        if len(weight.shape) != 1 and weight.shape != x.shape:
            raise ValueError("baybridge.layernorm currently supports only 1D or fully-expanded weight tensors")
        if len(bias.shape) != 1 and bias.shape != x.shape:
            raise ValueError("baybridge.layernorm currently supports only 1D or fully-expanded bias tensors")
        for prefix in _runtime_prefix_indices(x.shape[:-1]):
            mixed_values: list[float] = []
            for lane in range(hidden):
                index = _runtime_index(prefix + (lane,))
                mixed = float(x[index]) + float(residual[index])
                mixed_values.append(mixed)
                out_resid[index] = mixed
            mean = sum(mixed_values) / float(hidden)
            variance = sum((value - mean) * (value - mean) for value in mixed_values) / float(hidden)
            denom = py_math.sqrt(variance + float(epsilon))
            for lane, mixed in enumerate(mixed_values):
                index = _runtime_index(prefix + (lane,))
                weight_value = float(weight[lane] if len(weight.shape) == 1 else weight[index])
                bias_value = float(bias[lane] if len(bias.shape) == 1 else bias[index])
                out[index] = ((mixed - mean) / denom) * weight_value + bias_value
        return out
    if not all(isinstance(value, TensorValue) for value in (x, residual, out, out_resid, weight, bias)):
        raise TypeError("baybridge.layernorm expects all arguments to be runtime tensors or all to be traced tensors")
    if x.spec.shape != residual.spec.shape or x.spec.shape != out.spec.shape or x.spec.shape != out_resid.spec.shape:
        raise ValueError("baybridge.layernorm requires x, residual, out, and out_resid to have matching shapes")
    if len(x.spec.shape) < 1:
        raise ValueError("baybridge.layernorm requires rank >= 1 tensors")
    hidden = x.spec.shape[-1]
    if weight.spec.shape not in {(hidden,), x.spec.shape}:
        raise ValueError(
            f"baybridge.layernorm expects weight shape {(hidden,)} or {x.spec.shape}, got {weight.spec.shape}"
        )
    if bias.spec.shape not in {(hidden,), x.spec.shape}:
        raise ValueError(
            f"baybridge.layernorm expects bias shape {(hidden,)} or {x.spec.shape}, got {bias.spec.shape}"
        )
    if not all(value.spec.dtype == x.spec.dtype for value in (residual, out, out_resid, weight, bias)):
        raise ValueError("baybridge.layernorm requires matching tensor dtypes")
    require_builder().emit_void(
        "layernorm",
        x,
        residual,
        out,
        out_resid,
        weight,
        bias,
        attrs={"epsilon": float(epsilon)},
    )
    return out


def rmsnorm(
    x: TensorValue | RuntimeTensor,
    out: TensorValue | RuntimeTensor,
    gamma: TensorValue | RuntimeTensor,
    *,
    epsilon: float = 1e-5,
) -> Any:
    if all(isinstance(value, RuntimeTensor) for value in (x, out, gamma)):
        if x.shape != out.shape:
            raise ValueError("baybridge.rmsnorm requires x and out to have matching shapes")
        if len(x.shape) < 1:
            raise ValueError("baybridge.rmsnorm requires rank >= 1 tensors")
        hidden = x.shape[-1]
        if gamma.shape not in {(hidden,), x.shape}:
            raise ValueError(
                f"baybridge.rmsnorm expects gamma shape {(hidden,)} or {x.shape}, got {gamma.shape}"
            )
        for prefix in _runtime_prefix_indices(x.shape[:-1]):
            values = [float(x[_runtime_index(prefix + (lane,))]) for lane in range(hidden)]
            mean_square = sum(value * value for value in values) / float(hidden)
            inv_rms = 1.0 / py_math.sqrt(mean_square + float(epsilon))
            for lane, value in enumerate(values):
                index = _runtime_index(prefix + (lane,))
                gamma_value = float(gamma[lane] if len(gamma.shape) == 1 else gamma[index])
                out[index] = value * inv_rms * gamma_value
        return out
    if not all(isinstance(value, TensorValue) for value in (x, out, gamma)):
        raise TypeError("baybridge.rmsnorm expects all arguments to be runtime tensors or all to be traced tensors")
    if x.spec.shape != out.spec.shape:
        raise ValueError("baybridge.rmsnorm requires x and out to have matching shapes")
    if len(x.spec.shape) < 1:
        raise ValueError("baybridge.rmsnorm requires rank >= 1 tensors")
    hidden = x.spec.shape[-1]
    if gamma.spec.shape not in {(hidden,), x.spec.shape}:
        raise ValueError(
            f"baybridge.rmsnorm expects gamma shape {(hidden,)} or {x.spec.shape}, got {gamma.spec.shape}"
        )
    if gamma.spec.dtype != x.spec.dtype or out.spec.dtype != x.spec.dtype:
        raise ValueError("baybridge.rmsnorm requires matching tensor dtypes")
    require_builder().emit_void(
        "rmsnorm",
        x,
        out,
        gamma,
        attrs={"epsilon": float(epsilon)},
    )
    return out


def attention(
    q: TensorValue | RuntimeTensor,
    k: TensorValue | RuntimeTensor,
    v: TensorValue | RuntimeTensor,
    out: TensorValue | RuntimeTensor,
    lse: TensorValue | RuntimeTensor,
    *,
    causal: bool = False,
) -> Any:
    if all(isinstance(value, RuntimeTensor) for value in (q, k, v, out, lse)):
        if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4 or len(out.shape) != 4:
            raise ValueError("baybridge.attention requires rank-4 q, k, v, and out tensors")
        if q.shape != out.shape:
            raise ValueError("baybridge.attention requires q and out to have matching shapes")
        bsz, seqlen, heads, head_dim = q.shape
        kb, kn, kv_heads, kv_dim = k.shape
        if v.shape != k.shape:
            raise ValueError("baybridge.attention requires k and v to have matching shapes")
        if (kb, kn, kv_dim) != (bsz, seqlen, head_dim):
            raise ValueError("baybridge.attention requires compatible q/k/v shapes")
        if heads % kv_heads != 0:
            raise ValueError("baybridge.attention requires q heads to be divisible by kv heads")
        if lse.shape not in {(bsz, heads, 1, seqlen), (bsz, heads, seqlen)}:
            raise ValueError(
                f"baybridge.attention expects lse shape {(bsz, heads, 1, seqlen)} or {(bsz, heads, seqlen)}, got {lse.shape}"
            )
        scale = 1.0 / py_math.sqrt(float(head_dim))
        group_size = heads // kv_heads
        for batch in range(bsz):
            for head in range(heads):
                kv_head = head // group_size
                for query_idx in range(seqlen):
                    scores: list[float] = []
                    max_score = float("-inf")
                    for key_idx in range(seqlen):
                        if causal and key_idx > query_idx:
                            score = float("-inf")
                        else:
                            acc = 0.0
                            for dim_idx in range(head_dim):
                                acc += float(q[batch, query_idx, head, dim_idx]) * float(
                                    k[batch, key_idx, kv_head, dim_idx]
                                )
                            score = acc * scale
                        scores.append(score)
                        if score > max_score:
                            max_score = score
                    exp_scores = [0.0 if score == float("-inf") else py_math.exp(score - max_score) for score in scores]
                    norm = sum(exp_scores)
                    lse_value = float("-inf") if norm == 0.0 else py_math.log(norm) + max_score
                    if len(lse.shape) == 4:
                        lse[batch, head, 0, query_idx] = lse_value
                    else:
                        lse[batch, head, query_idx] = lse_value
                    for dim_idx in range(head_dim):
                        acc = 0.0
                        for key_idx, weight in enumerate(exp_scores):
                            if weight == 0.0:
                                continue
                            acc += (weight / norm) * float(v[batch, key_idx, kv_head, dim_idx])
                        out[batch, query_idx, head, dim_idx] = acc
        return out
    if not all(isinstance(value, TensorValue) for value in (q, k, v, out, lse)):
        raise TypeError("baybridge.attention expects all arguments to be runtime tensors or all to be traced tensors")
    if len(q.spec.shape) != 4 or len(k.spec.shape) != 4 or len(v.spec.shape) != 4 or len(out.spec.shape) != 4:
        raise ValueError("baybridge.attention requires rank-4 q, k, v, and out tensors")
    if q.spec.shape != out.spec.shape:
        raise ValueError("baybridge.attention requires q and out to have matching shapes")
    bsz, seqlen, heads, head_dim = q.spec.shape
    kb, kn, kv_heads, kv_dim = k.spec.shape
    if v.spec.shape != k.spec.shape:
        raise ValueError("baybridge.attention requires k and v to have matching shapes")
    if (kb, kn, kv_dim) != (bsz, seqlen, head_dim):
        raise ValueError("baybridge.attention requires compatible q/k/v shapes")
    if heads % kv_heads != 0:
        raise ValueError("baybridge.attention requires q heads to be divisible by kv heads")
    if lse.spec.shape not in {(bsz, heads, 1, seqlen), (bsz, heads, seqlen)}:
        raise ValueError(
            f"baybridge.attention expects lse shape {(bsz, heads, 1, seqlen)} or {(bsz, heads, seqlen)}, got {lse.spec.shape}"
        )
    if not all(value.spec.dtype == q.spec.dtype for value in (k, v, out)):
        raise ValueError("baybridge.attention requires matching q/k/v/out dtypes")
    require_builder().emit_void(
        "attention",
        q,
        k,
        v,
        out,
        lse,
        attrs={"causal": bool(causal)},
    )
    return out


def _default_accumulator_dtype(a_dtype: str, b_dtype: str) -> str:
    if a_dtype != b_dtype and {a_dtype, b_dtype} != {"fp8", "bf8"}:
        raise ValueError(f"mma currently requires matching operand dtypes, got {a_dtype} and {b_dtype}")
    if a_dtype in {"f16", "bf16", "fp8", "bf8"} or b_dtype in {"f16", "bf16", "fp8", "bf8"}:
        return "f32"
    return a_dtype


def _fragment_view(
    tensor: TensorValue,
    *,
    tile: tuple[int, int, int],
    role: str,
    axes: tuple[str, ...] | None = None,
    wave_size: int = 64,
    accumulator_dtype: str | None = None,
) -> TensorValue:
    accumulator = accumulator_dtype or _default_accumulator_dtype(tensor.spec.dtype, tensor.spec.dtype)
    descriptor = resolve_mfma_descriptor(tile, tensor.spec.dtype, accumulator, wave_size=wave_size)
    operand_tile = descriptor.operand_shape(role)
    wave_tile = partition_wave(tensor, operand_tile, axes=axes, wave_size=wave_size)
    lane_row, lane_col = lane_coords(descriptor.lane_shape)
    fragment_spec = TensorSpec(
        shape=operand_tile,
        dtype=tensor.spec.dtype,
        address_space=AddressSpace.REGISTER,
    )
    return require_builder().emit_tensor(
        "fragment",
        wave_tile,
        lane_row,
        lane_col,
        spec=fragment_spec,
        attrs={
            "role": role,
            "tile": list(tile),
            "lane_shape": list(descriptor.lane_shape),
            "wave_size": descriptor.wave_size,
            "variant": descriptor.variant_name,
            "llvm_intrinsic": descriptor.llvm_intrinsic,
        },
        name_hint=f"{tensor.name}_frag_{role}",
    )


def make_fragment_a(
    tensor: TensorValue,
    *,
    tile: tuple[int, int, int],
    axes: tuple[str, ...] | None = None,
    wave_size: int = 64,
    accumulator_dtype: str | None = None,
) -> TensorValue:
    return _fragment_view(
        tensor,
        tile=tile,
        role="a",
        axes=axes,
        wave_size=wave_size,
        accumulator_dtype=accumulator_dtype,
    )


def make_fragment_b(
    tensor: TensorValue,
    *,
    tile: tuple[int, int, int],
    axes: tuple[str, ...] | None = None,
    wave_size: int = 64,
    accumulator_dtype: str | None = None,
) -> TensorValue:
    return _fragment_view(
        tensor,
        tile=tile,
        role="b",
        axes=axes,
        wave_size=wave_size,
        accumulator_dtype=accumulator_dtype,
    )


def mma(
    a: TensorValue,
    b: TensorValue,
    c: TensorValue | None = None,
    *,
    tile: tuple[int, ...] | None = None,
    accumulate: bool = True,
    accumulator_dtype: str | None = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> TensorValue:
    if a.spec.dtype != b.spec.dtype and {a.spec.dtype, b.spec.dtype} != {"fp8", "bf8"}:
        raise ValueError(f"mma currently requires matching operand dtypes, got {a.spec.dtype} and {b.spec.dtype}")
    if c is not None:
        if accumulator_dtype is not None and accumulator_dtype != c.spec.dtype:
            raise ValueError(
                f"mma accumulator_dtype={accumulator_dtype} does not match explicit accumulator tensor dtype {c.spec.dtype}"
            )
        output = c
    else:
        output_dtype = accumulator_dtype or _default_accumulator_dtype(a.spec.dtype, b.spec.dtype)
        output = make_tensor(
            name="acc",
            shape=a.spec.shape,
            dtype=output_dtype,
            address_space="register",
        )
    require_builder().emit_void(
        "mma",
        a,
        b,
        output,
        attrs={
            "tile": list(tile) if tile else None,
            "accumulate": accumulate,
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
        },
    )
    return output


def printf(format_string: str, *args: Any) -> None:
    try:
        builder = require_builder()
    except CompilationError:
        if args:
            print(format_string.format(*(_runtime_printf_value(arg) for arg in args)))
            return
        print(format_string)
        return
    traced_inputs = [arg for arg in args if isinstance(arg, (ScalarValue, TensorValue))]
    builder.emit_void(
        "printf",
        *traced_inputs,
        attrs={"format": format_string, "arg_count": len(args)},
    )


class _UnsupportedNamespace:
    def __init__(self, namespace: str):
        self._namespace = namespace

    def __getattr__(self, attribute: str) -> Any:
        raise UnsupportedOperationError(
            f"baybridge.{self._namespace}.{attribute} is NVIDIA-specific and not implemented in the AMD port"
        )


class _CpAsyncLoadCacheMode:
    GLOBAL = "global"
    ALWAYS = "always"


class _CpAsyncNamespace(_UnsupportedNamespace):
    LoadCacheMode = _CpAsyncLoadCacheMode
    ReductionOp = _CpAsyncReductionOp

    def __init__(self):
        super().__init__("nvgpu.cpasync")

    def CopyG2SOp(self, *, cache_mode: str | None = None) -> CpAsyncCopyG2SOp:
        return CpAsyncCopyG2SOp(cache_mode=cache_mode)

    def CopyBulkTensorTileG2SOp(self, cta_group: Any = None) -> CpAsyncCopyBulkTensorTileG2SOp:
        return CpAsyncCopyBulkTensorTileG2SOp(cta_group=cta_group)

    def CopyBulkTensorTileG2SMulticastOp(self, cta_group: Any = None) -> CpAsyncCopyBulkTensorTileG2SMulticastOp:
        return CpAsyncCopyBulkTensorTileG2SMulticastOp(cta_group=cta_group)

    def CopyBulkTensorTileS2GOp(self, cta_group: Any = None) -> CpAsyncCopyBulkTensorTileS2GOp:
        return CpAsyncCopyBulkTensorTileS2GOp(cta_group=cta_group)

    def CopyReduceBulkTensorTileS2GOp(
        self,
        reduction_kind: Any = _CpAsyncReductionOp.ADD,
        cta_group: Any = None,
    ) -> CpAsyncCopyReduceBulkTensorTileS2GOp:
        return CpAsyncCopyReduceBulkTensorTileS2GOp(reduction_kind=reduction_kind, cta_group=cta_group)

    def CopyDsmemStoreOp(self) -> CpAsyncCopyDsmemStoreOp:
        return CpAsyncCopyDsmemStoreOp()

    def make_tiled_tma_atom(
        self,
        op: Any,
        tensor: TensorValue | RuntimeTensor,
        smem_layout: Any,
        cta_tiler: Any,
        *,
        num_multicast: int = 1,
        internal_type: Any | None = None,
    ) -> tuple[CopyAtom, TensorValue | RuntimeTensor]:
        value_type = internal_type or _tensor_element_type_name(tensor)
        atom = make_copy_atom(op, value_type)
        atom.set("smem_layout", smem_layout)
        atom.set("cta_tiler", cta_tiler)
        atom.set("num_multicast", int(num_multicast))
        return atom, tensor

    def create_tma_multicast_mask(
        self,
        cta_layout_vmnk: Layout | Any,
        cta_coord_vmnk: tuple[int, ...] | list[int],
        mcast_mode: str = "m",
    ) -> int:
        raw_shape = getattr(cta_layout_vmnk, "shape", None)
        if raw_shape is None:
            raw_shape = tuple(cta_layout_vmnk)
        shape = tuple(int(dim) for dim in raw_shape)
        coord = tuple(int(dim) for dim in tuple(cta_coord_vmnk))
        if len(shape) != len(coord):
            raise ValueError("create_tma_multicast_mask expects coord rank to match the CTA layout rank")
        axis_map = {"v": 0, "m": 0, "n": 1, "k": 2}
        axis = axis_map.get(str(mcast_mode).lower())
        if axis is None or axis >= len(shape):
            raise ValueError("unsupported multicast mode")
        strides = [prod(shape[index + 1:]) if index + 1 < len(shape) else 1 for index in range(len(shape))]
        mask = 0
        for axis_value in range(shape[axis]):
            linear_coord = list(coord)
            linear_coord[axis] = axis_value
            linear_index = sum(int(value) * int(strides[index]) for index, value in enumerate(linear_coord))
            mask |= 1 << linear_index
        return mask

    def copy_tensormap(self, tma_atom: Any, tensormap_ptr: Any) -> None:
        if isinstance(tma_atom, CopyAtom):
            tma_atom.set("tensormap_ptr", tensormap_ptr)

    def update_tma_descriptor(self, tma_atom: Any, gmem_tensor: Any, tma_desc_ptr: Any) -> None:
        if isinstance(tma_atom, CopyAtom):
            tma_atom.set("tma_desc_ptr", tma_desc_ptr)
            tma_atom.set("gmem_shape", getattr(gmem_tensor, "shape", None))
            tma_atom.set("gmem_stride", getattr(gmem_tensor, "stride", None))

    def fence_tma_desc_acquire(self, tma_desc_ptr: Any) -> None:
        del tma_desc_ptr

    def cp_fence_tma_desc_release(self, global_ptr: Any, shared_ptr: Any) -> None:
        del global_ptr
        del shared_ptr

    def fence_tma_desc_release(self) -> None:
        return None

    def is_tma_load(self, atom_or_op: Any) -> bool:
        op = atom_or_op.op if isinstance(atom_or_op, CopyAtom) else atom_or_op
        return isinstance(
            op,
            (
                CpAsyncCopyBulkTensorTileG2SOp,
                CpAsyncCopyBulkTensorTileG2SMulticastOp,
            ),
        )

    def is_tma_store(self, atom_or_op: Any) -> bool:
        op = atom_or_op.op if isinstance(atom_or_op, CopyAtom) else atom_or_op
        return isinstance(
            op,
            (
                CpAsyncCopyBulkTensorTileS2GOp,
                CpAsyncCopyReduceBulkTensorTileS2GOp,
                CpAsyncCopyDsmemStoreOp,
            ),
        )

    def is_tma_reduce(self, atom_or_op: Any) -> bool:
        op = atom_or_op.op if isinstance(atom_or_op, CopyAtom) else atom_or_op
        return isinstance(op, CpAsyncCopyReduceBulkTensorTileS2GOp)

    def get_tma_copy_properties(self, atom_or_op: Any) -> dict[str, Any]:
        op = atom_or_op.op if isinstance(atom_or_op, CopyAtom) else atom_or_op
        if isinstance(op, CpAsyncCopyBulkTensorTileG2SOp):
            return {"mode": "load", "variant": "g2s", "cta_group": op.cta_group}
        if isinstance(op, CpAsyncCopyBulkTensorTileG2SMulticastOp):
            return {"mode": "load", "variant": "g2s_multicast", "cta_group": op.cta_group}
        if isinstance(op, CpAsyncCopyBulkTensorTileS2GOp):
            return {"mode": "store", "variant": "s2g", "cta_group": op.cta_group}
        if isinstance(op, CpAsyncCopyReduceBulkTensorTileS2GOp):
            return {
                "mode": "store",
                "variant": "s2g_reduce",
                "cta_group": op.cta_group,
                "reduction": op.reduction_kind,
            }
        if isinstance(op, CpAsyncCopyDsmemStoreOp):
            return {"mode": "store", "variant": "dsmem_store", "cta_group": None}
        raise TypeError("cpasync helper expects a supported TMA copy op or baybridge CopyAtom")

    def prefetch_descriptor(self, descriptor: Any) -> None:
        del descriptor

    def tma_partition(
        self,
        tma_atom: Any,
        stage: Any,
        layout: Any,
        smem_tensor: Any,
        gmem_tensor: Any,
    ) -> tuple[Any, Any]:
        del tma_atom
        del stage
        del layout
        return smem_tensor, gmem_tensor


class _WarpNamespace(_UnsupportedNamespace):
    def __init__(self):
        super().__init__("nvgpu.warp")

    def MmaF16BF16Op(
        self,
        operand_dtype: Any,
        accumulator_dtype: Any,
        tile: tuple[int, int, int],
    ) -> WarpMmaF16BF16Op:
        return WarpMmaF16BF16Op(operand_dtype, accumulator_dtype, tile)

    def LdMatrix8x8x16bOp(
        self,
        *,
        transpose: bool = False,
        num_matrices: int = 1,
        ) -> WarpLdMatrix8x8x16bOp:
        return WarpLdMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices)


class _WarpgroupNamespace(_UnsupportedNamespace):
    def __init__(self, namespace: str = "nvgpu.warpgroup"):
        super().__init__(namespace)

    def MmaF16BF16Op(
        self,
        operand_dtype: Any,
        accumulator_dtype: Any,
        tile: tuple[int, int, int],
    ) -> WarpgroupMmaF16BF16Op:
        return WarpgroupMmaF16BF16Op(operand_dtype, accumulator_dtype, tile)

    def MmaTF32Op(
        self,
        accumulator_dtype: Any,
        tile: tuple[int, int, int],
    ) -> WarpgroupMmaTF32Op:
        return WarpgroupMmaTF32Op(accumulator_dtype, tile)

    def fence(self) -> None:
        barrier(kind="block", scope="warpgroup")

    def commit_batch(self) -> None:
        commit_group(group="warpgroup")

    def wait_batch(self, count: int = 0) -> None:
        wait_group(count=count, group="warpgroup")


class _Tcgen05CtaGroup:
    ONE = "one"
    TWO = "two"


class _Tcgen05OperandSource:
    SMEM = "smem"
    TMEM = "tmem"


class _Tcgen05OperandMajorMode:
    M = "m"
    N = "n"
    K = "k"


class _Tcgen05Repetition:
    x1 = "x1"
    x2 = "x2"
    x4 = "x4"
    x8 = "x8"
    x16 = "x16"
    x32 = "x32"
    x64 = "x64"
    x128 = "x128"


class _Tcgen05Field:
    ACCUMULATE = "accumulate"
    NEGATE_A = "negate_a"
    NEGATE_B = "negate_b"
    SCALE_A = "scale_a"
    SCALE_B = "scale_b"


class _Tcgen05TmemLoadRedOp:
    ADD = "add"
    MAX = "max"
    MIN = "min"


class _Tcgen05Pack:
    NONE = "none"
    PACK_16 = "pack_16"
    PACK_32 = "pack_32"
    PACK_16b_IN_32b = PACK_16


class _Tcgen05Unpack:
    NONE = "none"
    UNPACK_16 = "unpack_16"
    UNPACK_32 = "unpack_32"
    UNPACK_32b_IN_16b = UNPACK_32


class _Tcgen05SmemLayoutAtomKind:
    MN_INTER = "mn_inter"
    MN_SW32 = "mn_sw32"
    MN_SW64 = "mn_sw64"
    MN_SW128 = "mn_sw128"
    MN_SW128_32B = "mn_sw128_32b"
    K_INTER = "k_inter"
    K_SW32 = "k_sw32"
    K_SW64 = "k_sw64"
    K_SW128 = "k_sw128"


class _Tcgen05Namespace(_UnsupportedNamespace):
    CtaGroup = _Tcgen05CtaGroup
    OperandSource = _Tcgen05OperandSource
    OperandMajorMode = _Tcgen05OperandMajorMode
    Repetition = _Tcgen05Repetition
    Field = _Tcgen05Field
    TmemLoadRedOp = _Tcgen05TmemLoadRedOp
    Pack = _Tcgen05Pack
    Unpack = _Tcgen05Unpack
    SmemLayoutAtomKind = _Tcgen05SmemLayoutAtomKind

    def __init__(self):
        super().__init__("nvgpu.tcgen05")

    def MmaF16BF16Op(
        self,
        operand_dtype: Any,
        accumulator_dtype: Any,
        tile: tuple[int, int, int],
        cta_group: Any = None,
        operand_source: Any = None,
        operand_major_mode_a: Any = None,
        operand_major_mode_b: Any = None,
        ) -> Tcgen05MmaF16BF16Op:
        return Tcgen05MmaF16BF16Op(
            operand_dtype,
            accumulator_dtype,
            tile,
            cta_group,
            operand_source,
            operand_major_mode_a,
            operand_major_mode_b,
        )

    def MmaTF32Op(
        self,
        accumulator_dtype: Any,
        tile: tuple[int, int, int],
        cta_group: Any = None,
        operand_source: Any = None,
        operand_major_mode_a: Any = None,
        operand_major_mode_b: Any = None,
    ) -> Tcgen05MmaTF32Op:
        return Tcgen05MmaTF32Op(
            accumulator_dtype,
            tile,
            cta_group,
            operand_source,
            operand_major_mode_a,
            operand_major_mode_b,
        )

    def MmaI8Op(
        self,
        operand_dtype: Any,
        accumulator_dtype: Any,
        tile: tuple[int, int, int],
        cta_group: Any = None,
        operand_source: Any = None,
        operand_major_mode_a: Any = None,
        operand_major_mode_b: Any = None,
    ) -> Tcgen05MmaI8Op:
        return Tcgen05MmaI8Op(
            operand_dtype,
            accumulator_dtype,
            tile,
            cta_group,
            operand_source,
            operand_major_mode_a,
            operand_major_mode_b,
        )

    def Ld32x32bOp(self, repetition: Any = None) -> Tcgen05Ld32x32bOp:
        return Tcgen05Ld32x32bOp(repetition=repetition)

    def Ld16x64bOp(self, repetition: Any = None, pack: Any = None) -> Tcgen05Ld16x64bOp:
        return Tcgen05Ld16x64bOp(repetition=repetition, pack=pack)

    def Ld16x128bOp(self, repetition: Any = None, pack: Any = None) -> Tcgen05Ld16x128bOp:
        return Tcgen05Ld16x128bOp(repetition=repetition, pack=pack)

    def Ld16x256bOp(self, repetition: Any = None, pack: Any = None) -> Tcgen05Ld16x256bOp:
        return Tcgen05Ld16x256bOp(repetition=repetition, pack=pack)

    def Ld16x32bx2Op(self, repetition: Any = None, pack: Any = None) -> Tcgen05Ld16x32bx2Op:
        return Tcgen05Ld16x32bx2Op(repetition=repetition, pack=pack)

    def St16x64bOp(self, repetition: Any = None, unpack: Any = None) -> Tcgen05St16x64bOp:
        return Tcgen05St16x64bOp(repetition=repetition, unpack=unpack)

    def St16x128bOp(self, repetition: Any = None, unpack: Any = None) -> Tcgen05St16x128bOp:
        return Tcgen05St16x128bOp(repetition=repetition, unpack=unpack)

    def St16x256bOp(self, repetition: Any = None, unpack: Any = None) -> Tcgen05St16x256bOp:
        return Tcgen05St16x256bOp(repetition=repetition, unpack=unpack)

    def St16x32bx2Op(self, repetition: Any = None, unpack: Any = None) -> Tcgen05St16x32bx2Op:
        return Tcgen05St16x32bx2Op(repetition=repetition, unpack=unpack)

    def St32x32bOp(self, repetition: Any = None, unpack: Any = None) -> Tcgen05St32x32bOp:
        return Tcgen05St32x32bOp(repetition=repetition, unpack=unpack)

    def _repetition_count(self, repetition: Any) -> int:
        if repetition is None:
            return 1
        text = str(repetition)
        if text.startswith("x"):
            try:
                return int(text[1:])
            except ValueError:
                pass
        return 1

    def is_tmem_load(self, atom_or_op: Any) -> bool:
        op = atom_or_op.op if isinstance(atom_or_op, CopyAtom) else atom_or_op
        return isinstance(op, (Tcgen05Ld32x32bOp, Tcgen05Ld16x64bOp, Tcgen05Ld16x128bOp, Tcgen05Ld16x256bOp, Tcgen05Ld16x32bx2Op))

    def is_tmem_store(self, atom_or_op: Any) -> bool:
        op = atom_or_op.op if isinstance(atom_or_op, CopyAtom) else atom_or_op
        return isinstance(op, (Tcgen05St16x64bOp, Tcgen05St16x128bOp, Tcgen05St16x256bOp, Tcgen05St16x32bx2Op, Tcgen05St32x32bOp))

    def get_tmem_copy_properties(self, atom_or_op: Any) -> dict[str, Any]:
        op = atom_or_op.op if isinstance(atom_or_op, CopyAtom) else atom_or_op
        traits = {
            Tcgen05Ld32x32bOp: {"mode": "load", "lanes": 32, "bits": 32, "pack": getattr(op, "pack", None)},
            Tcgen05Ld16x64bOp: {"mode": "load", "lanes": 16, "bits": 64, "pack": getattr(op, "pack", None)},
            Tcgen05Ld16x128bOp: {"mode": "load", "lanes": 16, "bits": 128, "pack": getattr(op, "pack", None)},
            Tcgen05Ld16x256bOp: {"mode": "load", "lanes": 16, "bits": 256, "pack": getattr(op, "pack", None)},
            Tcgen05Ld16x32bx2Op: {"mode": "load", "lanes": 16, "bits": 64, "pack": getattr(op, "pack", None)},
            Tcgen05St16x64bOp: {"mode": "store", "lanes": 16, "bits": 64, "unpack": getattr(op, "unpack", None)},
            Tcgen05St16x128bOp: {"mode": "store", "lanes": 16, "bits": 128, "unpack": getattr(op, "unpack", None)},
            Tcgen05St16x256bOp: {"mode": "store", "lanes": 16, "bits": 256, "unpack": getattr(op, "unpack", None)},
            Tcgen05St16x32bx2Op: {"mode": "store", "lanes": 16, "bits": 64, "unpack": getattr(op, "unpack", None)},
            Tcgen05St32x32bOp: {"mode": "store", "lanes": 32, "bits": 32, "unpack": getattr(op, "unpack", None)},
        }
        for op_type, values in traits.items():
            if isinstance(op, op_type):
                return {
                    **values,
                    "repetition": self._repetition_count(getattr(op, "repetition", None)),
                }
        raise TypeError("tcgen05 helper expects a supported TMEM load/store op or baybridge CopyAtom")

    def find_tmem_tensor_col_offset(self, tensor: Any) -> int:
        stride = getattr(tensor, "stride", None)
        offset = int(getattr(tensor, "offset", 0))
        if callable(stride):
            stride = stride()
        if isinstance(stride, tuple) and len(stride) >= 2 and stride[0] > 0:
            return offset % int(stride[0])
        return offset

    def tile_to_mma_shape(self, value: Any) -> tuple[int, ...]:
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple):
            return tuple(int(dim) for dim in shape)
        raise TypeError("tcgen05.tile_to_mma_shape expects a tensor-like value with a shape")

    def make_tmem_copy(self, atom: CopyAtom, tensor: Any) -> TiledCopyTV:
        shape = getattr(tensor, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            raise ValueError("tcgen05.make_tmem_copy expects a tensor-like value with rank >= 2")
        properties = self.get_tmem_copy_properties(atom)
        lanes = min(int(properties["lanes"]), int(shape[1]))
        repetitions = max(1, int(properties["repetition"]))
        thread_layout = Layout.ordered((1, lanes), order=(1, 0))
        value_layout = Layout.ordered((max(1, int(shape[0]) * repetitions), 1), order=(1, 0))
        return make_tiled_copy_tv(atom, thread_layout, value_layout)

    def make_s2t_copy(self, atom: CopyAtom, tensor: Any) -> TiledCopyTV:
        return self.make_tmem_copy(atom, tensor)

    def get_s2t_smem_desc_tensor(self, atom_or_copy: Any, smem_tensor: Any) -> Any:
        if not self.is_tmem_load(atom_or_copy):
            raise TypeError("tcgen05.get_s2t_smem_desc_tensor expects a TMEM load op or baybridge CopyAtom")
        return smem_tensor

    def make_smem_layout_atom(self, kind: Any, element_dtype: Any) -> Layout:
        kind_name = str(kind).lower()
        element_bits = max(8, int(element_type(resolve_element_type_name(element_dtype)).width))
        element_bytes = max(1, element_bits // 8)
        swizzle_bytes = {
            self.SmemLayoutAtomKind.MN_INTER: 16,
            self.SmemLayoutAtomKind.K_INTER: 16,
            self.SmemLayoutAtomKind.MN_SW32: 32,
            self.SmemLayoutAtomKind.K_SW32: 32,
            self.SmemLayoutAtomKind.MN_SW64: 64,
            self.SmemLayoutAtomKind.K_SW64: 64,
            self.SmemLayoutAtomKind.MN_SW128: 128,
            self.SmemLayoutAtomKind.MN_SW128_32B: 128,
            self.SmemLayoutAtomKind.K_SW128: 128,
        }.get(kind_name, 16)
        return Layout.ordered((1, max(1, swizzle_bytes // element_bytes)), order=(1, 0))

    def make_umma_smem_desc(
        self,
        src: Pointer,
        layout: Layout | Any,
        major: Any,
        *,
        next_src: Pointer | None = None,
    ) -> Tcgen05SmemDesc:
        if not isinstance(src, Pointer):
            raise TypeError("tcgen05.make_umma_smem_desc expects a baybridge Pointer")
        normalized_layout = layout if isinstance(layout, Layout) else make_layout(layout)
        major_text = str(major).lower()
        if major_text not in {self.OperandMajorMode.M, self.OperandMajorMode.N, self.OperandMajorMode.K}:
            raise ValueError("tcgen05.make_umma_smem_desc expects a supported operand major mode")
        swizzle = "SWIZZLE_NONE"
        if normalized_layout.shape and normalized_layout.shape[-1] >= 32:
            swizzle = "SWIZZLE_128B"
        return Tcgen05SmemDesc(src=src, layout=normalized_layout, major=major_text, next_src=next_src, swizzle=swizzle)

    def commit(
        self,
        mbarrier_or_ptr: Mbarrier | Pointer | None = None,
        *,
        mask: int | None = None,
        cta_group: Any = None,
    ) -> None:
        barrier_id = getattr(mbarrier_or_ptr, "index", None)
        barrier(kind="block", scope="tcgen05_commit", barrier_id=barrier_id, mask=mask, cta_group=cta_group)


class _MbarrierNamespace(_UnsupportedNamespace):
    def __init__(self):
        super().__init__("nvgpu.mbarrier")

    def init(self, mbarrier: Mbarrier, arrival_count: int | None = None) -> None:
        mbarrier.init(arrival_count)

    def init_fence(self, mbarrier: Mbarrier, arrival_count: int | None = None) -> None:
        mbarrier.init_fence(arrival_count)

    def arrive(self, mbarrier: Mbarrier) -> None:
        mbarrier.arrive()

    def expect_tx(self, mbarrier: Mbarrier, bytes: int) -> None:
        mbarrier.expect_tx(bytes)

    def arrive_and_expect_tx(self, mbarrier: Mbarrier, bytes: int) -> None:
        mbarrier.arrive_and_expect_tx(bytes)

    def wait(self, mbarrier: Mbarrier) -> None:
        mbarrier.wait()

    def try_wait(self, mbarrier: Mbarrier, phase: int | None = None) -> bool | ScalarValue:
        return mbarrier.try_wait(phase)

    def test_wait(self, mbarrier: Mbarrier, phase: int | None = None) -> bool | ScalarValue:
        return mbarrier.test_wait(phase)


class _NvgpuNamespace(_UnsupportedNamespace):
    def __init__(self):
        super().__init__("nvgpu")
        self.cpasync = _CpAsyncNamespace()
        self.warp = _WarpNamespace()
        self.warpgroup = _WarpgroupNamespace("nvgpu.warpgroup")
        self.wgmma = _WarpgroupNamespace("nvgpu.wgmma")
        self.tcgen05 = _Tcgen05Namespace()
        self.mbarrier = _MbarrierNamespace()

    def CopyUniversalOp(self) -> CopyUniversalOp:
        return CopyUniversalOp()

    def MmaUniversalOp(self, accumulator_dtype: Any) -> MmaUniversalOp:
        return MmaUniversalOp(accumulator_dtype)

    def make_tiled_tma_atom_A(
        self,
        op: Any,
        tensor: TensorValue | RuntimeTensor,
        smem_layout: Any,
        mma_tiler: Any,
        tiled_mma: TiledMma,
    ) -> tuple[CopyAtom, TensorValue | RuntimeTensor]:
        del smem_layout
        del mma_tiler
        del tiled_mma
        return make_copy_atom(op, _tensor_element_type_name(tensor)), tensor

    def make_tiled_tma_atom_B(
        self,
        op: Any,
        tensor: TensorValue | RuntimeTensor,
        smem_layout: Any,
        mma_tiler: Any,
        tiled_mma: TiledMma,
    ) -> tuple[CopyAtom, TensorValue | RuntimeTensor]:
        del smem_layout
        del mma_tiler
        del tiled_mma
        return make_copy_atom(op, _tensor_element_type_name(tensor)), tensor


class _ArchNamespace:
    @staticmethod
    def thread_idx() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return (thread_idx("x"), thread_idx("y"), thread_idx("z"))

    @staticmethod
    def block_idx() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return (block_idx("x"), block_idx("y"), block_idx("z"))

    @staticmethod
    def block_dim() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return (block_dim("x"), block_dim("y"), block_dim("z"))

    @staticmethod
    def grid_dim() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return (grid_dim("x"), grid_dim("y"), grid_dim("z"))

    @staticmethod
    def cluster_dim() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return (cluster_dim("x"), cluster_dim("y"), cluster_dim("z"))

    @staticmethod
    def cluster_idx() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return (cluster_idx("x"), cluster_idx("y"), cluster_idx("z"))

    @staticmethod
    def cluster_size() -> ScalarValue | int:
        return cluster_size()

    @staticmethod
    def block_idx_in_cluster() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return block_in_cluster_idx()

    @staticmethod
    def block_rank_in_cluster() -> ScalarValue | int:
        return block_rank_in_cluster()

    @staticmethod
    def block_in_cluster_idx() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return block_in_cluster_idx()

    @staticmethod
    def block_in_cluster_dim() -> tuple[ScalarValue | int, ScalarValue | int, ScalarValue | int]:
        return block_in_cluster_dim()

    @staticmethod
    def lane_id() -> ScalarValue | int:
        return lane_id()

    @staticmethod
    def warp_idx(*, wave_size: int = 64) -> ScalarValue | int:
        return warp_idx(wave_size=wave_size)

    @staticmethod
    def make_warp_uniform(value: Any) -> Any:
        return make_warp_uniform(value)

    @staticmethod
    def sync_threads(*, loc: Any | None = None, ip: Any | None = None) -> None:
        del loc
        del ip
        barrier(kind="block")

    @staticmethod
    def barrier(*, loc: Any | None = None, ip: Any | None = None) -> None:
        del loc
        del ip
        barrier(kind="block")

    @staticmethod
    def barrier_arrive(*, loc: Any | None = None, ip: Any | None = None) -> None:
        del loc
        del ip
        barrier(kind="block", action="arrive")

    @staticmethod
    def sync_warp(*, loc: Any | None = None, ip: Any | None = None) -> None:
        sync_warp(loc=loc, ip=ip)

    @staticmethod
    def sync_grid(*, loc: Any | None = None, ip: Any | None = None) -> None:
        del loc
        del ip
        barrier(kind="grid")

    @staticmethod
    def elect_one() -> ScalarValue | bool:
        return elect_one()

    @staticmethod
    def get_dyn_smem_size() -> ScalarValue | int:
        return get_dyn_smem_size()

    @staticmethod
    def mbarrier_init(mbarrier: Mbarrier, arrival_count: int | None = None) -> None:
        mbarrier.init(arrival_count)

    @staticmethod
    def mbarrier_init_fence(mbarrier: Mbarrier, arrival_count: int | None = None) -> None:
        mbarrier.init_fence(arrival_count)

    @staticmethod
    def mbarrier_arrive(mbarrier: Mbarrier) -> None:
        mbarrier.arrive()

    @staticmethod
    def mbarrier_expect_tx(mbarrier: Mbarrier, bytes: int) -> None:
        mbarrier.expect_tx(bytes)

    @staticmethod
    def mbarrier_arrive_and_expect_tx(mbarrier: Mbarrier, bytes: int) -> None:
        mbarrier.arrive_and_expect_tx(bytes)

    @staticmethod
    def mbarrier_wait(mbarrier: Mbarrier) -> None:
        mbarrier.wait()

    @staticmethod
    def mbarrier_try_wait(mbarrier: Mbarrier, phase: int | None = None) -> bool | ScalarValue:
        return mbarrier.try_wait(phase)

    @staticmethod
    def mbarrier_conditional_try_wait(
        mbarrier: Mbarrier,
        predicate: ScalarValue | bool,
        phase: int | None = None,
    ) -> bool | ScalarValue:
        return mbarrier.conditional_try_wait(predicate, phase)

    @staticmethod
    def mbarrier_test_wait(mbarrier: Mbarrier, phase: int | None = None) -> bool | ScalarValue:
        return mbarrier.test_wait(phase)

    @staticmethod
    def fence_tma_store(fence: TmaStoreFence | None = None) -> None:
        (fence or TmaStoreFence()).arrive_and_wait()


nvgpu = _NvgpuNamespace()
arch = _ArchNamespace()

__all__ = [
    "ComposedLayout",
    "CopyAtom",
    "HierarchicalLayout",
    "IdentityLayout",
    "KernelDefinition",
    "LaunchConfig",
    "Mbarrier",
    "MbarrierArray",
    "NamedBarrier",
    "ScalarSpec",
    "SmemAllocator",
    "Swizzle",
    "TmaStoreFence",
    "TmemAllocator",
    "TiledCopy",
    "TiledMma",
    "TensorSpec",
    "AddressSpace",
    "arch",
    "all_",
    "any_",
    "autovec_copy",
    "barrier",
    "basic_copy",
    "basic_copy_if",
    "block_dim",
    "block_in_cluster_dim",
    "block_in_cluster_idx",
    "block_idx",
    "block_idx_in_cluster",
    "block_rank_in_cluster",
    "ceil_div",
    "cluster_dim",
    "cluster_idx",
    "cluster_size",
    "composition",
    "commit_group",
    "cosize",
    "copy",
    "copy_async",
    "dim",
    "domain_offset",
    "dsl_range",
    "dsl_range_constexpr",
    "empty_like",
    "elem_less",
    "full",
    "full_like",
    "gemm",
    "get_dyn_smem_size",
    "grid_dim",
    "group_modes",
    "jit",
    "kernel",
    "elect_one",
    "lane_id",
    "lane_idx",
    "lane_coords",
    "local_partition",
    "local_tile",
    "load",
    "logical_divide",
    "logical_product",
    "make_atom",
    "make_copy_atom",
    "make_fragment",
    "make_fragment_like",
    "make_layout",
    "make_identity_tensor",
    "make_layout_tv",
    "make_mma_atom",
    "make_ordered_layout",
    "make_fragment_a",
    "make_fragment_b",
    "make_cotiled_copy",
    "make_swizzle",
    "make_tiled_copy",
    "make_tiled_copy_A",
    "make_tiled_copy_B",
    "make_tiled_copy_C",
    "make_tiled_copy_C_atom",
    "make_tiled_copy_D",
    "make_tiled_copy_S",
    "make_tiled_mma",
    "make_rmem_tensor",
    "make_warp_uniform",
    "print_tensor",
    "recast_ptr",
    "recast_layout",
    "repeat",
    "repeat_as_tuple",
    "repeat_like",
    "make_tensor",
    "mma",
    "nvgpu",
    "partition",
    "partition_program",
    "partition_thread",
    "partition_wave",
    "prefetch",
    "ones_like",
    "printf",
    "product_each",
    "program_id",
    "select",
    "size",
    "size_in_bytes",
    "store",
    "tiled_divide",
    "tiled_product",
    "tile_to_shape",
    "thread_idx",
    "transform_apply",
    "tuple_cat",
    "sync_warp",
    "wait_group",
    "warp_idx",
    "wave_id",
    "where",
    "zeros_like",
    "flat_product",
    "zipped_product",
    "zipped_divide",
    "struct",
]
