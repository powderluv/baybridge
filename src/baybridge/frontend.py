from __future__ import annotations

from itertools import product
from math import prod
from dataclasses import dataclass
from typing import Any, Callable

from .diagnostics import CompilationError, UnsupportedOperationError
from .ir import AddressSpace, Layout, ScalarSpec, TensorSpec
from .mfma import resolve_mfma_descriptor
from .runtime import (
    ExecutionState,
    LaunchConfig,
    Pointer,
    RuntimeTensor,
    current_execution,
    execution_context,
    full,
    normalize_runtime_argument,
    zeros,
)
from .tracing import ScalarValue, TensorValue, require_builder
from .views import (
    IdentityTensor,
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
class CopyAtom:
    op: Any
    value_type: str


@dataclass(frozen=True)
class TiledCopyTV:
    atom: CopyAtom
    thread_layout: Layout
    value_layout: Layout

    @property
    def tv_layout(self) -> ThreadValueLayout:
        return ThreadValueLayout(self.thread_layout, self.value_layout)

    @property
    def tiler(self) -> tuple[int, ...]:
        return self.tv_layout.tile_shape

    def get_slice(self, thread_index: ScalarValue | int) -> "ThreadCopySlice":
        return ThreadCopySlice(self, thread_index)


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
        shared_mem_bytes: int = 0,
        cooperative: bool = False,
        stream: Any | None = None,
    ) -> None:
        del stream
        launch = LaunchConfig(
            grid=tuple(grid),
            block=tuple(block),
            shared_mem_bytes=shared_mem_bytes,
            cooperative=cooperative,
        )
        try:
            require_builder()
        except CompilationError:
            pass
        else:
            raise TraceKernelLaunch(self.definition, self.args, self.kwargs, launch)
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
                                    self.definition.fn(*self.args, **self.kwargs)

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


def _layout_size(layout: Layout) -> int:
    return prod(layout.shape) if layout.shape else 1


def depth(value: Any) -> int:
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
    tile = _normalize_tiler_dims(tiler, rank=len(layout.shape))
    rest = tuple((dim + step - 1) // step for dim, step in zip(layout.shape, tile))
    return Layout(
        shape=tile + rest,
        stride=layout.stride + tuple(step * stride for step, stride in zip(tile, layout.stride)),
        swizzle=layout.swizzle,
    )


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
        if isinstance(name.tensor, TensorValue):
            spec = TensorSpec(
                shape=target_layout.shape,
                dtype=str(name.value_type),
                layout=target_layout,
                address_space=AddressSpace.GLOBAL,
            )
            return require_builder().emit_tensor(
                "pointer_tensor",
                name.tensor,
                spec=spec,
                attrs={
                    "assumed_align": name.assumed_align,
                    "address_space": str(name.address_space),
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


@dataclass
class SmemAllocator:
    allocation_index: int = 0

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
        spec = TensorSpec(
            shape=target_layout.shape,
            dtype=str(element_type),
            layout=target_layout,
            address_space=AddressSpace.SHARED,
        )
        self.allocation_index += 1
        name = f"smem_alloc_{self.allocation_index}"
        try:
            builder = require_builder()
            return builder.make_tensor(name, spec, dynamic_shared=True, byte_alignment=byte_alignment)
        except CompilationError:
            return zeros(target_layout.shape, dtype=str(element_type))

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


def _runtime_tensor(value: TensorValue | RuntimeTensor) -> RuntimeTensor:
    if isinstance(value, RuntimeTensor):
        return value
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


def _runtime_printf_value(value: Any) -> Any:
    if isinstance(value, RuntimeTensor):
        return value.tolist()
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


def wave_id(*, axis: str = "x", wave_size: int = 64) -> ScalarValue:
    normalized_axis = _normalize_axis(axis)
    if wave_size <= 0:
        raise ValueError("wave_size must be > 0")
    return thread_idx(normalized_axis) // wave_size


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
    value: TensorValue | RuntimeTensor | IdentityTensor,
    tiler: tuple[int, ...],
) -> TiledTensorView:
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
    if isinstance(value, IdentityTileTensorValue) and isinstance(mapping, ThreadValueLayout):
        return ThreadValueComposedCoordinateView(value, mapping)
    if isinstance(value, TensorValue) and isinstance(mapping, ThreadValueLayout):
        return ThreadValueComposedTensorView(value, mapping)
    if mapping is not None and isinstance(mapping, ThreadValueLayout):
        from .views import IdentityTileTensor

        if isinstance(value, (RuntimeTensor, IdentityTileTensor)):
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
    predicate: ScalarValue | bool,
    true_value: ScalarValue | int | float,
    false_value: ScalarValue | int | float,
    *,
    loc: Any | None = None,
    ip: Any | None = None,
) -> ScalarValue | int | float:
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


def full_like(value: RuntimeTensor, fill_value: Any, dtype: str | None = None) -> RuntimeTensor:
    if isinstance(value, TensorValue):
        raise CompilationError("full_like is only implemented on the reference runtime today")
    return full(value.shape, fill_value, dtype=dtype or value.dtype)


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


def make_copy_atom(op: Any, value_type: str) -> CopyAtom:
    return CopyAtom(op=op, value_type=str(value_type))


def make_tiled_copy_tv(atom: CopyAtom, thr_layout: Layout, val_layout: Layout) -> TiledCopyTV:
    if not isinstance(atom, CopyAtom):
        raise TypeError("make_tiled_copy_tv expects a baybridge CopyAtom")
    return TiledCopyTV(atom=atom, thread_layout=thr_layout, value_layout=val_layout)


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


def copy(*args: Any, vector_bytes: int | None = None, pred: TensorValue | RuntimeTensor | None = None) -> None:
    copy_atom = None
    if len(args) == 2:
        src, dst = args
    elif len(args) == 3 and isinstance(args[0], CopyAtom):
        copy_atom, src, dst = args
    else:
        raise TypeError("copy expects (src, dst) or (copy_atom, src, dst)")

    del copy_atom

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

    if pred is not None:
        raise UnsupportedOperationError("predicated copy is only implemented for baybridge thread fragments today")

    try:
        require_builder().emit_void("copy", src, dst, attrs={"vector_bytes": vector_bytes})
    except CompilationError:
        del vector_bytes
        _copy_runtime_tensor(_runtime_tensor(src), _runtime_tensor(dst))


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


def barrier(*, kind: str = "block") -> None:
    if kind not in {"block", "grid"}:
        raise ValueError("barrier kind must be 'block' or 'grid'")
    try:
        require_builder().emit_void("barrier", attrs={"kind": kind})
    except CompilationError:
        if kind == "grid":
            raise UnsupportedOperationError(
                "baybridge grid-wide barriers require a compiled cooperative launch"
            ) from None
        del kind


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


def _default_accumulator_dtype(a_dtype: str, b_dtype: str) -> str:
    if a_dtype != b_dtype:
        raise ValueError(f"mma currently requires matching operand dtypes, got {a_dtype} and {b_dtype}")
    if a_dtype in {"f16", "bf16"}:
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
) -> TensorValue:
    if a.spec.dtype != b.spec.dtype:
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
        attrs={"tile": list(tile) if tile else None, "accumulate": accumulate},
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


class _NvgpuNamespace(_UnsupportedNamespace):
    def __init__(self):
        super().__init__("nvgpu")

    def CopyUniversalOp(self) -> CopyUniversalOp:
        return CopyUniversalOp()


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
    def lane_id() -> ScalarValue | int:
        return lane_id()

    @staticmethod
    def sync_threads(*, loc: Any | None = None, ip: Any | None = None) -> None:
        del loc
        del ip
        barrier(kind="block")

    @staticmethod
    def sync_grid(*, loc: Any | None = None, ip: Any | None = None) -> None:
        del loc
        del ip
        barrier(kind="grid")


nvgpu = _NvgpuNamespace()
arch = _ArchNamespace()

__all__ = [
    "KernelDefinition",
    "LaunchConfig",
    "ScalarSpec",
    "SmemAllocator",
    "TensorSpec",
    "AddressSpace",
    "arch",
    "barrier",
    "block_dim",
    "block_idx",
    "composition",
    "commit_group",
    "copy",
    "copy_async",
    "dim",
    "elem_less",
    "full_like",
    "grid_dim",
    "jit",
    "kernel",
    "lane_id",
    "lane_coords",
    "load",
    "make_fragment",
    "make_fragment_like",
    "make_layout",
    "make_identity_tensor",
    "make_layout_tv",
    "make_ordered_layout",
    "make_fragment_a",
    "make_fragment_b",
    "make_rmem_tensor",
    "print_tensor",
    "recast_layout",
    "repeat_like",
    "make_tensor",
    "mma",
    "nvgpu",
    "partition",
    "partition_program",
    "partition_thread",
    "partition_wave",
    "printf",
    "product_each",
    "program_id",
    "select",
    "size",
    "store",
    "thread_idx",
    "wait_group",
    "wave_id",
    "where",
    "zipped_divide",
]
