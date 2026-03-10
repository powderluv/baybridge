from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from .dtypes import element_type
from .ir import AddressSpace, Layout, TensorSpec
from .runtime import RuntimeTensor
from .tracing import ScalarValue, TensorValue, require_builder


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _normalize_tiler(shape: tuple[int, ...], tiler: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != len(tiler):
        raise ValueError(f"tiler rank {len(tiler)} must match tensor rank {len(shape)}")
    if any(dim <= 0 for dim in tiler):
        raise ValueError("tiler dimensions must be > 0")
    return tuple(int(dim) for dim in tiler)


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
            raise TypeError("identity tensor indices must be integers at runtime")
        result.append(item)
    return tuple(result)


def _unflatten_row_major(linear_index: ScalarValue | int, shape: tuple[int, ...]) -> tuple[ScalarValue | int, ...]:
    if len(shape) == 1:
        return (linear_index,)
    remainder = linear_index
    coords: list[ScalarValue | int] = []
    for axis in range(len(shape) - 1):
        stride = prod(shape[axis + 1 :])
        coords.append(remainder // stride)
        remainder = remainder % stride
    coords.append(remainder)
    return tuple(coords)


def _normalize_rest_selector(
    selector: ScalarValue | int | tuple[ScalarValue | int, ...],
    rest_shape: tuple[int, ...],
) -> tuple[ScalarValue | int, ...]:
    if isinstance(selector, tuple):
        if len(selector) != len(rest_shape):
            raise ValueError(f"expected {len(rest_shape)} rest-space indices, got {len(selector)}")
        return selector
    if isinstance(selector, (ScalarValue, int)):
        return _unflatten_row_major(selector, rest_shape)
    raise TypeError("tiled tensor selectors must be integers, baybridge scalars, or tuples of those values")


@dataclass(frozen=True)
class IdentityTensor:
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def layout(self) -> Layout:
        return Layout.row_major(self.shape)

    @property
    def element_type(self) -> str:
        return element_type("coord")

    @property
    def type(self) -> str:
        return f"IdentityTensor(shape={self.shape})"

    def __getitem__(self, index: Any) -> Any:
        normalized = _normalize_runtime_index(index, self.ndim)
        if self.ndim == 1:
            return normalized[0]
        return normalized


@dataclass(frozen=True)
class IdentityTileTensor:
    shape: tuple[int, ...]
    offset: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def element_type(self) -> str:
        return element_type("coord")

    @property
    def type(self) -> str:
        return f"IdentityTile(shape={self.shape}, offset={self.offset})"

    def __getitem__(self, index: Any) -> Any:
        normalized = _normalize_runtime_index(index, self.ndim)
        coords = tuple(base + local for base, local in zip(self.offset, normalized))
        if self.ndim == 1:
            return coords[0]
        return coords


@dataclass(frozen=True)
class IdentityTileTensorValue:
    shape: tuple[int, ...]
    offset: tuple[ScalarValue | int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def element_type(self) -> str:
        return element_type("coord")

    @property
    def type(self) -> str:
        return f"IdentityTile(shape={self.shape}, offset={self.offset})"

    def __getitem__(self, index: Any) -> Any:
        normalized = _normalize_runtime_index(index, self.ndim)
        coords = tuple(base + local for base, local in zip(self.offset, normalized))
        if self.ndim == 1:
            return coords[0]
        return coords


@dataclass(frozen=True)
class LocalCoordinateTensor:
    shape: tuple[int, ...]
    base_offsets: tuple[int, ...]
    tile_sizes: tuple[int, ...]
    tile_axes: tuple[int, ...]
    rest_axes: tuple[int | None, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def element_type(self) -> str:
        return element_type("coord")

    @property
    def type(self) -> str:
        return f"LocalCoordinateTensor(shape={self.shape})"

    def _base_coords(self, index: tuple[int, ...]) -> tuple[int, ...]:
        coords: list[int] = []
        for axis, base_offset in enumerate(self.base_offsets):
            coord = base_offset + index[self.tile_axes[axis]]
            rest_axis = self.rest_axes[axis]
            if rest_axis is not None:
                coord += index[rest_axis] * self.tile_sizes[axis]
            coords.append(coord)
        return tuple(coords)

    def __getitem__(self, index: Any) -> Any:
        normalized = _normalize_runtime_index(index, self.ndim)
        coords = self._base_coords(normalized)
        if len(coords) == 1:
            return coords[0]
        return coords


@dataclass(frozen=True)
class LocalCoordinateTensorValue:
    shape: tuple[int, ...]
    base_offsets: tuple[ScalarValue | int, ...]
    tile_sizes: tuple[int, ...]
    tile_axes: tuple[int, ...]
    rest_axes: tuple[int | None, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def element_type(self) -> str:
        return element_type("coord")

    @property
    def type(self) -> str:
        return f"LocalCoordinateTensor(shape={self.shape})"

    def _base_coords(self, index: tuple[int, ...]) -> tuple[ScalarValue | int, ...]:
        coords: list[ScalarValue | int] = []
        for axis, base_offset in enumerate(self.base_offsets):
            coord: ScalarValue | int = base_offset + index[self.tile_axes[axis]]
            rest_axis = self.rest_axes[axis]
            if rest_axis is not None:
                coord = coord + index[rest_axis] * self.tile_sizes[axis]
            coords.append(coord)
        return tuple(coords)

    def compose_thread_value_coord(
        self,
        thread_index: ScalarValue,
        value_index: int,
        tv_layout: "ThreadValueLayout",
    ) -> tuple[ScalarValue | int, ...]:
        thread_coords = _layout_coords(tv_layout.thread_layout, thread_index)
        value_coords = _invert_layout_index(tv_layout.value_layout, value_index)
        local_coords = tuple(
            thread_coord * value_extent + value_coord
            for thread_coord, value_extent, value_coord in zip(
                thread_coords,
                tv_layout.value_layout.shape,
                value_coords,
            )
        )
        return self._base_coords(local_coords)

    def __getitem__(self, index: Any) -> Any:
        normalized = _normalize_runtime_index(index, self.ndim)
        coords = self._base_coords(normalized)
        if len(coords) == 1:
            return coords[0]
        return coords


@dataclass(frozen=True)
class TiledTensorView:
    base: TensorValue | RuntimeTensor | IdentityTensor
    tile: tuple[int, ...]
    rest_shape: tuple[int, ...]
    rest_axes_map: tuple[int, ...]

    @classmethod
    def create(
        cls,
        base: TensorValue | RuntimeTensor | IdentityTensor,
        tiler: tuple[int, ...],
    ) -> "TiledTensorView":
        shape = tuple(base.shape)
        tile = _normalize_tiler(shape, tiler)
        rest_shape = tuple(_ceil_div(dim, step) for dim, step in zip(shape, tile))
        return cls(
            base=base,
            tile=tile,
            rest_shape=rest_shape,
            rest_axes_map=tuple(range(len(rest_shape))),
        )

    @property
    def shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.tile, self.rest_shape

    @property
    def element_type(self) -> str:
        return self.base.element_type

    @property
    def type(self) -> str:
        return f"TiledTensor(tile={self.tile}, rest={self.rest_shape}, element_type={self.element_type})"

    def remap_rest(self, permutation: tuple[int, ...]) -> "TiledTensorView":
        if len(permutation) != len(self.rest_shape):
            raise ValueError("rest permutation rank must match the tiled tensor rest rank")
        if tuple(sorted(permutation)) != tuple(range(len(self.rest_shape))):
            raise ValueError("rest permutation must be a permutation of the tiled tensor rest axes")
        return TiledTensorView(
            base=self.base,
            tile=self.tile,
            rest_shape=tuple(self.rest_shape[axis] for axis in permutation),
            rest_axes_map=tuple(self.rest_axes_map[axis] for axis in permutation),
        )

    def _resolve_selector(self, selector: Any) -> TensorValue | RuntimeTensor | IdentityTileTensor | IdentityTileTensorValue:
        if not isinstance(selector, tuple) or len(selector) != 2:
            raise TypeError("tiled tensor indexing expects a (tile_selector, rest_selector) pair")
        tile_selector, rest_selector = selector
        if tile_selector is None:
            normalized_tile_selector = (None,) * len(self.tile)
        else:
            normalized_tile_selector = tile_selector
        if not isinstance(normalized_tile_selector, tuple) or len(normalized_tile_selector) != len(self.tile):
            raise ValueError(f"tile selector must have rank {len(self.tile)}")
        if any(part is not None for part in normalized_tile_selector):
            raise NotImplementedError("tiled tensor indexing currently only supports whole-tile selection with None")
        rest_coords = _normalize_rest_selector(rest_selector, self.rest_shape)
        base_rest_coords: list[ScalarValue | int | None] = [None] * len(self.rest_axes_map)
        for view_axis, coord in enumerate(rest_coords):
            base_rest_coords[self.rest_axes_map[view_axis]] = coord
        offsets = tuple(coord * step for coord, step in zip(base_rest_coords, self.tile))
        if isinstance(self.base, IdentityTensor):
            if any(isinstance(offset, ScalarValue) for offset in offsets):
                return IdentityTileTensorValue(shape=self.tile, offset=offsets)
            return IdentityTileTensor(shape=self.tile, offset=tuple(offsets))
        from .frontend import partition

        return partition(self.base, self.tile, offset=offsets)

    def __getitem__(self, selector: Any) -> TensorValue | RuntimeTensor | IdentityTileTensor | IdentityTileTensorValue:
        return self._resolve_selector(selector)

    def __setitem__(self, selector: Any, value: Any) -> None:
        target = self._resolve_selector(selector)
        if isinstance(target, TensorValue):
            target.store(value)
            return
        if isinstance(target, (IdentityTileTensor, IdentityTileTensorValue)):
            raise TypeError("cannot store into coordinate tiles")
        target.store(value)


def _invert_layout_index(layout: Layout, linear_index: int) -> tuple[int, ...]:
    remaining = linear_index
    coords = [0] * len(layout.shape)
    for axis in sorted(range(len(layout.shape)), key=lambda item: layout.stride[item], reverse=True):
        stride = layout.stride[axis]
        coords[axis] = remaining // stride
        remaining %= stride
    return tuple(coords)


def _layout_coords(layout: Layout, linear_index: ScalarValue | int) -> tuple[ScalarValue | int, ...]:
    if isinstance(linear_index, int):
        return _invert_layout_index(layout, linear_index)
    remaining: ScalarValue | int = linear_index
    coords: list[ScalarValue | int] = [0] * len(layout.shape)
    for axis in sorted(range(len(layout.shape)), key=lambda item: layout.stride[item], reverse=True):
        stride = layout.stride[axis]
        coords[axis] = remaining // stride
        remaining = remaining % stride
    return tuple(coords)


@dataclass(frozen=True)
class ThreadValueLayout:
    thread_layout: Layout
    value_layout: Layout

    @property
    def shape(self) -> tuple[int, int]:
        return (prod(self.thread_layout.shape), prod(self.value_layout.shape))

    @property
    def tile_shape(self) -> tuple[int, ...]:
        return tuple(lhs * rhs for lhs, rhs in zip(self.thread_layout.shape, self.value_layout.shape))

    @property
    def type(self) -> str:
        return (
            f"ThreadValueLayout(threads={self.shape[0]}, values={self.shape[1]}, tile={self.tile_shape})"
        )

    def tile_coords(self, thread_index: int, value_index: int) -> tuple[int, ...]:
        thread_coords = _invert_layout_index(self.thread_layout, thread_index)
        value_coords = _invert_layout_index(self.value_layout, value_index)
        return tuple(
            thread_coord * value_extent + value_coord
            for thread_coord, value_extent, value_coord in zip(
                thread_coords, self.value_layout.shape, value_coords
            )
        )


@dataclass(frozen=True)
class ThreadFragmentView:
    base: RuntimeTensor | IdentityTileTensor | LocalCoordinateTensor
    tv_layout: ThreadValueLayout
    thread_index: int

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.tv_layout.shape[1],)

    @property
    def element_type(self) -> str:
        return self.base.element_type

    @property
    def type(self) -> str:
        return f"ThreadFragment(shape={self.shape}, element_type={self.element_type})"

    @property
    def layout(self) -> Layout:
        return Layout.row_major(self.shape)

    def __getitem__(self, index: Any) -> Any:
        normalized = _normalize_runtime_index(index, 1)
        coords = self.tv_layout.tile_coords(self.thread_index, normalized[0])
        return self.base[coords]

    def __setitem__(self, index: Any, value: Any) -> None:
        if not isinstance(self.base, RuntimeTensor):
            raise TypeError("cannot store into coordinate fragments")
        normalized = _normalize_runtime_index(index, 1)
        coords = self.tv_layout.tile_coords(self.thread_index, normalized[0])
        self.base[coords] = value

    def load(self, predicate: RuntimeTensor | None = None, *, else_value: Any = 0) -> RuntimeTensor:
        if predicate is not None:
            if predicate.dtype != "i1":
                raise ValueError("thread fragment load predicates must have dtype i1")
            if predicate.shape != self.shape:
                raise ValueError(f"thread fragment load predicates must have shape {self.shape}, got {predicate.shape}")
        values = [
            self[index] if predicate is None or predicate[index] else else_value
            for index in range(self.shape[0])
        ]
        return RuntimeTensor(values, self.shape, dtype=self.element_type)

    def store(self, value: RuntimeTensor | Any, predicate: RuntimeTensor | None = None) -> None:
        if predicate is not None:
            if predicate.dtype != "i1":
                raise ValueError("thread fragment store predicates must have dtype i1")
            if predicate.shape != self.shape:
                raise ValueError(f"thread fragment store predicates must have shape {self.shape}, got {predicate.shape}")
        if isinstance(value, RuntimeTensor):
            if value.shape != self.shape:
                raise ValueError(f"fragment store expects shape {self.shape}, got {value.shape}")
            for index in range(self.shape[0]):
                if predicate is None or predicate[index]:
                    self[index] = value[index]
            return
        if self.shape != (1,):
            raise TypeError("scalar fragment stores are only supported for size-1 fragments")
        if predicate is None or predicate[0]:
            self[0] = value


@dataclass(frozen=True)
class ThreadValueComposedView:
    base: RuntimeTensor | IdentityTileTensor | LocalCoordinateTensor
    tv_layout: ThreadValueLayout

    @property
    def shape(self) -> tuple[int, int]:
        return self.tv_layout.shape

    @property
    def element_type(self) -> str:
        return self.base.element_type

    @property
    def type(self) -> str:
        return f"ThreadValueView(shape={self.shape}, element_type={self.element_type})"

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int):
            return self.shape[index]
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("thread/value views expect (thread, value) indexing")
        thread_index, value_index = index
        if not isinstance(thread_index, int):
            raise TypeError("runtime thread/value views require integer thread indices")
        if value_index is None:
            return ThreadFragmentView(self.base, self.tv_layout, thread_index)
        normalized = _normalize_runtime_index(value_index, 1)
        return ThreadFragmentView(self.base, self.tv_layout, thread_index)[normalized]


@dataclass(frozen=True)
class ThreadCoordinateFragmentValue:
    base: IdentityTileTensorValue | LocalCoordinateTensorValue
    tv_layout: ThreadValueLayout
    thread_index: ScalarValue

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.tv_layout.shape[1],)

    @property
    def element_type(self) -> str:
        return self.base.element_type

    @property
    def type(self) -> str:
        return f"ThreadCoordinateFragment(shape={self.shape})"

    @property
    def layout(self) -> Layout:
        return Layout.row_major(self.shape)

    def __getitem__(self, index: Any) -> Any:
        normalized = _normalize_runtime_index(index, 1)
        value_index = normalized[0]
        if isinstance(self.base, LocalCoordinateTensorValue):
            coords = self.base.compose_thread_value_coord(self.thread_index, value_index, self.tv_layout)
            if len(coords) == 1:
                return coords[0]
            return coords
        thread_coords = _layout_coords(self.tv_layout.thread_layout, self.thread_index)
        value_coords = _invert_layout_index(self.tv_layout.value_layout, value_index)
        coords = tuple(
            base_offset + (thread_coord * value_extent) + value_coord
            for base_offset, thread_coord, value_extent, value_coord in zip(
                self.base.offset,
                thread_coords,
                self.tv_layout.value_layout.shape,
                value_coords,
            )
        )
        if len(coords) == 1:
            return coords[0]
        return coords


@dataclass(frozen=True)
class ThreadValueComposedCoordinateView:
    base: IdentityTileTensorValue | LocalCoordinateTensorValue
    tv_layout: ThreadValueLayout

    @property
    def shape(self) -> tuple[int, int]:
        return self.tv_layout.shape

    @property
    def element_type(self) -> str:
        return self.base.element_type

    @property
    def type(self) -> str:
        return f"ThreadValueCoord(shape={self.shape})"

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int):
            return self.shape[index]
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("thread/value views expect (thread, value) indexing")
        thread_index, value_index = index
        if not isinstance(thread_index, ScalarValue):
            raise TypeError("traced thread/value coord views require a baybridge scalar thread index")
        if value_index is None:
            return ThreadCoordinateFragmentValue(self.base, self.tv_layout, thread_index)
        normalized = _normalize_runtime_index(value_index, 1)
        return ThreadCoordinateFragmentValue(self.base, self.tv_layout, thread_index)[normalized]


@dataclass(frozen=True)
class ThreadFragmentTensorValue:
    base: TensorValue
    tv_layout: ThreadValueLayout
    thread_index: ScalarValue

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.tv_layout.shape[1],)

    @property
    def element_type(self) -> str:
        return self.base.element_type

    @property
    def type(self) -> str:
        return f"ThreadFragment(shape={self.shape}, element_type={self.element_type})"

    @property
    def layout(self) -> Layout:
        return Layout.row_major(self.shape)

    def load(self, predicate: TensorValue | None = None, *, else_value: int | float = 0) -> TensorValue:
        if predicate is not None:
            if predicate.spec.dtype != "i1":
                raise ValueError("thread fragment load predicates must have dtype i1")
            if predicate.spec.shape != self.shape:
                raise ValueError(
                    f"thread fragment load predicates must have shape {self.shape}, got {predicate.spec.shape}"
                )
        builder = require_builder()
        inputs = [self.base, self.thread_index]
        attrs = {
            "thread_layout": self.tv_layout.thread_layout.to_dict(),
            "value_layout": self.tv_layout.value_layout.to_dict(),
        }
        if predicate is not None:
            inputs.append(predicate)
            attrs["else_value"] = else_value
        return builder.emit_tensor(
            "thread_fragment_load",
            *inputs,
            spec=TensorSpec(
                shape=self.shape,
                dtype=self.base.spec.dtype,
                address_space=AddressSpace.REGISTER,
            ),
            attrs=attrs,
            name_hint=f"{self.base.name}_thread_frag",
        )

    def store(self, value: TensorValue, predicate: TensorValue | None = None) -> None:
        if not isinstance(value, TensorValue):
            raise TypeError("traced thread fragment stores require a baybridge tensor value")
        if value.spec.shape != self.shape:
            raise ValueError(f"fragment store expects shape {self.shape}, got {value.spec.shape}")
        if value.spec.dtype != self.base.spec.dtype:
            raise ValueError(f"fragment store expects dtype {self.base.spec.dtype}, got {value.spec.dtype}")
        inputs = [value, self.base, self.thread_index]
        if predicate is not None:
            if predicate.spec.dtype != "i1":
                raise ValueError("thread fragment store predicates must have dtype i1")
            if predicate.spec.shape != self.shape:
                raise ValueError(
                    f"thread fragment store predicates must have shape {self.shape}, got {predicate.spec.shape}"
                )
            inputs.append(predicate)
        require_builder().emit_void(
            "thread_fragment_store",
            *inputs,
            attrs={
                "thread_layout": self.tv_layout.thread_layout.to_dict(),
                "value_layout": self.tv_layout.value_layout.to_dict(),
            },
        )


@dataclass(frozen=True)
class ThreadValueComposedTensorView:
    base: TensorValue
    tv_layout: ThreadValueLayout

    @property
    def shape(self) -> tuple[int, int]:
        return self.tv_layout.shape

    @property
    def element_type(self) -> str:
        return self.base.element_type

    @property
    def type(self) -> str:
        return f"ThreadValueView(shape={self.shape}, element_type={self.element_type})"

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int):
            return self.shape[index]
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("thread/value views expect (thread, value) indexing")
        thread_index, value_index = index
        if not isinstance(thread_index, ScalarValue):
            raise TypeError("traced thread/value views require a baybridge scalar thread index")
        if value_index is None:
            return ThreadFragmentTensorValue(self.base, self.tv_layout, thread_index)
        raise TypeError("traced thread/value views currently only support (thread, None) slicing")
