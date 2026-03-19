import os
import sys
import math
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.flydsl_exec import FlyDslExecBackend


class FakeDLPackTensor:
    def __init__(self, data):
        self._data = self._clone_nested(data)
        self.shape = self._infer_shape(self._data)
        self.dtype = "f32"

    def __dlpack__(self, stream=None):
        del stream
        return object()

    def __dlpack_device__(self):
        return (10, 0)

    def data_ptr(self):
        return 0

    def stride(self):
        stride = [1] * len(self.shape)
        running = 1
        for index in range(len(self.shape) - 1, -1, -1):
            stride[index] = running
            running *= self.shape[index]
        return tuple(stride)

    def __getitem__(self, index):
        ref = self._data
        if isinstance(index, tuple):
            for item in index:
                ref = ref[item]
            return ref
        return ref[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            ref = self._data
            for item in index[:-1]:
                ref = ref[item]
            ref[index[-1]] = value
            return
        self._data[index] = value

    def tolist(self):
        return self._clone_nested(self._data)

    def _infer_shape(self, value):
        if isinstance(value, list):
            if not value:
                return (0,)
            return (len(value),) + self._infer_shape(value[0])
        return ()

    def _clone_nested(self, value):
        if isinstance(value, list):
            return [self._clone_nested(item) for item in value]
        return value


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_exec_add_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + other[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(4, 1, 1)))
def flydsl_exec_indexed_add_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(4, 1, 1)))
def flydsl_exec_indexed_sub_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] - other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(4, 1, 1)))
def flydsl_exec_indexed_mul_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] * other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(4, 1, 1)))
def flydsl_exec_indexed_div_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] / other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_exec_mul_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] * other[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_exec_sub_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] - other[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_exec_div_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] / other[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_exec_shared_stage_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor("smem", shape=(4,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    smem[tidx] = src[tidx]
    bb.barrier()
    dst[tidx] = smem[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def flydsl_exec_shared_stage_kernel_8(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor("smem", shape=(8,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    smem[tidx] = src[tidx]
    bb.barrier()
    dst[tidx] = smem[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_reduce_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor):
    loaded = src.load()
    dst_scalar[0] = loaded.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(loaded.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_reduce_mul_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor):
    loaded = src.load()
    dst_scalar[0] = loaded.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)
    dst_rows.store(loaded.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_broadcast_add_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_broadcast_sub_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_broadcast_mul_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_broadcast_div_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_tensor_factory_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_math_kernel(
    src: bb.Tensor,
    other: bb.Tensor,
    dst_exp: bb.Tensor,
    dst_log: bb.Tensor,
    dst_cos: bb.Tensor,
    dst_erf: bb.Tensor,
    dst_atan2: bb.Tensor,
):
    values = src.load()
    other_values = other.load()
    dst_exp.store(bb.math.exp(values))
    dst_log.store(bb.math.log(values))
    dst_cos.store(bb.math.cos(values))
    dst_erf.store(bb.math.erf(values))
    dst_atan2.store(bb.math.atan2(values, other_values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_unary_math_kernel(
    src: bb.Tensor,
    dst_exp: bb.Tensor,
    dst_log: bb.Tensor,
    dst_cos: bb.Tensor,
    dst_erf: bb.Tensor,
):
    values = src.load()
    dst_exp.store(bb.math.exp(values))
    dst_log.store(bb.math.log(values))
    dst_cos.store(bb.math.cos(values))
    dst_erf.store(bb.math.erf(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_unary_math_2d_kernel(
    src: bb.Tensor,
    dst_exp: bb.Tensor,
    dst_log: bb.Tensor,
    dst_cos: bb.Tensor,
    dst_erf: bb.Tensor,
):
    values = src.load()
    dst_exp.store(bb.math.exp(values))
    dst_log.store(bb.math.log(values))
    dst_cos.store(bb.math.cos(values))
    dst_erf.store(bb.math.erf(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_unary_math2_kernel(
    src: bb.Tensor,
    dst_exp2: bb.Tensor,
    dst_log2: bb.Tensor,
    dst_log10: bb.Tensor,
    dst_sqrt: bb.Tensor,
):
    values = src.load()
    dst_exp2.store(bb.math.exp2(values))
    dst_log2.store(bb.math.log2(values))
    dst_log10.store(bb.math.log10(values))
    dst_sqrt.store(bb.math.sqrt(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_unary_math3_kernel(
    src: bb.Tensor,
    dst_sin: bb.Tensor,
):
    values = src.load()
    dst_sin.store(bb.math.sin(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_exec_unary_rsqrt_kernel(
    src: bb.Tensor,
    dst_rsqrt: bb.Tensor,
):
    values = src.load()
    dst_rsqrt.store(bb.math.rsqrt(values))


@bb.kernel
def flydsl_exec_unsupported_kernel(src: bb.Tensor, dst: bb.Tensor):
    tile = bb.local_tile(src, tiler=(2, 4, 2), coord=(1, 0, None), proj=(1, None, 1))
    dst[0] = tile[0, 0, 0]


def _install_fake_flydsl(
    root: Path,
    *,
    built: bool = False,
    build_dir_name: str = "build-fly",
    with_mlir: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Fake FlyDSL\n", encoding="utf-8")
    package_root = root / build_dir_name / "python_packages" if built else root
    package = package_root / "flydsl"
    package.mkdir(parents=True, exist_ok=True)
    if with_mlir:
        mlir_package = package / "_mlir"
        dialects_package = mlir_package / "dialects"
        dialects_package.mkdir(parents=True, exist_ok=True)
        (mlir_package / "__init__.py").write_text("", encoding="utf-8")
        (dialects_package / "__init__.py").write_text("", encoding="utf-8")
        (dialects_package / "math.py").write_text(
            "import math\n"
            "def exp(value):\n"
            "    return math.exp(value)\n"
            "def log(value):\n"
            "    return math.log(value)\n"
            "def sin(value):\n"
            "    return math.sin(value)\n"
            "def cos(value):\n"
            "    return math.cos(value)\n"
            "def acos(value):\n"
            "    return math.acos(value)\n"
            "def asin(value):\n"
            "    return math.asin(value)\n"
            "def atan(value):\n"
            "    return math.atan(value)\n"
            "def erf(value):\n"
            "    return math.erf(value)\n"
            "def atan2(lhs, rhs):\n"
            "    return math.atan2(lhs, rhs)\n"
            "def exp2(value):\n"
            "    return 2.0 ** value\n"
            "def log2(value):\n"
            "    return math.log2(value)\n"
            "def log10(value):\n"
            "    return math.log10(value)\n"
            "def sqrt(value):\n"
            "    return math.sqrt(value)\n",
            encoding="utf-8",
        )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "expr.py").write_text(
        "def _normalize_shape(shape):\n"
        "    if isinstance(shape, int):\n"
        "        return (shape,)\n"
        "    return tuple(shape)\n"
        "def _zero_data(shape):\n"
        "    if not shape:\n"
        "        return 0\n"
        "    if len(shape) == 1:\n"
        "        return [0 for _ in range(shape[0])]\n"
        "    return [_zero_data(shape[1:]) for _ in range(shape[0])]\n"
        "def _normalize_indices(indices):\n"
        "    if isinstance(indices, (list, tuple)):\n"
        "        return tuple(indices)\n"
        "    return (indices,)\n"
        "class _MemRef:\n"
        "    def __init__(self, shape):\n"
        "        self.shape = _normalize_shape(shape)\n"
        "        self._data = _zero_data(self.shape)\n"
        "    def load(self, indices):\n"
        "        ref = self._data\n"
        "        for index in _normalize_indices(indices):\n"
        "            ref = ref[index]\n"
        "        return ref\n"
        "    def store(self, value, indices):\n"
        "        indices = _normalize_indices(indices)\n"
        "        ref = self._data\n"
        "        for index in indices[:-1]:\n"
        "            ref = ref[index]\n"
        "        ref[indices[-1]] = value\n"
        "class _Dim3:\n"
        "    def __init__(self):\n"
        "        self.x = 0\n"
        "        self.y = 0\n"
        "        self.z = 0\n"
        "class _Gpu:\n"
        "    def __init__(self):\n"
        "        self.thread_idx = _Dim3()\n"
        "        self.block_idx = _Dim3()\n"
        "        self.block_dim = _Dim3()\n"
        "        self.grid_dim = _Dim3()\n"
        "    def barrier(self):\n"
        "        return None\n"
        "gpu = _Gpu()\n"
        "thread_idx = gpu.thread_idx\n"
        "block_idx = gpu.block_idx\n"
        "block_dim = gpu.block_dim\n"
        "grid_dim = gpu.grid_dim\n"
        "barrier = gpu.barrier\n"
        "class _Rocdl:\n"
        "    @staticmethod\n"
        "    def make_buffer_tensor(value):\n"
        "        return value\n"
        "rocdl = _Rocdl()\n"
        "class _LogicalView:\n"
        "    def __init__(self, base, tile):\n"
        "        self.base = base\n"
        "        self.tile = tile\n"
        "class _TileView:\n"
        "    def __init__(self, base, start, tile):\n"
        "        self.base = base\n"
        "        self.start = start\n"
        "        self.tile = tile\n"
        "class _ElementView:\n"
        "    def __init__(self, base, index):\n"
        "        self.base = base\n"
        "        self.index = index\n"
        "    def load(self):\n"
        "        return self.base[self.index]\n"
        "    def store(self, value):\n"
        "        self.base[self.index] = value\n"
        "class AddressSpace:\n"
        "    Register = 'register'\n"
        "    Shared = 'shared'\n"
        "    Workgroup = Shared\n"
        "class LayoutType:\n"
        "    @staticmethod\n"
        "    def get(shape, stride):\n"
        "        return (_normalize_shape(shape), _normalize_shape(stride))\n"
        "class MemRefType:\n"
        "    @staticmethod\n"
        "    def get(element_type, layout_type, address_space):\n"
        "        shape, stride = layout_type\n"
        "        return {'element_type': element_type, 'shape': shape, 'stride': stride, 'address_space': address_space}\n"
        "class _T:\n"
        "    @staticmethod\n"
        "    def f32():\n"
        "        return 'f32'\n"
        "    @staticmethod\n"
        "    def f16():\n"
        "        return 'f16'\n"
        "    @staticmethod\n"
        "    def bf16():\n"
        "        return 'bf16'\n"
        "    @staticmethod\n"
        "    def i8():\n"
        "        return 'i8'\n"
        "    @staticmethod\n"
        "    def i32():\n"
        "        return 'i32'\n"
        "    @staticmethod\n"
        "    def i64():\n"
        "        return 'i64'\n"
        "    @staticmethod\n"
        "    def vec(width, elem_type):\n"
        "        return ('vec', width, elem_type)\n"
        "    @staticmethod\n"
        "    def vector(*shape, element_type=None, scalable=None, scalable_dims=None):\n"
        "        del scalable, scalable_dims\n"
        "        return ('vec',) + tuple(shape) + (element_type,)\n"
        "T = _T()\n"
        "Float32 = 'f32'\n"
        "def make_layout(shape, stride):\n"
        "    return (_normalize_shape(shape), _normalize_shape(stride))\n"
        "def _tile_from_layout(layout):\n"
        "    shape = layout[0] if isinstance(layout, tuple) else layout\n"
        "    if isinstance(shape, tuple):\n"
        "        return shape[0]\n"
        "    if isinstance(shape, list):\n"
        "        return shape[0]\n"
        "    return shape\n"
        "def logical_divide(value, layout):\n"
        "    return _LogicalView(value, _tile_from_layout(layout))\n"
        "def slice(value, selector):\n"
        "    selector = _normalize_indices(selector)\n"
        "    if not isinstance(value, (_LogicalView, _TileView)) and any(item is None for item in selector):\n"
        "        ref = value\n"
        "        for item in selector:\n"
        "            if item is None:\n"
        "                return ref\n"
        "            ref = ref[int(item)]\n"
        "        return ref\n"
        "    index = selector[-1]\n"
        "    if isinstance(value, _LogicalView):\n"
        "        if isinstance(value.base, _ElementView):\n"
        "            return value.base\n"
        "        if isinstance(value.base, _TileView):\n"
        "            if value.tile == 1:\n"
        "                return _ElementView(value.base.base, value.base.start + index)\n"
        "            return _TileView(value.base.base, value.base.start + index * value.tile, value.tile)\n"
        "        if value.tile == 1:\n"
        "            return _ElementView(value.base, index)\n"
        "        return _TileView(value.base, index * value.tile, value.tile)\n"
        "    if isinstance(value, _TileView):\n"
        "        return _ElementView(value.base, value.start + index)\n"
        "    return _ElementView(value, index)\n"
        "def memref_alloca(memref_type, layout):\n"
        "    del layout\n"
        "    return _MemRef(memref_type['shape'])\n"
        "def memref_load(memref, indices):\n"
        "    if hasattr(memref, 'load'):\n"
        "        return memref.load(indices)\n"
        "    indices = _normalize_indices(indices)\n"
        "    ref = memref\n"
        "    for index in indices:\n"
        "        ref = ref[index]\n"
        "    return ref\n"
        "def memref_store(value, memref, indices):\n"
        "    if hasattr(memref, 'store'):\n"
        "        memref.store(value, indices)\n"
        "        return\n"
        "    indices = _normalize_indices(indices)\n"
        "    ref = memref\n"
        "    for index in indices[:-1]:\n"
        "        ref = ref[index]\n"
        "    ref[indices[-1]] = value\n"
        "def memref_load_vec(memref):\n"
        "    return memref_load(memref, (0,))\n"
        "def memref_store_vec(value, memref):\n"
        "    memref_store(value, memref, (0,))\n"
        "class UniversalCopy32b:\n"
        "    pass\n"
        "def make_copy_atom(copy_op, dtype):\n"
        "    return (copy_op, dtype)\n"
        "def copy_atom_call(copy_atom, src, dst):\n"
        "    del copy_atom\n"
        "    if isinstance(src, _ElementView):\n"
        "        value = src.load()\n"
        "    elif isinstance(src, _TileView):\n"
        "        value = src.base[src.start]\n"
        "    elif hasattr(src, 'load'):\n"
        "        value = memref_load_vec(src)\n"
        "    else:\n"
        "        value = src\n"
        "    if isinstance(dst, _ElementView):\n"
        "        dst.store(value)\n"
        "    elif isinstance(dst, _TileView):\n"
        "        dst.base[dst.start] = value\n"
        "    elif hasattr(dst, 'store'):\n"
        "        memref_store_vec(value, dst)\n"
        "    else:\n"
        "        raise TypeError('unsupported destination for copy_atom_call')\n"
        "class _Arith:\n"
        "    @staticmethod\n"
        "    def constant(value, type=None, index=False):\n"
        "        del type, index\n"
        "        return value\n"
        "    @staticmethod\n"
        "    def addf(lhs, rhs):\n"
        "        return lhs + rhs\n"
        "    @staticmethod\n"
        "    def subf(lhs, rhs):\n"
        "        return lhs - rhs\n"
        "    @staticmethod\n"
        "    def mulf(lhs, rhs):\n"
        "        return lhs * rhs\n"
        "    @staticmethod\n"
        "    def divf(lhs, rhs):\n"
        "        return lhs / rhs\n"
        "arith = _Arith()\n"
        "class _Vector:\n"
        "    @staticmethod\n"
        "    def from_elements(vec_ty, elements):\n"
        "        del vec_ty\n"
        "        return elements[0] if len(elements) == 1 else list(elements)\n"
        "vector = _Vector()\n"
        "class Tensor: pass\n"
        "Int32 = int\n"
        "class Stream:\n"
        "    def __init__(self, value=None):\n"
        "        self.value = value\n",
        encoding="utf-8",
    )
    (package / "compiler.py").write_text(
        "from . import expr as fx\n"
        "from_dlpack_calls = []\n"
        "def from_dlpack(value):\n"
        "    from_dlpack_calls.append(type(value).__name__)\n"
        "    return getattr(value, 'source', value)\n"
        "class _Launchable:\n"
        "    def __init__(self, fn, args, kwargs):\n"
        "        self.fn = fn\n"
        "        self.args = args\n"
        "        self.kwargs = kwargs\n"
        "    def launch(self, grid, block, smem=None, stream=None):\n"
        "        del smem\n"
        "        del stream\n"
        "        gx, gy, gz = tuple(grid)\n"
        "        bx, by, bz = tuple(block)\n"
        "        fx.grid_dim.x, fx.grid_dim.y, fx.grid_dim.z = gx, gy, gz\n"
        "        fx.block_dim.x, fx.block_dim.y, fx.block_dim.z = bx, by, bz\n"
        "        for block_z in range(gz):\n"
        "            for block_y in range(gy):\n"
        "                for block_x in range(gx):\n"
        "                    fx.block_idx.x, fx.block_idx.y, fx.block_idx.z = block_x, block_y, block_z\n"
        "                    for thread_z in range(bz):\n"
        "                        for thread_y in range(by):\n"
        "                            for thread_x in range(bx):\n"
        "                                fx.thread_idx.x, fx.thread_idx.y, fx.thread_idx.z = thread_x, thread_y, thread_z\n"
        "                                self.fn(*self.args, **self.kwargs)\n"
        "        return None\n"
        "class _KernelWrapper:\n"
        "    def __init__(self, fn):\n"
        "        self.fn = fn\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return _Launchable(self.fn, args, kwargs)\n"
        "def kernel(fn):\n"
        "    return _KernelWrapper(fn)\n"
        "def jit(fn):\n"
        "    return fn\n",
        encoding="utf-8",
    )
    return package_root


def _install_broken_source_checkout(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Fake FlyDSL\n", encoding="utf-8")
    package = root / "python" / "flydsl"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "expr.py").write_text("raise ModuleNotFoundError('missing embedded _mlir package')\n", encoding="utf-8")
    (package / "compiler.py").write_text("import torch\n", encoding="utf-8")


def _install_fake_torch(root: Path, *, device_available: bool = True) -> Path:
    package = root / "torch"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        "float32 = 'float32'\n"
        "float16 = 'float16'\n"
        "bfloat16 = 'bfloat16'\n"
        "int8 = 'int8'\n"
        "int32 = 'int32'\n"
        "int64 = 'int64'\n"
        "bool = 'bool'\n"
        "class Tensor:\n"
        "    def __init__(self, data, dtype=None):\n"
        "        self._data = list(data)\n"
        "        self.dtype = dtype\n"
        "        self.shape = (len(self._data),)\n"
        "    def __getitem__(self, index):\n"
        "        return self._data[index]\n"
        "    def __setitem__(self, index, value):\n"
        "        self._data[index] = value\n"
        "    def __dlpack__(self, stream=None):\n"
        "        del stream\n"
        "        return object()\n"
        "    def __dlpack_device__(self):\n"
        "        return (1, 0)\n"
        "    def detach(self):\n"
        "        return self\n"
        "    def cpu(self):\n"
        "        return self\n"
        "    def tolist(self):\n"
        "        return list(self._data)\n"
        f"class cuda:\n"
        f"    @staticmethod\n"
        f"    def is_available():\n"
        f"        return {str(device_available)}\n"
        "def tensor(data, dtype=None, device=None):\n"
        "    del device\n"
        "    return Tensor(data, dtype=dtype)\n",
        encoding="utf-8",
    )
    return root


def _load_real_torch():
    module = sys.modules.get("torch")
    if module is not None and hasattr(module, "zeros"):
        return module
    if module is not None:
        sys.modules.pop("torch", None)
    return pytest.importorskip("torch")


def _skip_if_real_runtime_tensor_exec_unavailable(backend: FlyDslExecBackend) -> None:
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")


def _skip_if_real_exec_unavailable(backend: FlyDslExecBackend) -> None:
    if not backend.real_exec_enabled():
        pytest.skip(backend.real_exec_note())


def _skip_if_real_torch_device_unavailable(torch) -> None:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        pytest.skip("real FlyDSL torch-backed execution requires device-backed torch tensors")


def test_flydsl_exec_lowers_python_module(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(flydsl_exec_add_kernel, src, other, dst, cache_dir=tmp_path, backend="flydsl_exec")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "flydsl_python"
    assert "@flyc.kernel" in artifact.lowered_module.text
    assert "@flyc.jit" in artifact.lowered_module.text
    text = artifact.lowered_module.text
    assert "fx.thread_idx.x" in text
    if "src[thread_idx_x_1]" in text:
        assert "dst[thread_idx_x_1] =" in text
        assert " + " in text
    elif "fx.memref_load(src, (thread_idx_x_1,))" in text:
        assert "fx.memref_load(other, (thread_idx_x_1,))" in text
        assert "fx.memref_store(add_6, dst, (thread_idx_x_1,))" in text
    else:
        assert "fx.logical_divide(src, fx.make_layout(4, 1))" in text
        assert "fx.copy_atom_call(copyAtom, fx.slice(t_src, (None, tid)), r_src)" in text
        assert "fx.arith.addf(" in text


def test_flydsl_exec_available_is_false_for_source_only_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "flydsl_src_only"
    _install_broken_source_checkout(fake_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    backend = FlyDslExecBackend()
    assert backend.available() is False


def test_flydsl_exec_available_detects_custom_build_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "flydsl_custom_build"
    package_root = _install_fake_flydsl(fake_root, built=True, build_dir_name="build-fly-fullmlir")
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    assert backend.available() is True
    assert str(package_root) in environment.search_paths


def test_flydsl_exec_runs_with_fake_built_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    package_root = _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    backend = FlyDslExecBackend()
    assert backend.available() is True
    assert str(package_root) in backend._bridge.exec_environment().search_paths

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    src_handle = bb.from_dlpack(FakeDLPackTensor([1.0, 2.0, 3.0, 4.0]))
    other_handle = bb.from_dlpack(FakeDLPackTensor([10.0, 20.0, 30.0, 40.0]))
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_handle = bb.from_dlpack(dst_obj)

    artifact(src_handle, other_handle, dst_handle, stream=object())
    assert dst_obj.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_compile_auto_prefers_flydsl_exec_with_built_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0])
    other_obj = FakeDLPackTensor([10.0, 20.0, 30.0, 40.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    src = bb.from_dlpack(src_obj)
    other = bb.from_dlpack(other_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst_obj.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_compile_auto_prefers_flydsl_exec_for_raw_dlpack_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0])
    other = FakeDLPackTensor([10.0, 20.0, 30.0, 40.0])
    dst = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_flydsl_exec_indexed_add_runs_with_fake_built_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([float(i) for i in range(16)]))
    other = bb.from_dlpack(FakeDLPackTensor([10.0 + float(i) for i in range(16)]))
    dst_obj = FakeDLPackTensor([0.0 for _ in range(16)])
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_indexed_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    artifact(src, other, dst)
    assert dst_obj.tolist() == [10.0 + 2.0 * float(i) for i in range(16)]


def test_compile_auto_prefers_flydsl_exec_for_runtime_tensors_when_torch_is_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    fake_torch_root = tmp_path / "fake_torch"
    _install_fake_flydsl(fake_root, built=True)
    _install_fake_torch(fake_torch_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]
    sys.modules.pop("torch", None)


def test_compile_does_not_auto_prefer_flydsl_exec_for_unsupported_runtime_tensor_dtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    fake_torch_root = tmp_path / "fake_torch"
    _install_fake_flydsl(fake_root, built=True)
    _install_fake_torch(fake_torch_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([1, 2, 3, 4], dtype="index")
    other = bb.tensor([10, 20, 30, 40], dtype="index")
    dst = bb.zeros((4,), dtype="index")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name != "flydsl_exec"
    sys.modules.pop("torch", None)


def test_compile_does_not_auto_prefer_flydsl_exec_for_runtime_tensors_with_cpu_only_torch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    fake_torch_root = tmp_path / "fake_torch_cpu"
    _install_fake_flydsl(fake_root, built=True)
    _install_fake_torch(fake_torch_root, device_available=False)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name != "flydsl_exec"
    sys.modules.pop("torch", None)


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_built_root_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    fake_torch_root = tmp_path / "fake_torch"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    _install_fake_torch(fake_torch_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]
    sys.modules.pop("torch", None)


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_shared_stage_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    src = bb.from_dlpack(src_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_shared_stage_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst)
    assert dst_obj.tolist() == [1.0, 2.0, 3.0, 4.0]
    sys.modules.pop("torch", None)


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_shared_stage_len8_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    src = bb.from_dlpack(src_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_shared_stage_kernel_8,
        src,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst)
    assert dst_obj.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_copy_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([5.0, 6.0, 7.0, 8.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    src = bb.from_dlpack(src_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst)
    assert dst_obj.tolist() == [5.0, 6.0, 7.0, 8.0]


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_mul_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    fake_torch_root = tmp_path / "fake_torch"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    _install_fake_torch(fake_torch_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([2.0, 3.0, 4.0, 5.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_mul_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [20.0, 60.0, 120.0, 200.0]
    sys.modules.pop("torch", None)


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_indexed_add_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([float(i) for i in range(16)]))
    other = bb.from_dlpack(FakeDLPackTensor([10.0 + float(i) for i in range(16)]))
    dst = bb.from_dlpack(FakeDLPackTensor([0.0 for _ in range(16)]))

    artifact = bb.compile(
        flydsl_exec_indexed_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_indexed_sub_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([20.0 + float(i) for i in range(16)]))
    other = bb.from_dlpack(FakeDLPackTensor([10.0 + float(i) for i in range(16)]))
    dst = bb.from_dlpack(FakeDLPackTensor([0.0 for _ in range(16)]))

    artifact = bb.compile(
        flydsl_exec_indexed_sub_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_indexed_mul_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([1.0 + float(i) for i in range(16)]))
    other = bb.from_dlpack(FakeDLPackTensor([2.0 + float(i) for i in range(16)]))
    dst = bb.from_dlpack(FakeDLPackTensor([0.0 for _ in range(16)]))

    artifact = bb.compile(
        flydsl_exec_indexed_mul_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_indexed_div_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([20.0 + float(i) for i in range(16)]))
    other = bb.from_dlpack(FakeDLPackTensor([2.0 + float(i) for i in range(16)]))
    dst = bb.from_dlpack(FakeDLPackTensor([0.0 for _ in range(16)]))

    artifact = bb.compile(
        flydsl_exec_indexed_div_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_sub_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    fake_torch_root = tmp_path / "fake_torch"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    _install_fake_torch(fake_torch_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([21.0, 22.0, 23.0, 24.0], dtype="f32")
    other = bb.tensor([10.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_sub_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [11.0, 20.0, 20.0, 20.0]
    sys.modules.pop("torch", None)


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_div_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    fake_torch_root = tmp_path / "fake_torch"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    _install_fake_torch(fake_torch_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([20.0, 60.0, 120.0, 200.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_div_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [2.0, 3.0, 4.0, 5.0]
    sys.modules.pop("torch", None)


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_add_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs_obj = FakeDLPackTensor([[1.0], [2.0]])
    rhs_obj = FakeDLPackTensor([[10.0, 20.0, 30.0]])
    dst_obj = FakeDLPackTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    lhs = bb.from_dlpack(lhs_obj)
    rhs = bb.from_dlpack(rhs_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_add_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs_obj = FakeDLPackTensor([[1.0], [2.0], [3.0]])
    rhs_obj = FakeDLPackTensor([[10.0, 20.0]])
    dst_obj = FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    lhs = bb.from_dlpack(lhs_obj)
    rhs = bb.from_dlpack(rhs_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_sub_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[1.0], [2.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[10.0, 20.0, 30.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_sub_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_mul_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[1.0], [2.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[10.0, 20.0, 30.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_mul_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_div_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[10.0], [20.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[2.0, 5.0, 10.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_div_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_sub_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[1.0], [2.0], [3.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[10.0, 20.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_sub_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_mul_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[1.0], [2.0], [3.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[10.0, 20.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_mul_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_broadcast_div_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[10.0], [20.0], [30.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[2.0, 5.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_div_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_tensor_factory_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    dst_zero_obj = FakeDLPackTensor([[9.0, 9.0], [9.0, 9.0]])
    dst_one_obj = FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]])
    dst_full_obj = FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]])

    dst_zero = bb.from_dlpack(dst_zero_obj)
    dst_one = bb.from_dlpack(dst_one_obj)
    dst_full = bb.from_dlpack(dst_full_obj)

    artifact = bb.compile(
        flydsl_exec_tensor_factory_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_tensor_factory_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    dst_zero_obj = FakeDLPackTensor([[9.0], [9.0], [9.0]])
    dst_one_obj = FakeDLPackTensor([[0.0], [0.0], [0.0]])
    dst_full_obj = FakeDLPackTensor([[0.0], [0.0], [0.0]])

    dst_zero = bb.from_dlpack(dst_zero_obj)
    dst_one = bb.from_dlpack(dst_one_obj)
    dst_full = bb.from_dlpack(dst_full_obj)

    artifact = bb.compile(
        flydsl_exec_tensor_factory_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_does_not_auto_prefer_flydsl_exec_for_unvalidated_realish_math_bundle_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0])
    other_obj = FakeDLPackTensor([1.0, 2.0, 8.0])
    dst_exp_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_log_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_cos_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_erf_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_atan2_obj = FakeDLPackTensor([0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_math_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(other_obj),
        bb.from_dlpack(dst_exp_obj),
        bb.from_dlpack(dst_log_obj),
        bb.from_dlpack(dst_cos_obj),
        bb.from_dlpack(dst_erf_obj),
        bb.from_dlpack(dst_atan2_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name != "flydsl_exec"


def test_compile_does_not_auto_prefer_flydsl_exec_for_unvalidated_realish_math_bundle_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0, 8.0])
    other_obj = FakeDLPackTensor([1.0, 2.0, 8.0, 16.0])
    dst_exp_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_log_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_cos_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_erf_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_atan2_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_math_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(other_obj),
        bb.from_dlpack(dst_exp_obj),
        bb.from_dlpack(dst_log_obj),
        bb.from_dlpack(dst_cos_obj),
        bb.from_dlpack(dst_erf_obj),
        bb.from_dlpack(dst_atan2_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name != "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math_bundle_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0])
    dst_exp_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_log_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_cos_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_erf_obj = FakeDLPackTensor([0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_math_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_exp_obj),
        bb.from_dlpack(dst_log_obj),
        bb.from_dlpack(dst_cos_obj),
        bb.from_dlpack(dst_erf_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math_bundle_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0, 8.0])
    dst_exp_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_log_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_cos_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_erf_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_math_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_exp_obj),
        bb.from_dlpack(dst_log_obj),
        bb.from_dlpack(dst_cos_obj),
        bb.from_dlpack(dst_erf_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math_2d_bundle_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([[1.0, 2.0], [4.0, 8.0]]))
    dst_exp = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))
    dst_log = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))
    dst_cos = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))
    dst_erf = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_unary_math_2d_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math_2d_bundle_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([[1.0], [2.0], [4.0]]))
    dst_exp = bb.from_dlpack(FakeDLPackTensor([[0.0], [0.0], [0.0]]))
    dst_log = bb.from_dlpack(FakeDLPackTensor([[0.0], [0.0], [0.0]]))
    dst_cos = bb.from_dlpack(FakeDLPackTensor([[0.0], [0.0], [0.0]]))
    dst_erf = bb.from_dlpack(FakeDLPackTensor([[0.0], [0.0], [0.0]]))

    artifact = bb.compile(
        flydsl_exec_unary_math_2d_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math2_bundle_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0])
    dst_exp2_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_log2_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_log10_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_sqrt_obj = FakeDLPackTensor([0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_math2_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_exp2_obj),
        bb.from_dlpack(dst_log2_obj),
        bb.from_dlpack(dst_log10_obj),
        bb.from_dlpack(dst_sqrt_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math2_bundle_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0, 8.0])
    dst_exp2_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_log2_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_log10_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    dst_sqrt_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_math2_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_exp2_obj),
        bb.from_dlpack(dst_log2_obj),
        bb.from_dlpack(dst_log10_obj),
        bb.from_dlpack(dst_sqrt_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math3_bundle_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([0.0, 0.5, 1.0])
    dst_sin_obj = FakeDLPackTensor([0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_math3_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_sin_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_unary_math3_bundle_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([0.0, 0.25, 0.5, 1.0])
    dst_sin_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_math3_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_sin_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_rsqrt_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0])
    dst_rsqrt_obj = FakeDLPackTensor([0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_rsqrt_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_rsqrt_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_rsqrt_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0, 16.0])
    dst_rsqrt_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_unary_rsqrt_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(dst_rsqrt_obj),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_reduce_bundle_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dst_scalar_obj = FakeDLPackTensor([0.0])
    dst_rows_obj = FakeDLPackTensor([0.0, 0.0])

    src = bb.from_dlpack(src_obj)
    dst_scalar = bb.from_dlpack(dst_scalar_obj)
    dst_rows = bb.from_dlpack(dst_rows_obj)

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_reduce_bundle_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dst_scalar_obj = FakeDLPackTensor([0.0])
    dst_rows_obj = FakeDLPackTensor([0.0, 0.0, 0.0])

    src = bb.from_dlpack(src_obj)
    dst_scalar = bb.from_dlpack(dst_scalar_obj)
    dst_rows = bb.from_dlpack(dst_rows_obj)

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_reduce_mul_bundle_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    dst_scalar = bb.from_dlpack(FakeDLPackTensor([0.0]))
    dst_rows = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_reduce_mul_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_compile_auto_prefers_flydsl_exec_for_validated_realish_reduce_mul_bundle_second_shape_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    dst_scalar = bb.from_dlpack(FakeDLPackTensor([0.0]))
    dst_rows = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_reduce_mul_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"


def test_flydsl_exec_realish_built_root_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[[1.0]], [[2.0]]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]]))
    dst = bb.from_dlpack(
        FakeDLPackTensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]
        )
    )

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    with pytest.raises(bb.BackendNotImplementedError, match="BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC=1"):
        artifact(lhs, rhs, dst)


def test_flydsl_exec_realish_built_root_emits_upstream_pointwise_pattern_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    monkeypatch.setenv("BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC", "1")

    src = bb.from_dlpack(FakeDLPackTensor([1.0, 2.0, 3.0, 4.0]))
    other = bb.from_dlpack(FakeDLPackTensor([10.0, 20.0, 30.0, 40.0]))
    dst = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "fx.logical_divide(src, fx.make_layout(4, 1))" in text
    assert "fx.copy_atom_call(copyAtom, fx.slice(t_src, (None, tid)), r_src)" in text
    assert "fx.arith.addf(" in text


def test_flydsl_exec_realish_built_root_emits_upstream_indexed_pointwise_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([float(i) for i in range(16)]))
    other = bb.from_dlpack(FakeDLPackTensor([10.0 + float(i) for i in range(16)]))
    dst = bb.from_dlpack(FakeDLPackTensor([0.0 for _ in range(16)]))

    artifact = bb.compile(
        flydsl_exec_indexed_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "bid = fx.block_idx.x" in text
    assert "tid = fx.thread_idx.x" in text
    assert "fx.logical_divide(src, fx.make_layout(4, 1))" in text
    assert "fx.copy_atom_call(copyAtom, fx.slice(t_src, (None, tid)), r_src)" in text
    assert "fx.arith.addf(" in text


def test_flydsl_exec_realish_built_root_emits_upstream_broadcast_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[1.0], [2.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[10.0, 20.0, 30.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "row_lhs = fx.slice(lhs, (row_idx, None))" in text
    assert "row_rhs = fx.slice(rhs, (fx.Int32(0), None))" in text
    assert "fx.copy_atom_call(copyAtom, fx.slice(t_lhs, (None, fx.Int32(0))), r_lhs)" in text
    assert "fx.copy_atom_call(copyAtom, fx.slice(t_rhs, (None, col_idx)), r_rhs)" in text


def test_flydsl_exec_realish_built_root_emits_upstream_broadcast_div_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs = bb.from_dlpack(FakeDLPackTensor([[10.0], [20.0]]))
    rhs = bb.from_dlpack(FakeDLPackTensor([[2.0, 5.0, 10.0]]))
    dst = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_broadcast_div_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "fx.arith.divf(" in text
    assert "row_lhs = fx.slice(lhs, (row_idx, None))" in text
    assert "row_rhs = fx.slice(rhs, (fx.Int32(0), None))" in text


def test_flydsl_exec_realish_built_root_emits_upstream_reduce_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    dst_scalar = bb.from_dlpack(FakeDLPackTensor([0.0]))
    dst_rows = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "row_src = fx.slice(src, (row_idx, None))" in text


def test_flydsl_exec_realish_built_root_emits_upstream_reduce_mul_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    dst_scalar = bb.from_dlpack(FakeDLPackTensor([0.0]))
    dst_rows = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_reduce_mul_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "fx.arith.mulf(" in text
    assert "row_src = fx.slice(src, (row_idx, None))" in text


def test_flydsl_exec_realish_built_root_emits_upstream_tensor_factory_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    dst_zero = bb.from_dlpack(FakeDLPackTensor([[9.0, 9.0], [9.0, 9.0]]))
    dst_one = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))
    dst_full = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_tensor_factory_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "row_dst_zero = fx.slice(dst_zero, (row_idx, None))" in text
    assert "row_dst_one = fx.slice(dst_one, (row_idx, None))" in text
    assert "row_dst_full = fx.slice(dst_full, (row_idx, None))" in text
    assert "fx.copy_atom_call(copyAtom, r_dst_zero, fx.slice(t_dst_zero, (None, col_idx)))" in text
    assert "fx.copy_atom_call(copyAtom, r_dst_one, fx.slice(t_dst_one, (None, col_idx)))" in text
    assert "fx.copy_atom_call(copyAtom, r_dst_full, fx.slice(t_dst_full, (None, col_idx)))" in text


def test_flydsl_exec_realish_built_root_emits_upstream_unary_math_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([1.0, 2.0, 4.0]))
    dst_exp = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))
    dst_log = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))
    dst_cos = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))
    dst_erf = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_unary_math_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "fx.copy_atom_call(copyAtom, fx.slice(t_src, (None, math_idx)), r_src)" in text
    assert "fx.memref_store_vec(mlir_math.exp(v_src), r_dst_exp)" in text
    assert "fx.memref_store_vec(mlir_math.erf(v_src), r_dst_erf)" in text


def test_flydsl_exec_realish_built_root_emits_upstream_unary_math_2d_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([[1.0, 2.0], [4.0, 8.0]]))
    dst_exp = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))
    dst_log = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))
    dst_cos = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))
    dst_erf = bb.from_dlpack(FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]]))

    artifact = bb.compile(
        flydsl_exec_unary_math_2d_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "row_src = fx.slice(src, (row_idx, None))" in text
    assert "fx.memref_store_vec(mlir_math.exp(v_src), r_dst_exp)" in text
    assert "fx.memref_store_vec(mlir_math.erf(v_src), r_dst_erf)" in text


def test_flydsl_exec_realish_built_root_emits_upstream_unary_math2_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([1.0, 2.0, 4.0]))
    dst_exp2 = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))
    dst_log2 = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))
    dst_log10 = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))
    dst_sqrt = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_unary_math2_kernel,
        src,
        dst_exp2,
        dst_log2,
        dst_log10,
        dst_sqrt,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "fx.copy_atom_call(copyAtom, fx.slice(t_src, (None, math_idx)), r_src)" in text
    assert "fx.memref_store_vec(mlir_math.exp2(v_src), r_dst_exp2)" in text
    assert "fx.memref_store_vec(mlir_math.log10(v_src), r_dst_log10)" in text
    assert "fx.memref_store_vec(mlir_math.sqrt(v_src), r_dst_sqrt)" in text


def test_flydsl_exec_realish_built_root_emits_upstream_unary_math3_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([0.0, 0.5, 1.0]))
    dst_sin = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_unary_math3_kernel,
        src,
        dst_sin,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "fx.copy_atom_call(copyAtom, fx.slice(t_src, (None, math_idx)), r_src)" in text
    assert "fx.memref_store_vec(mlir_math.sin(v_src), r_dst_sin)" in text


def test_flydsl_exec_realish_built_root_emits_upstream_rsqrt_pattern_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.from_dlpack(FakeDLPackTensor([1.0, 2.0, 4.0]))
    dst_rsqrt = bb.from_dlpack(FakeDLPackTensor([0.0, 0.0, 0.0]))

    artifact = bb.compile(
        flydsl_exec_unary_rsqrt_kernel,
        src,
        dst_rsqrt,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    text = artifact.lowered_module.text
    assert "v_one = fx.vector.from_elements(vec1f32, [c_one])" in text
    assert "fx.memref_store_vec(fx.arith.divf(v_one, mlir_math.sqrt(v_src)), r_dst_rsqrt)" in text


def test_flydsl_exec_adapts_tensor_handles_via_from_dlpack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0])
    other_obj = FakeDLPackTensor([10.0, 20.0, 30.0, 40.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    src = bb.from_dlpack(src_obj)
    other = bb.from_dlpack(other_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst_obj.tolist() == [11.0, 22.0, 33.0, 44.0]

    import sys

    compiler_module = sys.modules.get("flydsl.compiler")
    assert compiler_module is not None
    assert compiler_module.from_dlpack_calls[-3:] == ["TensorHandle", "TensorHandle", "TensorHandle"]
    assert str(fake_root.resolve()) in artifact.lowered_module.text


def test_flydsl_exec_reuses_from_dlpack_adaptation_for_repeated_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0])
    other_obj = FakeDLPackTensor([10.0, 20.0, 30.0, 40.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    src = bb.from_dlpack(src_obj)
    other = bb.from_dlpack(other_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    import sys

    compiler_module = sys.modules.get("flydsl.compiler")
    assert compiler_module is not None
    before = len(compiler_module.from_dlpack_calls)

    artifact(src, other, dst)
    after_first = len(compiler_module.from_dlpack_calls)
    artifact(src, other, dst)
    after_second = len(compiler_module.from_dlpack_calls)

    assert after_first - before == 3
    assert after_second == after_first
    assert dst_obj.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_flydsl_exec_adapts_runtime_tensors_via_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    fake_torch_root = tmp_path / "fake_torch"
    _install_fake_flydsl(fake_root, built=True)
    _install_fake_torch(fake_torch_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(fake_torch_root) if not existing_pythonpath else f"{fake_torch_root}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.syspath_prepend(str(fake_torch_root))
    sys.modules.pop("torch", None)

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]

    compiler_module = sys.modules.get("flydsl.compiler")
    assert compiler_module is not None
    assert compiler_module.from_dlpack_calls[-3:] == ["Tensor", "Tensor", "Tensor"]
    sys.modules.pop("torch", None)


def test_flydsl_exec_shared_stage_kernel_uses_memref_alloca(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_shared_stage_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    assert "fx.memref_alloca" in artifact.lowered_module.text
    assert "gpu.barrier()" in artifact.lowered_module.text
    assert "fx.AddressSpace.Shared" in artifact.lowered_module.text

    src_obj = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])
    artifact(bb.from_dlpack(src_obj), bb.from_dlpack(dst_obj))
    assert dst_obj.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_flydsl_exec_copy_kernel_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 3.0, 4.0])
    dst_obj = FakeDLPackTensor([0.0, 0.0, 0.0, 0.0])

    src = bb.from_dlpack(src_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst)

    assert dst_obj.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert "for _bb_copy_i0 in range(4):" in artifact.lowered_module.text


def test_flydsl_exec_reduce_kernel_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dst_scalar_obj = FakeDLPackTensor([0.0])
    dst_rows_obj = FakeDLPackTensor([0.0, 0.0])

    src = bb.from_dlpack(src_obj)
    dst_scalar = bb.from_dlpack(dst_scalar_obj)
    dst_rows = bb.from_dlpack(dst_rows_obj)

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst_scalar, dst_rows)

    assert dst_scalar_obj.tolist() == [21.0]
    assert dst_rows_obj.tolist() == [6.0, 15.0]
    assert "reduce_add" in artifact.lowered_module.text


def test_flydsl_exec_broadcast_add_kernel_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    lhs_obj = FakeDLPackTensor([[1.0], [2.0]])
    rhs_obj = FakeDLPackTensor([[10.0, 20.0, 30.0]])
    dst_obj = FakeDLPackTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    lhs = bb.from_dlpack(lhs_obj)
    rhs = bb.from_dlpack(rhs_obj)
    dst = bb.from_dlpack(dst_obj)

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(lhs, rhs, dst)

    assert dst_obj.tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
    assert "broadcast_to" in artifact.lowered_module.text
    assert "tensor_add" in artifact.lowered_module.text


def test_flydsl_exec_tensor_factory_kernel_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    dst_zero_obj = FakeDLPackTensor([[9.0, 9.0], [9.0, 9.0]])
    dst_one_obj = FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]])
    dst_full_obj = FakeDLPackTensor([[0.0, 0.0], [0.0, 0.0]])

    dst_zero = bb.from_dlpack(dst_zero_obj)
    dst_one = bb.from_dlpack(dst_one_obj)
    dst_full = bb.from_dlpack(dst_full_obj)

    artifact = bb.compile(
        flydsl_exec_tensor_factory_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(dst_zero, dst_one, dst_full)

    assert dst_zero_obj.tolist() == [[0, 0], [0, 0]]
    assert dst_one_obj.tolist() == [[1, 1], [1, 1]]
    assert dst_full_obj.tolist() == [[7.0, 7.0], [7.0, 7.0]]
    assert "fill" in artifact.lowered_module.text


def test_flydsl_exec_math_kernel_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src_obj = FakeDLPackTensor([1.0, 2.0, 4.0])
    other_obj = FakeDLPackTensor([1.0, 2.0, 8.0])
    dst_exp_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_log_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_cos_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_erf_obj = FakeDLPackTensor([0.0, 0.0, 0.0])
    dst_atan2_obj = FakeDLPackTensor([0.0, 0.0, 0.0])

    artifact = bb.compile(
        flydsl_exec_math_kernel,
        bb.from_dlpack(src_obj),
        bb.from_dlpack(other_obj),
        bb.from_dlpack(dst_exp_obj),
        bb.from_dlpack(dst_log_obj),
        bb.from_dlpack(dst_cos_obj),
        bb.from_dlpack(dst_erf_obj),
        bb.from_dlpack(dst_atan2_obj),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(
        bb.from_dlpack(src_obj),
        bb.from_dlpack(other_obj),
        bb.from_dlpack(dst_exp_obj),
        bb.from_dlpack(dst_log_obj),
        bb.from_dlpack(dst_cos_obj),
        bb.from_dlpack(dst_erf_obj),
        bb.from_dlpack(dst_atan2_obj),
    )

    assert dst_exp_obj.tolist() == pytest.approx([math.exp(1.0), math.exp(2.0), math.exp(4.0)], rel=1e-6, abs=1e-6)
    assert dst_log_obj.tolist() == pytest.approx([math.log(1.0), math.log(2.0), math.log(4.0)], rel=1e-6, abs=1e-6)
    assert dst_cos_obj.tolist() == pytest.approx([math.cos(1.0), math.cos(2.0), math.cos(4.0)], rel=1e-6, abs=1e-6)
    assert dst_erf_obj.tolist() == pytest.approx([math.erf(1.0), math.erf(2.0), math.erf(4.0)], rel=1e-6, abs=1e-6)
    assert dst_atan2_obj.tolist() == pytest.approx(
        [math.atan2(1.0, 1.0), math.atan2(2.0, 2.0), math.atan2(4.0, 8.0)],
        rel=1e-6,
        abs=1e-6,
    )
    assert "math.exp" in artifact.lowered_module.text
    assert "math.atan2" in artifact.lowered_module.text


def test_flydsl_exec_launcher_explains_unusable_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "flydsl_src_only"
    _install_broken_source_checkout(fake_root)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    with pytest.raises(bb.BackendNotImplementedError, match="built and importable FlyDSL environment"):
        artifact(src, other, dst)


def test_flydsl_exec_rejects_unsupported_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0] * 8 for _ in range(4)], dtype="f32")
    dst = bb.zeros((1,), dtype="f32")

    with pytest.raises(bb.BackendNotImplementedError):
        bb.compile(flydsl_exec_unsupported_kernel, src, dst, cache_dir=tmp_path, backend="flydsl_exec")


def test_flydsl_exec_generated_module_imports_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    assert artifact.lowered_path is not None
    module = backend._load_module("_baybridge_real_flydsl_probe", artifact.lowered_path)
    assert hasattr(module, "launch_flydsl_exec_add_kernel")


def test_flydsl_exec_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    other = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(other),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_flydsl_exec_indexed_add_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([10.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == pytest.approx([10.0 + 2.0 * float(i) for i in range(16)], rel=1e-6, abs=1e-6)


def test_flydsl_exec_indexed_sub_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([20.0 + float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([10.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_sub_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == pytest.approx([10.0 for _ in range(16)], rel=1e-6, abs=1e-6)


def test_flydsl_exec_indexed_mul_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0 + float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([2.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_mul_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == pytest.approx(
        [(1.0 + float(i)) * (2.0 + float(i)) for i in range(16)],
        rel=1e-6,
        abs=1e-6,
    )


def test_flydsl_exec_indexed_div_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([20.0 + float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([2.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_div_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == pytest.approx(
        [(20.0 + float(i)) / (2.0 + float(i)) for i in range(16)],
        rel=1e-6,
        abs=1e-6,
    )


def test_flydsl_exec_mul_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
    other = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_mul_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(other),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == [20.0, 60.0, 120.0, 200.0]


def test_flydsl_exec_sub_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([21.0, 22.0, 23.0, 24.0], dtype=torch.float32, device="cuda")
    other = torch.tensor([10.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_sub_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(other),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == [11.0, 20.0, 20.0, 20.0]


def test_flydsl_exec_div_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([20.0, 60.0, 120.0, 200.0], dtype=torch.float32, device="cuda")
    other = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_div_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(other),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_flydsl_exec_copy_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_copy_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst)

    assert dst.tolist() == [5.0, 6.0, 7.0, 8.0]


def test_flydsl_exec_runtime_tensors_run_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst)

    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_compile_auto_prefers_flydsl_exec_for_real_torch_inputs_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    other = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_compile_auto_prefers_flydsl_exec_for_real_runtime_tensors_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_compile_auto_prefers_flydsl_exec_for_real_indexed_add_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    src = bb.tensor([float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([10.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == pytest.approx([10.0 + 2.0 * float(i) for i in range(16)], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_indexed_sub_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    src = bb.tensor([20.0 + float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([10.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_sub_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == pytest.approx([10.0 for _ in range(16)], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_indexed_mul_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    src = bb.tensor([1.0 + float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([2.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_mul_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == pytest.approx(
        [(1.0 + float(i)) * (2.0 + float(i)) for i in range(16)],
        rel=1e-6,
        abs=1e-6,
    )


def test_compile_auto_prefers_flydsl_exec_for_real_indexed_div_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    src = bb.tensor([20.0 + float(i) for i in range(16)], dtype="f32")
    other = bb.tensor([2.0 + float(i) for i in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_indexed_div_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, other, dst)
    assert dst.tolist() == pytest.approx(
        [(20.0 + float(i)) / (2.0 + float(i)) for i in range(16)],
        rel=1e-6,
        abs=1e-6,
    )


def test_compile_auto_prefers_flydsl_exec_for_real_broadcast_add_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    lhs = bb.tensor([[1.0], [2.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0, 30.0]], dtype="f32")
    dst = bb.zeros((2, 3), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]


def test_compile_auto_prefers_flydsl_exec_for_real_broadcast_add_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    lhs = bb.tensor([[1.0], [2.0], [3.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0]], dtype="f32")
    dst = bb.zeros((3, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]


def test_compile_auto_prefers_flydsl_exec_for_real_broadcast_sub_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    lhs = bb.tensor([[1.0], [2.0], [3.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0]], dtype="f32")
    dst = bb.zeros((3, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_sub_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[-9.0, -19.0], [-8.0, -18.0], [-7.0, -17.0]]


def test_compile_auto_prefers_flydsl_exec_for_real_broadcast_mul_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    lhs = bb.tensor([[1.0], [2.0], [3.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0]], dtype="f32")
    dst = bb.zeros((3, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_mul_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]]


def test_compile_auto_prefers_flydsl_exec_for_real_broadcast_div_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    if not environment.torch_available:
        pytest.skip("real FlyDSL runtime-tensor adaptation requires torch")
    if not backend._runtime_tensor_device_available():
        pytest.skip("real FlyDSL RuntimeTensor execution requires a GPU-capable torch build")

    lhs = bb.tensor([[10.0], [20.0], [30.0]], dtype="f32")
    rhs = bb.tensor([[2.0, 5.0]], dtype="f32")
    dst = bb.zeros((3, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_div_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[5.0, 2.0], [10.0, 4.0], [15.0, 6.0]]


def test_compile_auto_prefers_flydsl_exec_for_real_shared_stage_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_exec_unavailable(backend)

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_shared_stage_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst)
    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_compile_auto_prefers_flydsl_exec_for_real_shared_stage_len8_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_exec_unavailable(backend)

    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(8, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_shared_stage_kernel_8,
        bb.from_dlpack(src),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst)
    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_flydsl_exec_reduce_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst_scalar, dst_rows)

    assert dst_scalar.tolist() == [21.0]
    assert dst_rows.tolist() == [6.0, 15.0]


def test_flydsl_exec_reduce_second_shape_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst_scalar, dst_rows)

    assert dst_scalar.tolist() == [21.0]
    assert dst_rows.tolist() == [3.0, 7.0, 11.0]


def test_compile_auto_prefers_flydsl_exec_for_real_reduce_bundle_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_scalar, dst_rows)
    assert dst_scalar.tolist() == [21.0]
    assert dst_rows.tolist() == [6.0, 15.0]


def test_compile_auto_prefers_flydsl_exec_for_real_reduce_bundle_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_reduce_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_scalar, dst_rows)
    assert dst_scalar.tolist() == [21.0]
    assert dst_rows.tolist() == [3.0, 7.0, 11.0]


def test_flydsl_exec_reduce_mul_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_reduce_mul_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst_scalar, dst_rows)

    assert dst_scalar.tolist() == pytest.approx([720.0], rel=1e-6, abs=1e-6)
    assert dst_rows.tolist() == pytest.approx([6.0, 120.0], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_reduce_mul_bundle_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_reduce_mul_kernel,
        src,
        dst_scalar,
        dst_rows,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_scalar, dst_rows)
    assert dst_scalar.tolist() == pytest.approx([720.0], rel=1e-6, abs=1e-6)
    assert dst_rows.tolist() == pytest.approx([2.0, 12.0, 30.0], rel=1e-6, abs=1e-6)


def test_flydsl_exec_broadcast_add_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    lhs = bb.tensor([[1.0], [2.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0, 30.0]], dtype="f32")
    dst = bb.zeros((2, 3), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(lhs, rhs, dst)

    assert dst.tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]


def test_flydsl_exec_broadcast_add_second_shape_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    lhs = bb.tensor([[1.0], [2.0], [3.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0]], dtype="f32")
    dst = bb.zeros((3, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_add_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(lhs, rhs, dst)

    assert dst.tolist() == [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]


def test_flydsl_exec_broadcast_sub_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    lhs = bb.tensor([[1.0], [2.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0, 30.0]], dtype="f32")
    dst = bb.zeros((2, 3), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_sub_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(lhs, rhs, dst)

    assert dst.tolist() == [[-9.0, -19.0, -29.0], [-8.0, -18.0, -28.0]]


def test_flydsl_exec_broadcast_mul_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    lhs = bb.tensor([[1.0], [2.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0, 30.0]], dtype="f32")
    dst = bb.zeros((2, 3), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_mul_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(lhs, rhs, dst)

    assert dst.tolist() == [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]]


def test_flydsl_exec_broadcast_div_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    lhs = bb.tensor([[10.0], [20.0]], dtype="f32")
    rhs = bb.tensor([[2.0, 5.0, 10.0]], dtype="f32")
    dst = bb.zeros((2, 3), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_broadcast_div_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(lhs, rhs, dst)

    assert dst.tolist() == [[5.0, 2.0, 1.0], [10.0, 4.0, 2.0]]


def test_flydsl_exec_tensor_factory_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    dst_zero = bb.tensor([[9.0, 9.0], [9.0, 9.0]], dtype="f32")
    dst_one = bb.zeros((2, 2), dtype="f32")
    dst_full = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_tensor_factory_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(dst_zero, dst_one, dst_full)

    assert dst_zero.tolist() == [[0, 0], [0, 0]]
    assert dst_one.tolist() == [[1, 1], [1, 1]]
    assert dst_full.tolist() == [[7.0, 7.0], [7.0, 7.0]]


def test_compile_auto_prefers_flydsl_exec_for_real_tensor_factory_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    dst_zero = bb.tensor([[9.0, 9.0], [9.0, 9.0]], dtype="f32")
    dst_one = bb.zeros((2, 2), dtype="f32")
    dst_full = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_tensor_factory_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist() == [[0, 0], [0, 0]]
    assert dst_one.tolist() == [[1, 1], [1, 1]]
    assert dst_full.tolist() == [[7.0, 7.0], [7.0, 7.0]]


def test_compile_auto_prefers_flydsl_exec_for_real_tensor_factory_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    dst_zero = bb.tensor([[9.0], [9.0], [9.0]], dtype="f32")
    dst_one = bb.zeros((3, 1), dtype="f32")
    dst_full = bb.zeros((3, 1), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_tensor_factory_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist() == [[0], [0], [0]]
    assert dst_one.tolist() == [[1], [1], [1]]
    assert dst_full.tolist() == [[7.0], [7.0], [7.0]]


def test_flydsl_exec_math_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)
    pytest.xfail("real FlyDSL lowering still does not support atan2 cleanly in the current upstream pipeline")

    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    other = bb.tensor([1.0, 2.0, 8.0], dtype="f32")
    dst_exp = bb.zeros((3,), dtype="f32")
    dst_log = bb.zeros((3,), dtype="f32")
    dst_cos = bb.zeros((3,), dtype="f32")
    dst_erf = bb.zeros((3,), dtype="f32")
    dst_atan2 = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_math_kernel,
        src,
        other,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        dst_atan2,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, other, dst_exp, dst_log, dst_cos, dst_erf, dst_atan2)

    assert dst_exp.tolist() == pytest.approx([math.exp(1.0), math.exp(2.0), math.exp(4.0)], rel=1e-6, abs=1e-6)
    assert dst_log.tolist() == pytest.approx([math.log(1.0), math.log(2.0), math.log(4.0)], rel=1e-6, abs=1e-6)
    assert dst_cos.tolist() == pytest.approx([math.cos(1.0), math.cos(2.0), math.cos(4.0)], rel=1e-6, abs=1e-6)
    assert dst_erf.tolist() == pytest.approx([math.erf(1.0), math.erf(2.0), math.erf(4.0)], rel=1e-6, abs=1e-6)
    assert dst_atan2.tolist() == pytest.approx(
        [math.atan2(1.0, 1.0), math.atan2(2.0, 2.0), math.atan2(4.0, 8.0)],
        rel=1e-6,
        abs=1e-6,
    )


def test_flydsl_exec_unary_math_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    dst_exp = bb.zeros((3,), dtype="f32")
    dst_log = bb.zeros((3,), dtype="f32")
    dst_cos = bb.zeros((3,), dtype="f32")
    dst_erf = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    artifact(src, dst_exp, dst_log, dst_cos, dst_erf)
    assert dst_exp.tolist() == pytest.approx([math.exp(1.0), math.exp(2.0), math.exp(4.0)], rel=1e-6, abs=1e-6)
    assert dst_log.tolist() == pytest.approx([math.log(1.0), math.log(2.0), math.log(4.0)], rel=1e-6, abs=1e-6)
    assert dst_cos.tolist() == pytest.approx([math.cos(1.0), math.cos(2.0), math.cos(4.0)], rel=1e-6, abs=1e-6)
    assert dst_erf.tolist() == pytest.approx([math.erf(1.0), math.erf(2.0), math.erf(4.0)], rel=1e-6, abs=1e-6)


def test_flydsl_exec_unary_math_2d_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0], [4.0, 8.0]], dtype="f32")
    dst_exp = bb.zeros((2, 2), dtype="f32")
    dst_log = bb.zeros((2, 2), dtype="f32")
    dst_cos = bb.zeros((2, 2), dtype="f32")
    dst_erf = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math_2d_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    artifact(src, dst_exp, dst_log, dst_cos, dst_erf)
    assert dst_exp.tolist()[0] == pytest.approx([math.exp(1.0), math.exp(2.0)], rel=1e-6, abs=1e-6)
    assert dst_exp.tolist()[1] == pytest.approx([math.exp(4.0), math.exp(8.0)], rel=1e-6, abs=1e-6)
    assert dst_log.tolist()[0] == pytest.approx([math.log(1.0), math.log(2.0)], rel=1e-6, abs=1e-6)
    assert dst_log.tolist()[1] == pytest.approx([math.log(4.0), math.log(8.0)], rel=1e-6, abs=1e-6)
    assert dst_cos.tolist()[0] == pytest.approx([math.cos(1.0), math.cos(2.0)], rel=1e-6, abs=1e-6)
    assert dst_cos.tolist()[1] == pytest.approx([math.cos(4.0), math.cos(8.0)], rel=1e-6, abs=1e-6)
    assert dst_erf.tolist()[0] == pytest.approx([math.erf(1.0), math.erf(2.0)], rel=1e-6, abs=1e-6)
    assert dst_erf.tolist()[1] == pytest.approx([math.erf(4.0), math.erf(8.0)], rel=1e-6, abs=1e-6)


def test_flydsl_exec_unary_math2_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    dst_exp2 = bb.zeros((3,), dtype="f32")
    dst_log2 = bb.zeros((3,), dtype="f32")
    dst_log10 = bb.zeros((3,), dtype="f32")
    dst_sqrt = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math2_kernel,
        src,
        dst_exp2,
        dst_log2,
        dst_log10,
        dst_sqrt,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    artifact(src, dst_exp2, dst_log2, dst_log10, dst_sqrt)
    assert dst_exp2.tolist() == pytest.approx([2.0, 4.0, 16.0], rel=1e-6, abs=1e-6)
    assert dst_log2.tolist() == pytest.approx([0.0, 1.0, 2.0], rel=1e-6, abs=1e-6)
    assert dst_log10.tolist() == pytest.approx([0.0, math.log10(2.0), math.log10(4.0)], rel=1e-6, abs=1e-6)
    assert dst_sqrt.tolist() == pytest.approx([1.0, math.sqrt(2.0), 2.0], rel=1e-6, abs=1e-6)


def test_flydsl_exec_unary_math3_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([0.0, 0.5, 1.0], dtype="f32")
    dst_sin = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math3_kernel,
        src,
        dst_sin,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    artifact(src, dst_sin)
    assert dst_sin.tolist() == pytest.approx([math.sin(0.0), math.sin(0.5), math.sin(1.0)], rel=1e-6, abs=1e-6)


def test_flydsl_exec_rsqrt_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    dst_rsqrt = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_rsqrt_kernel,
        src,
        dst_rsqrt,
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )

    artifact(src, dst_rsqrt)
    assert dst_rsqrt.tolist() == pytest.approx([1.0, 1.0 / math.sqrt(2.0), 0.5], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math_bundle_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    dst_exp = bb.zeros((3,), dtype="f32")
    dst_log = bb.zeros((3,), dtype="f32")
    dst_cos = bb.zeros((3,), dtype="f32")
    dst_erf = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_exp, dst_log, dst_cos, dst_erf)
    assert dst_exp.tolist() == pytest.approx(
        [math.exp(1.0), math.exp(2.0), math.exp(4.0)],
        rel=1e-6,
        abs=1e-6,
    )


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math_bundle_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0, 8.0], dtype="f32")
    dst_exp = bb.zeros((4,), dtype="f32")
    dst_log = bb.zeros((4,), dtype="f32")
    dst_cos = bb.zeros((4,), dtype="f32")
    dst_erf = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_exp, dst_log, dst_cos, dst_erf)
    assert dst_exp.tolist() == pytest.approx(
        [math.exp(1.0), math.exp(2.0), math.exp(4.0), math.exp(8.0)],
        rel=1e-6,
        abs=1e-6,
    )


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math_2d_bundle_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0, 2.0], [4.0, 8.0]], dtype="f32")
    dst_exp = bb.zeros((2, 2), dtype="f32")
    dst_log = bb.zeros((2, 2), dtype="f32")
    dst_cos = bb.zeros((2, 2), dtype="f32")
    dst_erf = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math_2d_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_exp, dst_log, dst_cos, dst_erf)
    assert dst_exp.tolist()[0] == pytest.approx([math.exp(1.0), math.exp(2.0)], rel=1e-6, abs=1e-6)
    assert dst_exp.tolist()[1] == pytest.approx([math.exp(4.0), math.exp(8.0)], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math_2d_bundle_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([[1.0], [2.0], [4.0]], dtype="f32")
    dst_exp = bb.zeros((3, 1), dtype="f32")
    dst_log = bb.zeros((3, 1), dtype="f32")
    dst_cos = bb.zeros((3, 1), dtype="f32")
    dst_erf = bb.zeros((3, 1), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math_2d_kernel,
        src,
        dst_exp,
        dst_log,
        dst_cos,
        dst_erf,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_exp, dst_log, dst_cos, dst_erf)
    assert dst_exp.tolist()[0] == pytest.approx([math.exp(1.0)], rel=1e-6, abs=1e-6)
    assert dst_exp.tolist()[1] == pytest.approx([math.exp(2.0)], rel=1e-6, abs=1e-6)
    assert dst_exp.tolist()[2] == pytest.approx([math.exp(4.0)], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math2_bundle_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    dst_exp2 = bb.zeros((3,), dtype="f32")
    dst_log2 = bb.zeros((3,), dtype="f32")
    dst_log10 = bb.zeros((3,), dtype="f32")
    dst_sqrt = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math2_kernel,
        src,
        dst_exp2,
        dst_log2,
        dst_log10,
        dst_sqrt,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_exp2, dst_log2, dst_log10, dst_sqrt)
    assert dst_exp2.tolist() == pytest.approx([2.0, 4.0, 16.0], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math2_bundle_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0, 8.0], dtype="f32")
    dst_exp2 = bb.zeros((4,), dtype="f32")
    dst_log2 = bb.zeros((4,), dtype="f32")
    dst_log10 = bb.zeros((4,), dtype="f32")
    dst_sqrt = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math2_kernel,
        src,
        dst_exp2,
        dst_log2,
        dst_log10,
        dst_sqrt,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_exp2, dst_log2, dst_log10, dst_sqrt)
    assert dst_exp2.tolist() == pytest.approx([2.0, 4.0, 16.0, 256.0], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math3_bundle_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([0.0, 0.5, 1.0], dtype="f32")
    dst_sin = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math3_kernel,
        src,
        dst_sin,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_sin)
    assert dst_sin.tolist() == pytest.approx([math.sin(0.0), math.sin(0.5), math.sin(1.0)], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_unary_math3_bundle_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([0.0, 0.25, 0.5, 1.0], dtype="f32")
    dst_sin = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_math3_kernel,
        src,
        dst_sin,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_sin)
    assert dst_sin.tolist() == pytest.approx(
        [math.sin(0.0), math.sin(0.25), math.sin(0.5), math.sin(1.0)],
        rel=1e-6,
        abs=1e-6,
    )


def test_compile_auto_prefers_flydsl_exec_for_real_rsqrt_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    dst_rsqrt = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_rsqrt_kernel,
        src,
        dst_rsqrt,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_rsqrt)
    assert dst_rsqrt.tolist() == pytest.approx([1.0, 1.0 / math.sqrt(2.0), 0.5], rel=1e-6, abs=1e-6)


def test_compile_auto_prefers_flydsl_exec_for_real_rsqrt_second_shape_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_runtime_tensor_exec_unavailable(backend)

    src = bb.tensor([1.0, 2.0, 4.0, 16.0], dtype="f32")
    dst_rsqrt = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        flydsl_exec_unary_rsqrt_kernel,
        src,
        dst_rsqrt,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "flydsl_exec"
    artifact(src, dst_rsqrt)
    assert dst_rsqrt.tolist() == pytest.approx(
        [1.0, 1.0 / math.sqrt(2.0), 0.5, 0.25],
        rel=1e-6,
        abs=1e-6,
    )


def test_flydsl_exec_shared_stage_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    _skip_if_real_torch_device_unavailable(torch)
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")
    _skip_if_real_exec_unavailable(backend)

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    dst = torch.zeros(4, dtype=torch.float32, device="cuda")

    artifact = bb.compile(
        flydsl_exec_shared_stage_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst)

    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0]
