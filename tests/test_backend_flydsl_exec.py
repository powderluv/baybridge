import os
import sys
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


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_exec_shared_stage_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor("smem", shape=(4,), dtype="f32", address_space=bb.AddressSpace.SHARED)
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


@bb.kernel
def flydsl_exec_unsupported_kernel(src: bb.Tensor, dst: bb.Tensor):
    tile = bb.local_tile(src, tiler=(2, 4, 2), coord=(1, 0, None), proj=(1, None, 1))
    dst[0] = tile[0, 0, 0]


def _install_fake_flydsl(root: Path, *, built: bool = False, build_dir_name: str = "build-fly") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Fake FlyDSL\n", encoding="utf-8")
    package_root = root / build_dir_name / "python_packages" if built else root
    package = package_root / "flydsl"
    package.mkdir(parents=True, exist_ok=True)
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
        "class AddressSpace:\n"
        "    Register = 'register'\n"
        "    Workgroup = 'workgroup'\n"
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
        "T = _T()\n"
        "def make_layout(shape, stride):\n"
        "    return (_normalize_shape(shape), _normalize_shape(stride))\n"
        "def memref_alloca(memref_type, layout):\n"
        "    del layout\n"
        "    return _MemRef(memref_type['shape'])\n"
        "def memref_load(memref, indices):\n"
        "    return memref.load(indices)\n"
        "def memref_store(value, memref, indices):\n"
        "    memref.store(value, indices)\n"
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


def _install_fake_torch(root: Path) -> Path:
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
        "def tensor(data, dtype=None):\n"
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


def test_flydsl_exec_lowers_python_module(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(flydsl_exec_add_kernel, src, other, dst, cache_dir=tmp_path, backend="flydsl_exec")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "flydsl_python"
    assert "@flyc.kernel" in artifact.lowered_module.text
    assert "@flyc.jit" not in artifact.lowered_module.text
    assert "fx.thread_idx.x" in artifact.lowered_module.text
    assert "dst[" in artifact.lowered_module.text
    assert " + " in artifact.lowered_module.text


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


def test_compile_does_not_auto_prefer_flydsl_exec_for_runtime_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
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
    )

    assert artifact.backend_name != "flydsl_exec"


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
    assert "fx.AddressSpace.Workgroup" in artifact.lowered_module.text

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
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    other = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    dst = torch.zeros(4, dtype=torch.float32)

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


def test_flydsl_exec_copy_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
    dst = torch.zeros(4, dtype=torch.float32)

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


def test_flydsl_exec_reduce_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

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


def test_flydsl_exec_shared_stage_runs_with_real_flydsl_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    torch = _load_real_torch()
    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    dst = torch.zeros(4, dtype=torch.float32)

    artifact = bb.compile(
        flydsl_exec_shared_stage_kernel,
        bb.from_dlpack(src),
        bb.from_dlpack(dst),
        cache_dir=tmp_path / "cache",
        backend="flydsl_exec",
    )
    artifact(src, dst)

    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0]
