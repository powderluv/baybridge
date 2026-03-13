import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.flydsl_exec import FlyDslExecBackend
from baybridge.backends.waveasm_ref import WaveAsmRefBackend
from test_backend_flydsl_exec import _install_fake_flydsl, _install_fake_torch


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(64, 1, 1), cluster=(2, 1, 1), shared_mem_bytes=256))
def waveasm_scaffold_kernel(
    src: bb.TensorSpec(shape=(64,), dtype="f32"),
    dst: bb.TensorSpec(shape=(64,), dtype="f32"),
):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor("smem", shape=(64,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    bb.copy(src, smem, vector_bytes=16)
    bb.barrier()
    dst[tidx] = smem[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def waveasm_reduce_kernel(
    src: bb.TensorSpec(shape=(2, 3), dtype="f32"),
    dst: bb.TensorSpec(shape=(2,), dtype="f32"),
):
    dst.store(src.load().reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


def test_gpu_mlir_backend_emits_module_scaffold(tmp_path: Path) -> None:
    artifact = bb.compile(waveasm_scaffold_kernel, cache_dir=tmp_path, backend="gpu_mlir")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert artifact.lowered_module.dialect == "gpu_mlir"
    assert 'module attributes {baybridge.target = "gfx942", rocdl.wave_size = 64} {' in text
    assert 'gpu.module @kernels attributes {rocdl.target = "gfx942", rocdl.wave_size = 64}' in text
    assert (
        'gpu.func @waveasm_scaffold_kernel(%src: memref<64xf32, strided<[1], offset: 0>, 1>, '
        '%dst: memref<64xf32, strided<[1], offset: 0>, 1>) kernel attributes '
        '{gpu.grid = [4, 1, 1], gpu.block = [64, 1, 1], gpu.dynamic_shared_memory = 256, '
        'rocdl.target = "gfx942", gpu.cluster = [2, 1, 1]}'
    ) in text
    assert "gpu.barrier" in text
    assert "gpu.return" in text


def test_waveasm_ref_backend_emits_waveasm_tool_hints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    tool_path = wave_root / "build" / "bin" / "waveasm-translate"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))

    artifact = bb.compile(waveasm_scaffold_kernel, cache_dir=tmp_path / "cache", backend="waveasm_ref")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert artifact.lowered_module.dialect == "waveasm_mlir"
    assert "// baybridge.waveasm_ref" in text
    assert f"// configured_root: {wave_root}" in text
    assert f"// waveasm_translate: {tool_path}" in text
    assert "// suggested_pipeline:" in text
    assert 'module attributes {waveasm.target = "gfx942", waveasm.wave_size = 64,' in text
    assert "memref<64xf32, #gpu.address_space<workgroup>>" in text
    assert artifact.debug_bundle_dir is not None
    repro_dir = artifact.debug_bundle_dir
    assert (repro_dir / "kernel.mlir").exists()
    assert (repro_dir / "manifest.json").exists()
    repro_script = repro_dir / "repro.sh"
    assert repro_script.exists()
    repro_text = repro_script.read_text(encoding="utf-8")
    assert f"--target=gfx942" in repro_text
    assert "kernel.waveasm.mlir" in repro_text


def test_waveasm_ref_backend_detects_tool_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir(parents=True)
    tool_path = tool_dir / "waveasm-translate"
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)
    monkeypatch.delenv("BAYBRIDGE_WAVEASM_ROOT", raising=False)
    monkeypatch.setenv("PATH", f"{tool_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    backend = WaveAsmRefBackend()
    assert backend.available() is True
    assert backend._bridge.tool_path("waveasm-translate") == str(tool_path)


def test_waveasm_ref_backend_preserves_reduction_ops(tmp_path: Path) -> None:
    artifact = bb.compile(waveasm_reduce_kernel, cache_dir=tmp_path, backend="waveasm_ref")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '"baybridge.reduce_add"' in text
    assert "memref<2xf32>" in text


def test_emit_waveasm_repro_returns_bundle_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    tool_path = wave_root / "build" / "bin" / "waveasm-translate"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))

    bundle_dir = bb.emit_waveasm_repro(
        waveasm_scaffold_kernel,
        backend="waveasm_ref",
        cache_dir=tmp_path / "cache",
    )
    assert bundle_dir.exists()
    assert (bundle_dir / "kernel.mlir").exists()
    assert (bundle_dir / "repro.sh").exists()
    assert (bundle_dir / "manifest.json").exists()


def test_emit_waveasm_repro_script_prints_bundle_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    tool_path = wave_root / "build" / "bin" / "waveasm-translate"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))

    module_path = tmp_path / "sample_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src: bb.TensorSpec(shape=(4,), dtype='f32'), dst: bb.TensorSpec(shape=(4,), dtype='f32')):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx]\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "emit_waveasm_repro.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--backend",
            "waveasm_ref",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "BAYBRIDGE_WAVEASM_ROOT": str(wave_root),
        },
    )
    bundle_dir = Path(result.stdout.strip())
    assert bundle_dir.exists()
    assert (bundle_dir / "kernel.mlir").exists()


def test_emit_waveasm_repro_script_supports_sample_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    tool_path = wave_root / "build" / "bin" / "waveasm-translate"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))

    module_path = tmp_path / "sample_kernel_factory.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "@bb.kernel\n"
        "def sample_kernel(src, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx]\n"
        "def sample_args():\n"
        "    return (bb.tensor([1.0, 2.0, 3.0, 4.0], dtype='f32'), bb.zeros((4,), dtype='f32'))\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "emit_waveasm_repro.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backend",
            "waveasm_ref",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "BAYBRIDGE_WAVEASM_ROOT": str(wave_root),
        },
    )
    bundle_dir = Path(result.stdout.strip())
    assert bundle_dir.exists()
    assert (bundle_dir / "kernel.mlir").exists()


def test_compare_backends_script_reports_waveasm_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    tool_path = wave_root / "build" / "bin" / "waveasm-translate"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))

    module_path = tmp_path / "compare_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src: bb.TensorSpec(shape=(4,), dtype='f32'), dst: bb.TensorSpec(shape=(4,), dtype='f32')):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx]\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--backends",
            "waveasm_ref,gpu_mlir",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "BAYBRIDGE_WAVEASM_ROOT": str(wave_root),
        },
    )
    payload = json.loads(result.stdout)
    assert [entry["backend"] for entry in payload] == ["waveasm_ref", "gpu_mlir"]
    assert payload[0]["status"] == "ok"
    assert payload[0]["debug_bundle_dir"] is not None
    assert payload[1]["status"] == "ok"
    assert payload[1]["debug_bundle_dir"] is None


def test_compare_backends_script_can_include_environment_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wave_root = tmp_path / "wave"
    tool_path = wave_root / "build" / "bin" / "waveasm-translate"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text("#!/bin/sh\necho waveasm-translate 0.0\n", encoding="utf-8")
    tool_path.chmod(0o755)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))

    module_path = tmp_path / "compare_env_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src: bb.TensorSpec(shape=(4,), dtype='f32'), dst: bb.TensorSpec(shape=(4,), dtype='f32')):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx]\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--backends",
            "waveasm_ref",
            "--include-env",
            "--target",
            "gfx950",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PATH": f"{tool_path.parent}{os.pathsep}{os.environ.get('PATH', '')}",
            "BAYBRIDGE_WAVEASM_ROOT": str(wave_root),
        },
    )
    payload = json.loads(result.stdout)
    assert payload["environment"]["target"] == "gfx950"
    assert payload["environment"]["env"]["BAYBRIDGE_WAVEASM_ROOT"] == str(wave_root)
    assert payload["environment"]["tools"]["waveasm-translate"] == "waveasm-translate 0.0"
    assert "torch_device_available" in payload["environment"]["modules"]
    assert payload["results"][0]["backend"] == "waveasm_ref"
    assert payload["results"][0]["status"] == "ok"


def test_compare_backends_script_can_execute_flydsl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    module_path = tmp_path / "compare_exec_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "class FakeDLPackTensor:\n"
        "    def __init__(self, data):\n"
        "        self._data = list(data)\n"
        "        self.shape = (len(self._data),)\n"
        "        self.dtype = 'f32'\n"
        "    def __dlpack__(self, stream=None):\n"
        "        del stream\n"
        "        return object()\n"
        "    def __dlpack_device__(self):\n"
        "        return (10, 0)\n"
        "    def data_ptr(self):\n"
        "        return 0\n"
        "    def stride(self):\n"
        "        return (1,)\n"
        "    def __getitem__(self, index):\n"
        "        return self._data[index]\n"
        "    def __setitem__(self, index, value):\n"
        "        self._data[index] = value\n"
        "    def tolist(self):\n"
        "        return list(self._data)\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src, other, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx] + other[tidx]\n"
        "def sample_args():\n"
        "    return {\n"
        "        'args': (\n"
        "            FakeDLPackTensor([1.0, 2.0, 3.0, 4.0]),\n"
        "            FakeDLPackTensor([10.0, 20.0, 30.0, 40.0]),\n"
        "            FakeDLPackTensor([0.0, 0.0, 0.0, 0.0]),\n"
        "        ),\n"
        "        'result_indices': (2,),\n"
        "    }\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backends",
            "flydsl_exec",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--execute",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "BAYBRIDGE_FLYDSL_ROOT": str(fake_root),
        },
    )
    payload = json.loads(result.stdout)
    assert payload[0]["backend"] == "flydsl_exec"
    assert payload[0]["status"] == "ok"
    assert payload[0]["execute_status"] == "ok"
    assert len(payload[0]["timings_ms"]) == 1
    assert payload[0]["result_summaries"]["2"]["value"] == [11.0, 22.0, 33.0, 44.0]


def test_compare_backends_script_supports_split_compile_and_run_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    module_path = tmp_path / "compare_split_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "class FakeDLPackTensor:\n"
        "    def __init__(self, data):\n"
        "        self._data = list(data)\n"
        "        self.shape = (len(self._data),)\n"
        "        self.dtype = 'f32'\n"
        "    def __dlpack__(self, stream=None):\n"
        "        del stream\n"
        "        return object()\n"
        "    def __dlpack_device__(self):\n"
        "        return (10, 0)\n"
        "    def data_ptr(self):\n"
        "        return 0\n"
        "    def stride(self):\n"
        "        return (1,)\n"
        "    def __getitem__(self, index):\n"
        "        return self._data[index]\n"
        "    def __setitem__(self, index, value):\n"
        "        self._data[index] = value\n"
        "    def tolist(self):\n"
        "        return list(self._data)\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src, other, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx] + other[tidx]\n"
        "def sample_args():\n"
        "    return {\n"
        "        'compile_args': (\n"
        "            bb.tensor([1.0, 2.0, 3.0, 4.0], dtype='f32'),\n"
        "            bb.tensor([10.0, 20.0, 30.0, 40.0], dtype='f32'),\n"
        "            bb.zeros((4,), dtype='f32'),\n"
        "        ),\n"
        "        'run_args': (\n"
        "            FakeDLPackTensor([1.0, 2.0, 3.0, 4.0]),\n"
        "            FakeDLPackTensor([10.0, 20.0, 30.0, 40.0]),\n"
        "            FakeDLPackTensor([0.0, 0.0, 0.0, 0.0]),\n"
        "        ),\n"
        "        'result_indices': (2,),\n"
        "    }\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backends",
            "flydsl_exec",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--execute",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "BAYBRIDGE_FLYDSL_ROOT": str(fake_root),
        },
    )
    payload = json.loads(result.stdout)
    assert payload[0]["execute_status"] == "ok"
    assert payload[0]["result_summaries"]["2"]["value"] == [11.0, 22.0, 33.0, 44.0]


def test_compare_backends_script_passes_backend_and_target_to_sample_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    _install_fake_flydsl(fake_root, built=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    module_path = tmp_path / "compare_backend_aware_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "class FakeDLPackTensor:\n"
        "    def __init__(self, data):\n"
        "        self._data = list(data)\n"
        "        self.shape = (len(self._data),)\n"
        "        self.dtype = 'f32'\n"
        "    def __dlpack__(self, stream=None):\n"
        "        del stream\n"
        "        return object()\n"
        "    def __dlpack_device__(self):\n"
        "        return (10, 0)\n"
        "    def data_ptr(self):\n"
        "        return 0\n"
        "    def stride(self):\n"
        "        return (1,)\n"
        "    def __getitem__(self, index):\n"
        "        return self._data[index]\n"
        "    def __setitem__(self, index, value):\n"
        "        self._data[index] = value\n"
        "    def tolist(self):\n"
        "        return list(self._data)\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src, other, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx] + other[tidx]\n"
        "def sample_args(backend_name, target_arch):\n"
        "    if target_arch != 'gfx950':\n"
        "        raise RuntimeError(f'unexpected target: {target_arch}')\n"
        "    if backend_name == 'flydsl_exec':\n"
        "        return {\n"
        "            'compile_args': (\n"
        "                FakeDLPackTensor([1.0, 2.0, 3.0, 4.0]),\n"
        "                FakeDLPackTensor([10.0, 20.0, 30.0, 40.0]),\n"
        "                FakeDLPackTensor([0.0, 0.0, 0.0, 0.0]),\n"
        "            ),\n"
        "            'run_args': (\n"
        "                FakeDLPackTensor([1.0, 2.0, 3.0, 4.0]),\n"
        "                FakeDLPackTensor([10.0, 20.0, 30.0, 40.0]),\n"
        "                FakeDLPackTensor([0.0, 0.0, 0.0, 0.0]),\n"
        "            ),\n"
        "            'result_indices': (2,),\n"
        "        }\n"
        "    if backend_name == 'gpu_mlir':\n"
        "        return {\n"
        "            'args': (\n"
        "                bb.tensor([1.0, 2.0, 3.0, 4.0], dtype='f32'),\n"
        "                bb.tensor([10.0, 20.0, 30.0, 40.0], dtype='f32'),\n"
        "                bb.zeros((4,), dtype='f32'),\n"
        "            ),\n"
        "            'result_indices': (2,),\n"
        "        }\n"
        "    raise RuntimeError(f'unexpected backend: {backend_name}')\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backends",
            "flydsl_exec,gpu_mlir",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--execute",
            "--target",
            "gfx950",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "BAYBRIDGE_FLYDSL_ROOT": str(fake_root),
        },
    )
    payload = json.loads(result.stdout)
    assert payload[0]["backend"] == "flydsl_exec"
    assert payload[0]["execute_status"] == "ok"
    assert payload[1]["backend"] == "gpu_mlir"
    assert payload[1]["status"] == "ok"
    assert payload[1]["execute_status"] == "skipped_non_exec_backend"


def test_compare_backends_script_can_execute_flydsl_with_runtime_tensors(
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

    module_path = tmp_path / "compare_runtime_tensor_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src, other, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx] + other[tidx]\n"
        "def sample_args():\n"
        "    return {\n"
        "        'args': (\n"
        "            bb.tensor([1.0, 2.0, 3.0, 4.0], dtype='f32'),\n"
        "            bb.tensor([10.0, 20.0, 30.0, 40.0], dtype='f32'),\n"
        "            bb.zeros((4,), dtype='f32'),\n"
        "        ),\n"
        "        'result_indices': (2,),\n"
        "    }\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backends",
            "flydsl_exec",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--execute",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": f"{fake_torch_root}{os.pathsep}{Path(__file__).resolve().parents[1] / 'src'}",
            "BAYBRIDGE_FLYDSL_ROOT": str(fake_root),
        },
    )
    payload = json.loads(result.stdout)
    assert payload[0]["execute_status"] == "ok"
    assert payload[0]["result_summaries"]["2"]["value"] == [11.0, 22.0, 33.0, 44.0]
    sys.modules.pop("torch", None)


def test_compare_backends_script_skips_flydsl_runtime_tensor_execution_with_cpu_only_torch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl"
    fake_torch_root = tmp_path / "fake_torch_cpu"
    _install_fake_flydsl(fake_root, built=True)
    _install_fake_torch(fake_torch_root, device_available=False)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    module_path = tmp_path / "compare_runtime_tensor_cpu_only_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src, other, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx] + other[tidx]\n"
        "def sample_args():\n"
        "    return {\n"
        "        'args': (\n"
        "            bb.tensor([1.0, 2.0, 3.0, 4.0], dtype='f32'),\n"
        "            bb.tensor([10.0, 20.0, 30.0, 40.0], dtype='f32'),\n"
        "            bb.zeros((4,), dtype='f32'),\n"
        "        ),\n"
        "        'result_indices': (2,),\n"
        "    }\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backends",
            "flydsl_exec",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--execute",
            "--include-env",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": f"{fake_torch_root}{os.pathsep}{Path(__file__).resolve().parents[1] / 'src'}",
            "BAYBRIDGE_FLYDSL_ROOT": str(fake_root),
        },
    )
    payload = json.loads(result.stdout)
    assert payload["environment"]["modules"]["torch_device_available"] is False
    assert payload["results"][0]["execute_status"] == "skipped_incompatible_runtime_tensors"


def test_compare_backends_script_skips_unvalidated_real_flydsl_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_flydsl_realish"
    _install_fake_flydsl(fake_root, built=True, with_mlir=True)
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(fake_root))

    module_path = tmp_path / "compare_realish_flydsl_kernel.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "class FakeDLPackTensor:\n"
        "    def __init__(self, data):\n"
        "        self._data = list(data)\n"
        "        self.shape = (len(self._data),)\n"
        "        self.dtype = 'f32'\n"
        "    def __dlpack__(self, stream=None):\n"
        "        del stream\n"
        "        return object()\n"
        "    def __dlpack_device__(self):\n"
        "        return (10, 0)\n"
        "    def data_ptr(self):\n"
        "        return 0\n"
        "    def stride(self):\n"
        "        return (1,)\n"
        "    def __getitem__(self, index):\n"
        "        return self._data[index]\n"
        "    def __setitem__(self, index, value):\n"
        "        self._data[index] = value\n"
        "    def tolist(self):\n"
        "        return list(self._data)\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    smem = bb.make_tensor('smem', shape=(4,), dtype='f32', address_space=bb.AddressSpace.SHARED)\n"
        "    smem[tidx] = src[tidx]\n"
        "    bb.barrier()\n"
        "    dst[tidx] = smem[tidx]\n"
        "def sample_args():\n"
        "    return {\n"
        "        'args': (\n"
        "            FakeDLPackTensor([1.0, 2.0, 3.0, 4.0]),\n"
        "            FakeDLPackTensor([0.0, 0.0, 0.0, 0.0]),\n"
        "        ),\n"
        "        'result_indices': (1,),\n"
        "    }\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backends",
            "flydsl_exec",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--execute",
            "--include-env",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "BAYBRIDGE_FLYDSL_ROOT": str(fake_root),
        },
    )
    payload = json.loads(result.stdout)
    assert payload["results"][0]["execute_status"] == "skipped_unvalidated_real_flydsl_exec"
    assert "BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC=1" in payload["results"][0]["execute_note"]


def test_compare_backends_script_executes_with_real_flydsl_device_tensors_if_enabled(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_REAL_FLYDSL_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_REAL_FLYDSL_TESTS=1 to probe a real FlyDSL environment")

    backend = FlyDslExecBackend()
    environment = backend._bridge.exec_environment()
    if not environment.ready:
        pytest.skip("real FlyDSL environment is not importable")

    if not environment.torch_available:
        pytest.skip("real compare_backends execution requires device-backed torch tensors")

    target_arch = bb.AMDTarget().arch
    module_path = tmp_path / "compare_real_backend_aware.py"
    module_path.write_text(
        "import baybridge as bb\n"
        "import torch\n"
        "@bb.kernel(launch=bb.LaunchConfig(grid=(1,1,1), block=(4,1,1)))\n"
        "def sample_kernel(src, other, dst):\n"
        "    tidx, _, _ = bb.arch.thread_idx()\n"
        "    dst[tidx] = src[tidx] + other[tidx]\n"
        "def sample_args(backend_name, target_arch):\n"
        f"    if target_arch != '{target_arch}':\n"
        "        raise RuntimeError(f'unexpected target: {target_arch}')\n"
        "    if backend_name == 'flydsl_exec':\n"
        "        return {\n"
        "            'compile_args': (\n"
        "                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device='cuda'),\n"
        "                torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device='cuda'),\n"
        "                torch.zeros(4, dtype=torch.float32, device='cuda'),\n"
        "            ),\n"
        "            'run_args': (\n"
        "                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device='cuda'),\n"
        "                torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device='cuda'),\n"
        "                torch.zeros(4, dtype=torch.float32, device='cuda'),\n"
        "            ),\n"
        "            'result_indices': (2,),\n"
        "        }\n"
        "    if backend_name == 'gpu_mlir':\n"
        "        return {\n"
        "            'args': (\n"
        "                bb.tensor([1.0, 2.0, 3.0, 4.0], dtype='f32'),\n"
        "                bb.tensor([10.0, 20.0, 30.0, 40.0], dtype='f32'),\n"
        "                bb.zeros((4,), dtype='f32'),\n"
        "            ),\n"
        "            'result_indices': (2,),\n"
        "        }\n"
        "    raise RuntimeError(f'unexpected backend: {backend_name}')\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[1] / "tools" / "compare_backends.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(module_path),
            "sample_kernel",
            "--sample-factory",
            "sample_args",
            "--backends",
            "flydsl_exec,gpu_mlir",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--execute",
            "--include-env",
            "--target",
            target_arch,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    payload = json.loads(result.stdout)
    assert payload["environment"]["modules"]["torch_device_available"] is True
    assert payload["results"][0]["backend"] == "flydsl_exec"
    assert payload["results"][0]["execute_status"] == "ok"
    assert payload["results"][0]["result_summaries"]["2"]["value"] == [11.0, 22.0, 33.0, 44.0]
    assert payload["results"][1]["backend"] == "gpu_mlir"
    assert payload["results"][1]["execute_status"] == "skipped_non_exec_backend"
