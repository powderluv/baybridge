import os
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.aster_exec import AsterExecBackend


@bb.kernel
def aster_exec_copy_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f32"),
    dst: bb.TensorSpec(shape=(16,), dtype="f32"),
):
    bb.copy(src, dst)


@bb.kernel
def aster_exec_copy_i32_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="i32"),
    dst: bb.TensorSpec(shape=(16,), dtype="i32"),
):
    bb.copy(src, dst)


@bb.kernel
def aster_exec_add_i32_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="i32"),
    other: bb.TensorSpec(shape=(16,), dtype="i32"),
    dst: bb.TensorSpec(shape=(16,), dtype="i32"),
):
    dst.store(src.load() + other.load())


@bb.kernel
def aster_exec_sub_i32_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="i32"),
    other: bb.TensorSpec(shape=(16,), dtype="i32"),
    dst: bb.TensorSpec(shape=(16,), dtype="i32"),
):
    dst.store(src.load() - other.load())


@bb.kernel
def aster_exec_mul_i32_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="i32"),
    other: bb.TensorSpec(shape=(16,), dtype="i32"),
    dst: bb.TensorSpec(shape=(16,), dtype="i32"),
):
    dst.store(src.load() * other.load())


@bb.kernel
def aster_exec_div_i32_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="i32"),
    other: bb.TensorSpec(shape=(16,), dtype="i32"),
    dst: bb.TensorSpec(shape=(16,), dtype="i32"),
):
    dst.store(src.load() / other.load())


@bb.kernel
def aster_exec_copy_f16_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f16"),
    dst: bb.TensorSpec(shape=(16,), dtype="f16"),
):
    bb.copy(src, dst)


@bb.kernel
def aster_exec_add_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f32"),
    other: bb.TensorSpec(shape=(16,), dtype="f32"),
    dst: bb.TensorSpec(shape=(16,), dtype="f32"),
):
    dst.store(src.load() + other.load())


@bb.kernel
def aster_exec_sub_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f32"),
    other: bb.TensorSpec(shape=(16,), dtype="f32"),
    dst: bb.TensorSpec(shape=(16,), dtype="f32"),
):
    dst.store(src.load() - other.load())


@bb.kernel
def aster_exec_mul_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f32"),
    other: bb.TensorSpec(shape=(16,), dtype="f32"),
    dst: bb.TensorSpec(shape=(16,), dtype="f32"),
):
    dst.store(src.load() * other.load())


@bb.kernel
def aster_exec_div_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f32"),
    other: bb.TensorSpec(shape=(16,), dtype="f32"),
    dst: bb.TensorSpec(shape=(16,), dtype="f32"),
):
    dst.store(src.load() / other.load())


@bb.kernel
def aster_exec_unsupported_pipeline_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f32"),
    other: bb.TensorSpec(shape=(16,), dtype="f32"),
    dst: bb.TensorSpec(shape=(16,), dtype="f32"),
):
    bb.copy(src, dst)
    dst.store(src.load() + other.load())


def _install_fake_aster(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    aster_root = tmp_path / "aster-root"
    package_root = tmp_path / "fake-site"
    package_dir = package_root / "aster"
    testing_dir = package_dir / "testing"
    build_bin = aster_root / ".aster-baybridge" / "bin"
    library_dir = aster_root / "mlir_kernels" / "library" / "common"

    build_bin.mkdir(parents=True, exist_ok=True)
    library_dir.mkdir(parents=True, exist_ok=True)
    testing_dir.mkdir(parents=True, exist_ok=True)

    for name in ("aster-opt", "aster-translate"):
        tool = build_bin / name
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)

    for name in ("register-init.mlir", "indexing.mlir", "copies.mlir"):
        (library_dir / name).write_text("// fake library\n", encoding="utf-8")

    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "ir.py").write_text(
        "class Context:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return False\n",
        encoding="utf-8",
    )
    (package_dir / "pass_pipelines.py").write_text(
        "def get_pass_pipeline(name):\n"
        "    return name\n",
        encoding="utf-8",
    )
    (package_dir / "utils.py").write_text(
        "from pathlib import Path\n"
        "def assemble_to_hsaco(asm, target='gfx942', wavefront_size=64, output_path=None):\n"
        "    path = Path(output_path)\n"
        "    path.write_bytes(b'fake-hsaco')\n"
        "    return str(path)\n",
        encoding="utf-8",
    )
    (package_dir / "hip.py").write_text(
        "def execute_hsaco(hsaco_path, kernel_name, input_arrays, output_arrays, grid_dim=(1,1,1), block_dim=(1,1,1), num_iterations=1):\n"
        "    if len(input_arrays) == 2:\n"
        "        is_array_like = all(hasattr(value, 'shape') or hasattr(value, 'tolist') for value in input_arrays)\n"
        "        if 'sub' in kernel_name:\n"
        "            if is_array_like:\n"
        "                output_arrays[0][:] = input_arrays[0] - input_arrays[1]\n"
        "            else:\n"
        "                output_arrays[0][:] = [left - right for left, right in zip(input_arrays[0], input_arrays[1])]\n"
        "        elif 'mul' in kernel_name:\n"
        "            if is_array_like:\n"
        "                output_arrays[0][:] = input_arrays[0] * input_arrays[1]\n"
        "            else:\n"
        "                output_arrays[0][:] = [left * right for left, right in zip(input_arrays[0], input_arrays[1])]\n"
        "        elif 'div' in kernel_name:\n"
        "            if is_array_like:\n"
        "                output_arrays[0][:] = input_arrays[0] / input_arrays[1]\n"
        "            else:\n"
        "                output_arrays[0][:] = [left / right for left, right in zip(input_arrays[0], input_arrays[1])]\n"
        "        elif is_array_like:\n"
        "            output_arrays[0][:] = input_arrays[0] + input_arrays[1]\n"
        "        else:\n"
        "            output_arrays[0][:] = [left + right for left, right in zip(input_arrays[0], input_arrays[1])]\n"
        "    else:\n"
        "        output_arrays[0][:] = input_arrays[0]\n"
        "    return [1234]\n",
        encoding="utf-8",
    )
    (testing_dir / "__init__.py").write_text(
        "from contextlib import contextmanager\n"
        "def compile_mlir_file_to_asm(mlir_file, kernel_name, pass_pipeline, ctx, preprocess=None, library_paths=None, **kwargs):\n"
        "    text = open(mlir_file, 'r', encoding='utf-8').read()\n"
        "    if preprocess is not None:\n"
        "        text = preprocess(text)\n"
        "    return f'// fake asm for {kernel_name}\\n' + text, object()\n"
        "@contextmanager\n"
        "def hsaco_file(path):\n"
        "    yield path\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(package_root))
    monkeypatch.setenv("BAYBRIDGE_ASTER_ROOT", str(aster_root))
    monkeypatch.setattr("baybridge.backends.aster_exec.load_hip_library", lambda global_scope=True: object())
    return aster_root


def test_aster_exec_lowers_copy_kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    aster_root = _install_fake_aster(monkeypatch, tmp_path)

    artifact = bb.compile(aster_exec_copy_kernel, cache_dir=tmp_path / "cache", backend="aster_exec")

    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "aster_exec_mlir"
    text = artifact.lowered_module.text
    assert "amdgcn.module @mod target = #amdgcn.target<" in text
    assert "#amdgcn.isa<" in text
    assert f"amdgcn.kernel @{aster_exec_copy_kernel.__name__}" in text
    assert "func.func private @copy_loop" in text
    assert str(aster_root) in os.environ["BAYBRIDGE_ASTER_ROOT"]
    assert artifact.debug_bundle_dir is not None
    assert (artifact.debug_bundle_dir / "kernel.mlir").exists()
    assert (artifact.debug_bundle_dir / "manifest.json").exists()
    assert (artifact.debug_bundle_dir / "repro.sh").exists()


def test_aster_exec_launches_copy_kernel_with_fake_aster(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, dst)

    assert dst.tolist() == src.tolist()
    assert artifact.lowered_path is not None
    assert artifact.lowered_path.with_suffix(".hsaco").exists()


def test_compile_auto_prefers_aster_exec_for_matching_copy_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_compile_does_not_auto_prefer_aster_exec_for_unsupported_ir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_unsupported_pipeline_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name != "aster_exec"


def test_compile_does_not_auto_prefer_aster_exec_for_unsupported_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_unsupported_pipeline_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name != "aster_exec"


def test_aster_exec_rejects_unsupported_ir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    with pytest.raises(bb.BackendNotImplementedError, match="single supported tensor binary pipeline"):
        bb.compile(
            aster_exec_unsupported_pipeline_kernel,
            src,
            other,
            dst,
            cache_dir=tmp_path / "cache",
            backend="aster_exec",
        )


def test_compile_auto_prefers_aster_exec_for_matching_add_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_add_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left + right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_sub_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_sub_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left - right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_mul_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.5 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_mul_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left * right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_div_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([float(index + 2) for index in range(16)], dtype="f32")
    other = bb.tensor([2.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_div_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left / right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_i32_copy_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor(list(range(16)), dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_copy_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_compile_auto_prefers_aster_exec_for_matching_i32_add_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_add_i32_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left + right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_i32_sub_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_sub_i32_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left - right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_i32_mul_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([3 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_mul_i32_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left * right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_i32_div_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([2 * (index + 1) for index in range(16)], dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_div_i32_kernel,
        src,
        other,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left // right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_matching_f16_copy_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_aster(monkeypatch, tmp_path)
    src = bb.tensor([index / 8.0 for index in range(16)], dtype="f16")
    dst = bb.zeros((16,), dtype="f16")

    artifact = bb.compile(
        aster_exec_copy_f16_kernel,
        src,
        dst,
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_aster_exec_runs_copy_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_copy_kernel,
        src,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, dst)

    assert dst.tolist() == src.tolist()


def test_aster_exec_runs_add_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_add_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left + right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_sub_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_sub_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left - right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_mul_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.5 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_mul_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left * right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_div_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index + 2) for index in range(16)], dtype="f32")
    other = bb.tensor([2.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_div_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left / right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_i32_copy_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_copy_i32_kernel,
        src,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_aster_exec_runs_i32_add_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_add_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left + right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_i32_sub_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_sub_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left - right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_i32_mul_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([3 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_mul_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left * right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_i32_div_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([2 * (index + 1) for index in range(16)], dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_div_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, other, dst)
    assert dst.tolist() == [left // right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_aster_exec_runs_f16_copy_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([index / 8.0 for index in range(16)], dtype="f16")
    dst = bb.zeros((16,), dtype="f16")

    artifact = bb.compile(
        aster_exec_copy_f16_kernel,
        src,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
        backend="aster_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_compile_auto_prefers_aster_exec_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_copy_kernel,
        src,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_compile_auto_prefers_aster_exec_for_add_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_add_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left + right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_sub_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_sub_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left - right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_mul_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index) for index in range(16)], dtype="f32")
    other = bb.tensor([1.5 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_mul_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left * right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_div_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([float(index + 2) for index in range(16)], dtype="f32")
    other = bb.tensor([2.0 for _ in range(16)], dtype="f32")
    dst = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        aster_exec_div_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left / right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_f16_copy_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([index / 8.0 for index in range(16)], dtype="f16")
    dst = bb.zeros((16,), dtype="f16")

    artifact = bb.compile(
        aster_exec_copy_f16_kernel,
        src,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_compile_auto_prefers_aster_exec_for_i32_copy_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_copy_i32_kernel,
        src,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_compile_auto_prefers_aster_exec_for_i32_add_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_add_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left + right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_i32_sub_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_sub_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left - right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_i32_mul_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor(list(range(16)), dtype="i32")
    other = bb.tensor([3 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_mul_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left * right for left, right in zip(src.tolist(), other.tolist(), strict=True)]


def test_compile_auto_prefers_aster_exec_for_i32_div_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to exercise ASTER exec on hardware")
    if "BAYBRIDGE_ASTER_ROOT" not in os.environ:
        pytest.skip("set BAYBRIDGE_ASTER_ROOT to an ASTER source checkout")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx942")
    backend = AsterExecBackend()
    if not backend.available(bb.AMDTarget(arch=target_arch)):
        pytest.skip(f"aster_exec is not ready for target {target_arch}")

    src = bb.tensor([2 * (index + 1) for index in range(16)], dtype="i32")
    other = bb.tensor([2 for _ in range(16)], dtype="i32")
    dst = bb.zeros((16,), dtype="i32")

    artifact = bb.compile(
        aster_exec_div_i32_kernel,
        src,
        other,
        dst,
        target=bb.AMDTarget(arch=target_arch),
        cache_dir=tmp_path / "cache",
    )

    assert artifact.backend_name == "aster_exec"
    artifact(src, other, dst)
    assert dst.tolist() == [left // right for left, right in zip(src.tolist(), other.tolist(), strict=True)]
