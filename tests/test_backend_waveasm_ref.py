import os
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.waveasm_ref import WaveAsmRefBackend


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
