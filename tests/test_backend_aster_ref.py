import os
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.aster_ref import AsterRefBackend


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1), cluster=(2, 1, 1), shared_mem_bytes=128))
def aster_elementwise_kernel(
    src: bb.TensorSpec(shape=(4,), dtype="f32"),
    dst: bb.TensorSpec(shape=(4,), dtype="f32"),
):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + 1.0


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def aster_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


def test_aster_ref_backend_emits_tool_hints_and_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    aster_root = tmp_path / "aster"
    opt_path = aster_root / "build" / "bin" / "aster-opt"
    translate_path = aster_root / "build" / "bin" / "aster-translate"
    opt_path.parent.mkdir(parents=True, exist_ok=True)
    opt_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    translate_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    opt_path.chmod(0o755)
    translate_path.chmod(0o755)
    (aster_root / "python" / "aster").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BAYBRIDGE_ASTER_ROOT", str(aster_root))

    artifact = bb.compile(aster_elementwise_kernel, cache_dir=tmp_path / "cache", backend="aster_ref")

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert artifact.lowered_module.dialect == "aster_mlir"
    assert "// baybridge.aster_ref" in text
    assert f"// configured_root: {aster_root.resolve()}" in text
    assert f"// aster_opt: {opt_path}" in text
    assert f"// aster_translate: {translate_path}" in text
    assert '"family": "elementwise"' in text
    assert "Baybridge currently emits GPU MLIR reference input, not ASTER AMDGCN IR." in text
    assert f'module attributes {{aster.target = "{artifact.target.arch}", aster.wave_size = 64,' in text
    assert artifact.debug_bundle_dir is not None
    repro_dir = artifact.debug_bundle_dir
    assert (repro_dir / "kernel.mlir").exists()
    assert (repro_dir / "manifest.json").exists()
    repro_script = repro_dir / "repro.sh"
    assert repro_script.exists()
    repro_text = repro_script.read_text(encoding="utf-8")
    assert "ASTER_OPT" in repro_text
    assert "ASTER_TRANSLATE" in repro_text
    assert "Baybridge currently emits GPU MLIR reference input" in repro_text


def test_aster_ref_backend_detects_tools_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir(parents=True)
    opt_path = tool_dir / "aster-opt"
    translate_path = tool_dir / "aster-translate"
    opt_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    translate_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    opt_path.chmod(0o755)
    translate_path.chmod(0o755)
    monkeypatch.delenv("BAYBRIDGE_ASTER_ROOT", raising=False)
    monkeypatch.setenv("PATH", f"{tool_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    backend = AsterRefBackend()
    assert backend.available() is True
    assert backend._bridge.tool_path("aster-opt") == str(opt_path)
    assert backend._bridge.tool_path("aster-translate") == str(translate_path)


def test_aster_ref_backend_classifies_gemm_family(tmp_path: Path) -> None:
    a = bb.tensor([[1.0] * 16 for _ in range(16)], dtype="f16")
    b = bb.tensor([[1.0] * 16 for _ in range(16)], dtype="f16")
    c = bb.zeros((16, 16), dtype="f32")

    artifact = bb.compile(aster_gemm_kernel, a, b, c, cache_dir=tmp_path, backend="aster_ref")

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '"family": "mfma_gemm"' in text
    assert "mlir_kernels/gemm_sched_1wave_dword4_mxnxk_16x16x16_f16f16f32.mlir" in text
