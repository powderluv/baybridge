from __future__ import annotations

import sys
from glob import glob as stdlib_glob
from pathlib import Path

from baybridge.backends.aster_bridge import AsterEnvironment
from baybridge.backends.aster_exec import AsterExecBackend
from baybridge.hip_runtime import _hip_library_candidates


def test_hip_library_candidates_prefer_active_python_environment(monkeypatch, tmp_path: Path) -> None:
    prefix = tmp_path / "prefix"
    rocm_sdk_lib = prefix / "lib" / "python3.10" / "site-packages" / "_rocm_sdk_devel" / "lib"
    torch_lib = prefix / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
    triton_lib = prefix / "lib" / "python3.10" / "site-packages" / "triton" / "backends" / "amd" / "lib"
    for path in (rocm_sdk_lib, torch_lib, triton_lib):
        path.mkdir(parents=True, exist_ok=True)
    (rocm_sdk_lib / "libamdhip64.so.7").write_text("", encoding="utf-8")
    (torch_lib / "libamdhip64.so").write_text("", encoding="utf-8")
    (triton_lib / "libamdhip64.so").write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(
        "baybridge.hip_runtime.glob",
        lambda pattern: [] if pattern == "/opt/rocm-*" else stdlib_glob(pattern),
    )

    candidates = _hip_library_candidates()

    assert candidates[0] == str(rocm_sdk_lib / "libamdhip64.so.7")
    assert candidates[1] == str(torch_lib / "libamdhip64.so")
    assert candidates[2] == str(triton_lib / "libamdhip64.so")


def test_aster_exec_preloads_mlir_support_libraries(monkeypatch, tmp_path: Path) -> None:
    package_root = tmp_path / "python_packages" / "aster"
    mlir_lib_root = package_root / "_mlir_libs"
    mlir_lib_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "libMLIRPythonSupport-mlir.so",
        "libnanobind-mlir.so",
        "libASTER.so.23.0git",
        "libASTER.so",
    ):
        (mlir_lib_root / name).write_text("", encoding="utf-8")

    environment = AsterEnvironment(
        configured_root=None,
        aster_opt="/tmp/aster-opt",
        aster_translate="/tmp/aster-translate",
        python_package_root=str(package_root),
        runtime_module_available=True,
        ready=True,
        notes=(),
    )
    backend = AsterExecBackend()
    monkeypatch.setattr(backend._bridge, "environment", lambda: environment)
    monkeypatch.setattr("baybridge.backends.aster_exec.load_hip_library", lambda global_scope=True: object())

    loaded_paths: list[str] = []

    def fake_cdll(path: str, mode: int = 0):
        loaded_paths.append(path)
        return object()

    def fake_import_module(name: str):
        return {"module": name}

    monkeypatch.setattr("baybridge.backends.aster_exec.ctypes.CDLL", fake_cdll)
    monkeypatch.setattr("baybridge.backends.aster_exec.importlib.import_module", fake_import_module)

    modules = backend._load_aster_modules()

    assert str(package_root.parent) in sys.path
    assert loaded_paths == [
        str(mlir_lib_root / "libMLIRPythonSupport-mlir.so"),
        str(mlir_lib_root / "libnanobind-mlir.so"),
        str(mlir_lib_root / "libASTER.so.23.0git"),
        str(mlir_lib_root / "libASTER.so"),
    ]
    assert modules["hip"] == {"module": "aster.hip"}
