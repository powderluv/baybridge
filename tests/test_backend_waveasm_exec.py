import math
import os
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.waveasm_exec import WaveAsmExecBackend

_WAVEASM_EXPERIMENTAL_ENV = "BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC"


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def waveasm_exec_inplace_scale_kernel(
    buf: bb.TensorSpec(shape=(4,), dtype="f32"),
    scale: bb.ScalarSpec(dtype="f32"),
):
    tidx, _, _ = bb.arch.thread_idx()
    buf[tidx] = buf[tidx] * scale


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1), shared_mem_bytes=16))
def waveasm_exec_shared_inplace_scale_kernel(
    buf: bb.TensorSpec(shape=(4,), dtype="f32"),
    scale: bb.ScalarSpec(dtype="f32"),
):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor("smem", shape=(4,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    smem[tidx] = buf[tidx]
    bb.barrier()
    buf[tidx] = smem[tidx] * scale


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def waveasm_exec_inplace_sqrt_scale_kernel(
    buf: bb.TensorSpec(shape=(4,), dtype="f32"),
    scale: bb.ScalarSpec(dtype="f32"),
):
    tidx, _, _ = bb.arch.thread_idx()
    buf[tidx] = bb.math.sqrt(buf[tidx]) / scale


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def waveasm_exec_multi_buffer_add_kernel(
    src: bb.TensorSpec(shape=(4,), dtype="f32"),
    other: bb.TensorSpec(shape=(4,), dtype="f32"),
    dst: bb.TensorSpec(shape=(4,), dtype="f32"),
):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + other[tidx]


@bb.kernel
def waveasm_exec_unsupported_reduce_kernel(
    src: bb.TensorSpec(shape=(2, 3), dtype="f32"),
    dst: bb.TensorSpec(shape=(2,), dtype="f32"),
):
    dst.store(src.load().reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


class _FakeModuleRuntime:
    def __init__(self) -> None:
        self.loaded_modules: list[Path] = []
        self.queried_functions: list[str] = []

    def load_module(self, hsaco_path: Path):
        self.loaded_modules.append(Path(hsaco_path))
        return "module"

    def get_function(self, module, kernel_name: str):
        del module
        self.queried_functions.append(kernel_name)
        return "function"


class _FakeStream:
    def __init__(self, value: int) -> None:
        self.value = value


def _install_fake_toolchain(root: Path) -> Path:
    tool_dir = root / "build" / "bin"
    tool_dir.mkdir(parents=True, exist_ok=True)
    translate = tool_dir / "waveasm-translate"
    translate.write_text("#!/bin/sh\nprintf '.text\\n'\n", encoding="utf-8")
    translate.chmod(0o755)
    clang_script = (
        "#!/bin/sh\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '-o' ]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "if [ -z \"$out\" ]; then exit 1; fi\n"
        "mkdir -p \"$(dirname \"$out\")\"\n"
        ": > \"$out\"\n"
        "exit 0\n"
    )
    for name in ("clang", "clang++", "ld.lld"):
        tool = tool_dir / name
        tool.write_text(clang_script, encoding="utf-8")
        tool.chmod(0o755)
    return tool_dir


def test_waveasm_exec_available_with_fake_toolchain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    _install_fake_toolchain(wave_root)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))
    monkeypatch.setenv(_WAVEASM_EXPERIMENTAL_ENV, "1")

    backend = WaveAsmExecBackend()
    assert backend.available() is True


def test_waveasm_exec_builds_hsaco_and_launches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    _install_fake_toolchain(wave_root)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))
    monkeypatch.setenv(_WAVEASM_EXPERIMENTAL_ENV, "1")

    backend = WaveAsmExecBackend()
    fake_runtime = _FakeModuleRuntime()
    launch_record: dict[str, object] = {}

    def fake_make_module_runtime():
        return fake_runtime

    def fake_launch(runtime, function, ir, args, stream):
        launch_record["runtime"] = runtime
        launch_record["function"] = function
        launch_record["kernel"] = ir.name
        launch_record["stream"] = stream
        buf, scale = args
        for index, value in enumerate(buf.tolist()):
            buf[index] = value * scale

    monkeypatch.setattr(backend, "_make_module_runtime", fake_make_module_runtime)
    monkeypatch.setattr(backend, "_launch", fake_launch)

    buf = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    scale = 2.0

    artifact = bb.compile(
        waveasm_exec_inplace_scale_kernel,
        buf,
        scale,
        cache_dir=tmp_path / "cache",
        backend=backend,
    )

    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "waveasm_exec_mlir"
    stream = _FakeStream(7)
    artifact(buf, scale, stream=stream)

    assert buf.tolist() == [2.0, 4.0, 6.0, 8.0]
    assert artifact.lowered_path is not None
    assert artifact.lowered_path.with_suffix(".s").exists()
    assert artifact.lowered_path.with_suffix(".o").exists()
    assert artifact.lowered_path.with_suffix(".hsaco").exists()
    assert artifact.debug_bundle_dir is not None
    repro_dir = artifact.debug_bundle_dir
    assert (repro_dir / "kernel.mlir").exists()
    assert (repro_dir / "manifest.json").exists()
    assert (repro_dir / "repro.sh").exists()
    assert fake_runtime.loaded_modules == [artifact.lowered_path.with_suffix(".hsaco")]
    assert fake_runtime.queried_functions == ["waveasm_exec_inplace_scale_kernel"]
    assert launch_record["runtime"] is fake_runtime
    assert launch_record["function"] == "function"
    assert launch_record["kernel"] == "waveasm_exec_inplace_scale_kernel"
    assert launch_record["stream"] is stream


def test_waveasm_exec_supports_shared_stage_kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    _install_fake_toolchain(wave_root)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))
    monkeypatch.setenv(_WAVEASM_EXPERIMENTAL_ENV, "1")

    backend = WaveAsmExecBackend()
    artifact = bb.compile(
        waveasm_exec_shared_inplace_scale_kernel,
        cache_dir=tmp_path / "cache",
        backend=backend,
    )
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "%smem = memref.alloca()" in text
    assert "gpu.barrier" in text


def test_waveasm_exec_supports_single_buffer_math_kernel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wave_root = tmp_path / "wave"
    _install_fake_toolchain(wave_root)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))
    monkeypatch.setenv(_WAVEASM_EXPERIMENTAL_ENV, "1")

    backend = WaveAsmExecBackend()
    fake_runtime = _FakeModuleRuntime()

    def fake_make_module_runtime():
        return fake_runtime

    def fake_launch(runtime, function, ir, args, stream):
        del runtime, function, ir, stream
        buf, scale = args
        for index, value in enumerate(buf.tolist()):
            buf[index] = math.sqrt(value) / scale

    monkeypatch.setattr(backend, "_make_module_runtime", fake_make_module_runtime)
    monkeypatch.setattr(backend, "_launch", fake_launch)

    buf = bb.tensor([1.0, 4.0, 9.0, 16.0], dtype="f32")
    artifact = bb.compile(
        waveasm_exec_inplace_sqrt_scale_kernel,
        buf,
        2.0,
        cache_dir=tmp_path / "cache",
        backend=backend,
    )

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "math.sqrt" in text
    assert "arith.divf" in text

    artifact(buf, 2.0)

    assert buf.tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0], rel=1e-6, abs=1e-6)


def test_waveasm_exec_rejects_multi_buffer_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    other = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    with pytest.raises(
        bb.BackendNotImplementedError,
        match="disabled by default|single-global-tensor",
    ):
        bb.compile(
            waveasm_exec_multi_buffer_add_kernel,
            src,
            other,
            dst,
            cache_dir=tmp_path,
            backend="waveasm_exec",
        )


def test_waveasm_exec_rejects_reduction_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst = bb.zeros((2,), dtype="f32")

    with pytest.raises(
        bb.BackendNotImplementedError,
        match="disabled by default|single-global-tensor",
    ):
        bb.compile(
            waveasm_exec_unsupported_reduce_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            backend="waveasm_exec",
        )


def test_waveasm_exec_rejects_when_not_opted_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    _install_fake_toolchain(wave_root)
    monkeypatch.setenv("BAYBRIDGE_WAVEASM_ROOT", str(wave_root))
    monkeypatch.delenv(_WAVEASM_EXPERIMENTAL_ENV, raising=False)

    buf = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")

    with pytest.raises(bb.BackendNotImplementedError, match="disabled by default"):
        bb.compile(
            waveasm_exec_inplace_scale_kernel,
            buf,
            2.0,
            cache_dir=tmp_path,
            backend="waveasm_exec",
        )


@pytest.mark.xfail(reason="Known upstream WaveASM execution correctness issue: iree-org/wave#1117", strict=False)
def test_waveasm_exec_runs_single_buffer_kernel_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable WaveASM backend tests")
    if not os.environ.get("BAYBRIDGE_WAVEASM_ROOT"):
        pytest.skip("set BAYBRIDGE_WAVEASM_ROOT to a Wave checkout to run executable WaveASM backend tests")
    if os.environ.get(_WAVEASM_EXPERIMENTAL_ENV) != "1":
        pytest.skip(f"set {_WAVEASM_EXPERIMENTAL_ENV}=1 to run the experimental WaveASM backend test")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    target = bb.AMDTarget(arch=target_arch)
    backend = WaveAsmExecBackend()
    if not backend.available(target):
        pytest.skip(f"waveasm_exec is not toolchain-ready for target {target_arch}")

    buf = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    artifact = bb.compile(
        waveasm_exec_inplace_scale_kernel,
        buf,
        2.0,
        cache_dir=tmp_path,
        backend="waveasm_exec",
        target=target,
    )

    artifact(buf, 2.0)

    assert buf.tolist() == [2.0, 4.0, 6.0, 8.0]
