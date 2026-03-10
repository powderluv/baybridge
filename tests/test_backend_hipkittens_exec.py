import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel

def bf16_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel

def unsupported_gemm(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


def _fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_root = tmp_path / "HipKittens"
    (fake_root / "include").mkdir(parents=True)
    (fake_root / "include" / "kittens.cuh").write_text("// stub\n", encoding="utf-8")
    monkeypatch.setenv("BAYBRIDGE_HIPKITTENS_ROOT", str(fake_root))


def _make_micro_inputs() -> tuple[bb.Tensor, bb.Tensor, bb.Tensor]:
    a = bb.tensor([[row * 16 + col + 1 for col in range(16)] for row in range(32)], dtype="bf16")
    b_data = []
    for row in range(16):
        values = []
        for col in range(32):
            values.append(1.0 if col == row else 0.0)
        b_data.append(values)
    b = bb.tensor(b_data, dtype="bf16")
    c = bb.zeros((32, 32), dtype="f32")
    return a, b, c


def _make_tiled_inputs() -> tuple[bb.Tensor, bb.Tensor, bb.Tensor]:
    a = bb.tensor([[row * 32 + col + 1 for col in range(32)] for row in range(64)], dtype="bf16")
    b_data = []
    for row in range(32):
        values = [0.0] * 64
        if row < 16:
            values[row] = 1.0
            values[row + 16] = 1.0
        else:
            values[(row - 16) + 32] = 1.0
            values[(row - 16) + 48] = 1.0
        b_data.append(values)
    b = bb.tensor(b_data, dtype="bf16")
    c = bb.zeros((64, 64), dtype="f32")
    return a, b, c


def test_hipkittens_exec_lowers_supported_bf16_gemm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    a, b, c = _make_micro_inputs()
    artifact = bb.compile(
        bf16_gemm_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch="gfx950"),
    )

    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_exec_cpp"
    text = artifact.lowered_module.text
    assert '#include "kittens.cuh"' in text
    assert 'mma_AB(c_accum, a_tile, b_tile, c_accum);' in text
    assert 'rt_bf<32, 16, ducks::rt_layout::row, ducks::rt_shape::rt_32x16> a_tile;' in text
    assert 'rt_bf<16, 32, ducks::rt_layout::col, ducks::rt_shape::rt_16x32> b_tile;' in text
    assert 'for (int k_tile = 0; k_tile < 1; ++k_tile)' in text
    assert '<<<dim3(1, 1, 1), dim3(kittens::WARP_THREADS, 1, 1)>>>' in text



def test_hipkittens_exec_lowers_tiled_bf16_gemm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    a, b, c = _make_tiled_inputs()
    artifact = bb.compile(
        bf16_gemm_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch="gfx950"),
    )

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '// full shapes: A=(64, 32), B=(32, 64), C=(64, 64)' in text
    assert '// tile grid: (2, 2, 1)' in text
    assert '// k_tiles: 2' in text
    assert 'for (int k_tile = 0; k_tile < 2; ++k_tile)' in text
    assert 'load(a_tile, a, {0, 0, tile_row, k_tile});' in text
    assert 'load(b_tile, b, {0, 0, k_tile, tile_col});' in text
    assert '<<<dim3(2, 2, 1), dim3(kittens::WARP_THREADS, 1, 1)>>>' in text



def test_hipkittens_exec_rejects_unsupported_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    a = bb.tensor([[1.0] * 16 for _ in range(16)], dtype="bf16")
    b = bb.tensor([[1.0] * 16 for _ in range(16)], dtype="bf16")
    c = bb.zeros((16, 16), dtype="f32")

    with pytest.raises(bb.BackendNotImplementedError, match="hipkittens_exec only supports bf16 GEMM shapes"):
        bb.compile(
            unsupported_gemm,
            a,
            b,
            c,
            cache_dir=tmp_path,
            backend="hipkittens_exec",
            target=bb.AMDTarget(arch="gfx950"),
        )


def test_compile_auto_prefers_hipkittens_exec_for_matching_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    a, b, c = _make_micro_inputs()

    artifact = bb.compile(bf16_gemm_kernel, a, b, c, cache_dir=tmp_path, target=bb.AMDTarget(arch="gfx950"))

    assert artifact.backend_name == "hipkittens_exec"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_exec_cpp"


def test_compile_keeps_default_backend_when_hipkittens_exec_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_root(tmp_path, monkeypatch)
    a, b, c = _make_micro_inputs()

    artifact = bb.compile(bf16_gemm_kernel, a, b, c, cache_dir=tmp_path, target=bb.AMDTarget(arch="gfx942"))

    assert artifact.backend_name == "mlir_text"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "baybridge"


def test_compile_uses_target_env_for_auto_preference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    monkeypatch.setenv("BAYBRIDGE_EXEC_ARCH", "gfx950")
    a, b, c = _make_micro_inputs()

    artifact = bb.compile(bf16_gemm_kernel, a, b, c, cache_dir=tmp_path)

    assert artifact.target.arch == "gfx950"
    assert artifact.backend_name == "hipkittens_exec"



def test_hipkittens_exec_runs_supported_bf16_gemm(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HipKittens backend tests")
    if not os.environ.get("BAYBRIDGE_HIPKITTENS_ROOT"):
        pytest.skip("set BAYBRIDGE_HIPKITTENS_ROOT to a HipKittens checkout to run executable HipKittens backend tests")

    a, b, c = _make_micro_inputs()
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    artifact = bb.compile(
        bf16_gemm_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, b, c)

    expected = [[0.0 for _ in range(32)] for _ in range(32)]
    rounded_a = a.tolist()
    for row in range(32):
        for col in range(16):
            expected[row][col] = rounded_a[row][col]
    assert c.tolist() == expected



def test_hipkittens_exec_runs_tiled_bf16_gemm(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HipKittens backend tests")
    if not os.environ.get("BAYBRIDGE_HIPKITTENS_ROOT"):
        pytest.skip("set BAYBRIDGE_HIPKITTENS_ROOT to a HipKittens checkout to run executable HipKittens backend tests")

    a, b, c = _make_tiled_inputs()
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    artifact = bb.compile(
        bf16_gemm_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, b, c)

    rounded_a = a.tolist()
    expected = [[0.0 for _ in range(64)] for _ in range(64)]
    for row in range(64):
        expected[row][0:16] = rounded_a[row][0:16]
        expected[row][16:32] = rounded_a[row][0:16]
        expected[row][32:48] = rounded_a[row][16:32]
        expected[row][48:64] = rounded_a[row][16:32]
    assert c.tolist() == expected
