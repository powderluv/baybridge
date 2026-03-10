import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel

def bf16_gemm_32x16x32(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel

def unsupported_gemm(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


def _make_supported_inputs() -> tuple[bb.Tensor, bb.Tensor, bb.Tensor]:
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


def test_hipkittens_exec_lowers_supported_bf16_gemm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_root = tmp_path / "HipKittens"
    (fake_root / "include").mkdir(parents=True)
    (fake_root / "include" / "kittens.cuh").write_text("// stub\n", encoding="utf-8")
    monkeypatch.setenv("BAYBRIDGE_HIPKITTENS_ROOT", str(fake_root))

    a, b, c = _make_supported_inputs()
    artifact = bb.compile(bf16_gemm_32x16x32, a, b, c, cache_dir=tmp_path, backend="hipkittens_exec")

    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_exec_cpp"
    text = artifact.lowered_module.text
    assert '#include "kittens.cuh"' in text
    assert 'mma_AB(c_accum, a_tile, b_tile, c_accum);' in text
    assert 'rt_bf<32, 16, ducks::rt_layout::row, ducks::rt_shape::rt_32x16> a_tile;' in text
    assert 'rt_bf<16, 32, ducks::rt_layout::col, ducks::rt_shape::rt_16x32> b_tile;' in text
    assert 'launch_bf16_gemm_32x16x32' in text



def test_hipkittens_exec_rejects_unsupported_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_root = tmp_path / "HipKittens"
    (fake_root / "include").mkdir(parents=True)
    (fake_root / "include" / "kittens.cuh").write_text("// stub\n", encoding="utf-8")
    monkeypatch.setenv("BAYBRIDGE_HIPKITTENS_ROOT", str(fake_root))

    a = bb.tensor([[1.0] * 16 for _ in range(16)], dtype="bf16")
    b = bb.tensor([[1.0] * 16 for _ in range(16)], dtype="bf16")
    c = bb.zeros((16, 16), dtype="f32")

    with pytest.raises(bb.BackendNotImplementedError, match="hipkittens_exec only supports bf16 GEMM shapes"):
        bb.compile(unsupported_gemm, a, b, c, cache_dir=tmp_path, backend="hipkittens_exec")



def test_hipkittens_exec_runs_supported_bf16_gemm(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HipKittens backend tests")
    if not os.environ.get("BAYBRIDGE_HIPKITTENS_ROOT"):
        pytest.skip("set BAYBRIDGE_HIPKITTENS_ROOT to a HipKittens checkout to run executable HipKittens backend tests")

    a, b, c = _make_supported_inputs()
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    artifact = bb.compile(
        bf16_gemm_32x16x32,
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
