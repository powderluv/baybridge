import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def static_range_kernel(out: bb.Tensor):
    acc = bb.Int32(0)
    for index in bb.range(4, prefetch_stages=2, unroll_full=True):
        acc = acc + bb.Int32(index)
    out[0] = acc


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def constexpr_range_kernel(out: bb.Tensor):
    acc = bb.Int32(0)
    for index in bb.range_constexpr(bb.size(out)):
        acc = acc + bb.Int32(index)
    out[0] = acc


def test_static_range_runtime_and_compile(tmp_path: Path) -> None:
    out = bb.zeros((4,), dtype="i32")
    static_range_kernel(out).launch(grid=(1, 1, 1), block=(1, 1, 1))
    assert out[0] == 6

    artifact = bb.compile(static_range_kernel, out, cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None
    loop_hints = artifact.ir.metadata.get("loop_hints")
    assert isinstance(loop_hints, list)
    assert loop_hints[0]["kind"] == "range"
    assert loop_hints[0]["prefetch_stages"] == 2
    assert loop_hints[0]["unroll_full"] is True


def test_constexpr_range_runtime_and_compile(tmp_path: Path) -> None:
    out = bb.zeros((4,), dtype="i32")
    constexpr_range_kernel(out).launch(grid=(1, 1, 1), block=(1, 1, 1))
    assert out[0] == 6

    artifact = bb.compile(constexpr_range_kernel, out, cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None
    loop_hints = artifact.ir.metadata.get("loop_hints")
    assert isinstance(loop_hints, list)
    assert loop_hints[0]["kind"] == "range_constexpr"
    assert loop_hints[0]["unroll_full"] is True


def test_generic_atom_and_tiled_copy_aliases() -> None:
    copy_atom = bb.make_atom(bb.nvgpu.CopyUniversalOp(), "f32", num_bits_per_copy=64)
    tiled_copy = bb.make_tiled_copy(
        copy_atom,
        bb.make_layout((1, 4), stride=(4, 1)),
        bb.make_layout((2, 1), stride=(1, 1)),
    )
    cotiled_copy = bb.make_cotiled_copy(
        copy_atom,
        bb.make_layout((1, 4), stride=(4, 1)),
        bb.make_layout((2, 1), stride=(1, 1)),
    )
    assert isinstance(copy_atom, bb.CopyAtom)
    assert isinstance(tiled_copy, bb.TiledCopy)
    assert isinstance(cotiled_copy, bb.TiledCopy)

    mma_atom = bb.make_atom(
        SimpleNamespace(tile=(16, 16, 4), dtype="f32", accumulator_dtype="f32", wave_size=64)
    )
    tiled_mma = bb.make_tiled_mma(mma_atom)
    copy_s = bb.make_tiled_copy_S(copy_atom, tiled_mma)
    copy_d = bb.make_tiled_copy_D(copy_atom, tiled_mma)
    copy_c_atom = bb.make_tiled_copy_C_atom(bb.nvgpu.CopyUniversalOp(), "f32", num_bits_per_copy=32)

    assert tiled_mma.shape == (16, 16, 4)
    assert isinstance(copy_s, bb.TiledCopy)
    assert isinstance(copy_d, bb.TiledCopy)
    assert isinstance(copy_c_atom, bb.CopyAtom)


def test_static_range_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    out = bb.zeros((4,), dtype="i32")
    artifact = bb.compile(
        static_range_kernel,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(out)
    assert out[0] == 6
