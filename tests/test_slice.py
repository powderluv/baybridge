import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def slice_row_copy_kernel(g_a: bb.Tensor, g_out: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    row = bb.slice_(g_a, (1, None))
    g_out[tidx] = row[tidx]


def test_slice_runtime_and_compile(tmp_path: Path) -> None:
    a = bb.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
        ],
        dtype="f32",
    )
    out = bb.zeros((4,), dtype="f32")

    slice_row_copy_kernel(a, out).launch(grid=(1, 1, 1), block=(4, 1, 1))
    assert out.tolist() == [10.0, 20.0, 30.0, 40.0]

    compiled_out = bb.zeros((4,), dtype="f32")
    artifact = bb.compile(slice_row_copy_kernel, a, compiled_out, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]
    assert "slice" in ops
    assert artifact.lowered_module is not None
    assert '"baybridge.slice"' in artifact.lowered_module.text


def test_slice_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    a = bb.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
        ],
        dtype="f32",
    )
    out = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        slice_row_copy_kernel,
        a,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, out)

    assert out.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_assume_is_noop() -> None:
    assert bb.assume(8, divby=4) == 8
