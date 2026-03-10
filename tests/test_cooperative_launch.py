import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel
def cooperative_grid_sync_kernel(
    flags: bb.Tensor,
    out: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    writer = tidx == 0
    leader = writer & (bidx == 0)
    bb.store(bb.where(writer, 1.0, 0.0), flags, bidx, predicate=writer)
    bb.arch.sync_grid()
    total = bb.load(flags, 0, predicate=leader, else_value=0.0) + bb.load(
        flags,
        1,
        predicate=leader,
        else_value=0.0,
    )
    bb.store(total, out, 0, predicate=leader)


def test_cooperative_launch_requires_compilation_for_grid_barrier() -> None:
    flags = bb.zeros((2,), dtype="f32")
    out = bb.zeros((1,), dtype="f32")

    with pytest.raises(
        bb.UnsupportedOperationError,
        match="grid-wide barriers require a compiled cooperative launch",
    ):
        cooperative_grid_sync_kernel(flags, out).launch(
            grid=(2, 1, 1),
            block=(64, 1, 1),
            cooperative=True,
        )


def test_grid_barrier_requires_cooperative_hip_launch(tmp_path: Path) -> None:
    flags = bb.zeros((2,), dtype="f32")
    out = bb.zeros((1,), dtype="f32")

    with pytest.raises(
        bb.BackendNotImplementedError,
        match="grid-wide barriers require cooperative launch",
    ):
        bb.compile(
            cooperative_grid_sync_kernel,
            flags,
            out,
            cache_dir=tmp_path,
            backend="hipcc_exec",
        )


@bb.jit
def cooperative_wrapper(
    flags: bb.Tensor,
    out: bb.Tensor,
):
    cooperative_grid_sync_kernel(flags, out).launch(
        grid=(2, 1, 1),
        block=(64, 1, 1),
        cooperative=True,
    )


def test_cooperative_wrapper_emits_cooperative_launch(tmp_path: Path) -> None:
    flags = bb.zeros((2,), dtype="f32")
    out = bb.zeros((1,), dtype="f32")

    artifact = bb.compile(
        cooperative_wrapper,
        flags,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )

    assert artifact.ir is not None
    assert artifact.ir.launch.cooperative is True
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "hipLaunchCooperativeKernel" in text
    assert "cg::this_grid().sync();" in text

    gpu_artifact = bb.compile(
        cooperative_wrapper,
        flags,
        out,
        cache_dir=tmp_path,
        backend="gpu_text",
    )
    assert gpu_artifact.ir is not None
    assert gpu_artifact.ir.launch.cooperative is True
    assert gpu_artifact.lowered_module is not None
    gpu_text = gpu_artifact.lowered_module.text
    assert "gpu.cooperative = true" in gpu_text
    assert '"amdgpu.grid_barrier"() : () -> ()' in gpu_text

def test_cooperative_launch_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    flags = bb.zeros((2,), dtype="f32")
    out = bb.zeros((1,), dtype="f32")

    artifact = bb.compile(
        cooperative_wrapper,
        flags,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(flags, out)

    assert out.tolist() == [2.0]
