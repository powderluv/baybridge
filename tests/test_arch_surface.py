import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(4, 1, 1),
        block=(64, 1, 1),
        cluster=(2, 1, 1),
        shared_mem_bytes=32,
    )
)
def arch_surface_kernel(out: bb.Tensor):
    block_x = bb.block_idx("x")
    base = block_x * 6
    predicate = bb.arch.elect_one()

    bb.store(bb.Int64(bb.arch.cluster_size()), out, base + 0, predicate=predicate)
    bb.store(bb.Int64(bb.arch.cluster_idx()[0]), out, base + 1, predicate=predicate)
    bb.store(bb.Int64(bb.arch.block_idx_in_cluster()[0]), out, base + 2, predicate=predicate)
    bb.store(bb.Int64(bb.arch.block_rank_in_cluster()), out, base + 3, predicate=predicate)
    bb.store(bb.Int64(bb.arch.get_dyn_smem_size()), out, base + 4, predicate=predicate)
    bb.store(bb.Int64(bb.arch.block_in_cluster_dim()[0]), out, base + 5, predicate=predicate)
    bb.arch.sync_warp()


def _expected_arch_surface() -> list[int]:
    return [
        2, 0, 0, 0, 32, 2,
        2, 0, 1, 1, 32, 2,
        2, 1, 0, 0, 32, 2,
        2, 1, 1, 1, 32, 2,
    ]


def test_arch_surface_runtime_and_compile(tmp_path: Path) -> None:
    out = bb.zeros((24,), dtype="i64")
    arch_surface_kernel(out).launch(
        grid=(4, 1, 1),
        block=(64, 1, 1),
        cluster=(2, 1, 1),
        shared_mem_bytes=32,
    )
    assert out.tolist() == _expected_arch_surface()

    artifact = bb.compile(arch_surface_kernel, out, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.ir is not None
    assert artifact.ir.launch.cluster == (2, 1, 1)
    assert any(operation.op == "barrier" and operation.attrs.get("kind") == "warp" for operation in artifact.ir.operations)


def test_barrier_and_tmem_compat_objects() -> None:
    named = bb.NamedBarrier(barrier_id=1, num_threads=64)
    named.arrive()
    named.wait()
    named.arrive_and_wait()

    mbarriers = bb.MbarrierArray(2)
    mbarriers[0].arrive_and_wait()

    fence = bb.TmaStoreFence()
    fence.arrive_and_wait()

    tmem = bb.TmemAllocator()
    reg = tmem.allocate(8, dtype="f32")
    assert reg.shape == (8,)
    assert reg.dtype == "f32"


def test_arch_surface_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    out = bb.zeros((24,), dtype="i64")
    artifact = bb.compile(
        arch_surface_kernel,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(out)
    assert out.tolist() == _expected_arch_surface()
