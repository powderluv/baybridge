import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(2, 1, 1)))
def local_partition_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tiler, tv = bb.make_layout_tv(
        bb.make_layout((2,), stride=(1,)),
        bb.make_layout((2,), stride=(1,)),
    )
    del tiler
    src_frag = bb.local_partition(src, tv, bb.thread_idx("x"))
    dst_frag = bb.local_partition(dst, tv, bb.thread_idx("x"))
    dst_frag.store(src_frag.load())


def test_local_partition_selects_runtime_tiles() -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    tile = bb.local_partition(src, (2,), 1)
    assert tile.tolist() == [3.0, 4.0]


def test_local_partition_thread_value_runtime_and_compile(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    local_partition_copy_kernel(src, dst).launch(grid=(1, 1, 1), block=(2, 1, 1))
    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0]

    artifact = bb.compile(local_partition_copy_kernel, src, dst, cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None
    assert any(operation.op == "thread_fragment_load" for operation in artifact.ir.operations)
    assert any(operation.op == "thread_fragment_store" for operation in artifact.ir.operations)


def test_local_partition_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")
    artifact = bb.compile(
        local_partition_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(src, dst)
    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0]
