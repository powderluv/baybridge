import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel
def dynamic_smem_stage_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    allocator = bb.SmemAllocator()
    smem = allocator.allocate_tensor(
        element_type="f32",
        layout=bb.make_layout((4,), stride=(1,)),
        byte_alignment=16,
    )
    smem[tidx] = src[tidx]
    bb.arch.sync_threads()
    dst[tidx] = smem[tidx]


@bb.jit
def dynamic_smem_stage_wrapper(src: bb.Tensor, dst: bb.Tensor):
    kernel = dynamic_smem_stage_kernel(src, dst)
    kernel.launch(grid=(1, 1, 1), block=(4, 1, 1))


def test_kernel_launch_reports_inferred_shared_memory_usage() -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    launch = dynamic_smem_stage_kernel(src, dst)

    assert launch.smem_usage() == 16


def test_dynamic_shared_memory_is_inferred_and_lowered(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        dynamic_smem_stage_wrapper,
        src,
        dst,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )

    assert artifact.ir is not None
    assert artifact.ir.launch.shared_mem_bytes == 16
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "extern __shared__ unsigned char baybridge_dynamic_smem[];" in text
    assert "reinterpret_cast<float*>(baybridge_dynamic_smem + 0)" in text


def test_dynamic_shared_memory_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        dynamic_smem_stage_wrapper,
        src,
        dst,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(src, dst)

    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0]
