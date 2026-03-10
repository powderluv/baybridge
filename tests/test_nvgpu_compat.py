import os
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.nvgpu import cpasync, tcgen05


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(16, 1, 1)))
def cpasync_copy_kernel(src: bb.Tensor, out: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor("smem", shape=(16,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    atom = bb.make_copy_atom(
        bb.nvgpu.cpasync.CopyG2SOp(cache_mode=bb.nvgpu.cpasync.LoadCacheMode.GLOBAL),
        src.element_type,
        num_bits_per_copy=128,
    )
    bb.copy(atom, src, smem)
    bb.barrier()
    out[tidx] = smem[tidx]


def test_cpasync_copy_atom_routes_to_async_copy(tmp_path: Path) -> None:
    src = bb.tensor([float(index + 1) for index in range(16)], dtype="f32")
    out = bb.zeros((16,), dtype="f32")

    cpasync_copy_kernel(src, out).launch(grid=(1, 1, 1), block=(16, 1, 1))
    assert out.tolist() == src.tolist()

    gpu_artifact = bb.compile(cpasync_copy_kernel, src, out, cache_dir=tmp_path, backend="gpu_text")
    assert gpu_artifact.ir is not None
    assert any(operation.op == "copy_async" for operation in gpu_artifact.ir.operations)
    assert gpu_artifact.lowered_module is not None
    assert "gpu.memcpy async" in gpu_artifact.lowered_module.text

    exec_artifact = bb.compile(cpasync_copy_kernel, src, out, cache_dir=tmp_path, backend="hipcc_exec")
    assert exec_artifact.lowered_module is not None
    assert "smem" in exec_artifact.lowered_module.text


def test_nvgpu_mma_compat_helpers_construct_shapes() -> None:
    simt = bb.make_tiled_mma(
        bb.nvgpu.MmaUniversalOp(bb.Float32),
        bb.make_layout((16, 16, 1), stride=(16, 1, 1)),
        permutation_mnk=(64, 64, 1),
    )
    assert simt.shape == (64, 64, 1)

    tensorop = bb.make_tiled_mma(
        bb.nvgpu.warp.MmaF16BF16Op(bb.Float16, bb.Float32, (16, 8, 16)),
        bb.make_layout((2, 2, 1), stride=(2, 1, 1)),
        permutation_mnk=(32, 16, 16),
    )
    assert tensorop.shape == (16, 8, 16)

    ld_a = bb.make_copy_atom(bb.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), bb.Float16)
    ld_b = bb.make_copy_atom(bb.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), bb.Float16)
    copy_a = bb.make_tiled_copy_A(ld_a, tensorop)
    copy_b = bb.make_tiled_copy_B(ld_b, tensorop)
    copy_c = bb.make_tiled_copy_C(bb.make_copy_atom(bb.nvgpu.CopyUniversalOp(), bb.Float32), tensorop)

    assert copy_a.tiler == (16, 16)
    assert copy_b.tiler == (16, 8)
    assert copy_c.tiler == (16, 8)
    assert ld_a.vector_bytes == 2


def test_tma_compat_surface_constructs_passthrough_objects() -> None:
    op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tiled_mma = bb.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            bb.Float16,
            bb.Float32,
            (16, 8, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
    )
    a = bb.tensor([[1.0] * 16 for _ in range(16)], dtype="f16")
    smem = bb.zeros((16, 16), dtype="f16")

    atom_a, tensor_a = bb.nvgpu.make_tiled_tma_atom_A(op, a, bb.make_layout((16, 16), stride=(16, 1)), (16, 8, 16), tiled_mma)
    atom_b, tensor_b = bb.nvgpu.make_tiled_tma_atom_B(op, a, bb.make_layout((16, 16), stride=(16, 1)), (16, 8, 16), tiled_mma)
    cpasync.prefetch_descriptor(atom_a)
    smem_part, gmem_part = cpasync.tma_partition(atom_a, 0, bb.make_layout((1,), stride=(1,)), smem, a)
    tmem_copy = tcgen05.make_tmem_copy(
        bb.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), bb.Float32),
        bb.zeros((4, 8), dtype="f32"),
    )

    assert isinstance(atom_a, bb.CopyAtom)
    assert isinstance(atom_b, bb.CopyAtom)
    assert tensor_a is a
    assert tensor_b is a
    assert smem_part is smem
    assert gmem_part is a
    assert isinstance(tmem_copy, bb.TiledCopy)
    assert tiled_mma.partition_shape_C((128, 256)) == (16, 8)
    assert tiled_mma.set(tcgen05.Field.ACCUMULATE, True) is tiled_mma


def test_cpasync_copy_kernel_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    src = bb.tensor([float(index + 1) for index in range(16)], dtype="f32")
    out = bb.zeros((16,), dtype="f32")

    artifact = bb.compile(
        cpasync_copy_kernel,
        src,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(src, out)

    assert out.tolist() == src.tolist()
