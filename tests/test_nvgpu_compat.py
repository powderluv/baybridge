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


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def cpasync_tma_store_kernel(src: bb.Tensor, out: bb.Tensor):
    smem = bb.make_tensor("smem", shape=(16,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    g2s = bb.make_copy_atom(
        bb.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        src.element_type,
        num_bits_per_copy=128,
    )
    s2g = bb.make_copy_atom(
        bb.nvgpu.cpasync.CopyBulkTensorTileS2GOp(tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(g2s, src, smem)
    bb.barrier()
    bb.copy(s2g, smem, out)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def cpasync_tma_reduce_kernel(src: bb.Tensor, out: bb.Tensor):
    smem = bb.make_tensor("smem", shape=(16,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    g2s = bb.make_copy_atom(
        bb.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        src.element_type,
        num_bits_per_copy=128,
    )
    reduce_store = bb.make_copy_atom(
        bb.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(bb.nvgpu.cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(g2s, src, smem)
    bb.barrier()
    bb.copy(reduce_store, smem, out)


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


def test_cpasync_tma_store_and_reduce_route_to_available_backends(tmp_path: Path) -> None:
    src = bb.tensor([float(index + 1) for index in range(16)], dtype="f32")
    out_store = bb.zeros((16,), dtype="f32")
    out_reduce = bb.tensor([10.0 for _ in range(16)], dtype="f32")

    cpasync_tma_store_kernel(src, out_store).launch(grid=(1, 1, 1), block=(1, 1, 1))
    cpasync_tma_reduce_kernel(src, out_reduce).launch(grid=(1, 1, 1), block=(1, 1, 1))
    assert out_store.tolist() == src.tolist()
    assert out_reduce.tolist() == [value + 10.0 for value in src.tolist()]

    store_gpu = bb.compile(cpasync_tma_store_kernel, src, out_store, cache_dir=tmp_path, backend="gpu_text")
    reduce_gpu = bb.compile(cpasync_tma_reduce_kernel, src, out_reduce, cache_dir=tmp_path, backend="gpu_text")
    assert store_gpu.ir is not None
    assert reduce_gpu.ir is not None
    assert any(operation.op == "copy" and operation.attrs.get("copy_variant") == "cpasync_tma_s2g" for operation in store_gpu.ir.operations)
    assert any(
        operation.op == "copy_reduce"
        and operation.attrs.get("copy_variant") == "cpasync_tma_s2g_reduce"
        and operation.attrs.get("reduction") == bb.nvgpu.cpasync.ReductionOp.ADD
        for operation in reduce_gpu.ir.operations
    )
    assert store_gpu.lowered_module is not None
    assert reduce_gpu.lowered_module is not None
    assert "memref.copy" in store_gpu.lowered_module.text
    assert "baybridge.copy_reduce" in reduce_gpu.lowered_module.text

    compiled_store = bb.compile(cpasync_tma_store_kernel, src, out_store, cache_dir=tmp_path, backend="hipcc_exec")
    compiled_reduce = bb.compile(cpasync_tma_reduce_kernel, src, out_reduce, cache_dir=tmp_path, backend="hipcc_exec")
    assert compiled_store.lowered_module is not None
    assert compiled_reduce.lowered_module is not None
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        return
    out_store.fill(0.0)
    out_reduce.fill(10.0)
    compiled_store(src, out_store)
    compiled_reduce(src, out_reduce)
    assert out_store.tolist() == src.tolist()
    assert out_reduce.tolist() == [value + 10.0 for value in src.tolist()]


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


def test_tcgen05_tmem_surface_and_tf32_op_helpers() -> None:
    load_op = tcgen05.Ld16x128bOp(tcgen05.Repetition.x4, tcgen05.Pack.PACK_16)
    store_op = tcgen05.St32x32bOp(tcgen05.Repetition.x2, tcgen05.Unpack.UNPACK_32)
    load_atom = bb.make_copy_atom(load_op, bb.Float16)
    store_atom = bb.make_copy_atom(store_op, bb.Float32)

    assert tcgen05.is_tmem_load(load_atom) is True
    assert tcgen05.is_tmem_store(store_atom) is True
    assert tcgen05.get_tmem_copy_properties(load_atom) == {
        "mode": "load",
        "lanes": 16,
        "bits": 128,
        "pack": tcgen05.Pack.PACK_16,
        "repetition": 4,
    }
    assert tcgen05.get_tmem_copy_properties(store_atom) == {
        "mode": "store",
        "lanes": 32,
        "bits": 32,
        "unpack": tcgen05.Unpack.UNPACK_32,
        "repetition": 2,
    }

    tensor = bb.zeros((4, 8), dtype="f32")
    assert tcgen05.find_tmem_tensor_col_offset(tensor) == 0
    assert tcgen05.tile_to_mma_shape(tensor) == (4, 8)
    assert isinstance(tcgen05.make_s2t_copy(load_atom, tensor), bb.TiledCopy)

    tf32_mma = tcgen05.MmaTF32Op(
        bb.Float32,
        (16, 8, 8),
        tcgen05.CtaGroup.TWO,
        tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.M,
        tcgen05.OperandMajorMode.N,
    )
    tiled_tf32 = bb.make_tiled_mma(tf32_mma)
    assert tiled_tf32.shape == (16, 8, 8)

    tcgen05.commit(bb.MbarrierArray(1)[0], mask=3, cta_group=tcgen05.CtaGroup.TWO)


def test_cpasync_tma_reduce_surface_and_helpers() -> None:
    load_atom = bb.make_copy_atom(cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO), bb.Float16)
    store_atom = bb.make_copy_atom(cpasync.CopyBulkTensorTileS2GOp(tcgen05.CtaGroup.ONE), bb.Float32)
    reduce_atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.TWO),
        bb.Float32,
    )
    dsmem_atom = bb.make_copy_atom(cpasync.CopyDsmemStoreOp(), bb.Float32)

    assert cpasync.is_tma_load(load_atom) is True
    assert cpasync.is_tma_store(store_atom) is True
    assert cpasync.is_tma_store(reduce_atom) is True
    assert cpasync.is_tma_reduce(reduce_atom) is True
    assert cpasync.is_tma_reduce(load_atom) is False
    assert cpasync.get_tma_copy_properties(load_atom) == {
        "mode": "load",
        "variant": "g2s_multicast",
        "cta_group": tcgen05.CtaGroup.TWO,
    }
    assert cpasync.get_tma_copy_properties(store_atom) == {
        "mode": "store",
        "variant": "s2g",
        "cta_group": tcgen05.CtaGroup.ONE,
    }
    assert cpasync.get_tma_copy_properties(reduce_atom) == {
        "mode": "store",
        "variant": "s2g_reduce",
        "cta_group": tcgen05.CtaGroup.TWO,
        "reduction": cpasync.ReductionOp.MAX,
    }
    assert cpasync.get_tma_copy_properties(dsmem_atom) == {
        "mode": "store",
        "variant": "dsmem_store",
        "cta_group": None,
    }


def test_cpasync_extended_tma_helper_surface() -> None:
    tensor = bb.zeros((16, 16), dtype="f16")
    atom, returned_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(tcgen05.CtaGroup.TWO),
        tensor,
        bb.make_layout((16, 16), stride=(16, 1)),
        (16, 8, 16),
        num_multicast=2,
        internal_type="f32",
    )
    cpasync.copy_tensormap(atom, 1234)
    cpasync.update_tma_descriptor(atom, tensor, 5678)
    cpasync.fence_tma_desc_acquire(5678)
    cpasync.cp_fence_tma_desc_release(1234, 5678)
    cpasync.fence_tma_desc_release()

    assert returned_tensor is tensor
    assert atom.type == "f32"
    assert atom.get("num_multicast") == 2
    assert atom.get("tensormap_ptr") == 1234
    assert atom.get("tma_desc_ptr") == 5678
    assert atom.get("gmem_shape") == (16, 16)
    assert cpasync.create_tma_multicast_mask(bb.make_layout((2, 2), stride=(2, 1)), (1, 0), "n") == 0b1100


def test_tcgen05_extended_helper_surface() -> None:
    load_atom = bb.make_copy_atom(tcgen05.Ld16x64bOp(tcgen05.Repetition.x128, tcgen05.Pack.PACK_16b_IN_32b), "i8")
    smem_tensor = bb.zeros((4, 8), dtype="i8")
    layout_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.MN_SW64, "f16")
    umma_desc = tcgen05.make_umma_smem_desc(
        bb.make_ptr("f16", 0x1000, address_space=bb.AddressSpace.SHARED, assumed_align=16),
        bb.make_layout((8, 64), stride=(64, 1)),
        tcgen05.OperandMajorMode.N,
    )
    tiled_mma = bb.make_tiled_mma(
        tcgen05.MmaI8Op(
            "i8",
            "i32",
            (16, 8, 32),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.N,
        )
    )

    assert tcgen05.is_tmem_load(load_atom) is True
    assert tcgen05.get_tmem_copy_properties(load_atom) == {
        "mode": "load",
        "lanes": 16,
        "bits": 64,
        "pack": tcgen05.Pack.PACK_16b_IN_32b,
        "repetition": 128,
    }
    assert tcgen05.get_s2t_smem_desc_tensor(load_atom, smem_tensor) is smem_tensor
    assert layout_atom.shape == (1, 32)
    assert umma_desc.major == tcgen05.OperandMajorMode.N
    assert umma_desc.swizzle == "SWIZZLE_128B"
    assert tiled_mma.shape == (16, 8, 32)


def test_tcgen05_i8_mma_resolves_real_mfma_descriptor() -> None:
    tiled_mma = bb.make_tiled_mma(
        tcgen05.MmaI8Op(
            "i8",
            "i32",
            (16, 16, 32),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.N,
        )
    )

    assert tiled_mma.shape == (16, 16, 32)
    assert tiled_mma.partition_shape_A() == (16, 32)
    assert tiled_mma.partition_shape_B() == (32, 16)
    assert tiled_mma.descriptor.variant_name == "mfma_i32_16x16x32i8"
    assert tiled_mma.descriptor.llvm_intrinsic == "llvm.amdgcn.mfma.i32.16x16x32.i8"


def test_tcgen05_tf32_mma_resolves_real_mfma_descriptor() -> None:
    tiled_mma = bb.make_tiled_mma(
        tcgen05.MmaTF32Op(
            "f32",
            (16, 16, 4),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.M,
            tcgen05.OperandMajorMode.N,
        )
    )

    assert tiled_mma.shape == (16, 16, 4)
    assert tiled_mma.partition_shape_A() == (16, 4)
    assert tiled_mma.partition_shape_B() == (4, 16)
    assert tiled_mma.descriptor.variant_name == "mfma_f32_16x16x4f32"
    assert tiled_mma.descriptor.llvm_intrinsic == "llvm.amdgcn.mfma.f32.16x16x4f32"


def test_tcgen05_f16_mma_resolves_real_mfma_descriptor() -> None:
    tiled_mma = bb.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            "f16",
            "f32",
            (16, 16, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.N,
        )
    )

    assert tiled_mma.shape == (16, 16, 16)
    assert tiled_mma.partition_shape_A() == (16, 16)
    assert tiled_mma.partition_shape_B() == (16, 16)
    assert tiled_mma.descriptor.variant_name == "mfma_f32_16x16x16f16"
    assert tiled_mma.descriptor.llvm_intrinsic == "llvm.amdgcn.mfma.f32.16x16x16f16"


def test_warp_bf16_mma_resolves_real_mfma_descriptor() -> None:
    tiled_mma = bb.make_tiled_mma(
        bb.nvgpu.warp.MmaF16BF16Op("bf16", "f32", (16, 16, 16))
    )

    assert tiled_mma.shape == (16, 16, 16)
    assert tiled_mma.partition_shape_A() == (16, 16)
    assert tiled_mma.partition_shape_B() == (16, 16)
    assert tiled_mma.descriptor.variant_name == "mfma_f32_16x16x16bf16"
    assert tiled_mma.descriptor.llvm_intrinsic == "llvm.amdgcn.mfma.f32.16x16x16bf16"


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def warpgroup_helper_kernel(out: bb.Tensor):
    bb.nvgpu.warpgroup.fence()
    bb.nvgpu.warpgroup.commit_batch()
    bb.nvgpu.warpgroup.wait_batch(0)
    out[0] = bb.Int32(1)


def test_warpgroup_wgmma_and_mbarrier_surface(tmp_path: Path) -> None:
    warpgroup_mma = bb.make_tiled_mma(
        bb.nvgpu.warpgroup.MmaF16BF16Op(bb.Float16, bb.Float32, (64, 8, 16))
    )
    wgmma_tf32 = bb.make_tiled_mma(
        bb.nvgpu.wgmma.MmaTF32Op(bb.Float32, (64, 8, 8))
    )
    assert warpgroup_mma.shape == (64, 8, 16)
    assert wgmma_tf32.shape == (64, 8, 8)

    mbarrier = bb.MbarrierArray(1)[0]
    bb.nvgpu.mbarrier.init(mbarrier, 64)
    bb.nvgpu.mbarrier.init_fence(mbarrier, 64)
    bb.nvgpu.mbarrier.arrive(mbarrier)
    bb.nvgpu.mbarrier.expect_tx(mbarrier, 32)
    bb.nvgpu.mbarrier.arrive_and_expect_tx(mbarrier, 64)
    bb.nvgpu.mbarrier.wait(mbarrier)
    assert bb.nvgpu.mbarrier.try_wait(mbarrier) is True
    assert bb.nvgpu.mbarrier.test_wait(mbarrier) is True

    out = bb.zeros((1,), dtype="i32")
    warpgroup_helper_kernel(out).launch(grid=(1, 1, 1), block=(1, 1, 1))
    assert out.tolist() == [1]

    artifact = bb.compile(warpgroup_helper_kernel, out, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]
    assert "barrier" in ops
    assert "commit_group" in ops
    assert "wait_group" in ops


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


def test_cpasync_tma_store_and_reduce_run_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    src = bb.tensor([float(index + 1) for index in range(16)], dtype="f32")
    out_store = bb.zeros((16,), dtype="f32")
    out_reduce = bb.tensor([10.0 for _ in range(16)], dtype="f32")

    store_artifact = bb.compile(
        cpasync_tma_store_kernel,
        src,
        out_store,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    reduce_artifact = bb.compile(
        cpasync_tma_reduce_kernel,
        src,
        out_reduce,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    store_artifact(src, out_store)
    reduce_artifact(src, out_reduce)

    assert out_store.tolist() == src.tolist()
    assert out_reduce.tolist() == [value + 10.0 for value in src.tolist()]
