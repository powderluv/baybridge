from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(120, 3, 1), block=(256, 1, 1), shared_mem_bytes=2048))
def topology_kernel(
    src: bb.TensorSpec(shape=(128,), dtype="f16"),
    dst: bb.TensorSpec(shape=(128,), dtype="f16"),
):
    bb.program_id("x")
    bb.block_idx("y")
    bb.thread_idx("x")
    bb.block_dim("x")
    bb.grid_dim("y")
    bb.lane_id()
    smem = bb.make_tensor("smem", shape=(128,), dtype="f16", address_space=bb.AddressSpace.SHARED)
    bb.copy(src, smem, vector_bytes=16)
    bb.barrier()
    bb.copy(smem, dst, vector_bytes=16)


@bb.jit(launch=bb.LaunchConfig(grid=(96, 2, 1), block=(128, 2, 1), shared_mem_bytes=8192))
def tiled_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
    b: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    a_tile = bb.partition(a, (16, 16))
    b_tile = bb.partition(b, (16, 16))
    bb.mma(a_tile, b_tile, tile=(16, 16, 16))
    bb.barrier()


def test_gpu_text_backend_emits_gpu_kernel_scaffold(tmp_path: Path) -> None:
    artifact = bb.compile(topology_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert 'gpu.module @kernels attributes {rocdl.target = "gfx942", rocdl.wave_size = 64}' in text
    assert "gpu.func @topology_kernel" in text
    assert 'attributes {gpu.grid = [120, 3, 1], gpu.block = [256, 1, 1], gpu.dynamic_shared_memory = 2048, rocdl.target = "gfx942"}' in text
    assert "gpu.return" in text


def test_gpu_text_backend_maps_topology_builtins(tmp_path: Path) -> None:
    artifact = bb.compile(topology_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "gpu.block_id x" in text
    assert "gpu.block_id y" in text
    assert "gpu.thread_id x" in text
    assert "gpu.block_dim x" in text
    assert "gpu.grid_dim y" in text
    assert '"rocdl.lane_id"() : () -> index' in text


def test_gpu_text_backend_uses_gpu_style_types(tmp_path: Path) -> None:
    artifact = bb.compile(topology_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "memref<128xf16, strided<[1], offset: 0>, 1>" in text
    assert "memref<128xf16, strided<[1], offset: 0>, 3>" in text
    assert "%smem = memref.alloca() : memref<128xf16, strided<[1], offset: 0>, 3>" in text
    assert "memref.copy %src, %smem : memref<128xf16, strided<[1], offset: 0>, 1> to memref<128xf16, strided<[1], offset: 0>, 3>" in text
    assert "gpu.barrier" in text


def test_gpu_text_backend_keeps_register_tensor_as_staging(tmp_path: Path) -> None:
    artifact = bb.compile(tiled_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "%a_part_1 = memref.subview %a[0, 0] [16, 16] [1, 1]" in text
    assert "%b_part_2 = memref.subview %b[0, 0] [16, 16] [1, 1]" in text
    assert '%acc = "amdgpu.alloca_register"()' in text
    assert "!baybridge.reg<16x16xf32>" in text
    assert '"amdgpu.mfma"(%a_part_1, %b_part_2, %acc) {tile = [16, 16, 16], accumulate = true, transpose_a = false, transpose_b = false, variant = "mfma_f32_16x16x16f16", operand_dtype = "f16", accumulator_dtype = "f32", wave_size = 64}' in text


@bb.jit
def bf16_mixed_precision_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="bf16"),
    b: bb.TensorSpec(shape=(64, 64), dtype="bf16"),
):
    a_tile = bb.partition(a, (16, 16))
    b_tile = bb.partition(b, (16, 16))
    acc = bb.make_tensor("acc", shape=(16, 16), dtype="f32", address_space=bb.AddressSpace.REGISTER)
    bb.mma(a_tile, b_tile, c=acc, tile=(16, 16, 16))


def test_gpu_text_backend_selects_mixed_precision_bf16_variant(tmp_path: Path) -> None:
    artifact = bb.compile(bf16_mixed_precision_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "!baybridge.reg<16x16xf32>" in text
    assert '"amdgpu.mfma"(%a_part_1, %b_part_2, %acc) {tile = [16, 16, 16], accumulate = true, transpose_a = false, transpose_b = false, variant = "mfma_f32_16x16x16bf16", operand_dtype = "bf16", accumulator_dtype = "f32", wave_size = 64}' in text


@bb.jit
def fragment_mma_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
    b: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    a_frag = bb.make_fragment_a(a, tile=(16, 16, 16))
    b_frag = bb.make_fragment_b(b, tile=(16, 16, 16))
    bb.mma(a_frag, b_frag, tile=(16, 16, 16))


def test_gpu_text_backend_uses_fragment_views_and_llvm_mfma_intrinsic(tmp_path: Path) -> None:
    artifact = bb.compile(fragment_mma_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert text.count('"amdgpu.fragment_view"') == 2
    assert '"llvm.amdgcn.mfma.f32.16x16x16f16"' in text
    assert 'variant = "mfma_f32_16x16x16f16"' in text


@bb.jit
def unsupported_mma_kernel(
    a: bb.TensorSpec(shape=(8, 8), dtype="f16"),
    b: bb.TensorSpec(shape=(8, 8), dtype="f16"),
):
    a_tile = bb.partition(a, (8, 8))
    b_tile = bb.partition(b, (8, 8))
    bb.mma(a_tile, b_tile, tile=(8, 8, 4))


def test_gpu_text_backend_rejects_unsupported_mfma_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not support mma lowering"):
        bb.compile(unsupported_mma_kernel, cache_dir=tmp_path, backend="gpu_text")
