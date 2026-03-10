from pathlib import Path

import baybridge as bb


@bb.kernel
def copy_kernel(
    src: bb.TensorSpec(shape=(128,), dtype="f16"),
    dst: bb.TensorSpec(shape=(128,), dtype="f16"),
):
    bb.copy(src, dst, vector_bytes=16)


@bb.jit(launch=bb.LaunchConfig(grid=(96, 2, 1), block=(128, 2, 1), shared_mem_bytes=8192))
def tiled_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
    b: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    a_tile = bb.partition(a, (16, 16))
    b_tile = bb.partition(b, (16, 16))
    bb.mma(a_tile, b_tile, tile=(16, 16, 16))
    bb.barrier()


def test_mlir_text_backend_emits_module(tmp_path: Path) -> None:
    artifact = bb.compile(copy_kernel, cache_dir=tmp_path, backend="mlir_text")
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.backend_name == "mlir_text"
    assert 'module attributes { baybridge.target = "gfx942", baybridge.backend = "mlir_text", baybridge.wave_size = 64 }' in artifact.lowered_module.text
    assert "func.func @copy_kernel" in artifact.lowered_module.text
    assert "attributes {baybridge.grid = [1, 1, 1], baybridge.block = [1, 1, 1], baybridge.shared_mem_bytes = 0}" in artifact.lowered_module.text
    assert '"baybridge.copy"(%src, %dst) {vector_bytes = 16}' in artifact.lowered_module.text
    assert "!baybridge.tensor<shape=[128], dtype=f16, stride=[1], swizzle=null, space=global>" in artifact.lowered_module.text


def test_mlir_text_backend_emits_partition_and_mma(tmp_path: Path) -> None:
    artifact = bb.compile(tiled_kernel, cache_dir=tmp_path, backend="mlir_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "attributes {baybridge.grid = [96, 2, 1], baybridge.block = [128, 2, 1], baybridge.shared_mem_bytes = 8192}" in text
    assert '"baybridge.partition"(%a)' in text
    assert '"baybridge.partition"(%b)' in text
    assert '"baybridge.make_tensor"()' in text
    assert '"baybridge.mma"(%a_part_1, %b_part_2, %acc)' in text
    assert '"baybridge.barrier"() {kind = "block"}' in text


def test_portable_backend_skips_lowering(tmp_path: Path) -> None:
    artifact = bb.compile(copy_kernel, cache_dir=tmp_path, backend="portable")
    assert artifact.backend_name == "portable"
    assert artifact.lowered_module is None
    assert artifact.lowered_path is None


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(64, 1, 1), shared_mem_bytes=2048))
def staged_copy(
    src: bb.TensorSpec(shape=(64,), dtype="f16"),
    dst: bb.TensorSpec(shape=(64,), dtype="f16"),
):
    smem = bb.make_tensor("smem", shape=(64,), dtype="f16", address_space=bb.AddressSpace.SHARED)
    bb.copy(src, smem, vector_bytes=16)
    bb.barrier()
    bb.copy(smem, dst, vector_bytes=16)


def test_shared_tensor_address_space_reaches_lowering(tmp_path: Path) -> None:
    artifact = bb.compile(staged_copy, cache_dir=tmp_path, backend="mlir_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '!baybridge.tensor<shape=[64], dtype=f16, stride=[1], swizzle=null, space=shared>' in text
    assert '"baybridge.make_tensor"()' in text
    assert 'address_space = "shared"' in text
