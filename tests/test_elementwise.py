from pathlib import Path

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(256, 1, 1)))
def elementwise_copy(
    src: bb.TensorSpec(shape=(1024,), dtype="f16"),
    dst: bb.TensorSpec(shape=(1024,), dtype="f16"),
):
    idx = bb.program_id("x") * bb.block_dim("x") + bb.thread_idx("x")
    value = bb.load(src, idx)
    bb.store(value, dst, idx)


def test_elementwise_kernel_reaches_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(elementwise_copy, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "program_id",
        "block_dim",
        "mul",
        "thread_idx",
        "add",
        "load",
        "store",
    ]


def test_elementwise_kernel_reaches_gpu_text_backend(tmp_path: Path) -> None:
    artifact = bb.compile(elementwise_copy, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "gpu.block_id x" in text
    assert "gpu.block_dim x" in text
    assert "gpu.thread_id x" in text
    assert "arith.muli" in text
    assert "arith.addi" in text
    assert "memref.load %src[%add_5] : memref<1024xf16, strided<[1], offset: 0>, 1>" in text
    assert "memref.store %src_val_6, %dst[%add_5] : memref<1024xf16, strided<[1], offset: 0>, 1>" in text
