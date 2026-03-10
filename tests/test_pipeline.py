from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(32, 1, 1), block=(256, 1, 1), shared_mem_bytes=4096))
def async_stage(
    src: bb.TensorSpec(shape=(128,), dtype="f16"),
    dst: bb.TensorSpec(shape=(128,), dtype="f16"),
):
    smem = bb.make_tensor("smem", shape=(128,), dtype="f16", address_space=bb.AddressSpace.SHARED)
    bb.copy_async(src, smem, vector_bytes=16, stages=2)
    bb.commit_group()
    bb.wait_group(count=0)
    bb.barrier()
    bb.copy(smem, dst, vector_bytes=16)


def test_pipeline_ops_enter_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(async_stage, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "make_tensor",
        "copy_async",
        "commit_group",
        "wait_group",
        "barrier",
        "copy",
    ]
    assert artifact.ir.operations[1].attrs["stages"] == 2
    assert artifact.ir.operations[3].attrs["count"] == 0


def test_pipeline_ops_reach_mlir_text_backend(tmp_path: Path) -> None:
    artifact = bb.compile(async_stage, cache_dir=tmp_path, backend="mlir_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '"baybridge.copy_async"(%src, %smem)' in text
    assert '{vector_bytes = 16, stages = 2}' in text
    assert '"baybridge.commit_group"() {group = "default"}' in text
    assert '"baybridge.wait_group"() {count = 0, group = "default"}' in text


def test_pipeline_ops_reach_gpu_text_backend(tmp_path: Path) -> None:
    artifact = bb.compile(async_stage, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert 'gpu.memcpy async %smem, %src {vector_bytes = 16, stages = 2}' in text
    assert '"gpu.async.commit_group"() {group = "default"} : () -> ()' in text
    assert '"gpu.async.wait_group"() {count = 0, group = "default"} : () -> ()' in text
    assert "gpu.barrier" in text


@bb.kernel
def invalid_async_destination(
    src: bb.TensorSpec(shape=(16,), dtype="f16"),
):
    reg = bb.make_tensor("reg", shape=(16,), dtype="f16", address_space=bb.AddressSpace.REGISTER)
    bb.copy_async(src, reg)


def test_copy_async_rejects_non_shared_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="destination tensor to be in shared memory"):
        bb.compile(invalid_async_destination, cache_dir=tmp_path, backend="portable")
