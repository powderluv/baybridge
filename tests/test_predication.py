from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(5, 1, 1), block=(256, 1, 1)))
def bounded_copy(
    src: bb.TensorSpec(shape=(1024,), dtype="f16"),
    dst: bb.TensorSpec(shape=(1024,), dtype="f16"),
    n: bb.ScalarSpec(dtype="index"),
):
    idx = bb.program_id("x") * bb.block_dim("x") + bb.thread_idx("x")
    in_bounds = idx < n
    value = bb.load(src, idx, predicate=in_bounds, else_value=0)
    bb.store(value, dst, idx, predicate=in_bounds)


def test_bounded_copy_reaches_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(bounded_copy, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "program_id",
        "block_dim",
        "mul",
        "thread_idx",
        "add",
        "cmp_lt",
        "constant",
        "masked_load",
        "masked_store",
    ]


def test_bounded_copy_reaches_gpu_text_backend(tmp_path: Path) -> None:
    artifact = bb.compile(bounded_copy, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "arith.cmpi slt %add_5, %n : index" in text
    assert "scf.if %cmp_lt_6 -> (f16)" in text
    assert "scf.if %cmp_lt_6 {" in text
    assert "scf.yield %cst_7 : f16" in text


@bb.kernel
def invalid_predicate_store(
    src: bb.TensorSpec(shape=(16,), dtype="f16"),
    dst: bb.TensorSpec(shape=(16,), dtype="f16"),
):
    idx = bb.thread_idx("x")
    value = bb.load(src, idx)
    bb.store(value, dst, idx, predicate=idx)


def test_store_rejects_non_boolean_predicate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predicate values must have dtype i1"):
        bb.compile(invalid_predicate_store, cache_dir=tmp_path, backend="portable")


@bb.kernel(launch=bb.LaunchConfig(grid=(5, 1, 1), block=(256, 1, 1)))
def bounded_copy_static(
    src: bb.TensorSpec(shape=(1024,), dtype="f16"),
    dst: bb.TensorSpec(shape=(1024,), dtype="f16"),
):
    idx = bb.program_id("x") * bb.block_dim("x") + bb.thread_idx("x")
    in_bounds = idx < bb.dim(src, 0)
    value = bb.load(src, idx, predicate=in_bounds, else_value=0)
    bb.store(value, dst, idx, predicate=in_bounds)


def test_static_bounded_copy_uses_tensor_dim_in_ir(tmp_path: Path) -> None:
    artifact = bb.compile(bounded_copy_static, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "program_id",
        "block_dim",
        "mul",
        "thread_idx",
        "add",
        "tensor_dim",
        "cmp_lt",
        "constant",
        "masked_load",
        "masked_store",
    ]


def test_static_bounded_copy_uses_tensor_dim_in_gpu_text(tmp_path: Path) -> None:
    artifact = bb.compile(bounded_copy_static, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "arith.constant 1024 : index" in text
    assert "arith.cmpi slt %add_5, %src_dim_0_6 : index" in text
