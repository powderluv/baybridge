import math
import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def unary_math_kernel(src: bb.Tensor, dst: bb.Tensor):
    src_vec = src.load()
    sqrt_res = bb.math.sqrt(src_vec)
    dst.store(sqrt_res)


@bb.jit
def unary_math_wrapper(src: bb.Tensor, dst: bb.Tensor):
    unary_math_kernel(src, dst).launch(grid=(1, 1, 1), block=(1, 1, 1))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduction_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.jit
def reduction_wrapper(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    reduction_kernel(src, dst_scalar, dst_rows, dst_cols).launch(grid=(1, 1, 1), block=(1, 1, 1))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def tensor_factory_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.jit
def tensor_factory_wrapper(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    tensor_factory_kernel(dst_zero, dst_one, dst_full).launch(grid=(1, 1, 1), block=(1, 1, 1))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def broadcast_add_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.jit
def broadcast_add_wrapper(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    broadcast_add_kernel(lhs, rhs, dst).launch(grid=(1, 1, 1), block=(1, 1, 1))


def test_tensor_ssa_runtime_slice_math_and_reduce() -> None:
    src = bb.tensor(
        [
            [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]],
            [[49.0, 64.0, 81.0], [100.0, 121.0, 144.0]],
        ],
        dtype="f32",
    )
    loaded = src.load()

    assert isinstance(loaded, bb.TensorSSA)

    sliced = loaded[(None, 1, None)]
    assert sliced.shape == (2, 3)
    assert sliced.tolist() == [[16.0, 25.0, 36.0], [100.0, 121.0, 144.0]]

    sqrt_res = bb.math.sqrt(loaded)
    assert sqrt_res.tolist() == [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ]

    sin_res = bb.math.sin(bb.tensor([4.0, 4.0, 4.0], dtype="f32"))
    assert all(abs(value - math.sin(4.0)) < 1e-6 for value in sin_res.tolist())

    exp2_res = bb.math.exp2(bb.tensor([4.0, 4.0, 4.0], dtype="f32"))
    assert exp2_res.tolist() == [16.0, 16.0, 16.0]

    reduced_all = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32").reduce(
        bb.ReductionOp.ADD,
        0.0,
        reduction_profile=0,
    )
    assert reduced_all == 21.0

    reduced_rows = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32").reduce(
        bb.ReductionOp.ADD,
        0.0,
        reduction_profile=(None, 1),
    )
    assert reduced_rows.tolist() == [6.0, 15.0]

    reduced_cols = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32").reduce(
        bb.ReductionOp.ADD,
        1.0,
        reduction_profile=(1, None),
    )
    assert reduced_cols.tolist() == [6.0, 8.0, 10.0]

    broadcasted = bb.tensor([[1.0], [2.0]], dtype="f32").broadcast_to((2, 3))
    assert broadcasted.shape == (2, 3)
    assert broadcasted.tolist() == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

    sum_res = bb.tensor([[1.0], [2.0]], dtype="f32") + bb.tensor([[10.0, 20.0, 30.0]], dtype="f32")
    assert sum_res.tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]

    empty = bb.empty_like(bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32"))
    assert empty.shape == (2, 2)
    assert empty.dtype == "f32"

    ones = bb.ones_like(bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32"))
    assert ones.tolist() == [[1, 1], [1, 1]]

    zeroes = bb.zeros_like(bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32"))
    assert zeroes.tolist() == [[0, 0], [0, 0]]

    sevens = bb.full((2, 2), 7.0, dtype="f32")
    assert sevens.tolist() == [[7.0, 7.0], [7.0, 7.0]]


def test_unary_math_and_reduction_runtime_and_compile(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 4.0, 9.0], dtype="f32")
    dst = bb.zeros((3,), dtype="f32")
    unary_math_wrapper(src, dst)
    assert dst.tolist() == [1.0, 2.0, 3.0]

    artifact = bb.compile(unary_math_wrapper, src, dst, cache_dir=tmp_path, backend="hipcc_exec")

    assert artifact.ir is not None
    assert any(operation.op == "math_sqrt" for operation in artifact.ir.operations)

    red_src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")
    reduction_wrapper(red_src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == [21.0]
    assert dst_rows.tolist() == [6.0, 15.0]
    assert dst_cols.tolist() == [6.0, 8.0, 10.0]

    red_artifact = bb.compile(
        reduction_wrapper,
        red_src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )

    assert red_artifact.ir is not None
    assert any(operation.op == "reduce_add" for operation in red_artifact.ir.operations)

    dst_zero = bb.zeros((2, 2), dtype="f32")
    dst_one = bb.zeros((2, 2), dtype="f32")
    dst_full = bb.zeros((2, 2), dtype="f32")
    tensor_factory_wrapper(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist() == [[0, 0], [0, 0]]
    assert dst_one.tolist() == [[1, 1], [1, 1]]
    assert dst_full.tolist() == [[7.0, 7.0], [7.0, 7.0]]

    factory_artifact = bb.compile(
        tensor_factory_wrapper,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )
    assert factory_artifact.ir is not None
    assert any(operation.op == "fill" for operation in factory_artifact.ir.operations)

    lhs = bb.tensor([[1.0], [2.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0, 30.0]], dtype="f32")
    dst = bb.zeros((2, 3), dtype="f32")
    broadcast_add_wrapper(lhs, rhs, dst)
    assert dst.tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]

    broadcast_artifact = bb.compile(
        broadcast_add_wrapper,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )
    assert broadcast_artifact.ir is not None
    assert any(operation.op == "broadcast_to" for operation in broadcast_artifact.ir.operations)


def test_unary_math_and_reduction_run_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    src = bb.tensor([1.0, 4.0, 9.0], dtype="f32")
    dst = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        unary_math_wrapper,
        src,
        dst,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(src, dst)
    assert dst.tolist() == [1.0, 2.0, 3.0]

    red_src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")

    red_artifact = bb.compile(
        reduction_wrapper,
        red_src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    red_artifact(red_src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == [21.0]
    assert dst_rows.tolist() == [6.0, 15.0]
    assert dst_cols.tolist() == [6.0, 8.0, 10.0]

    dst_zero = bb.zeros((2, 2), dtype="f32")
    dst_one = bb.zeros((2, 2), dtype="f32")
    dst_full = bb.zeros((2, 2), dtype="f32")
    factory_artifact = bb.compile(
        tensor_factory_wrapper,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    factory_artifact(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist() == [[0, 0], [0, 0]]
    assert dst_one.tolist() == [[1, 1], [1, 1]]
    assert dst_full.tolist() == [[7.0, 7.0], [7.0, 7.0]]

    lhs = bb.tensor([[1.0], [2.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0, 30.0]], dtype="f32")
    dst = bb.zeros((2, 3), dtype="f32")
    broadcast_artifact = bb.compile(
        broadcast_add_wrapper,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    broadcast_artifact(lhs, rhs, dst)
    assert dst.tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
