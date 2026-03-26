from __future__ import annotations

import math
import warnings
from pathlib import Path

import ctypes
import pytest

import baybridge as bb
from baybridge.cuda_driver import CudaDriver, CudaDriverError
from baybridge.nvgpu import cpasync, tcgen05

pytestmark = pytest.mark.filterwarnings(
    "ignore:ptx_exec is staging RuntimeTensor arguments through host memory"
)


def _assert_nested_tensor_values(actual, expected) -> None:
    if expected and isinstance(expected[0], list):
        if expected[0] and isinstance(expected[0][0], float):
            assert len(actual) == len(expected)
            for actual_row, expected_row in zip(actual, expected):
                assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
            return
    if expected and isinstance(expected[0], float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
        return
    assert actual == expected


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_add_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] + b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_bitand_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] & b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_bitor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] | b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_bitxor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] ^ b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_bitand_i1_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] & b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_cmp_lt_f32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] < b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_select_f32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_select_i32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_select_i1_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_select_scalar_f32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Float32, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_select_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Boolean, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_select_tensor_scalar_i32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], alpha[0], a[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_select_tensor_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], alpha[0], a[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_scalar_cmp_lt_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] < alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_tensor_scalar_cmp_eq_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] == alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_scalar_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] | alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] ^ alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_tensor_scalar_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] & alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_tensor_scalar_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] | alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_tensor_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] ^ alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_copy_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_copy_reduce_max_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_copy_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_copy_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_copy_reduce_max_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_copy_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_add_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] + b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_mul_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] * b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_bitand_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] & b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_bitor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] | b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_bitor_i1_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] | b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_bitxor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] ^ b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_cmp_eq_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] == b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_select_f32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_select_i32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_select_i1_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_select_scalar_f32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Float32, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], alpha, a[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_select_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Boolean, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], alpha, a[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_select_tensor_scalar_i32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_select_tensor_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_scalar_cmp_lt_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] < alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_tensor_scalar_cmp_eq_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] == alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_scalar_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] | alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] ^ alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_scalar_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] & alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_tensor_scalar_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] & alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_tensor_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] ^ alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_cmp_lt_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() < rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_cmp_lt_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() < alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_cmp_eq_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() == alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_select_f32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_select_f32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_select_i32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), alpha[0], src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_sub_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_mul_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_div_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_sub_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_mul_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_div_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_bitand_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_copy_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_copy_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_copy_reduce_add_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_copy_reduce_max_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_copy_reduce_or_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_cmp_eq_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() == rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_select_i32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_sub_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_mul_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_div_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_sub_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_mul_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_div_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_bitor_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_sqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_rsqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.rsqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_sin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_cos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.cos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_exp2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_exp_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_log2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_log_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_log10_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log10(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_round_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_floor_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.floor(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_ceil_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.ceil(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_trunc_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.trunc(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_atan_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_asin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_acos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_erf_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_neg_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_neg_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_abs_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_abs_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_bitnot_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_bitnot_i1_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha[0]))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_sub_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_mul_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_div_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_sub_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() - alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_mul_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() * alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_div_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() / alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() | alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() ^ alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() & alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() ^ alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_sub_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_mul_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_tensor_scalar_div_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_sub_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() - alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_mul_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() * alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_scalar_div_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() / alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_cmp_lt_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() < rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_cmp_lt_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() < alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_bitxor_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_copy_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_copy_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_copy_reduce_add_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_copy_reduce_max_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_copy_reduce_or_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_sqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_sin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_cos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.cos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_exp2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_exp_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_log2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_log_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_log10_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log10(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_round_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_floor_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.floor(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_ceil_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.ceil(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_trunc_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.trunc(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_atan_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_asin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_acos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_erf_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_neg_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_neg_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_abs_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_abs_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_bitnot_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_bitnot_i1_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_sub_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_mul_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_div_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() | alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() ^ alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() & alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() ^ alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_sub_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_mul_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_div_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_cmp_eq_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() == alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_select_f32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_scalar_select_f32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_dense_tensor_scalar_select_i32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), alpha[0], src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_cmp_eq_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() == rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_broadcast_select_i32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_cmp_lt_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() < rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_scalar_cmp_lt_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() < alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_copy_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_copy_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_broadcast_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_broadcast_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_broadcast_cmp_eq_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() == rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_broadcast_bitand_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_tensor_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_tensor_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha[0]))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_tensor_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_tensor_scalar_cmp_eq_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() == alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_select_f32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_scalar_select_f32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_tensor_scalar_select_i32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), alpha[0], src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_broadcast_select_i32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_sqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_sin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_cos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.cos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_exp2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_exp_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_log2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_log_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_log10_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log10(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_round_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_floor_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.floor(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_ceil_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.ceil(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_trunc_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.trunc(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_atan_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_asin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_acos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_broadcast_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_erf_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_neg_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_neg_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_abs_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_abs_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_bitnot_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_bitnot_i1_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_copy_reduce_add_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_copy_reduce_or_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_tensor_factory_bundle_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_tensor_factory_bundle_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_scalar_param_broadcast_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_tensor_scalar_broadcast_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_atan2_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.math.atan2(lhs[tidx], rhs[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_tensor_scalar_atan2_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.math.atan2(src[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_scalar_param_broadcast_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_tensor_scalar_broadcast_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_atan2_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.atan2(lhs[idx], rhs[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_scalar_atan2_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.atan2(src[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_sqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.sqrt(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_rsqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.rsqrt(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_sin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.sin(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_cos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.cos(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_exp2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.exp2(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_exp_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.exp(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_log2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.log2(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_log_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.log(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_log10_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.log10(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_round_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.round(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_floor_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.floor(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_ceil_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.ceil(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_trunc_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.trunc(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_atan_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.atan(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_asin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.asin(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_acos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.acos(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_erf_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.erf(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_neg_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = -src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_neg_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = -src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_abs_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = abs(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_abs_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = abs(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_bitnot_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = ~src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_bitnot_i1_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = ~src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_sqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.sqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_rsqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.rsqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_sin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.sin(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_cos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.cos(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_exp2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.exp2(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_exp_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.exp(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_log2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.log2(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_log_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.log(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_log10_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.log10(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_round_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.round(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_floor_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.floor(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_ceil_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.ceil(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_trunc_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.trunc(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_atan_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.atan(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_asin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.asin(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_acos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.acos(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_erf_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.math.erf(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_neg_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = -src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_neg_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = -src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_abs_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = abs(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_abs_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = abs(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_bitnot_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = ~src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_bitnot_i1_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = ~src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_add_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_mul_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_max_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_min_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_mul_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MUL, 1, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_max_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_min_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MIN, 999, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_and_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.AND, -1, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_exec_parallel_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.XOR, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_add_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_add_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_add_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_add_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_mul_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_mul_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_max_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_max_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_min_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_min_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_mul_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_mul_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_max_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_max_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_min_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_min_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_and_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_and_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_or_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_or_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_rows_xor_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_cols_xor_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_add_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_add_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_add_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_add_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_mul_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_mul_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_max_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_max_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_min_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_min_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_mul_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_mul_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_max_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_max_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_min_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_min_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_and_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_and_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_or_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_or_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_rows_xor_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_cols_xor_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_reduce_rows_add_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_reduce_cols_add_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_reduce_rows_or_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_reduce_cols_or_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_add_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_add_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_mul_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_max_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_min_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_mul_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_max_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_min_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_and_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_or_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_xor_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_tensor_factory_bundle_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_tensor_factory_bundle_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_tensor_factory_bundle_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_exec_parallel_tensor_factory_bundle_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_max_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_add_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_mul_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_max_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_min_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_add_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_mul_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_max_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_min_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_and_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_or_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_xor_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_add_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_reduce_or_i32_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_add_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_reduce_or_i32_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_reduce_add_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_reduce_or_i32_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


def _skip_if_cuda_driver_unavailable() -> None:
    try:
        driver = CudaDriver()
    except (bb.BackendNotImplementedError, CudaDriverError) as exc:
        pytest.skip(f"CUDA driver not usable on this host: {exc}")
    if driver.device_count() < 1:
        pytest.skip("no NVIDIA devices are visible to the CUDA driver")


def _target() -> bb.NvidiaTarget:
    return bb.NvidiaTarget(sm="sm_80", ptx_version="8.0")


class _FakeCudaInteropTensor:
    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: str, stride: tuple[int, ...] | None = None) -> None:
        self._ptr = int(ptr)
        self.shape = shape
        self.dtype = dtype
        self._stride = stride

    def __dlpack__(self):
        return "cuda-capsule"

    def __dlpack_device__(self):
        return (2, 0)

    def data_ptr(self):
        return self._ptr

    def stride(self):
        if self._stride is None:
            raise AttributeError("no stride")
        return self._stride


class _FakeStreamAwareCudaInteropTensor(_FakeCudaInteropTensor):
    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: str, stride: tuple[int, ...] | None = None) -> None:
        super().__init__(ptr, shape, dtype, stride)
        self.dlpack_streams: list[object | None] = []

    def __dlpack__(self, stream=None):
        self.dlpack_streams.append(stream)
        return "cuda-capsule"


class _FakeCpuInteropTensor:
    shape = (128,)
    dtype = "torch.float32"

    def __dlpack__(self):
        return "cpu-capsule"

    def __dlpack_device__(self):
        return (1, 0)

    def data_ptr(self):
        return 1234

    def stride(self):
        return (1,)


def test_ptx_exec_runs_indexed_copy_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_ptx_exec_runs_indexed_copy_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([True, False] * 64, dtype="i1")
    dst = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_ptx_exec_runs_indexed_add_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(index * 2) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs + rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_bitand_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([7 + index for index in range(128)], dtype="i32")
    b = bb.tensor([3 + (index % 5) for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_bitand_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs & rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_bitor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([1 + index for index in range(128)], dtype="i32")
    b = bb.tensor([8 + (index % 7) for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_bitor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs | rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_bitxor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([1 + index for index in range(128)], dtype="i32")
    b = bb.tensor([8 + (index % 7) for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_bitxor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs ^ rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_bitand_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([True, False] * 64, dtype="i1")
    b = bb.tensor([True, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_bitand_i1_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs and rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_cmp_lt_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(index + (index % 3)) for index in range(128)], dtype="f32")
    c = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_cmp_lt_f32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs < rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_select_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(1000 + index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_select_f32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, b, c)
    assert c.tolist() == [lhs if flag else rhs for flag, lhs, rhs in zip(pred.tolist(), a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_select_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    b = bb.tensor([1000 + index for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_select_i32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, b, c)
    assert c.tolist() == [lhs if flag else rhs for flag, lhs, rhs in zip(pred.tolist(), a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_select_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([True, True] * 64, dtype="i1")
    b = bb.tensor([False, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_select_i1_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, b, c)
    assert c.tolist() == [lhs if flag else rhs for flag, lhs, rhs in zip(pred.tolist(), a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_select_scalar_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_select_scalar_f32_kernel,
        pred,
        a,
        bb.Float32(3.5),
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, bb.Float32(3.5), c)
    assert c.tolist() == [lhs if flag else 3.5 for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_indexed_select_scalar_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool((index + 1) % 3) for index in range(128)], dtype="i1")
    c = bb.zeros((128,), dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_select_scalar_i1_kernel,
        pred,
        a,
        bb.Boolean(True),
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, bb.Boolean(True), c)
    assert c.tolist() == [lhs if flag else True for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_indexed_select_tensor_scalar_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([17], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_select_tensor_scalar_i32_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, alpha, c)
    assert c.tolist() == [17 if flag else lhs for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_indexed_select_tensor_scalar_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool(index % 3) for index in range(128)], dtype="i1")
    alpha = bb.tensor([True], dtype="i1")
    c = bb.zeros((128,), dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_select_tensor_scalar_i1_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, alpha, c)
    assert c.tolist() == [True if flag else lhs for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_indexed_scalar_cmp_lt_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_scalar_cmp_lt_f32_kernel,
        src,
        bb.Float32(64.0),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Float32(64.0), dst)
    assert dst.tolist() == [value < 64.0 for value in src.tolist()]


def test_ptx_exec_runs_indexed_tensor_scalar_cmp_eq_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([17], dtype="i32")
    dst = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_tensor_scalar_cmp_eq_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value == 17 for value in src.tolist()]


def test_ptx_exec_warns_once_for_runtime_tensor_staging(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(index * 2) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact(a, b, c)
        artifact(a, b, c)

    staging_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "ptx_exec is staging RuntimeTensor arguments through host memory" in str(warning.message)
    ]
    assert len(staging_warnings) == 1


def test_ptx_exec_reuses_runtime_tensor_device_allocations_across_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_if_cuda_driver_unavailable()

    alloc_count = 0
    original_mem_alloc = CudaDriver.mem_alloc

    def counting_mem_alloc(self, size: int):
        nonlocal alloc_count
        alloc_count += 1
        return original_mem_alloc(self, size)

    monkeypatch.setattr(CudaDriver, "mem_alloc", counting_mem_alloc)

    a0 = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b0 = bb.tensor([float(index * 2) for index in range(128)], dtype="f32")
    c0 = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_add_kernel,
        a0,
        b0,
        c0,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a0, b0, c0)
    first_launch_allocs = alloc_count
    assert first_launch_allocs == 3

    a1 = bb.tensor([float(index + 1) for index in range(128)], dtype="f32")
    b1 = bb.tensor([float(index * 3) for index in range(128)], dtype="f32")
    c1 = bb.zeros((128,), dtype="f32")
    artifact(a1, b1, c1)

    assert alloc_count == first_launch_allocs
    assert c1.tolist()[:8] == pytest.approx([1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0])


def test_ptx_exec_forwards_stream_to_driver_launch_and_stream_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_if_cuda_driver_unavailable()

    launched_streams: list[int | None] = []
    synchronized_streams: list[int] = []

    def fake_launch_kernel(self, function, *, grid, block, shared_mem_bytes=0, stream=None, kernel_params=None) -> None:
        del self, function, grid, block, shared_mem_bytes, kernel_params
        launched_streams.append(stream)

    def fail_context_synchronize(self) -> None:
        del self
        raise AssertionError("ptx_exec should use synchronize_stream when stream= is supplied")

    def fake_synchronize_stream(self, stream) -> None:
        del self
        synchronized_streams.append(int(stream))

    monkeypatch.setattr(CudaDriver, "launch_kernel", fake_launch_kernel)
    monkeypatch.setattr(CudaDriver, "synchronize", fail_context_synchronize)
    monkeypatch.setattr(CudaDriver, "synchronize_stream", fake_synchronize_stream)

    src = bb.from_dlpack(_FakeCudaInteropTensor(1234, (128,), "torch.float32", (1,)))
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (128,), "torch.float32", (1,)))
    artifact = bb.compile(
        ptx_exec_direct_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst, stream=1234)

    assert launched_streams == [1234]
    assert synchronized_streams == [1234]


def test_ptx_exec_passes_stream_to_raw_cuda_dlpack_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_if_cuda_driver_unavailable()

    launched_streams: list[int | None] = []
    synchronized_streams: list[int] = []

    def fake_launch_kernel(self, function, *, grid, block, shared_mem_bytes=0, stream=None, kernel_params=None) -> None:
        del self, function, grid, block, shared_mem_bytes, kernel_params
        launched_streams.append(stream)

    def fail_context_synchronize(self) -> None:
        del self
        raise AssertionError("ptx_exec should use synchronize_stream when stream= is supplied")

    def fake_synchronize_stream(self, stream) -> None:
        del self
        synchronized_streams.append(int(stream))

    monkeypatch.setattr(CudaDriver, "launch_kernel", fake_launch_kernel)
    monkeypatch.setattr(CudaDriver, "synchronize", fail_context_synchronize)
    monkeypatch.setattr(CudaDriver, "synchronize_stream", fake_synchronize_stream)

    src = _FakeStreamAwareCudaInteropTensor(1234, (128,), "torch.float32", (1,))
    dst = _FakeStreamAwareCudaInteropTensor(2345, (128,), "torch.float32", (1,))
    artifact = bb.compile(
        ptx_exec_direct_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )
    src.dlpack_streams.clear()
    dst.dlpack_streams.clear()

    artifact(src, dst, stream=1234)

    assert src.dlpack_streams == [1234]
    assert dst.dlpack_streams == [1234]
    assert launched_streams == [1234]
    assert synchronized_streams == [1234]


def test_ptx_exec_runs_direct_copy_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_ptx_exec_runs_direct_copy_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([True, False] * 64, dtype="i1")
    dst = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == src.tolist()


def test_ptx_exec_runs_direct_add_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(index * 2) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs + rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_bitor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([1 + index for index in range(128)], dtype="i32")
    b = bb.tensor([8 + (index % 7) for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_bitor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs | rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_bitxor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([1 + index for index in range(128)], dtype="i32")
    b = bb.tensor([8 + (index % 7) for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_bitxor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs ^ rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_bitor_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([True, False] * 64, dtype="i1")
    b = bb.tensor([False, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_bitor_i1_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs or rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_cmp_eq_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([index for index in range(128)], dtype="i32")
    b = bb.tensor([index if index % 5 else index + 1 for index in range(128)], dtype="i32")
    c = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_cmp_eq_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs == rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_select_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(1000 + index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_select_f32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, b, c)
    assert c.tolist() == [lhs if flag else rhs for flag, lhs, rhs in zip(pred.tolist(), a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_select_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    b = bb.tensor([1000 + index for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_select_i32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, b, c)
    assert c.tolist() == [lhs if flag else rhs for flag, lhs, rhs in zip(pred.tolist(), a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_select_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([True, True] * 64, dtype="i1")
    b = bb.tensor([False, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_select_i1_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, b, c)
    assert c.tolist() == [lhs if flag else rhs for flag, lhs, rhs in zip(pred.tolist(), a.tolist(), b.tolist())]


def test_ptx_exec_runs_direct_select_scalar_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_select_scalar_f32_kernel,
        pred,
        a,
        bb.Float32(5.5),
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, bb.Float32(5.5), c)
    assert c.tolist() == [5.5 if flag else lhs for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_direct_select_scalar_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool(index % 3) for index in range(128)], dtype="i1")
    c = bb.zeros((128,), dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_select_scalar_i1_kernel,
        pred,
        a,
        bb.Boolean(True),
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, bb.Boolean(True), c)
    assert c.tolist() == [True if flag else lhs for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_direct_select_tensor_scalar_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([23], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_select_tensor_scalar_i32_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, alpha, c)
    assert c.tolist() == [lhs if flag else 23 for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_direct_select_tensor_scalar_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool((index + 1) % 3) for index in range(128)], dtype="i1")
    alpha = bb.tensor([False], dtype="i1")
    c = bb.zeros((128,), dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_select_tensor_scalar_i1_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, a, alpha, c)
    assert c.tolist() == [lhs if flag else False for flag, lhs in zip(pred.tolist(), a.tolist())]


def test_ptx_exec_runs_direct_tensor_scalar_cmp_eq_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([17], dtype="i32")
    dst = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_tensor_scalar_cmp_eq_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value == 17 for value in src.tolist()]


def test_ptx_exec_runs_direct_scalar_cmp_lt_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.tensor([False] * 128, dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_scalar_cmp_lt_f32_kernel,
        src,
        bb.Float32(64.0),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Float32(64.0), dst)
    assert dst.tolist() == [value < 64.0 for value in src.tolist()]


def test_ptx_exec_runs_direct_bitand_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([7 + index for index in range(128)], dtype="i32")
    b = bb.tensor([3 + (index % 5) for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_bitand_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs & rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_indexed_scalar_bitor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_scalar_bitor_i32_kernel,
        src,
        bb.Int32(24),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Int32(24), dst)
    assert dst.tolist() == [value | 24 for value in src.tolist()]


def test_ptx_exec_runs_indexed_scalar_bitxor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_scalar_bitxor_i32_kernel,
        src,
        bb.Int32(24),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Int32(24), dst)
    assert dst.tolist() == [value ^ 24 for value in src.tolist()]


def test_ptx_exec_runs_indexed_tensor_scalar_bitand_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_tensor_scalar_bitand_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value & 11 for value in src.tolist()]


def test_ptx_exec_runs_indexed_tensor_scalar_bitor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_tensor_scalar_bitor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value | 11 for value in src.tolist()]


def test_ptx_exec_runs_indexed_tensor_scalar_bitxor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_tensor_scalar_bitxor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value ^ 11 for value in src.tolist()]


def test_ptx_exec_runs_direct_scalar_bitor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_scalar_bitor_i32_kernel,
        src,
        bb.Int32(24),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Int32(24), dst)
    assert dst.tolist() == [value | 24 for value in src.tolist()]


def test_ptx_exec_runs_direct_scalar_bitxor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_scalar_bitxor_i32_kernel,
        src,
        bb.Int32(24),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Int32(24), dst)
    assert dst.tolist() == [value ^ 24 for value in src.tolist()]


def test_ptx_exec_runs_direct_scalar_bitand_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_scalar_bitand_i32_kernel,
        src,
        bb.Int32(11),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Int32(11), dst)
    assert dst.tolist() == [value & 11 for value in src.tolist()]


def test_ptx_exec_runs_direct_tensor_scalar_bitand_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_tensor_scalar_bitand_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value & 11 for value in src.tolist()]


def test_ptx_exec_runs_direct_tensor_scalar_bitxor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([7 + index for index in range(128)], dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_tensor_scalar_bitxor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value ^ 11 for value in src.tolist()]


def test_ptx_exec_runs_copy_reduce_add_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    artifact = bb.compile(
        ptx_exec_copy_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([11.0, 22.0, 33.0, 44.0], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_copy_reduce_max_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.tensor([0.5, 2.5, 2.0, 8.0], dtype="f32")
    artifact = bb.compile(
        ptx_exec_copy_reduce_max_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([1.0, 2.5, 3.0, 8.0], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_copy_reduce_xor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([1, 2, 3, 4], dtype="i32")
    dst = bb.tensor([8, 8, 8, 8], dtype="i32")
    artifact = bb.compile(
        ptx_exec_copy_reduce_xor_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [9, 10, 11, 12]


def test_ptx_exec_runs_copy_reduce_or_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([1, 2, 4, 8], dtype="i32")
    dst = bb.tensor([16, 16, 16, 16], dtype="i32")
    artifact = bb.compile(
        ptx_exec_copy_reduce_or_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [17, 18, 20, 24]


def test_ptx_exec_runs_indexed_copy_reduce_add_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.tensor([1.0 for _ in range(128)], dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_copy_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([float(index) + 1.0 for index in range(128)], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_indexed_copy_reduce_max_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.tensor([float(index % 5) for index in range(128)], dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_copy_reduce_max_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx(
        [max(float(index), float(index % 5)) for index in range(128)],
        rel=1e-6,
        abs=1e-6,
    )


def test_ptx_exec_runs_indexed_copy_reduce_xor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index for index in range(128)], dtype="i32")
    dst = bb.tensor([3 for _ in range(128)], dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_copy_reduce_xor_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [index ^ 3 for index in range(128)]


def test_ptx_exec_runs_direct_copy_reduce_or_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index for index in range(128)], dtype="i32")
    dst = bb.tensor([3 for _ in range(128)], dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_copy_reduce_or_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [index | 3 for index in range(128)]


def test_ptx_exec_runs_direct_copy_reduce_xor_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index for index in range(128)], dtype="i32")
    dst = bb.tensor([3 for _ in range(128)], dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_copy_reduce_xor_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [index ^ 3 for index in range(128)]


def test_ptx_exec_runs_direct_mul_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    a = bb.tensor([index for index in range(128)], dtype="i32")
    b = bb.tensor([2 for _ in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_mul_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(a, b, c)
    assert c.tolist() == [lhs * rhs for lhs, rhs in zip(a.tolist(), b.tolist())]


def test_ptx_exec_runs_scalar_param_broadcast_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_scalar_param_broadcast_kernel,
        src,
        bb.Float32(1.5),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, 1.5, dst)
    assert dst.tolist() == [value + 1.5 for value in src.tolist()]


def test_ptx_exec_runs_tensor_scalar_broadcast_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    alpha = bb.tensor([1.5], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_tensor_scalar_broadcast_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value + 1.5 for value in src.tolist()]


def test_ptx_exec_runs_indexed_scalar_param_broadcast_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_scalar_param_broadcast_kernel,
        src,
        bb.Float32(1.5),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, 1.5, dst)
    assert dst.tolist() == [value + 1.5 for value in src.tolist()]


def test_ptx_exec_runs_indexed_tensor_scalar_broadcast_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    alpha = bb.tensor([1.5], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_tensor_scalar_broadcast_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [value + 1.5 for value in src.tolist()]


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values"),
    [
        (ptx_exec_dense_copy_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]]),
        (ptx_exec_dense_copy_i32_2d_kernel, "i32", [[1, 2], [3, 4]]),
        (ptx_exec_dense_copy_2d_kernel, "i1", [[True, False], [False, True]]),
    ],
)
def test_ptx_exec_runs_dense_tensor_copy_2d_if_cuda_available(tmp_path: Path, kernel, dtype: str, src_values) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros((len(src_values), len(src_values[0])), dtype=dtype)
    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), src_values):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == src_values


def test_ptx_exec_runs_dense_copy_reduce_add_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")
    artifact = bb.compile(
        ptx_exec_dense_copy_reduce_add_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    for actual_row, expected_row in zip(dst.tolist(), [[11.0, 22.0], [33.0, 44.0]]):
        assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_dense_copy_reduce_max_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[10.0, 1.0], [2.0, 7.0]], dtype="f32")
    artifact = bb.compile(
        ptx_exec_dense_copy_reduce_max_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    for actual_row, expected_row in zip(dst.tolist(), [[10.0, 2.0], [3.0, 7.0]]):
        assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_dense_copy_reduce_or_i32_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1, 2], [4, 8]], dtype="i32")
    dst = bb.tensor([[16, 16], [16, 16]], dtype="i32")
    artifact = bb.compile(
        ptx_exec_dense_copy_reduce_or_i32_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [[17, 18], [20, 24]]


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values"),
    [
        (ptx_exec_parallel_dense_copy_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]]),
        (ptx_exec_parallel_dense_copy_i32_2d_kernel, "i32", [[1, 2], [3, 4]]),
        (ptx_exec_parallel_dense_copy_2d_kernel, "i1", [[True, False], [False, True]]),
    ],
)
def test_ptx_exec_runs_parallel_dense_tensor_copy_2d_if_cuda_available(tmp_path: Path, kernel, dtype: str, src_values) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros((len(src_values), len(src_values[0])), dtype=dtype)
    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), src_values):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == src_values


def test_ptx_exec_runs_parallel_dense_copy_reduce_add_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")
    artifact = bb.compile(
        ptx_exec_parallel_dense_copy_reduce_add_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    for actual_row, expected_row in zip(dst.tolist(), [[11.0, 22.0], [33.0, 44.0]]):
        assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_parallel_dense_copy_reduce_max_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[10.0, 1.0], [2.0, 7.0]], dtype="f32")
    artifact = bb.compile(
        ptx_exec_parallel_dense_copy_reduce_max_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    for actual_row, expected_row in zip(dst.tolist(), [[10.0, 2.0], [3.0, 7.0]]):
        assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_parallel_dense_copy_reduce_or_i32_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1, 2], [4, 8]], dtype="i32")
    dst = bb.tensor([[16, 16], [16, 16]], dtype="i32")
    artifact = bb.compile(
        ptx_exec_parallel_dense_copy_reduce_or_i32_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [[17, 18], [20, 24]]


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha", "expected"),
    [
        (ptx_exec_dense_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], bb.Float32(1.5), [[2.5, 3.5], [4.5, 5.5]]),
        (ptx_exec_dense_scalar_sub_2d_kernel, "f32", [[10.0, 20.0], [30.0, 40.0]], bb.Float32(1.5), [[8.5, 18.5], [28.5, 38.5]]),
        (ptx_exec_dense_scalar_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], bb.Float32(2.0), [[2.0, 4.0], [6.0, 8.0]]),
        (ptx_exec_dense_scalar_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], bb.Float32(2.0), [[4.0, 6.0], [9.0, 14.0]]),
        (ptx_exec_dense_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], bb.Int32(7), [[8, 9], [10, 11]]),
        (ptx_exec_dense_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(5), [[5, 1], [5, 1]]),
        (ptx_exec_dense_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), [[31, 27], [29, 25]]),
        (ptx_exec_dense_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), [[31, 19], [21, 9]]),
        (ptx_exec_dense_scalar_sub_i32_2d_kernel, "i32", [[10, 20], [30, 40]], bb.Int32(7), [[3, 13], [23, 33]]),
        (ptx_exec_dense_scalar_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], bb.Int32(2), [[2, 4], [6, 8]]),
        (ptx_exec_dense_scalar_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], bb.Int32(2), [[4, 6], [9, 14]]),
    ],
)
def test_ptx_exec_runs_dense_tensor_scalar_broadcast_2d_if_cuda_available(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha,
    expected,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, alpha, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha_values", "expected"),
    [
        (ptx_exec_dense_tensor_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [1.5], [[2.5, 3.5], [4.5, 5.5]]),
        (ptx_exec_dense_tensor_scalar_sub_2d_kernel, "f32", [[10.0, 20.0], [30.0, 40.0]], [1.5], [[8.5, 18.5], [28.5, 38.5]]),
        (ptx_exec_dense_tensor_scalar_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [2.0], [[2.0, 4.0], [6.0, 8.0]]),
        (ptx_exec_dense_tensor_scalar_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], [2.0], [[4.0, 6.0], [9.0, 14.0]]),
        (ptx_exec_dense_tensor_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [7], [[8, 9], [10, 11]]),
        (ptx_exec_dense_tensor_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [5], [[5, 1], [5, 1]]),
        (ptx_exec_dense_tensor_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], [[31, 27], [29, 25]]),
        (ptx_exec_dense_tensor_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], [[31, 19], [21, 9]]),
        (ptx_exec_dense_tensor_scalar_sub_i32_2d_kernel, "i32", [[10, 20], [30, 40]], [7], [[3, 13], [23, 33]]),
        (ptx_exec_dense_tensor_scalar_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [2], [[2, 4], [6, 8]]),
        (ptx_exec_dense_tensor_scalar_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], [2], [[4, 6], [9, 14]]),
    ],
)
def test_ptx_exec_runs_dense_tensor_extent1_scalar_broadcast_2d_if_cuda_available(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha_values,
    expected,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype=dtype)
    alpha = bb.tensor(alpha_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, alpha, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha", "expected"),
    [
        (ptx_exec_parallel_dense_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], bb.Float32(1.5), [[2.5, 3.5], [4.5, 5.5]]),
        (ptx_exec_parallel_dense_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], bb.Int32(7), [[8, 9], [10, 11]]),
        (ptx_exec_parallel_dense_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(5), [[5, 1], [5, 1]]),
        (ptx_exec_parallel_dense_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), [[31, 27], [29, 25]]),
        (ptx_exec_parallel_dense_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), [[31, 19], [21, 9]]),
    ],
)
def test_ptx_exec_runs_parallel_dense_tensor_scalar_broadcast_2d_if_cuda_available(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha,
    expected,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, alpha, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha_values", "expected"),
    [
        (ptx_exec_parallel_dense_tensor_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [1.5], [[2.5, 3.5], [4.5, 5.5]]),
        (ptx_exec_parallel_dense_tensor_scalar_sub_2d_kernel, "f32", [[10.0, 20.0], [30.0, 40.0]], [1.5], [[8.5, 18.5], [28.5, 38.5]]),
        (ptx_exec_parallel_dense_tensor_scalar_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [2.0], [[2.0, 4.0], [6.0, 8.0]]),
        (ptx_exec_parallel_dense_tensor_scalar_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], [2.0], [[4.0, 6.0], [9.0, 14.0]]),
        (ptx_exec_parallel_dense_tensor_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [7], [[8, 9], [10, 11]]),
        (ptx_exec_parallel_dense_tensor_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [5], [[5, 1], [5, 1]]),
        (ptx_exec_parallel_dense_tensor_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], [[31, 27], [29, 25]]),
        (ptx_exec_parallel_dense_tensor_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], [[31, 19], [21, 9]]),
        (ptx_exec_parallel_dense_tensor_scalar_sub_i32_2d_kernel, "i32", [[10, 20], [30, 40]], [7], [[3, 13], [23, 33]]),
        (ptx_exec_parallel_dense_tensor_scalar_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [2], [[2, 4], [6, 8]]),
        (ptx_exec_parallel_dense_tensor_scalar_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], [2], [[4, 6], [9, 14]]),
    ],
)
def test_ptx_exec_runs_parallel_dense_tensor_extent1_scalar_broadcast_2d_if_cuda_available(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha_values,
    expected,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype=dtype)
    alpha = bb.tensor(alpha_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, alpha, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "expected"),
    [
        (ptx_exec_dense_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]], [[11.0, 22.0], [33.0, 44.0]]),
        (ptx_exec_dense_sub_2d_kernel, "f32", [[10.0, 20.0], [30.0, 40.0]], [[1.0, 2.0], [3.0, 4.0]], [[9.0, 18.0], [27.0, 36.0]]),
        (ptx_exec_dense_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]], [[10.0, 40.0], [90.0, 160.0]]),
        (ptx_exec_dense_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], [[2.0, 3.0], [6.0, 7.0]], [[4.0, 4.0], [3.0, 4.0]]),
        (ptx_exec_dense_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [[10, 20], [30, 40]], [[11, 22], [33, 44]]),
        (ptx_exec_dense_sub_i32_2d_kernel, "i32", [[10, 20], [30, 40]], [[1, 2], [3, 4]], [[9, 18], [27, 36]]),
        (ptx_exec_dense_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [[10, 20], [30, 40]], [[10, 40], [90, 160]]),
        (ptx_exec_dense_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], [[2, 3], [6, 7]], [[4, 4], [3, 4]]),
        (ptx_exec_dense_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], [[3, 1], [5, 1]]),
        (ptx_exec_dense_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], [[7, 15], [15, 25]]),
        (ptx_exec_dense_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], [[4, 14], [10, 24]]),
        (ptx_exec_dense_bitand_i1_2d_kernel, "i1", [[True, False], [True, False]], [[True, True], [False, False]], [[True, False], [False, False]]),
    ],
)
def test_ptx_exec_runs_dense_tensor_binary_2d_if_cuda_available(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(lhs, rhs, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


def test_ptx_exec_runs_dense_tensor_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[2.0, 1.0], [3.0, 5.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_dense_cmp_lt_f32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[True, False], [False, True]]


def test_ptx_exec_runs_dense_scalar_tensor_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_dense_scalar_cmp_lt_f32_2d_kernel,
        src,
        bb.Float32(3.5),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Float32(3.5), dst)
    assert dst.tolist() == [[True, True], [True, False]]


def test_ptx_exec_runs_dense_tensor_scalar_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_dense_tensor_scalar_cmp_eq_i32_2d_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [[False, False], [True, False]]


def test_ptx_exec_runs_dense_tensor_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(
        ptx_exec_dense_select_f32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, lhs, rhs, dst)
    _assert_nested_tensor_values(dst.tolist(), [[1.0, 20.0], [30.0, 4.0]])


def test_ptx_exec_runs_dense_scalar_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(
        ptx_exec_dense_scalar_select_f32_2d_kernel,
        pred,
        src,
        bb.Float32(9.5),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, src, bb.Float32(9.5), dst)
    _assert_nested_tensor_values(dst.tolist(), [[1.0, 9.5], [9.5, 4.0]])


def test_ptx_exec_runs_dense_tensor_scalar_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.zeros((2, 2), dtype="i32")
    artifact = bb.compile(
        ptx_exec_dense_tensor_scalar_select_i32_2d_kernel,
        pred,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, src, alpha, dst)
    assert dst.tolist() == [[13, 11], [13, 13]]


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "expected"),
    [
        (ptx_exec_parallel_dense_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]], [[11.0, 22.0], [33.0, 44.0]]),
        (ptx_exec_parallel_dense_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [[10, 20], [30, 40]], [[11, 22], [33, 44]]),
        (ptx_exec_parallel_dense_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], [[3, 1], [5, 1]]),
        (ptx_exec_parallel_dense_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], [[7, 15], [15, 25]]),
        (ptx_exec_parallel_dense_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], [[4, 14], [10, 24]]),
        (ptx_exec_parallel_dense_bitxor_i1_2d_kernel, "i1", [[True, False], [True, False]], [[False, True], [True, False]], [[True, True], [False, False]]),
    ],
)
def test_ptx_exec_runs_parallel_dense_tensor_binary_2d_if_cuda_available(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(lhs, rhs, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "expected"),
    [
        (ptx_exec_broadcast_add_2d_kernel, "f32", [[1.0], [2.0]], [[10.0, 20.0, 30.0]], [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]),
        (ptx_exec_broadcast_sub_2d_kernel, "f32", [[10.0], [20.0]], [[1.0, 2.0, 3.0]], [[9.0, 8.0, 7.0], [19.0, 18.0, 17.0]]),
        (ptx_exec_broadcast_mul_2d_kernel, "f32", [[2.0], [3.0]], [[4.0, 5.0, 6.0]], [[8.0, 10.0, 12.0], [12.0, 15.0, 18.0]]),
        (ptx_exec_broadcast_div_2d_kernel, "f32", [[8.0], [12.0]], [[2.0, 4.0, 8.0]], [[4.0, 2.0, 1.0], [6.0, 3.0, 1.5]]),
        (ptx_exec_broadcast_add_i32_2d_kernel, "i32", [[1], [2]], [[10, 20, 30]], [[11, 21, 31], [12, 22, 32]]),
        (ptx_exec_broadcast_sub_i32_2d_kernel, "i32", [[10], [20]], [[1, 2, 3]], [[9, 8, 7], [19, 18, 17]]),
        (ptx_exec_broadcast_mul_i32_2d_kernel, "i32", [[2], [3]], [[4, 5, 6]], [[8, 10, 12], [12, 15, 18]]),
        (ptx_exec_broadcast_div_i32_2d_kernel, "i32", [[8], [12]], [[2, 4, 8]], [[4, 2, 1], [6, 3, 1]]),
        (ptx_exec_broadcast_bitand_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], [[3, 5, 1], [1, 5, 9]]),
        (ptx_exec_broadcast_bitor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], [[7, 7, 15], [15, 13, 13]]),
        (ptx_exec_broadcast_bitxor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], [[4, 2, 14], [14, 8, 4]]),
        (ptx_exec_broadcast_bitor_i1_2d_kernel, "i1", [[True], [False]], [[False, True, False]], [[True, True, True], [False, True, False]]),
    ],
)
def test_ptx_exec_runs_broadcast_tensor_binary_2d_if_cuda_available(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(lhs, rhs, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


def test_ptx_exec_runs_broadcast_tensor_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[7, 5, 7]], dtype="i32")
    dst = bb.tensor([[False, False, False], [False, False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_broadcast_cmp_eq_i32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[True, False, True], [False, False, False]]


def test_ptx_exec_runs_broadcast_tensor_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False, True], [False, True, False]], dtype="i1")
    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[1, 3, 5]], dtype="i32")
    dst = bb.zeros((2, 3), dtype="i32")
    artifact = bb.compile(
        ptx_exec_broadcast_select_i32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, lhs, rhs, dst)
    assert dst.tolist() == [[7, 3, 7], [1, 11, 5]]


def test_ptx_exec_runs_parallel_dense_tensor_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[2.0, 1.0], [3.0, 5.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_parallel_dense_cmp_lt_f32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[True, False], [False, True]]


def test_ptx_exec_runs_parallel_dense_tensor_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(
        ptx_exec_parallel_dense_select_f32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, lhs, rhs, dst)
    _assert_nested_tensor_values(dst.tolist(), [[1.0, 20.0], [30.0, 4.0]])


def test_ptx_exec_runs_parallel_dense_scalar_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(
        ptx_exec_parallel_dense_scalar_select_f32_2d_kernel,
        pred,
        src,
        bb.Float32(9.5),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, src, bb.Float32(9.5), dst)
    _assert_nested_tensor_values(dst.tolist(), [[1.0, 9.5], [9.5, 4.0]])


def test_ptx_exec_runs_parallel_dense_tensor_scalar_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.zeros((2, 2), dtype="i32")
    artifact = bb.compile(
        ptx_exec_parallel_dense_tensor_scalar_select_i32_2d_kernel,
        pred,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, src, alpha, dst)
    assert dst.tolist() == [[13, 11], [13, 13]]


def test_ptx_exec_runs_parallel_dense_scalar_tensor_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_parallel_dense_scalar_cmp_lt_f32_2d_kernel,
        src,
        bb.Float32(3.5),
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, bb.Float32(3.5), dst)
    assert dst.tolist() == [[True, True], [True, False]]


def test_ptx_exec_runs_parallel_dense_tensor_scalar_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_parallel_dense_tensor_scalar_cmp_eq_i32_2d_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, alpha, dst)
    assert dst.tolist() == [[False, False], [True, False]]


def test_ptx_exec_runs_parallel_broadcast_tensor_compare_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[7, 5, 7]], dtype="i32")
    dst = bb.tensor([[False, False, False], [False, False, False]], dtype="i1")
    artifact = bb.compile(
        ptx_exec_parallel_broadcast_cmp_eq_i32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(lhs, rhs, dst)
    assert dst.tolist() == [[True, False, True], [False, False, False]]


def test_ptx_exec_runs_parallel_broadcast_tensor_select_2d_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    pred = bb.tensor([[True, False, True], [False, True, False]], dtype="i1")
    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[1, 3, 5]], dtype="i32")
    dst = bb.zeros((2, 3), dtype="i32")
    artifact = bb.compile(
        ptx_exec_parallel_broadcast_select_i32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(pred, lhs, rhs, dst)
    assert dst.tolist() == [[7, 3, 7], [1, 11, 5]]


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "expected"),
    [
        (ptx_exec_parallel_broadcast_add_2d_kernel, "f32", [[1.0], [2.0]], [[10.0, 20.0, 30.0]], [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]),
        (ptx_exec_parallel_broadcast_add_i32_2d_kernel, "i32", [[1], [2]], [[10, 20, 30]], [[11, 21, 31], [12, 22, 32]]),
        (ptx_exec_parallel_broadcast_bitand_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], [[3, 5, 1], [1, 5, 9]]),
        (ptx_exec_parallel_broadcast_bitor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], [[7, 7, 15], [15, 13, 13]]),
        (ptx_exec_parallel_broadcast_bitxor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], [[4, 2, 14], [14, 8, 4]]),
    ],
)
def test_ptx_exec_runs_parallel_broadcast_tensor_binary_2d_if_cuda_available(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros((len(expected), len(expected[0])), dtype=dtype)
    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(lhs, rhs, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (ptx_exec_dense_sqrt_2d_kernel, [[2.0, 3.0], [4.0, 5.0]]),
        (ptx_exec_dense_rsqrt_2d_kernel, [[0.5, 1.0 / 3.0], [0.25, 0.2]]),
        (ptx_exec_dense_neg_f32_2d_kernel, [[-1.0, 2.0], [-3.5, 4.5]]),
        (ptx_exec_dense_neg_i32_2d_kernel, [[-1, 2], [-3, 4]]),
        (ptx_exec_dense_abs_f32_2d_kernel, [[1.0, 2.0], [3.5, 4.5]]),
        (ptx_exec_dense_abs_i32_2d_kernel, [[1, 2], [3, 4]]),
        (ptx_exec_dense_bitnot_i32_2d_kernel, [[~1, ~2], [~3, ~4]]),
        (ptx_exec_dense_bitnot_i1_2d_kernel, [[False, True], [True, False]]),
    ],
)
def test_ptx_exec_runs_dense_tensor_unary_2d_if_cuda_available(tmp_path: Path, kernel, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    if kernel is ptx_exec_dense_neg_i32_2d_kernel:
        src = bb.tensor([[1, -2], [3, -4]], dtype="i32")
        dst = bb.zeros((2, 2), dtype="i32")
    elif kernel is ptx_exec_dense_abs_i32_2d_kernel:
        src = bb.tensor([[1, -2], [3, -4]], dtype="i32")
        dst = bb.zeros((2, 2), dtype="i32")
    elif kernel is ptx_exec_dense_bitnot_i32_2d_kernel:
        src = bb.tensor([[1, 2], [3, 4]], dtype="i32")
        dst = bb.zeros((2, 2), dtype="i32")
    elif kernel is ptx_exec_dense_bitnot_i1_2d_kernel:
        src = bb.tensor([[True, False], [False, True]], dtype="i1")
        dst = bb.zeros((2, 2), dtype="i1")
    elif kernel is ptx_exec_dense_abs_f32_2d_kernel:
        src = bb.tensor([[1.0, -2.0], [3.5, -4.5]], dtype="f32")
        dst = bb.zeros((2, 2), dtype="f32")
    elif kernel is ptx_exec_dense_neg_f32_2d_kernel:
        src = bb.tensor([[1.0, -2.0], [3.5, -4.5]], dtype="f32")
        dst = bb.zeros((2, 2), dtype="f32")
    else:
        src = bb.tensor([[4.0, 9.0], [16.0, 25.0]], dtype="f32")
        dst = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, dst)
    if dst.dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=5e-4, abs=5e-4)
    else:
        _assert_nested_tensor_values(dst.tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "src", "expected"),
    [
        (ptx_exec_dense_sin_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], [[math.sin(0.0), math.sin(0.5)], [math.sin(1.0), math.sin(1.5)]]),
        (ptx_exec_dense_cos_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], [[math.cos(0.0), math.cos(0.5)], [math.cos(1.0), math.cos(1.5)]]),
        (ptx_exec_dense_exp2_2d_kernel, [[0.0, 1.0], [2.0, 3.0]], [[2.0**0.0, 2.0**1.0], [2.0**2.0, 2.0**3.0]]),
        (ptx_exec_dense_exp_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], [[math.exp(0.0), math.exp(0.5)], [math.exp(1.0), math.exp(1.5)]]),
        (ptx_exec_dense_log2_2d_kernel, [[1.0, 2.0], [4.0, 8.0]], [[math.log2(1.0), math.log2(2.0)], [math.log2(4.0), math.log2(8.0)]]),
        (ptx_exec_dense_log_2d_kernel, [[1.0, 2.0], [4.0, 8.0]], [[math.log(1.0), math.log(2.0)], [math.log(4.0), math.log(8.0)]]),
        (ptx_exec_dense_log10_2d_kernel, [[1.0, 10.0], [100.0, 1000.0]], [[math.log10(1.0), math.log10(10.0)], [math.log10(100.0), math.log10(1000.0)]]),
        (ptx_exec_dense_round_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-2.0, 0.0], [0.0, 2.0]]),
        (ptx_exec_dense_floor_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-2.0, -1.0], [0.0, 1.0]]),
        (ptx_exec_dense_ceil_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-1.0, 0.0], [1.0, 2.0]]),
        (ptx_exec_dense_trunc_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-1.0, 0.0], [0.0, 1.0]]),
        (ptx_exec_dense_atan_2d_kernel, [[-1.5, -0.75], [0.0, 0.75]], [[math.atan(-1.5), math.atan(-0.75)], [math.atan(0.0), math.atan(0.75)]]),
        (ptx_exec_dense_asin_2d_kernel, [[-0.875, -0.5], [0.0, 0.625]], [[math.asin(-0.875), math.asin(-0.5)], [math.asin(0.0), math.asin(0.625)]]),
        (ptx_exec_dense_acos_2d_kernel, [[-0.875, -0.5], [0.0, 0.625]], [[math.acos(-0.875), math.acos(-0.5)], [math.acos(0.0), math.acos(0.625)]]),
        (ptx_exec_dense_erf_2d_kernel, [[-1.5, -0.75], [0.0, 0.75]], [[math.erf(-1.5), math.erf(-0.75)], [math.erf(0.0), math.erf(0.75)]]),
    ],
)
def test_ptx_exec_runs_dense_native_math_tensor_unary_2d_if_cuda_available(tmp_path: Path, kernel, src, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src, dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")

    artifact(src, dst)
    tolerance = 1.2e-2 if kernel in {ptx_exec_dense_asin_2d_kernel, ptx_exec_dense_acos_2d_kernel} else (8e-3 if kernel in {ptx_exec_dense_atan_2d_kernel, ptx_exec_dense_erf_2d_kernel} else 2e-3)
    for actual_row, expected_row in zip(dst.tolist(), expected):
        assert actual_row == pytest.approx(expected_row, rel=tolerance, abs=tolerance)


@pytest.mark.parametrize(
    ("kernel", "src", "dtype", "expected"),
    [
        (ptx_exec_parallel_dense_sqrt_2d_kernel, [[4.0, 9.0], [16.0, 25.0]], "f32", [[2.0, 3.0], [4.0, 5.0]]),
        (ptx_exec_parallel_dense_neg_f32_2d_kernel, [[1.0, -2.0], [3.5, -4.5]], "f32", [[-1.0, 2.0], [-3.5, 4.5]]),
        (ptx_exec_parallel_dense_neg_i32_2d_kernel, [[1, -2], [3, -4]], "i32", [[-1, 2], [-3, 4]]),
        (ptx_exec_parallel_dense_abs_f32_2d_kernel, [[1.0, -2.0], [3.5, -4.5]], "f32", [[1.0, 2.0], [3.5, 4.5]]),
        (ptx_exec_parallel_dense_abs_i32_2d_kernel, [[1, -2], [3, -4]], "i32", [[1, 2], [3, 4]]),
        (ptx_exec_parallel_dense_bitnot_i32_2d_kernel, [[1, 2], [3, 4]], "i32", [[~1, ~2], [~3, ~4]]),
        (ptx_exec_parallel_dense_bitnot_i1_2d_kernel, [[True, False], [False, True]], "i1", [[False, True], [True, False]]),
    ],
)
def test_ptx_exec_runs_parallel_dense_tensor_unary_2d_if_cuda_available(tmp_path: Path, kernel, src, dtype: str, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src, dtype=dtype)
    dst = bb.zeros((2, 2), dtype=dtype)
    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    if dtype == "f32":
        for actual_row, expected_row in zip(dst.tolist(), expected):
            assert actual_row == pytest.approx(expected_row, rel=1e-6, abs=1e-6)
    else:
        _assert_nested_tensor_values(dst.tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "src", "expected"),
    [
        (ptx_exec_parallel_dense_sin_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], [[math.sin(0.0), math.sin(0.5)], [math.sin(1.0), math.sin(1.5)]]),
        (ptx_exec_parallel_dense_cos_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], [[math.cos(0.0), math.cos(0.5)], [math.cos(1.0), math.cos(1.5)]]),
        (ptx_exec_parallel_dense_exp2_2d_kernel, [[0.0, 1.0], [2.0, 3.0]], [[2.0**0.0, 2.0**1.0], [2.0**2.0, 2.0**3.0]]),
        (ptx_exec_parallel_dense_exp_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], [[math.exp(0.0), math.exp(0.5)], [math.exp(1.0), math.exp(1.5)]]),
        (ptx_exec_parallel_dense_log2_2d_kernel, [[1.0, 2.0], [4.0, 8.0]], [[math.log2(1.0), math.log2(2.0)], [math.log2(4.0), math.log2(8.0)]]),
        (ptx_exec_parallel_dense_log_2d_kernel, [[1.0, 2.0], [4.0, 8.0]], [[math.log(1.0), math.log(2.0)], [math.log(4.0), math.log(8.0)]]),
        (ptx_exec_parallel_dense_log10_2d_kernel, [[1.0, 10.0], [100.0, 1000.0]], [[math.log10(1.0), math.log10(10.0)], [math.log10(100.0), math.log10(1000.0)]]),
        (ptx_exec_parallel_dense_round_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-2.0, 0.0], [0.0, 2.0]]),
        (ptx_exec_parallel_dense_floor_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-2.0, -1.0], [0.0, 1.0]]),
        (ptx_exec_parallel_dense_ceil_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-1.0, 0.0], [1.0, 2.0]]),
        (ptx_exec_parallel_dense_trunc_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], [[-1.0, 0.0], [0.0, 1.0]]),
        (ptx_exec_parallel_dense_atan_2d_kernel, [[-1.5, -0.75], [0.0, 0.75]], [[math.atan(-1.5), math.atan(-0.75)], [math.atan(0.0), math.atan(0.75)]]),
        (ptx_exec_parallel_dense_asin_2d_kernel, [[-0.875, -0.5], [0.0, 0.625]], [[math.asin(-0.875), math.asin(-0.5)], [math.asin(0.0), math.asin(0.625)]]),
        (ptx_exec_parallel_dense_acos_2d_kernel, [[-0.875, -0.5], [0.0, 0.625]], [[math.acos(-0.875), math.acos(-0.5)], [math.acos(0.0), math.acos(0.625)]]),
        (ptx_exec_parallel_dense_erf_2d_kernel, [[-1.5, -0.75], [0.0, 0.75]], [[math.erf(-1.5), math.erf(-0.75)], [math.erf(0.0), math.erf(0.75)]]),
    ],
)
def test_ptx_exec_runs_parallel_dense_native_math_tensor_unary_2d_if_cuda_available(tmp_path: Path, kernel, src, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src, dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    tolerance = 1.2e-2 if kernel in {ptx_exec_parallel_dense_asin_2d_kernel, ptx_exec_parallel_dense_acos_2d_kernel} else (8e-3 if kernel in {ptx_exec_parallel_dense_atan_2d_kernel, ptx_exec_parallel_dense_erf_2d_kernel} else 2e-3)
    for actual_row, expected_row in zip(dst.tolist(), expected):
        assert actual_row == pytest.approx(expected_row, rel=tolerance, abs=tolerance)


def test_ptx_exec_runs_indexed_sqrt_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([4.0 + float(index % 5) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_sqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([value**0.5 for value in src.tolist()], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_indexed_rsqrt_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([4.0 + float(index % 5) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_rsqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([1.0 / (value**0.5) for value in src.tolist()], rel=5e-4, abs=5e-4)


@pytest.mark.parametrize(
    ("kernel", "src_values", "expected"),
    [
        (ptx_exec_indexed_sin_kernel, [0.5 + 0.01 * float(index % 16) for index in range(128)], [math.sin(0.5 + 0.01 * float(index % 16)) for index in range(128)]),
        (ptx_exec_indexed_cos_kernel, [0.5 + 0.01 * float(index % 16) for index in range(128)], [math.cos(0.5 + 0.01 * float(index % 16)) for index in range(128)]),
        (ptx_exec_indexed_exp2_kernel, [float(index % 8) * 0.25 for index in range(128)], [2.0 ** (float(index % 8) * 0.25) for index in range(128)]),
        (ptx_exec_indexed_exp_kernel, [0.125 * float(index % 8) for index in range(128)], [math.exp(0.125 * float(index % 8)) for index in range(128)]),
        (ptx_exec_indexed_log2_kernel, [float(1 << (index % 8)) for index in range(128)], [float(index % 8) for index in range(128)]),
        (ptx_exec_indexed_log_kernel, [float(1 << (index % 8)) for index in range(128)], [math.log(float(1 << (index % 8))) for index in range(128)]),
        (ptx_exec_indexed_log10_kernel, [float(10 ** (index % 4)) for index in range(128)], [math.log10(float(10 ** (index % 4))) for index in range(128)]),
        (ptx_exec_indexed_round_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(round(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_indexed_floor_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(math.floor(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_indexed_ceil_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(math.ceil(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_indexed_trunc_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(math.trunc(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_indexed_atan_kernel, [0.25 * float((index % 16) - 8) for index in range(128)], [math.atan(0.25 * float((index % 16) - 8)) for index in range(128)]),
        (ptx_exec_indexed_asin_kernel, [-0.875 + 0.125 * float(index % 15) for index in range(128)], [math.asin(-0.875 + 0.125 * float(index % 15)) for index in range(128)]),
        (ptx_exec_indexed_acos_kernel, [-0.875 + 0.125 * float(index % 15) for index in range(128)], [math.acos(-0.875 + 0.125 * float(index % 15)) for index in range(128)]),
        (ptx_exec_indexed_erf_kernel, [0.25 * float((index % 16) - 8) for index in range(128)], [math.erf(0.25 * float((index % 16) - 8)) for index in range(128)]),
    ],
)
def test_ptx_exec_runs_indexed_native_math_if_cuda_available(tmp_path: Path, kernel, src_values, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    tolerance = 1.2e-2 if kernel in {ptx_exec_indexed_asin_kernel, ptx_exec_indexed_acos_kernel} else (8e-3 if kernel in {ptx_exec_indexed_atan_kernel, ptx_exec_indexed_erf_kernel} else 2e-3)
    assert dst.tolist() == pytest.approx(expected, rel=tolerance, abs=tolerance)


def test_ptx_exec_runs_indexed_neg_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) - 64.0 for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_neg_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([-value for value in src.tolist()], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_indexed_neg_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_neg_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [-value for value in src.tolist()]


def test_ptx_exec_runs_indexed_abs_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) - 64.0 for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_indexed_abs_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([abs(value) for value in src.tolist()], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_indexed_abs_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_abs_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [abs(value) for value in src.tolist()]


def test_ptx_exec_runs_indexed_bitnot_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_indexed_bitnot_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [~value for value in src.tolist()]


def test_ptx_exec_runs_indexed_bitnot_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    dst = bb.zeros((128,), dtype="i1")
    artifact = bb.compile(
        ptx_exec_indexed_bitnot_i1_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [not value for value in src.tolist()]


def test_ptx_exec_runs_direct_sqrt_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([4.0 + float(index % 5) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_sqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([value**0.5 for value in src.tolist()], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("kernel", "src_values", "expected"),
    [
        (ptx_exec_direct_sin_kernel, [0.5 + 0.01 * float(index % 16) for index in range(128)], [math.sin(0.5 + 0.01 * float(index % 16)) for index in range(128)]),
        (ptx_exec_direct_cos_kernel, [0.5 + 0.01 * float(index % 16) for index in range(128)], [math.cos(0.5 + 0.01 * float(index % 16)) for index in range(128)]),
        (ptx_exec_direct_exp2_kernel, [float(index % 8) * 0.25 for index in range(128)], [2.0 ** (float(index % 8) * 0.25) for index in range(128)]),
        (ptx_exec_direct_exp_kernel, [0.125 * float(index % 8) for index in range(128)], [math.exp(0.125 * float(index % 8)) for index in range(128)]),
        (ptx_exec_direct_log2_kernel, [float(1 << (index % 8)) for index in range(128)], [float(index % 8) for index in range(128)]),
        (ptx_exec_direct_log_kernel, [float(1 << (index % 8)) for index in range(128)], [math.log(float(1 << (index % 8))) for index in range(128)]),
        (ptx_exec_direct_log10_kernel, [float(10 ** (index % 4)) for index in range(128)], [math.log10(float(10 ** (index % 4))) for index in range(128)]),
        (ptx_exec_direct_round_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(round(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_direct_floor_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(math.floor(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_direct_ceil_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(math.ceil(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_direct_trunc_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], [float(math.trunc(0.5 * float((index % 16) - 8) + 0.25)) for index in range(128)]),
        (ptx_exec_direct_atan_kernel, [0.25 * float((index % 16) - 8) for index in range(128)], [math.atan(0.25 * float((index % 16) - 8)) for index in range(128)]),
        (ptx_exec_direct_asin_kernel, [-0.875 + 0.125 * float(index % 15) for index in range(128)], [math.asin(-0.875 + 0.125 * float(index % 15)) for index in range(128)]),
        (ptx_exec_direct_acos_kernel, [-0.875 + 0.125 * float(index % 15) for index in range(128)], [math.acos(-0.875 + 0.125 * float(index % 15)) for index in range(128)]),
        (ptx_exec_direct_erf_kernel, [0.25 * float((index % 16) - 8) for index in range(128)], [math.erf(0.25 * float((index % 16) - 8)) for index in range(128)]),
    ],
)
def test_ptx_exec_runs_direct_native_math_if_cuda_available(tmp_path: Path, kernel, src_values, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src_values, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    tolerance = 1.2e-2 if kernel in {ptx_exec_direct_asin_kernel, ptx_exec_direct_acos_kernel} else (8e-3 if kernel in {ptx_exec_direct_atan_kernel, ptx_exec_direct_erf_kernel} else 2e-3)
    assert dst.tolist() == pytest.approx(expected, rel=tolerance, abs=tolerance)


def test_ptx_exec_runs_direct_neg_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) - 64.0 for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_neg_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([-value for value in src.tolist()], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_direct_neg_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_neg_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [-value for value in src.tolist()]


def test_ptx_exec_runs_direct_abs_f32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index) - 64.0 for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_abs_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([abs(value) for value in src.tolist()], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_direct_abs_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_abs_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [abs(value) for value in src.tolist()]


def test_ptx_exec_runs_direct_bitnot_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_direct_bitnot_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [~value for value in src.tolist()]


def test_ptx_exec_runs_direct_rsqrt_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([4.0 + float(index % 5) for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_direct_rsqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([1.0 / (value**0.5) for value in src.tolist()], rel=5e-4, abs=5e-4)


def test_ptx_exec_runs_direct_bitnot_i1_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    dst = bb.zeros((128,), dtype="i1")
    artifact = bb.compile(
        ptx_exec_direct_bitnot_i1_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [not value for value in src.tolist()]


def test_ptx_exec_runs_reduce_add_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((1,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([10.0], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_parallel_reduce_add_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([float(index % 13) for index in range(4096)], dtype="f32")
    dst = bb.zeros((1,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_parallel_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == pytest.approx([sum(src.tolist())], rel=1e-5, abs=1e-5)


def test_ptx_exec_runs_parallel_reduce_add_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([index % 13 for index in range(4096)], dtype="i32")
    dst = bb.zeros((1,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_parallel_reduce_add_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [sum(src.tolist())]


def test_ptx_exec_runs_reduce_or_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([1, 2, 4, 8], dtype="i32")
    dst = bb.zeros((1,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_reduce_or_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [15]


@pytest.mark.parametrize(
    ("kernel", "src", "dtype", "expected"),
    [
        (ptx_exec_parallel_reduce_mul_kernel, [1.0, 2.0, 3.0, 4.0], "f32", pytest.approx([24.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_max_kernel, [1.0, 9.0, 3.0, 7.0], "f32", pytest.approx([9.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_min_kernel, [1.0, 9.0, 3.0, 7.0], "f32", pytest.approx([1.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_mul_i32_kernel, [1, 2, 3, 4], "i32", [24]),
        (ptx_exec_parallel_reduce_max_i32_kernel, [1, 9, 3, 7], "i32", [9]),
        (ptx_exec_parallel_reduce_min_i32_kernel, [1, 9, 3, 7], "i32", [1]),
        (ptx_exec_parallel_reduce_and_i32_kernel, [15, 7, 3], "i32", [3]),
        (ptx_exec_parallel_reduce_or_i32_kernel, [1, 2, 4, 8], "i32", [15]),
        (ptx_exec_parallel_reduce_xor_i32_kernel, [1, 2, 3, 4], "i32", [4]),
    ],
)
def test_ptx_exec_runs_other_parallel_reduce_if_cuda_available(tmp_path: Path, kernel, src, dtype: str, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src_tensor = bb.tensor(src, dtype=dtype)
    dst = bb.zeros((1,), dtype=dtype)
    artifact = bb.compile(
        kernel,
        src_tensor,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src_tensor, dst)
    assert dst.tolist() == expected


def test_ptx_exec_runs_reduce_max_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([1, 9, 3, 7], dtype="i32")
    dst = bb.zeros((1,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_reduce_max_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    assert dst.tolist() == [9]


@pytest.mark.parametrize(
    ("kernel", "src", "dst_shape", "dtype", "expected"),
    [
        (ptx_exec_reduce_rows_add_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_cols_add_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([5.0, 7.0, 9.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_rows_mul_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([6.0, 120.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_cols_mul_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([4.0, 10.0, 18.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_rows_max_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([3.0, 6.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_cols_max_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([4.0, 5.0, 6.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_rows_min_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([1.0, 4.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_cols_min_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([1.0, 2.0, 3.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_rows_add_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [6, 15]),
        (ptx_exec_reduce_cols_add_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [5, 7, 9]),
        (ptx_exec_reduce_rows_mul_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [6, 120]),
        (ptx_exec_reduce_cols_mul_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [4, 10, 18]),
        (ptx_exec_reduce_rows_max_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [3, 6]),
        (ptx_exec_reduce_cols_max_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [4, 5, 6]),
        (ptx_exec_reduce_rows_min_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [1, 4]),
        (ptx_exec_reduce_cols_min_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [1, 2, 3]),
        (ptx_exec_reduce_rows_and_i32_2d_kernel, [[7, 3, 1], [15, 7, 3]], (2,), "i32", [1, 3]),
        (ptx_exec_reduce_cols_and_i32_2d_kernel, [[7, 3, 1], [15, 7, 3]], (3,), "i32", [7, 3, 1]),
        (ptx_exec_reduce_rows_or_i32_2d_kernel, [[1, 2, 4], [8, 1, 2]], (2,), "i32", [7, 11]),
        (ptx_exec_reduce_cols_or_i32_2d_kernel, [[1, 2, 4], [8, 1, 2]], (3,), "i32", [9, 3, 6]),
        (ptx_exec_reduce_rows_xor_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [0, 7]),
        (ptx_exec_reduce_cols_xor_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [5, 7, 5]),
    ],
)
def test_ptx_exec_runs_tensor_reduce_2d_if_cuda_available(tmp_path: Path, kernel, src, dst_shape, dtype: str, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src_tensor = bb.tensor(src, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)
    artifact = bb.compile(
        kernel,
        src_tensor,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src_tensor, dst)
    assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "src", "dst_shape", "dtype", "expected"),
    [
        (ptx_exec_parallel_reduce_rows_add_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_cols_add_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([5.0, 7.0, 9.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_rows_mul_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([6.0, 120.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_cols_mul_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([4.0, 10.0, 18.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_rows_max_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([3.0, 6.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_cols_max_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([4.0, 5.0, 6.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_rows_min_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([1.0, 4.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_cols_min_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([1.0, 2.0, 3.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_rows_add_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [6, 15]),
        (ptx_exec_parallel_reduce_cols_add_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [5, 7, 9]),
        (ptx_exec_parallel_reduce_rows_mul_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [6, 120]),
        (ptx_exec_parallel_reduce_cols_mul_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [4, 10, 18]),
        (ptx_exec_parallel_reduce_rows_max_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [3, 6]),
        (ptx_exec_parallel_reduce_cols_max_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [4, 5, 6]),
        (ptx_exec_parallel_reduce_rows_min_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [1, 4]),
        (ptx_exec_parallel_reduce_cols_min_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [1, 2, 3]),
        (ptx_exec_parallel_reduce_rows_and_i32_2d_kernel, [[7, 3, 1], [15, 7, 3]], (2,), "i32", [1, 3]),
        (ptx_exec_parallel_reduce_cols_and_i32_2d_kernel, [[7, 3, 1], [15, 7, 3]], (3,), "i32", [7, 3, 1]),
        (ptx_exec_parallel_reduce_rows_or_i32_2d_kernel, [[1, 2, 4], [8, 1, 2]], (2,), "i32", [7, 11]),
        (ptx_exec_parallel_reduce_cols_or_i32_2d_kernel, [[1, 2, 4], [8, 1, 2]], (3,), "i32", [9, 3, 6]),
        (ptx_exec_parallel_reduce_rows_xor_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (2,), "i32", [0, 7]),
        (ptx_exec_parallel_reduce_cols_xor_i32_2d_kernel, [[1, 2, 3], [4, 5, 6]], (3,), "i32", [5, 7, 5]),
    ],
)
def test_ptx_exec_runs_parallel_tensor_reduce_2d_if_cuda_available(
    tmp_path: Path, kernel, src, dst_shape, dtype: str, expected
) -> None:
    _skip_if_cuda_driver_unavailable()

    src_tensor = bb.tensor(src, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)
    artifact = bb.compile(
        kernel,
        src_tensor,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src_tensor, dst)
    assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "src", "dst_shape", "dtype", "expected"),
    [
        (ptx_exec_multiblock_reduce_rows_add_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2,), "f32", pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_multiblock_reduce_cols_add_2d_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), "f32", pytest.approx([5.0, 7.0, 9.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_multiblock_reduce_rows_or_i32_2d_kernel, [[1, 2, 4], [8, 1, 2]], (2,), "i32", [7, 11]),
        (ptx_exec_multiblock_reduce_cols_or_i32_2d_kernel, [[1, 2, 4], [8, 1, 2]], (3,), "i32", [9, 3, 6]),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_reduce_2d_if_cuda_available(
    tmp_path: Path, kernel, src, dst_shape, dtype: str, expected
) -> None:
    _skip_if_cuda_driver_unavailable()

    src_tensor = bb.tensor(src, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)
    artifact = bb.compile(
        kernel,
        src_tensor,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src_tensor, dst)
    assert dst.tolist() == expected


def test_ptx_exec_runs_reduce_add_2d_bundle_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_reduce_add_2d_bundle_kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == pytest.approx([21.0], rel=1e-6, abs=1e-6)
    assert dst_rows.tolist() == pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6)
    assert dst_cols.tolist() == pytest.approx([6.0, 8.0, 10.0], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("kernel", "expected_scalar", "expected_rows", "expected_cols"),
    [
        (ptx_exec_reduce_mul_2d_bundle_kernel, [720.0], [6.0, 120.0], [4.0, 10.0, 18.0]),
        (ptx_exec_reduce_max_2d_bundle_kernel, [6.0], [3.0, 6.0], [4.0, 5.0, 6.0]),
        (ptx_exec_reduce_min_2d_bundle_kernel, [1.0], [1.0, 4.0], [1.0, 2.0, 3.0]),
    ],
)
def test_ptx_exec_runs_other_reduce_2d_bundle_if_cuda_available(
    tmp_path: Path,
    kernel,
    expected_scalar,
    expected_rows,
    expected_cols,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")
    artifact = bb.compile(
        kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == pytest.approx(expected_scalar, rel=1e-6, abs=1e-6)
    assert dst_rows.tolist() == pytest.approx(expected_rows, rel=1e-6, abs=1e-6)
    assert dst_cols.tolist() == pytest.approx(expected_cols, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("kernel", "expected_scalar", "expected_rows", "expected_cols"),
    [
        (ptx_exec_reduce_add_i32_2d_bundle_kernel, [21], [6, 15], [5, 7, 9]),
        (ptx_exec_reduce_mul_i32_2d_bundle_kernel, [720], [6, 120], [4, 10, 18]),
        (ptx_exec_reduce_max_i32_2d_bundle_kernel, [6], [3, 6], [4, 5, 6]),
        (ptx_exec_reduce_min_i32_2d_bundle_kernel, [1], [1, 4], [1, 2, 3]),
        (ptx_exec_reduce_and_i32_2d_bundle_kernel, [0], [0, 4], [0, 0, 2]),
        (ptx_exec_reduce_or_i32_2d_bundle_kernel, [7], [3, 7], [5, 7, 7]),
        (ptx_exec_reduce_xor_i32_2d_bundle_kernel, [7], [0, 7], [5, 7, 5]),
    ],
)
def test_ptx_exec_runs_i32_reduce_2d_bundle_if_cuda_available(
    tmp_path: Path,
    kernel,
    expected_scalar,
    expected_rows,
    expected_cols,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="i32")
    dst_scalar = bb.zeros((1,), dtype="i32")
    dst_rows = bb.zeros((2,), dtype="i32")
    dst_cols = bb.zeros((3,), dtype="i32")
    artifact = bb.compile(
        kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == expected_scalar
    assert dst_rows.tolist() == expected_rows
    assert dst_cols.tolist() == expected_cols


@pytest.mark.parametrize(
    ("kernel", "src", "dtype", "expected_rows", "expected_cols"),
    [
        (ptx_exec_reduce_add_rowcol_2d_bundle_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "f32", pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6), pytest.approx([6.0, 8.0, 10.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_reduce_or_i32_rowcol_2d_bundle_kernel, [[1, 2, 4], [8, 1, 2]], "i32", [7, 11], [9, 3, 6]),
    ],
)
def test_ptx_exec_runs_rowcol_reduce_2d_bundle_if_cuda_available(
    tmp_path: Path, kernel, src, dtype: str, expected_rows, expected_cols
) -> None:
    _skip_if_cuda_driver_unavailable()

    src_tensor = bb.tensor(src, dtype=dtype)
    dst_rows = bb.zeros((2,), dtype=dtype)
    dst_cols = bb.zeros((3,), dtype=dtype)
    artifact = bb.compile(
        kernel,
        src_tensor,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src_tensor, dst_rows, dst_cols)
    assert dst_rows.tolist() == expected_rows
    assert dst_cols.tolist() == expected_cols


def test_ptx_exec_runs_parallel_reduce_add_2d_bundle_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")
    artifact = bb.compile(
        ptx_exec_parallel_reduce_add_2d_bundle_kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == pytest.approx([21.0], rel=1e-6, abs=1e-6)
    assert dst_rows.tolist() == pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6)
    assert dst_cols.tolist() == pytest.approx([6.0, 8.0, 10.0], rel=1e-6, abs=1e-6)


def test_ptx_exec_runs_parallel_reduce_add_i32_2d_bundle_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="i32")
    dst_scalar = bb.zeros((1,), dtype="i32")
    dst_rows = bb.zeros((2,), dtype="i32")
    dst_cols = bb.zeros((3,), dtype="i32")
    artifact = bb.compile(
        ptx_exec_parallel_reduce_add_i32_2d_bundle_kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == [21]
    assert dst_rows.tolist() == [6, 15]
    assert dst_cols.tolist() == [5, 7, 9]


@pytest.mark.parametrize(
    ("kernel", "src", "dtype", "expected_rows", "expected_cols"),
    [
        (ptx_exec_parallel_reduce_add_rowcol_2d_bundle_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "f32", pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6), pytest.approx([6.0, 8.0, 10.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_or_i32_rowcol_2d_bundle_kernel, [[1, 2, 4], [8, 1, 2]], "i32", [7, 11], [9, 3, 6]),
    ],
)
def test_ptx_exec_runs_parallel_rowcol_reduce_2d_bundle_if_cuda_available(
    tmp_path: Path, kernel, src, dtype: str, expected_rows, expected_cols
) -> None:
    _skip_if_cuda_driver_unavailable()

    src_tensor = bb.tensor(src, dtype=dtype)
    dst_rows = bb.zeros((2,), dtype=dtype)
    dst_cols = bb.zeros((3,), dtype=dtype)
    artifact = bb.compile(
        kernel,
        src_tensor,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src_tensor, dst_rows, dst_cols)
    assert dst_rows.tolist() == expected_rows
    assert dst_cols.tolist() == expected_cols


@pytest.mark.parametrize(
    ("kernel", "src", "dtype", "expected_rows", "expected_cols"),
    [
        (ptx_exec_multiblock_reduce_add_rowcol_2d_bundle_kernel, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "f32", pytest.approx([6.0, 15.0], rel=1e-6, abs=1e-6), pytest.approx([6.0, 8.0, 10.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_multiblock_reduce_or_i32_rowcol_2d_bundle_kernel, [[1, 2, 4], [8, 1, 2]], "i32", [7, 11], [9, 3, 6]),
    ],
)
def test_ptx_exec_runs_multiblock_rowcol_reduce_2d_bundle_if_cuda_available(
    tmp_path: Path, kernel, src, dtype: str, expected_rows, expected_cols
) -> None:
    _skip_if_cuda_driver_unavailable()

    src_tensor = bb.tensor(src, dtype=dtype)
    dst_rows = bb.zeros((2,), dtype=dtype)
    dst_cols = bb.zeros((3,), dtype=dtype)
    artifact = bb.compile(
        kernel,
        src_tensor,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src_tensor, dst_rows, dst_cols)
    assert dst_rows.tolist() == expected_rows
    assert dst_cols.tolist() == expected_cols


@pytest.mark.parametrize(
    ("kernel", "expected_scalar", "expected_rows", "expected_cols"),
    [
        (ptx_exec_parallel_reduce_mul_2d_bundle_kernel, [720.0], [6.0, 120.0], [4.0, 10.0, 18.0]),
        (ptx_exec_parallel_reduce_max_2d_bundle_kernel, [6.0], [3.0, 6.0], [4.0, 5.0, 6.0]),
        (ptx_exec_parallel_reduce_min_2d_bundle_kernel, [1.0], [1.0, 4.0], [1.0, 2.0, 3.0]),
    ],
)
def test_ptx_exec_runs_other_parallel_reduce_2d_bundle_if_cuda_available(
    tmp_path: Path,
    kernel,
    expected_scalar,
    expected_rows,
    expected_cols,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")
    artifact = bb.compile(
        kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == pytest.approx(expected_scalar, rel=1e-6, abs=1e-6)
    assert dst_rows.tolist() == pytest.approx(expected_rows, rel=1e-6, abs=1e-6)
    assert dst_cols.tolist() == pytest.approx(expected_cols, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("kernel", "expected_scalar", "expected_rows", "expected_cols"),
    [
        (ptx_exec_parallel_reduce_mul_i32_2d_bundle_kernel, [720], [6, 120], [4, 10, 18]),
        (ptx_exec_parallel_reduce_max_i32_2d_bundle_kernel, [6], [3, 6], [4, 5, 6]),
        (ptx_exec_parallel_reduce_min_i32_2d_bundle_kernel, [1], [1, 4], [1, 2, 3]),
        (ptx_exec_parallel_reduce_and_i32_2d_bundle_kernel, [0], [0, 4], [0, 0, 2]),
        (ptx_exec_parallel_reduce_or_i32_2d_bundle_kernel, [7], [3, 7], [5, 7, 7]),
        (ptx_exec_parallel_reduce_xor_i32_2d_bundle_kernel, [7], [0, 7], [5, 7, 5]),
    ],
)
def test_ptx_exec_runs_other_parallel_i32_reduce_2d_bundle_if_cuda_available(
    tmp_path: Path,
    kernel,
    expected_scalar,
    expected_rows,
    expected_cols,
) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="i32")
    dst_scalar = bb.zeros((1,), dtype="i32")
    dst_rows = bb.zeros((2,), dtype="i32")
    dst_cols = bb.zeros((3,), dtype="i32")
    artifact = bb.compile(
        kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst_scalar, dst_rows, dst_cols)
    assert dst_scalar.tolist() == expected_scalar
    assert dst_rows.tolist() == expected_rows
    assert dst_cols.tolist() == expected_cols


def test_ptx_exec_runs_tensor_factory_bundle_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    dst_zero = bb.zeros((2, 2), dtype="f32")
    dst_one = bb.zeros((2, 2), dtype="f32")
    dst_full = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(
        ptx_exec_tensor_factory_bundle_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert dst_one.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert dst_full.tolist() == [[7.0, 7.0], [7.0, 7.0]]


def test_ptx_exec_runs_tensor_factory_bundle_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    dst_zero = bb.zeros((2, 2), dtype="i32")
    dst_one = bb.zeros((2, 2), dtype="i32")
    dst_full = bb.zeros((2, 2), dtype="i32")
    artifact = bb.compile(
        ptx_exec_tensor_factory_bundle_i32_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist() == [[0, 0], [0, 0]]
    assert dst_one.tolist() == [[1, 1], [1, 1]]
    assert dst_full.tolist() == [[7, 7], [7, 7]]


def test_ptx_exec_runs_parallel_tensor_factory_bundle_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    dst_zero = bb.zeros((64, 64), dtype="f32")
    dst_one = bb.zeros((64, 64), dtype="f32")
    dst_full = bb.zeros((64, 64), dtype="f32")
    artifact = bb.compile(
        ptx_exec_parallel_tensor_factory_bundle_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist()[0][:4] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert dst_one.tolist()[0][:4] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert dst_full.tolist()[0][:4] == pytest.approx([7.0, 7.0, 7.0, 7.0])


def test_ptx_exec_runs_parallel_tensor_factory_bundle_i32_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    dst_zero = bb.zeros((64, 64), dtype="i32")
    dst_one = bb.zeros((64, 64), dtype="i32")
    dst_full = bb.zeros((64, 64), dtype="i32")
    artifact = bb.compile(
        ptx_exec_parallel_tensor_factory_bundle_i32_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(dst_zero, dst_one, dst_full)
    assert dst_zero.tolist()[0][:4] == [0, 0, 0, 0]
    assert dst_one.tolist()[0][:4] == [1, 1, 1, 1]
    assert dst_full.tolist()[0][:4] == [7, 7, 7, 7]


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_multiblock_dense_copy_2d_kernel,
            (bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"), bb.zeros((4, 3), dtype="f32")),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ),
        (
            ptx_exec_multiblock_dense_copy_i32_2d_kernel,
            (bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"), bb.zeros((4, 3), dtype="i32")),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        ),
        (
            ptx_exec_multiblock_dense_copy_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.zeros((4, 3), dtype="i1"),
            ),
            [[True, False, True], [False, True, False], [True, True, False], [False, False, True]],
        ),
        (
            ptx_exec_multiblock_dense_add_2d_kernel,
            (
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
            [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0], [77.0, 88.0, 99.0], [110.0, 121.0, 132.0]],
        ),
        (
            ptx_exec_multiblock_dense_add_i32_2d_kernel,
            (
                bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"),
                bb.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            [[11, 22, 33], [44, 55, 66], [77, 88, 99], [110, 121, 132]],
        ),
        (
            ptx_exec_multiblock_dense_bitxor_i32_2d_kernel,
            (
                bb.tensor([[7, 11, 13], [17, 19, 23], [29, 31, 37], [41, 43, 47]], dtype="i32"),
                bb.tensor([[3, 5, 9], [6, 10, 12], [15, 17, 19], [21, 22, 23]], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            [[4, 14, 4], [23, 25, 27], [18, 14, 54], [60, 61, 56]],
        ),
        (
            ptx_exec_multiblock_broadcast_add_2d_kernel,
            (bb.tensor([[1.0], [2.0], [3.0], [4.0]], dtype="f32"), bb.tensor([[10.0, 20.0, 30.0]], dtype="f32"), bb.zeros((4, 3), dtype="f32")),
            [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0], [13.0, 23.0, 33.0], [14.0, 24.0, 34.0]],
        ),
        (
            ptx_exec_multiblock_broadcast_bitor_i32_2d_kernel,
            (bb.tensor([[1], [2], [4], [8]], dtype="i32"), bb.tensor([[16, 32, 64]], dtype="i32"), bb.zeros((4, 3), dtype="i32")),
            [[17, 33, 65], [18, 34, 66], [20, 36, 68], [24, 40, 72]],
        ),
        (
            ptx_exec_multiblock_broadcast_bitand_i1_2d_kernel,
            (
                bb.tensor([[True], [False], [True], [False]], dtype="i1"),
                bb.tensor([[True, False, True]], dtype="i1"),
                bb.zeros((4, 3), dtype="i1"),
            ),
            [[True, False, True], [False, False, False], [True, False, True], [False, False, False]],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_elementwise_2d_if_cuda_available(
    tmp_path: Path, kernel, args, expected
) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    _assert_nested_tensor_values(args[-1].tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_multiblock_dense_scalar_add_2d_kernel,
            (bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"), bb.Float32(1.5), bb.zeros((4, 3), dtype="f32")),
            [[2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5], [11.5, 12.5, 13.5]],
        ),
        (
            ptx_exec_multiblock_dense_scalar_bitand_i32_2d_kernel,
            (bb.tensor([[7, 11, 13], [17, 19, 23], [29, 31, 37], [41, 43, 47]], dtype="i32"), bb.Int32(5), bb.zeros((4, 3), dtype="i32")),
            [[5, 1, 5], [1, 1, 5], [5, 5, 5], [1, 1, 5]],
        ),
        (
            ptx_exec_multiblock_dense_tensor_scalar_add_2d_kernel,
            (bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"), bb.tensor([1.5], dtype="f32"), bb.zeros((4, 3), dtype="f32")),
            [[2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5], [11.5, 12.5, 13.5]],
        ),
        (
            ptx_exec_multiblock_dense_tensor_scalar_bitor_i32_2d_kernel,
            (bb.tensor([[1, 2, 4], [8, 16, 32], [3, 5, 9], [6, 10, 12]], dtype="i32"), bb.tensor([24], dtype="i32"), bb.zeros((4, 3), dtype="i32")),
            [[25, 26, 28], [24, 24, 56], [27, 29, 25], [30, 26, 28]],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_scalar_broadcast_2d_if_cuda_available(
    tmp_path: Path, kernel, args, expected
) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    _assert_nested_tensor_values(args[-1].tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_multiblock_dense_cmp_lt_f32_2d_kernel,
            (
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.tensor([[2.0, 1.0, 4.0], [4.0, 6.0, 5.0], [8.0, 7.0, 9.0], [11.0, 10.0, 13.0]], dtype="f32"),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            [[True, False, True], [False, True, False], [True, False, False], [True, False, True]],
        ),
        (
            ptx_exec_multiblock_broadcast_cmp_eq_i32_2d_kernel,
            (
                bb.tensor([[1], [2], [4], [8]], dtype="i32"),
                bb.tensor([[1, 3, 1]], dtype="i32"),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            [[True, False, True], [False, False, False], [False, False, False], [False, False, False]],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_compare_2d_if_cuda_available(tmp_path: Path, kernel, args, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    _assert_nested_tensor_values(args[-1].tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_multiblock_dense_scalar_cmp_lt_f32_2d_kernel,
            (
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.Float32(8.5),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            [[True, True, True], [True, True, True], [True, True, False], [False, False, False]],
        ),
        (
            ptx_exec_multiblock_dense_tensor_scalar_cmp_eq_i32_2d_kernel,
            (
                bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"),
                bb.tensor([8], dtype="i32"),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            [[False, False, False], [False, False, False], [False, True, False], [False, False, False]],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_scalar_compare_2d_if_cuda_available(
    tmp_path: Path, kernel, args, expected
) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    _assert_nested_tensor_values(args[-1].tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "src", "dtype", "expected"),
    [
        (ptx_exec_multiblock_dense_sqrt_2d_kernel, [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0], [49.0, 64.0, 81.0], [100.0, 121.0, 144.0]], "f32", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        (ptx_exec_multiblock_dense_sin_2d_kernel, [[0.0, 0.5, 1.0], [1.5, 0.25, 0.75], [1.25, 0.1, 0.9], [0.2, 0.4, 0.6]], "f32", [[math.sin(0.0), math.sin(0.5), math.sin(1.0)], [math.sin(1.5), math.sin(0.25), math.sin(0.75)], [math.sin(1.25), math.sin(0.1), math.sin(0.9)], [math.sin(0.2), math.sin(0.4), math.sin(0.6)]]),
        (ptx_exec_multiblock_dense_cos_2d_kernel, [[0.0, 0.5, 1.0], [1.5, 0.25, 0.75], [1.25, 0.1, 0.9], [0.2, 0.4, 0.6]], "f32", [[math.cos(0.0), math.cos(0.5), math.cos(1.0)], [math.cos(1.5), math.cos(0.25), math.cos(0.75)], [math.cos(1.25), math.cos(0.1), math.cos(0.9)], [math.cos(0.2), math.cos(0.4), math.cos(0.6)]]),
        (ptx_exec_multiblock_dense_exp2_2d_kernel, [[0.0, 1.0, 2.0], [3.0, 0.5, 1.5], [2.5, 0.25, 1.25], [1.75, 2.25, 0.75]], "f32", [[2.0**0.0, 2.0**1.0, 2.0**2.0], [2.0**3.0, 2.0**0.5, 2.0**1.5], [2.0**2.5, 2.0**0.25, 2.0**1.25], [2.0**1.75, 2.0**2.25, 2.0**0.75]]),
        (ptx_exec_multiblock_dense_exp_2d_kernel, [[0.0, 0.5, 1.0], [1.5, 0.25, 0.75], [1.25, 0.1, 0.9], [0.2, 0.4, 0.6]], "f32", [[math.exp(0.0), math.exp(0.5), math.exp(1.0)], [math.exp(1.5), math.exp(0.25), math.exp(0.75)], [math.exp(1.25), math.exp(0.1), math.exp(0.9)], [math.exp(0.2), math.exp(0.4), math.exp(0.6)]]),
        (ptx_exec_multiblock_dense_log2_2d_kernel, [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0], [512.0, 1024.0, 2048.0]], "f32", [[math.log2(1.0), math.log2(2.0), math.log2(4.0)], [math.log2(8.0), math.log2(16.0), math.log2(32.0)], [math.log2(64.0), math.log2(128.0), math.log2(256.0)], [math.log2(512.0), math.log2(1024.0), math.log2(2048.0)]]),
        (ptx_exec_multiblock_dense_log_2d_kernel, [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0], [512.0, 1024.0, 2048.0]], "f32", [[math.log(1.0), math.log(2.0), math.log(4.0)], [math.log(8.0), math.log(16.0), math.log(32.0)], [math.log(64.0), math.log(128.0), math.log(256.0)], [math.log(512.0), math.log(1024.0), math.log(2048.0)]]),
        (ptx_exec_multiblock_dense_log10_2d_kernel, [[1.0, 10.0, 100.0], [1000.0, 1.0, 10.0], [100.0, 1000.0, 1.0], [10.0, 100.0, 1000.0]], "f32", [[math.log10(1.0), math.log10(10.0), math.log10(100.0)], [math.log10(1000.0), math.log10(1.0), math.log10(10.0)], [math.log10(100.0), math.log10(1000.0), math.log10(1.0)], [math.log10(10.0), math.log10(100.0), math.log10(1000.0)]]),
        (ptx_exec_multiblock_dense_round_2d_kernel, [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]], "f32", [[-4.0, -2.0, -2.0], [0.0, 0.0, 2.0], [2.0, 4.0, -1.0], [1.0, -1.0, 1.0]]),
        (ptx_exec_multiblock_dense_floor_2d_kernel, [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]], "f32", [[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, -2.0], [0.0, -1.0, 1.0]]),
        (ptx_exec_multiblock_dense_ceil_2d_kernel, [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]], "f32", [[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0], [3.0, 4.0, -1.0], [1.0, 0.0, 2.0]]),
        (ptx_exec_multiblock_dense_trunc_2d_kernel, [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]], "f32", [[-3.0, -2.0, -1.0], [0.0, 0.0, 1.0], [2.0, 3.0, -1.0], [0.0, 0.0, 1.0]]),
        (ptx_exec_multiblock_dense_atan_2d_kernel, [[-1.5, -0.75, 0.0], [0.75, 1.0, 1.25], [-1.25, -0.5, 0.5], [1.5, 0.25, -0.25]], "f32", [[math.atan(-1.5), math.atan(-0.75), math.atan(0.0)], [math.atan(0.75), math.atan(1.0), math.atan(1.25)], [math.atan(-1.25), math.atan(-0.5), math.atan(0.5)], [math.atan(1.5), math.atan(0.25), math.atan(-0.25)]]),
        (ptx_exec_multiblock_dense_asin_2d_kernel, [[-0.875, -0.625, -0.375], [-0.125, 0.0, 0.125], [0.375, 0.625, 0.875], [-0.75, 0.5, -0.25]], "f32", [[math.asin(-0.875), math.asin(-0.625), math.asin(-0.375)], [math.asin(-0.125), math.asin(0.0), math.asin(0.125)], [math.asin(0.375), math.asin(0.625), math.asin(0.875)], [math.asin(-0.75), math.asin(0.5), math.asin(-0.25)]]),
        (ptx_exec_multiblock_dense_acos_2d_kernel, [[-0.875, -0.625, -0.375], [-0.125, 0.0, 0.125], [0.375, 0.625, 0.875], [-0.75, 0.5, -0.25]], "f32", [[math.acos(-0.875), math.acos(-0.625), math.acos(-0.375)], [math.acos(-0.125), math.acos(0.0), math.acos(0.125)], [math.acos(0.375), math.acos(0.625), math.acos(0.875)], [math.acos(-0.75), math.acos(0.5), math.acos(-0.25)]]),
        (ptx_exec_multiblock_dense_erf_2d_kernel, [[-1.5, -0.75, 0.0], [0.75, 1.0, 1.25], [-1.25, -0.5, 0.5], [1.5, 0.25, -0.25]], "f32", [[math.erf(-1.5), math.erf(-0.75), math.erf(0.0)], [math.erf(0.75), math.erf(1.0), math.erf(1.25)], [math.erf(-1.25), math.erf(-0.5), math.erf(0.5)], [math.erf(1.5), math.erf(0.25), math.erf(-0.25)]]),
        (ptx_exec_multiblock_dense_neg_f32_2d_kernel, [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], "f32", [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]]),
        (ptx_exec_multiblock_dense_neg_i32_2d_kernel, [[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, -12]], "i32", [[-1, 2, -3], [4, -5, 6], [-7, 8, -9], [10, -11, 12]]),
        (ptx_exec_multiblock_dense_abs_f32_2d_kernel, [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], "f32", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        (ptx_exec_multiblock_dense_abs_i32_2d_kernel, [[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, -12]], "i32", [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        (ptx_exec_multiblock_dense_bitnot_i32_2d_kernel, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], "i32", [[~1, ~2, ~3], [~4, ~5, ~6], [~7, ~8, ~9], [~10, ~11, ~12]]),
        (ptx_exec_multiblock_dense_bitnot_i1_2d_kernel, [[True, False, True], [False, True, False], [True, True, False], [False, False, True]], "i1", [[False, True, False], [True, False, True], [False, False, True], [True, True, False]]),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_unary_2d_if_cuda_available(tmp_path: Path, kernel, src, dtype: str, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    src = bb.tensor(src, dtype=dtype)
    dst = bb.zeros((4, 3), dtype=dtype)
    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    artifact(src, dst)
    if kernel in {
        ptx_exec_multiblock_dense_sin_2d_kernel,
        ptx_exec_multiblock_dense_cos_2d_kernel,
        ptx_exec_multiblock_dense_exp_2d_kernel,
        ptx_exec_multiblock_dense_exp2_2d_kernel,
        ptx_exec_multiblock_dense_atan_2d_kernel,
        ptx_exec_multiblock_dense_asin_2d_kernel,
        ptx_exec_multiblock_dense_acos_2d_kernel,
        ptx_exec_multiblock_dense_erf_2d_kernel,
        ptx_exec_multiblock_dense_log_2d_kernel,
        ptx_exec_multiblock_dense_log2_2d_kernel,
        ptx_exec_multiblock_dense_log10_2d_kernel,
    }:
        for actual_row, expected_row in zip(dst.tolist(), expected):
            tolerance = 1.2e-2 if kernel in {ptx_exec_multiblock_dense_asin_2d_kernel, ptx_exec_multiblock_dense_acos_2d_kernel} else (8e-3 if kernel in {ptx_exec_multiblock_dense_atan_2d_kernel, ptx_exec_multiblock_dense_erf_2d_kernel} else 2e-3)
            assert actual_row == pytest.approx(expected_row, rel=tolerance, abs=tolerance)
    elif dtype == "f32":
        _assert_nested_tensor_values(dst.tolist(), expected)
    else:
        _assert_nested_tensor_values(dst.tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_multiblock_dense_select_f32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
            [[1.0, 20.0, 3.0], [40.0, 5.0, 60.0], [7.0, 8.0, 90.0], [100.0, 110.0, 12.0]],
        ),
        (
            ptx_exec_multiblock_broadcast_select_i32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1], [2], [4], [8]], dtype="i32"),
                bb.tensor([[1, 3, 5]], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            [[1, 3, 1], [1, 2, 5], [4, 4, 5], [1, 3, 8]],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_select_2d_if_cuda_available(tmp_path: Path, kernel, args, expected) -> None:
    _skip_if_cuda_driver_unavailable()

    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    dst = args[-1]
    if dst.dtype == "f32":
        _assert_nested_tensor_values(dst.tolist(), expected)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_multiblock_dense_scalar_select_f32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.Float32(8.5),
                bb.zeros((4, 3), dtype="f32"),
            ),
            [[1.0, 8.5, 3.0], [8.5, 5.0, 8.5], [7.0, 8.0, 8.5], [8.5, 8.5, 12.0]],
        ),
        (
            ptx_exec_multiblock_dense_tensor_scalar_select_i32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"),
                bb.tensor([8], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            [[8, 2, 8], [4, 8, 6], [8, 8, 9], [10, 11, 8]],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_scalar_select_2d_if_cuda_available(
    tmp_path: Path, kernel, args, expected
) -> None:
    _skip_if_cuda_driver_unavailable()

    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    dst = args[-1]
    if dst.dtype == "f32":
        _assert_nested_tensor_values(dst.tolist(), expected)
    else:
        assert dst.tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "dst_values", "expected"),
    [
        (ptx_exec_multiblock_dense_copy_reduce_add_2d_kernel, "f32", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]),
        (ptx_exec_multiblock_dense_copy_reduce_or_i32_2d_kernel, "i32", [[1, 2, 4], [8, 16, 32], [3, 5, 9], [6, 10, 12]], [[16, 32, 64], [16, 32, 64], [16, 32, 64], [16, 32, 64]], [[17, 34, 68], [24, 48, 96], [19, 37, 73], [22, 42, 76]]),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_copy_reduce_2d_if_cuda_available(
    tmp_path: Path, kernel, dtype: str, src_values, dst_values, expected
) -> None:
    _skip_if_cuda_driver_unavailable()
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.tensor(dst_values, dtype=dtype)
    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(src, dst)
    _assert_nested_tensor_values(dst.tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "dtype", "expected_full"),
    [
        (ptx_exec_multiblock_tensor_factory_bundle_kernel, "f32", 7.0),
        (ptx_exec_multiblock_tensor_factory_bundle_i32_kernel, "i32", 7),
    ],
)
def test_ptx_exec_runs_multiblock_parallel_tensor_factory_bundle_if_cuda_available(
    tmp_path: Path, kernel, dtype: str, expected_full
) -> None:
    _skip_if_cuda_driver_unavailable()
    dst_zero = bb.zeros((4, 3), dtype=dtype)
    dst_one = bb.zeros((4, 3), dtype=dtype)
    dst_full = bb.zeros((4, 3), dtype=dtype)
    artifact = bb.compile(kernel, dst_zero, dst_one, dst_full, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(dst_zero, dst_one, dst_full)
    if dtype == "f32":
        _assert_nested_tensor_values(dst_zero.tolist(), [[0.0, 0.0, 0.0]] * 4)
        _assert_nested_tensor_values(dst_one.tolist(), [[1.0, 1.0, 1.0]] * 4)
        _assert_nested_tensor_values(dst_full.tolist(), [[expected_full, expected_full, expected_full]] * 4)
    else:
        assert dst_zero.tolist() == [[0, 0, 0]] * 4
        assert dst_one.tolist() == [[1, 1, 1]] * 4
        assert dst_full.tolist() == [[expected_full, expected_full, expected_full]] * 4


def test_compile_auto_prefers_ptx_exec_for_nvidia_target_if_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.from_dlpack(_FakeCudaInteropTensor(1234, (128,), "torch.float32", (1,)))
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (128,), "torch.float32", (1,)))

    artifact = bb.compile(
        ptx_exec_direct_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_target(),
    )

    assert artifact.backend_name == "ptx_exec"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "ptx"


def test_compile_auto_prefers_ptx_ref_when_all_tensor_args_are_staged_runtime_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.tensor([1.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    with pytest.warns(RuntimeWarning, match="auto-selected ptx_ref.*RuntimeTensor.*backend='ptx_exec'"):
        artifact = bb.compile(
            ptx_exec_direct_copy_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_target(),
        )

    assert artifact.backend_name == "ptx_ref"


def test_compile_auto_warns_when_ptx_exec_will_stage_mixed_tensor_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.tensor([1.0] * 128, dtype="f32")
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (128,), "torch.float32", (1,)))

    with pytest.warns(RuntimeWarning, match="auto-selected ptx_exec.*staged RuntimeTensor tensor arguments"):
        artifact = bb.compile(
            ptx_exec_direct_copy_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_target(),
        )

    assert artifact.backend_name == "ptx_exec"


def test_compile_auto_prefers_ptx_ref_for_mixed_staged_copy_reduce_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.tensor([1.0] * 128, dtype="f32")
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (128,), "torch.float32", (1,)))

    with pytest.warns(RuntimeWarning, match="auto-selected ptx_ref.*reduction-style kernels.*device-resident"):
        artifact = bb.compile(
            ptx_exec_indexed_copy_reduce_add_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_target(),
        )

    assert artifact.backend_name == "ptx_ref"


def test_compile_auto_prefers_ptx_exec_for_device_resident_copy_reduce_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.from_dlpack(_FakeCudaInteropTensor(1234, (128,), "torch.float32", (1,)))
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (128,), "torch.float32", (1,)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = bb.compile(
            ptx_exec_indexed_copy_reduce_add_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_target(),
        )

    assert artifact.backend_name == "ptx_exec"
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning) and "auto-selected ptx_ref" in str(warning.message)
    ]


def test_compile_auto_prefers_ptx_ref_for_mixed_staged_bitwise_reduce_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.tensor([1, 2, 3, 4], dtype="i32")
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (1,), "torch.int32", (1,)))

    with pytest.warns(RuntimeWarning, match="auto-selected ptx_ref.*reduction-style kernels.*device-resident"):
        artifact = bb.compile(
            ptx_exec_parallel_reduce_xor_i32_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_target(),
        )

    assert artifact.backend_name == "ptx_ref"


def test_compile_auto_prefers_ptx_exec_for_device_resident_bitwise_reduce_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.from_dlpack(_FakeCudaInteropTensor(1234, (4,), "torch.int32", (1,)))
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (1,), "torch.int32", (1,)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = bb.compile(
            ptx_exec_parallel_reduce_xor_i32_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_target(),
        )

    assert artifact.backend_name == "ptx_exec"
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning) and "auto-selected ptx_ref" in str(warning.message)
    ]


def test_compile_auto_does_not_warn_for_cuda_tensor_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.from_dlpack(_FakeCudaInteropTensor(1234, (128,), "torch.float32", (1,)))
    dst = bb.from_dlpack(_FakeCudaInteropTensor(2345, (128,), "torch.float32", (1,)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = bb.compile(
            ptx_exec_direct_copy_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_target(),
        )

    assert artifact.backend_name == "ptx_exec"
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning) and "auto-selected ptx_exec" in str(warning.message)
    ]


def test_ptx_exec_runs_on_cuda_tensor_handles_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    driver = CudaDriver()
    count = 128
    host_a = (ctypes.c_float * count)(*[float(index) for index in range(count)])
    host_b = (ctypes.c_float * count)(*[float(index * 2) for index in range(count)])
    host_c = (ctypes.c_float * count)()
    ptr_a = driver.mem_alloc(ctypes.sizeof(host_a))
    ptr_b = driver.mem_alloc(ctypes.sizeof(host_b))
    ptr_c = driver.mem_alloc(ctypes.sizeof(host_c))
    try:
        driver.memcpy_htod(ptr_a, ctypes.cast(host_a, ctypes.c_void_p), ctypes.sizeof(host_a))
        driver.memcpy_htod(ptr_b, ctypes.cast(host_b, ctypes.c_void_p), ctypes.sizeof(host_b))
        handle_a = bb.from_dlpack(_FakeCudaInteropTensor(int(ptr_a.value), (count,), "torch.float32", (1,)))
        handle_b = bb.from_dlpack(_FakeCudaInteropTensor(int(ptr_b.value), (count,), "torch.float32", (1,)))
        handle_c = bb.from_dlpack(_FakeCudaInteropTensor(int(ptr_c.value), (count,), "torch.float32", (1,)))
        artifact = bb.compile(
            ptx_exec_add_kernel,
            handle_a,
            handle_b,
            handle_c,
            cache_dir=tmp_path,
            target=_target(),
            backend="ptx_exec",
        )

        artifact(handle_a, handle_b, handle_c)
        driver.memcpy_dtoh(ctypes.cast(host_c, ctypes.c_void_p), ptr_c, ctypes.sizeof(host_c))
        assert list(host_c)[:8] == pytest.approx([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0])
    finally:
        driver.mem_free(ptr_a)
        driver.mem_free(ptr_b)
        driver.mem_free(ptr_c)
        driver.release_primary_context(0)


def test_ptx_exec_runs_on_raw_cuda_dlpack_objects_if_cuda_available(tmp_path: Path) -> None:
    _skip_if_cuda_driver_unavailable()

    driver = CudaDriver()
    count = 128
    host_a = (ctypes.c_float * count)(*[float(index) for index in range(count)])
    host_b = (ctypes.c_float * count)(*[float(index * 2) for index in range(count)])
    host_c = (ctypes.c_float * count)()
    ptr_a = driver.mem_alloc(ctypes.sizeof(host_a))
    ptr_b = driver.mem_alloc(ctypes.sizeof(host_b))
    ptr_c = driver.mem_alloc(ctypes.sizeof(host_c))
    try:
        driver.memcpy_htod(ptr_a, ctypes.cast(host_a, ctypes.c_void_p), ctypes.sizeof(host_a))
        driver.memcpy_htod(ptr_b, ctypes.cast(host_b, ctypes.c_void_p), ctypes.sizeof(host_b))
        tensor_a = _FakeCudaInteropTensor(int(ptr_a.value), (count,), "torch.float32", (1,))
        tensor_b = _FakeCudaInteropTensor(int(ptr_b.value), (count,), "torch.float32", (1,))
        tensor_c = _FakeCudaInteropTensor(int(ptr_c.value), (count,), "torch.float32", (1,))
        artifact = bb.compile(
            ptx_exec_add_kernel,
            tensor_a,
            tensor_b,
            tensor_c,
            cache_dir=tmp_path,
            target=_target(),
            backend="ptx_exec",
        )

        artifact(tensor_a, tensor_b, tensor_c)
        driver.memcpy_dtoh(ctypes.cast(host_c, ctypes.c_void_p), ptr_c, ctypes.sizeof(host_c))
        assert list(host_c)[:8] == pytest.approx([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0])
    finally:
        driver.mem_free(ptr_a)
        driver.mem_free(ptr_b)
        driver.mem_free(ptr_c)
        driver.release_primary_context(0)


def test_ptx_exec_rejects_non_cuda_tensor_handles(tmp_path: Path) -> None:
    artifact = bb.compile(
        ptx_exec_copy_kernel,
        bb.tensor([1.0] * 128, dtype="f32"),
        bb.zeros((128,), dtype="f32"),
        cache_dir=tmp_path,
        target=_target(),
        backend="ptx_exec",
    )

    with pytest.raises(bb.BackendNotImplementedError, match="requires CUDA TensorHandle inputs"):
        artifact(bb.from_dlpack(_FakeCpuInteropTensor()), bb.from_dlpack(_FakeCpuInteropTensor()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_max_f32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.maximum(a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_min_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.minimum(a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_exec_indexed_scalar_max_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.maximum(src[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_tensor_scalar_min_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.minimum(src[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_dense_max_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.maximum(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_exec_broadcast_min_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.minimum(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_exec_parallel_dense_scalar_max_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.maximum(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_exec_multiblock_dense_tensor_scalar_min_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.minimum(src.load(), alpha[0]))


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_indexed_max_f32_kernel,
            (
                bb.tensor([1.0, -2.5, 3.0], dtype="f32"),
                bb.tensor([0.5, -3.0, 4.0], dtype="f32"),
                bb.zeros((3,), dtype="f32"),
            ),
            [1.0, -2.5, 4.0],
        ),
        (
            ptx_exec_direct_min_i32_kernel,
            (
                bb.tensor([7, 11, 13, 17], dtype="i32"),
                bb.tensor([3, 15, 9, 21], dtype="i32"),
                bb.zeros((4,), dtype="i32"),
            ),
            [3, 11, 9, 17],
        ),
    ],
)
def test_ptx_exec_runs_extrema_rank1_binary_if_cuda_available(tmp_path: Path, kernel, args, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    assert args[-1].tolist() == pytest.approx(expected) if expected and isinstance(expected[0], float) else args[-1].tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_indexed_scalar_max_f32_kernel,
            (
                bb.tensor([1.0, -2.5, 3.0], dtype="f32"),
                bb.Float32(2.0),
                bb.zeros((3,), dtype="f32"),
            ),
            [2.0, 2.0, 3.0],
        ),
        (
            ptx_exec_direct_tensor_scalar_min_i32_kernel,
            (
                bb.tensor([7, 11, 13, 17], dtype="i32"),
                bb.tensor([10], dtype="i32"),
                bb.zeros((4,), dtype="i32"),
            ),
            [7, 10, 10, 10],
        ),
    ],
)
def test_ptx_exec_runs_extrema_rank1_scalar_broadcast_if_cuda_available(tmp_path: Path, kernel, args, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    assert args[-1].tolist() == pytest.approx(expected) if expected and isinstance(expected[0], float) else args[-1].tolist() == expected


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_dense_max_f32_2d_kernel,
            (
                bb.tensor([[1.0, -2.5], [7.0, 4.25]], dtype="f32"),
                bb.tensor([[0.5, -3.0], [8.0, 2.0]], dtype="f32"),
                bb.zeros((2, 2), dtype="f32"),
            ),
            [[1.0, -2.5], [8.0, 4.25]],
        ),
        (
            ptx_exec_broadcast_min_i32_2d_kernel,
            (
                bb.tensor([[7], [13]], dtype="i32"),
                bb.tensor([[3, 15, 9]], dtype="i32"),
                bb.zeros((2, 3), dtype="i32"),
            ),
            [[3, 7, 7], [3, 13, 9]],
        ),
    ],
)
def test_ptx_exec_runs_extrema_tensor_binary_2d_if_cuda_available(tmp_path: Path, kernel, args, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    _assert_nested_tensor_values(args[-1].tolist(), expected)


@pytest.mark.parametrize(
    ("kernel", "args", "expected"),
    [
        (
            ptx_exec_parallel_dense_scalar_max_f32_2d_kernel,
            (
                bb.tensor([[1.0, -2.5, 3.0], [0.5, 7.0, 4.25]], dtype="f32"),
                bb.Float32(2.0),
                bb.zeros((2, 3), dtype="f32"),
            ),
            [[2.0, 2.0, 3.0], [2.0, 7.0, 4.25]],
        ),
        (
            ptx_exec_multiblock_dense_tensor_scalar_min_i32_2d_kernel,
            (
                bb.tensor([[7, 11, 13], [17, 19, 23], [29, 31, 37], [41, 43, 47]], dtype="i32"),
                bb.tensor([24], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            [[7, 11, 13], [17, 19, 23], [24, 24, 24], [24, 24, 24]],
        ),
    ],
)
def test_ptx_exec_runs_extrema_tensor_scalar_broadcast_2d_if_cuda_available(tmp_path: Path, kernel, args, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    _assert_nested_tensor_values(args[-1].tolist(), expected)


def _atan2_vector_values_exec() -> tuple[list[float], list[float]]:
    y_values = [0.5 * float((index % 16) - 8) for index in range(128)]
    x_pattern = [1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5, 0.25]
    x_values = [x_pattern[index % len(x_pattern)] for index in range(128)]
    return y_values, x_values


@pytest.mark.parametrize(
    ("kernel", "args_fn", "expected_fn"),
    [
        (
            ptx_exec_indexed_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values_exec()[0], dtype="f32"),
                bb.tensor(_atan2_vector_values_exec()[1], dtype="f32"),
                bb.zeros((128,), dtype="f32"),
            ),
            lambda lhs, rhs: [math.atan2(y, x) for y, x in zip(lhs.tolist(), rhs.tolist())],
        ),
        (
            ptx_exec_direct_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values_exec()[0], dtype="f32"),
                bb.tensor(_atan2_vector_values_exec()[1], dtype="f32"),
                bb.zeros((128,), dtype="f32"),
            ),
            lambda lhs, rhs: [math.atan2(y, x) for y, x in zip(lhs.tolist(), rhs.tolist())],
        ),
    ],
)
def test_ptx_exec_runs_rank1_atan2_binary_if_cuda_available(tmp_path: Path, kernel, args_fn, expected_fn) -> None:
    _skip_if_cuda_driver_unavailable()
    args = args_fn()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    expected = expected_fn(args[0], args[1])
    assert args[-1].tolist() == pytest.approx(expected, rel=2e-2, abs=2e-2)


@pytest.mark.parametrize(
    ("kernel", "args_fn", "expected_fn"),
    [
        (
            ptx_exec_indexed_scalar_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values_exec()[0], dtype="f32"),
                bb.Float32(-1.25),
                bb.zeros((128,), dtype="f32"),
            ),
            lambda src, alpha: [math.atan2(value, float(alpha)) for value in src.tolist()],
        ),
        (
            ptx_exec_direct_tensor_scalar_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values_exec()[0], dtype="f32"),
                bb.tensor([-1.25], dtype="f32"),
                bb.zeros((128,), dtype="f32"),
            ),
            lambda src, alpha: [math.atan2(value, alpha.tolist()[0]) for value in src.tolist()],
        ),
    ],
)
def test_ptx_exec_runs_rank1_atan2_scalar_broadcast_if_cuda_available(
    tmp_path: Path, kernel, args_fn, expected_fn
) -> None:
    _skip_if_cuda_driver_unavailable()
    args = args_fn()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    expected = expected_fn(args[0], args[1])
    assert args[-1].tolist() == pytest.approx(expected, rel=2e-2, abs=2e-2)


@pytest.mark.parametrize(
    ("kernel", "args_fn", "expected"),
    [
        (
            ptx_exec_dense_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5]], dtype="f32"),
                bb.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.25]], dtype="f32"),
                bb.zeros((2, 3), dtype="f32"),
            ),
            [[math.atan2(1.0, 1.0), math.atan2(-1.0, 1.0), math.atan2(0.0, 0.0)], [math.atan2(2.0, -1.0), math.atan2(-2.0, -1.0), math.atan2(0.5, 0.25)]],
        ),
        (
            ptx_exec_broadcast_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0], [-1.0]], dtype="f32"),
                bb.tensor([[1.0, -1.0, 0.0]], dtype="f32"),
                bb.zeros((2, 3), dtype="f32"),
            ),
            [[math.atan2(1.0, 1.0), math.atan2(1.0, -1.0), math.atan2(1.0, 0.0)], [math.atan2(-1.0, 1.0), math.atan2(-1.0, -1.0), math.atan2(-1.0, 0.0)]],
        ),
    ],
)
def test_ptx_exec_runs_serial_tensor_atan2_2d_if_cuda_available(tmp_path: Path, kernel, args_fn, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    args = args_fn()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    for actual_row, expected_row in zip(args[-1].tolist(), expected):
        assert actual_row == pytest.approx(expected_row, rel=2e-2, abs=2e-2)


@pytest.mark.parametrize(
    ("kernel", "args_fn", "expected"),
    [
        (
            ptx_exec_dense_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5]], dtype="f32"),
                bb.Float32(-1.25),
                bb.zeros((2, 3), dtype="f32"),
            ),
            [[math.atan2(1.0, -1.25), math.atan2(-1.0, -1.25), math.atan2(0.0, -1.25)], [math.atan2(2.0, -1.25), math.atan2(-2.0, -1.25), math.atan2(0.5, -1.25)]],
        ),
        (
            ptx_exec_dense_tensor_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5]], dtype="f32"),
                bb.tensor([-1.25], dtype="f32"),
                bb.zeros((2, 3), dtype="f32"),
            ),
            [[math.atan2(1.0, -1.25), math.atan2(-1.0, -1.25), math.atan2(0.0, -1.25)], [math.atan2(2.0, -1.25), math.atan2(-2.0, -1.25), math.atan2(0.5, -1.25)]],
        ),
    ],
)
def test_ptx_exec_runs_serial_tensor_scalar_atan2_2d_if_cuda_available(tmp_path: Path, kernel, args_fn, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    args = args_fn()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    for actual_row, expected_row in zip(args[-1].tolist(), expected):
        assert actual_row == pytest.approx(expected_row, rel=2e-2, abs=2e-2)


@pytest.mark.parametrize(
    ("kernel", "args_fn", "expected"),
    [
        (
            ptx_exec_multiblock_dense_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5], [-0.5, 0.75, -1.25], [1.5, -1.5, 2.0]], dtype="f32"),
                bb.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.25], [0.5, -0.75, 1.25], [2.0, -2.0, -0.5]], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
            [
                [math.atan2(1.0, 1.0), math.atan2(-1.0, 1.0), math.atan2(0.0, 0.0)],
                [math.atan2(2.0, -1.0), math.atan2(-2.0, -1.0), math.atan2(0.5, 0.25)],
                [math.atan2(-0.5, 0.5), math.atan2(0.75, -0.75), math.atan2(-1.25, 1.25)],
                [math.atan2(1.5, 2.0), math.atan2(-1.5, -2.0), math.atan2(2.0, -0.5)],
            ],
        ),
        (
            ptx_exec_multiblock_broadcast_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0], [-1.0], [0.5], [-0.5]], dtype="f32"),
                bb.tensor([[1.0, -1.0, 0.0]], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
            [
                [math.atan2(1.0, 1.0), math.atan2(1.0, -1.0), math.atan2(1.0, 0.0)],
                [math.atan2(-1.0, 1.0), math.atan2(-1.0, -1.0), math.atan2(-1.0, 0.0)],
                [math.atan2(0.5, 1.0), math.atan2(0.5, -1.0), math.atan2(0.5, 0.0)],
                [math.atan2(-0.5, 1.0), math.atan2(-0.5, -1.0), math.atan2(-0.5, 0.0)],
            ],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_tensor_atan2_2d_if_cuda_available(tmp_path: Path, kernel, args_fn, expected) -> None:
    _skip_if_cuda_driver_unavailable()
    args = args_fn()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    for actual_row, expected_row in zip(args[-1].tolist(), expected):
        assert actual_row == pytest.approx(expected_row, rel=2e-2, abs=2e-2)


@pytest.mark.parametrize(
    ("kernel", "args_fn", "expected"),
    [
        (
            ptx_exec_multiblock_dense_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5], [-0.5, 0.75, -1.25], [1.5, -1.5, 2.0]], dtype="f32"),
                bb.Float32(-1.25),
                bb.zeros((4, 3), dtype="f32"),
            ),
            [
                [math.atan2(1.0, -1.25), math.atan2(-1.0, -1.25), math.atan2(0.0, -1.25)],
                [math.atan2(2.0, -1.25), math.atan2(-2.0, -1.25), math.atan2(0.5, -1.25)],
                [math.atan2(-0.5, -1.25), math.atan2(0.75, -1.25), math.atan2(-1.25, -1.25)],
                [math.atan2(1.5, -1.25), math.atan2(-1.5, -1.25), math.atan2(2.0, -1.25)],
            ],
        ),
        (
            ptx_exec_multiblock_dense_tensor_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5], [-0.5, 0.75, -1.25], [1.5, -1.5, 2.0]], dtype="f32"),
                bb.tensor([-1.25], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
            [
                [math.atan2(1.0, -1.25), math.atan2(-1.0, -1.25), math.atan2(0.0, -1.25)],
                [math.atan2(2.0, -1.25), math.atan2(-2.0, -1.25), math.atan2(0.5, -1.25)],
                [math.atan2(-0.5, -1.25), math.atan2(0.75, -1.25), math.atan2(-1.25, -1.25)],
                [math.atan2(1.5, -1.25), math.atan2(-1.5, -1.25), math.atan2(2.0, -1.25)],
            ],
        ),
    ],
)
def test_ptx_exec_runs_multiblock_tensor_scalar_atan2_2d_if_cuda_available(
    tmp_path: Path, kernel, args_fn, expected
) -> None:
    _skip_if_cuda_driver_unavailable()
    args = args_fn()
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_target(), backend="ptx_exec")
    artifact(*args)
    for actual_row, expected_row in zip(args[-1].tolist(), expected):
        assert actual_row == pytest.approx(expected_row, rel=2e-2, abs=2e-2)
