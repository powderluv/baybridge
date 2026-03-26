from __future__ import annotations

from pathlib import Path

import pytest

import baybridge as bb
from baybridge.cuda_driver import CudaDriver, CudaDriverError
from baybridge.nvgpu import cpasync, tcgen05


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(64, 1, 1)))
def ptx_indexed_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(64, 1, 1)))
def ptx_indexed_add_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] + b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_mul_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] * b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_bitand_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] & b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_bitor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] | b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_bitxor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] ^ b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_bitand_i1_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] & b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_cmp_lt_f32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] < b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_select_f32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_select_i32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_select_i1_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_select_scalar_f32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Float32, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_select_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Boolean, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], a[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_select_tensor_scalar_i32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], alpha[0], a[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_select_tensor_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.where(pred[idx], alpha[0], a[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_scalar_cmp_lt_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] < alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_tensor_scalar_cmp_eq_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] == alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_scalar_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] | alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] ^ alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_tensor_scalar_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] & alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_tensor_scalar_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] | alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_tensor_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] ^ alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_copy_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_copy_reduce_max_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_copy_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_copy_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_copy_reduce_max_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_copy_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_add_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] + b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_mul_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] * b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_bitand_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] & b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_bitor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] | b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_bitor_i1_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] | b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_bitxor_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] ^ b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_cmp_eq_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] == b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_select_f32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_select_i32_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_select_i1_kernel(pred: bb.Tensor, a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_select_scalar_f32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Float32, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], alpha, a[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_select_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Boolean, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], alpha, a[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_select_tensor_scalar_i32_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_select_tensor_scalar_i1_kernel(pred: bb.Tensor, a: bb.Tensor, alpha: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.where(pred[tidx], a[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_scalar_cmp_lt_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] < alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_tensor_scalar_cmp_eq_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] == alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_scalar_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] | alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] ^ alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_scalar_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] & alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_tensor_scalar_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] & alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_tensor_scalar_bitxor_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] ^ alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_cmp_lt_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() < rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_cmp_lt_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() < alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_cmp_eq_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() == alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_select_f32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_select_f32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_select_i32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), alpha[0], src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_sub_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_mul_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_div_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_sub_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_mul_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_div_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_bitand_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_copy_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_copy_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_copy_reduce_add_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_copy_reduce_max_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_copy_reduce_or_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_cmp_eq_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() == rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_select_i32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_sub_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_mul_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_div_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_sub_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() - rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_mul_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() * rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_div_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() / rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_bitor_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_sqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_rsqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.rsqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_sin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_cos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.cos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_exp2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_exp_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_log2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_log_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_log10_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log10(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_round_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_floor_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.floor(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_ceil_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.ceil(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_trunc_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.trunc(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_atan_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_asin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_acos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_erf_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_neg_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_neg_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_abs_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_abs_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_bitnot_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_bitnot_i1_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha[0]))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_sub_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_mul_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_div_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_sub_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() - alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_mul_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() * alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_div_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() / alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() | alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() ^ alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() & alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() ^ alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_sub_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_mul_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_tensor_scalar_div_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_sub_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() - alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_mul_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() * alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_scalar_div_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() / alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_cmp_lt_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() < rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_cmp_lt_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() < alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_bitxor_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_copy_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_copy_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_copy_reduce_add_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_copy_reduce_max_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_copy_reduce_or_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_bitand_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_sqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_sin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_cos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.cos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_exp2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_exp_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_log2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_log_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_log10_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log10(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_round_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_floor_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.floor(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_ceil_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.ceil(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_trunc_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.trunc(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_atan_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_asin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_acos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_erf_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_neg_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_neg_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_abs_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_abs_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_bitnot_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_bitnot_i1_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_sub_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_mul_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_div_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() | alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() ^ alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_add_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() & alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_bitxor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() ^ alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_sub_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_mul_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_div_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() / alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_cmp_eq_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() == alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_select_f32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_scalar_select_f32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_dense_tensor_scalar_select_i32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), alpha[0], src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_cmp_eq_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() == rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_broadcast_select_i32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_cmp_lt_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() < rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_scalar_cmp_lt_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() < alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_add_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_copy_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_copy_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_broadcast_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_broadcast_bitor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_broadcast_cmp_eq_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() == rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_bitxor_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() ^ rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_broadcast_bitand_i1_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_scalar_bitand_i32_2d_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_tensor_scalar_add_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_tensor_scalar_atan2_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(src.load(), alpha[0]))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_tensor_scalar_bitor_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_tensor_scalar_cmp_eq_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() == alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_select_f32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_scalar_select_f32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_tensor_scalar_select_i32_2d_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), alpha[0], src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_broadcast_select_i32_2d_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_sqrt_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_sin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_cos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.cos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_exp2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_exp_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_log2_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_log_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_log10_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log10(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_round_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_floor_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.floor(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_ceil_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.ceil(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_trunc_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.trunc(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_atan_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_asin_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_acos_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_broadcast_atan2_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_erf_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_neg_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_neg_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(-src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_abs_f32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_abs_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_bitnot_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_bitnot_i1_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_copy_reduce_add_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_copy_reduce_or_i32_2d_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_tensor_factory_bundle_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_tensor_factory_bundle_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_scalar_param_broadcast_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_tensor_scalar_broadcast_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_atan2_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.math.atan2(lhs[tidx], rhs[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_tensor_scalar_atan2_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.math.atan2(src[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_scalar_param_broadcast_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_tensor_scalar_broadcast_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_atan2_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.atan2(lhs[idx], rhs[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_scalar_atan2_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.atan2(src[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_sqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.sqrt(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_rsqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.rsqrt(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_sin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.sin(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_cos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.cos(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_exp2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.exp2(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_exp_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.exp(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_log2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.log2(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_log_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.log(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_log10_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.log10(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_round_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.round(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_floor_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.floor(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_ceil_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.ceil(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_trunc_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.trunc(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_atan_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.atan(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_asin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.asin(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_acos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.acos(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_erf_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.erf(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_neg_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = -src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_neg_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = -src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_abs_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = abs(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_abs_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = abs(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_bitnot_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = ~src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_bitnot_i1_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = ~src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_sqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.sqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_rsqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.rsqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_sin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.sin(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_cos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.cos(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_exp2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.exp2(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_exp_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.exp(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_log2_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.log2(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_log_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.log(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_log10_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.log10(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_round_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.round(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_floor_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.floor(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_ceil_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.ceil(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_trunc_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.trunc(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_atan_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.atan(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_asin_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.asin(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_acos_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.acos(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_erf_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.math.erf(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_neg_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = -src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_neg_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = -src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_abs_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = abs(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_abs_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = abs(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_bitnot_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = ~src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_bitnot_i1_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = ~src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_add_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_add_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_mul_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_max_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_min_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_mul_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MUL, 1, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_max_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_min_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MIN, 999, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_and_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.AND, -1, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def ptx_parallel_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.XOR, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_add_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_add_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_add_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_add_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_mul_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_mul_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_max_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_max_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_min_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_min_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_mul_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_mul_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_max_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_max_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_min_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_min_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_and_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_and_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_or_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_or_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_rows_xor_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_cols_xor_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_add_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_add_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_add_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_add_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_mul_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_mul_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_max_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_max_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_min_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_min_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_mul_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_mul_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_max_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_max_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_min_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_min_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_and_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_and_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_or_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_or_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_rows_xor_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_cols_xor_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_multiblock_reduce_rows_add_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_multiblock_reduce_cols_add_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_multiblock_reduce_rows_or_i32_2d_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_multiblock_reduce_cols_or_i32_2d_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_add_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_add_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_mul_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_max_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_min_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_mul_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_max_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_min_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_and_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_or_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_xor_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_tensor_factory_bundle_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_tensor_factory_bundle_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_tensor_factory_bundle_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def ptx_parallel_tensor_factory_bundle_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_max_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_add_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_mul_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_max_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_min_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_add_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_mul_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_max_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_min_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_and_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_or_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_xor_i32_2d_bundle_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_add_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_reduce_or_i32_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_add_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_reduce_or_i32_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_multiblock_reduce_add_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(2, 1, 1)))
def ptx_multiblock_reduce_or_i32_rowcol_2d_bundle_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_unsupported_math_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.acos(src[tidx])


def _nvidia_target() -> bb.NvidiaTarget:
    return bb.NvidiaTarget(sm="sm_80", ptx_version="8.0")


def test_ptx_ref_lowers_indexed_copy_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 256, dtype="f32")
    dst = bb.zeros((256,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    assert artifact.backend_name == "ptx_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "ptx"
    assert artifact.lowered_path is not None
    assert artifact.lowered_path.suffix == ".ptx"
    text = artifact.lowered_module.text
    assert ".version 8.0" in text
    assert ".target sm_80" in text
    assert ".visible .entry ptx_indexed_copy_kernel(" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "st.global.f32 [%rd5], %f1;" in text


def test_ptx_ref_lowers_indexed_copy_i1_kernel(tmp_path: Path) -> None:
    src = bb.tensor([True, False] * 64, dtype="i1")
    dst = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_indexed_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_copy_kernel(" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_indexed_add_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1.0] * 256, dtype="f32")
    b = bb.tensor([2.0] * 256, dtype="f32")
    c = bb.zeros((256,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_add_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd6];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd7], %f3;" in text


def test_ptx_ref_lowers_indexed_mul_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1] * 128, dtype="i32")
    b = bb.tensor([2] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_mul_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_mul_i32_kernel(" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "mul.lo.s32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_indexed_bitand_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([7] * 128, dtype="i32")
    b = bb.tensor([3] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_bitand_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_bitand_i32_kernel(" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "and.b32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_indexed_bitor_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([7] * 128, dtype="i32")
    b = bb.tensor([3] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_bitor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_bitor_i32_kernel(" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "or.b32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_indexed_bitxor_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([7] * 128, dtype="i32")
    b = bb.tensor([3] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_bitxor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_bitxor_i32_kernel(" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "xor.b32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_indexed_bitand_i1_kernel(tmp_path: Path) -> None:
    a = bb.tensor([True, False] * 64, dtype="i1")
    b = bb.tensor([True, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_indexed_bitand_i1_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_bitand_i1_kernel(" in text
    assert "ld.global.u8 %r5, [%rd5];" in text
    assert "ld.global.u8 %r2, [%rd6];" in text
    assert "and.b32 %r5, %r5, %r2;" in text
    assert "st.global.u8 [%rd7], %r5;" in text


def test_ptx_ref_lowers_indexed_cmp_lt_f32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(index + 1) for index in range(128)], dtype="f32")
    c = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_indexed_cmp_lt_f32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_cmp_lt_f32_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd6];" in text
    assert "setp.lt.f32 %p2, %f1, %f2;" in text
    assert "st.global.u8 [%rd7], %r5;" in text


def test_ptx_ref_lowers_indexed_select_f32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(1000 + index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_select_f32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_select_f32_kernel(" in text
    assert "ld.global.u8 %r5, [%rd6];" in text
    assert "setp.ne.u32 %p2, %r5, 0;" in text
    assert "selp.f32 %f3, %f1, %f2, %p2;" in text
    assert "st.global.f32 [%rd10], %f3;" in text


def test_ptx_ref_lowers_indexed_select_i32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    b = bb.tensor([1000 + index for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_select_i32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_select_i32_kernel(" in text
    assert "ld.global.u8 %r5, [%rd6];" in text
    assert "setp.ne.u32 %p2, %r5, 0;" in text
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.s32 [%rd10], %r6;" in text


def test_ptx_ref_lowers_indexed_select_i1_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([True, True] * 64, dtype="i1")
    b = bb.tensor([False, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_indexed_select_i1_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_select_i1_kernel(" in text
    assert "ld.global.u8 %r5, [%rd6];" in text
    assert "ld.global.u8 %r6, [%rd8];" in text
    assert "ld.global.u8 %r7, [%rd9];" in text
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.u8 [%rd10], %r6;" in text


def test_ptx_ref_lowers_indexed_select_scalar_f32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_select_scalar_f32_kernel,
        pred,
        a,
        bb.Float32(3.5),
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_select_scalar_f32_kernel(" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "ld.global.f32 %f1, [%rd6];" in text
    assert "selp.f32 %f3, %f1, %f2, %p2;" in text
    assert "st.global.f32 [%rd8], %f3;" in text


def test_ptx_ref_lowers_indexed_select_scalar_i1_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool((index + 1) % 3) for index in range(128)], dtype="i1")
    c = bb.zeros((128,), dtype="i1")

    artifact = bb.compile(
        ptx_indexed_select_scalar_i1_kernel,
        pred,
        a,
        bb.Boolean(True),
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_select_scalar_i1_kernel(" in text
    assert "ld.param.u8 %r7, [alpha_param];" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "ld.global.u8 %r6, [%rd6];" in text
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.u8 [%rd8], %r6;" in text


def test_ptx_ref_lowers_indexed_select_tensor_scalar_i32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([17], dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_select_tensor_scalar_i32_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_select_tensor_scalar_i32_kernel(" in text
    assert "ld.param.u64 %rd7, [alpha_param];" in text
    assert "ld.global.s32 %r7, [%rd7];" in text
    assert "selp.b32 %r6, %r7, %r6, %p2;" in text
    assert "st.global.s32 [%rd8], %r6;" in text


def test_ptx_ref_lowers_indexed_select_tensor_scalar_i1_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool(index % 3) for index in range(128)], dtype="i1")
    alpha = bb.tensor([True], dtype="i1")
    c = bb.zeros((128,), dtype="i1")

    artifact = bb.compile(
        ptx_indexed_select_tensor_scalar_i1_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_select_tensor_scalar_i1_kernel(" in text
    assert "ld.param.u64 %rd7, [alpha_param];" in text
    assert "ld.global.u8 %r7, [%rd7];" in text
    assert "selp.b32 %r6, %r7, %r6, %p2;" in text
    assert "st.global.u8 [%rd8], %r6;" in text


def test_ptx_ref_lowers_indexed_scalar_cmp_lt_f32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_indexed_scalar_cmp_lt_f32_kernel,
        src,
        bb.Float32(64.0),
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_scalar_cmp_lt_f32_kernel(" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "setp.lt.f32 %p2, %f1, %f2;" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_indexed_tensor_scalar_cmp_eq_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([17], dtype="i32")
    dst = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_indexed_tensor_scalar_cmp_eq_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_tensor_scalar_cmp_eq_i32_kernel(" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert "ld.global.s32 %r2, [%rd4];" in text
    assert "ld.global.s32 %r3, [%rd2];" in text
    assert "setp.eq.s32 %p2, %r2, %r3;" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_direct_copy_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_copy_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "st.global.f32 [%rd5], %f1;" in text


def test_ptx_ref_lowers_direct_copy_i1_kernel(tmp_path: Path) -> None:
    src = bb.tensor([True, False] * 64, dtype="i1")
    dst = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_direct_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_copy_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_direct_add_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1.0] * 128, dtype="f32")
    b = bb.tensor([2.0] * 128, dtype="f32")
    c = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_add_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd6];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd7], %f3;" in text


def test_ptx_ref_lowers_direct_mul_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1] * 128, dtype="i32")
    b = bb.tensor([2] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_mul_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_mul_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "mul.lo.s32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_direct_bitand_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([7] * 128, dtype="i32")
    b = bb.tensor([3] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_bitand_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_bitand_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "and.b32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_direct_bitor_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1] * 128, dtype="i32")
    b = bb.tensor([8] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_bitor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_bitor_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "or.b32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_direct_bitxor_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1] * 128, dtype="i32")
    b = bb.tensor([8] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_bitxor_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_bitxor_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "xor.b32 %r5, %r5, %r2;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_ptx_ref_lowers_direct_bitor_i1_kernel(tmp_path: Path) -> None:
    a = bb.tensor([True, False] * 64, dtype="i1")
    b = bb.tensor([False, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_direct_bitor_i1_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_bitor_i1_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.u8 %r5, [%rd5];" in text
    assert "ld.global.u8 %r2, [%rd6];" in text
    assert "or.b32 %r5, %r5, %r2;" in text
    assert "st.global.u8 [%rd7], %r5;" in text


def test_ptx_ref_lowers_direct_cmp_eq_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([index for index in range(128)], dtype="i32")
    b = bb.tensor([index for index in range(128)], dtype="i32")
    c = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_direct_cmp_eq_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_cmp_eq_i32_kernel(" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r2, [%rd6];" in text
    assert "setp.eq.s32 %p2, %r5, %r2;" in text
    assert "st.global.u8 [%rd7], %r5;" in text


def test_ptx_ref_lowers_direct_select_f32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    b = bb.tensor([float(1000 + index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_select_f32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_select_f32_kernel(" in text
    assert "ld.global.u8 %r5, [%rd6];" in text
    assert "setp.ne.u32 %p2, %r5, 0;" in text
    assert "selp.f32 %f3, %f1, %f2, %p2;" in text
    assert "st.global.f32 [%rd10], %f3;" in text


def test_ptx_ref_lowers_direct_select_i32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    b = bb.tensor([1000 + index for index in range(128)], dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_select_i32_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_select_i32_kernel(" in text
    assert "ld.global.u8 %r5, [%rd6];" in text
    assert "setp.ne.u32 %p2, %r5, 0;" in text
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.s32 [%rd10], %r6;" in text


def test_ptx_ref_lowers_direct_select_i1_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([True, True] * 64, dtype="i1")
    b = bb.tensor([False, True] * 64, dtype="i1")
    c = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_direct_select_i1_kernel,
        pred,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_select_i1_kernel(" in text
    assert "ld.global.u8 %r5, [%rd6];" in text
    assert "ld.global.u8 %r6, [%rd8];" in text
    assert "ld.global.u8 %r7, [%rd9];" in text
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.u8 [%rd10], %r6;" in text


def test_ptx_ref_lowers_direct_select_scalar_f32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([float(index) for index in range(128)], dtype="f32")
    c = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_select_scalar_f32_kernel,
        pred,
        a,
        bb.Float32(5.5),
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_select_scalar_f32_kernel(" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "ld.global.f32 %f1, [%rd6];" in text
    assert "selp.f32 %f3, %f2, %f1, %p2;" in text
    assert "st.global.f32 [%rd8], %f3;" in text


def test_ptx_ref_lowers_direct_select_scalar_i1_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool(index % 3) for index in range(128)], dtype="i1")
    c = bb.zeros((128,), dtype="i1")

    artifact = bb.compile(
        ptx_direct_select_scalar_i1_kernel,
        pred,
        a,
        bb.Boolean(True),
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_select_scalar_i1_kernel(" in text
    assert "ld.param.u8 %r7, [alpha_param];" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "ld.global.u8 %r6, [%rd6];" in text
    assert "selp.b32 %r6, %r7, %r6, %p2;" in text
    assert "st.global.u8 [%rd8], %r6;" in text


def test_ptx_ref_lowers_direct_select_tensor_scalar_i32_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([23], dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_select_tensor_scalar_i32_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_select_tensor_scalar_i32_kernel(" in text
    assert "ld.param.u64 %rd7, [alpha_param];" in text
    assert "ld.global.s32 %r7, [%rd7];" in text
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.s32 [%rd8], %r6;" in text


def test_ptx_ref_lowers_direct_select_tensor_scalar_i1_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    a = bb.tensor([bool((index + 1) % 3) for index in range(128)], dtype="i1")
    alpha = bb.tensor([False], dtype="i1")
    c = bb.zeros((128,), dtype="i1")

    artifact = bb.compile(
        ptx_direct_select_tensor_scalar_i1_kernel,
        pred,
        a,
        alpha,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_select_tensor_scalar_i1_kernel(" in text
    assert "ld.param.u64 %rd7, [alpha_param];" in text
    assert "ld.global.u8 %r7, [%rd7];" in text
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.u8 [%rd8], %r6;" in text


def test_ptx_ref_lowers_direct_scalar_cmp_lt_f32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_direct_scalar_cmp_lt_f32_kernel,
        src,
        bb.Float32(64.0),
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_scalar_cmp_lt_f32_kernel(" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "setp.lt.f32 %p2, %f1, %f2;" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_direct_tensor_scalar_cmp_eq_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index for index in range(128)], dtype="i32")
    alpha = bb.tensor([17], dtype="i32")
    dst = bb.tensor([False] * 128, dtype="i1")

    artifact = bb.compile(
        ptx_direct_tensor_scalar_cmp_eq_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_tensor_scalar_cmp_eq_i32_kernel(" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert "ld.global.s32 %r2, [%rd4];" in text
    assert "ld.global.s32 %r3, [%rd2];" in text
    assert "setp.eq.s32 %p2, %r2, %r3;" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_indexed_scalar_bitor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.Int32(24)
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_scalar_bitor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_scalar_bitor_i32_kernel(" in text
    assert "ld.param.s32" in text
    assert "or.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_indexed_scalar_bitxor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.Int32(24)
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_scalar_bitxor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_scalar_bitxor_i32_kernel(" in text
    assert "ld.param.s32" in text
    assert "xor.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_indexed_tensor_scalar_bitand_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_tensor_scalar_bitand_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_tensor_scalar_bitand_i32_kernel(" in text
    assert "ld.param.u64" in text
    assert "and.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_indexed_tensor_scalar_bitor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_tensor_scalar_bitor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_tensor_scalar_bitor_i32_kernel(" in text
    assert "ld.param.u64" in text
    assert "or.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_indexed_tensor_scalar_bitxor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_tensor_scalar_bitxor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_tensor_scalar_bitxor_i32_kernel(" in text
    assert "ld.param.u64" in text
    assert "xor.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_direct_scalar_bitor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.Int32(24)
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_scalar_bitor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_scalar_bitor_i32_kernel(" in text
    assert "ld.param.s32" in text
    assert "or.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_direct_scalar_bitxor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.Int32(24)
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_scalar_bitxor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_scalar_bitxor_i32_kernel(" in text
    assert "ld.param.s32" in text
    assert "xor.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_direct_scalar_bitand_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.Int32(11)
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_scalar_bitand_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_scalar_bitand_i32_kernel(" in text
    assert "ld.param.s32" in text
    assert "and.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_direct_tensor_scalar_bitand_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_tensor_scalar_bitand_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_tensor_scalar_bitand_i32_kernel(" in text
    assert "ld.param.u64" in text
    assert "ld.global.s32" in text
    assert "and.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_direct_tensor_scalar_bitxor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([7] * 128, dtype="i32")
    alpha = bb.tensor([11], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_tensor_scalar_bitxor_i32_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_tensor_scalar_bitxor_i32_kernel(" in text
    assert "ld.param.u64" in text
    assert "ld.global.s32" in text
    assert "xor.b32" in text
    assert "st.global.s32" in text


def test_ptx_ref_lowers_copy_reduce_add_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")

    artifact = bb.compile(
        ptx_copy_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_copy_reduce_add_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd4];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_copy_reduce_max_f32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.tensor([0.5, 2.5, 2.0, 8.0], dtype="f32")

    artifact = bb.compile(
        ptx_copy_reduce_max_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_copy_reduce_max_f32_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd4];" in text
    assert "max.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_copy_reduce_xor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1, 2, 3, 4], dtype="i32")
    dst = bb.tensor([8, 8, 8, 8], dtype="i32")

    artifact = bb.compile(
        ptx_copy_reduce_xor_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_copy_reduce_xor_i32_kernel(" in text
    assert "ld.global.s32 %r3, [%rd5];" in text
    assert "ld.global.s32 %r4, [%rd4];" in text
    assert "xor.b32 %r3, %r3, %r4;" in text
    assert "st.global.s32 [%rd5], %r3;" in text


def test_ptx_ref_lowers_copy_reduce_or_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1, 2, 4, 8], dtype="i32")
    dst = bb.tensor([16, 16, 16, 16], dtype="i32")

    artifact = bb.compile(
        ptx_copy_reduce_or_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_copy_reduce_or_i32_kernel(" in text
    assert "ld.global.s32 %r3, [%rd5];" in text
    assert "ld.global.s32 %r4, [%rd4];" in text
    assert "or.b32 %r3, %r3, %r4;" in text
    assert "st.global.s32 [%rd5], %r3;" in text


def test_ptx_ref_lowers_indexed_copy_reduce_add_kernel(tmp_path: Path) -> None:
    src = bb.tensor([float(index) for index in range(128)], dtype="f32")
    dst = bb.tensor([1.0 for _ in range(128)], dtype="f32")

    artifact = bb.compile(
        ptx_indexed_copy_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_copy_reduce_add_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "L_copy_reduce:" not in text


def test_ptx_ref_lowers_indexed_copy_reduce_max_f32_kernel(tmp_path: Path) -> None:
    artifact = bb.compile(
        ptx_indexed_copy_reduce_max_f32_kernel,
        bb.zeros((128,), dtype="f32"),
        bb.zeros((128,), dtype="f32"),
        backend="ptx_ref",
        target=_nvidia_target(),
        cache_dir=tmp_path,
    )
    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_copy_reduce_max_f32_kernel(" in text
    assert "max.f32" in text
    assert "L_copy_reduce:" not in text


def test_ptx_ref_lowers_indexed_copy_reduce_xor_i32_kernel(tmp_path: Path) -> None:
    artifact = bb.compile(
        ptx_indexed_copy_reduce_xor_i32_kernel,
        bb.zeros((128,), dtype="i32"),
        bb.zeros((128,), dtype="i32"),
        backend="ptx_ref",
        target=_nvidia_target(),
        cache_dir=tmp_path,
    )
    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_copy_reduce_xor_i32_kernel(" in text
    assert "xor.b32" in text
    assert "L_copy_reduce:" not in text


def test_ptx_ref_lowers_direct_copy_reduce_or_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index for index in range(128)], dtype="i32")
    dst = bb.tensor([3 for _ in range(128)], dtype="i32")

    artifact = bb.compile(
        ptx_direct_copy_reduce_or_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_copy_reduce_or_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "or.b32 %r3, %r3, %r4;" in text
    assert "L_copy_reduce:" not in text


def test_ptx_ref_lowers_direct_copy_reduce_xor_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index for index in range(128)], dtype="i32")
    dst = bb.tensor([3 for _ in range(128)], dtype="i32")

    artifact = bb.compile(
        ptx_direct_copy_reduce_xor_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_copy_reduce_xor_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "xor.b32 %r3, %r3, %r4;" in text
    assert "L_copy_reduce:" not in text


def test_ptx_ref_rejects_underfilled_direct_rank1_launch(tmp_path: Path) -> None:
    target = bb.NvidiaTarget(sm="sm_80", ptx_version="8.0")
    with pytest.raises(bb.CompilationError, match="ptx_ref currently supports only exact"):
        bb.compile(
            ptx_direct_copy_kernel,
            bb.zeros((129,), dtype="f32"),
            bb.zeros((129,), dtype="f32"),
            backend="ptx_ref",
            target=target,
            cache_dir=tmp_path,
        )


def test_ptx_ref_rejects_undercovered_indexed_rank1_launch(tmp_path: Path) -> None:
    target = bb.NvidiaTarget(sm="sm_80", ptx_version="8.0")
    with pytest.raises(bb.CompilationError, match="ptx_ref currently supports only exact"):
        bb.compile(
            ptx_indexed_copy_kernel,
            bb.zeros((257,), dtype="f32"),
            bb.zeros((257,), dtype="f32"),
            backend="ptx_ref",
            target=target,
            cache_dir=tmp_path,
        )


def test_ptx_ref_lowers_direct_scalar_param_broadcast_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 128, dtype="f32")
    alpha = bb.Float32(2.5)
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_scalar_param_broadcast_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_scalar_param_broadcast_kernel(" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_direct_tensor_scalar_broadcast_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 128, dtype="f32")
    alpha = bb.tensor([2.5], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_tensor_scalar_broadcast_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_tensor_scalar_broadcast_kernel(" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert "ld.global.f32 %f2, [%rd2];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_indexed_scalar_param_broadcast_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 128, dtype="f32")
    alpha = bb.Float32(2.5)
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_scalar_param_broadcast_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_scalar_param_broadcast_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_indexed_tensor_scalar_broadcast_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 128, dtype="f32")
    alpha = bb.tensor([2.5], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_tensor_scalar_broadcast_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_tensor_scalar_broadcast_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert "ld.global.f32 %f2, [%rd2];" in text
    assert "st.global.f32 [%rd5], %f3;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "dst_shape", "load_instr", "store_instr"),
    [
        (ptx_dense_copy_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], (2, 2), "ld.global.f32 %f1, [%rd4];", "st.global.f32 [%rd5], %f1;"),
        (ptx_dense_copy_i32_2d_kernel, "i32", [[1, 2], [3, 4]], (2, 2), "ld.global.s32 %r7, [%rd4];", "st.global.s32 [%rd5], %r7;"),
        (ptx_dense_copy_2d_kernel, "i1", [[True, False], [False, True]], (2, 2), "ld.global.u8 %r7, [%rd4];", "st.global.u8 [%rd5], %r7;"),
    ],
)
def test_ptx_ref_lowers_dense_tensor_copy_2d_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    dst_shape,
    load_instr: str,
    store_instr: str,
) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert "mad.lo.u32 %r7, %r1, 2, %r2;" in text
    assert load_instr in text
    assert store_instr in text


def test_ptx_ref_lowers_dense_copy_reduce_add_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")

    artifact = bb.compile(
        ptx_dense_copy_reduce_add_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_copy_reduce_add_2d_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd4];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_dense_copy_reduce_max_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[10.0, 1.0], [2.0, 7.0]], dtype="f32")

    artifact = bb.compile(
        ptx_dense_copy_reduce_max_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_copy_reduce_max_2d_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd4];" in text
    assert "max.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_dense_copy_reduce_or_i32_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1, 2], [4, 8]], dtype="i32")
    dst = bb.tensor([[16, 16], [16, 16]], dtype="i32")

    artifact = bb.compile(
        ptx_dense_copy_reduce_or_i32_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_copy_reduce_or_i32_2d_kernel(" in text
    assert "or.b32 %r3, %r3, %r4;" in text
    assert "st.global.s32 [%rd5], %r3;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "dst_shape", "load_instr", "store_instr"),
    [
        (ptx_parallel_dense_copy_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], (2, 2), "ld.global.f32 %f1, [%rd4];", "st.global.f32 [%rd5], %f1;"),
        (ptx_parallel_dense_copy_i32_2d_kernel, "i32", [[1, 2], [3, 4]], (2, 2), "ld.global.s32 %r7, [%rd4];", "st.global.s32 [%rd5], %r7;"),
        (ptx_parallel_dense_copy_2d_kernel, "i1", [[True, False], [False, True]], (2, 2), "ld.global.u8 %r7, [%rd4];", "st.global.u8 [%rd5], %r7;"),
    ],
)
def test_ptx_ref_lowers_parallel_dense_tensor_copy_2d_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    dst_shape,
    load_instr: str,
    store_instr: str,
) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "mad.lo.u32 %r7, %r2, 2, %r1;" in text
    assert load_instr in text
    assert store_instr in text


def test_ptx_ref_lowers_parallel_dense_copy_reduce_add_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")

    artifact = bb.compile(
        ptx_parallel_dense_copy_reduce_add_2d_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_parallel_dense_copy_reduce_add_2d_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd4];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_parallel_dense_copy_reduce_max_2d_kernel(tmp_path: Path) -> None:
    artifact = bb.compile(
        ptx_parallel_dense_copy_reduce_max_2d_kernel,
        bb.zeros((2, 2), dtype="f32"),
        bb.zeros((2, 2), dtype="f32"),
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_parallel_dense_copy_reduce_max_2d_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "max.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd5], %f3;" in text


def test_ptx_ref_lowers_parallel_dense_copy_reduce_or_i32_2d_kernel(tmp_path: Path) -> None:
    artifact = bb.compile(
        ptx_parallel_dense_copy_reduce_or_i32_2d_kernel,
        bb.zeros((2, 2), dtype="i32"),
        bb.zeros((2, 2), dtype="i32"),
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_parallel_dense_copy_reduce_or_i32_2d_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "or.b32 %r3, %r3, %r4;" in text
    assert "st.global.s32 [%rd5], %r3;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha", "dst_shape", "scalar_load", "instr"),
    [
        (ptx_dense_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], bb.Float32(1.5), (2, 2), "ld.param.f32 %f2, [alpha_param];", "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_scalar_sub_2d_kernel, "f32", [[10.0, 20.0], [30.0, 40.0]], bb.Float32(1.5), (2, 2), "ld.param.f32 %f2, [alpha_param];", "sub.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_scalar_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], bb.Float32(2.0), (2, 2), "ld.param.f32 %f2, [alpha_param];", "mul.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_scalar_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], bb.Float32(2.0), (2, 2), "ld.param.f32 %f2, [alpha_param];", "div.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], bb.Int32(7), (2, 2), "ld.param.s32 %r3, [alpha_param];", "add.s32 %r7, %r7, %r3;"),
        (ptx_dense_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(5), (2, 2), "ld.param.s32 %r3, [alpha_param];", "and.b32 %r7, %r7, %r3;"),
        (ptx_dense_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), (2, 2), "ld.param.s32 %r3, [alpha_param];", "or.b32 %r7, %r7, %r3;"),
        (ptx_dense_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), (2, 2), "ld.param.s32 %r3, [alpha_param];", "xor.b32 %r7, %r7, %r3;"),
        (ptx_dense_scalar_sub_i32_2d_kernel, "i32", [[10, 20], [30, 40]], bb.Int32(7), (2, 2), "ld.param.s32 %r3, [alpha_param];", "sub.s32 %r7, %r7, %r3;"),
        (ptx_dense_scalar_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], bb.Int32(2), (2, 2), "ld.param.s32 %r3, [alpha_param];", "mul.lo.s32 %r7, %r7, %r3;"),
        (ptx_dense_scalar_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], bb.Int32(2), (2, 2), "ld.param.s32 %r3, [alpha_param];", "div.s32 %r7, %r7, %r3;"),
    ],
)
def test_ptx_ref_lowers_dense_tensor_scalar_broadcast_2d_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha,
    dst_shape,
    scalar_load: str,
    instr: str,
) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert "mad.lo.u32 %r7, %r1, 2, %r2;" in text
    assert scalar_load in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha_values", "dst_shape", "scalar_load", "instr"),
    [
        (ptx_dense_tensor_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [1.5], (2, 2), "ld.global.f32 %f2, [%rd2];", "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_tensor_scalar_sub_2d_kernel, "f32", [[10.0, 20.0], [30.0, 40.0]], [1.5], (2, 2), "ld.global.f32 %f2, [%rd2];", "sub.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_tensor_scalar_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [2.0], (2, 2), "ld.global.f32 %f2, [%rd2];", "mul.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_tensor_scalar_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], [2.0], (2, 2), "ld.global.f32 %f2, [%rd2];", "div.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_tensor_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [7], (2, 2), "ld.global.s32 %r3, [%rd2];", "add.s32 %r7, %r7, %r3;"),
        (ptx_dense_tensor_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [5], (2, 2), "ld.global.s32 %r3, [%rd2];", "and.b32 %r7, %r7, %r3;"),
        (ptx_dense_tensor_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], (2, 2), "ld.global.s32 %r3, [%rd2];", "or.b32 %r7, %r7, %r3;"),
        (ptx_dense_tensor_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], (2, 2), "ld.global.s32 %r3, [%rd2];", "xor.b32 %r7, %r7, %r3;"),
        (ptx_dense_tensor_scalar_sub_i32_2d_kernel, "i32", [[10, 20], [30, 40]], [7], (2, 2), "ld.global.s32 %r3, [%rd2];", "sub.s32 %r7, %r7, %r3;"),
        (ptx_dense_tensor_scalar_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [2], (2, 2), "ld.global.s32 %r3, [%rd2];", "mul.lo.s32 %r7, %r7, %r3;"),
        (ptx_dense_tensor_scalar_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], [2], (2, 2), "ld.global.s32 %r3, [%rd2];", "div.s32 %r7, %r7, %r3;"),
    ],
)
def test_ptx_ref_lowers_dense_tensor_extent1_scalar_broadcast_2d_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha_values,
    dst_shape,
    scalar_load: str,
    instr: str,
) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    alpha = bb.tensor(alpha_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert scalar_load in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha", "dst_shape", "scalar_load", "instr"),
    [
        (ptx_parallel_dense_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], bb.Float32(1.5), (2, 2), "ld.param.f32 %f2, [alpha_param];", "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_parallel_dense_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], bb.Int32(7), (2, 2), "ld.param.s32 %r3, [alpha_param];", "add.s32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(5), (2, 2), "ld.param.s32 %r3, [alpha_param];", "and.b32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), (2, 2), "ld.param.s32 %r3, [alpha_param];", "or.b32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], bb.Int32(24), (2, 2), "ld.param.s32 %r3, [alpha_param];", "xor.b32 %r7, %r7, %r3;"),
    ],
)
def test_ptx_ref_lowers_parallel_dense_tensor_scalar_broadcast_2d_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha,
    dst_shape,
    scalar_load: str,
    instr: str,
) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "mad.lo.u32 %r7, %r2, 2, %r1;" in text
    assert scalar_load in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "alpha_values", "dst_shape", "scalar_load", "instr"),
    [
        (ptx_parallel_dense_tensor_scalar_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [1.5], (2, 2), "ld.global.f32 %f2, [%rd2];", "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_parallel_dense_tensor_scalar_sub_2d_kernel, "f32", [[10.0, 20.0], [30.0, 40.0]], [1.5], (2, 2), "ld.global.f32 %f2, [%rd2];", "sub.rn.f32 %f3, %f1, %f2;"),
        (ptx_parallel_dense_tensor_scalar_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [2.0], (2, 2), "ld.global.f32 %f2, [%rd2];", "mul.rn.f32 %f3, %f1, %f2;"),
        (ptx_parallel_dense_tensor_scalar_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], [2.0], (2, 2), "ld.global.f32 %f2, [%rd2];", "div.rn.f32 %f3, %f1, %f2;"),
        (ptx_parallel_dense_tensor_scalar_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [7], (2, 2), "ld.global.s32 %r3, [%rd2];", "add.s32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_tensor_scalar_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [5], (2, 2), "ld.global.s32 %r3, [%rd2];", "and.b32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_tensor_scalar_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], (2, 2), "ld.global.s32 %r3, [%rd2];", "or.b32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_tensor_scalar_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [24], (2, 2), "ld.global.s32 %r3, [%rd2];", "xor.b32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_tensor_scalar_sub_i32_2d_kernel, "i32", [[10, 20], [30, 40]], [7], (2, 2), "ld.global.s32 %r3, [%rd2];", "sub.s32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_tensor_scalar_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [2], (2, 2), "ld.global.s32 %r3, [%rd2];", "mul.lo.s32 %r7, %r7, %r3;"),
        (ptx_parallel_dense_tensor_scalar_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], [2], (2, 2), "ld.global.s32 %r3, [%rd2];", "div.s32 %r7, %r7, %r3;"),
    ],
)
def test_ptx_ref_lowers_parallel_dense_tensor_extent1_scalar_broadcast_2d_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    src_values,
    alpha_values,
    dst_shape,
    scalar_load: str,
    instr: str,
) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    alpha = bb.tensor(alpha_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, src, alpha, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert scalar_load in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "dst_shape", "instr"),
    [
        (ptx_dense_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]], (2, 2), "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_sub_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]], (2, 2), "sub.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_mul_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]], (2, 2), "mul.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_div_2d_kernel, "f32", [[8.0, 12.0], [18.0, 28.0]], [[2.0, 3.0], [6.0, 7.0]], (2, 2), "div.rn.f32 %f3, %f1, %f2;"),
        (ptx_dense_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [[10, 20], [30, 40]], (2, 2), "add.s32 %r11, %r11, %r12;"),
        (ptx_dense_sub_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [[10, 20], [30, 40]], (2, 2), "sub.s32 %r11, %r11, %r12;"),
        (ptx_dense_mul_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [[10, 20], [30, 40]], (2, 2), "mul.lo.s32 %r11, %r11, %r12;"),
        (ptx_dense_div_i32_2d_kernel, "i32", [[8, 12], [18, 28]], [[2, 3], [6, 7]], (2, 2), "div.s32 %r11, %r11, %r12;"),
        (ptx_dense_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], (2, 2), "and.b32 %r11, %r11, %r12;"),
        (ptx_dense_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], (2, 2), "or.b32 %r11, %r11, %r12;"),
        (ptx_dense_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], (2, 2), "xor.b32 %r11, %r11, %r12;"),
        (ptx_dense_bitand_i1_2d_kernel, "i1", [[True, False], [True, False]], [[True, True], [False, False]], (2, 2), "and.b32 %r11, %r11, %r12;"),
    ],
)
def test_ptx_ref_lowers_dense_tensor_binary_2d_kernels(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, dst_shape, instr: str) -> None:
    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert instr in text
    assert "mad.lo.u32 %r10, %r1, 2, %r2;" in text


def test_ptx_ref_lowers_dense_tensor_compare_2d_kernel(tmp_path: Path) -> None:
    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[2.0, 1.0], [3.0, 5.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_dense_cmp_lt_f32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_cmp_lt_f32_2d_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd6];" in text
    assert "setp.lt.f32 %p2, %f1, %f2;" in text
    assert "st.global.u8 [%rd7], %r11;" in text


def test_ptx_ref_lowers_dense_scalar_tensor_compare_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_dense_scalar_cmp_lt_f32_2d_kernel,
        src,
        bb.Float32(3.5),
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_scalar_cmp_lt_f32_2d_kernel(" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "setp.lt.f32 %p3, %f1, %f2;" in text
    assert "st.global.u8 [%rd5], %r6;" in text


def test_ptx_ref_lowers_dense_tensor_scalar_compare_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_dense_tensor_scalar_cmp_eq_i32_2d_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_tensor_scalar_cmp_eq_i32_2d_kernel(" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert "ld.global.s32 %r3, [%rd2];" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "setp.eq.s32 %p3, %r5, %r3;" in text
    assert "st.global.u8 [%rd6], %r6;" in text


def test_ptx_ref_lowers_dense_tensor_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        ptx_dense_select_f32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_select_f32_2d_kernel(" in text
    assert "ld.global.u8 %r5, [%rd5];" in text
    assert "selp.f32 %f3, %f1, %f2, %p2;" in text
    assert "st.global.f32 [%rd8], %f3;" in text


def test_ptx_ref_lowers_dense_scalar_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        ptx_dense_scalar_select_f32_2d_kernel,
        pred,
        src,
        bb.Float32(9.5),
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_scalar_select_f32_2d_kernel(" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "selp.f32 %f3, %f1, %f2, %p2;" in text
    assert "st.global.f32 [%rd8], %f3;" in text


def test_ptx_ref_lowers_dense_tensor_scalar_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.zeros((2, 2), dtype="i32")

    artifact = bb.compile(
        ptx_dense_tensor_scalar_select_i32_2d_kernel,
        pred,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_dense_tensor_scalar_select_i32_2d_kernel(" in text
    assert "ld.param.u64 %rd7, [alpha_param];" in text
    assert "ld.global.s32 %r7, [%rd7];" in text
    assert "selp.b32 %r6, %r7, %r6, %p2;" in text
    assert "st.global.s32 [%rd8], %r6;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "dst_shape", "instr"),
    [
        (ptx_parallel_dense_add_2d_kernel, "f32", [[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]], (2, 2), "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_parallel_dense_add_i32_2d_kernel, "i32", [[1, 2], [3, 4]], [[10, 20], [30, 40]], (2, 2), "add.s32 %r11, %r11, %r12;"),
        (ptx_parallel_dense_bitand_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], (2, 2), "and.b32 %r11, %r11, %r12;"),
        (ptx_parallel_dense_bitor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], (2, 2), "or.b32 %r11, %r11, %r12;"),
        (ptx_parallel_dense_bitxor_i32_2d_kernel, "i32", [[7, 11], [13, 17]], [[3, 5], [7, 9]], (2, 2), "xor.b32 %r11, %r11, %r12;"),
        (ptx_parallel_dense_bitxor_i1_2d_kernel, "i1", [[True, False], [True, False]], [[False, True], [True, False]], (2, 2), "xor.b32 %r11, %r11, %r12;"),
    ],
)
def test_ptx_ref_lowers_parallel_dense_tensor_binary_2d_kernels(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, dst_shape, instr: str) -> None:
    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "mad.lo.u32 %r10, %r2, 2, %r1;" in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "dst_shape", "instr"),
    [
        (ptx_broadcast_add_2d_kernel, "f32", [[1.0], [2.0]], [[10.0, 20.0, 30.0]], (2, 3), "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_broadcast_sub_2d_kernel, "f32", [[10.0], [20.0]], [[1.0, 2.0, 3.0]], (2, 3), "sub.rn.f32 %f3, %f1, %f2;"),
        (ptx_broadcast_mul_2d_kernel, "f32", [[2.0], [3.0]], [[4.0, 5.0, 6.0]], (2, 3), "mul.rn.f32 %f3, %f1, %f2;"),
        (ptx_broadcast_div_2d_kernel, "f32", [[8.0], [12.0]], [[2.0, 4.0, 8.0]], (2, 3), "div.rn.f32 %f3, %f1, %f2;"),
        (ptx_broadcast_add_i32_2d_kernel, "i32", [[1], [2]], [[10, 20, 30]], (2, 3), "add.s32 %r11, %r11, %r12;"),
        (ptx_broadcast_sub_i32_2d_kernel, "i32", [[10], [20]], [[1, 2, 3]], (2, 3), "sub.s32 %r11, %r11, %r12;"),
        (ptx_broadcast_mul_i32_2d_kernel, "i32", [[2], [3]], [[4, 5, 6]], (2, 3), "mul.lo.s32 %r11, %r11, %r12;"),
        (ptx_broadcast_div_i32_2d_kernel, "i32", [[8], [12]], [[2, 4, 8]], (2, 3), "div.s32 %r11, %r11, %r12;"),
        (ptx_broadcast_bitand_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], (2, 3), "and.b32 %r11, %r11, %r12;"),
        (ptx_broadcast_bitor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], (2, 3), "or.b32 %r11, %r11, %r12;"),
        (ptx_broadcast_bitxor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], (2, 3), "xor.b32 %r11, %r11, %r12;"),
        (ptx_broadcast_bitor_i1_2d_kernel, "i1", [[True], [False]], [[False, True, False]], (2, 3), "or.b32 %r11, %r11, %r12;"),
    ],
)
def test_ptx_ref_lowers_broadcast_tensor_binary_2d_kernels(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, dst_shape, instr: str) -> None:
    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert instr in text
    assert "mov.u32 %r3, 0;" in text or "mov.u32 %r4, 0;" in text


def test_ptx_ref_lowers_broadcast_tensor_compare_2d_kernel(tmp_path: Path) -> None:
    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[7, 5, 7]], dtype="i32")
    dst = bb.tensor([[False, False, False], [False, False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_broadcast_cmp_eq_i32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_broadcast_cmp_eq_i32_2d_kernel(" in text
    assert "ld.global.s32 %r11, [%rd5];" in text
    assert "ld.global.s32 %r12, [%rd6];" in text
    assert "setp.eq.s32 %p2, %r11, %r12;" in text
    assert "st.global.u8 [%rd7], %r11;" in text


def test_ptx_ref_lowers_broadcast_tensor_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False, True], [False, True, False]], dtype="i1")
    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[1, 3, 5]], dtype="i32")
    dst = bb.zeros((2, 3), dtype="i32")

    artifact = bb.compile(
        ptx_broadcast_select_i32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_broadcast_select_i32_2d_kernel(" in text
    assert any(line in text for line in ("mov.u32 %r3, 0;", "mov.u32 %r4, 0;", "mov.u32 %r5, 0;", "mov.u32 %r6, 0;"))
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.s32 [%rd8], %r6;" in text


def test_ptx_ref_lowers_parallel_dense_tensor_compare_2d_kernel(tmp_path: Path) -> None:
    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[2.0, 1.0], [3.0, 5.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_parallel_dense_cmp_lt_f32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "mad.lo.u32 %r10, %r2, 2, %r1;" in text
    assert "setp.lt.f32 %p3, %f1, %f2;" in text
    assert "st.global.u8 [%rd7], %r11;" in text


def test_ptx_ref_lowers_parallel_dense_scalar_tensor_compare_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_parallel_dense_scalar_cmp_lt_f32_2d_kernel,
        src,
        bb.Float32(3.5),
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "setp.lt.f32 %p3, %f1, %f2;" in text
    assert "st.global.u8 [%rd5], %r6;" in text


def test_ptx_ref_lowers_parallel_dense_tensor_scalar_compare_2d_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.tensor([[False, False], [False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_parallel_dense_tensor_scalar_cmp_eq_i32_2d_kernel,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "ld.param.u64 %rd2, [alpha_param];" in text
    assert "setp.eq.s32 %p3, %r5, %r3;" in text
    assert "st.global.u8 [%rd6], %r6;" in text


def test_ptx_ref_lowers_parallel_dense_tensor_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    lhs = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    rhs = bb.tensor([[10.0, 20.0], [30.0, 40.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        ptx_parallel_dense_select_f32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "selp.f32 %f3, %f1, %f2, %p2;" in text
    assert "st.global.f32 [%rd8], %f3;" in text


def test_ptx_ref_lowers_parallel_dense_scalar_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        ptx_parallel_dense_scalar_select_f32_2d_kernel,
        pred,
        src,
        bb.Float32(9.5),
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.param.f32 %f2, [alpha_param];" in text
    assert "selp.f32 %f3, %f1, %f2, %p2;" in text
    assert "st.global.f32 [%rd8], %f3;" in text


def test_ptx_ref_lowers_parallel_dense_tensor_scalar_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False], [False, True]], dtype="i1")
    src = bb.tensor([[7, 11], [13, 17]], dtype="i32")
    alpha = bb.tensor([13], dtype="i32")
    dst = bb.zeros((2, 2), dtype="i32")

    artifact = bb.compile(
        ptx_parallel_dense_tensor_scalar_select_i32_2d_kernel,
        pred,
        src,
        alpha,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.param.u64 %rd7, [alpha_param];" in text
    assert "selp.b32 %r6, %r7, %r6, %p2;" in text
    assert "st.global.s32 [%rd8], %r6;" in text


def test_ptx_ref_lowers_parallel_broadcast_tensor_compare_2d_kernel(tmp_path: Path) -> None:
    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[7, 5, 7]], dtype="i32")
    dst = bb.tensor([[False, False, False], [False, False, False]], dtype="i1")

    artifact = bb.compile(
        ptx_parallel_broadcast_cmp_eq_i32_2d_kernel,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 3;" in text
    assert "mov.u32 %r4, 0;" in text or "mov.u32 %r5, 0;" in text
    assert "setp.eq.s32 %p3, %r11, %r12;" in text
    assert "st.global.u8 [%rd7], %r11;" in text


def test_ptx_ref_lowers_parallel_broadcast_tensor_select_2d_kernel(tmp_path: Path) -> None:
    pred = bb.tensor([[True, False, True], [False, True, False]], dtype="i1")
    lhs = bb.tensor([[7], [11]], dtype="i32")
    rhs = bb.tensor([[1, 3, 5]], dtype="i32")
    dst = bb.zeros((2, 3), dtype="i32")

    artifact = bb.compile(
        ptx_parallel_broadcast_select_i32_2d_kernel,
        pred,
        lhs,
        rhs,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert any(line in text for line in ("mov.u32 %r3, 0;", "mov.u32 %r4, 0;", "mov.u32 %r5, 0;", "mov.u32 %r6, 0;"))
    assert "selp.b32 %r6, %r6, %r7, %p2;" in text
    assert "st.global.s32 [%rd8], %r6;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "lhs_values", "rhs_values", "dst_shape", "instr"),
    [
        (ptx_parallel_broadcast_add_2d_kernel, "f32", [[1.0], [2.0]], [[10.0, 20.0, 30.0]], (2, 3), "add.rn.f32 %f3, %f1, %f2;"),
        (ptx_parallel_broadcast_add_i32_2d_kernel, "i32", [[1], [2]], [[10, 20, 30]], (2, 3), "add.s32 %r11, %r11, %r12;"),
        (ptx_parallel_broadcast_bitand_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], (2, 3), "and.b32 %r11, %r11, %r12;"),
        (ptx_parallel_broadcast_bitor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], (2, 3), "or.b32 %r11, %r11, %r12;"),
        (ptx_parallel_broadcast_bitxor_i32_2d_kernel, "i32", [[7], [13]], [[3, 5, 9]], (2, 3), "xor.b32 %r11, %r11, %r12;"),
    ],
)
def test_ptx_ref_lowers_parallel_broadcast_tensor_binary_2d_kernels(tmp_path: Path, kernel, dtype: str, lhs_values, rhs_values, dst_shape, instr: str) -> None:
    lhs = bb.tensor(lhs_values, dtype=dtype)
    rhs = bb.tensor(rhs_values, dtype=dtype)
    dst = bb.zeros(dst_shape, dtype=dtype)

    artifact = bb.compile(kernel, lhs, rhs, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 3;" in text
    assert instr in text
    assert "mov.u32 %r3, 0;" in text or "mov.u32 %r4, 0;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "load_line", "instr", "store_line"),
    [
        (ptx_dense_sqrt_2d_kernel, "f32", [[4.0, 9.0], [16.0, 25.0]], "ld.global.f32 %f1, [%rd4];", "sqrt.rn.f32 %f2, %f1;", "st.global.f32 [%rd5], %f2;"),
        (ptx_dense_rsqrt_2d_kernel, "f32", [[4.0, 9.0], [16.0, 25.0]], "ld.global.f32 %f1, [%rd4];", "rsqrt.approx.f32 %f2, %f1;", "st.global.f32 [%rd5], %f2;"),
        (ptx_dense_neg_f32_2d_kernel, "f32", [[1.0, -2.0], [3.5, -4.5]], "ld.global.f32 %f1, [%rd4];", "neg.f32 %f2, %f1;", "st.global.f32 [%rd5], %f2;"),
        (ptx_dense_neg_i32_2d_kernel, "i32", [[1, -2], [3, -4]], "ld.global.s32 %r5, [%rd4];", "neg.s32 %r5, %r5;", "st.global.s32 [%rd5], %r5;"),
        (ptx_dense_abs_f32_2d_kernel, "f32", [[1.0, -2.0], [3.5, -4.5]], "ld.global.f32 %f1, [%rd4];", "abs.f32 %f2, %f1;", "st.global.f32 [%rd5], %f2;"),
        (ptx_dense_abs_i32_2d_kernel, "i32", [[1, -2], [3, -4]], "ld.global.s32 %r5, [%rd4];", "abs.s32 %r5, %r5;", "st.global.s32 [%rd5], %r5;"),
        (ptx_dense_bitnot_i32_2d_kernel, "i32", [[1, 2], [3, 4]], "ld.global.s32 %r5, [%rd4];", "not.b32 %r5, %r5;", "st.global.s32 [%rd5], %r5;"),
        (ptx_dense_bitnot_i1_2d_kernel, "i1", [[True, False], [False, True]], "ld.global.u8 %r5, [%rd4];", "xor.b32 %r5, %r5, 1;", "st.global.u8 [%rd5], %r5;"),
    ],
)
def test_ptx_ref_lowers_dense_tensor_unary_2d_kernels(tmp_path: Path, kernel, dtype: str, src_values, load_line: str, instr: str, store_line: str) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros((2, 2), dtype=dtype)

    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert "mad.lo.u32 %r8, %r1, 2, %r2;" in text
    assert load_line in text
    assert instr in text
    assert store_line in text


@pytest.mark.parametrize(
    ("kernel", "src_values", "expected_lines"),
    [
        (ptx_dense_sin_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], ("sin.approx.f32 %f2, %f1;",)),
        (ptx_dense_cos_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], ("cos.approx.f32 %f2, %f1;",)),
        (ptx_dense_exp2_2d_kernel, [[0.0, 1.0], [2.0, 3.0]], ("ex2.approx.f32 %f2, %f1;",)),
        (
            ptx_dense_exp_2d_kernel,
            [[0.0, 0.5], [1.0, 1.5]],
            ("mov.f32 %f2, 0f3FB8AA3B;", "mul.rn.f32 %f1, %f1, %f2;", "ex2.approx.f32 %f2, %f1;"),
        ),
        (ptx_dense_log2_2d_kernel, [[1.0, 2.0], [4.0, 8.0]], ("lg2.approx.f32 %f2, %f1;",)),
        (
            ptx_dense_log_2d_kernel,
            [[1.0, 2.0], [4.0, 8.0]],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3F317218;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (
            ptx_dense_log10_2d_kernel,
            [[1.0, 10.0], [100.0, 1000.0]],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3E9A209B;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (ptx_dense_round_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rni.f32.f32 %f2, %f1;",)),
        (ptx_dense_floor_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rmi.f32.f32 %f2, %f1;",)),
        (ptx_dense_ceil_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rpi.f32.f32 %f2, %f1;",)),
        (ptx_dense_trunc_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rzi.f32.f32 %f2, %f1;",)),
        (
            ptx_dense_atan_2d_kernel,
            [[-1.5, -0.75], [0.0, 0.75]],
            ("rcp.approx.f32 %f4, %f2;", "selp.f32 %f2, %f4, %f2, %p0;", "selp.f32 %f2, %f5, %f3, %p0;"),
        ),
        (
            ptx_dense_asin_2d_kernel,
            [[-0.875, -0.5], [0.0, 0.625]],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "selp.f32 %f3, %f8, %f3, %p2;"),
        ),
        (
            ptx_dense_acos_2d_kernel,
            [[-0.875, -0.5], [0.0, 0.625]],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "abs.f32 %f4, %f2;", "abs.f32 %f5, %f1;"),
        ),
        (
            ptx_dense_erf_2d_kernel,
            [[-1.5, -0.75], [0.0, 0.75]],
            ("rcp.approx.f32 %f3, %f3;", "ex2.approx.f32 %f5, %f5;", "selp.f32 %f2, %f5, %f4, %p0;"),
        ),
    ],
)
def test_ptx_ref_lowers_dense_native_math_tensor_unary_2d_kernels(tmp_path: Path, kernel, src_values, expected_lines: tuple[str, ...]) -> None:
    src = bb.tensor(src_values, dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert "mad.lo.u32 %r8, %r1, 2, %r2;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    for line in expected_lines:
        assert line in text
    assert any(line in text for line in ("st.global.f32 [%rd5], %f2;", "st.global.f32 [%rd5], %f3;"))


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "instr"),
    [
        (ptx_parallel_dense_sqrt_2d_kernel, "f32", [[4.0, 9.0], [16.0, 25.0]], "sqrt.rn.f32 %f2, %f1;"),
        (ptx_parallel_dense_neg_f32_2d_kernel, "f32", [[1.0, -2.0], [3.5, -4.5]], "neg.f32 %f2, %f1;"),
        (ptx_parallel_dense_neg_i32_2d_kernel, "i32", [[1, -2], [3, -4]], "neg.s32 %r5, %r5;"),
        (ptx_parallel_dense_abs_f32_2d_kernel, "f32", [[1.0, -2.0], [3.5, -4.5]], "abs.f32 %f2, %f1;"),
        (ptx_parallel_dense_abs_i32_2d_kernel, "i32", [[1, -2], [3, -4]], "abs.s32 %r5, %r5;"),
        (ptx_parallel_dense_bitnot_i32_2d_kernel, "i32", [[1, 2], [3, 4]], "not.b32 %r5, %r5;"),
        (ptx_parallel_dense_bitnot_i1_2d_kernel, "i1", [[True, False], [False, True]], "xor.b32 %r5, %r5, 1;"),
    ],
)
def test_ptx_ref_lowers_parallel_dense_tensor_unary_2d_kernel(tmp_path: Path, kernel, dtype: str, src_values, instr: str) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros((2, 2), dtype=dtype)

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "mad.lo.u32 %r8, %r2, 2, %r1;" in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "src_values", "expected_lines"),
    [
        (ptx_parallel_dense_sin_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], ("sin.approx.f32 %f2, %f1;",)),
        (ptx_parallel_dense_cos_2d_kernel, [[0.0, 0.5], [1.0, 1.5]], ("cos.approx.f32 %f2, %f1;",)),
        (ptx_parallel_dense_exp2_2d_kernel, [[0.0, 1.0], [2.0, 3.0]], ("ex2.approx.f32 %f2, %f1;",)),
        (
            ptx_parallel_dense_exp_2d_kernel,
            [[0.0, 0.5], [1.0, 1.5]],
            ("mov.f32 %f2, 0f3FB8AA3B;", "mul.rn.f32 %f1, %f1, %f2;", "ex2.approx.f32 %f2, %f1;"),
        ),
        (ptx_parallel_dense_log2_2d_kernel, [[1.0, 2.0], [4.0, 8.0]], ("lg2.approx.f32 %f2, %f1;",)),
        (
            ptx_parallel_dense_log_2d_kernel,
            [[1.0, 2.0], [4.0, 8.0]],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3F317218;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (
            ptx_parallel_dense_log10_2d_kernel,
            [[1.0, 10.0], [100.0, 1000.0]],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3E9A209B;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (ptx_parallel_dense_floor_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rmi.f32.f32 %f2, %f1;",)),
        (ptx_parallel_dense_ceil_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rpi.f32.f32 %f2, %f1;",)),
        (ptx_parallel_dense_trunc_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rzi.f32.f32 %f2, %f1;",)),
        (ptx_parallel_dense_round_2d_kernel, [[-1.75, -0.25], [0.25, 1.75]], ("cvt.rni.f32.f32 %f2, %f1;",)),
        (
            ptx_parallel_dense_atan_2d_kernel,
            [[-1.5, -0.75], [0.0, 0.75]],
            ("rcp.approx.f32 %f4, %f2;", "selp.f32 %f2, %f4, %f2, %p0;", "selp.f32 %f2, %f5, %f3, %p0;"),
        ),
        (
            ptx_parallel_dense_asin_2d_kernel,
            [[-0.875, -0.5], [0.0, 0.625]],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "selp.f32 %f3, %f8, %f3, %p2;"),
        ),
        (
            ptx_parallel_dense_acos_2d_kernel,
            [[-0.875, -0.5], [0.0, 0.625]],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "abs.f32 %f4, %f2;", "abs.f32 %f5, %f1;"),
        ),
        (
            ptx_parallel_dense_erf_2d_kernel,
            [[-1.5, -0.75], [0.0, 0.75]],
            ("rcp.approx.f32 %f3, %f3;", "ex2.approx.f32 %f5, %f5;", "selp.f32 %f2, %f5, %f4, %p0;"),
        ),
    ],
)
def test_ptx_ref_lowers_parallel_dense_native_math_tensor_unary_2d_kernel(tmp_path: Path, kernel, src_values, expected_lines: tuple[str, ...]) -> None:
    src = bb.tensor(src_values, dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 2;" in text
    assert "mad.lo.u32 %r8, %r2, 2, %r1;" in text
    for line in expected_lines:
        assert line in text


def test_ptx_ref_lowers_indexed_sqrt_kernel(tmp_path: Path) -> None:
    src = bb.tensor([4.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_sqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_sqrt_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "sqrt.rn.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


def test_ptx_ref_lowers_indexed_rsqrt_kernel(tmp_path: Path) -> None:
    src = bb.tensor([4.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_rsqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_rsqrt_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "rsqrt.approx.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


def test_ptx_ref_lowers_indexed_neg_f32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([4.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_neg_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_neg_f32_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "neg.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


def test_ptx_ref_lowers_indexed_neg_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_neg_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_neg_i32_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.s32 %r5, [%rd4];" in text
    assert "neg.s32 %r5, %r5;" in text
    assert "st.global.s32 [%rd5], %r5;" in text


def test_ptx_ref_lowers_indexed_abs_f32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([float(index) - 64.0 for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_abs_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_abs_f32_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "abs.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


def test_ptx_ref_lowers_indexed_abs_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_abs_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_abs_i32_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.s32 %r5, [%rd4];" in text
    assert "abs.s32 %r5, %r5;" in text
    assert "st.global.s32 [%rd5], %r5;" in text


def test_ptx_ref_lowers_indexed_bitnot_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_bitnot_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_bitnot_i32_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.s32 %r5, [%rd4];" in text
    assert "not.b32 %r5, %r5;" in text
    assert "st.global.s32 [%rd5], %r5;" in text


def test_ptx_ref_lowers_indexed_bitnot_i1_kernel(tmp_path: Path) -> None:
    src = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    dst = bb.zeros((128,), dtype="i1")

    artifact = bb.compile(
        ptx_indexed_bitnot_i1_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_bitnot_i1_kernel(" in text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "xor.b32 %r5, %r5, 1;" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_direct_sqrt_kernel(tmp_path: Path) -> None:
    src = bb.tensor([4.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_sqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_sqrt_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "sqrt.rn.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


def test_ptx_ref_lowers_direct_neg_f32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([4.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_neg_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_neg_f32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "neg.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


def test_ptx_ref_lowers_direct_neg_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_neg_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_neg_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.s32 %r5, [%rd4];" in text
    assert "neg.s32 %r5, %r5;" in text
    assert "st.global.s32 [%rd5], %r5;" in text


def test_ptx_ref_lowers_direct_abs_f32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([float(index) - 64.0 for index in range(128)], dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_abs_f32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_abs_f32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "abs.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


def test_ptx_ref_lowers_direct_abs_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index - 64 for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_abs_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_abs_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.s32 %r5, [%rd4];" in text
    assert "abs.s32 %r5, %r5;" in text
    assert "st.global.s32 [%rd5], %r5;" in text


def test_ptx_ref_lowers_direct_bitnot_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([index for index in range(128)], dtype="i32")
    dst = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_direct_bitnot_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_bitnot_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.s32 %r5, [%rd4];" in text
    assert "not.b32 %r5, %r5;" in text
    assert "st.global.s32 [%rd5], %r5;" in text


def test_ptx_ref_lowers_direct_rsqrt_kernel(tmp_path: Path) -> None:
    src = bb.tensor([4.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        ptx_direct_rsqrt_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_rsqrt_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "rsqrt.approx.f32 %f2, %f1;" in text
    assert "st.global.f32 [%rd5], %f2;" in text


@pytest.mark.parametrize(
    ("kernel", "src_values", "expected_lines"),
    [
        (ptx_indexed_sin_kernel, [0.5 for _ in range(128)], ("sin.approx.f32 %f2, %f1;",)),
        (ptx_indexed_cos_kernel, [0.5 for _ in range(128)], ("cos.approx.f32 %f2, %f1;",)),
        (ptx_indexed_exp2_kernel, [1.0 for _ in range(128)], ("ex2.approx.f32 %f2, %f1;",)),
        (
            ptx_indexed_exp_kernel,
            [0.125 * float(index % 8) for index in range(128)],
            ("mov.f32 %f2, 0f3FB8AA3B;", "mul.rn.f32 %f1, %f1, %f2;", "ex2.approx.f32 %f2, %f1;"),
        ),
        (ptx_indexed_log2_kernel, [2.0 for _ in range(128)], ("lg2.approx.f32 %f2, %f1;",)),
        (
            ptx_indexed_log_kernel,
            [float(1 << (index % 8)) for index in range(128)],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3F317218;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (
            ptx_indexed_log10_kernel,
            [float(10 ** (index % 4)) for index in range(128)],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3E9A209B;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (ptx_indexed_round_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rni.f32.f32 %f2, %f1;",)),
        (ptx_indexed_floor_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rmi.f32.f32 %f2, %f1;",)),
        (ptx_indexed_ceil_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rpi.f32.f32 %f2, %f1;",)),
        (ptx_indexed_trunc_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rzi.f32.f32 %f2, %f1;",)),
        (
            ptx_indexed_atan_kernel,
            [0.25 * float((index % 16) - 8) for index in range(128)],
            ("rcp.approx.f32 %f4, %f2;", "selp.f32 %f2, %f4, %f2, %p0;", "selp.f32 %f2, %f5, %f3, %p0;"),
        ),
        (
            ptx_indexed_asin_kernel,
            [-0.875 + 0.125 * float(index % 15) for index in range(128)],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "selp.f32 %f3, %f8, %f3, %p2;"),
        ),
        (
            ptx_indexed_acos_kernel,
            [-0.875 + 0.125 * float(index % 15) for index in range(128)],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "abs.f32 %f4, %f2;", "abs.f32 %f5, %f1;"),
        ),
        (
            ptx_indexed_erf_kernel,
            [0.25 * float((index % 16) - 8) for index in range(128)],
            ("rcp.approx.f32 %f3, %f3;", "ex2.approx.f32 %f5, %f5;", "selp.f32 %f2, %f5, %f4, %p0;"),
        ),
    ],
)
def test_ptx_ref_lowers_indexed_native_math_kernel(tmp_path: Path, kernel, src_values, expected_lines: tuple[str, ...]) -> None:
    src = bb.tensor(src_values, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mad.lo.u32 %r4, %r2, %r3, %r1;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    for line in expected_lines:
        assert line in text
    assert any(line in text for line in ("st.global.f32 [%rd5], %f2;", "st.global.f32 [%rd5], %f3;"))


@pytest.mark.parametrize(
    ("kernel", "src_values", "expected_lines"),
    [
        (ptx_direct_sin_kernel, [0.5 for _ in range(128)], ("sin.approx.f32 %f2, %f1;",)),
        (ptx_direct_cos_kernel, [0.5 for _ in range(128)], ("cos.approx.f32 %f2, %f1;",)),
        (ptx_direct_exp2_kernel, [1.0 for _ in range(128)], ("ex2.approx.f32 %f2, %f1;",)),
        (
            ptx_direct_exp_kernel,
            [0.125 * float(index % 8) for index in range(128)],
            ("mov.f32 %f2, 0f3FB8AA3B;", "mul.rn.f32 %f1, %f1, %f2;", "ex2.approx.f32 %f2, %f1;"),
        ),
        (ptx_direct_log2_kernel, [2.0 for _ in range(128)], ("lg2.approx.f32 %f2, %f1;",)),
        (
            ptx_direct_log_kernel,
            [float(1 << (index % 8)) for index in range(128)],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3F317218;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (
            ptx_direct_log10_kernel,
            [float(10 ** (index % 4)) for index in range(128)],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3E9A209B;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (ptx_direct_round_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rni.f32.f32 %f2, %f1;",)),
        (ptx_direct_floor_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rmi.f32.f32 %f2, %f1;",)),
        (ptx_direct_ceil_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rpi.f32.f32 %f2, %f1;",)),
        (ptx_direct_trunc_kernel, [0.5 * float((index % 16) - 8) + 0.25 for index in range(128)], ("cvt.rzi.f32.f32 %f2, %f1;",)),
        (
            ptx_direct_atan_kernel,
            [0.25 * float((index % 16) - 8) for index in range(128)],
            ("rcp.approx.f32 %f4, %f2;", "selp.f32 %f2, %f4, %f2, %p0;", "selp.f32 %f2, %f5, %f3, %p0;"),
        ),
        (
            ptx_direct_asin_kernel,
            [-0.875 + 0.125 * float(index % 15) for index in range(128)],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "selp.f32 %f3, %f8, %f3, %p2;"),
        ),
        (
            ptx_direct_acos_kernel,
            [-0.875 + 0.125 * float(index % 15) for index in range(128)],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "abs.f32 %f4, %f2;", "abs.f32 %f5, %f1;"),
        ),
        (
            ptx_direct_erf_kernel,
            [0.25 * float((index % 16) - 8) for index in range(128)],
            ("rcp.approx.f32 %f3, %f3;", "ex2.approx.f32 %f5, %f5;", "selp.f32 %f2, %f5, %f4, %p0;"),
        ),
    ],
)
def test_ptx_ref_lowers_direct_native_math_kernel(tmp_path: Path, kernel, src_values, expected_lines: tuple[str, ...]) -> None:
    src = bb.tensor(src_values, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    for line in expected_lines:
        assert line in text
    assert any(line in text for line in ("st.global.f32 [%rd5], %f2;", "st.global.f32 [%rd5], %f3;"))


def test_ptx_ref_lowers_direct_bitnot_i1_kernel(tmp_path: Path) -> None:
    src = bb.tensor([bool(index % 2) for index in range(128)], dtype="i1")
    dst = bb.zeros((128,), dtype="i1")

    artifact = bb.compile(
        ptx_direct_bitnot_i1_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_direct_bitnot_i1_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "ld.global.u8 %r5, [%rd4];" in text
    assert "xor.b32 %r5, %r5, 1;" in text
    assert "st.global.u8 [%rd5], %r5;" in text


def test_ptx_ref_lowers_reduce_add_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((1,), dtype="f32")

    artifact = bb.compile(
        ptx_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_reduce_add_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "mov.f32 %f1, 0f00000000;" in text
    assert "ld.global.f32 %f2, [%rd4];" in text
    assert "add.rn.f32 %f1, %f1, %f2;" in text
    assert "st.global.f32 [%rd2], %f1;" in text


def test_ptx_ref_lowers_parallel_reduce_add_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 1024, dtype="f32")
    dst = bb.zeros((1,), dtype="f32")

    artifact = bb.compile(
        ptx_parallel_reduce_add_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".shared .align 4 .f32 smem[256];" in text
    assert "bar.sync 0;" in text
    assert "L_reduce_load:" in text
    assert "ld.shared.f32 %f3, [%rd5];" in text
    assert "st.global.f32 [%rd2], %f3;" in text


def test_ptx_ref_lowers_parallel_reduce_add_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1] * 1024, dtype="i32")
    dst = bb.zeros((1,), dtype="i32")

    artifact = bb.compile(
        ptx_parallel_reduce_add_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".shared .align 4 .s32 smem[256];" in text
    assert "bar.sync 0;" in text
    assert "L_reduce_load:" in text
    assert "ld.shared.s32 %r5, [%rd5];" in text
    assert "st.global.s32 [%rd2], %r5;" in text


def test_ptx_ref_lowers_reduce_or_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1, 2, 4, 8], dtype="i32")
    dst = bb.zeros((1,), dtype="i32")

    artifact = bb.compile(
        ptx_reduce_or_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_reduce_or_i32_kernel(" in text
    assert "mov.s32 %r3, 0;" in text
    assert "ld.global.s32 %r4, [%rd4];" in text
    assert "or.b32 %r3, %r3, %r4;" in text
    assert "st.global.s32 [%rd2], %r3;" in text


@pytest.mark.parametrize(
    ("kernel", "instr"),
    [
        (ptx_parallel_reduce_and_i32_kernel, "and.b32 %r4, %r4, %r5;"),
        (ptx_parallel_reduce_or_i32_kernel, "or.b32 %r4, %r4, %r5;"),
        (ptx_parallel_reduce_xor_i32_kernel, "xor.b32 %r4, %r4, %r5;"),
    ],
)
def test_ptx_ref_lowers_parallel_bitwise_reduce_i32_kernels(tmp_path: Path, kernel, instr: str) -> None:
    src = bb.tensor([1, 2, 4, 8], dtype="i32")
    dst = bb.zeros((1,), dtype="i32")

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".shared .align 4 .s32 smem[256];" in text
    assert "bar.sync 0;" in text
    assert "L_reduce_load:" in text
    assert instr in text
    assert "st.global.s32 [%rd2], %r5;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "mad_line", "bound_line", "load_line", "store_line", "instr"),
    [
        (ptx_reduce_rows_add_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_reduce_cols_add_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_reduce_rows_mul_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "mul.rn.f32 %f1, %f1, %f2;"),
        (ptx_reduce_cols_mul_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "mul.rn.f32 %f1, %f1, %f2;"),
        (ptx_reduce_rows_max_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "max.f32 %f1, %f1, %f2;"),
        (ptx_reduce_cols_max_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "max.f32 %f1, %f1, %f2;"),
        (ptx_reduce_rows_min_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "min.f32 %f1, %f1, %f2;"),
        (ptx_reduce_cols_min_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "min.f32 %f1, %f1, %f2;"),
        (ptx_reduce_rows_add_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "add.s32 %r3, %r3, %r4;"),
        (ptx_reduce_cols_add_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "add.s32 %r3, %r3, %r4;"),
        (ptx_reduce_rows_mul_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "mul.lo.s32 %r3, %r3, %r4;"),
        (ptx_reduce_cols_mul_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "mul.lo.s32 %r3, %r3, %r4;"),
        (ptx_reduce_rows_max_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "max.s32 %r3, %r3, %r4;"),
        (ptx_reduce_cols_max_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "max.s32 %r3, %r3, %r4;"),
        (ptx_reduce_rows_min_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "min.s32 %r3, %r3, %r4;"),
        (ptx_reduce_cols_min_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "min.s32 %r3, %r3, %r4;"),
        (ptx_reduce_rows_and_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "and.b32 %r3, %r3, %r4;"),
        (ptx_reduce_cols_and_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "and.b32 %r3, %r3, %r4;"),
        (ptx_reduce_rows_or_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "or.b32 %r3, %r3, %r4;"),
        (ptx_reduce_cols_or_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "or.b32 %r3, %r3, %r4;"),
        (ptx_reduce_rows_xor_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "xor.b32 %r3, %r3, %r4;"),
        (ptx_reduce_cols_xor_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "xor.b32 %r3, %r3, %r4;"),
    ],
)
def test_ptx_ref_lowers_tensor_reduce_2d_kernels(
    tmp_path: Path, kernel, dtype: str, mad_line: str, bound_line: str, load_line: str, store_line: str, instr: str
) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32") if dtype == "f32" else bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="i32")
    dst = bb.zeros((2,), dtype="f32") if "rows" in kernel.__name__ and dtype == "f32" else None
    if dst is None:
        if "rows" in kernel.__name__:
            dst = bb.zeros((2,), dtype="i32" if dtype == "i32" else "f32")
        else:
            dst = bb.zeros((3,), dtype="i32" if dtype == "i32" else "f32")

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry " + kernel.__name__ + "(" in text
    assert bound_line in text
    assert mad_line in text
    assert load_line in text
    assert instr in text
    assert store_line in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "mad_line", "bound_line", "load_line", "store_line", "instr"),
    [
        (ptx_parallel_reduce_rows_add_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_cols_add_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_rows_mul_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "mul.rn.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_cols_mul_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "mul.rn.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_rows_max_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "max.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_cols_max_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "max.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_rows_min_2d_kernel, "f32", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "min.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_cols_min_2d_kernel, "f32", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "min.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_rows_add_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "add.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_cols_add_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "add.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_rows_mul_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "mul.lo.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_cols_mul_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "mul.lo.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_rows_max_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "max.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_cols_max_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "max.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_rows_min_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "min.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_cols_min_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "min.s32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_rows_and_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "and.b32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_cols_and_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "and.b32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_rows_or_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "or.b32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_cols_or_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "or.b32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_rows_xor_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "xor.b32 %r3, %r3, %r4;"),
        (ptx_parallel_reduce_cols_xor_i32_2d_kernel, "i32", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "xor.b32 %r3, %r3, %r4;"),
    ],
)
def test_ptx_ref_lowers_parallel_tensor_reduce_2d_kernels(
    tmp_path: Path, kernel, dtype: str, mad_line: str, bound_line: str, load_line: str, store_line: str, instr: str
) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32") if dtype == "f32" else bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="i32")
    dst = bb.zeros((2,), dtype="f32") if "rows" in kernel.__name__ and dtype == "f32" else None
    if dst is None:
        if "rows" in kernel.__name__:
            dst = bb.zeros((2,), dtype="i32" if dtype == "i32" else "f32")
        else:
            dst = bb.zeros((3,), dtype="i32" if dtype == "i32" else "f32")

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry " + kernel.__name__ + "(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert bound_line in text
    assert mad_line in text
    assert load_line in text
    assert instr in text
    assert store_line in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "global_index_line", "mad_line", "bound_line", "load_line", "store_line", "instr"),
    [
        (ptx_multiblock_reduce_rows_add_2d_kernel, "f32", "mad.lo.u32 %r1, %r1, 2, %r0;", "mad.lo.u32 %r3, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_multiblock_reduce_cols_add_2d_kernel, "f32", "mad.lo.u32 %r1, %r1, 2, %r0;", "mad.lo.u32 %r3, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.f32 %f2, [%rd4];", "st.global.f32 [%rd4], %f1;", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_multiblock_reduce_rows_or_i32_2d_kernel, "i32", "mad.lo.u32 %r1, %r1, 2, %r0;", "mad.lo.u32 %r5, %r1, 3, %r2;", "setp.ge.u32 %p1, %r1, 2;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "or.b32 %r3, %r3, %r4;"),
        (ptx_multiblock_reduce_cols_or_i32_2d_kernel, "i32", "mad.lo.u32 %r1, %r1, 2, %r0;", "mad.lo.u32 %r5, %r2, 3, %r1;", "setp.ge.u32 %p1, %r1, 3;", "ld.global.s32 %r4, [%rd4];", "st.global.s32 [%rd4], %r3;", "or.b32 %r3, %r3, %r4;"),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_reduce_2d_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    global_index_line: str,
    mad_line: str,
    bound_line: str,
    load_line: str,
    store_line: str,
    instr: str,
) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32") if dtype == "f32" else bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="i32")
    dst = bb.zeros((2,), dtype="f32") if "rows" in kernel.__name__ and dtype == "f32" else None
    if dst is None:
        if "rows" in kernel.__name__:
            dst = bb.zeros((2,), dtype="i32" if dtype == "i32" else "f32")
        else:
            dst = bb.zeros((3,), dtype="i32" if dtype == "i32" else "f32")

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry " + kernel.__name__ + "(" in text
    assert "mov.u32 %r0, %tid.x;" in text
    assert "mov.u32 %r1, %ctaid.x;" in text
    assert global_index_line in text
    assert bound_line in text
    assert mad_line in text
    assert load_line in text
    assert instr in text
    assert store_line in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "instr", "shared_decl", "store_instr"),
    [
        (ptx_parallel_reduce_mul_kernel, "f32", "mul.rn.f32 %f1, %f1, %f2;", ".shared .align 4 .f32 smem[256];", "st.global.f32 [%rd2], %f3;"),
        (ptx_parallel_reduce_max_kernel, "f32", "max.f32 %f1, %f1, %f2;", ".shared .align 4 .f32 smem[256];", "st.global.f32 [%rd2], %f3;"),
        (ptx_parallel_reduce_min_kernel, "f32", "min.f32 %f1, %f1, %f2;", ".shared .align 4 .f32 smem[256];", "st.global.f32 [%rd2], %f3;"),
        (ptx_parallel_reduce_mul_i32_kernel, "i32", "mul.lo.s32 %r4, %r4, %r5;", ".shared .align 4 .s32 smem[256];", "st.global.s32 [%rd2], %r5;"),
        (ptx_parallel_reduce_max_i32_kernel, "i32", "max.s32 %r4, %r4, %r5;", ".shared .align 4 .s32 smem[256];", "st.global.s32 [%rd2], %r5;"),
        (ptx_parallel_reduce_min_i32_kernel, "i32", "min.s32 %r4, %r4, %r5;", ".shared .align 4 .s32 smem[256];", "st.global.s32 [%rd2], %r5;"),
    ],
)
def test_ptx_ref_lowers_other_parallel_reduce_kernels(tmp_path: Path, kernel, dtype: str, instr: str, shared_decl: str, store_instr: str) -> None:
    src = bb.tensor([1.0] * 1024, dtype="f32") if dtype == "f32" else bb.tensor([1] * 1024, dtype="i32")
    dst = bb.zeros((1,), dtype=dtype)

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert shared_decl in text
    assert "bar.sync 0;" in text
    assert "L_reduce_load:" in text
    assert instr in text
    assert store_instr in text


def test_ptx_ref_lowers_reduce_max_i32_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1, 9, 3, 7], dtype="i32")
    dst = bb.zeros((1,), dtype="i32")

    artifact = bb.compile(
        ptx_reduce_max_i32_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_reduce_max_i32_kernel(" in text
    assert "mov.s32 %r3, -99;" in text
    assert "ld.global.s32 %r4, [%rd4];" in text
    assert "max.s32 %r3, %r3, %r4;" in text
    assert "st.global.s32 [%rd2], %r3;" in text


def test_ptx_ref_lowers_reduce_add_2d_bundle_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        ptx_reduce_add_2d_bundle_kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_reduce_add_2d_bundle_kernel(" in text
    assert "L_scalar_rows:" in text
    assert "L_rows_outer:" in text
    assert "L_cols_outer:" in text
    assert "add.rn.f32 %f1, %f1, %f2;" in text
    assert "st.global.f32 [%rd2], %f1;" in text
    assert "st.global.f32 [%rd6], %f1;" in text


@pytest.mark.parametrize(
    ("kernel", "instr"),
    [
        (ptx_reduce_mul_2d_bundle_kernel, "mul.rn.f32"),
        (ptx_reduce_max_2d_bundle_kernel, "max.f32"),
        (ptx_reduce_min_2d_bundle_kernel, "min.f32"),
    ],
)
def test_ptx_ref_lowers_other_reduce_2d_bundle_kernels(tmp_path: Path, kernel, instr: str) -> None:
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
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "L_scalar_rows:" in text
    assert "L_rows_outer:" in text
    assert "L_cols_outer:" in text
    assert f"{instr} %f1, %f1, %f2;" in text


@pytest.mark.parametrize(
    ("kernel", "instr"),
    [
        (ptx_reduce_add_i32_2d_bundle_kernel, "add.s32"),
        (ptx_reduce_mul_i32_2d_bundle_kernel, "mul.lo.s32"),
        (ptx_reduce_max_i32_2d_bundle_kernel, "max.s32"),
        (ptx_reduce_min_i32_2d_bundle_kernel, "min.s32"),
        (ptx_reduce_and_i32_2d_bundle_kernel, "and.b32"),
        (ptx_reduce_or_i32_2d_bundle_kernel, "or.b32"),
        (ptx_reduce_xor_i32_2d_bundle_kernel, "xor.b32"),
    ],
)
def test_ptx_ref_lowers_i32_reduce_2d_bundle_kernels(tmp_path: Path, kernel, instr: str) -> None:
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
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "L_scalar_rows:" in text
    assert "L_rows_outer:" in text
    assert "L_cols_outer:" in text
    assert f"{instr} %r3, %r3, %r4;" in text
    assert "st.global.s32 [%rd2], %r3;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "instr"),
    [
        (ptx_reduce_add_rowcol_2d_bundle_kernel, "f32", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_reduce_or_i32_rowcol_2d_bundle_kernel, "i32", "or.b32 %r3, %r3, %r4;"),
    ],
)
def test_ptx_ref_lowers_rowcol_reduce_2d_bundle_kernels(tmp_path: Path, kernel, dtype: str, instr: str) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32") if dtype == "f32" else bb.tensor([[1, 2, 4], [8, 1, 2]], dtype="i32")
    dst_rows = bb.zeros((2,), dtype=dtype)
    dst_cols = bb.zeros((3,), dtype=dtype)

    artifact = bb.compile(
        kernel,
        src,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_outer:" in text
    assert instr in text
    assert "st.global" in text


def test_ptx_ref_lowers_parallel_reduce_add_2d_bundle_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    dst_scalar = bb.zeros((1,), dtype="f32")
    dst_rows = bb.zeros((2,), dtype="f32")
    dst_cols = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        ptx_parallel_reduce_add_2d_bundle_kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".shared .align 4 .f32 smem[8];" in text
    assert "bar.sync 0;" in text
    assert "L_scalar_load:" in text
    assert "L_rows_phase:" in text
    assert "L_cols_phase:" in text
    assert "st.global.f32 [%rd2], %f3;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "instr"),
    [
        (ptx_parallel_reduce_add_rowcol_2d_bundle_kernel, "f32", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_parallel_reduce_or_i32_rowcol_2d_bundle_kernel, "i32", "or.b32 %r3, %r3, %r4;"),
    ],
)
def test_ptx_ref_lowers_parallel_rowcol_reduce_2d_bundle_kernels(tmp_path: Path, kernel, dtype: str, instr: str) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32") if dtype == "f32" else bb.tensor([[1, 2, 4], [8, 1, 2]], dtype="i32")
    dst_rows = bb.zeros((2,), dtype=dtype)
    dst_cols = bb.zeros((3,), dtype=dtype)

    artifact = bb.compile(
        kernel,
        src,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "L_rows_inner:" in text
    assert "L_cols_inner:" in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "global_index_line", "instr"),
    [
        (ptx_multiblock_reduce_add_rowcol_2d_bundle_kernel, "f32", "mad.lo.u32 %r1, %r1, 2, %r0;", "add.rn.f32 %f1, %f1, %f2;"),
        (ptx_multiblock_reduce_or_i32_rowcol_2d_bundle_kernel, "i32", "mad.lo.u32 %r1, %r1, 2, %r0;", "or.b32 %r3, %r3, %r4;"),
    ],
)
def test_ptx_ref_lowers_multiblock_rowcol_reduce_2d_bundle_kernels(
    tmp_path: Path, kernel, dtype: str, global_index_line: str, instr: str
) -> None:
    src = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32") if dtype == "f32" else bb.tensor([[1, 2, 4], [8, 1, 2]], dtype="i32")
    dst_rows = bb.zeros((2,), dtype=dtype)
    dst_cols = bb.zeros((3,), dtype=dtype)

    artifact = bb.compile(
        kernel,
        src,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r0, %tid.x;" in text
    assert "mov.u32 %r1, %ctaid.x;" in text
    assert global_index_line in text
    assert "L_rows_inner:" in text
    assert "L_cols_inner:" in text
    assert instr in text


def test_ptx_ref_lowers_parallel_reduce_add_i32_2d_bundle_kernel(tmp_path: Path) -> None:
    src = bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="i32")
    dst_scalar = bb.zeros((1,), dtype="i32")
    dst_rows = bb.zeros((2,), dtype="i32")
    dst_cols = bb.zeros((3,), dtype="i32")

    artifact = bb.compile(
        ptx_parallel_reduce_add_i32_2d_bundle_kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".shared .align 4 .s32 smem[8];" in text
    assert "bar.sync 0;" in text
    assert "L_scalar_load:" in text
    assert "L_rows_phase:" in text
    assert "L_cols_phase:" in text
    assert "st.global.s32 [%rd2], %r5;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "shared_decl", "instr", "store_line"),
    [
        (ptx_parallel_reduce_mul_2d_bundle_kernel, "f32", ".shared .align 4 .f32 smem[8];", "mul.rn.f32 %f1, %f1, %f3;", "st.global.f32 [%rd2], %f3;"),
        (ptx_parallel_reduce_max_2d_bundle_kernel, "f32", ".shared .align 4 .f32 smem[8];", "max.f32 %f1, %f1, %f3;", "st.global.f32 [%rd2], %f3;"),
        (ptx_parallel_reduce_min_2d_bundle_kernel, "f32", ".shared .align 4 .f32 smem[8];", "min.f32 %f1, %f1, %f3;", "st.global.f32 [%rd2], %f3;"),
        (ptx_parallel_reduce_mul_i32_2d_bundle_kernel, "i32", ".shared .align 4 .s32 smem[8];", "mul.lo.s32 %r4, %r4, %r5;", "st.global.s32 [%rd2], %r5;"),
        (ptx_parallel_reduce_max_i32_2d_bundle_kernel, "i32", ".shared .align 4 .s32 smem[8];", "max.s32 %r4, %r4, %r5;", "st.global.s32 [%rd2], %r5;"),
        (ptx_parallel_reduce_min_i32_2d_bundle_kernel, "i32", ".shared .align 4 .s32 smem[8];", "min.s32 %r4, %r4, %r5;", "st.global.s32 [%rd2], %r5;"),
        (ptx_parallel_reduce_and_i32_2d_bundle_kernel, "i32", ".shared .align 4 .s32 smem[8];", "and.b32 %r4, %r4, %r5;", "st.global.s32 [%rd2], %r5;"),
        (ptx_parallel_reduce_or_i32_2d_bundle_kernel, "i32", ".shared .align 4 .s32 smem[8];", "or.b32 %r4, %r4, %r5;", "st.global.s32 [%rd2], %r5;"),
        (ptx_parallel_reduce_xor_i32_2d_bundle_kernel, "i32", ".shared .align 4 .s32 smem[8];", "xor.b32 %r4, %r4, %r5;", "st.global.s32 [%rd2], %r5;"),
    ],
)
def test_ptx_ref_lowers_other_parallel_reduce_2d_bundle_kernels(
    tmp_path: Path,
    kernel,
    dtype: str,
    shared_decl: str,
    instr: str,
    store_line: str,
) -> None:
    values = [[1, 2, 3], [4, 5, 6]] if dtype == "i32" else [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    src = bb.tensor(values, dtype=dtype)
    dst_scalar = bb.zeros((1,), dtype=dtype)
    dst_rows = bb.zeros((2,), dtype=dtype)
    dst_cols = bb.zeros((3,), dtype=dtype)

    artifact = bb.compile(
        kernel,
        src,
        dst_scalar,
        dst_rows,
        dst_cols,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert shared_decl in text
    assert "bar.sync 0;" in text
    assert "L_scalar_load:" in text
    assert "L_rows_phase:" in text
    assert "L_cols_phase:" in text
    assert instr in text
    assert store_line in text


def test_ptx_ref_lowers_tensor_factory_bundle_kernel(tmp_path: Path) -> None:
    dst_zero = bb.zeros((2, 2), dtype="f32")
    dst_one = bb.zeros((2, 2), dtype="f32")
    dst_full = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        ptx_tensor_factory_bundle_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_tensor_factory_bundle_kernel(" in text
    assert "mov.f32 %f1, 0f00000000;" in text
    assert "mov.f32 %f1, 0f3F800000;" in text
    assert "mov.f32 %f1, 0f40E00000;" in text
    assert "L_zero_loop:" in text
    assert "L_one_loop:" in text
    assert "L_full_loop:" in text
    assert "st.global.f32 [%rd5], %f1;" in text


def test_ptx_ref_lowers_tensor_factory_bundle_i32_kernel(tmp_path: Path) -> None:
    dst_zero = bb.zeros((2, 2), dtype="i32")
    dst_one = bb.zeros((2, 2), dtype="i32")
    dst_full = bb.zeros((2, 2), dtype="i32")

    artifact = bb.compile(
        ptx_tensor_factory_bundle_i32_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_tensor_factory_bundle_i32_kernel(" in text
    assert "mov.s32 %r3, 0;" in text
    assert "mov.s32 %r3, 1;" in text
    assert "mov.s32 %r3, 7;" in text
    assert "L_zero_loop:" in text
    assert "L_one_loop:" in text
    assert "L_full_loop:" in text
    assert "st.global.s32 [%rd5], %r3;" in text


def test_ptx_ref_lowers_parallel_tensor_factory_bundle_kernel(tmp_path: Path) -> None:
    dst_zero = bb.zeros((64, 64), dtype="f32")
    dst_one = bb.zeros((64, 64), dtype="f32")
    dst_full = bb.zeros((64, 64), dtype="f32")

    artifact = bb.compile(
        ptx_parallel_tensor_factory_bundle_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_parallel_tensor_factory_bundle_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 64;" in text
    assert "st.global.f32 [%rd5], %f1;" in text
    assert "st.global.f32 [%rd6], %f2;" in text
    assert "st.global.f32 [%rd7], %f3;" in text


def test_ptx_ref_lowers_parallel_tensor_factory_bundle_i32_kernel(tmp_path: Path) -> None:
    dst_zero = bb.zeros((64, 64), dtype="i32")
    dst_one = bb.zeros((64, 64), dtype="i32")
    dst_full = bb.zeros((64, 64), dtype="i32")

    artifact = bb.compile(
        ptx_parallel_tensor_factory_bundle_i32_kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_parallel_tensor_factory_bundle_i32_kernel(" in text
    assert "mov.u32 %r1, %tid.x;" in text
    assert "setp.ge.u32 %p1, %r1, 64;" in text
    assert "mov.s32 %r8, 0;" in text
    assert "mov.s32 %r9, 1;" in text
    assert "mov.s32 %r10, 7;" in text


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        (ptx_multiblock_dense_copy_2d_kernel, (bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"), bb.zeros((4, 3), dtype="f32"))),
        (ptx_multiblock_dense_copy_i32_2d_kernel, (bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"), bb.zeros((4, 3), dtype="i32"))),
        (ptx_multiblock_dense_copy_2d_kernel, (bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"), bb.zeros((4, 3), dtype="i1"))),
        (ptx_multiblock_dense_add_2d_kernel, (bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"), bb.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]], dtype="f32"), bb.zeros((4, 3), dtype="f32"))),
        (ptx_multiblock_dense_add_i32_2d_kernel, (bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"), bb.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]], dtype="i32"), bb.zeros((4, 3), dtype="i32"))),
        (ptx_multiblock_dense_bitxor_i32_2d_kernel, (bb.tensor([[7, 11, 13], [17, 19, 23], [29, 31, 37], [41, 43, 47]], dtype="i32"), bb.tensor([[3, 5, 9], [6, 10, 12], [15, 17, 19], [21, 22, 23]], dtype="i32"), bb.zeros((4, 3), dtype="i32"))),
        (ptx_multiblock_broadcast_add_2d_kernel, (bb.tensor([[1.0], [2.0], [3.0], [4.0]], dtype="f32"), bb.tensor([[10.0, 20.0, 30.0]], dtype="f32"), bb.zeros((4, 3), dtype="f32"))),
        (ptx_multiblock_broadcast_bitor_i32_2d_kernel, (bb.tensor([[1], [2], [4], [8]], dtype="i32"), bb.tensor([[16, 32, 64]], dtype="i32"), bb.zeros((4, 3), dtype="i32"))),
        (ptx_multiblock_broadcast_bitand_i1_2d_kernel, (bb.tensor([[True], [False], [True], [False]], dtype="i1"), bb.tensor([[True, False, True]], dtype="i1"), bb.zeros((4, 3), dtype="i1"))),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_elementwise_2d_kernels(tmp_path: Path, kernel, args) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "mad.lo.u32 %r1, %r2, 2, %r1;" in text
    assert "add.s32 %r2, %r2, 2;" in text


@pytest.mark.parametrize(
    ("kernel", "args", "instr"),
    [
        (ptx_multiblock_dense_scalar_add_2d_kernel, (bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"), bb.Float32(1.5), bb.zeros((4, 3), dtype="f32")), "add.rn.f32"),
        (ptx_multiblock_dense_scalar_bitand_i32_2d_kernel, (bb.tensor([[7, 11, 13], [17, 19, 23], [29, 31, 37], [41, 43, 47]], dtype="i32"), bb.Int32(5), bb.zeros((4, 3), dtype="i32")), "and.b32"),
        (ptx_multiblock_dense_tensor_scalar_add_2d_kernel, (bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"), bb.tensor([1.5], dtype="f32"), bb.zeros((4, 3), dtype="f32")), "add.rn.f32"),
        (ptx_multiblock_dense_tensor_scalar_bitor_i32_2d_kernel, (bb.tensor([[1, 2, 4], [8, 16, 32], [3, 5, 9], [6, 10, 12]], dtype="i32"), bb.tensor([24], dtype="i32"), bb.zeros((4, 3), dtype="i32")), "or.b32"),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_scalar_broadcast_2d_kernels(tmp_path: Path, kernel, args, instr: str) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text
    assert instr in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_line"),
    [
        (
            ptx_multiblock_dense_cmp_lt_f32_2d_kernel,
            (
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.tensor([[2.0, 1.0, 4.0], [4.0, 6.0, 5.0], [8.0, 7.0, 9.0], [11.0, 10.0, 13.0]], dtype="f32"),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            "setp.lt.f32 %p3, %f1, %f2;",
        ),
        (
            ptx_multiblock_broadcast_cmp_eq_i32_2d_kernel,
            (
                bb.tensor([[1], [2], [4], [8]], dtype="i32"),
                bb.tensor([[1, 3, 1]], dtype="i32"),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            "setp.eq.s32 %p3, %r11, %r12;",
        ),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_compare_2d_kernels(tmp_path: Path, kernel, args, expected_line: str) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text
    assert expected_line in text
    assert "st.global.u8 [%rd7], %r11;" in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_line"),
    [
        (
            ptx_multiblock_dense_scalar_cmp_lt_f32_2d_kernel,
            (
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.Float32(8.5),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            "setp.lt.f32 %p3, %f1, %f2;",
        ),
        (
            ptx_multiblock_dense_tensor_scalar_cmp_eq_i32_2d_kernel,
            (
                bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"),
                bb.tensor([8], dtype="i32"),
                bb.tensor([[False, False, False], [False, False, False], [False, False, False], [False, False, False]], dtype="i1"),
            ),
            "setp.eq.s32 %p3, %r5, %r3;",
        ),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_scalar_compare_2d_kernels(
    tmp_path: Path, kernel, args, expected_line: str
) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text
    assert expected_line in text
    assert ("st.global.u8 [%rd5], %r6;" in text) or ("st.global.u8 [%rd6], %r6;" in text)


@pytest.mark.parametrize(
    ("kernel", "args", "expected_line"),
    [
        (
            ptx_multiblock_dense_select_f32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0], [100.0, 110.0, 120.0]], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
            "selp.f32 %f3, %f1, %f2, %p2;",
        ),
        (
            ptx_multiblock_broadcast_select_i32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1], [2], [4], [8]], dtype="i32"),
                bb.tensor([[1, 3, 5]], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            "selp.b32 %r6, %r6, %r7, %p2;",
        ),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_select_2d_kernels(tmp_path: Path, kernel, args, expected_line: str) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text
    assert expected_line in text
    assert ("st.global.f32 [%rd8], %f3;" in text) or ("st.global.s32 [%rd8], %r6;" in text)


@pytest.mark.parametrize(
    ("kernel", "args", "expected_line"),
    [
        (
            ptx_multiblock_dense_scalar_select_f32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="f32"),
                bb.Float32(8.5),
                bb.zeros((4, 3), dtype="f32"),
            ),
            "selp.f32 %f3, %f1, %f2, %p2;",
        ),
        (
            ptx_multiblock_dense_tensor_scalar_select_i32_2d_kernel,
            (
                bb.tensor([[True, False, True], [False, True, False], [True, True, False], [False, False, True]], dtype="i1"),
                bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="i32"),
                bb.tensor([8], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            "selp.b32 %r6, %r7, %r6, %p2;",
        ),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_scalar_select_2d_kernels(
    tmp_path: Path, kernel, args, expected_line: str
) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text
    assert expected_line in text
    assert ("st.global.f32 [%rd8], %f3;" in text) or ("st.global.s32 [%rd8], %r6;" in text)


@pytest.mark.parametrize(
    ("kernel", "dtype", "src_values", "expected_lines"),
    [
        (ptx_multiblock_dense_sqrt_2d_kernel, "f32", [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0], [49.0, 64.0, 81.0], [100.0, 121.0, 144.0]], ("sqrt.rn.f32 %f2, %f1;",)),
        (ptx_multiblock_dense_sin_2d_kernel, "f32", [[0.0, 0.5, 1.0], [1.5, 0.25, 0.75], [1.25, 0.1, 0.9], [0.2, 0.4, 0.6]], ("sin.approx.f32 %f2, %f1;",)),
        (ptx_multiblock_dense_cos_2d_kernel, "f32", [[0.0, 0.5, 1.0], [1.5, 0.25, 0.75], [1.25, 0.1, 0.9], [0.2, 0.4, 0.6]], ("cos.approx.f32 %f2, %f1;",)),
        (ptx_multiblock_dense_exp2_2d_kernel, "f32", [[0.0, 1.0, 2.0], [3.0, 0.5, 1.5], [2.5, 0.25, 1.25], [1.75, 2.25, 0.75]], ("ex2.approx.f32 %f2, %f1;",)),
        (
            ptx_multiblock_dense_exp_2d_kernel,
            "f32",
            [[0.0, 0.5, 1.0], [1.5, 0.25, 0.75], [1.25, 0.1, 0.9], [0.2, 0.4, 0.6]],
            ("mov.f32 %f2, 0f3FB8AA3B;", "mul.rn.f32 %f1, %f1, %f2;", "ex2.approx.f32 %f2, %f1;"),
        ),
        (ptx_multiblock_dense_log2_2d_kernel, "f32", [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0], [512.0, 1024.0, 2048.0]], ("lg2.approx.f32 %f2, %f1;",)),
        (
            ptx_multiblock_dense_log_2d_kernel,
            "f32",
            [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0], [512.0, 1024.0, 2048.0]],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3F317218;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (
            ptx_multiblock_dense_log10_2d_kernel,
            "f32",
            [[1.0, 10.0, 100.0], [1000.0, 1.0, 10.0], [100.0, 1000.0, 1.0], [10.0, 100.0, 1000.0]],
            ("lg2.approx.f32 %f2, %f1;", "mov.f32 %f1, 0f3E9A209B;", "mul.rn.f32 %f2, %f2, %f1;"),
        ),
        (
            ptx_multiblock_dense_floor_2d_kernel,
            "f32",
            [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]],
            ("cvt.rmi.f32.f32 %f2, %f1;",),
        ),
        (
            ptx_multiblock_dense_ceil_2d_kernel,
            "f32",
            [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]],
            ("cvt.rpi.f32.f32 %f2, %f1;",),
        ),
        (
            ptx_multiblock_dense_trunc_2d_kernel,
            "f32",
            [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]],
            ("cvt.rzi.f32.f32 %f2, %f1;",),
        ),
        (
            ptx_multiblock_dense_atan_2d_kernel,
            "f32",
            [[-1.5, -0.75, 0.0], [0.75, 1.0, 1.25], [-1.25, -0.5, 0.5], [1.5, 0.25, -0.25]],
            ("rcp.approx.f32 %f4, %f2;", "selp.f32 %f2, %f4, %f2, %p0;", "selp.f32 %f2, %f5, %f3, %p0;"),
        ),
        (
            ptx_multiblock_dense_asin_2d_kernel,
            "f32",
            [[-0.875, -0.625, -0.375], [-0.125, 0.0, 0.125], [0.375, 0.625, 0.875], [-0.75, 0.5, -0.25]],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "selp.f32 %f3, %f8, %f3, %p2;"),
        ),
        (
            ptx_multiblock_dense_acos_2d_kernel,
            "f32",
            [[-0.875, -0.625, -0.375], [-0.125, 0.0, 0.125], [0.375, 0.625, 0.875], [-0.75, 0.5, -0.25]],
            ("max.f32 %f2, %f2, %f3;", "sqrt.rn.f32 %f2, %f2;", "abs.f32 %f4, %f2;", "abs.f32 %f5, %f1;"),
        ),
        (
            ptx_multiblock_dense_erf_2d_kernel,
            "f32",
            [[-1.5, -0.75, 0.0], [0.75, 1.0, 1.25], [-1.25, -0.5, 0.5], [1.5, 0.25, -0.25]],
            ("rcp.approx.f32 %f3, %f3;", "ex2.approx.f32 %f5, %f5;", "selp.f32 %f2, %f5, %f4, %p0;"),
        ),
        (ptx_multiblock_dense_round_2d_kernel, "f32", [[-3.75, -2.25, -1.75], [-0.25, 0.25, 1.75], [2.25, 3.75, -1.25], [0.75, -0.75, 1.25]], ("cvt.rni.f32.f32 %f2, %f1;",)),
        (ptx_multiblock_dense_neg_f32_2d_kernel, "f32", [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], ("neg.f32 %f2, %f1;",)),
        (ptx_multiblock_dense_neg_i32_2d_kernel, "i32", [[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, -12]], ("neg.s32 %r5, %r5;",)),
        (ptx_multiblock_dense_abs_f32_2d_kernel, "f32", [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], ("abs.f32 %f2, %f1;",)),
        (ptx_multiblock_dense_abs_i32_2d_kernel, "i32", [[1, -2, 3], [-4, 5, -6], [7, -8, 9], [-10, 11, -12]], ("abs.s32 %r5, %r5;",)),
        (ptx_multiblock_dense_bitnot_i32_2d_kernel, "i32", [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], ("not.b32 %r5, %r5;",)),
        (ptx_multiblock_dense_bitnot_i1_2d_kernel, "i1", [[True, False, True], [False, True, False], [True, True, False], [False, False, True]], ("xor.b32 %r5, %r5, 1;",)),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_unary_2d_kernel(tmp_path: Path, kernel, dtype: str, src_values, expected_lines: tuple[str, ...]) -> None:
    src = bb.tensor(src_values, dtype=dtype)
    dst = bb.zeros((4, 3), dtype=dtype)

    artifact = bb.compile(
        kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    for line in expected_lines:
        assert line in text
    assert "add.s32 %r2, %r2, 2;" in text


@pytest.mark.parametrize(
    ("kernel", "dtype", "expected_line"),
    [
        (ptx_multiblock_dense_copy_reduce_add_2d_kernel, "f32", "add.rn.f32"),
        (ptx_multiblock_dense_copy_reduce_or_i32_2d_kernel, "i32", "or.b32"),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_copy_reduce_2d_kernels(
    tmp_path: Path, kernel, dtype: str, expected_line: str
) -> None:
    src = bb.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=dtype)
    dst_values = [[1.0, 1.0, 1.0]] * 4 if dtype == "f32" else [[1, 1, 1]] * 4
    dst = bb.tensor(dst_values, dtype=dtype)

    artifact = bb.compile(kernel, src, dst, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text
    assert expected_line in text


@pytest.mark.parametrize(
    ("kernel", "dtype"),
    [
        (ptx_multiblock_tensor_factory_bundle_kernel, "f32"),
        (ptx_multiblock_tensor_factory_bundle_i32_kernel, "i32"),
    ],
)
def test_ptx_ref_lowers_multiblock_parallel_tensor_factory_bundle_kernels(tmp_path: Path, kernel, dtype: str) -> None:
    dst_zero = bb.zeros((4, 3), dtype=dtype)
    dst_one = bb.zeros((4, 3), dtype=dtype)
    dst_full = bb.zeros((4, 3), dtype=dtype)

    artifact = bb.compile(
        kernel,
        dst_zero,
        dst_one,
        dst_full,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text


def test_compile_auto_falls_back_to_ptx_ref_for_nvidia_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: False)
    src = bb.tensor([1.0] * 256, dtype="f32")
    dst = bb.zeros((256,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
    )

    assert artifact.backend_name == "ptx_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "ptx"


def test_ptx_ref_rejects_unsupported_math_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 8, dtype="f16")
    dst = bb.zeros((8,), dtype="f16")

    with pytest.raises(bb.CompilationError, match="ptx_ref currently supports only exact rank-1 dense copy"):
        bb.compile(
            ptx_unsupported_math_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_nvidia_target(),
            backend="ptx_ref",
        )


def test_ptx_ref_generated_module_loads_in_cuda_driver_if_available(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 256, dtype="f32")
    dst = bb.zeros((256,), dtype="f32")
    artifact = bb.compile(
        ptx_indexed_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    try:
        driver = CudaDriver()
    except (bb.BackendNotImplementedError, CudaDriverError) as exc:
        pytest.skip(f"CUDA driver not usable on this host: {exc}")

    module = driver.load_module_from_ptx(artifact.lowered_module.text)
    try:
        function = driver.function(module, artifact.lowered_module.entry_point)
        assert int(function.value or 0) != 0
    finally:
        driver.unload_module(module)
        driver.release_primary_context(0)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_max_f32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = bb.maximum(a[idx], b[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_min_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = bb.minimum(a[tidx], b[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_scalar_max_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.maximum(src[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_tensor_scalar_min_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.minimum(src[tidx], alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_dense_max_f32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.maximum(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def ptx_broadcast_min_i32_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.minimum(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_parallel_dense_scalar_max_f32_2d_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.maximum(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(2, 1, 1)))
def ptx_multiblock_dense_tensor_scalar_min_i32_2d_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.minimum(src.load(), alpha[0]))


@pytest.mark.parametrize(
    ("kernel", "args", "expected_instr"),
    [
        (
            ptx_indexed_max_f32_kernel,
            (
                bb.tensor([1.0, -2.5, 3.0], dtype="f32"),
                bb.tensor([0.5, -3.0, 4.0], dtype="f32"),
                bb.zeros((3,), dtype="f32"),
            ),
            "max.f32",
        ),
        (
            ptx_direct_min_i32_kernel,
            (
                bb.tensor([7, 11, 13, 17], dtype="i32"),
                bb.tensor([3, 15, 9, 21], dtype="i32"),
                bb.zeros((4,), dtype="i32"),
            ),
            "min.s32",
        ),
    ],
)
def test_ptx_ref_lowers_extrema_rank1_binary_kernels(tmp_path: Path, kernel, args, expected_instr: str) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert expected_instr in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_instr"),
    [
        (
            ptx_indexed_scalar_max_f32_kernel,
            (
                bb.tensor([1.0, -2.5, 3.0], dtype="f32"),
                bb.Float32(2.0),
                bb.zeros((3,), dtype="f32"),
            ),
            "max.f32",
        ),
        (
            ptx_direct_tensor_scalar_min_i32_kernel,
            (
                bb.tensor([7, 11, 13, 17], dtype="i32"),
                bb.tensor([10], dtype="i32"),
                bb.zeros((4,), dtype="i32"),
            ),
            "min.s32",
        ),
    ],
)
def test_ptx_ref_lowers_extrema_rank1_scalar_broadcast_kernels(tmp_path: Path, kernel, args, expected_instr: str) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert expected_instr in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_instr"),
    [
        (
            ptx_dense_max_f32_2d_kernel,
            (
                bb.tensor([[1.0, -2.5], [7.0, 4.25]], dtype="f32"),
                bb.tensor([[0.5, -3.0], [8.0, 2.0]], dtype="f32"),
                bb.zeros((2, 2), dtype="f32"),
            ),
            "max.f32",
        ),
        (
            ptx_broadcast_min_i32_2d_kernel,
            (
                bb.tensor([[7], [13]], dtype="i32"),
                bb.tensor([[3, 15, 9]], dtype="i32"),
                bb.zeros((2, 3), dtype="i32"),
            ),
            "min.s32",
        ),
    ],
)
def test_ptx_ref_lowers_extrema_tensor_binary_2d_kernels(tmp_path: Path, kernel, args, expected_instr: str) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert expected_instr in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_instr"),
    [
        (
            ptx_parallel_dense_scalar_max_f32_2d_kernel,
            (
                bb.tensor([[1.0, -2.5, 3.0], [0.5, 7.0, 4.25]], dtype="f32"),
                bb.Float32(2.0),
                bb.zeros((2, 3), dtype="f32"),
            ),
            "max.f32",
        ),
        (
            ptx_multiblock_dense_tensor_scalar_min_i32_2d_kernel,
            (
                bb.tensor([[7, 11, 13], [17, 19, 23], [29, 31, 37], [41, 43, 47]], dtype="i32"),
                bb.tensor([24], dtype="i32"),
                bb.zeros((4, 3), dtype="i32"),
            ),
            "min.s32",
        ),
    ],
)
def test_ptx_ref_lowers_extrema_tensor_scalar_broadcast_2d_kernels(
    tmp_path: Path,
    kernel,
    args,
    expected_instr: str,
) -> None:
    artifact = bb.compile(kernel, *args, cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert expected_instr in text


def _atan2_vector_values() -> tuple[list[float], list[float]]:
    y_values = [0.5 * float((index % 16) - 8) for index in range(128)]
    x_pattern = [1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5, 0.25]
    x_values = [x_pattern[index % len(x_pattern)] for index in range(128)]
    return y_values, x_values


@pytest.mark.parametrize(
    ("kernel", "args", "expected_lines"),
    [
        (
            ptx_indexed_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values()[0], dtype="f32"),
                bb.tensor(_atan2_vector_values()[1], dtype="f32"),
                bb.zeros((128,), dtype="f32"),
            ),
            ("sub.rn.f32 %f8, %f11, %f3;", "selp.f32 %f3, %f10, %f3, %p1;"),
        ),
        (
            ptx_direct_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values()[0], dtype="f32"),
                bb.tensor(_atan2_vector_values()[1], dtype="f32"),
                bb.zeros((128,), dtype="f32"),
            ),
            ("sub.rn.f32 %f8, %f11, %f3;", "selp.f32 %f3, %f10, %f3, %p1;"),
        ),
    ],
)
def test_ptx_ref_lowers_rank1_atan2_binary_kernels(tmp_path: Path, kernel, args, expected_lines: tuple[str, ...]) -> None:
    artifact = bb.compile(*((kernel,) + args()), cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "abs.f32 %f4" in text
    assert "and.pred %p0, %p1, %p2;" in text
    for line in expected_lines:
        assert line in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_load"),
    [
        (
            ptx_indexed_scalar_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values()[0], dtype="f32"),
                bb.Float32(-1.25),
                bb.zeros((128,), dtype="f32"),
            ),
            "ld.param.f32 %f2, [alpha_param];",
        ),
        (
            ptx_direct_tensor_scalar_atan2_kernel,
            lambda: (
                bb.tensor(_atan2_vector_values()[0], dtype="f32"),
                bb.tensor([-1.25], dtype="f32"),
                bb.zeros((128,), dtype="f32"),
            ),
            "ld.global.f32 %f2, [%rd2];",
        ),
    ],
)
def test_ptx_ref_lowers_rank1_atan2_scalar_broadcast_kernels(tmp_path: Path, kernel, args, expected_load: str) -> None:
    artifact = bb.compile(*((kernel,) + args()), cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert expected_load in text
    assert "selp.f32 %f3, %f10, %f3, %p1;" in text
    assert "st.global.f32" in text


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        (
            ptx_dense_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5]], dtype="f32"),
                bb.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.25]], dtype="f32"),
                bb.zeros((2, 3), dtype="f32"),
            ),
        ),
        (
            ptx_broadcast_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0], [-1.0]], dtype="f32"),
                bb.tensor([[1.0, -1.0, 0.0]], dtype="f32"),
                bb.zeros((2, 3), dtype="f32"),
            ),
        ),
    ],
)
def test_ptx_ref_lowers_serial_tensor_atan2_2d_kernels(tmp_path: Path, kernel, args) -> None:
    artifact = bb.compile(*((kernel,) + args()), cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "L_rows_outer:" in text
    assert "L_cols_inner:" in text
    assert "sub.rn.f32 %f8, %f11, %f3;" in text
    assert "selp.f32 %f3, %f10, %f3, %p1;" in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_load"),
    [
        (
            ptx_dense_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5]], dtype="f32"),
                bb.Float32(-1.25),
                bb.zeros((2, 3), dtype="f32"),
            ),
            "ld.param.f32 %f2, [alpha_param];",
        ),
        (
            ptx_dense_tensor_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5]], dtype="f32"),
                bb.tensor([-1.25], dtype="f32"),
                bb.zeros((2, 3), dtype="f32"),
            ),
            "ld.global.f32 %f2, [%rd2];",
        ),
    ],
)
def test_ptx_ref_lowers_serial_tensor_scalar_atan2_2d_kernels(
    tmp_path: Path, kernel, args, expected_load: str
) -> None:
    artifact = bb.compile(*((kernel,) + args()), cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert expected_load in text
    assert "selp.f32 %f3, %f10, %f3, %p1;" in text


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        (
            ptx_multiblock_dense_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5], [-0.5, 0.75, -1.25], [1.5, -1.5, 2.0]], dtype="f32"),
                bb.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.25], [0.5, -0.75, 1.25], [2.0, -2.0, -0.5]], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
        ),
        (
            ptx_multiblock_broadcast_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0], [-1.0], [0.5], [-0.5]], dtype="f32"),
                bb.tensor([[1.0, -1.0, 0.0]], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
        ),
    ],
)
def test_ptx_ref_lowers_multiblock_tensor_atan2_2d_kernels(tmp_path: Path, kernel, args) -> None:
    artifact = bb.compile(*((kernel,) + args()), cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert "add.s32 %r2, %r2, 2;" in text
    assert "selp.f32 %f3, %f10, %f3, %p1;" in text


@pytest.mark.parametrize(
    ("kernel", "args", "expected_load"),
    [
        (
            ptx_multiblock_dense_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5], [-0.5, 0.75, -1.25], [1.5, -1.5, 2.0]], dtype="f32"),
                bb.Float32(-1.25),
                bb.zeros((4, 3), dtype="f32"),
            ),
            "ld.param.f32 %f2, [alpha_param];",
        ),
        (
            ptx_multiblock_dense_tensor_scalar_atan2_2d_kernel,
            lambda: (
                bb.tensor([[1.0, -1.0, 0.0], [2.0, -2.0, 0.5], [-0.5, 0.75, -1.25], [1.5, -1.5, 2.0]], dtype="f32"),
                bb.tensor([-1.25], dtype="f32"),
                bb.zeros((4, 3), dtype="f32"),
            ),
            "ld.global.f32 %f2, [%rd2];",
        ),
    ],
)
def test_ptx_ref_lowers_multiblock_tensor_scalar_atan2_2d_kernels(
    tmp_path: Path, kernel, args, expected_load: str
) -> None:
    artifact = bb.compile(*((kernel,) + args()), cache_dir=tmp_path, target=_nvidia_target(), backend="ptx_ref")
    text = artifact.lowered_module.text
    assert "mov.u32 %r2, %ctaid.x;" in text
    assert "mov.u32 %r2, %ctaid.y;" in text
    assert expected_load in text
    assert "selp.f32 %f3, %f10, %f3, %p1;" in text
