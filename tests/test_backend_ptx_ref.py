from __future__ import annotations

from pathlib import Path

import pytest

import baybridge as bb
from baybridge.cuda_driver import CudaDriver, CudaDriverError


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


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_add_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] + b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_mul_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] * b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_scalar_param_broadcast_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_tensor_scalar_broadcast_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha[0]


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


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_sqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.sqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_direct_rsqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.rsqrt(src[tidx])


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
    src = bb.tensor([1.0] * 8, dtype="f32")
    dst = bb.zeros((8,), dtype="f32")

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
