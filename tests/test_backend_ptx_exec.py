from __future__ import annotations

from pathlib import Path

import ctypes
import pytest

import baybridge as bb
from baybridge.cuda_driver import CudaDriver, CudaDriverError


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


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_add_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] + b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_mul_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    c[tidx] = a[tidx] * b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_scalar_param_broadcast_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_tensor_scalar_broadcast_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + alpha[0]


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


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_sqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.sqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def ptx_exec_direct_rsqrt_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.rsqrt(src[tidx])


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


@pytest.mark.parametrize(
    ("kernel", "src", "dtype", "expected"),
    [
        (ptx_exec_parallel_reduce_mul_kernel, [1.0, 2.0, 3.0, 4.0], "f32", pytest.approx([24.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_max_kernel, [1.0, 9.0, 3.0, 7.0], "f32", pytest.approx([9.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_min_kernel, [1.0, 9.0, 3.0, 7.0], "f32", pytest.approx([1.0], rel=1e-6, abs=1e-6)),
        (ptx_exec_parallel_reduce_mul_i32_kernel, [1, 2, 3, 4], "i32", [24]),
        (ptx_exec_parallel_reduce_max_i32_kernel, [1, 9, 3, 7], "i32", [9]),
        (ptx_exec_parallel_reduce_min_i32_kernel, [1, 9, 3, 7], "i32", [1]),
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


def test_compile_auto_prefers_ptx_exec_for_nvidia_target_if_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("baybridge.compiler.PtxExecBackend.available", lambda self, target=None: True)

    src = bb.tensor([1.0] * 128, dtype="f32")
    dst = bb.zeros((128,), dtype="f32")

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
