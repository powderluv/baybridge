from __future__ import annotations

from pathlib import Path

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


def _skip_if_cuda_driver_unavailable() -> None:
    try:
        driver = CudaDriver()
    except (bb.BackendNotImplementedError, CudaDriverError) as exc:
        pytest.skip(f"CUDA driver not usable on this host: {exc}")
    if driver.device_count() < 1:
        pytest.skip("no NVIDIA devices are visible to the CUDA driver")


def _target() -> bb.NvidiaTarget:
    return bb.NvidiaTarget(sm="sm_80", ptx_version="8.0")


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
