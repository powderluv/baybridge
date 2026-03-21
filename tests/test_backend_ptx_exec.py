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
