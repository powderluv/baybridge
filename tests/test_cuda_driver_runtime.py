from __future__ import annotations

import ctypes

import pytest

from baybridge.cuda_driver import (
    CUcontext,
    CUdevice,
    CUfunction,
    CUmodule,
    CudaDriver,
    CudaDriverError,
    load_cuda_driver_library,
)
from baybridge.diagnostics import BackendNotImplementedError


def _reset_cuda_loader(monkeypatch) -> None:
    monkeypatch.setattr("baybridge.cuda_driver._CUDA_DRIVER_LIBRARY_HANDLE", None)


def _write_int_ptr(out, value: int) -> None:
    ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = value


def _write_char_pp(out, value: bytes) -> None:
    ctypes.cast(out, ctypes.POINTER(ctypes.c_char_p))[0] = value


def _write_device_ptr(out, value: int) -> None:
    ctypes.cast(out, ctypes.POINTER(CUdevice))[0] = CUdevice(value)


def _write_context_ptr(out, value: int) -> None:
    ctypes.cast(out, ctypes.POINTER(CUcontext))[0] = CUcontext(value)


def _write_module_ptr(out, value: int) -> None:
    ctypes.cast(out, ctypes.POINTER(CUmodule))[0] = CUmodule(value)


def _write_function_ptr(out, value: int) -> None:
    ctypes.cast(out, ctypes.POINTER(CUfunction))[0] = CUfunction(value)


def test_load_cuda_driver_library_raises_when_missing(monkeypatch) -> None:
    _reset_cuda_loader(monkeypatch)
    monkeypatch.setattr("baybridge.cuda_driver._cuda_library_candidates", lambda: ["missing-a", "missing-b"])

    def fake_cdll(path: str, mode: int = 0):
        raise OSError(f"cannot open {path}")

    monkeypatch.setattr("baybridge.cuda_driver.ctypes.CDLL", fake_cdll)

    with pytest.raises(BackendNotImplementedError, match="libcuda.so.1"):
        load_cuda_driver_library()


def test_cuda_driver_fake_module_load_and_launch(monkeypatch) -> None:
    _reset_cuda_loader(monkeypatch)

    class FakeLibrary:
        def __init__(self) -> None:
            self.loaded_ptx = ""
            self.launches: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
            self.context = 0xCAFE

            def cuInit(flags: int) -> int:
                return 0

            def cuDriverGetVersion(out) -> int:
                _write_int_ptr(out, 13020)
                return 0

            def cuGetErrorName(status: int, out) -> int:
                _write_char_pp(out, b"CUDA_ERROR_UNKNOWN")
                return 0

            def cuGetErrorString(status: int, out) -> int:
                _write_char_pp(out, b"unknown")
                return 0

            def cuDeviceGetCount(out) -> int:
                _write_int_ptr(out, 1)
                return 0

            def cuDeviceGet(out, ordinal: int) -> int:
                _write_device_ptr(out, ordinal)
                return 0

            def cuDeviceGetName(buffer, size: int, device: CUdevice) -> int:
                ctypes.memmove(buffer, b"Fake NVIDIA\0", len(b"Fake NVIDIA\0"))
                return 0

            def cuDeviceComputeCapability(major, minor, device: CUdevice) -> int:
                _write_int_ptr(major, 9)
                _write_int_ptr(minor, 0)
                return 0

            def cuCtxGetCurrent(out) -> int:
                _write_context_ptr(out, self.context)
                return 0

            def cuCtxSetCurrent(context: CUcontext) -> int:
                self.context = int(context.value or 0)
                return 0

            def cuCtxSynchronize() -> int:
                return 0

            def cuDevicePrimaryCtxRetain(out, device: CUdevice) -> int:
                self.context = 0xCAFE
                _write_context_ptr(out, self.context)
                return 0

            def cuDevicePrimaryCtxRelease(device: CUdevice) -> int:
                return 0

            def cuModuleLoadDataEx(out, image, num_options: int, options, option_values) -> int:
                self.loaded_ptx = ctypes.string_at(image).decode("utf-8")
                _write_module_ptr(out, 0x1234)
                return 0

            def cuModuleUnload(module: CUmodule) -> int:
                return 0

            def cuModuleGetFunction(out, module: CUmodule, name: bytes) -> int:
                assert name == b"noop_kernel"
                _write_function_ptr(out, 0x5678)
                return 0

            def cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared_mem, stream, kernel_params, extra) -> int:
                self.launches.append(
                    (
                        (int(gx.value), int(gy.value), int(gz.value)),
                        (int(bx.value), int(by.value), int(bz.value)),
                    )
                )
                return 0

            self.cuInit = cuInit
            self.cuDriverGetVersion = cuDriverGetVersion
            self.cuGetErrorName = cuGetErrorName
            self.cuGetErrorString = cuGetErrorString
            self.cuDeviceGetCount = cuDeviceGetCount
            self.cuDeviceGet = cuDeviceGet
            self.cuDeviceGetName = cuDeviceGetName
            self.cuDeviceComputeCapability = cuDeviceComputeCapability
            self.cuCtxGetCurrent = cuCtxGetCurrent
            self.cuCtxSetCurrent = cuCtxSetCurrent
            self.cuCtxSynchronize = cuCtxSynchronize
            self.cuDevicePrimaryCtxRetain = cuDevicePrimaryCtxRetain
            self.cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease
            self.cuModuleLoadDataEx = cuModuleLoadDataEx
            self.cuModuleUnload = cuModuleUnload
            self.cuModuleGetFunction = cuModuleGetFunction
            self.cuLaunchKernel = cuLaunchKernel

    fake_library = FakeLibrary()
    monkeypatch.setattr("baybridge.cuda_driver.load_cuda_driver_library", lambda global_scope=False: fake_library)

    driver = CudaDriver()
    module = driver.load_module_from_ptx(
        ".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry noop_kernel() { ret; }\n"
    )
    function = driver.function(module, "noop_kernel")
    driver.launch_kernel(function, grid=(1, 1, 1), block=(1, 1, 1))
    driver.synchronize()
    driver.unload_module(module)

    assert "noop_kernel" in fake_library.loaded_ptx
    assert fake_library.launches == [((1, 1, 1), (1, 1, 1))]
    assert driver.device_info(0).name == "Fake NVIDIA"
    assert driver.device_info(0).compute_capability == (9, 0)
    assert driver.driver_version() == 13020


def test_cuda_driver_error_includes_name_and_string(monkeypatch) -> None:
    _reset_cuda_loader(monkeypatch)

    class FakeLibrary:
        def __init__(self) -> None:
            def cuInit(flags: int) -> int:
                return 1

            def cuDriverGetVersion(out) -> int:
                _write_int_ptr(out, 0)
                return 0

            def cuGetErrorName(status: int, out) -> int:
                _write_char_pp(out, b"CUDA_ERROR_INVALID_VALUE")
                return 0

            def cuGetErrorString(status: int, out) -> int:
                _write_char_pp(out, b"invalid value")
                return 0

            def cuDeviceGetCount(out) -> int:
                _write_int_ptr(out, 0)
                return 0

            def cuDeviceGet(out, ordinal: int) -> int:
                return 0

            def cuDeviceGetName(buffer, size: int, device: CUdevice) -> int:
                return 0

            def cuDeviceComputeCapability(major, minor, device: CUdevice) -> int:
                return 0

            def cuCtxGetCurrent(out) -> int:
                return 0

            def cuCtxSetCurrent(context: CUcontext) -> int:
                return 0

            def cuCtxSynchronize() -> int:
                return 0

            def cuDevicePrimaryCtxRetain(out, device: CUdevice) -> int:
                return 0

            def cuDevicePrimaryCtxRelease(device: CUdevice) -> int:
                return 0

            def cuModuleLoadDataEx(out, image, num_options: int, options, option_values) -> int:
                return 0

            def cuModuleUnload(module: CUmodule) -> int:
                return 0

            def cuModuleGetFunction(out, module: CUmodule, name: bytes) -> int:
                return 0

            def cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared_mem, stream, kernel_params, extra) -> int:
                return 0

            self.cuInit = cuInit
            self.cuDriverGetVersion = cuDriverGetVersion
            self.cuGetErrorName = cuGetErrorName
            self.cuGetErrorString = cuGetErrorString
            self.cuDeviceGetCount = cuDeviceGetCount
            self.cuDeviceGet = cuDeviceGet
            self.cuDeviceGetName = cuDeviceGetName
            self.cuDeviceComputeCapability = cuDeviceComputeCapability
            self.cuCtxGetCurrent = cuCtxGetCurrent
            self.cuCtxSetCurrent = cuCtxSetCurrent
            self.cuCtxSynchronize = cuCtxSynchronize
            self.cuDevicePrimaryCtxRetain = cuDevicePrimaryCtxRetain
            self.cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease
            self.cuModuleLoadDataEx = cuModuleLoadDataEx
            self.cuModuleUnload = cuModuleUnload
            self.cuModuleGetFunction = cuModuleGetFunction
            self.cuLaunchKernel = cuLaunchKernel

    monkeypatch.setattr("baybridge.cuda_driver.load_cuda_driver_library", lambda global_scope=False: FakeLibrary())

    with pytest.raises(CudaDriverError, match="CUDA_ERROR_INVALID_VALUE"):
        CudaDriver()


def test_cuda_driver_can_load_and_launch_noop_ptx_if_available() -> None:
    try:
        driver = CudaDriver()
    except BackendNotImplementedError:
        pytest.skip("libcuda.so.1 is not available on this host")
    except CudaDriverError as exc:
        pytest.skip(f"CUDA driver is present but not usable: {exc}")

    if driver.device_count() < 1:
        pytest.skip("no NVIDIA devices are visible to the CUDA driver")

    device = driver.device(0)
    major, minor = driver.device_compute_capability(device)
    module = driver.load_module_from_ptx(
        (
            ".version 8.0\n"
            ".target sm_80\n"
            ".address_size 64\n"
            ".visible .entry noop_kernel()\n"
            "{\n"
            "    ret;\n"
            "}\n"
        )
    )
    try:
        function = driver.function(module, "noop_kernel")
        driver.launch_kernel(function, grid=(1, 1, 1), block=(1, 1, 1))
        driver.synchronize()
        assert driver.device_name(device)
    finally:
        driver.unload_module(module)
        driver.release_primary_context(device)
