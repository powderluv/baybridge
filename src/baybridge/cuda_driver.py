from __future__ import annotations

import ctypes
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .diagnostics import BackendNotImplementedError

_CUDA_SUCCESS = 0
_CUDA_DRIVER_LIBRARY_HANDLE: ctypes.CDLL | None = None


class CUdevice(ctypes.c_int):
    pass


class CUcontext(ctypes.c_void_p):
    pass


class CUmodule(ctypes.c_void_p):
    pass


class CUfunction(ctypes.c_void_p):
    pass


class CUstream(ctypes.c_void_p):
    pass


@dataclass(frozen=True)
class CudaDeviceInfo:
    ordinal: int
    name: str
    compute_capability: tuple[int, int]


class CudaDriverError(RuntimeError):
    def __init__(self, op: str, status: int, name: str | None = None, message: str | None = None) -> None:
        self.op = op
        self.status = int(status)
        self.name = name
        self.message = message
        detail = f"{op} failed with CUDA status {status}"
        if name:
            detail += f" ({name})"
        if message:
            detail += f": {message}"
        super().__init__(detail)


def _cuda_library_candidates() -> list[str]:
    candidates: list[str] = []
    discovered = find_library("cuda")
    if discovered:
        candidates.append(discovered)
    for candidate in (
        "/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/lib64/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so.1",
        "/lib/wsl/lib/libcuda.so.1",
        "libcuda.so.1",
        "libcuda.so",
    ):
        if candidate.startswith("/"):
            if not Path(candidate).exists():
                continue
        candidates.append(candidate)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _cuda_driver_has_device(handle: ctypes.CDLL) -> bool:
    try:
        handle.cuInit.argtypes = [ctypes.c_uint]
        handle.cuInit.restype = ctypes.c_int
        handle.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        handle.cuDeviceGetCount.restype = ctypes.c_int
    except AttributeError:
        return False
    if handle.cuInit(0) != _CUDA_SUCCESS:
        return False
    count = ctypes.c_int()
    status = handle.cuDeviceGetCount(ctypes.byref(count))
    return status == _CUDA_SUCCESS and count.value >= 1


def load_cuda_driver_library(*, global_scope: bool = False) -> ctypes.CDLL:
    global _CUDA_DRIVER_LIBRARY_HANDLE
    if _CUDA_DRIVER_LIBRARY_HANDLE is not None:
        return _CUDA_DRIVER_LIBRARY_HANDLE
    mode = getattr(ctypes, "RTLD_GLOBAL", 0) if global_scope else getattr(ctypes, "RTLD_LOCAL", 0)
    fallback_handle: ctypes.CDLL | None = None
    for candidate in _cuda_library_candidates():
        try:
            handle = ctypes.CDLL(candidate, mode=mode)
        except OSError:
            continue
        if fallback_handle is None:
            fallback_handle = handle
        if _cuda_driver_has_device(handle):
            _CUDA_DRIVER_LIBRARY_HANDLE = handle
            return _CUDA_DRIVER_LIBRARY_HANDLE
    if fallback_handle is not None:
        _CUDA_DRIVER_LIBRARY_HANDLE = fallback_handle
        return _CUDA_DRIVER_LIBRARY_HANDLE
    raise BackendNotImplementedError("libcuda.so.1 was not found; install the NVIDIA driver to enable the PTX backend")


class CudaDriver:
    def __init__(self) -> None:
        self._lib = load_cuda_driver_library(global_scope=False)
        self._bind()
        self._check(self._lib.cuInit(0), "cuInit")

    def _bind(self) -> None:
        self._lib.cuInit.argtypes = [ctypes.c_uint]
        self._lib.cuInit.restype = ctypes.c_int
        self._lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.cuDriverGetVersion.restype = ctypes.c_int
        self._lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._lib.cuGetErrorName.restype = ctypes.c_int
        self._lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._lib.cuGetErrorString.restype = ctypes.c_int
        self._lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.cuDeviceGetCount.restype = ctypes.c_int
        self._lib.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
        self._lib.cuDeviceGet.restype = ctypes.c_int
        self._lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, CUdevice]
        self._lib.cuDeviceGetName.restype = ctypes.c_int
        self._lib.cuDeviceComputeCapability.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            CUdevice,
        ]
        self._lib.cuDeviceComputeCapability.restype = ctypes.c_int
        self._lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(CUcontext)]
        self._lib.cuCtxGetCurrent.restype = ctypes.c_int
        self._lib.cuCtxSetCurrent.argtypes = [CUcontext]
        self._lib.cuCtxSetCurrent.restype = ctypes.c_int
        self._lib.cuCtxSynchronize.argtypes = []
        self._lib.cuCtxSynchronize.restype = ctypes.c_int
        self._lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(CUcontext), CUdevice]
        self._lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        self._lib.cuDevicePrimaryCtxRelease.argtypes = [CUdevice]
        self._lib.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
        self._lib.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(CUmodule),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.cuModuleLoadDataEx.restype = ctypes.c_int
        self._lib.cuModuleUnload.argtypes = [CUmodule]
        self._lib.cuModuleUnload.restype = ctypes.c_int
        self._lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p]
        self._lib.cuModuleGetFunction.restype = ctypes.c_int
        self._lib.cuLaunchKernel.argtypes = [
            CUfunction,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            CUstream,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.cuLaunchKernel.restype = ctypes.c_int

    def _error_name(self, status: int) -> str | None:
        name = ctypes.c_char_p()
        if self._lib.cuGetErrorName(status, ctypes.byref(name)) != _CUDA_SUCCESS:
            return None
        if not name.value:
            return None
        return name.value.decode("utf-8", errors="replace")

    def _error_message(self, status: int) -> str | None:
        message = ctypes.c_char_p()
        if self._lib.cuGetErrorString(status, ctypes.byref(message)) != _CUDA_SUCCESS:
            return None
        if not message.value:
            return None
        return message.value.decode("utf-8", errors="replace")

    def _check(self, status: int, op: str) -> None:
        if status == _CUDA_SUCCESS:
            return
        raise CudaDriverError(op, status, self._error_name(status), self._error_message(status))

    def driver_version(self) -> int:
        version = ctypes.c_int()
        self._check(self._lib.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
        return int(version.value)

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(self._lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        return int(count.value)

    def device(self, ordinal: int = 0) -> CUdevice:
        device = CUdevice()
        self._check(self._lib.cuDeviceGet(ctypes.byref(device), int(ordinal)), "cuDeviceGet")
        return device

    def device_name(self, device_or_ordinal: CUdevice | int = 0) -> str:
        device = self.device(device_or_ordinal) if isinstance(device_or_ordinal, int) else device_or_ordinal
        buffer = ctypes.create_string_buffer(256)
        self._check(self._lib.cuDeviceGetName(buffer, len(buffer), device), "cuDeviceGetName")
        return buffer.value.decode("utf-8", errors="replace")

    def device_compute_capability(self, device_or_ordinal: CUdevice | int = 0) -> tuple[int, int]:
        device = self.device(device_or_ordinal) if isinstance(device_or_ordinal, int) else device_or_ordinal
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self._check(
            self._lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device),
            "cuDeviceComputeCapability",
        )
        return int(major.value), int(minor.value)

    def device_info(self, ordinal: int = 0) -> CudaDeviceInfo:
        return CudaDeviceInfo(
            ordinal=int(ordinal),
            name=self.device_name(ordinal),
            compute_capability=self.device_compute_capability(ordinal),
        )

    def current_context(self) -> CUcontext | None:
        context = CUcontext()
        self._check(self._lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
        if not context.value:
            return None
        return context

    def ensure_primary_context(self, ordinal: int = 0) -> CUcontext:
        current = self.current_context()
        if current is not None:
            return current
        device = self.device(ordinal)
        context = CUcontext()
        self._check(self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
        self._check(self._lib.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        return context

    def release_primary_context(self, device_or_ordinal: CUdevice | int = 0) -> None:
        device = self.device(device_or_ordinal) if isinstance(device_or_ordinal, int) else device_or_ordinal
        self._check(self._lib.cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease")

    def synchronize(self) -> None:
        self._check(self._lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def load_module_from_ptx(self, ptx: str | bytes, *, ordinal: int = 0) -> CUmodule:
        self.ensure_primary_context(ordinal)
        blob = ptx.encode("utf-8") if isinstance(ptx, str) else ptx
        buffer = ctypes.create_string_buffer(blob)
        module = CUmodule()
        self._check(
            self._lib.cuModuleLoadDataEx(
                ctypes.byref(module),
                ctypes.cast(buffer, ctypes.c_void_p),
                0,
                None,
                None,
            ),
            "cuModuleLoadDataEx",
        )
        return module

    def unload_module(self, module: CUmodule) -> None:
        self._check(self._lib.cuModuleUnload(module), "cuModuleUnload")

    def function(self, module: CUmodule, name: str) -> CUfunction:
        function = CUfunction()
        self._check(self._lib.cuModuleGetFunction(ctypes.byref(function), module, name.encode("utf-8")), "cuModuleGetFunction")
        return function

    def launch_kernel(
        self,
        function: CUfunction,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_mem_bytes: int = 0,
        stream: CUstream | None = None,
        kernel_params: Sequence[int | ctypes.c_void_p] | None = None,
    ) -> None:
        params_array = None
        if kernel_params:
            packed = [ctypes.c_void_p(int(value)) if not isinstance(value, ctypes.c_void_p) else value for value in kernel_params]
            params_array = (ctypes.c_void_p * len(packed))(*packed)
        self._check(
            self._lib.cuLaunchKernel(
                function,
                ctypes.c_uint(grid[0]),
                ctypes.c_uint(grid[1]),
                ctypes.c_uint(grid[2]),
                ctypes.c_uint(block[0]),
                ctypes.c_uint(block[1]),
                ctypes.c_uint(block[2]),
                ctypes.c_uint(shared_mem_bytes),
                stream or CUstream(),
                params_array,
                None,
            ),
            "cuLaunchKernel",
        )


__all__ = [
    "CUcontext",
    "CUdevice",
    "CUfunction",
    "CUmodule",
    "CUstream",
    "CudaDeviceInfo",
    "CudaDriver",
    "CudaDriverError",
    "load_cuda_driver_library",
]
