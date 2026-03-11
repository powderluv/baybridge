from __future__ import annotations

import ctypes
import sys
import shutil
import struct
from glob import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import BackendNotImplementedError
from .runtime import RuntimeTensor

_HIP_MEMCPY_HOST_TO_DEVICE = 1
_HIP_MEMCPY_DEVICE_TO_HOST = 2
_HIP_LIBRARY_HANDLE: ctypes.CDLL | None = None


def require_hipcc() -> str:
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        raise BackendNotImplementedError("hipcc was not found on PATH; the hipcc_exec backend cannot build kernels")
    return hipcc


def _hip_library_candidates() -> list[str]:
    candidates: list[Path] = []
    prefix = Path(sys.prefix)
    for candidate in glob(str(prefix / "lib" / "python*" / "site-packages" / "_rocm_sdk_*" / "lib")):
        libdir = Path(candidate)
        candidates.extend(
            [
                libdir / "libamdhip64.so",
                libdir / "libamdhip64.so.7",
            ]
        )
        candidates.extend(Path(path) for path in sorted(glob(str(libdir / "libamdhip64.so.*"))))
    for root in [Path("/opt/rocm/lib"), Path("/opt/rocm/lib64"), Path("/usr/lib"), Path("/usr/lib64")]:
        candidates.extend([root / "libamdhip64.so", root / "libamdhip64.so.7"])
    for root in sorted(glob("/opt/rocm-*")):
        path_root = Path(root)
        for subdir in ("lib", "lib64"):
            libdir = path_root / subdir
            candidates.extend([libdir / "libamdhip64.so", libdir / "libamdhip64.so.7"])
    deduped: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        deduped.append(str(candidate))
    deduped.append("libamdhip64.so")
    return deduped


def load_hip_library(*, global_scope: bool = False) -> ctypes.CDLL:
    global _HIP_LIBRARY_HANDLE
    if _HIP_LIBRARY_HANDLE is not None:
        return _HIP_LIBRARY_HANDLE
    mode = getattr(ctypes, "RTLD_GLOBAL", 0) if global_scope else getattr(ctypes, "RTLD_LOCAL", 0)
    fallback_handle: ctypes.CDLL | None = None
    for candidate in _hip_library_candidates():
        try:
            handle = ctypes.CDLL(candidate, mode=mode)
        except OSError:
            continue
        if fallback_handle is None:
            fallback_handle = handle
        if _hip_runtime_has_device(handle):
            _HIP_LIBRARY_HANDLE = handle
            return _HIP_LIBRARY_HANDLE
    if fallback_handle is not None:
        _HIP_LIBRARY_HANDLE = fallback_handle
        return _HIP_LIBRARY_HANDLE
    raise BackendNotImplementedError("libamdhip64 was not found; install ROCm or TheRock runtime libraries in the active environment")


def _hip_runtime_has_device(handle: ctypes.CDLL) -> bool:
    try:
        handle.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        handle.hipGetDeviceCount.restype = ctypes.c_int
    except AttributeError:
        return False
    count = ctypes.c_int()
    status = handle.hipGetDeviceCount(ctypes.byref(count))
    return status == 0 and count.value >= 1


def scalar_ctype(dtype: str):
    table = {
        "f32": ctypes.c_float,
        "i1": ctypes.c_bool,
        "i8": ctypes.c_int8,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "index": ctypes.c_longlong,
    }
    try:
        return table[dtype]
    except KeyError as exc:
        raise BackendNotImplementedError(f"hipcc_exec does not support scalar dtype '{dtype}' yet") from exc


def tensor_ctype(dtype: str):
    table = {
        "f32": ctypes.c_float,
        "f16": ctypes.c_uint16,
        "bf16": ctypes.c_uint16,
        "i8": ctypes.c_int8,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
    }
    try:
        return table[dtype]
    except KeyError as exc:
        raise BackendNotImplementedError(f"hipcc_exec does not support tensor dtype '{dtype}' yet") from exc


def contiguous_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


@dataclass
class DeviceTensor:
    tensor: RuntimeTensor
    ctype: Any
    ptr: ctypes.c_void_p
    byte_size: int
    _host_array: Any

    def copy_back(self, hip: "HipRuntime") -> None:
        host_array = (self.ctype * len(self.tensor._storage))()
        hip.memcpy(
            ctypes.cast(host_array, ctypes.c_void_p),
            self.ptr,
            self.byte_size,
            _HIP_MEMCPY_DEVICE_TO_HOST,
        )
        self.tensor._storage[:] = [unpack_tensor_value(item, self.tensor.dtype) for item in host_array]

    def free(self, hip: "HipRuntime") -> None:
        hip.free(self.ptr)


class HipRuntime:
    def __init__(self) -> None:
        self._lib = load_hip_library(global_scope=True)
        self._lib.hipGetErrorString.argtypes = [ctypes.c_int]
        self._lib.hipGetErrorString.restype = ctypes.c_char_p
        self._lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.hipMalloc.restype = ctypes.c_int
        self._lib.hipFree.argtypes = [ctypes.c_void_p]
        self._lib.hipFree.restype = ctypes.c_int
        self._lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._lib.hipMemcpy.restype = ctypes.c_int
        self._lib.hipDeviceSynchronize.argtypes = []
        self._lib.hipDeviceSynchronize.restype = ctypes.c_int

    def _check(self, status: int, op: str) -> None:
        if status == 0:
            return
        message = self._lib.hipGetErrorString(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"{op} failed with HIP status {status}: {message}")

    def malloc(self, byte_size: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self._lib.hipMalloc(ctypes.byref(ptr), byte_size), "hipMalloc")
        return ptr

    def memcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p, byte_size: int, kind: int) -> None:
        self._check(self._lib.hipMemcpy(dst, src, byte_size, kind), "hipMemcpy")

    def synchronize(self) -> None:
        self._check(self._lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def free(self, ptr: ctypes.c_void_p) -> None:
        self._check(self._lib.hipFree(ptr), "hipFree")

    def upload_tensor(self, value: RuntimeTensor) -> DeviceTensor:
        expected_size = contiguous_size(value.shape)
        if value.offset != 0 or value.stride != _canonical_stride(value.shape):
            raise BackendNotImplementedError("hipcc_exec currently requires contiguous runtime tensors without offsets")
        if len(value._storage) != expected_size:
            raise BackendNotImplementedError("hipcc_exec currently requires densely packed runtime tensors")
        ctype = tensor_ctype(value.dtype)
        host_array = (ctype * expected_size)(*(pack_tensor_value(item, value.dtype) for item in value._storage))
        byte_size = ctypes.sizeof(host_array)
        ptr = self.malloc(byte_size)
        self.memcpy(ptr, ctypes.cast(host_array, ctypes.c_void_p), byte_size, _HIP_MEMCPY_HOST_TO_DEVICE)
        return DeviceTensor(
            tensor=value,
            ctype=ctype,
            ptr=ptr,
            byte_size=byte_size,
            _host_array=host_array,
        )


def _canonical_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return tuple(stride)


def pack_tensor_value(value: Any, dtype: str) -> Any:
    if dtype == "f32":
        return float(value)
    if dtype == "i8":
        return int(value)
    if dtype == "i32":
        return int(value)
    if dtype == "i64":
        return int(value)
    if dtype == "f16":
        return struct.unpack("<H", struct.pack("<e", float(value)))[0]
    if dtype == "bf16":
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        return (bits + rounding_bias) >> 16
    raise BackendNotImplementedError(f"hipcc_exec does not support tensor dtype '{dtype}' yet")


def unpack_tensor_value(value: Any, dtype: str) -> Any:
    if dtype == "f32":
        return float(value)
    if dtype == "i8":
        return int(value)
    if dtype == "i32":
        return int(value)
    if dtype == "i64":
        return int(value)
    if dtype == "f16":
        return float(struct.unpack("<e", struct.pack("<H", int(value)))[0])
    if dtype == "bf16":
        return float(struct.unpack("<f", struct.pack("<I", int(value) << 16))[0])
    raise BackendNotImplementedError(f"hipcc_exec does not support tensor dtype '{dtype}' yet")
