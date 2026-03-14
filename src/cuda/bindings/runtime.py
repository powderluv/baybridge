from __future__ import annotations

import ctypes
from enum import IntEnum
from typing import Any

from baybridge.hip_runtime import load_hip_library
from .driver import CUgraph, CUgraphExec, CUstream, CUstreamCaptureMode, CUstreamCaptureStatus


class cudaError_t(IntEnum):
    cudaSuccess = 0
    cudaErrorCooperativeLaunchTooLarge = 82


_GRAPH_COUNTER = 1
_CAPTURED_STREAMS: dict[int, int] = {}


def _library() -> ctypes.CDLL:
    handle = load_hip_library(global_scope=True)
    handle.hipGetErrorString.argtypes = [ctypes.c_int]
    handle.hipGetErrorString.restype = ctypes.c_char_p
    handle.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    handle.hipMalloc.restype = ctypes.c_int
    handle.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    handle.hipMemset.restype = ctypes.c_int
    handle.hipFree.argtypes = [ctypes.c_void_p]
    handle.hipFree.restype = ctypes.c_int
    handle.hipDeviceSynchronize.argtypes = []
    handle.hipDeviceSynchronize.restype = ctypes.c_int
    return handle


def _pointer_value(ptr: Any) -> int:
    if isinstance(ptr, ctypes.c_void_p):
        return int(ptr.value or 0)
    if hasattr(ptr, "_pointer"):
        value = getattr(ptr, "_pointer")
        if value is not None:
            return int(value)
    return int(ptr)


def _check_known_status(value: int) -> int | cudaError_t:
    if value == 0:
        return cudaError_t.cudaSuccess
    if value == cudaError_t.cudaErrorCooperativeLaunchTooLarge:
        return cudaError_t.cudaErrorCooperativeLaunchTooLarge
    return int(value)


def cudaMalloc(byte_count: int) -> tuple[int | cudaError_t, int]:
    handle = _library()
    ptr = ctypes.c_void_p()
    status = handle.hipMalloc(ctypes.byref(ptr), int(byte_count))
    return _check_known_status(status), int(ptr.value or 0)


def cudaMemset(ptr: Any, value: int, byte_count: int) -> tuple[int | cudaError_t]:
    handle = _library()
    status = handle.hipMemset(ctypes.c_void_p(_pointer_value(ptr)), int(value), int(byte_count))
    return (_check_known_status(status),)


def cudaFree(ptr: Any) -> tuple[int | cudaError_t]:
    handle = _library()
    status = handle.hipFree(ctypes.c_void_p(_pointer_value(ptr)))
    return (_check_known_status(status),)


def cudaDeviceSynchronize() -> tuple[int | cudaError_t]:
    handle = _library()
    status = handle.hipDeviceSynchronize()
    return (_check_known_status(status),)


def cudaStreamBeginCapture(
    stream: CUstream | int,
    mode: CUstreamCaptureMode | int = CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL,
) -> tuple[int | cudaError_t]:
    del mode
    _CAPTURED_STREAMS[int(stream)] = 1
    return (cudaError_t.cudaSuccess,)


def cudaStreamEndCapture(stream: CUstream | int) -> tuple[int | cudaError_t, CUgraph]:
    global _GRAPH_COUNTER
    stream_id = int(stream)
    _CAPTURED_STREAMS.pop(stream_id, None)
    graph = CUgraph(_GRAPH_COUNTER)
    _GRAPH_COUNTER += 1
    return cudaError_t.cudaSuccess, graph


def cudaStreamIsCapturing(stream: CUstream | int) -> tuple[int | cudaError_t, CUstreamCaptureStatus]:
    stream_id = int(stream)
    if stream_id in _CAPTURED_STREAMS:
        return cudaError_t.cudaSuccess, CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE
    return cudaError_t.cudaSuccess, CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE


def cudaGraphInstantiate(graph: CUgraph | int, flags: int = 0) -> tuple[int | cudaError_t, CUgraphExec]:
    del flags
    return cudaError_t.cudaSuccess, CUgraphExec(int(graph))


def cudaGraphLaunch(graph_exec: CUgraphExec | int, stream: CUstream | int) -> tuple[int | cudaError_t]:
    del graph_exec
    del stream
    return (cudaError_t.cudaSuccess,)


def cudaGraphDestroy(graph: CUgraph | int) -> tuple[int | cudaError_t]:
    del graph
    return (cudaError_t.cudaSuccess,)


def cudaGraphExecDestroy(graph_exec: CUgraphExec | int) -> tuple[int | cudaError_t]:
    del graph_exec
    return (cudaError_t.cudaSuccess,)


__all__ = [
    "cudaDeviceSynchronize",
    "cudaError_t",
    "cudaFree",
    "cudaGraphDestroy",
    "cudaGraphExecDestroy",
    "cudaGraphInstantiate",
    "cudaGraphLaunch",
    "cudaMalloc",
    "cudaMemset",
    "cudaStreamIsCapturing",
    "cudaStreamBeginCapture",
    "cudaStreamEndCapture",
]
