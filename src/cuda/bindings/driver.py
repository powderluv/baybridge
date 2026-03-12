from __future__ import annotations

from enum import IntEnum


class CUstream(int):
    def __new__(cls, value: int = 0):
        return int.__new__(cls, int(value))


class CUgraph(int):
    def __new__(cls, value: int = 0):
        return int.__new__(cls, int(value))


class CUgraphExec(int):
    def __new__(cls, value: int = 0):
        return int.__new__(cls, int(value))


class CUstream_flags(IntEnum):
    CU_STREAM_DEFAULT = 0
    CU_STREAM_NON_BLOCKING = 1


class CUstreamCaptureMode(IntEnum):
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
    CU_STREAM_CAPTURE_MODE_RELAXED = 2


__all__ = [
    "CUgraph",
    "CUgraphExec",
    "CUstream",
    "CUstreamCaptureMode",
    "CUstream_flags",
]
