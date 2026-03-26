from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

from ..backend import LoweredModule
from ..cuda_driver import CudaDriver, CUdeviceptr, load_cuda_driver_library
from ..diagnostics import BackendNotImplementedError
from ..dtypes import normalize_dtype_name
from ..hip_runtime import contiguous_size, pack_tensor_value, unpack_tensor_value
from ..ir import KernelArgument, PortableKernelIR, ScalarSpec, TensorSpec
from ..runtime import RuntimeScalar, RuntimeTensor, TensorHandle
from ..target import NvidiaTarget
from .ptx_bridge import PtxBridge

_DLPACK_DEVICE_TYPE_CUDA = 2


def _tensor_ctype(dtype: str):
    table = {
        "f32": ctypes.c_float,
        "i1": ctypes.c_bool,
        "i32": ctypes.c_int32,
    }
    try:
        return table[dtype]
    except KeyError as exc:
        raise BackendNotImplementedError(f"ptx_exec does not support tensor dtype '{dtype}' yet") from exc


def _scalar_ctype(dtype: str):
    table = {
        "f32": ctypes.c_float,
        "i1": ctypes.c_bool,
        "i32": ctypes.c_int32,
    }
    try:
        return table[dtype]
    except KeyError as exc:
        raise BackendNotImplementedError(f"ptx_exec does not support scalar dtype '{dtype}' yet") from exc


@dataclass
class _CachedCudaTensorSlot:
    ctype: Any
    ptr: CUdeviceptr
    byte_size: int
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def matches(self, value: RuntimeTensor, stride: tuple[int, ...], byte_size: int) -> bool:
        return (
            self.dtype == value.dtype
            and self.shape == value.shape
            and self.stride == stride
            and self.byte_size == byte_size
        )


class PtxExecBackend:
    name = "ptx_exec"
    artifact_extension = ".ptx"

    def __init__(self) -> None:
        self._bridge = PtxBridge()

    def available(self, target: NvidiaTarget | None = None) -> bool:
        del target
        try:
            load_cuda_driver_library()
            driver = CudaDriver()
        except Exception:
            return False
        return driver.device_count() >= 1

    def supports(self, ir: PortableKernelIR, target: NvidiaTarget) -> bool:
        return self._bridge.supports(ir, target)

    def lower(self, ir: PortableKernelIR, target: NvidiaTarget) -> LoweredModule:
        return self._bridge.lower(ir, target, backend_name=self.name)

    def build_launcher(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        lowered_module: LoweredModule,
        source_path: Path,
    ):
        del source_path
        state: dict[str, Any] = {}

        def launcher(*args: Any, **kwargs: Any) -> None:
            stream = kwargs.pop("stream", None)
            if kwargs:
                raise TypeError("ptx_exec launcher only supports positional arguments and an optional stream=")
            if len(args) != len(ir.arguments):
                raise TypeError(f"{ir.name} expects {len(ir.arguments)} arguments, got {len(args)}")
            driver: CudaDriver = state.get("driver")  # type: ignore[assignment]
            function = state.get("function")
            if function is None:
                driver = CudaDriver()
                module = driver.load_module_from_ptx(lowered_module.text)
                function = driver.function(module, ir.name)
                state["driver"] = driver
                state["module"] = module
                state["function"] = function
            else:
                driver = state["driver"]
            self._launch(driver, function, ir, args, stream=stream, state=state)

        return launcher

    def _launch(
        self,
        driver: CudaDriver,
        function: Any,
        ir: PortableKernelIR,
        args: tuple[Any, ...],
        *,
        stream: Any = None,
        state: dict[str, Any],
    ) -> None:
        staged_runtime_tensors: list[tuple[RuntimeTensor, _CachedCudaTensorSlot]] = []
        kernel_params: list[object] = []
        staged_argument_names = [
            argument.name
            for argument, value in zip(ir.arguments, args)
            if isinstance(argument.spec, TensorSpec) and isinstance(value, RuntimeTensor)
        ]
        if staged_argument_names and not state.get("warned_runtime_tensor_staging"):
            warnings.warn(
                "ptx_exec is staging RuntimeTensor arguments through host memory for tensor arguments "
                f"{staged_argument_names}; pass CUDA TensorHandle values for tensor arguments to avoid "
                "host-copy dominated timings",
                RuntimeWarning,
                stacklevel=3,
            )
            state["warned_runtime_tensor_staging"] = True
        try:
            for arg_index, (argument, value) in enumerate(zip(ir.arguments, args)):
                if isinstance(argument.spec, TensorSpec):
                    if not isinstance(value, RuntimeTensor):
                        if isinstance(value, TensorHandle) or self._is_dlpack_tensor(value):
                            kernel_params.append(self._tensor_handle_ptr(self._coerce_tensor_handle(value, stream), argument))
                            continue
                        raise TypeError(
                            "ptx_exec currently expects RuntimeTensor values, CUDA TensorHandle values, "
                            f"or CUDA DLPack-capable tensor objects for tensor argument '{argument.name}', got {type(value).__name__}"
                        )
                    slot = self._stage_runtime_tensor(driver, value, arg_index, state)
                    staged_runtime_tensors.append((value, slot))
                    kernel_params.append(int(slot.ptr.value))
                    continue
                kernel_params.append(self._scalar_kernel_param(value, argument))
            driver.launch_kernel(
                function,
                grid=ir.launch.grid,
                block=ir.launch.block,
                shared_mem_bytes=ir.launch.shared_mem_bytes,
                stream=stream,
                kernel_params=kernel_params,
            )
            if stream is None:
                driver.synchronize()
            else:
                driver.synchronize_stream(stream)
            for tensor, slot in staged_runtime_tensors:
                self._copy_back_runtime_tensor(driver, tensor, slot)
        except Exception:
            self._release_cached_tensor_slots(driver, state)
            raise

    def _stage_runtime_tensor(
        self,
        driver: CudaDriver,
        value: RuntimeTensor,
        arg_index: int,
        state: dict[str, Any],
    ) -> _CachedCudaTensorSlot:
        expected_size = contiguous_size(value.shape)
        canonical_stride = self._canonical_stride(value.shape)
        if value.offset != 0 or value.stride != canonical_stride:
            raise BackendNotImplementedError("ptx_exec currently requires contiguous runtime tensors without offsets")
        if len(value._storage) != expected_size:
            raise BackendNotImplementedError("ptx_exec currently requires densely packed runtime tensors")
        ctype = _tensor_ctype(value.dtype)
        byte_size = expected_size * ctypes.sizeof(ctype)
        slots: dict[int, _CachedCudaTensorSlot] = state.setdefault("tensor_slots", {})
        slot = slots.get(arg_index)
        if slot is None or not slot.matches(value, canonical_stride, byte_size):
            if slot is not None:
                driver.mem_free(slot.ptr)
            slot = _CachedCudaTensorSlot(
                ctype=ctype,
                ptr=driver.mem_alloc(byte_size),
                byte_size=byte_size,
                dtype=value.dtype,
                shape=value.shape,
                stride=canonical_stride,
            )
            slots[arg_index] = slot
        host_array = (ctype * expected_size)(*(pack_tensor_value(item, value.dtype) for item in value._storage))
        driver.memcpy_htod(slot.ptr, ctypes.cast(host_array, ctypes.c_void_p), byte_size)
        return slot

    def _copy_back_runtime_tensor(
        self,
        driver: CudaDriver,
        tensor: RuntimeTensor,
        slot: _CachedCudaTensorSlot,
    ) -> None:
        host_array = (slot.ctype * len(tensor._storage))()
        driver.memcpy_dtoh(ctypes.cast(host_array, ctypes.c_void_p), slot.ptr, slot.byte_size)
        tensor._storage[:] = [unpack_tensor_value(item, tensor.dtype) for item in host_array]

    def _release_cached_tensor_slots(self, driver: CudaDriver, state: dict[str, Any]) -> None:
        slots: dict[int, _CachedCudaTensorSlot] = state.pop("tensor_slots", {})
        for slot in slots.values():
            driver.mem_free(slot.ptr)

    def _tensor_handle_ptr(self, value: TensorHandle, argument: KernelArgument) -> int:
        if int(value.device_type) != _DLPACK_DEVICE_TYPE_CUDA:
            raise BackendNotImplementedError(
                f"ptx_exec currently requires CUDA TensorHandle inputs for tensor argument '{argument.name}'"
            )
        data_ptr = value.data_ptr() or value.raw_address
        if not data_ptr:
            raise TypeError(
                f"ptx_exec tensor handle for argument '{argument.name}' does not expose a usable data_ptr()"
            )
        if value.stride is not None and tuple(int(dim) for dim in value.stride) != self._canonical_stride(value.shape):
            raise BackendNotImplementedError(
                f"ptx_exec currently requires contiguous CUDA TensorHandle inputs for tensor argument '{argument.name}'"
            )
        return int(data_ptr)

    def _coerce_tensor_handle(self, value: Any, stream: Any) -> TensorHandle:
        if isinstance(value, TensorHandle):
            return value
        return self._tensor_handle_from_dlpack(value, stream)

    def _tensor_handle_from_dlpack(self, value: Any, stream: Any) -> TensorHandle:
        device_type, device_id = value.__dlpack_device__()
        if stream is None:
            capsule = value.__dlpack__()
        else:
            try:
                capsule = value.__dlpack__(stream=stream)
            except TypeError:
                capsule = value.__dlpack__()
        shape = tuple(getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", "unknown"))
        stride_value = getattr(value, "stride", None)
        if callable(stride_value):
            stride_value = stride_value()
        stride = tuple(int(dim) for dim in stride_value) if stride_value is not None else None
        raw_address = int(value.data_ptr()) if hasattr(value, "data_ptr") else None
        return TensorHandle(
            capsule=capsule,
            shape=shape,
            dtype=dtype,
            device_type=device_type,
            device_id=device_id,
            source=value,
            stride=stride,
            raw_address=raw_address,
        )

    def _scalar_kernel_param(self, value: Any, argument: KernelArgument) -> ctypes._SimpleCData:
        if not isinstance(argument.spec, ScalarSpec):
            raise TypeError("scalar kernel parameter adaptation requires a scalar argument spec")
        ctype = _scalar_ctype(argument.spec.dtype)
        if isinstance(value, RuntimeScalar):
            if normalize_dtype_name(value.dtype) != argument.spec.dtype:
                raise TypeError(
                    f"ptx_exec scalar argument '{argument.name}' expects dtype {argument.spec.dtype}, got {value.dtype}"
                )
            return ctype(value.value)
        return ctype(value)

    def _canonical_stride(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if not shape:
            return ()
        stride = [1] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            stride[index] = running
            running *= shape[index]
        return tuple(stride)

    def _is_dlpack_tensor(self, value: Any) -> bool:
        return hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__")
