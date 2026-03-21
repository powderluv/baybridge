from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backend import LoweredModule
from ..cuda_driver import CudaDriver, CUdeviceptr, load_cuda_driver_library
from ..diagnostics import BackendNotImplementedError
from ..hip_runtime import contiguous_size, pack_tensor_value, unpack_tensor_value
from ..ir import KernelArgument, PortableKernelIR, TensorSpec
from ..runtime import RuntimeTensor, TensorHandle
from ..target import NvidiaTarget
from .ptx_bridge import PtxBridge

_DLPACK_DEVICE_TYPE_CUDA = 2


def _tensor_ctype(dtype: str):
    table = {
        "f32": ctypes.c_float,
        "i32": ctypes.c_int32,
    }
    try:
        return table[dtype]
    except KeyError as exc:
        raise BackendNotImplementedError(f"ptx_exec does not support tensor dtype '{dtype}' yet") from exc


@dataclass
class _CudaDeviceTensor:
    tensor: RuntimeTensor
    ctype: Any
    ptr: CUdeviceptr
    byte_size: int
    host_array: Any

    def copy_back(self, driver: CudaDriver) -> None:
        host_array = (self.ctype * len(self.tensor._storage))()
        driver.memcpy_dtoh(ctypes.cast(host_array, ctypes.c_void_p), self.ptr, self.byte_size)
        self.tensor._storage[:] = [unpack_tensor_value(item, self.tensor.dtype) for item in host_array]

    def free(self, driver: CudaDriver) -> None:
        driver.mem_free(self.ptr)


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
            del stream
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
            self._launch(driver, function, ir, args)

        return launcher

    def _launch(self, driver: CudaDriver, function: Any, ir: PortableKernelIR, args: tuple[Any, ...]) -> None:
        allocations: list[_CudaDeviceTensor] = []
        kernel_params: list[object] = []
        try:
            for argument, value in zip(ir.arguments, args):
                if not isinstance(argument.spec, TensorSpec):
                    raise BackendNotImplementedError("ptx_exec currently supports tensor arguments only")
                if not isinstance(value, RuntimeTensor):
                    if isinstance(value, TensorHandle):
                        kernel_params.append(self._tensor_handle_ptr(value, argument))
                        continue
                    raise TypeError(
                        f"ptx_exec currently expects RuntimeTensor or CUDA TensorHandle values for tensor argument '{argument.name}', got {type(value).__name__}"
                    )
                allocation = self._upload_tensor(driver, value)
                allocations.append(allocation)
                kernel_params.append(int(allocation.ptr.value))
            driver.launch_kernel(
                function,
                grid=ir.launch.grid,
                block=ir.launch.block,
                shared_mem_bytes=ir.launch.shared_mem_bytes,
                kernel_params=kernel_params,
            )
            driver.synchronize()
            for allocation in allocations:
                allocation.copy_back(driver)
        finally:
            for allocation in allocations:
                allocation.free(driver)

    def _upload_tensor(self, driver: CudaDriver, value: RuntimeTensor) -> _CudaDeviceTensor:
        expected_size = contiguous_size(value.shape)
        if value.offset != 0 or value.stride != self._canonical_stride(value.shape):
            raise BackendNotImplementedError("ptx_exec currently requires contiguous runtime tensors without offsets")
        if len(value._storage) != expected_size:
            raise BackendNotImplementedError("ptx_exec currently requires densely packed runtime tensors")
        ctype = _tensor_ctype(value.dtype)
        host_array = (ctype * expected_size)(*(pack_tensor_value(item, value.dtype) for item in value._storage))
        byte_size = ctypes.sizeof(host_array)
        ptr = driver.mem_alloc(byte_size)
        driver.memcpy_htod(ptr, ctypes.cast(host_array, ctypes.c_void_p), byte_size)
        return _CudaDeviceTensor(
            tensor=value,
            ctype=ctype,
            ptr=ptr,
            byte_size=byte_size,
            host_array=host_array,
        )

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

    def _canonical_stride(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if not shape:
            return ()
        stride = [1] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            stride[index] = running
            running *= shape[index]
        return tuple(stride)
