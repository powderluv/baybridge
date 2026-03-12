from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..hip_runtime import HipRuntime, load_hip_library, scalar_ctype
from ..ir import KernelArgument, PortableKernelIR, TensorSpec
from ..runtime import RuntimeTensor, TensorHandle
from ..target import AMDTarget
from .waveasm_bridge import WaveAsmBridge

_EXPERIMENTAL_WAVEASM_EXEC_ENV = "BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC"
_SUPPORTED_EXEC_OPS = {
    "add",
    "and",
    "barrier",
    "block_dim",
    "block_idx",
    "compare",
    "constant",
    "copy",
    "div",
    "floordiv",
    "grid_dim",
    "load",
    "make_tensor",
    "masked_load",
    "masked_store",
    "mod",
    "mul",
    "or",
    "program_id",
    "select",
    "store",
    "sub",
    "thread_idx",
}


class _HipModuleRuntime:
    def __init__(self) -> None:
        self._lib = load_hip_library(global_scope=True)
        self._lib.hipGetErrorString.argtypes = [ctypes.c_int]
        self._lib.hipGetErrorString.restype = ctypes.c_char_p
        self._lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self._lib.hipModuleLoad.restype = ctypes.c_int
        self._lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
        self._lib.hipModuleUnload.restype = ctypes.c_int
        self._lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self._lib.hipModuleGetFunction.restype = ctypes.c_int
        self._lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self._lib.hipModuleLaunchKernel.restype = ctypes.c_int

    def load_module(self, hsaco_path: Path) -> ctypes.c_void_p:
        module = ctypes.c_void_p()
        self._check(self._lib.hipModuleLoad(ctypes.byref(module), str(hsaco_path).encode("utf-8")), "hipModuleLoad")
        return module

    def unload_module(self, module: ctypes.c_void_p) -> None:
        self._check(self._lib.hipModuleUnload(module), "hipModuleUnload")

    def get_function(self, module: ctypes.c_void_p, kernel_name: str) -> ctypes.c_void_p:
        function = ctypes.c_void_p()
        self._check(
            self._lib.hipModuleGetFunction(ctypes.byref(function), module, kernel_name.encode("utf-8")),
            "hipModuleGetFunction",
        )
        return function

    def launch(
        self,
        function: ctypes.c_void_p,
        launch,
        kernel_args: list[Any],
        stream: Any | None = None,
    ) -> None:
        params = (ctypes.c_void_p * len(kernel_args))()
        for index, arg in enumerate(kernel_args):
            params[index] = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)
        stream_ptr = ctypes.c_void_p(0)
        if stream is not None:
            if hasattr(stream, "value"):
                stream_ptr = ctypes.c_void_p(int(stream.value or 0))
            else:
                stream_ptr = ctypes.c_void_p(int(stream))
        self._check(
            self._lib.hipModuleLaunchKernel(
                function,
                launch.grid[0],
                launch.grid[1],
                launch.grid[2],
                launch.block[0],
                launch.block[1],
                launch.block[2],
                launch.shared_mem_bytes,
                stream_ptr,
                params,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _check(self, status: int, op: str) -> None:
        if status == 0:
            return
        message = self._lib.hipGetErrorString(status).decode("utf-8", errors="replace")
        raise RuntimeError(f"{op} failed with HIP status {status}: {message}")


class WaveAsmExecBackend:
    name = "waveasm_exec"
    artifact_extension = ".wave.mlir"

    def __init__(self) -> None:
        self._bridge = WaveAsmBridge()

    def available(self, target: AMDTarget | None = None) -> bool:
        del target
        return self._bridge.environment().ready and self._experimental_enabled()

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        del target
        if not self._experimental_enabled():
            return False
        if self._count_global_tensor_arguments(ir) > 1:
            return False
        if any(operation.op not in _SUPPORTED_EXEC_OPS and not operation.op.startswith("cmp_") for operation in ir.operations):
            return False
        if ir.launch.cooperative:
            return False
        for operation in ir.operations:
            if operation.op == "barrier" and operation.attrs.get("kind", "block") != "block":
                return False
            if operation.op == "make_tensor":
                if operation.attrs.get("dynamic_shared"):
                    return False
                if operation.attrs.get("address_space") != "shared":
                    return False
        return True

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        if not self.supports(ir, target):
            if not self._experimental_enabled():
                raise BackendNotImplementedError(
                    "waveasm_exec is disabled by default; set BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC=1 to enable the experimental backend. "
                    "Current upstream correctness issues are tracked in iree-org/wave#1117."
                )
            raise BackendNotImplementedError(
                "waveasm_exec currently supports only experimental single-global-tensor pointwise/shared-memory kernels lowered fully to standard GPU/MLIR ops"
            )
        lowered = self._bridge.lower_mlir(ir, target)
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="waveasm_exec_mlir",
            text=lowered.text,
        )

    def build_launcher(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        source_path: Path,
    ):
        hsaco_path = source_path.with_suffix(".hsaco")
        state: dict[str, Any] = {}

        def launcher(*args: Any, **kwargs: Any) -> None:
            stream = kwargs.pop("stream", None)
            if kwargs:
                raise TypeError("waveasm_exec launcher only supports positional arguments and an optional stream=")
            if len(args) != len(ir.arguments):
                raise TypeError(f"{ir.name} expects {len(ir.arguments)} arguments, got {len(args)}")
            if not hsaco_path.exists():
                self._bridge.compile_to_hsaco(
                    source_path,
                    hsaco_path,
                    target,
                    lowered_module.text,
                    workgroup_size=ir.launch.block,
                )
            runtime = state.get("runtime")
            function = state.get("function")
            if runtime is None or function is None:
                runtime = self._make_module_runtime()
                module = runtime.load_module(hsaco_path)
                function = runtime.get_function(module, ir.name)
                state["runtime"] = runtime
                state["module"] = module
                state["function"] = function
            self._launch(state["runtime"], state["function"], ir, args, stream)

        return launcher

    def emit_bundle(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        lowered_path: Path,
    ) -> Path:
        return self._bridge.write_repro_bundle(ir, target, lowered_module, lowered_path)

    def _make_module_runtime(self) -> _HipModuleRuntime:
        return _HipModuleRuntime()

    def _experimental_enabled(self) -> bool:
        import os

        return os.environ.get(_EXPERIMENTAL_WAVEASM_EXEC_ENV) == "1"

    def _count_global_tensor_arguments(self, ir: PortableKernelIR) -> int:
        return sum(
            1
            for argument in ir.arguments
            if isinstance(argument.spec, TensorSpec) and argument.spec.address_space.value == "global"
        )

    def _launch(
        self,
        runtime: _HipModuleRuntime,
        function: Any,
        ir: PortableKernelIR,
        args: tuple[Any, ...],
        stream: Any | None,
    ) -> None:
        hip = HipRuntime()
        tensor_allocations = []
        kernel_args: list[Any] = []
        try:
            for argument, value in zip(ir.arguments, args):
                if isinstance(argument.spec, TensorSpec):
                    if isinstance(value, RuntimeTensor):
                        allocation = hip.upload_tensor(value)
                        tensor_allocations.append(allocation)
                        kernel_args.append(ctypes.c_void_p(int(allocation.ptr.value or 0)))
                        continue
                    if isinstance(value, TensorHandle):
                        data_ptr = value.data_ptr()
                        if not data_ptr:
                            raise TypeError(
                                f"waveasm_exec tensor handle for argument '{argument.name}' does not expose a usable data_ptr()"
                            )
                        kernel_args.append(ctypes.c_void_p(int(data_ptr)))
                        continue
                    raise TypeError(
                        f"waveasm_exec expects RuntimeTensor or TensorHandle values for tensor argument '{argument.name}', got {type(value).__name__}"
                    )
                kernel_args.append(scalar_ctype(argument.spec.dtype)(value))
            runtime.launch(function, ir.launch, kernel_args, stream)
            hip.synchronize()
            for allocation in tensor_allocations:
                allocation.copy_back(hip)
        finally:
            for allocation in tensor_allocations:
                allocation.free(hip)
