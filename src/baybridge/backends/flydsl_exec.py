from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..ir import PortableKernelIR
from ..runtime import Pointer, RuntimeTensor, TensorHandle, tensor as make_runtime_tensor
from ..target import AMDTarget
from .flydsl_bridge import FlyDslBridge


class FlyDslExecBackend:
    name = "flydsl_exec"
    artifact_extension = ".fly.py"
    _SUPPORTED_RUNTIME_DTYPES = frozenset({"i1", "i8", "i32", "i64", "f16", "bf16", "f32"})
    _EXPERIMENTAL_REAL_EXEC_ENV = "BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC"

    def __init__(self) -> None:
        self._bridge = FlyDslBridge()

    def available(self, target: AMDTarget | None = None) -> bool:
        del target
        return self._bridge.exec_environment().ready

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        return self._bridge.supports_exec(ir, target)

    def supports_inputs(self, values: tuple[Any, ...]) -> bool:
        if not values:
            return False
        environment = self._bridge.exec_environment()

        def visit(value: Any) -> bool:
            if isinstance(value, TensorHandle):
                return True
            if isinstance(value, RuntimeTensor):
                return (
                    environment.torch_available
                    and self._runtime_tensor_device_available()
                    and value.dtype in self._SUPPORTED_RUNTIME_DTYPES
                )
            if isinstance(value, Pointer):
                return False
            if hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__"):
                return True
            if isinstance(value, list):
                return all(visit(item) for item in value)
            if isinstance(value, tuple):
                return all(visit(item) for item in value)
            if isinstance(value, dict):
                return all(visit(item) for item in value.values())
            return True

        return all(visit(value) for value in values)

    def supports_auto_selection(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        values: tuple[Any, ...],
    ) -> bool:
        if not self.available(target):
            return False
        if not self.supports(ir, target):
            return False
        if not self.supports_inputs(values):
            return False
        environment = self._bridge.exec_environment()
        if self._requires_real_exec_gate(environment, ir) and not self.real_exec_enabled():
            return False
        return True

    def real_exec_enabled(self) -> bool:
        environment = self._bridge.exec_environment()
        if not self._requires_real_exec_gate(environment):
            return True
        return os.environ.get(self._EXPERIMENTAL_REAL_EXEC_ENV) == "1"

    def real_exec_note(self) -> str:
        return (
            "real upstream FlyDSL execution is disabled by default because Baybridge's current lowering "
            "does not yet match FlyDSL's buffer/tile access model; "
            f"set {self._EXPERIMENTAL_REAL_EXEC_ENV}=1 to override"
        )

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        if not self.supports(ir, target):
            raise BackendNotImplementedError("flydsl_exec currently supports a narrow pointwise tensor subset only")
        return self._bridge.lower_exec(ir, target, backend_name=self.name)

    def build_launcher(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        source_path: Path,
    ):
        del target
        module_name = f"_baybridge_flydsl_exec_{ir.name}_{source_path.stem.replace('-', '_')}"
        state: dict[str, Any] = {}

        def launcher(*args: Any, **kwargs: Any) -> Any:
            environment = self._bridge.exec_environment()
            if not environment.ready:
                details = "; ".join(environment.notes) or "FlyDSL compiler/expr imports are not available"
                raise BackendNotImplementedError(
                    "flydsl_exec requires a built and importable FlyDSL environment; "
                    f"{details}"
                )
            if self._requires_real_exec_gate(environment, ir) and not self.real_exec_enabled():
                raise BackendNotImplementedError(self.real_exec_note())
            stream = kwargs.pop("stream", None)
            if kwargs:
                raise TypeError("flydsl_exec launcher only supports positional arguments and an optional stream=")
            launch_fn = state.get("launch_fn")
            if launch_fn is None:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(lowered_module.text, encoding="utf-8")
                module = self._load_module(module_name, source_path)
                launch_fn = getattr(module, f"launch_{ir.name}")
                state["module"] = module
                state["launch_fn"] = launch_fn
            adapted_args, sync_callbacks = self._adapt_runtime_arguments(args)
            try:
                return launch_fn(*adapted_args, stream=stream)
            finally:
                for sync in sync_callbacks:
                    sync()

        return launcher

    def _load_module(self, module_name: str, source_path: Path):
        with self._pythonpath():
            spec = importlib.util.spec_from_file_location(module_name, source_path)
            if spec is None or spec.loader is None:
                raise BackendNotImplementedError(f"failed to load FlyDSL module scaffold from {source_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def _adapt_runtime_arguments(self, args: tuple[Any, ...]) -> tuple[tuple[Any, ...], list[Callable[[], None]]]:
        adapted: list[Any] = []
        sync_callbacks: list[Callable[[], None]] = []
        for value in args:
            adapted_value, sync = self._adapt_argument(value)
            adapted.append(adapted_value)
            if sync is not None:
                sync_callbacks.append(sync)
        return tuple(adapted), sync_callbacks

    def _adapt_argument(self, value: Any) -> tuple[Any, Callable[[], None] | None]:
        if not isinstance(value, RuntimeTensor):
            return value, None
        torch = self._import_torch()
        if not self._torch_device_available(torch):
            raise BackendNotImplementedError(
                "flydsl_exec requires a GPU-capable torch build to adapt baybridge RuntimeTensor inputs; "
                "use device-backed DLPack inputs instead"
            )
        try:
            torch_tensor = torch.tensor(value.tolist(), dtype=self._torch_dtype(torch, value.dtype), device="cuda")
        except TypeError:
            torch_tensor = torch.tensor(value.tolist(), dtype=self._torch_dtype(torch, value.dtype))

        def sync() -> None:
            copied = torch_tensor.detach().cpu().tolist() if hasattr(torch_tensor, "detach") else torch_tensor.tolist()
            if value.shape:
                value.store(make_runtime_tensor(copied, dtype=value.dtype))
            else:
                value.store(copied)

        return torch_tensor, sync

    def _import_torch(self):
        try:
            return importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise BackendNotImplementedError(
                "flydsl_exec requires torch to adapt baybridge RuntimeTensor inputs; "
                "use baybridge.from_dlpack(...) handles or install torch in the active environment"
            ) from exc

    def _torch_dtype(self, torch: Any, dtype: str):
        mapping = {
            "i1": "bool",
            "i8": "int8",
            "i32": "int32",
            "i64": "int64",
            "f16": "float16",
            "bf16": "bfloat16",
            "f32": "float32",
        }
        try:
            return getattr(torch, mapping[dtype])
        except KeyError as exc:
            raise BackendNotImplementedError(
                f"flydsl_exec does not support RuntimeTensor adaptation for dtype '{dtype}'"
            ) from exc

    def _runtime_tensor_device_available(self) -> bool:
        try:
            torch = self._import_torch()
        except BackendNotImplementedError:
            return False
        return self._torch_device_available(torch)

    def _torch_device_available(self, torch: Any) -> bool:
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return True
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available):
            try:
                return bool(is_available())
            except Exception:
                return False
        return True

    def _requires_real_exec_gate(self, environment, ir: PortableKernelIR | None = None) -> bool:
        if not (environment.ready and environment.built_package_available):
            return False
        if ir is not None and self._bridge.has_validated_real_exec(ir):
            return False
        return True

    @contextmanager
    def _pythonpath(self) -> Iterator[None]:
        search_paths = [str(path) for path in self._bridge.search_paths()]
        previous = list(sys.path)
        try:
            if search_paths:
                sys.path[:0] = search_paths
            importlib.invalidate_caches()
            yield
        finally:
            sys.path[:] = previous
