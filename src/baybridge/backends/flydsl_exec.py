from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..ir import PortableKernelIR
from ..runtime import RuntimeTensor, tensor as make_runtime_tensor
from ..target import AMDTarget
from .flydsl_bridge import FlyDslBridge


class FlyDslExecBackend:
    name = "flydsl_exec"
    artifact_extension = ".fly.py"

    def __init__(self) -> None:
        self._bridge = FlyDslBridge()

    def available(self, target: AMDTarget | None = None) -> bool:
        del target
        return self._bridge.exec_environment().ready

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        return self._bridge.supports_exec(ir, target)

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
