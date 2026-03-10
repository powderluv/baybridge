from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .backend import Backend, LoweredModule
from .backends import GpuTextBackend, HipKittensExecBackend, HipKittensRefBackend, HipccExecBackend, MlirTextBackend
from .diagnostics import BackendNotImplementedError, CompilationError
from .frontend import KernelDefinition, TraceKernelLaunch
from .ir import PortableKernelIR, ScalarSpec, TensorSpec
from .runtime import LaunchConfig, Pointer, RuntimeTensor, infer_runtime_spec, normalize_runtime_argument
from .target import AMDTarget
from .tracing import IRBuilder, ScalarValue, TensorValue, tracing
from .views import TiledTensorView

_CACHE_ENV = "BAYBRIDGE_CACHE_DIR"
_DEFAULT_BACKEND = "mlir_text"


@dataclass(frozen=True)
class CompiledKernel:
    ir: PortableKernelIR | None
    target: AMDTarget
    backend_name: str
    cache_key: str
    artifact_path: Path
    lowered_path: Path | None
    lowered_module: LoweredModule | None
    from_cache: bool
    runtime_callable: Callable[..., Any] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.runtime_callable is not None:
            normalized_args = tuple(normalize_runtime_argument(arg) for arg in args)
            normalized_kwargs = {name: normalize_runtime_argument(value) for name, value in kwargs.items()}
            return self.runtime_callable(*normalized_args, **normalized_kwargs)
        raise BackendNotImplementedError(
            "portable IR compilation succeeded, but AMD lowering and runtime launch are not implemented yet"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "cache_key": self.cache_key,
            "target": self.target.to_dict(),
            "ir": self.ir.to_dict() if self.ir else None,
            "lowered_module": self.lowered_module.to_dict() if self.lowered_module else None,
        }


@dataclass(frozen=True)
class TraceBinding:
    name: str
    parameter: str
    path: tuple[int | str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter": self.parameter,
            "path": list(self.path),
        }


@dataclass(frozen=True)
class GenerateLineInfo:
    enabled: bool = True


def _path_suffix(path: tuple[int | str, ...]) -> str:
    if not path:
        return ""
    parts = [str(item).replace("-", "_") for item in path]
    return "_" + "_".join(parts)


def _default_cache_dir() -> Path:
    configured = os.environ.get(_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "baybridge"


def _resolve_backend(backend: str | Backend | None) -> tuple[str, Backend | None]:
    if backend is None:
        backend = _DEFAULT_BACKEND
    if isinstance(backend, str):
        if backend == "portable":
            return backend, None
        if backend == "mlir_text":
            return backend, MlirTextBackend()
        if backend == "gpu_text":
            return backend, GpuTextBackend()
        if backend == "hipkittens_exec":
            return backend, HipKittensExecBackend()
        if backend == "hipkittens_ref":
            return backend, HipKittensRefBackend()
        if backend == "hipcc_exec":
            return backend, HipccExecBackend()
        raise CompilationError(f"unknown backend '{backend}'")
    return backend.name, backend


def _normalize_kernel(kernel: KernelDefinition | Any) -> KernelDefinition:
    if isinstance(kernel, KernelDefinition):
        return kernel
    if callable(kernel):
        return KernelDefinition(fn=kernel, kind="kernel", launch=LaunchConfig())
    raise CompilationError("compile expects a @kernel/@jit function or a callable")


def _argument_specs(
    definition: KernelDefinition,
    sample_args: tuple[Any, ...] = (),
    sample_kwargs: dict[str, Any] | None = None,
) -> list[tuple[str, TensorSpec | ScalarSpec]]:
    signature = inspect.signature(definition.fn)
    bound_values = signature.bind_partial(*sample_args, **(sample_kwargs or {})).arguments
    bound: list[tuple[str, TensorSpec | ScalarSpec]] = []
    for name, parameter in signature.parameters.items():
        annotation = parameter.annotation
        if isinstance(annotation, (TensorSpec, ScalarSpec)):
            bound.append((name, annotation))
            continue
        if name in bound_values:
            bound.append((name, _spec_from_value(bound_values[name], parameter_name=name)))
            continue
        raise CompilationError(
            f"parameter '{name}' must be annotated with baybridge.TensorSpec(...) or baybridge.ScalarSpec(...), "
            "or compile(...) must be given sample arguments"
        )
    return bound


def _spec_from_value(value: Any, *, parameter_name: str) -> TensorSpec | ScalarSpec:
    if isinstance(value, TensorValue):
        return value.spec
    if isinstance(value, ScalarValue):
        return value.spec
    dtype, shape = infer_runtime_spec(value)
    if dtype == "python_object":
        raise CompilationError(
            f"parameter '{parameter_name}' could not be inferred for tracing; add a baybridge.TensorSpec/ScalarSpec annotation"
        )
    if shape is None:
        return ScalarSpec(dtype=dtype)
    return TensorSpec(shape=shape, dtype=dtype)


def _trace(
    definition: KernelDefinition,
    sample_args: tuple[Any, ...] = (),
    sample_kwargs: dict[str, Any] | None = None,
) -> PortableKernelIR:
    signature = inspect.signature(definition.fn)
    bound_values = signature.bind_partial(*sample_args, **(sample_kwargs or {})).arguments
    builder = IRBuilder(kernel_name=definition.__name__, launch=definition.launch)
    trace_bindings: list[TraceBinding] = []
    proxy_arguments: list[Any] = []
    proxy_keywords: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        annotation = parameter.annotation
        if isinstance(annotation, (TensorSpec, ScalarSpec)):
            value = _bind_trace_leaf(builder, trace_bindings, name, name, annotation)
        elif name in bound_values:
            value = _build_trace_proxy(name, bound_values[name], builder, trace_bindings)
        elif parameter.default is not inspect._empty:
            value = parameter.default
        else:
            raise CompilationError(
                f"parameter '{name}' must be annotated with baybridge.TensorSpec(...) or baybridge.ScalarSpec(...), "
                "or compile(...) must be given sample arguments"
            )
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            proxy_keywords[name] = value
            continue
        proxy_arguments.append(value)
    builder.metadata["trace_bindings"] = [binding.to_dict() for binding in trace_bindings]
    with tracing(builder):
        try:
            definition.fn(*proxy_arguments, **proxy_keywords)
        except TraceKernelLaunch as launch_trace:
            return _trace_launched_kernel(definition, builder, launch_trace)
    builder.metadata["frontend"] = definition.kind
    return builder.finalize()


def _rehydrate_launch_value(
    value: Any,
    *,
    tensor_values: dict[str, TensorValue],
    scalar_values: dict[str, ScalarValue],
) -> Any:
    if isinstance(value, Pointer):
        if value.tensor is None:
            return value
        return replace(
            value,
            tensor=_rehydrate_launch_value(value.tensor, tensor_values=tensor_values, scalar_values=scalar_values),
        )
    if isinstance(value, TensorValue):
        try:
            return tensor_values[value.name]
        except KeyError as exc:
            raise CompilationError(
                f"launched kernel captured tensor '{value.name}' that is not a wrapper argument; this trace path is not implemented yet"
            ) from exc
    if isinstance(value, ScalarValue):
        try:
            return scalar_values[value.name]
        except KeyError as exc:
            raise CompilationError(
                f"launched kernel captured scalar '{value.name}' from the wrapper; this trace path is not implemented yet"
            ) from exc
    if isinstance(value, TiledTensorView):
        return TiledTensorView(
            base=_rehydrate_launch_value(value.base, tensor_values=tensor_values, scalar_values=scalar_values),
            tile=value.tile,
            rest_shape=value.rest_shape,
            rest_axes_map=value.rest_axes_map,
        )
    if isinstance(value, tuple):
        return tuple(_rehydrate_launch_value(item, tensor_values=tensor_values, scalar_values=scalar_values) for item in value)
    if isinstance(value, list):
        return [_rehydrate_launch_value(item, tensor_values=tensor_values, scalar_values=scalar_values) for item in value]
    if isinstance(value, dict):
        return {
            key: _rehydrate_launch_value(item, tensor_values=tensor_values, scalar_values=scalar_values)
            for key, item in value.items()
        }
    return value


def _trace_launched_kernel(
    wrapper_definition: KernelDefinition,
    wrapper_builder: IRBuilder,
    launch_trace: TraceKernelLaunch,
) -> PortableKernelIR:
    builder = IRBuilder(kernel_name=launch_trace.definition.__name__, launch=launch_trace.launch)
    tensor_values: dict[str, TensorValue] = {}
    scalar_values: dict[str, ScalarValue] = {}
    for argument in wrapper_builder.arguments:
        value = builder.bind_argument(argument.name, argument.spec)
        if isinstance(value, TensorValue):
            tensor_values[value.name] = value
        else:
            scalar_values[value.name] = value
    proxy_arguments = [
        _rehydrate_launch_value(value, tensor_values=tensor_values, scalar_values=scalar_values)
        for value in launch_trace.args
    ]
    proxy_kwargs = {
        name: _rehydrate_launch_value(value, tensor_values=tensor_values, scalar_values=scalar_values)
        for name, value in launch_trace.kwargs.items()
    }
    with tracing(builder):
        launch_trace.definition.fn(*proxy_arguments, **proxy_kwargs)
    builder.metadata["frontend"] = launch_trace.definition.kind
    builder.metadata["compiled_from"] = "launch_wrapper"
    builder.metadata["trace_bindings"] = list(wrapper_builder.metadata.get("trace_bindings", []))
    builder.metadata["wrapped_by"] = wrapper_definition.__name__
    builder.metadata["wrapper_frontend"] = wrapper_definition.kind
    return builder.finalize()


def _bind_trace_leaf(
    builder: IRBuilder,
    trace_bindings: list[TraceBinding],
    parameter_name: str,
    leaf_name: str,
    spec: TensorSpec | ScalarSpec,
    *,
    path: tuple[int | str, ...] = (),
) -> TensorValue | ScalarValue:
    trace_bindings.append(TraceBinding(name=leaf_name, parameter=parameter_name, path=path))
    return builder.bind_argument(leaf_name, spec)


def _container_has_tensor_leaves(value: Any) -> bool:
    if isinstance(value, (TensorValue, RuntimeTensor)):
        return True
    if isinstance(value, Pointer) and value.tensor is not None:
        return _container_has_tensor_leaves(value.tensor)
    if isinstance(value, (list, tuple)):
        return any(_container_has_tensor_leaves(item) for item in value)
    if isinstance(value, dict):
        return any(_container_has_tensor_leaves(item) for item in value.values())
    return False


def _build_trace_proxy(
    parameter_name: str,
    value: Any,
    builder: IRBuilder,
    trace_bindings: list[TraceBinding],
    *,
    path: tuple[int | str, ...] = (),
) -> Any:
    if isinstance(value, Pointer):
        if value.tensor is None:
            return value
        tensor_proxy = _build_trace_proxy(parameter_name, value.tensor, builder, trace_bindings, path=path + ("tensor",))
        return replace(value, tensor=tensor_proxy)
    if isinstance(value, TensorValue):
        return value
    if isinstance(value, ScalarValue):
        return value
    dtype, shape = infer_runtime_spec(value)
    if dtype != "python_object":
        spec: TensorSpec | ScalarSpec
        if shape is None:
            spec = ScalarSpec(dtype=dtype)
        else:
            spec = TensorSpec(shape=shape, dtype=dtype)
        leaf_name = parameter_name if not path else f"{parameter_name}{_path_suffix(path)}"
        return _bind_trace_leaf(builder, trace_bindings, parameter_name, leaf_name, spec, path=path)
    if isinstance(value, list):
        if not _container_has_tensor_leaves(value):
            return value
        return [
            _build_trace_proxy(parameter_name, item, builder, trace_bindings, path=path + (index,))
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        if not _container_has_tensor_leaves(value):
            return value
        return tuple(
            _build_trace_proxy(parameter_name, item, builder, trace_bindings, path=path + (index,))
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if not _container_has_tensor_leaves(value):
            return value
        return {
            key: _build_trace_proxy(parameter_name, item, builder, trace_bindings, path=path + (key,))
            for key, item in value.items()
        }
    return value


def _extract_bound_value(value: Any, path: tuple[int | str, ...]) -> Any:
    extracted = value
    for item in path:
        if isinstance(extracted, dict):
            extracted = extracted[item]
            continue
        if isinstance(item, str) and hasattr(extracted, item):
            extracted = getattr(extracted, item)
            continue
        extracted = extracted[item]
    return extracted


def _wrap_backend_launcher(
    definition: KernelDefinition,
    backend_launcher: Callable[..., Any],
    trace_bindings: list[dict[str, Any]],
) -> Callable[..., Any]:
    signature = inspect.signature(definition.fn)

    def launcher(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs).arguments
        extracted_args = [
            normalize_runtime_argument(
                _extract_bound_value(bound[binding["parameter"]], tuple(binding["path"]))
            )
            for binding in trace_bindings
        ]
        return backend_launcher(*extracted_args)

    return launcher


def _cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trace_cache_payload(ir: PortableKernelIR, target: AMDTarget, backend_name: str) -> dict[str, Any]:
    return {
        "backend_name": backend_name,
        "ir": ir.to_dict(),
        "mode": "traced",
        "target": target.to_dict(),
    }


def _runtime_cache_payload(
    definition: KernelDefinition,
    target: AMDTarget,
    backend_name: str,
    sample_args: tuple[Any, ...],
) -> dict[str, Any]:
    try:
        source = inspect.getsource(definition.fn)
    except (OSError, TypeError):
        source = definition.fn.__qualname__
    argument_specs = [
        {"name": name, "spec": spec.to_dict()}
        for name, spec in _argument_specs(definition, sample_args)
    ]
    return {
        "backend_name": backend_name,
        "function": definition.fn.__qualname__,
        "kind": definition.kind,
        "launch": definition.launch.to_dict(),
        "mode": "runtime_only",
        "source": source,
        "specialization": argument_specs,
        "target": target.to_dict(),
    }


def _can_runtime_compile(definition: KernelDefinition, sample_args: tuple[Any, ...]) -> bool:
    if definition.kind != "jit":
        return False
    try:
        _argument_specs(definition, sample_args)
    except CompilationError:
        return len(inspect.signature(definition.fn).parameters) == 0
    return True


def _execute_definition(definition: KernelDefinition, *args: Any, **kwargs: Any) -> Any:
    if definition.kind == "kernel":
        launch = definition(*args, **kwargs)
        return launch.launch(
            grid=definition.launch.grid,
            block=definition.launch.block,
            shared_mem_bytes=definition.launch.shared_mem_bytes,
        )
    return definition(*args, **kwargs)


def _write_artifact(
    manifest_path: Path,
    *,
    target: AMDTarget,
    backend_name: str,
    cache_key: str,
    ir: PortableKernelIR | None,
    lowered_module: LoweredModule | None,
    lowered_path: Path | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend_name": backend_name,
        "cache_key": cache_key,
        "target": target.to_dict(),
        "ir": ir.to_dict() if ir else None,
        "lowered_module": lowered_module.to_dict() if lowered_module else None,
    }
    if metadata:
        payload["metadata"] = metadata
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if lowered_module and lowered_path is not None:
        lowered_path.write_text(lowered_module.text, encoding="utf-8")


def _artifact_paths(
    cache_dir: Path,
    target: AMDTarget,
    backend_name: str,
    cache_key: str,
    *,
    artifact_extension: str = ".mlir",
) -> tuple[Path, Path | None]:
    root = cache_dir / backend_name / target.arch
    manifest_path = root / f"{cache_key}.json"
    lowered_path = root / f"{cache_key}{artifact_extension}" if backend_name != "portable" else None
    return manifest_path, lowered_path


def compile(
    kernel: KernelDefinition | Any,
    *sample_args: Any,
    target: AMDTarget | None = None,
    backend: str | Backend | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    use_cache: bool = True,
) -> CompiledKernel:
    definition = _normalize_kernel(kernel)
    selected_target = target or AMDTarget()
    backend_name, resolved_backend = _resolve_backend(backend)
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    normalized_sample_args = tuple(normalize_runtime_argument(arg) for arg in sample_args)
    ir: PortableKernelIR | None = None
    lowered_module: LoweredModule | None = None
    runtime_metadata: dict[str, Any] | None = None
    artifact_extension = getattr(resolved_backend, "artifact_extension", ".mlir")
    try:
        ir = _trace(definition, normalized_sample_args)
    except CompilationError as exc:
        if not _can_runtime_compile(definition, normalized_sample_args):
            raise
        runtime_metadata = {"mode": "runtime_only", "trace_error": str(exc)}
    if ir is not None:
        cache_key = _cache_key(_trace_cache_payload(ir, selected_target, backend_name))
        lowered_module = resolved_backend.lower(ir, selected_target) if resolved_backend else None
    else:
        cache_key = _cache_key(_runtime_cache_payload(definition, selected_target, backend_name, normalized_sample_args))
    artifact_path, lowered_path = _artifact_paths(
        resolved_cache_dir,
        selected_target,
        backend_name,
        cache_key,
        artifact_extension=artifact_extension,
    )
    if ir is None:
        lowered_path = None
    from_cache = use_cache and artifact_path.exists()
    if not from_cache:
        _write_artifact(
            artifact_path,
            target=selected_target,
            backend_name=backend_name,
            cache_key=cache_key,
            ir=ir,
            lowered_module=lowered_module,
            lowered_path=lowered_path,
            metadata=runtime_metadata,
        )
    launcher_callable = lambda *args, **kwargs: _execute_definition(definition, *args, **kwargs)
    if ir is not None and lowered_module is not None and lowered_path is not None and hasattr(resolved_backend, "build_launcher"):
        backend_launcher = resolved_backend.build_launcher(ir, selected_target, lowered_module, lowered_path)
        trace_bindings = ir.metadata.get("trace_bindings")
        if isinstance(trace_bindings, list) and trace_bindings:
            launcher_callable = _wrap_backend_launcher(definition, backend_launcher, trace_bindings)
        else:
            launcher_callable = backend_launcher
    return CompiledKernel(
        ir=ir,
        target=selected_target,
        backend_name=backend_name,
        cache_key=cache_key,
        artifact_path=artifact_path,
        lowered_path=lowered_path,
        lowered_module=lowered_module,
        from_cache=from_cache,
        runtime_callable=launcher_callable,
    )


def _normalize_compile_options(options: Any) -> tuple[Any, ...]:
    if isinstance(options, tuple):
        normalized = options
    else:
        normalized = (options,)
    for option in normalized:
        if not isinstance(option, GenerateLineInfo):
            raise CompilationError(f"unsupported compile option '{type(option).__name__}'")
    return normalized


class _CompileDispatcher:
    def __init__(self, fn: Callable[..., CompiledKernel]):
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> CompiledKernel:
        return self._fn(*args, **kwargs)

    def __getitem__(self, options: Any) -> Callable[..., CompiledKernel]:
        _normalize_compile_options(options)

        def compile_with_options(*args: Any, **kwargs: Any) -> CompiledKernel:
            return self._fn(*args, **kwargs)

        return compile_with_options


compile = _CompileDispatcher(compile)
