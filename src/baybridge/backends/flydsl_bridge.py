from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..backend import LoweredModule
from ..ir import KernelArgument, Layout, Operation, PortableKernelIR, ScalarSpec, TensorSpec
from ..target import AMDTarget

_FLYDSL_ENV = "BAYBRIDGE_FLYDSL_ROOT"

_SUPPORTED_OPS = {
    "add",
    "and",
    "atan2",
    "barrier",
    "bitnot",
    "block_dim",
    "block_idx",
    "cast",
    "commit_group",
    "compare",
    "constant",
    "copy",
    "copy_async",
    "cos",
    "div",
    "elect_one",
    "erf",
    "exp",
    "exp2",
    "floordiv",
    "gemm",
    "grid_dim",
    "group_modes",
    "lane_id",
    "load",
    "local_tile",
    "log",
    "log10",
    "log2",
    "make_tensor",
    "math_rsqrt",
    "math_sqrt",
    "mma",
    "mod",
    "mul",
    "neg",
    "or",
    "partition",
    "program_id",
    "reduce_add",
    "reduce_max",
    "reduce_min",
    "reduce_mul",
    "select",
    "slice",
    "sin",
    "store",
    "sub",
    "sync_grid",
    "sync_warp",
    "thread_fragment_load",
    "thread_fragment_store",
    "thread_idx",
    "wait_group",
    "wave_id",
}

_EXEC_SUPPORTED_OPS = {
    "add",
    "and",
    "barrier",
    "bitnot",
    "block_dim",
    "block_idx",
    "compare",
    "constant",
    "copy",
    "div",
    "fill",
    "floordiv",
    "grid_dim",
    "load",
    "make_tensor",
    "math_atan2",
    "math_cos",
    "math_erf",
    "math_exp",
    "math_exp2",
    "math_log",
    "math_log10",
    "math_log2",
    "math_rsqrt",
    "math_sin",
    "math_sqrt",
    "masked_load",
    "masked_store",
    "mod",
    "mul",
    "neg",
    "or",
    "program_id",
    "reduce_add",
    "reduce_max",
    "reduce_min",
    "reduce_mul",
    "select",
    "store",
    "sub",
    "broadcast_to",
    "tensor_add",
    "tensor_div",
    "tensor_mul",
    "tensor_sub",
    "thread_idx",
}


@dataclass(frozen=True)
class FlyDslMatch:
    family: str
    op_counts: dict[str, int]
    configured_root: str | None
    import_available: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FlyDslExecEnvironment:
    configured_root: str | None
    search_paths: tuple[str, ...]
    library_paths: tuple[str, ...]
    active_spec_available: bool
    torch_available: bool
    built_package_available: bool
    expr_import_available: bool
    compiler_import_available: bool
    ready: bool
    import_error: str | None
    notes: tuple[str, ...]


class FlyDslBridge:
    def reference_available(self) -> bool:
        return self._configured_root() is not None

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        del target
        return all(operation.op in _SUPPORTED_OPS for operation in ir.operations)

    def lower(self, ir: PortableKernelIR, target: AMDTarget, *, backend_name: str) -> LoweredModule:
        match = self.analyze(ir)
        text = self.render(ir, target, match, backend_name=backend_name)
        return LoweredModule(
            backend_name=backend_name,
            entry_point=ir.name,
            dialect="flydsl_ref",
            text=text,
        )

    def supports_exec(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        del target
        if not all(operation.op in _EXEC_SUPPORTED_OPS or operation.op.startswith("cmp_") for operation in ir.operations):
            return False
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return False
        for operation in ir.operations:
            if operation.op == "make_tensor":
                if operation.attrs.get("dynamic_shared"):
                    return False
                if operation.attrs.get("address_space") not in {"shared", "register"}:
                    return False
            if operation.op == "barrier" and operation.attrs.get("kind", "block") == "grid":
                return False
        return True

    def lower_exec(self, ir: PortableKernelIR, target: AMDTarget, *, backend_name: str) -> LoweredModule:
        match = self.analyze(ir)
        text = self.render_exec_python(ir, target, match, backend_name=backend_name)
        return LoweredModule(
            backend_name=backend_name,
            entry_point=ir.name,
            dialect="flydsl_python",
            text=text,
        )

    def analyze(self, ir: PortableKernelIR) -> FlyDslMatch:
        op_counts = Counter(operation.op for operation in ir.operations)
        family = self._family(op_counts)
        notes: list[str] = []
        if op_counts.get("local_tile") or op_counts.get("partition") or op_counts.get("group_modes"):
            notes.append("layout and tiled-view operations are present")
        if op_counts.get("copy_async") or op_counts.get("wait_group") or op_counts.get("commit_group"):
            notes.append("async copy/pipeline operations are present")
        if op_counts.get("mma"):
            notes.append("mma or gemm lowering is present")
        if op_counts.get("thread_fragment_load") or op_counts.get("thread_fragment_store"):
            notes.append("thread-value fragment movement is present")
        root = self._configured_root()
        return FlyDslMatch(
            family=family,
            op_counts=dict(sorted(op_counts.items())),
            configured_root=str(root) if root is not None else None,
            import_available=importlib.util.find_spec("flydsl") is not None,
            notes=tuple(notes),
        )

    def exec_environment(self) -> FlyDslExecEnvironment:
        root = self._configured_root()
        search_paths = tuple(str(path) for path in self.search_paths())
        library_paths = tuple(str(path) for path in self.library_paths())
        active_spec_available = importlib.util.find_spec("flydsl") is not None
        torch_available = importlib.util.find_spec("torch") is not None
        built_package_available = any((Path(path) / "flydsl" / "_mlir").exists() for path in search_paths)
        expr_import_available, compiler_import_available, import_error = self._probe_import(
            tuple(Path(path) for path in search_paths),
            tuple(Path(path) for path in library_paths),
        )
        notes: list[str] = []
        if root is not None and not built_package_available:
            notes.append("configured FlyDSL root does not contain a built python_packages tree")
        if not torch_available:
            notes.append("torch is not importable in the current Python environment")
        if built_package_available and library_paths:
            notes.append("embedded FlyDSL MLIR libs may require LD_LIBRARY_PATH to be set before starting Python")
        if import_error:
            notes.append(f"import probe failed: {import_error}")
        if root is None and not active_spec_available:
            notes.append(f"set {_FLYDSL_ENV} to a built FlyDSL checkout or install flydsl in the active environment")
        ready = expr_import_available and compiler_import_available
        return FlyDslExecEnvironment(
            configured_root=str(root) if root is not None else None,
            search_paths=search_paths,
            library_paths=library_paths,
            active_spec_available=active_spec_available,
            torch_available=torch_available,
            built_package_available=built_package_available,
            expr_import_available=expr_import_available,
            compiler_import_available=compiler_import_available,
            ready=ready,
            import_error=import_error,
            notes=tuple(notes),
        )

    def render(self, ir: PortableKernelIR, target: AMDTarget, match: FlyDslMatch, *, backend_name: str) -> str:
        manifest = {
            "backend": backend_name,
            "entry_point": ir.name,
            "family": match.family,
            "target_arch": target.arch,
            "wave_size": target.wave_size,
            "configured_flydsl_root": match.configured_root,
            "flydsl_import_available": match.import_available,
            "op_counts": match.op_counts,
            "notes": list(match.notes),
        }
        manifest_text = "\n".join(f"// {line}" for line in json.dumps(manifest, indent=2, sort_keys=True).splitlines())
        root_hint = (
            f"// Build hint: use FlyDSL from {match.configured_root}\n"
            if match.configured_root
            else f"// Build hint: set {_FLYDSL_ENV} to a FlyDSL checkout or install flydsl in the active environment\n"
        )
        ops = "\n".join(f"    {self._render_operation(operation)}" for operation in ir.operations) or "    fly.return"
        arguments = ", ".join(self._render_argument(argument) for argument in ir.arguments)
        return (
            f"{manifest_text}\n"
            f"{root_hint}"
            f"module attributes {{ baybridge.target = \"{target.arch}\", baybridge.backend = \"{backend_name}\" }} {{\n"
            f"  fly.kernel @{ir.name}({arguments}) attributes {{ "
            f"grid = {list(ir.launch.grid)}, block = {list(ir.launch.block)}, "
            f"shared_mem_bytes = {ir.launch.shared_mem_bytes}"
            f"{', cooperative = true' if ir.launch.cooperative else ''} }} {{\n"
            f"{ops}\n"
            "  }\n"
            "}\n"
        )

    def _family(self, op_counts: Counter[str]) -> str:
        if op_counts.get("mma") or op_counts.get("gemm"):
            return "mfma_gemm"
        if op_counts.get("local_tile") or op_counts.get("partition") or op_counts.get("group_modes"):
            return "layout_tiled"
        if any(name.startswith("reduce_") for name in op_counts):
            return "elementwise_reduce"
        return "elementwise"

    def _configured_root(self) -> Path | None:
        configured = os.environ.get(_FLYDSL_ENV)
        if not configured:
            return None
        path = Path(configured).expanduser().resolve()
        if (path / "README.md").exists() and (
            (path / "python").exists()
            or (path / "build-fly").exists()
            or any(candidate.is_dir() for candidate in path.glob("build-*"))
        ):
            return path
        return None

    def search_paths(self) -> list[Path]:
        root = self._configured_root()
        if root is None:
            return []
        candidates: list[Path] = []
        for name in ("build-fly", "build"):
            candidates.append(root / name / "python_packages")
        candidates.extend(
            path / "python_packages"
            for path in sorted(root.glob("build-*"))
            if (path / "python_packages").exists()
        )
        candidates.extend([root / "python", root])
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path.exists() and path not in seen:
                unique.append(path)
                seen.add(path)
        return unique

    def library_paths(self) -> list[Path]:
        root = self._configured_root()
        if root is None:
            return []
        candidates: list[Path] = []
        for name in ("build-fly", "build"):
            candidates.append(root / name / "python_packages" / "flydsl" / "_mlir" / "_mlir_libs")
        candidates.extend(
            path / "python_packages" / "flydsl" / "_mlir" / "_mlir_libs"
            for path in sorted(root.glob("build-*"))
            if (path / "python_packages" / "flydsl" / "_mlir" / "_mlir_libs").exists()
        )
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path.exists() and path not in seen:
                unique.append(path)
                seen.add(path)
        return unique

    def _render_argument(self, argument: KernelArgument) -> str:
        return f"%{argument.name}: {self._render_spec(argument.spec)}"

    def _render_spec(self, spec: TensorSpec | ScalarSpec) -> str:
        if isinstance(spec, ScalarSpec):
            return spec.dtype
        layout = spec.resolved_layout()
        mem = spec.address_space.value
        return f"!fly.tensor<{list(spec.shape)}x{spec.dtype}, stride={list(layout.stride)}, memory={mem}>"

    def _render_result_type(self, operation: Operation) -> str | None:
        result = operation.attrs.get("result")
        if not isinstance(result, dict):
            return None
        kind = result.get("kind")
        if kind == "scalar":
            return str(result["dtype"])
        if kind == "tensor":
            shape = list(result["shape"])
            dtype = str(result["dtype"])
            layout_dict = result.get("layout")
            stride = list(layout_dict["stride"]) if isinstance(layout_dict, dict) else None
            memory = str(result.get("address_space", "global"))
            suffix = f", stride={stride}" if stride is not None else ""
            return f"!fly.tensor<{shape}x{dtype}{suffix}, memory={memory}>"
        return None

    def _render_attrs(self, operation: Operation) -> str:
        attrs: list[str] = []
        for key, value in sorted(operation.attrs.items()):
            if key == "result":
                continue
            attrs.append(f"{key}={value!r}")
        return f" {{{', '.join(attrs)}}}" if attrs else ""

    def _render_operation(self, operation: Operation) -> str:
        result_type = self._render_result_type(operation)
        rendered_inputs = ", ".join(f"%{name}" for name in operation.inputs)
        attrs = self._render_attrs(operation)
        if operation.outputs:
            rendered_outputs = ", ".join(f"%{name}" for name in operation.outputs)
            if result_type is not None:
                return f"{rendered_outputs} = fly.{operation.op}({rendered_inputs}){attrs} : {result_type}"
            return f"{rendered_outputs} = fly.{operation.op}({rendered_inputs}){attrs}"
        return f"fly.{operation.op}({rendered_inputs}){attrs}"

    def render_exec_python(self, ir: PortableKernelIR, target: AMDTarget, match: FlyDslMatch, *, backend_name: str) -> str:
        environment = self.exec_environment()
        manifest = {
            "backend": backend_name,
            "entry_point": ir.name,
            "family": match.family,
            "target_arch": target.arch,
            "wave_size": target.wave_size,
            "configured_flydsl_root": match.configured_root,
            "flydsl_import_available": match.import_available,
            "flydsl_exec_ready": environment.ready,
            "flydsl_torch_available": environment.torch_available,
            "flydsl_search_paths": list(environment.search_paths),
            "flydsl_import_error": environment.import_error,
            "op_counts": match.op_counts,
            "notes": list(match.notes),
        }
        manifest_text = "\n".join(f"# {line}" for line in json.dumps(manifest, indent=2, sort_keys=True).splitlines())
        args = ", ".join(argument.name for argument in ir.arguments)
        kernel_args = ", ".join(f"{argument.name}: fx.Tensor" for argument in ir.arguments)
        launch_grid = tuple(ir.launch.grid)
        launch_block = tuple(ir.launch.block)
        kernel_body = "\n".join(f"    {line}" for line in self._render_exec_body(ir))
        launch_kwargs = [f"grid={launch_grid}", f"block={launch_block}"]
        if ir.launch.shared_mem_bytes > 0:
            launch_kwargs.append(f"smem={ir.launch.shared_mem_bytes}")
        launch_kwargs.append("stream=stream")
        root_hint = (
            f"# FlyDSL root hint: {match.configured_root}\n"
            if match.configured_root
            else f"# FlyDSL root hint: set {_FLYDSL_ENV} or install flydsl in the active environment\n"
        )
        return (
            f"{manifest_text}\n"
            f"{root_hint}"
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n\n"
            "import math\n\n"
            "gpu = getattr(fx, 'gpu', fx)\n\n"
            "def _baybridge_adapt_tensor_arg(value):\n"
            "    if hasattr(value, '__fly_types__') or hasattr(value, '__fly_ptrs__'):\n"
            "        return value\n"
            "    if hasattr(value, '__dlpack__') and hasattr(value, '__dlpack_device__'):\n"
            "        return flyc.from_dlpack(value)\n"
            "    raise TypeError('flydsl_exec currently requires FlyDSL JIT arguments or DLPack-capable tensor inputs')\n\n"
            f"@flyc.kernel\n"
            f"def {ir.name}_kernel({kernel_args}):\n"
            f"{kernel_body}\n\n"
            f"@flyc.jit\n"
            f"def {ir.name}_jit({kernel_args}, stream: fx.Stream = fx.Stream(None)):\n"
            f"    return {ir.name}_kernel({args}).launch({', '.join(launch_kwargs)})\n\n"
            f"def launch_{ir.name}({args}, stream=None):\n"
            + "".join(f"    {argument.name} = _baybridge_adapt_tensor_arg({argument.name})\n" for argument in ir.arguments)
            + "    if stream is not None and not isinstance(stream, fx.Stream):\n"
            + "        stream = fx.Stream(stream)\n"
            + f"    return {ir.name}_jit({args}, stream=stream if stream is not None else fx.Stream(None))\n"
        )

    def _render_exec_body(self, ir: PortableKernelIR) -> list[str]:
        tensor_specs = self._collect_exec_tensor_specs(ir)
        memref_names = set(tensor_specs)
        lines: list[str] = []
        for operation in ir.operations:
            lines.extend(self._render_exec_operation(operation, tensor_specs=tensor_specs, memref_names=memref_names))
        if not lines:
            lines.append("pass")
        return lines

    def _collect_exec_tensor_specs(self, ir: PortableKernelIR) -> dict[str, TensorSpec]:
        specs = {
            argument.name: argument.spec
            for argument in ir.arguments
            if isinstance(argument.spec, TensorSpec)
        }
        for operation in ir.operations:
            if operation.op == "make_tensor":
                specs[operation.outputs[0]] = self._tensor_spec_from_attrs(operation.attrs)
                continue
            result = operation.attrs.get("result")
            if operation.outputs and isinstance(result, dict) and result.get("kind") == "tensor":
                specs[operation.outputs[0]] = self._tensor_spec_from_result(result)
        return specs

    def _tensor_spec_from_attrs(self, attrs: dict[str, Any]) -> TensorSpec:
        layout_dict = attrs["layout"]
        layout = Layout(
            shape=tuple(layout_dict["shape"]),
            stride=tuple(layout_dict["stride"]),
            swizzle=layout_dict.get("swizzle"),
        )
        return TensorSpec(
            shape=tuple(attrs["shape"]),
            dtype=str(attrs["dtype"]),
            layout=layout,
            address_space=str(attrs["address_space"]),
        )

    def _tensor_spec_from_result(self, result: dict[str, Any]) -> TensorSpec:
        layout_dict = result["layout"]
        layout = Layout(
            shape=tuple(layout_dict["shape"]),
            stride=tuple(layout_dict["stride"]),
            swizzle=layout_dict.get("swizzle"),
        )
        return TensorSpec(
            shape=tuple(result["shape"]),
            dtype=str(result["dtype"]),
            layout=layout,
            address_space=str(result["address_space"]),
        )

    def _render_exec_operation(
        self,
        operation: Operation,
        *,
        tensor_specs: dict[str, TensorSpec],
        memref_names: set[str],
    ) -> list[str]:
        if operation.op in {"thread_idx", "block_idx", "block_dim", "grid_dim", "program_id"}:
            builtin = {
                "thread_idx": "thread_idx",
                "block_idx": "block_idx",
                "block_dim": "block_dim",
                "grid_dim": "grid_dim",
                "program_id": "block_idx",
            }[operation.op]
            axis = operation.attrs["axis"]
            return [f"{operation.outputs[0]} = fx.{builtin}.{axis}"]
        if operation.op == "constant":
            return [f"{operation.outputs[0]} = {operation.attrs['value']!r}"]
        if operation.op == "make_tensor":
            spec = tensor_specs[operation.outputs[0]]
            shape_expr = self._render_exec_shape_tuple(spec.shape)
            stride_expr = self._render_exec_shape_tuple(spec.resolved_layout().stride)
            layout_name = f"{operation.outputs[0]}_layout"
            type_name = f"{operation.outputs[0]}_type"
            address_space = self._render_exec_address_space(spec)
            dtype_expr = self._render_exec_dtype(spec.dtype)
            return [
                f"{layout_name} = fx.make_layout({shape_expr}, {stride_expr})",
                f"{type_name} = fx.MemRefType.get({dtype_expr}, fx.LayoutType.get({shape_expr}, {stride_expr}), {address_space})",
                f"{operation.outputs[0]} = fx.memref_alloca({type_name}, {layout_name})",
            ]
        if operation.op in {
            "math_sqrt",
            "math_rsqrt",
            "math_sin",
            "math_cos",
            "math_exp",
            "math_exp2",
            "math_log",
            "math_log2",
            "math_log10",
            "math_erf",
        }:
            source_name = operation.inputs[0]
            if operation.outputs[0] in tensor_specs:
                return self._render_exec_tensor_unary_math(
                    operation.op,
                    source_name,
                    tensor_specs[source_name],
                    operation.outputs[0],
                    tensor_specs[operation.outputs[0]],
                    memref_names=memref_names,
                )
            return [f"{operation.outputs[0]} = {self._render_exec_scalar_unary_math_expr(operation.op, source_name)}"]
        if operation.op == "math_atan2":
            lhs_name, rhs_name = operation.inputs
            if operation.outputs[0] in tensor_specs:
                return self._render_exec_tensor_binary_math(
                    operation.op,
                    lhs_name,
                    rhs_name,
                    tensor_specs=tensor_specs,
                    output_name=operation.outputs[0],
                    output_spec=tensor_specs[operation.outputs[0]],
                    memref_names=memref_names,
                )
            return [f"{operation.outputs[0]} = math.atan2({lhs_name}, {rhs_name})"]
        if operation.op in {"add", "sub", "mul", "div", "floordiv", "mod", "and", "or"}:
            lhs, rhs = operation.inputs
            symbol = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "floordiv": "//",
                "mod": "%",
                "and": "&",
                "or": "|",
            }[operation.op]
            return [f"{operation.outputs[0]} = {lhs} {symbol} {rhs}"]
        if operation.op == "broadcast_to":
            source_name = operation.inputs[0]
            return self._render_exec_broadcast_to(
                source_name,
                tensor_specs[source_name],
                operation.outputs[0],
                tensor_specs[operation.outputs[0]],
                memref_names=memref_names,
            )
        if operation.op in {"tensor_add", "tensor_sub", "tensor_mul", "tensor_div"}:
            lhs_name, rhs_name = operation.inputs
            return self._render_exec_tensor_binary(
                operation.op,
                lhs_name,
                rhs_name,
                tensor_specs=tensor_specs,
                output_name=operation.outputs[0],
                output_spec=tensor_specs[operation.outputs[0]],
                memref_names=memref_names,
            )
        if operation.op == "neg":
            return [f"{operation.outputs[0]} = -{operation.inputs[0]}"]
        if operation.op == "bitnot":
            return [f"{operation.outputs[0]} = ~{operation.inputs[0]}"]
        if operation.op == "copy":
            src, dst = operation.inputs
            return self._render_exec_copy(src, dst, tensor_specs=tensor_specs, memref_names=memref_names)
        if operation.op == "fill":
            tensor_name, value_name = operation.inputs
            return self._render_exec_fill(
                tensor_name,
                tensor_specs[tensor_name],
                value_name,
                memref_names=memref_names,
            )
        if operation.op in {"reduce_add", "reduce_mul", "reduce_max", "reduce_min"}:
            source_name, init_name = operation.inputs
            if operation.outputs[0] in tensor_specs:
                return self._render_exec_reduce_tensor(
                    operation,
                    source_name,
                    tensor_specs[source_name],
                    init_name,
                    operation.outputs[0],
                    tensor_specs[operation.outputs[0]],
                    memref_names=memref_names,
                )
            return self._render_exec_reduce_scalar(
                operation,
                source_name,
                tensor_specs[source_name],
                init_name,
                memref_names=memref_names,
            )
        if operation.op == "barrier":
            kind = operation.attrs.get("kind", "block")
            if kind in {"block", "warp"}:
                return ["gpu.barrier()"]
            raise ValueError(f"flydsl_exec does not support barrier kind '{kind}'")
        if operation.op.startswith("cmp_"):
            lhs, rhs = operation.inputs
            symbol = {
                "cmp_lt": "<",
                "cmp_le": "<=",
                "cmp_gt": ">",
                "cmp_ge": ">=",
                "cmp_eq": "==",
                "cmp_ne": "!=",
            }[operation.op]
            return [f"{operation.outputs[0]} = {lhs} {symbol} {rhs}"]
        if operation.op == "select":
            pred, true_value, false_value = operation.inputs
            return [f"{operation.outputs[0]} = {true_value} if {pred} else {false_value}"]
        if operation.op == "load":
            tensor, *indices = operation.inputs
            return [f"{operation.outputs[0]} = {self._render_exec_load(tensor, indices, memref_names)}"]
        if operation.op == "masked_load":
            tensor, *rest = operation.inputs
            predicate = rest[-2]
            fallback = rest[-1]
            indices = rest[:-2]
            return [f"{operation.outputs[0]} = {self._render_exec_load(tensor, indices, memref_names)} if {predicate} else {fallback}"]
        if operation.op == "store":
            value, tensor, *indices = operation.inputs
            return [self._render_exec_store(tensor, value, indices, memref_names)]
        if operation.op == "masked_store":
            value, tensor, *rest = operation.inputs
            predicate = rest[-1]
            indices = rest[:-1]
            return [
                f"if {predicate}:",
                f"    {self._render_exec_store(tensor, value, indices, memref_names)}",
            ]
        raise ValueError(f"flydsl_exec does not support operation '{operation.op}'")

    def _render_exec_index(self, indices: list[str] | tuple[str, ...]) -> str:
        if len(indices) == 1:
            return indices[0]
        return ", ".join(indices)

    def _render_exec_shape_tuple(self, values: tuple[int, ...]) -> str:
        if len(values) == 1:
            return str(values[0])
        return repr(tuple(values))

    def _render_exec_dtype(self, dtype: str) -> str:
        mapping = {
            "f32": "fx.T.f32()",
            "f16": "fx.T.f16()",
            "bf16": "fx.T.bf16()",
            "i8": "fx.T.i8()",
            "i32": "fx.T.i32()",
            "i64": "fx.T.i64()",
        }
        try:
            return mapping[dtype]
        except KeyError as exc:
            raise ValueError(f"flydsl_exec does not support dtype '{dtype}' for memref allocation") from exc

    def _render_exec_address_space(self, spec: TensorSpec) -> str:
        if spec.address_space.value == "shared":
            return "fx.AddressSpace.Workgroup"
        if spec.address_space.value == "register":
            return "fx.AddressSpace.Register"
        raise ValueError(f"flydsl_exec does not support address_space '{spec.address_space.value}' for make_tensor")

    def _render_exec_memref_alloca(self, name: str, spec: TensorSpec) -> list[str]:
        shape_expr = self._render_exec_shape_tuple(spec.shape)
        stride_expr = self._render_exec_shape_tuple(spec.resolved_layout().stride)
        layout_name = f"{name}_layout"
        type_name = f"{name}_type"
        address_space = self._render_exec_address_space(spec)
        dtype_expr = self._render_exec_dtype(spec.dtype)
        return [
            f"{layout_name} = fx.make_layout({shape_expr}, {stride_expr})",
            f"{type_name} = fx.MemRefType.get({dtype_expr}, fx.LayoutType.get({shape_expr}, {stride_expr}), {address_space})",
            f"{name} = fx.memref_alloca({type_name}, {layout_name})",
        ]

    def _render_exec_load(self, tensor: str, indices: list[str] | tuple[str, ...], memref_names: set[str]) -> str:
        if tensor in memref_names:
            return f"fx.memref_load({tensor}, [{self._render_exec_index(indices)}])"
        return f"{tensor}[{self._render_exec_index(indices)}]"

    def _render_exec_store(
        self,
        tensor: str,
        value: str,
        indices: list[str] | tuple[str, ...],
        memref_names: set[str],
    ) -> str:
        if tensor in memref_names:
            return f"fx.memref_store({value}, {tensor}, [{self._render_exec_index(indices)}])"
        return f"{tensor}[{self._render_exec_index(indices)}] = {value}"

    def _render_exec_copy(
        self,
        src: str,
        dst: str,
        *,
        tensor_specs: dict[str, TensorSpec],
        memref_names: set[str],
    ) -> list[str]:
        src_spec = tensor_specs.get(src)
        dst_spec = tensor_specs.get(dst)
        if src_spec is None or dst_spec is None:
            raise ValueError("flydsl_exec copy requires tensor specs for both source and destination")
        if src_spec.shape != dst_spec.shape:
            raise ValueError("flydsl_exec copy requires matching tensor shapes")
        return self._render_exec_loop_nest(
            src_spec.shape,
            prefix="_bb_copy_i",
            body_fn=lambda loop_vars: [
                self._render_exec_store(
                    dst,
                    self._render_exec_load(src, loop_vars, memref_names),
                    loop_vars,
                    memref_names,
                )
            ],
        )

    def _render_exec_fill(
        self,
        tensor_name: str,
        tensor_spec: TensorSpec,
        value_name: str,
        *,
        memref_names: set[str],
    ) -> list[str]:
        return self._render_exec_loop_nest(
            tensor_spec.shape,
            prefix=f"_{tensor_name}_fill_i",
            body_fn=lambda index_names: [
                self._render_exec_store(tensor_name, value_name, index_names, memref_names)
            ],
        )

    def _render_exec_tensor_unary_math(
        self,
        op_name: str,
        source_name: str,
        source_spec: TensorSpec,
        output_name: str,
        output_spec: TensorSpec,
        *,
        memref_names: set[str],
    ) -> list[str]:
        lines = self._render_exec_memref_alloca(output_name, output_spec)
        lines.extend(
            self._render_exec_loop_nest(
                source_spec.shape,
                prefix=f"_{output_name}_math_i",
                body_fn=lambda index_names: [
                    self._render_exec_store(
                        output_name,
                        self._render_exec_scalar_unary_math_expr(
                            op_name,
                            self._render_exec_load(source_name, index_names, memref_names),
                        ),
                        index_names,
                        memref_names,
                    )
                ],
            )
        )
        return lines

    def _render_exec_tensor_binary_math(
        self,
        op_name: str,
        lhs_name: str,
        rhs_name: str,
        *,
        tensor_specs: dict[str, TensorSpec],
        output_name: str,
        output_spec: TensorSpec,
        memref_names: set[str],
    ) -> list[str]:
        lines = self._render_exec_memref_alloca(output_name, output_spec)

        def operand_expr(name: str, index_names: list[str]) -> str:
            if name in tensor_specs:
                return self._render_exec_load(name, index_names, memref_names)
            return name

        lines.extend(
            self._render_exec_loop_nest(
                output_spec.shape,
                prefix=f"_{output_name}_math_i",
                body_fn=lambda index_names: [
                    self._render_exec_store(
                        output_name,
                        self._render_exec_scalar_binary_math_expr(
                            op_name,
                            operand_expr(lhs_name, index_names),
                            operand_expr(rhs_name, index_names),
                        ),
                        index_names,
                        memref_names,
                    )
                ],
            )
        )
        return lines

    def _render_exec_scalar_unary_math_expr(self, op_name: str, value: str) -> str:
        mapping = {
            "math_sqrt": f"math.sqrt({value})",
            "math_rsqrt": f"(1.0 / math.sqrt({value}))",
            "math_sin": f"math.sin({value})",
            "math_cos": f"math.cos({value})",
            "math_exp": f"math.exp({value})",
            "math_exp2": f"(2.0 ** ({value}))",
            "math_log": f"math.log({value})",
            "math_log2": f"math.log2({value})",
            "math_log10": f"math.log10({value})",
            "math_erf": f"math.erf({value})",
        }
        try:
            return mapping[op_name]
        except KeyError as exc:
            raise ValueError(f"flydsl_exec does not support unary math op '{op_name}'") from exc

    def _render_exec_scalar_binary_math_expr(self, op_name: str, lhs: str, rhs: str) -> str:
        if op_name == "math_atan2":
            return f"math.atan2({lhs}, {rhs})"
        raise ValueError(f"flydsl_exec does not support binary math op '{op_name}'")

    def _render_exec_broadcast_to(
        self,
        source_name: str,
        source_spec: TensorSpec,
        output_name: str,
        output_spec: TensorSpec,
        *,
        memref_names: set[str],
    ) -> list[str]:
        lines = self._render_exec_memref_alloca(output_name, output_spec)
        rank_delta = len(output_spec.shape) - len(source_spec.shape)

        def body(index_names: list[str]) -> list[str]:
            source_indices: list[str] = []
            for axis, extent in enumerate(source_spec.shape):
                output_axis = axis + rank_delta
                source_indices.append("0" if extent == 1 else index_names[output_axis])
            return [
                self._render_exec_store(
                    output_name,
                    self._render_exec_load(source_name, source_indices, memref_names),
                    index_names,
                    memref_names,
                )
            ]

        lines.extend(self._render_exec_loop_nest(output_spec.shape, prefix=f"_{output_name}_bcast_i", body_fn=body))
        return lines

    def _render_exec_tensor_binary(
        self,
        op_name: str,
        lhs_name: str,
        rhs_name: str,
        *,
        tensor_specs: dict[str, TensorSpec],
        output_name: str,
        output_spec: TensorSpec,
        memref_names: set[str],
    ) -> list[str]:
        symbol = {
            "tensor_add": "+",
            "tensor_sub": "-",
            "tensor_mul": "*",
            "tensor_div": "/",
        }[op_name]
        lines = self._render_exec_memref_alloca(output_name, output_spec)

        def operand_expr(name: str, index_names: list[str]) -> str:
            if name in tensor_specs:
                return self._render_exec_load(name, index_names, memref_names)
            return name

        lines.extend(
            self._render_exec_loop_nest(
                output_spec.shape,
                prefix=f"_{output_name}_bin_i",
                body_fn=lambda index_names: [
                    self._render_exec_store(
                        output_name,
                        f"{operand_expr(lhs_name, index_names)} {symbol} {operand_expr(rhs_name, index_names)}",
                        index_names,
                        memref_names,
                    )
                ],
            )
        )
        return lines

    def _render_exec_reduce_scalar(
        self,
        operation: Operation,
        source_name: str,
        source_spec: TensorSpec,
        init_name: str,
        *,
        memref_names: set[str],
    ) -> list[str]:
        output_name = operation.outputs[0]
        op_name = operation.op.removeprefix("reduce_")
        return [
            f"{output_name} = {init_name}",
            *self._render_exec_loop_nest(
                source_spec.shape,
                prefix=f"_{output_name}_reduce_i",
                body_fn=lambda index_names: [
                    f"{output_name} = {self._render_exec_reduction_expr(op_name, output_name, self._render_exec_load(source_name, index_names, memref_names))}"
                ],
            ),
        ]

    def _render_exec_reduce_tensor(
        self,
        operation: Operation,
        source_name: str,
        source_spec: TensorSpec,
        init_name: str,
        output_name: str,
        output_spec: TensorSpec,
        *,
        memref_names: set[str],
    ) -> list[str]:
        reduction_profile = operation.attrs.get("reduction_profile", 0)
        if not isinstance(reduction_profile, list):
            raise ValueError("flydsl_exec tensor reductions require an explicit reduction_profile list")
        keep_axes = [axis for axis, marker in enumerate(reduction_profile) if marker is None]
        if tuple(source_spec.shape[axis] for axis in keep_axes) != output_spec.shape:
            raise ValueError("flydsl_exec tensor reductions require output shapes matching the kept axes")
        lines = self._render_exec_memref_alloca(output_name, output_spec)
        lines.extend(
            self._render_exec_loop_nest(
                output_spec.shape,
                prefix=f"_{output_name}_init_i",
                body_fn=lambda out_indices: [
                    self._render_exec_store(output_name, init_name, out_indices, memref_names),
                ],
            )
        )
        lines.extend(
            self._render_exec_loop_nest(
                source_spec.shape,
                prefix=f"_{output_name}_reduce_i",
                body_fn=lambda source_indices: self._render_exec_reduce_tensor_body(
                    op_name=operation.op.removeprefix("reduce_"),
                    source_name=source_name,
                    output_name=output_name,
                    source_indices=source_indices,
                    output_indices=[source_indices[axis] for axis in keep_axes],
                    memref_names=memref_names,
                ),
            )
        )
        return lines

    def _render_exec_reduce_tensor_body(
        self,
        *,
        op_name: str,
        source_name: str,
        output_name: str,
        source_indices: list[str],
        output_indices: list[str],
        memref_names: set[str],
    ) -> list[str]:
        current_name = f"_{output_name}_cur"
        value_name = f"_{output_name}_val"
        return [
            f"{current_name} = {self._render_exec_load(output_name, output_indices, memref_names)}",
            f"{value_name} = {self._render_exec_load(source_name, source_indices, memref_names)}",
            self._render_exec_store(
                output_name,
                self._render_exec_reduction_expr(op_name, current_name, value_name),
                output_indices,
                memref_names,
            ),
        ]

    def _render_exec_reduction_expr(self, op_name: str, lhs: str, rhs: str) -> str:
        if op_name == "add":
            return f"{lhs} + {rhs}"
        if op_name == "mul":
            return f"{lhs} * {rhs}"
        if op_name == "max":
            return f"{lhs} if {lhs} > {rhs} else {rhs}"
        if op_name == "min":
            return f"{lhs} if {lhs} < {rhs} else {rhs}"
        raise ValueError(f"flydsl_exec does not support reduction op '{op_name}'")

    def _render_exec_loop_nest(self, shape: tuple[int, ...], *, prefix: str, body_fn) -> list[str]:
        if not shape:
            return body_fn([])
        index_names = [f"{prefix}{axis}" for axis in range(len(shape))]
        lines: list[str] = []
        indent = ""
        for index_name, extent in zip(index_names, shape):
            lines.append(f"{indent}for {index_name} in range({extent}):")
            indent += "    "
        for body_line in body_fn(index_names):
            lines.append(f"{indent}{body_line}")
        return lines

    def _probe_import(
        self,
        search_paths: tuple[Path, ...],
        library_paths: tuple[Path, ...],
    ) -> tuple[bool, bool, str | None]:
        env = os.environ.copy()
        pythonpath_parts = [str(path) for path in search_paths]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        if pythonpath_parts:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        ld_library_parts = [str(path) for path in library_paths]
        if env.get("LD_LIBRARY_PATH"):
            ld_library_parts.append(env["LD_LIBRARY_PATH"])
        if ld_library_parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_parts)
        script = (
            "import importlib, json\n"
            "result = {'expr': False, 'compiler': False, 'error': None}\n"
            "try:\n"
            "    importlib.import_module('flydsl.expr')\n"
            "    result['expr'] = True\n"
            "    importlib.import_module('flydsl.compiler')\n"
            "    result['compiler'] = True\n"
            "except Exception as exc:\n"
            "    result['error'] = f'{type(exc).__name__}: {exc}'\n"
            "print(json.dumps(result))\n"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        except Exception as exc:
            return False, False, f"{type(exc).__name__}: {exc}"
        stdout = completed.stdout.strip()
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            detail = stderr or stdout or f"subprocess exited with code {completed.returncode}"
            return False, False, f"subprocess import probe failed: {detail}"
        try:
            result = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except json.JSONDecodeError:
            return False, False, f"subprocess import probe produced invalid output: {stdout or '<empty>'}"
        return bool(result.get("expr")), bool(result.get("compiler")), result.get("error")

    @contextmanager
    def _pythonpath(self, search_paths: tuple[Path, ...]) -> Iterator[None]:
        previous = list(sys.path)
        try:
            if search_paths:
                sys.path[:0] = [str(path) for path in search_paths]
            yield
        finally:
            sys.path[:] = previous

    @contextmanager
    def _isolated_flydsl_modules(self) -> Iterator[None]:
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name == "flydsl" or name.startswith("flydsl.")
        }
        try:
            for name in list(sys.modules):
                if name == "flydsl" or name.startswith("flydsl."):
                    sys.modules.pop(name, None)
            yield
        finally:
            for name in list(sys.modules):
                if name == "flydsl" or name.startswith("flydsl."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)
