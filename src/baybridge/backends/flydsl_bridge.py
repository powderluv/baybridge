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
    "math_acos",
    "math_asin",
    "math_atan",
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
    "math_acos",
    "math_asin",
    "math_atan",
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
        real_exec_enabled = os.environ.get("BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC") == "1"
        validated_real_exec = self.has_validated_real_exec(ir)
        if environment.built_package_available and (validated_real_exec or real_exec_enabled):
            special = self._render_real_exec_python_specialized(ir)
            if special is not None:
                kernel_body, prologue_lines = special
            else:
                kernel_body = "\n".join(f"    {line}" for line in self._render_exec_body(ir))
                prologue_lines = []
        else:
            kernel_body = "\n".join(f"    {line}" for line in self._render_exec_body(ir))
            prologue_lines = []
        bufferize_lines = "".join(f"    {line}\n" for line in prologue_lines)
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
            f"{bufferize_lines}"
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

    def has_validated_real_exec(self, ir: PortableKernelIR) -> bool:
        return (
            self._match_real_exec_pointwise_binary_1d(ir) is not None
            or self._match_real_exec_copy_1d(ir) is not None
            or self._match_real_exec_broadcast_add_2d(ir) is not None
            or self._match_real_exec_reduce_bundle_2d(ir) is not None
            or self._match_real_exec_tensor_factory_2d(ir) is not None
            or self._match_real_exec_shared_stage_1d(ir) is not None
        )

    def _render_real_exec_python_specialized(self, ir: PortableKernelIR) -> tuple[str, list[str]] | None:
        matched = self._match_real_exec_pointwise_binary_1d(ir)
        if matched is not None:
            return self._render_real_exec_pointwise_binary_1d(ir, *matched)
        matched_copy = self._match_real_exec_copy_1d(ir)
        if matched_copy is not None:
            return self._render_real_exec_copy_1d(ir, *matched_copy)
        matched_broadcast = self._match_real_exec_broadcast_add_2d(ir)
        if matched_broadcast is not None:
            return self._render_real_exec_broadcast_add_2d(ir, *matched_broadcast)
        matched_reduce = self._match_real_exec_reduce_bundle_2d(ir)
        if matched_reduce is not None:
            return self._render_real_exec_reduce_bundle_2d(ir, *matched_reduce)
        matched_tensor_factory = self._match_real_exec_tensor_factory_2d(ir)
        if matched_tensor_factory is not None:
            return self._render_real_exec_tensor_factory_2d(ir, *matched_tensor_factory)
        matched_shared_stage = self._match_real_exec_shared_stage_1d(ir)
        if matched_shared_stage is not None:
            return self._render_real_exec_copy_1d(ir, *matched_shared_stage)
        return None

    def _match_real_exec_pointwise_binary_1d(self, ir: PortableKernelIR) -> tuple[str, str, str, str, str] | None:
        if len(ir.arguments) != 3:
            return None
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return None
        src_arg, other_arg, dst_arg = ir.arguments
        for argument in (src_arg, other_arg, dst_arg):
            spec = argument.spec
            if spec.shape != (4,) and len(spec.shape) != 1:
                return None
            if spec.dtype != "f32":
                return None
            if spec.address_space.value != "global":
                return None
        ops = ir.operations
        if len(ops) != 7:
            return None
        if [op.op for op in ops[:3]] != ["thread_idx", "thread_idx", "thread_idx"]:
            return None
        thread_x = ops[0]
        if thread_x.attrs.get("axis") != "x" or len(thread_x.outputs) != 1:
            return None
        thread_index_name = thread_x.outputs[0]
        load_a, load_b, add_op, store_op = ops[3:]
        if load_a.op != "load" or load_b.op != "load" or store_op.op != "store":
            return None
        if add_op.op not in {"add", "sub", "mul", "div"}:
            return None
        if tuple(load_a.inputs) != (src_arg.name, thread_index_name):
            return None
        if tuple(load_b.inputs) != (other_arg.name, thread_index_name):
            return None
        if tuple(add_op.inputs) != (load_a.outputs[0], load_b.outputs[0]):
            return None
        if tuple(store_op.inputs) != (add_op.outputs[0], dst_arg.name, thread_index_name):
            return None
        return src_arg.name, other_arg.name, dst_arg.name, thread_index_name, add_op.op

    def _render_real_exec_pointwise_binary_1d(
        self,
        ir: PortableKernelIR,
        src_name: str,
        other_name: str,
        dst_name: str,
        thread_index_name: str,
        op_name: str,
    ) -> tuple[str, list[str]]:
        del thread_index_name
        block_dim = ir.launch.block[0]
        arith_op = {
            "add": "addf",
            "sub": "subf",
            "mul": "mulf",
            "div": "divf",
        }[op_name]
        prologue = [
            "bid = fx.block_idx.x",
            "tid = fx.thread_idx.x",
            f"t_{src_name} = fx.logical_divide({src_name}, fx.make_layout({block_dim}, 1))",
            f"t_{other_name} = fx.logical_divide({other_name}, fx.make_layout({block_dim}, 1))",
            f"t_{dst_name} = fx.logical_divide({dst_name}, fx.make_layout({block_dim}, 1))",
            f"t_{src_name} = fx.slice(t_{src_name}, (None, bid))",
            f"t_{other_name} = fx.slice(t_{other_name}, (None, bid))",
            f"t_{dst_name} = fx.slice(t_{dst_name}, (None, bid))",
            f"t_{src_name} = fx.logical_divide(t_{src_name}, fx.make_layout(1, 1))",
            f"t_{other_name} = fx.logical_divide(t_{other_name}, fx.make_layout(1, 1))",
            f"t_{dst_name} = fx.logical_divide(t_{dst_name}, fx.make_layout(1, 1))",
            "RMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)",
            "copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)",
            f"r_{src_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{other_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{dst_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
        ]
        body_lines = [
            f"fx.copy_atom_call(copyAtom, fx.slice(t_{src_name}, (None, tid)), r_{src_name})",
            f"fx.copy_atom_call(copyAtom, fx.slice(t_{other_name}, (None, tid)), r_{other_name})",
            f"val_{dst_name} = fx.arith.{arith_op}(fx.memref_load_vec(r_{src_name}), fx.memref_load_vec(r_{other_name}))",
            f"fx.memref_store_vec(val_{dst_name}, r_{dst_name})",
            f"fx.copy_atom_call(copyAtom, r_{dst_name}, fx.slice(t_{dst_name}, (None, tid)))",
        ]
        return "\n".join(f"    {line}" for line in body_lines), prologue

    def _match_real_exec_copy_1d(self, ir: PortableKernelIR) -> tuple[str, str] | None:
        if len(ir.arguments) != 2:
            return None
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return None
        src_arg, dst_arg = ir.arguments
        for argument in (src_arg, dst_arg):
            spec = argument.spec
            if len(spec.shape) != 1:
                return None
            if spec.dtype != "f32":
                return None
            if spec.address_space.value != "global":
                return None
        if len(ir.operations) != 1 or ir.operations[0].op != "copy":
            return None
        copy_op = ir.operations[0]
        if tuple(copy_op.inputs) != (src_arg.name, dst_arg.name):
            return None
        return src_arg.name, dst_arg.name

    def _render_real_exec_copy_1d(
        self,
        ir: PortableKernelIR,
        src_name: str,
        dst_name: str,
    ) -> tuple[str, list[str]]:
        extent = ir.arguments[0].spec.shape[0] if isinstance(ir.arguments[0].spec, TensorSpec) else 0
        prologue = [
            f"t_{src_name} = fx.logical_divide({src_name}, fx.make_layout(1, 1))",
            f"t_{dst_name} = fx.logical_divide({dst_name}, fx.make_layout(1, 1))",
            "RMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)",
            "copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)",
            f"r_{src_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
        ]
        body_lines = [
            f"for copy_i0 in range({extent}):",
            f"    fx.copy_atom_call(copyAtom, fx.slice(t_{src_name}, (None, fx.Int32(copy_i0))), r_{src_name})",
            f"    fx.copy_atom_call(copyAtom, r_{src_name}, fx.slice(t_{dst_name}, (None, fx.Int32(copy_i0))))",
        ]
        return "\n".join(f"    {line}" for line in body_lines), prologue

    def _match_real_exec_broadcast_add_2d(self, ir: PortableKernelIR) -> tuple[str, str, str] | None:
        if len(ir.arguments) != 3:
            return None
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return None
        if tuple(ir.launch.grid) != (1, 1, 1):
            return None
        if tuple(ir.launch.block) != (1, 1, 1):
            return None
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        if len(lhs_arg.spec.shape) != 2 or len(rhs_arg.spec.shape) != 2 or len(dst_arg.spec.shape) != 2:
            return None
        for argument in (lhs_arg, rhs_arg, dst_arg):
            spec = argument.spec
            if spec.dtype != "f32":
                return None
            if spec.address_space.value != "global":
                return None
        dst_shape = tuple(dst_arg.spec.shape)
        for src_shape in (tuple(lhs_arg.spec.shape), tuple(rhs_arg.spec.shape)):
            if len(src_shape) != len(dst_shape):
                return None
            if any(src_dim not in (1, dst_dim) for src_dim, dst_dim in zip(src_shape, dst_shape, strict=True)):
                return None
        ops = ir.operations
        if len(ops) != 8:
            return None
        if [op.op for op in ops] != [
            "make_tensor",
            "copy",
            "make_tensor",
            "copy",
            "broadcast_to",
            "broadcast_to",
            "tensor_add",
            "copy",
        ]:
            return None
        lhs_tensor = ops[0].outputs[0]
        rhs_tensor = ops[2].outputs[0]
        lhs_broadcast = ops[4].outputs[0]
        rhs_broadcast = ops[5].outputs[0]
        result_tensor = ops[6].outputs[0]
        if tuple(ops[1].inputs) != (lhs_arg.name, lhs_tensor):
            return None
        if tuple(ops[3].inputs) != (rhs_arg.name, rhs_tensor):
            return None
        if tuple(ops[4].inputs) != (lhs_tensor,):
            return None
        if tuple(ops[5].inputs) != (rhs_tensor,):
            return None
        if tuple(ops[6].inputs) != (lhs_broadcast, rhs_broadcast):
            return None
        if tuple(ops[7].inputs) != (result_tensor, dst_arg.name):
            return None
        lhs_result = ops[4].attrs.get("result")
        rhs_result = ops[5].attrs.get("result")
        add_result = ops[6].attrs.get("result")
        if not isinstance(lhs_result, dict) or tuple(lhs_result.get("shape", ())) != dst_shape:
            return None
        if not isinstance(rhs_result, dict) or tuple(rhs_result.get("shape", ())) != dst_shape:
            return None
        if not isinstance(add_result, dict) or tuple(add_result.get("shape", ())) != dst_shape:
            return None
        return lhs_arg.name, rhs_arg.name, dst_arg.name

    def _render_real_exec_broadcast_add_2d(
        self,
        ir: PortableKernelIR,
        lhs_name: str,
        rhs_name: str,
        dst_name: str,
    ) -> tuple[str, list[str]]:
        lhs_spec = ir.arguments[0].spec
        rhs_spec = ir.arguments[1].spec
        dst_spec = ir.arguments[2].spec
        if not isinstance(lhs_spec, TensorSpec) or not isinstance(rhs_spec, TensorSpec) or not isinstance(dst_spec, TensorSpec):
            raise ValueError("validated real FlyDSL broadcast-add lowering requires tensor specs")
        dst_rows, dst_cols = dst_spec.shape
        lhs_row_expr = "fx.Int32(0)" if lhs_spec.shape[0] == 1 else "row_idx"
        rhs_row_expr = "fx.Int32(0)" if rhs_spec.shape[0] == 1 else "row_idx"
        lhs_col_expr = "fx.Int32(0)" if lhs_spec.shape[1] == 1 else "col_idx"
        rhs_col_expr = "fx.Int32(0)" if rhs_spec.shape[1] == 1 else "col_idx"
        prologue = [
            "RMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)",
            "copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)",
            f"r_{lhs_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{rhs_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{dst_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
        ]
        body_lines = [
            f"for row_i in range({dst_rows}):",
            "    row_idx = fx.Int32(row_i)",
            f"    row_{lhs_name} = fx.slice({lhs_name}, ({lhs_row_expr}, None))",
            f"    row_{rhs_name} = fx.slice({rhs_name}, ({rhs_row_expr}, None))",
            f"    row_{dst_name} = fx.slice({dst_name}, (row_idx, None))",
            f"    t_{lhs_name} = fx.logical_divide(row_{lhs_name}, fx.make_layout(1, 1))",
            f"    t_{rhs_name} = fx.logical_divide(row_{rhs_name}, fx.make_layout(1, 1))",
            f"    t_{dst_name} = fx.logical_divide(row_{dst_name}, fx.make_layout(1, 1))",
            f"    for col_i in range({dst_cols}):",
            "        col_idx = fx.Int32(col_i)",
            f"        fx.copy_atom_call(copyAtom, fx.slice(t_{lhs_name}, (None, {lhs_col_expr})), r_{lhs_name})",
            f"        fx.copy_atom_call(copyAtom, fx.slice(t_{rhs_name}, (None, {rhs_col_expr})), r_{rhs_name})",
            f"        val_{dst_name} = fx.arith.addf(fx.memref_load_vec(r_{lhs_name}), fx.memref_load_vec(r_{rhs_name}))",
            f"        fx.memref_store_vec(val_{dst_name}, r_{dst_name})",
            f"        fx.copy_atom_call(copyAtom, r_{dst_name}, fx.slice(t_{dst_name}, (None, col_idx)))",
        ]
        return "\n".join(f"    {line}" for line in body_lines), prologue

    def _match_real_exec_shared_stage_1d(self, ir: PortableKernelIR) -> tuple[str, str] | None:
        if len(ir.arguments) != 2:
            return None
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return None
        src_arg, dst_arg = ir.arguments
        if src_arg.spec.shape != dst_arg.spec.shape:
            return None
        if len(src_arg.spec.shape) != 1:
            return None
        extent = src_arg.spec.shape[0]
        if extent <= 0:
            return None
        if tuple(ir.launch.grid) != (1, 1, 1):
            return None
        if ir.launch.block[0] != extent:
            return None
        for argument in (src_arg, dst_arg):
            spec = argument.spec
            if spec.dtype != "f32":
                return None
            if spec.address_space.value != "global":
                return None
        ops = ir.operations
        if len(ops) != 9:
            return None
        if [op.op for op in ops] != [
            "thread_idx",
            "thread_idx",
            "thread_idx",
            "make_tensor",
            "load",
            "store",
            "barrier",
            "load",
            "store",
        ]:
            return None
        thread_x = ops[0]
        if thread_x.attrs.get("axis") != "x":
            return None
        for op, axis in zip(ops[:3], ("x", "y", "z"), strict=True):
            if op.attrs.get("axis") != axis or len(op.outputs) != 1:
                return None
        shared_tensor = ops[3]
        if shared_tensor.attrs.get("address_space") != "shared":
            return None
        if tuple(shared_tensor.attrs.get("shape", ())) != (extent,):
            return None
        if shared_tensor.attrs.get("dtype") != "f32":
            return None
        shared_name = shared_tensor.outputs[0]
        thread_index_name = thread_x.outputs[0]
        src_load = ops[4]
        smem_store = ops[5]
        barrier = ops[6]
        smem_load = ops[7]
        dst_store = ops[8]
        if tuple(src_load.inputs) != (src_arg.name, thread_index_name):
            return None
        if tuple(smem_store.inputs) != (src_load.outputs[0], shared_name, thread_index_name):
            return None
        if barrier.attrs.get("kind") != "block":
            return None
        if tuple(smem_load.inputs) != (shared_name, thread_index_name):
            return None
        if tuple(dst_store.inputs) != (smem_load.outputs[0], dst_arg.name, thread_index_name):
            return None
        return src_arg.name, dst_arg.name

    def _match_real_exec_tensor_factory_2d(self, ir: PortableKernelIR) -> tuple[str, str, str] | None:
        if len(ir.arguments) != 3:
            return None
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return None
        if tuple(ir.launch.grid) != (1, 1, 1):
            return None
        if tuple(ir.launch.block) != (1, 1, 1):
            return None
        dst_zero_arg, dst_one_arg, dst_full_arg = ir.arguments
        shape = tuple(dst_zero_arg.spec.shape)
        if len(shape) != 2:
            return None
        if shape[0] <= 0 or shape[1] <= 0:
            return None
        for argument in (dst_zero_arg, dst_one_arg, dst_full_arg):
            spec = argument.spec
            if tuple(spec.shape) != shape:
                return None
            if spec.dtype != "f32":
                return None
            if spec.address_space.value != "global":
                return None
        ops = ir.operations
        if len(ops) != 12:
            return None
        expected_pattern = ["make_tensor", "constant", "fill", "copy"] * 3
        if [op.op for op in ops] != expected_pattern:
            return None
        expected_args = [
            (dst_zero_arg.name, 0),
            (dst_one_arg.name, 1),
            (dst_full_arg.name, 7.0),
        ]
        for offset, (dst_name, fill_value) in zip((0, 4, 8), expected_args, strict=True):
            tensor_op, constant_op, fill_op, copy_op = ops[offset : offset + 4]
            tensor_name = tensor_op.outputs[0]
            constant_name = constant_op.outputs[0]
            if tuple(fill_op.inputs) != (tensor_name, constant_name):
                return None
            if tuple(copy_op.inputs) != (tensor_name, dst_name):
                return None
            result = constant_op.attrs.get("result")
            if not isinstance(result, dict) or result.get("dtype") != "f32":
                return None
            if constant_op.attrs.get("value") != fill_value:
                return None
        return dst_zero_arg.name, dst_one_arg.name, dst_full_arg.name

    def _render_real_exec_tensor_factory_2d(
        self,
        ir: PortableKernelIR,
        dst_zero_name: str,
        dst_one_name: str,
        dst_full_name: str,
    ) -> tuple[str, list[str]]:
        dst_spec = ir.arguments[0].spec
        if not isinstance(dst_spec, TensorSpec):
            raise ValueError("validated real FlyDSL tensor-factory lowering requires tensor specs")
        rows, cols = dst_spec.shape
        prologue = [
            "RMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)",
            "copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)",
            f"r_{dst_zero_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{dst_one_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{dst_full_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            "vec1f32 = fx.T.vector(1, element_type=fx.T.f32())",
            "c_zero = fx.arith.constant(0.0, type=fx.T.f32())",
            "c_one = fx.arith.constant(1.0, type=fx.T.f32())",
            "c_full = fx.arith.constant(7.0, type=fx.T.f32())",
            "v_zero = fx.vector.from_elements(vec1f32, [c_zero])",
            "v_one = fx.vector.from_elements(vec1f32, [c_one])",
            "v_full = fx.vector.from_elements(vec1f32, [c_full])",
            f"fx.memref_store_vec(v_zero, r_{dst_zero_name})",
            f"fx.memref_store_vec(v_one, r_{dst_one_name})",
            f"fx.memref_store_vec(v_full, r_{dst_full_name})",
        ]
        body_lines = [
            f"for row_i in range({rows}):",
            "    row_idx = fx.Int32(row_i)",
            f"    row_{dst_zero_name} = fx.slice({dst_zero_name}, (row_idx, None))",
            f"    row_{dst_one_name} = fx.slice({dst_one_name}, (row_idx, None))",
            f"    row_{dst_full_name} = fx.slice({dst_full_name}, (row_idx, None))",
            f"    t_{dst_zero_name} = fx.logical_divide(row_{dst_zero_name}, fx.make_layout(1, 1))",
            f"    t_{dst_one_name} = fx.logical_divide(row_{dst_one_name}, fx.make_layout(1, 1))",
            f"    t_{dst_full_name} = fx.logical_divide(row_{dst_full_name}, fx.make_layout(1, 1))",
            f"    for col_i in range({cols}):",
            "        col_idx = fx.Int32(col_i)",
            f"        fx.copy_atom_call(copyAtom, r_{dst_zero_name}, fx.slice(t_{dst_zero_name}, (None, col_idx)))",
            f"        fx.copy_atom_call(copyAtom, r_{dst_one_name}, fx.slice(t_{dst_one_name}, (None, col_idx)))",
            f"        fx.copy_atom_call(copyAtom, r_{dst_full_name}, fx.slice(t_{dst_full_name}, (None, col_idx)))",
        ]
        return "\n".join(f"    {line}" for line in body_lines), prologue

    def _match_real_exec_math_bundle_1d(self, ir: PortableKernelIR) -> tuple[str, str] | None:
        if len(ir.arguments) != 7:
            return None
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return None
        src_arg, other_arg, *dst_args = ir.arguments
        shape = tuple(src_arg.spec.shape)
        if len(shape) != 1:
            return None
        for argument in ir.arguments:
            spec = argument.spec
            if tuple(spec.shape) != shape:
                return None
            if spec.dtype != "f32":
                return None
            if spec.address_space.value != "global":
                return None
        ops = ir.operations
        if len(ops) != 14:
            return None
        if [op.op for op in ops[:4]] != ["make_tensor", "copy", "make_tensor", "copy"]:
            return None
        src_tensor = ops[0].outputs[0]
        other_tensor = ops[2].outputs[0]
        if tuple(ops[1].inputs) != (src_arg.name, src_tensor):
            return None
        if tuple(ops[3].inputs) != (other_arg.name, other_tensor):
            return None
        expected_math = [
            ("math_exp", (src_tensor,), dst_args[0].name),
            ("math_log", (src_tensor,), dst_args[1].name),
            ("math_cos", (src_tensor,), dst_args[2].name),
            ("math_erf", (src_tensor,), dst_args[3].name),
            ("math_atan2", (src_tensor, other_tensor), dst_args[4].name),
        ]
        for (math_op_name, expected_inputs, dst_name), math_op, copy_op in zip(
            expected_math,
            ops[4::2],
            ops[5::2],
            strict=True,
        ):
            if math_op.op != math_op_name:
                return None
            if tuple(math_op.inputs) != expected_inputs:
                return None
            if tuple(copy_op.inputs) != (math_op.outputs[0], dst_name):
                return None
        return src_arg.name, other_arg.name

    def _match_real_exec_reduce_bundle_2d(self, ir: PortableKernelIR) -> tuple[str, str, str] | None:
        if len(ir.arguments) != 3:
            return None
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            return None
        if tuple(ir.launch.grid) != (1, 1, 1):
            return None
        if tuple(ir.launch.block) != (1, 1, 1):
            return None
        src_arg, dst_scalar_arg, dst_rows_arg = ir.arguments
        if len(src_arg.spec.shape) != 2:
            return None
        if src_arg.spec.shape[0] <= 0 or src_arg.spec.shape[1] <= 0:
            return None
        if tuple(dst_scalar_arg.spec.shape) != (1,):
            return None
        if tuple(dst_rows_arg.spec.shape) != (src_arg.spec.shape[0],):
            return None
        for argument in ir.arguments:
            spec = argument.spec
            if spec.dtype != "f32":
                return None
            if spec.address_space.value != "global":
                return None
        ops = ir.operations
        if len(ops) != 9:
            return None
        if [op.op for op in ops] != [
            "make_tensor",
            "copy",
            "constant",
            "reduce_add",
            "constant",
            "store",
            "constant",
            "reduce_add",
            "copy",
        ]:
            return None
        src_tensor = ops[0].outputs[0]
        if tuple(ops[1].inputs) != (src_arg.name, src_tensor):
            return None
        if tuple(ops[3].inputs) != (src_tensor, ops[2].outputs[0]):
            return None
        if tuple(ops[5].inputs) != (ops[3].outputs[0], dst_scalar_arg.name, ops[4].outputs[0]):
            return None
        if tuple(ops[7].inputs) != (src_tensor, ops[6].outputs[0]):
            return None
        if tuple(ops[8].inputs) != (ops[7].outputs[0], dst_rows_arg.name):
            return None
        if ops[3].attrs.get("reduction_profile") != 0:
            return None
        if ops[7].attrs.get("reduction_profile") != [None, 1]:
            return None
        return src_arg.name, dst_scalar_arg.name, dst_rows_arg.name

    def _render_real_exec_reduce_bundle_2d(
        self,
        ir: PortableKernelIR,
        src_name: str,
        dst_scalar_name: str,
        dst_rows_name: str,
    ) -> tuple[str, list[str]]:
        src_spec = ir.arguments[0].spec
        if not isinstance(src_spec, TensorSpec):
            raise ValueError("validated real FlyDSL reduction lowering requires tensor specs")
        rows, cols = src_spec.shape
        prologue = [
            "RMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)",
            "copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)",
            f"t_{dst_rows_name} = fx.logical_divide({dst_rows_name}, fx.make_layout(1, 1))",
            f"t_{dst_scalar_name} = fx.logical_divide({dst_scalar_name}, fx.make_layout(1, 1))",
            f"r_{src_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{dst_rows_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
            f"r_{dst_scalar_name} = fx.memref_alloca(RMemRefTy, fx.make_layout(1, 1))",
        ]
        body_lines = [
            f"for row_i in range({rows}):",
            "    row_idx = fx.Int32(row_i)",
            f"    row_{src_name} = fx.slice({src_name}, (row_idx, None))",
            f"    t_{src_name} = fx.logical_divide(row_{src_name}, fx.make_layout(1, 1))",
            f"    fx.copy_atom_call(copyAtom, fx.slice(t_{src_name}, (None, fx.Int32(0))), r_{dst_rows_name})",
            f"    for col_i in range(1, {cols}):",
            "        col_idx = fx.Int32(col_i)",
            f"        fx.copy_atom_call(copyAtom, fx.slice(t_{src_name}, (None, col_idx)), r_{src_name})",
            f"        fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(r_{dst_rows_name}), fx.memref_load_vec(r_{src_name})), r_{dst_rows_name})",
            f"    fx.copy_atom_call(copyAtom, r_{dst_rows_name}, fx.slice(t_{dst_rows_name}, (None, row_idx)))",
            f"fx.copy_atom_call(copyAtom, fx.slice(t_{dst_rows_name}, (None, fx.Int32(0))), r_{dst_scalar_name})",
            f"for row_i in range(1, {rows}):",
            "    row_idx = fx.Int32(row_i)",
            f"    fx.copy_atom_call(copyAtom, fx.slice(t_{dst_rows_name}, (None, row_idx)), r_{src_name})",
            f"    fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(r_{dst_scalar_name}), fx.memref_load_vec(r_{src_name})), r_{dst_scalar_name})",
            f"fx.copy_atom_call(copyAtom, r_{dst_scalar_name}, fx.slice(t_{dst_scalar_name}, (None, fx.Int32(0))))",
        ]
        return "\n".join(f"    {line}" for line in body_lines), prologue

    def _render_exec_body(self, ir: PortableKernelIR) -> list[str]:
        tensor_specs = self._collect_exec_tensor_specs(ir)
        memref_names: set[str] = set()
        lines: list[str] = []
        for operation in ir.operations:
            if self._operation_allocates_exec_memref(operation):
                memref_names.update(operation.outputs)
            lines.extend(self._render_exec_operation(operation, tensor_specs=tensor_specs, memref_names=memref_names))
        if not lines:
            lines.append("pass")
        return lines

    def _operation_allocates_exec_memref(self, operation: Operation) -> bool:
        if operation.op == "make_tensor":
            return True
        if operation.op in {"broadcast_to", "tensor_add", "tensor_sub", "tensor_mul", "tensor_div"}:
            return True
        if operation.op.startswith("math_"):
            result = operation.attrs.get("result")
            return isinstance(result, dict) and result.get("kind") == "tensor"
        if operation.op.startswith("reduce_"):
            result = operation.attrs.get("result")
            return isinstance(result, dict) and result.get("kind") == "tensor"
        return False

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
            "math_acos",
            "math_asin",
            "math_atan",
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

    def _render_exec_index_tuple(self, indices: list[str] | tuple[str, ...]) -> str:
        if len(indices) == 1:
            return f"({indices[0]},)"
        return f"({self._render_exec_index(indices)})"

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
            return "fx.AddressSpace.Shared"
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
            return f"fx.memref_load({tensor}, {self._render_exec_index_tuple(indices)})"
        return f"{tensor}[{self._render_exec_index(indices)}]"

    def _render_exec_store(
        self,
        tensor: str,
        value: str,
        indices: list[str] | tuple[str, ...],
        memref_names: set[str],
    ) -> str:
        if tensor in memref_names:
            return f"fx.memref_store({value}, {tensor}, {self._render_exec_index_tuple(indices)})"
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
            "math_acos": f"math.acos({value})",
            "math_asin": f"math.asin({value})",
            "math_atan": f"math.atan({value})",
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
