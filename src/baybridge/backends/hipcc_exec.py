from __future__ import annotations

import ctypes
import subprocess
from math import prod
from pathlib import Path
from typing import Any

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..hip_runtime import HipRuntime, require_hipcc, scalar_ctype
from ..ir import KernelArgument, Layout, Operation, PortableKernelIR, ScalarSpec, TensorSpec
from ..runtime import RuntimeTensor
from ..target import AMDTarget


class HipccExecBackend:
    name = "hipcc_exec"
    artifact_extension = ".hip.cpp"

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        self._loop_index = 0
        self._cooperative_launch = ir.launch.cooperative
        argument_list = ", ".join(self._kernel_arg(argument) for argument in ir.arguments)
        body_lines: list[str] = []
        if any(operation.op == "make_tensor" and operation.attrs.get("dynamic_shared") for operation in ir.operations):
            body_lines.append("extern __shared__ unsigned char baybridge_dynamic_smem[];")
        body_lines.extend(self._emit_body(ir))
        body = "\n".join(f"  {line}" for line in body_lines)
        wrapper_args = ", ".join(
            [self._wrapper_arg(argument) for argument in ir.arguments]
            + [
                "unsigned int grid_x",
                "unsigned int grid_y",
                "unsigned int grid_z",
                "unsigned int block_x",
                "unsigned int block_y",
                "unsigned int block_z",
                "std::size_t shared_mem_bytes",
            ]
        )
        wrapper_call_args = ", ".join(argument.name for argument in ir.arguments)
        if ir.launch.cooperative:
            kernel_args = ", ".join(f"&{argument.name}" for argument in ir.arguments)
            launch_line = (
                f"  void* kernel_args[] = {{{kernel_args}}};\n"
                f"  hipError_t status = hipLaunchCooperativeKernel((void*){ir.name}, "
                "dim3(grid_x, grid_y, grid_z), dim3(block_x, block_y, block_z), "
                "kernel_args, shared_mem_bytes, 0);\n"
                "  if (status != hipSuccess) {\n"
                "    return static_cast<int>(status);\n"
                "  }\n"
            )
        else:
            launch_line = (
                f"  hipLaunchKernelGGL({ir.name}, dim3(grid_x, grid_y, grid_z), "
                f"dim3(block_x, block_y, block_z), shared_mem_bytes, 0, {wrapper_call_args});\n"
            )
        text = (
            "#include <hip/hip_runtime.h>\n"
            "#include <hip/hip_cooperative_groups.h>\n"
            "#include <hip/hip_fp16.h>\n"
            "#include <hip/hip_bfloat16.h>\n"
            "#include <cmath>\n"
            "#include <cstddef>\n"
            "#include <cstdint>\n\n"
            "namespace cg = cooperative_groups;\n\n"
            f"extern \"C\" __global__ void {ir.name}({argument_list}) {{\n"
            f"{body}\n"
            "}\n\n"
            f"extern \"C\" int launch_{ir.name}({wrapper_args}) {{\n"
            f"{launch_line}"
            "  return static_cast<int>(hipDeviceSynchronize());\n"
            "}\n"
        )
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="hip_cpp",
            text=text,
        )

    def build_launcher(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        source_path: Path,
    ):
        shared_path = source_path.with_suffix("").with_suffix(".so")
        state: dict[str, Any] = {}

        def launcher(*args: Any, **kwargs: Any) -> None:
            if kwargs:
                raise TypeError("hipcc_exec launcher only supports positional arguments")
            if len(args) != len(ir.arguments):
                raise TypeError(f"{ir.name} expects {len(ir.arguments)} arguments, got {len(args)}")
            if not shared_path.exists():
                self._compile_shared_object(source_path, shared_path, target, lowered_module.text)
            function = state.get("function")
            if function is None:
                library = ctypes.CDLL(str(shared_path))
                function = getattr(library, f"launch_{ir.name}")
                function.argtypes = self._launcher_argtypes(ir.arguments)
                function.restype = ctypes.c_int
                state["library"] = library
                state["function"] = function
            self._launch(function, ir, args)

        return launcher

    def _compile_shared_object(
        self,
        source_path: Path,
        shared_path: Path,
        target: AMDTarget,
        source_text: str,
    ) -> None:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source_text, encoding="utf-8")
        hipcc = require_hipcc()
        command = [
            hipcc,
            "-shared",
            "-fPIC",
            "-O2",
            f"--offload-arch={target.arch}",
            str(source_path),
            "-o",
            str(shared_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"hipcc failed for backend {self.name} with exit code {exc.returncode}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

    def _launch(self, function: Any, ir: PortableKernelIR, args: tuple[Any, ...]) -> None:
        hip = HipRuntime()
        tensor_allocations = []
        c_args: list[Any] = []
        try:
            for argument, value in zip(ir.arguments, args):
                if isinstance(argument.spec, TensorSpec):
                    if not isinstance(value, RuntimeTensor):
                        raise TypeError(
                            f"hipcc_exec expects RuntimeTensor values for tensor argument '{argument.name}', got {type(value).__name__}"
                        )
                    allocation = hip.upload_tensor(value)
                    tensor_allocations.append(allocation)
                    c_args.append(allocation.ptr)
                else:
                    ctype = scalar_ctype(argument.spec.dtype)
                    c_args.append(ctype(value))
            launch = ir.launch
            status = function(
                *c_args,
                launch.grid[0],
                launch.grid[1],
                launch.grid[2],
                launch.block[0],
                launch.block[1],
                launch.block[2],
                launch.shared_mem_bytes,
            )
            if status != 0:
                raise RuntimeError(f"launch_{ir.name} returned HIP status {status}")
            for allocation in tensor_allocations:
                allocation.copy_back(hip)
        finally:
            for allocation in tensor_allocations:
                allocation.free(hip)

    def _launcher_argtypes(self, arguments: tuple[KernelArgument, ...]) -> list[Any]:
        argtypes: list[Any] = []
        for argument in arguments:
            if isinstance(argument.spec, TensorSpec):
                argtypes.append(ctypes.c_void_p)
            else:
                argtypes.append(scalar_ctype(argument.spec.dtype))
        argtypes.extend(
            [
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_size_t,
            ]
        )
        return argtypes

    def _emit_body(self, ir: PortableKernelIR) -> list[str]:
        tensor_specs = {
            argument.name: argument.spec
            for argument in ir.arguments
            if isinstance(argument.spec, TensorSpec)
        }
        value_types = {
            argument.name: argument.spec
            for argument in ir.arguments
            if isinstance(argument.spec, ScalarSpec)
        }
        lines: list[str] = []
        for operation in ir.operations:
            if operation.op == "make_tensor":
                tensor_specs[operation.outputs[0]] = TensorSpec(
                    shape=tuple(operation.attrs["shape"]),
                    dtype=operation.attrs["dtype"],
                    layout=Layout(
                        shape=tuple(operation.attrs["layout"]["shape"]),
                        stride=tuple(operation.attrs["layout"]["stride"]),
                        swizzle=operation.attrs["layout"].get("swizzle"),
                    ),
                    address_space=operation.attrs["address_space"],
                )
            result = operation.attrs.get("result")
            if result is not None and result.get("kind") == "tensor":
                layout = result.get("layout")
                tensor_specs[operation.outputs[0]] = TensorSpec(
                    shape=tuple(result["shape"]),
                    dtype=result["dtype"],
                    layout=Layout(
                        shape=tuple(layout["shape"]),
                        stride=tuple(layout["stride"]),
                        swizzle=layout.get("swizzle"),
                    )
                    if layout is not None
                    else None,
                    address_space=result["address_space"],
                )
            lines.extend(self._emit_operation(operation, tensor_specs, value_types))
        return lines

    def _emit_operation(
        self,
        operation: Operation,
        tensor_specs: dict[str, TensorSpec],
        value_types: dict[str, ScalarSpec],
    ) -> list[str]:
        output = operation.outputs[0] if operation.outputs else None
        if output is not None:
            result = operation.attrs.get("result")
            if result is not None and result.get("kind") == "scalar":
                value_types[output] = ScalarSpec(dtype=result["dtype"])
        if operation.op in {"thread_idx", "block_idx", "program_id", "block_dim", "grid_dim"}:
            axis = operation.attrs["axis"]
            builtin = {
                "thread_idx": "threadIdx",
                "block_idx": "blockIdx",
                "program_id": "blockIdx",
                "block_dim": "blockDim",
                "grid_dim": "gridDim",
            }[operation.op]
            return [f"const std::int64_t {output} = static_cast<std::int64_t>({builtin}.{axis});"]
        if operation.op == "lane_id":
            return [f"const std::int64_t {output} = static_cast<std::int64_t>(__lane_id());"]
        if operation.op == "make_tensor":
            spec = tensor_specs[output]
            if operation.attrs.get("dynamic_shared"):
                byte_offset = int(operation.attrs.get("byte_offset", 0))
                return [
                    f"{self._cpp_tensor_base(spec.dtype)}* {output} = "
                    f"reinterpret_cast<{self._cpp_tensor_base(spec.dtype)}*>(baybridge_dynamic_smem + {byte_offset});"
                ]
            size = prod(spec.shape)
            qualifier = "__shared__ " if spec.address_space.value == "shared" else ""
            return [f"{qualifier}{self._cpp_tensor_base(spec.dtype)} {output}[{size}];"]
        if operation.op == "partition":
            source_name, *index_names = operation.inputs
            spec = tensor_specs[operation.outputs[0]]
            source_spec = tensor_specs[source_name]
            offset = self._offset_expr(source_spec, index_names) if index_names else "0"
            return [
                f"{self._cpp_tensor_base(spec.dtype)}* {operation.outputs[0]} = &{source_name}[{offset}];"
            ]
        if operation.op == "domain_offset":
            source_name, *index_names = operation.inputs
            spec = tensor_specs[operation.outputs[0]]
            source_spec = tensor_specs[source_name]
            offset = self._offset_expr(source_spec, index_names) if index_names else "0"
            return [
                f"{self._cpp_tensor_base(spec.dtype)}* {operation.outputs[0]} = &{source_name}[{offset}];"
            ]
        if operation.op == "local_tile":
            source_name, *index_names = operation.inputs
            spec = tensor_specs[operation.outputs[0]]
            source_spec = tensor_specs[source_name]
            fixed_axes = operation.attrs.get("fixed_axes", [])
            if len(fixed_axes) != len(index_names):
                raise BackendNotImplementedError(
                    f"hipcc_exec local_tile expected {len(fixed_axes)} fixed indices, got {len(index_names)}"
                )
            layout = source_spec.resolved_layout()
            terms = [f"({index}) * {layout.stride[axis]}" for axis, index in zip(fixed_axes, index_names)]
            offset = " + ".join(terms) if terms else "0"
            return [
                f"{self._cpp_tensor_base(spec.dtype)}* {operation.outputs[0]} = &{source_name}[{offset}];"
            ]
        if operation.op == "pointer_tensor":
            source_name = operation.inputs[0]
            spec = tensor_specs[operation.outputs[0]]
            return [
                f"{self._cpp_tensor_base(spec.dtype)}* {operation.outputs[0]} = "
                f"reinterpret_cast<{self._cpp_tensor_base(spec.dtype)}*>({source_name});"
            ]
        if operation.op == "slice":
            source_name, *index_names = operation.inputs
            spec = tensor_specs[operation.outputs[0]]
            source_spec = tensor_specs[source_name]
            offset = self._slice_offset_expr(source_spec, operation.attrs.get("fixed_axes", []), index_names)
            return [
                f"{self._cpp_tensor_base(spec.dtype)}* {operation.outputs[0]} = &{source_name}[{offset}];"
            ]
        if operation.op == "copy_async":
            src_name, dst_name = operation.inputs
            src_spec = tensor_specs[src_name]
            dst_spec = tensor_specs[dst_name]
            if src_spec.shape != dst_spec.shape:
                raise BackendNotImplementedError(
                    f"hipcc_exec copy_async requires matching tensor shapes, got {src_spec.shape} and {dst_spec.shape}"
                )
            return self._emit_tensor_copy(src_name, src_spec, dst_name, dst_spec)
        if operation.op == "copy":
            src_name, dst_name = operation.inputs
            src_spec = tensor_specs[src_name]
            dst_spec = tensor_specs[dst_name]
            if src_spec.shape != dst_spec.shape:
                raise BackendNotImplementedError(
                    f"hipcc_exec copy requires matching tensor shapes, got {src_spec.shape} and {dst_spec.shape}"
                )
            return self._emit_tensor_copy(src_name, src_spec, dst_name, dst_spec)
        if operation.op == "fill":
            tensor_name, value_name = operation.inputs
            tensor_spec = tensor_specs[tensor_name]
            return self._emit_tensor_fill(tensor_name, tensor_spec, value_name)
        if operation.op in {"tensor_add", "tensor_sub", "tensor_mul"}:
            lhs_name, rhs_name = operation.inputs
            out_name = operation.outputs[0]
            lhs_spec = tensor_specs[lhs_name]
            rhs_spec = tensor_specs[rhs_name]
            out_spec = tensor_specs[out_name]
            if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != out_spec.shape:
                raise BackendNotImplementedError(
                    "hipcc_exec tensor elementwise ops require matching tensor shapes"
                )
            if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != out_spec.dtype:
                raise BackendNotImplementedError(
                    "hipcc_exec tensor elementwise ops require matching tensor dtypes"
                )
            return self._emit_tensor_binary(operation.op, lhs_name, lhs_spec, rhs_name, rhs_spec, out_name, out_spec)
        if operation.op in {"math_sqrt", "math_sin", "math_exp2"}:
            source_name = operation.inputs[0]
            out_name = operation.outputs[0]
            source_spec = tensor_specs[source_name]
            out_spec = tensor_specs[out_name]
            if source_spec.shape != out_spec.shape or source_spec.dtype != out_spec.dtype:
                raise BackendNotImplementedError("hipcc_exec tensor math ops require matching tensor shapes and dtypes")
            return self._emit_tensor_unary_math(operation.op, source_name, source_spec, out_name, out_spec)
        if operation.op in {"reduce_add", "reduce_mul", "reduce_max", "reduce_min"}:
            source_name, init_name = operation.inputs
            source_spec = tensor_specs[source_name]
            init_spec = value_types[init_name]
            if init_spec.dtype != source_spec.dtype:
                raise BackendNotImplementedError("hipcc_exec reductions require init values matching the tensor dtype")
            if output is not None and output in tensor_specs:
                return self._emit_tensor_reduce_tensor(operation, source_name, source_spec, init_name, tensor_specs[output])
            if output is not None and output in value_types:
                return self._emit_tensor_reduce_scalar(operation, source_name, source_spec, init_name, value_types[output])
        if operation.op == "thread_fragment_load":
            base_name, thread_name, *rest = operation.inputs
            out_name = operation.outputs[0]
            base_spec = tensor_specs[base_name]
            out_spec = tensor_specs[out_name]
            predicate_name = rest[0] if rest else None
            predicate_spec = tensor_specs[predicate_name] if predicate_name is not None else None
            return self._emit_thread_fragment_load(
                base_name,
                base_spec,
                thread_name,
                out_name,
                out_spec,
                operation.attrs,
                predicate_name=predicate_name,
                predicate_spec=predicate_spec,
            )
        if operation.op == "thread_fragment_store":
            value_name, base_name, thread_name, *rest = operation.inputs
            value_spec = tensor_specs[value_name]
            base_spec = tensor_specs[base_name]
            predicate_name = rest[0] if rest else None
            predicate_spec = tensor_specs[predicate_name] if predicate_name is not None else None
            return self._emit_thread_fragment_store(
                value_name,
                value_spec,
                base_name,
                base_spec,
                thread_name,
                operation.attrs,
                predicate_name=predicate_name,
                predicate_spec=predicate_spec,
            )
        if operation.op == "barrier":
            kind = operation.attrs.get("kind", "block")
            if kind == "grid":
                if not self._cooperative_launch:
                    raise BackendNotImplementedError("grid-wide barriers require cooperative launch")
                return ["cg::this_grid().sync();"]
            return ["__syncthreads();"]
        if operation.op == "tensor_dim":
            return [f"const std::int64_t {output} = {int(operation.attrs['value'])};"]
        if operation.op == "constant":
            dtype = value_types[output].dtype
            return [f"const {self._cpp_scalar_type(dtype)} {output} = {self._literal(operation.attrs['value'], dtype)};"]
        if operation.op in {"add", "sub", "mul", "div", "floordiv", "mod", "and", "or", "bitand", "bitor"}:
            lhs, rhs = operation.inputs
            dtype = value_types[output].dtype
            op = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "floordiv": "/",
                "mod": "%",
                "and": "&&",
                "or": "||",
                "bitand": "&",
                "bitor": "|",
            }[operation.op]
            return [f"const {self._cpp_scalar_type(dtype)} {output} = {lhs} {op} {rhs};"]
        if operation.op == "neg":
            dtype = value_types[output].dtype
            source_name = operation.inputs[0]
            return [f"const {self._cpp_scalar_type(dtype)} {output} = -{source_name};"]
        if operation.op == "bitnot":
            dtype = value_types[output].dtype
            source_name = operation.inputs[0]
            return [f"const {self._cpp_scalar_type(dtype)} {output} = ~{source_name};"]
        if operation.op == "cast":
            source_name = operation.inputs[0]
            source_dtype = value_types[source_name].dtype
            target_dtype = value_types[output].dtype
            return [
                f"const {self._cpp_scalar_type(target_dtype)} {output} = "
                f"{self._cast_expr(source_name, source_dtype, target_dtype)};"
            ]
        if operation.op.startswith("cmp_"):
            lhs, rhs = operation.inputs
            op = {
                "cmp_lt": "<",
                "cmp_le": "<=",
                "cmp_gt": ">",
                "cmp_ge": ">=",
                "cmp_eq": "==",
                "cmp_ne": "!=",
            }[operation.op]
            return [f"const bool {output} = {lhs} {op} {rhs};"]
        if operation.op == "select":
            predicate_name, true_name, false_name = operation.inputs
            dtype = value_types[output].dtype
            return [
                f"const {self._cpp_scalar_type(dtype)} {output} = {predicate_name} ? {true_name} : {false_name};"
            ]
        if operation.op == "load":
            tensor_name, *index_names = operation.inputs
            spec = tensor_specs[tensor_name]
            offset = self._offset_expr(spec, index_names)
            return [f"const {self._cpp_tensor_base(spec.dtype)} {output} = {tensor_name}[{offset}];"]
        if operation.op == "store":
            value_name, tensor_name, *index_names = operation.inputs
            spec = tensor_specs[tensor_name]
            offset = self._offset_expr(spec, index_names)
            return [f"{tensor_name}[{offset}] = {value_name};"]
        if operation.op == "masked_load":
            tensor_name, *rest = operation.inputs
            predicate_name = rest[-2]
            fallback_name = rest[-1]
            index_names = rest[:-2]
            spec = tensor_specs[tensor_name]
            offset = self._offset_expr(spec, index_names)
            return [
                f"const {self._cpp_tensor_base(spec.dtype)} {output} = {predicate_name} ? {tensor_name}[{offset}] : {fallback_name};"
            ]
        if operation.op == "masked_store":
            value_name, tensor_name, *rest = operation.inputs
            predicate_name = rest[-1]
            index_names = rest[:-1]
            spec = tensor_specs[tensor_name]
            offset = self._offset_expr(spec, index_names)
            return [f"if ({predicate_name}) {{ {tensor_name}[{offset}] = {value_name}; }}"]
        raise BackendNotImplementedError(
            f"hipcc_exec does not support operation '{operation.op}' yet; use gpu_text/mlir_text or runtime fallback"
        )

    def _offset_expr(self, spec: TensorSpec, index_names: list[str]) -> str:
        layout = spec.resolved_layout()
        if len(index_names) != len(layout.stride):
            raise BackendNotImplementedError(
                f"hipcc_exec expected {len(layout.stride)} indices for tensor shape {spec.shape}, got {len(index_names)}"
            )
        terms = [f"({index}) * {stride}" for index, stride in zip(index_names, layout.stride)]
        return " + ".join(terms) if terms else "0"

    def _slice_offset_expr(self, spec: TensorSpec, fixed_axes: list[int], index_names: list[str]) -> str:
        layout = spec.resolved_layout()
        if len(fixed_axes) != len(index_names):
            raise BackendNotImplementedError(
                f"hipcc_exec expected {len(fixed_axes)} slice indices for fixed axes {fixed_axes}, got {len(index_names)}"
            )
        terms = [f"({index}) * {layout.stride[axis]}" for axis, index in zip(fixed_axes, index_names)]
        return " + ".join(terms) if terms else "0"

    def _next_loop_prefix(self) -> str:
        self._loop_index += 1
        return f"idx_{self._loop_index}"

    def _emit_loop_nest(self, shape: tuple[int, ...], body_lines_fn) -> list[str]:
        if not shape:
            return body_lines_fn([])
        prefix = self._next_loop_prefix()
        index_names = [f"{prefix}_{axis}" for axis in range(len(shape))]
        lines: list[str] = []
        indent = ""
        for index_name, extent in zip(index_names, shape):
            lines.append(f"{indent}for (std::int64_t {index_name} = 0; {index_name} < {extent}; ++{index_name}) {{")
            indent += "  "
        for body_line in body_lines_fn(index_names):
            lines.append(f"{indent}{body_line}")
        for depth in range(len(shape) - 1, -1, -1):
            indent = "  " * depth
            lines.append(f"{indent}}}")
        return lines

    def _emit_tensor_copy(
        self,
        src_name: str,
        src_spec: TensorSpec,
        dst_name: str,
        dst_spec: TensorSpec,
    ) -> list[str]:
        def body(index_names: list[str]) -> list[str]:
            src_offset = self._offset_expr(src_spec, index_names)
            dst_offset = self._offset_expr(dst_spec, index_names)
            return [f"{dst_name}[{dst_offset}] = {src_name}[{src_offset}];"]

        return self._emit_loop_nest(dst_spec.shape, body)

    def _emit_tensor_fill(
        self,
        tensor_name: str,
        tensor_spec: TensorSpec,
        value_name: str,
    ) -> list[str]:
        def body(index_names: list[str]) -> list[str]:
            tensor_offset = self._offset_expr(tensor_spec, index_names)
            return [f"{tensor_name}[{tensor_offset}] = {value_name};"]

        return self._emit_loop_nest(tensor_spec.shape, body)

    def _emit_tensor_binary(
        self,
        op: str,
        lhs_name: str,
        lhs_spec: TensorSpec,
        rhs_name: str,
        rhs_spec: TensorSpec,
        out_name: str,
        out_spec: TensorSpec,
    ) -> list[str]:
        symbol = {
            "tensor_add": "+",
            "tensor_sub": "-",
            "tensor_mul": "*",
        }[op]
        lines = [f"{self._cpp_tensor_base(out_spec.dtype)} {out_name}[{prod(out_spec.shape)}];"]

        def body(index_names: list[str]) -> list[str]:
            lhs_offset = self._offset_expr(lhs_spec, index_names)
            rhs_offset = self._offset_expr(rhs_spec, index_names)
            out_offset = self._offset_expr(out_spec, index_names)
            return [f"{out_name}[{out_offset}] = {lhs_name}[{lhs_offset}] {symbol} {rhs_name}[{rhs_offset}];"]

        lines.extend(self._emit_loop_nest(out_spec.shape, body))
        return lines

    def _emit_tensor_unary_math(
        self,
        op: str,
        source_name: str,
        source_spec: TensorSpec,
        out_name: str,
        out_spec: TensorSpec,
    ) -> list[str]:
        expr = {
            "math_sqrt": lambda value: f"sqrtf({value})",
            "math_sin": lambda value: f"sinf({value})",
            "math_exp2": lambda value: f"exp2f({value})",
        }[op]
        lines = [f"{self._cpp_tensor_base(out_spec.dtype)} {out_name}[{prod(out_spec.shape)}];"]

        def body(index_names: list[str]) -> list[str]:
            source_offset = self._offset_expr(source_spec, index_names)
            out_offset = self._offset_expr(out_spec, index_names)
            return [f"{out_name}[{out_offset}] = {expr(f'{source_name}[{source_offset}]')};"]

        lines.extend(self._emit_loop_nest(out_spec.shape, body))
        return lines

    def _emit_tensor_reduce_scalar(
        self,
        operation: Operation,
        source_name: str,
        source_spec: TensorSpec,
        init_name: str,
        output_spec: ScalarSpec,
    ) -> list[str]:
        op_name = operation.op.removeprefix("reduce_")
        if output_spec.dtype != source_spec.dtype:
            raise BackendNotImplementedError("hipcc_exec scalar reductions require matching output and source dtypes")
        output_name = operation.outputs[0]
        lines = [f"{self._cpp_scalar_type(output_spec.dtype)} {output_name} = {init_name};"]

        def body(index_names: list[str]) -> list[str]:
            source_offset = self._offset_expr(source_spec, index_names)
            value_expr = f"{source_name}[{source_offset}]"
            return [self._reduction_update(op_name, output_name, value_expr)]

        lines.extend(self._emit_loop_nest(source_spec.shape, body))
        return lines

    def _emit_tensor_reduce_tensor(
        self,
        operation: Operation,
        source_name: str,
        source_spec: TensorSpec,
        init_name: str,
        output_spec: TensorSpec,
    ) -> list[str]:
        op_name = operation.op.removeprefix("reduce_")
        output_name = operation.outputs[0]
        reduction_profile = operation.attrs.get("reduction_profile", 0)
        if not isinstance(reduction_profile, list):
            raise BackendNotImplementedError("hipcc_exec tensor reductions require an explicit reduction_profile list")
        keep_axes = [axis for axis, marker in enumerate(reduction_profile) if marker is None]
        if tuple(source_spec.shape[axis] for axis in keep_axes) != output_spec.shape:
            raise BackendNotImplementedError("hipcc_exec tensor reductions require output shapes matching the kept axes")
        lines = [f"{self._cpp_tensor_base(output_spec.dtype)} {output_name}[{prod(output_spec.shape)}];"]

        def init_body(index_names: list[str]) -> list[str]:
            out_offset = self._offset_expr(output_spec, index_names)
            return [f"{output_name}[{out_offset}] = {init_name};"]

        lines.extend(self._emit_loop_nest(output_spec.shape, init_body))

        def reduce_body(index_names: list[str]) -> list[str]:
            source_offset = self._offset_expr(source_spec, index_names)
            out_indices = [index_names[axis] for axis in keep_axes]
            out_offset = self._offset_expr(output_spec, out_indices) if out_indices else "0"
            value_expr = f"{source_name}[{source_offset}]"
            return [self._reduction_update(op_name, f"{output_name}[{out_offset}]", value_expr)]

        lines.extend(self._emit_loop_nest(source_spec.shape, reduce_body))
        return lines

    def _reduction_update(self, op_name: str, lhs: str, rhs: str) -> str:
        if op_name == "add":
            return f"{lhs} = {lhs} + {rhs};"
        if op_name == "mul":
            return f"{lhs} = {lhs} * {rhs};"
        if op_name == "max":
            return f"{lhs} = {lhs} > {rhs} ? {lhs} : {rhs};"
        if op_name == "min":
            return f"{lhs} = {lhs} < {rhs} ? {lhs} : {rhs};"
        raise BackendNotImplementedError(f"hipcc_exec does not support reduction op '{op_name}'")

    def _layout_from_dict(self, value: Any) -> Layout:
        return Layout(
            shape=tuple(value["shape"]),
            stride=tuple(value["stride"]),
            swizzle=value.get("swizzle"),
        )

    def _emit_layout_coords(
        self,
        linear_name: str,
        layout: Layout,
        *,
        prefix: str,
    ) -> tuple[list[str], list[str]]:
        axes = sorted(range(len(layout.shape)), key=lambda axis: layout.stride[axis], reverse=True)
        remaining_name = f"{prefix}_remaining"
        lines = [f"const std::int64_t {remaining_name} = {linear_name};"]
        coords = ["0"] * len(layout.shape)
        for index, axis in enumerate(axes):
            coord_name = f"{prefix}_coord_{axis}"
            stride = layout.stride[axis]
            lines.append(f"const std::int64_t {coord_name} = {remaining_name} / {stride};")
            coords[axis] = coord_name
            if index != len(axes) - 1:
                next_remaining = f"{prefix}_remaining_{index}"
                lines.append(f"const std::int64_t {next_remaining} = {remaining_name} % {stride};")
                remaining_name = next_remaining
        return lines, coords

    def _emit_thread_value_coords(
        self,
        thread_name: str,
        value_name: str,
        *,
        attrs: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        thread_layout = self._layout_from_dict(attrs["thread_layout"])
        value_layout = self._layout_from_dict(attrs["value_layout"])
        prefix = self._next_loop_prefix()
        thread_lines, thread_coords = self._emit_layout_coords(thread_name, thread_layout, prefix=f"{prefix}_thread")
        value_lines, value_coords = self._emit_layout_coords(value_name, value_layout, prefix=f"{prefix}_value")
        lines = thread_lines + value_lines
        coords = [
            f"({thread_coord}) * {value_extent} + ({value_coord})"
            for thread_coord, value_extent, value_coord in zip(
                thread_coords,
                value_layout.shape,
                value_coords,
            )
        ]
        return lines, coords

    def _emit_thread_fragment_load(
        self,
        base_name: str,
        base_spec: TensorSpec,
        thread_name: str,
        out_name: str,
        out_spec: TensorSpec,
        attrs: dict[str, Any],
        *,
        predicate_name: str | None = None,
        predicate_spec: TensorSpec | None = None,
    ) -> list[str]:
        lines = [f"{self._cpp_tensor_base(out_spec.dtype)} {out_name}[{prod(out_spec.shape)}];"]
        else_value = self._literal(attrs.get("else_value", 0), out_spec.dtype)

        def body(index_names: list[str]) -> list[str]:
            value_name = index_names[0] if index_names else "0"
            coord_lines, base_indices = self._emit_thread_value_coords(thread_name, value_name, attrs=attrs)
            out_offset = self._offset_expr(out_spec, index_names)
            base_offset = self._offset_expr(base_spec, base_indices)
            if predicate_name is None:
                return coord_lines + [f"{out_name}[{out_offset}] = {base_name}[{base_offset}];"]
            assert predicate_spec is not None
            predicate_offset = self._offset_expr(predicate_spec, index_names)
            return coord_lines + [
                f"{out_name}[{out_offset}] = {predicate_name}[{predicate_offset}] ? {base_name}[{base_offset}] : {else_value};"
            ]

        lines.extend(self._emit_loop_nest(out_spec.shape, body))
        return lines

    def _emit_thread_fragment_store(
        self,
        value_name: str,
        value_spec: TensorSpec,
        base_name: str,
        base_spec: TensorSpec,
        thread_name: str,
        attrs: dict[str, Any],
        *,
        predicate_name: str | None = None,
        predicate_spec: TensorSpec | None = None,
    ) -> list[str]:
        def body(index_names: list[str]) -> list[str]:
            fragment_index = index_names[0] if index_names else "0"
            coord_lines, base_indices = self._emit_thread_value_coords(thread_name, fragment_index, attrs=attrs)
            value_offset = self._offset_expr(value_spec, index_names)
            base_offset = self._offset_expr(base_spec, base_indices)
            if predicate_name is None:
                return coord_lines + [f"{base_name}[{base_offset}] = {value_name}[{value_offset}];"]
            assert predicate_spec is not None
            predicate_offset = self._offset_expr(predicate_spec, index_names)
            return coord_lines + [
                f"if ({predicate_name}[{predicate_offset}]) {{ {base_name}[{base_offset}] = {value_name}[{value_offset}]; }}"
            ]

        return self._emit_loop_nest(value_spec.shape, body)

    def _kernel_arg(self, argument: KernelArgument) -> str:
        if isinstance(argument.spec, TensorSpec):
            return f"{self._cpp_tensor_base(argument.spec.dtype)}* {argument.name}"
        return f"{self._cpp_scalar_type(argument.spec.dtype)} {argument.name}"

    def _wrapper_arg(self, argument: KernelArgument) -> str:
        return self._kernel_arg(argument)

    def _cpp_scalar_type(self, dtype: str) -> str:
        table = {
            "f32": "float",
            "f16": "__half",
            "bf16": "hip_bfloat16",
            "i1": "bool",
            "i8": "std::int8_t",
            "i32": "std::int32_t",
            "i64": "std::int64_t",
            "index": "std::int64_t",
        }
        try:
            return table[dtype]
        except KeyError as exc:
            raise BackendNotImplementedError(f"hipcc_exec does not support scalar dtype '{dtype}' yet") from exc

    def _cpp_unsigned_scalar_type(self, dtype: str) -> str:
        table = {
            "i8": "std::uint8_t",
            "i32": "std::uint32_t",
            "i64": "std::uint64_t",
            "index": "std::uint64_t",
        }
        try:
            return table[dtype]
        except KeyError as exc:
            raise BackendNotImplementedError(
                f"hipcc_exec does not support unsigned scalar casts for dtype '{dtype}' yet"
            ) from exc

    def _cpp_tensor_base(self, dtype: str) -> str:
        table = {
            "f32": "float",
            "f16": "__half",
            "bf16": "hip_bfloat16",
            "i1": "bool",
            "i8": "std::int8_t",
            "i32": "std::int32_t",
            "i64": "std::int64_t",
        }
        try:
            return table[dtype]
        except KeyError as exc:
            raise BackendNotImplementedError(f"hipcc_exec does not support tensor dtype '{dtype}' yet") from exc

    def _literal(self, value: Any, dtype: str) -> str:
        if dtype == "f32":
            return f"{float(value)}f"
        if dtype == "f16":
            return f"__float2half({float(value)}f)"
        if dtype == "bf16":
            return f"hip_bfloat16({float(value)}f)"
        if dtype == "i1":
            return "true" if value else "false"
        return str(int(value))

    def _cast_expr(self, source_name: str, source_dtype: str, target_dtype: str) -> str:
        if source_dtype == target_dtype:
            return source_name
        if target_dtype == "f16":
            if source_dtype == "bf16":
                return f"__float2half(static_cast<float>({source_name}))"
            return f"__float2half(static_cast<float>({source_name}))"
        if target_dtype == "bf16":
            return f"hip_bfloat16(static_cast<float>({source_name}))"
        if source_dtype in {"f16", "bf16"} and target_dtype in {"i1", "i8", "i32", "i64", "index", "f32"}:
            return f"static_cast<{self._cpp_scalar_type(target_dtype)}>(static_cast<float>({source_name}))"
        if target_dtype in {"i8", "i32", "i64", "index"} and source_dtype in {"i1", "i8", "i32", "i64", "index"}:
            return (
                f"static_cast<{self._cpp_scalar_type(target_dtype)}>"
                f"(static_cast<{self._cpp_unsigned_scalar_type(target_dtype)}>({source_name}))"
            )
        return f"static_cast<{self._cpp_scalar_type(target_dtype)}>({source_name})"
