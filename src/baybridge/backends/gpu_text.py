from __future__ import annotations

import re
from typing import Any

from ..backend import LoweredModule
from ..ir import KernelArgument, Operation, PortableKernelIR, ScalarSpec, TensorSpec
from ..mfma import MFMA_DESCRIPTORS, resolve_mfma_descriptor
from ..target import AMDTarget


class GpuTextBackend:
    name = "gpu_text"

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        value_types = self._seed_value_types(ir.arguments)
        body: list[str] = []
        for operation in ir.operations:
            body.extend(self._emit_operation(operation, value_types))
        argument_list = ", ".join(
            f"%{argument.name}: {value_types[argument.name]}" for argument in ir.arguments
        )
        kernel_attrs = self._kernel_attrs(ir, target)
        body_text = "\n".join(f"      {line}" for line in body + ["gpu.return"])
        text = (
            f"gpu.module @kernels attributes {{rocdl.target = \"{target.arch}\", rocdl.wave_size = {target.wave_size}}} {{\n"
            f"  gpu.func @{ir.name}({argument_list}) kernel attributes {{{kernel_attrs}}} {{\n"
            f"{body_text}\n"
            f"  }}\n"
            f"}}\n"
        )
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="gpu+rocdl",
            text=text,
        )

    def _seed_value_types(self, arguments: tuple[KernelArgument, ...]) -> dict[str, str]:
        value_types: dict[str, str] = {}
        for argument in arguments:
            value_types[argument.name] = self._spec_type(argument.spec)
        return value_types

    def _kernel_attrs(self, ir: PortableKernelIR, target: AMDTarget) -> str:
        grid = ", ".join(str(dim) for dim in ir.launch.grid)
        block = ", ".join(str(dim) for dim in ir.launch.block)
        attrs = (
            f"gpu.grid = [{grid}], "
            f"gpu.block = [{block}], "
            f"gpu.dynamic_shared_memory = {ir.launch.shared_mem_bytes}, "
            f"rocdl.target = \"{target.arch}\""
        )
        if ir.launch.cooperative:
            attrs += ", gpu.cooperative = true"
        return attrs

    def _emit_operation(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
        if operation.op == "make_tensor":
            return self._emit_make_tensor(operation, value_types)
        if operation.op == "partition":
            return self._emit_partition(operation, value_types)
        if operation.op == "fragment":
            return self._emit_fragment(operation, value_types)
        if operation.op == "mma":
            return self._emit_mma(operation, value_types)
        if operation.op in {"copy", "copy_async", "copy_reduce", "commit_group", "wait_group", "barrier"}:
            return self._emit_gpu_semantic_op(operation, value_types)
        if operation.op in {
            "program_id",
            "block_idx",
            "thread_idx",
            "block_dim",
            "grid_dim",
            "lane_id",
            "tensor_dim",
            "constant",
            "add",
            "sub",
            "mul",
            "div",
            "floordiv",
            "mod",
            "cmp_lt",
            "cmp_le",
            "cmp_gt",
            "cmp_ge",
            "cmp_eq",
            "cmp_ne",
            "and",
            "or",
            "select",
            "load",
            "store",
            "masked_load",
            "masked_store",
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
            "math_atan2",
        }:
            return self._emit_builtin(operation, value_types)

        outputs = []
        if operation.outputs:
            for output in operation.outputs:
                output_spec = operation.attrs.get("result")
                if output_spec is not None:
                    value_types[output] = self._result_type(output_spec)
                else:
                    value_types[output] = self._op_result_type(operation)
                outputs.append(f"%{output}")
        prefix = f"{', '.join(outputs)} = " if outputs else ""
        inputs = ", ".join(f"%{value}" for value in operation.inputs)
        input_types = ", ".join(value_types[value] for value in operation.inputs)
        result_types = ", ".join(value_types[value] for value in operation.outputs)
        attr_items = {key: value for key, value in operation.attrs.items() if key != "result"}
        attr_text = self._attrs(attr_items)
        signature = f"({input_types}) -> ({result_types})" if result_types else f"({input_types}) -> ()"
        if not operation.inputs:
            signature = "() -> ({})".format(result_types) if result_types else "() -> ()"
        return [f'{prefix}"baybridge.{operation.op}"({inputs}){attr_text} : {signature}']

    def _emit_make_tensor(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
        output = operation.outputs[0]
        spec = {
            "shape": operation.attrs["shape"],
            "dtype": operation.attrs["dtype"],
            "layout": operation.attrs["layout"],
            "address_space": operation.attrs["address_space"],
        }
        tensor_type = self._tensor_type(spec)
        value_types[output] = tensor_type
        address_space = spec["address_space"]
        if address_space in {"shared", "local"}:
            return [f"%{output} = memref.alloca() : {tensor_type}"]
        if address_space == "global":
            return [f"%{output} = memref.alloc() : {tensor_type}"]
        attrs = self._attrs(spec)
        return [f'%{output} = "amdgpu.alloca_register"(){attrs} : () -> {tensor_type}']

    def _emit_partition(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
        source_name = operation.inputs[0]
        offset_names = operation.inputs[1:]
        output_name = operation.outputs[0]
        output_spec = operation.attrs["result"]
        result_type = self._result_type(output_spec)
        value_types[output_name] = result_type
        source_type = value_types[source_name]
        if source_type.startswith("memref<") and operation.attrs.get("policy") == "blocked":
            tile = operation.attrs["tile"]
            if offset_names:
                offsets = ", ".join(f"%{name}" for name in offset_names)
            else:
                offsets = ", ".join("0" for _ in tile)
            sizes = ", ".join(str(dim) for dim in tile)
            strides = ", ".join("1" for _ in tile)
            return [
                f"%{output_name} = memref.subview %{source_name}[{offsets}] [{sizes}] [{strides}] : {source_type} to {result_type}"
            ]
        attr_items = {key: value for key, value in operation.attrs.items() if key != "result"}
        attr_text = self._attrs(attr_items)
        return [f'%{output_name} = "baybridge.partition"(%{source_name}){attr_text} : ({source_type}) -> ({result_type})']

    def _emit_fragment(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
        source_name, lane_row_name, lane_col_name = operation.inputs
        output_name = operation.outputs[0]
        output_spec = operation.attrs["result"]
        result_type = self._result_type(output_spec)
        value_types[output_name] = result_type
        source_type = value_types[source_name]
        lane_row_type = value_types[lane_row_name]
        lane_col_type = value_types[lane_col_name]
        attr_items = {key: value for key, value in operation.attrs.items() if key != "result"}
        attr_text = self._attrs(attr_items)
        return [
            f'%{output_name} = "amdgpu.fragment_view"(%{source_name}, %{lane_row_name}, %{lane_col_name}){attr_text} : ({source_type}, {lane_row_type}, {lane_col_type}) -> ({result_type})'
        ]

    def _emit_mma(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
        a_name, b_name, acc_name = operation.inputs
        a_type = value_types[a_name]
        b_type = value_types[b_name]
        acc_type = value_types[acc_name]
        variant = self._mfma_variant(operation, a_type, b_type, acc_type)
        attrs = {key: value for key, value in operation.attrs.items() if value is not None}
        attrs["variant"] = variant.variant_name
        attrs["operand_dtype"] = variant.operand_dtype
        attrs["accumulator_dtype"] = variant.accumulator_dtype
        attrs["wave_size"] = variant.wave_size
        attr_text = self._attrs(attrs)
        if a_type.startswith("!baybridge.reg<") and b_type.startswith("!baybridge.reg<"):
            return [
                f'"{variant.llvm_intrinsic}"(%{a_name}, %{b_name}, %{acc_name}){attr_text} : ({a_type}, {b_type}, {acc_type}) -> ()'
            ]
        return [f'"amdgpu.mfma"(%{a_name}, %{b_name}, %{acc_name}){attr_text} : ({a_type}, {b_type}, {acc_type}) -> ()']

    def _emit_gpu_semantic_op(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
        if operation.op == "barrier":
            kind = operation.attrs.get("kind", "block")
            if kind == "grid":
                return ['"amdgpu.grid_barrier"() : () -> ()']
            if kind == "warp":
                return ['"amdgpu.wave_barrier"() : () -> ()']
            return ["gpu.barrier"]
        if operation.op == "commit_group":
            group = operation.attrs["group"]
            return [f'"gpu.async.commit_group"() {{group = "{group}"}} : () -> ()']
        if operation.op == "wait_group":
            count = operation.attrs["count"]
            group = operation.attrs["group"]
            return [f'"gpu.async.wait_group"() {{count = {count}, group = "{group}"}} : () -> ()']
        if operation.op in {"copy", "copy_async"}:
            src_name, dst_name = operation.inputs
            src_type = value_types[src_name]
            dst_type = value_types[dst_name]
            attrs = {key: value for key, value in operation.attrs.items() if value is not None}
            attr_text = self._attrs(attrs)
            if operation.op == "copy_async":
                return [f"gpu.memcpy async %{dst_name}, %{src_name}{attr_text} : {dst_type}, {src_type}"]
            return [f"memref.copy %{src_name}, %{dst_name} : {src_type} to {dst_type}"]
        if operation.op == "copy_reduce":
            src_name, dst_name = operation.inputs
            src_type = value_types[src_name]
            dst_type = value_types[dst_name]
            attrs = {key: value for key, value in operation.attrs.items() if value is not None}
            attr_text = self._attrs(attrs)
            return [
                f'"baybridge.copy_reduce"(%{src_name}, %{dst_name}){attr_text} : ({src_type}, {dst_type}) -> ()'
            ]
        raise ValueError(f"unsupported gpu semantic op '{operation.op}'")

    def _emit_builtin(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
        output = operation.outputs[0] if operation.outputs else None
        output_spec = operation.attrs.get("result")
        if output is not None:
            value_types[output] = self._result_type(output_spec) if output_spec is not None else "index"
        axis = operation.attrs.get("axis")
        if operation.op == "program_id":
            return [f"%{output} = gpu.block_id {axis}"]
        if operation.op == "block_idx":
            return [f"%{output} = gpu.block_id {axis}"]
        if operation.op == "thread_idx":
            return [f"%{output} = gpu.thread_id {axis}"]
        if operation.op == "block_dim":
            return [f"%{output} = gpu.block_dim {axis}"]
        if operation.op == "grid_dim":
            return [f"%{output} = gpu.grid_dim {axis}"]
        if operation.op == "lane_id":
            return [f'%{output} = "rocdl.lane_id"() : () -> index']
        if operation.op == "tensor_dim":
            return [f"%{output} = arith.constant {operation.attrs['value']} : index"]
        if operation.op == "constant":
            return [f"%{output} = arith.constant {operation.attrs['value']} : {value_types[output]}"]
        if operation.op in {"add", "sub", "mul"}:
            arith_op = self._arith_op(operation.op, value_types[output])
            lhs, rhs = operation.inputs
            operand_type = value_types[output]
            return [f"%{output} = {arith_op} %{lhs}, %{rhs} : {operand_type}"]
        if operation.op == "div":
            lhs, rhs = operation.inputs
            operand_type = value_types[output]
            arith_op = "arith.divf" if operand_type.startswith("f") else "arith.divsi"
            return [f"%{output} = {arith_op} %{lhs}, %{rhs} : {operand_type}"]
        if operation.op == "floordiv":
            lhs, rhs = operation.inputs
            operand_type = value_types[output]
            return [f"%{output} = arith.floordivsi %{lhs}, %{rhs} : {operand_type}"]
        if operation.op == "mod":
            lhs, rhs = operation.inputs
            operand_type = value_types[output]
            return [f"%{output} = arith.remsi %{lhs}, %{rhs} : {operand_type}"]
        if operation.op.startswith("cmp_"):
            lhs, rhs = operation.inputs
            operand_type = value_types[lhs]
            cmp_op = self._cmp_op(operation.op, operand_type)
            return [f"%{output} = {cmp_op} %{lhs}, %{rhs} : {operand_type}"]
        if operation.op in {"and", "or"}:
            lhs, rhs = operation.inputs
            arith_op = "arith.andi" if operation.op == "and" else "arith.ori"
            return [f"%{output} = {arith_op} %{lhs}, %{rhs} : i1"]
        if operation.op == "select":
            predicate_name, true_name, false_name = operation.inputs
            result_type = value_types[output]
            return [f"%{output} = arith.select %{predicate_name}, %{true_name}, %{false_name} : {result_type}"]
        if operation.op == "load":
            tensor_name, *index_names = operation.inputs
            tensor_type = value_types[tensor_name]
            indices = ", ".join(f"%{name}" for name in index_names)
            return [f"%{output} = memref.load %{tensor_name}[{indices}] : {tensor_type}"]
        if operation.op == "store":
            value_name, tensor_name, *index_names = operation.inputs
            tensor_type = value_types[tensor_name]
            indices = ", ".join(f"%{name}" for name in index_names)
            return [f"memref.store %{value_name}, %{tensor_name}[{indices}] : {tensor_type}"]
        if operation.op == "masked_load":
            tensor_name, *rest = operation.inputs
            predicate_name = rest[-2]
            fallback_name = rest[-1]
            index_names = rest[:-2]
            tensor_type = value_types[tensor_name]
            result_type = value_types[output]
            indices = ", ".join(f"%{name}" for name in index_names)
            return [
                f"%{output} = scf.if %{predicate_name} -> ({result_type}) {{",
                f"  %masked_load_val = memref.load %{tensor_name}[{indices}] : {tensor_type}",
                f"  scf.yield %masked_load_val : {result_type}",
                "} else {",
                f"  scf.yield %{fallback_name} : {result_type}",
                "}",
            ]
        if operation.op == "masked_store":
            value_name, tensor_name, *rest = operation.inputs
            predicate_name = rest[-1]
            index_names = rest[:-1]
            tensor_type = value_types[tensor_name]
            indices = ", ".join(f"%{name}" for name in index_names)
            return [
                f"scf.if %{predicate_name} {{",
                f"  memref.store %{value_name}, %{tensor_name}[{indices}] : {tensor_type}",
                "}",
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
            operand_type = value_types[output]
            return [f"%{output} = {self._math_unary_op(operation.op)} %{source_name} : {operand_type}"]
        if operation.op == "math_atan2":
            lhs, rhs = operation.inputs
            operand_type = value_types[output]
            return [f"%{output} = math.atan2 %{lhs}, %{rhs} : {operand_type}"]
        raise ValueError(f"unsupported builtin op '{operation.op}'")

    def _arith_op(self, op: str, result_type: str) -> str:
        is_float = result_type.startswith("f")
        table = {
            ("add", False): "arith.addi",
            ("sub", False): "arith.subi",
            ("mul", False): "arith.muli",
            ("add", True): "arith.addf",
            ("sub", True): "arith.subf",
            ("mul", True): "arith.mulf",
        }
        return table[(op, is_float)]

    def _cmp_op(self, op: str, operand_type: str) -> str:
        is_float = operand_type.startswith("f")
        suffix = {
            "cmp_lt": "olt" if is_float else "slt",
            "cmp_le": "ole" if is_float else "sle",
            "cmp_gt": "ogt" if is_float else "sgt",
            "cmp_ge": "oge" if is_float else "sge",
            "cmp_eq": "oeq" if is_float else "eq",
            "cmp_ne": "one" if is_float else "ne",
        }[op]
        return f"arith.cmp{'f' if is_float else 'i'} {suffix}"

    def _math_unary_op(self, op: str) -> str:
        return {
            "math_acos": "math.acos",
            "math_asin": "math.asin",
            "math_atan": "math.atan",
            "math_sqrt": "math.sqrt",
            "math_rsqrt": "math.rsqrt",
            "math_sin": "math.sin",
            "math_cos": "math.cos",
            "math_exp": "math.exp",
            "math_exp2": "math.exp2",
            "math_log": "math.log",
            "math_log2": "math.log2",
            "math_log10": "math.log10",
            "math_erf": "math.erf",
        }[op]

    def _mfma_variant(self, operation: Operation, a_type: str, b_type: str, acc_type: str):
        tile = tuple(operation.attrs.get("tile") or ())
        a_info = self._tensor_info(a_type)
        b_info = self._tensor_info(b_type)
        acc_info = self._tensor_info(acc_type)
        if a_info["dtype"] != b_info["dtype"]:
            raise ValueError(
                "gpu_text requires matching mma operand dtypes, "
                f"got a_dtype={a_info['dtype']} and b_dtype={b_info['dtype']}"
            )
        try:
            return resolve_mfma_descriptor(tile, a_info["dtype"], acc_info["dtype"])
        except ValueError:
            pass
        supported = ", ".join(
            f"{descriptor.variant_name}[tile={descriptor.tile}, operand_dtype={descriptor.operand_dtype}, accumulator_dtype={descriptor.accumulator_dtype}]"
            for descriptor in MFMA_DESCRIPTORS
        )
        if not supported:
            supported = "none"
        raise ValueError(
            "gpu_text does not support mma lowering for "
            f"tile={tile}, operand_dtype={a_info['dtype']}, acc_dtype={acc_info['dtype']}. "
            f"Supported variants: {supported}"
        )

    def _op_result_type(self, operation: Operation) -> str:
        if operation.op == "make_tensor":
            return self._tensor_type(
                {
                    "shape": operation.attrs["shape"],
                    "dtype": operation.attrs["dtype"],
                    "layout": operation.attrs["layout"],
                    "address_space": operation.attrs["address_space"],
                }
            )
        raise ValueError(f"cannot infer result type for operation '{operation.op}'")

    def _attrs(self, attrs: dict[str, Any]) -> str:
        if not attrs:
            return ""
        rendered = ", ".join(f"{key} = {self._attr_value(value)}" for key, value in attrs.items())
        return f" {{{rendered}}}"

    def _attr_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, dict):
            rendered = ", ".join(f"{key} = {self._attr_value(item)}" for key, item in value.items())
            return "{" + rendered + "}"
        if isinstance(value, list):
            return "[" + ", ".join(self._attr_value(item) for item in value) + "]"
        return str(value)

    def _spec_type(self, spec: TensorSpec | ScalarSpec) -> str:
        if isinstance(spec, TensorSpec):
            return self._tensor_type(spec.to_dict())
        return spec.dtype

    def _result_type(self, spec: dict[str, Any]) -> str:
        kind = spec.get("kind", "tensor")
        if kind == "tensor":
            return self._tensor_type(spec)
        if kind == "scalar":
            return spec["dtype"]
        raise ValueError(f"unknown result kind '{kind}'")

    def _tensor_type(self, spec: dict[str, Any]) -> str:
        shape_prefix = "x".join(str(dim) for dim in spec["shape"])
        dtype = spec["dtype"]
        address_space = spec["address_space"]
        layout = spec["layout"]
        stride = ", ".join(str(dim) for dim in layout["stride"])
        if address_space == "register":
            return f"!baybridge.reg<{shape_prefix}x{dtype}>"
        memory_space = {
            "global": "1",
            "shared": "3",
            "local": "5",
        }.get(address_space, "0")
        return f"memref<{shape_prefix}x{dtype}, strided<[{stride}], offset: 0>, {memory_space}>"

    def _tensor_info(self, tensor_type: str) -> dict[str, Any]:
        reg_match = re.fullmatch(r"!baybridge\.reg<([0-9x]+)x([A-Za-z0-9]+)>", tensor_type)
        if reg_match:
            shape = tuple(int(dim) for dim in reg_match.group(1).split("x"))
            return {"kind": "reg", "shape": shape, "dtype": reg_match.group(2)}
        memref_match = re.fullmatch(r"memref<([0-9x]+)x([A-Za-z0-9]+),.*>", tensor_type)
        if memref_match:
            shape = tuple(int(dim) for dim in memref_match.group(1).split("x"))
            return {"kind": "memref", "shape": shape, "dtype": memref_match.group(2)}
        raise ValueError(f"unable to parse tensor type '{tensor_type}'")
