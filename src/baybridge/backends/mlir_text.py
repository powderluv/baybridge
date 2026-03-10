from __future__ import annotations

from typing import Any

from ..backend import LoweredModule
from ..ir import KernelArgument, Operation, PortableKernelIR, ScalarSpec, TensorSpec
from ..target import AMDTarget


class MlirTextBackend:
    name = "mlir_text"

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        value_types = self._seed_value_types(ir.arguments)
        body: list[str] = []
        for operation in ir.operations:
            body.extend(self._emit_operation(operation, value_types))
        argument_list = ", ".join(
            f"%{argument.name}: {value_types[argument.name]}" for argument in ir.arguments
        )
        launch_attrs = self._launch_attrs(ir)
        body_text = "\n".join(f"    {line}" for line in body + ["return"])
        text = (
            f'module attributes {{ baybridge.target = "{target.arch}", baybridge.backend = "{self.name}", '
            f'baybridge.wave_size = {target.wave_size} }} {{\n'
            f"  func.func @{ir.name}({argument_list}) attributes {{{launch_attrs}}} {{\n"
            f"{body_text}\n"
            f"  }}\n"
            f"}}\n"
        )
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="baybridge",
            text=text,
        )

    def _seed_value_types(self, arguments: tuple[KernelArgument, ...]) -> dict[str, str]:
        value_types: dict[str, str] = {}
        for argument in arguments:
            value_types[argument.name] = self._spec_type(argument.spec)
        return value_types

    def _launch_attrs(self, ir: PortableKernelIR) -> str:
        grid = ", ".join(str(dim) for dim in ir.launch.grid)
        block = ", ".join(str(dim) for dim in ir.launch.block)
        attrs = (
            f"baybridge.grid = [{grid}], "
            f"baybridge.block = [{block}], "
            f"baybridge.shared_mem_bytes = {ir.launch.shared_mem_bytes}"
        )
        if ir.launch.cooperative:
            attrs += ", baybridge.cooperative = true"
        return attrs

    def _emit_operation(self, operation: Operation, value_types: dict[str, str]) -> list[str]:
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
        attr_items = {
            key: value for key, value in operation.attrs.items() if key != "result"
        }
        attr_text = self._attrs(attr_items)
        signature = f"({input_types}) -> ({result_types})" if result_types else f"({input_types}) -> ()"
        if not operation.inputs:
            signature = "() -> ({})".format(result_types) if result_types else "() -> ()"
        return [f'{prefix}"baybridge.{operation.op}"({inputs}){attr_text} : {signature}']

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
        shape = "[" + ", ".join(str(dim) for dim in spec["shape"]) + "]"
        layout = spec["layout"]
        stride = "[" + ", ".join(str(dim) for dim in layout["stride"]) + "]"
        swizzle = "null" if layout["swizzle"] is None else f'"{layout["swizzle"]}"'
        return (
            f'!baybridge.tensor<shape={shape}, dtype={spec["dtype"]}, stride={stride}, swizzle={swizzle}, '
            f'space={spec["address_space"]}>'
        )
