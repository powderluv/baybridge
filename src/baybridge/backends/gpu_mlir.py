from __future__ import annotations

from ..backend import LoweredModule
from ..ir import PortableKernelIR
from ..target import AMDTarget
from .gpu_text import GpuTextBackend


class GpuMlirBackend(GpuTextBackend):
    name = "gpu_mlir"

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        value_types = self._seed_value_types(ir.arguments)
        body: list[str] = []
        for operation in ir.operations:
            body.extend(self._emit_operation(operation, value_types))
        argument_list = ", ".join(
            f"%{argument.name}: {value_types[argument.name]}" for argument in ir.arguments
        )
        kernel_attrs = self._kernel_attrs(ir, target)
        module_attrs = self._module_attrs(target)
        gpu_module_attrs = self._gpu_module_attrs(target)
        body_text = "\n".join(f"        {line}" for line in body + ["gpu.return"])
        text = (
            f"module attributes {{{module_attrs}}} {{\n"
            f"  gpu.module @kernels attributes {{{gpu_module_attrs}}} {{\n"
            f"    gpu.func @{ir.name}({argument_list}) kernel attributes {{{kernel_attrs}}} {{\n"
            f"{body_text}\n"
            f"    }}\n"
            f"  }}\n"
            f"}}\n"
        )
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="gpu_mlir",
            text=text,
        )

    def _module_attrs(self, target: AMDTarget) -> str:
        attrs = [
            f'baybridge.target = "{target.arch}"',
            f"rocdl.wave_size = {target.wave_size}",
        ]
        if target.rocm_version is not None:
            attrs.append(f'rocm.version = "{target.rocm_version}"')
        return ", ".join(attrs)

    def _gpu_module_attrs(self, target: AMDTarget) -> str:
        attrs = [
            f'rocdl.target = "{target.arch}"',
            f"rocdl.wave_size = {target.wave_size}",
        ]
        return ", ".join(attrs)

    def _kernel_attrs(self, ir: PortableKernelIR, target: AMDTarget) -> str:
        attrs = super()._kernel_attrs(ir, target)
        cluster = ", ".join(str(dim) for dim in ir.launch.cluster)
        return attrs + f", gpu.cluster = [{cluster}]"
