from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..ir import AddressSpace, KernelArgument, Layout, Operation, PortableKernelIR, ScalarSpec, TensorSpec, normalize_address_space
from ..target import AMDTarget

_HIPKITTENS_ENV = "BAYBRIDGE_HIPKITTENS_ROOT"


@dataclass(frozen=True)
class HipKittensMatch:
    family: str
    operand_dtype: str
    accumulator_dtype: str
    tile: tuple[int, int, int]
    reference_paths: tuple[str, ...]
    notes: tuple[str, ...]
    op_counts: dict[str, int]


class HipKittensRefBackend:
    name = "hipkittens_ref"
    artifact_extension = ".cpp"

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        try:
            match = self._analyze(ir)
        except BackendNotImplementedError:
            return False
        return match.family in {"tensorop_gemm", "attention"}

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        match = self._analyze(ir)
        root = self._configured_root()
        text = self._render_cpp(ir, target, match, root)
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="hipkittens_cpp",
            text=text,
        )

    def _analyze(self, ir: PortableKernelIR) -> HipKittensMatch:
        value_specs = self._seed_value_specs(ir.arguments)
        for operation in ir.operations:
            self._record_result_specs(operation, value_specs)
        return self._match_family(ir, value_specs)

    def _seed_value_specs(self, arguments: tuple[KernelArgument, ...]) -> dict[str, TensorSpec | ScalarSpec]:
        return {argument.name: argument.spec for argument in arguments}

    def _record_result_specs(self, operation: Operation, value_specs: dict[str, TensorSpec | ScalarSpec]) -> None:
        result = operation.attrs.get("result")
        if not isinstance(result, dict):
            return
        outputs = operation.outputs
        if not outputs:
            return
        output = outputs[0]
        kind = result.get("kind")
        if kind == "tensor":
            layout_dict = result.get("layout")
            layout = None
            if isinstance(layout_dict, dict):
                layout = Layout(
                    shape=tuple(layout_dict["shape"]),
                    stride=tuple(layout_dict["stride"]),
                    swizzle=layout_dict.get("swizzle"),
                )
            value_specs[output] = TensorSpec(
                shape=tuple(result["shape"]),
                dtype=str(result["dtype"]),
                layout=layout,
                address_space=normalize_address_space(result.get("address_space", AddressSpace.GLOBAL.value)),
            )
            return
        if kind == "scalar":
            value_specs[output] = ScalarSpec(dtype=str(result["dtype"]))

    def _match_family(self, ir: PortableKernelIR, value_specs: dict[str, TensorSpec | ScalarSpec]) -> HipKittensMatch:
        op_counts = Counter(operation.op for operation in ir.operations)
        mma_ops = [operation for operation in ir.operations if operation.op == "mma"]
        if not mma_ops:
            raise BackendNotImplementedError(
                "hipkittens_ref currently targets Baybridge GEMM and attention kernel families"
            )
        mma_op = mma_ops[0]
        a_name, b_name, acc_name = mma_op.inputs
        a_spec = value_specs.get(a_name)
        b_spec = value_specs.get(b_name)
        acc_spec = value_specs.get(acc_name)
        if not isinstance(a_spec, TensorSpec) or not isinstance(b_spec, TensorSpec) or not isinstance(acc_spec, TensorSpec):
            raise BackendNotImplementedError("hipkittens_ref requires tensor specs for mma operands")
        tile_attr = mma_op.attrs.get("tile")
        if isinstance(tile_attr, list) and len(tile_attr) == 3:
            tile = tuple(int(dim) for dim in tile_attr)
        elif isinstance(tile_attr, tuple) and len(tile_attr) == 3:
            tile = tuple(int(dim) for dim in tile_attr)
        elif len(a_spec.shape) == 2 and len(b_spec.shape) == 2:
            tile = (int(a_spec.shape[0]), int(b_spec.shape[1]), int(a_spec.shape[1]))
        else:
            raise BackendNotImplementedError("hipkittens_ref could not infer an mma tile for this kernel")

        notes: list[str] = []
        if op_counts.get("copy_async"):
            notes.append("async global-to-shared staging is present")
        if op_counts.get("barrier"):
            notes.append("block or grid barriers are present")
        if op_counts.get("local_tile") or op_counts.get("partition"):
            notes.append("tiled tensor views are present")
        if op_counts.get("thread_fragment_load") or op_counts.get("thread_fragment_store"):
            notes.append("thread fragment movement is present")

        has_attention_math = op_counts.get("math_exp2", 0) > 0 or any(name.startswith("reduce_") for name in op_counts)
        if has_attention_math:
            family = "attention"
            references = (
                "kernels/attn/gqa/",
                "kernels/attn/gqa_causal/",
                "kernels/attn/gqa_backwards/",
                "kernels/attn/gqa_causal_backwards/",
            )
            if op_counts.get("math_exp2"):
                notes.append("softmax-style exp2 is present")
        elif a_spec.dtype in {"f16", "bf16"}:
            family = "tensorop_gemm"
            references = ("kernels/gemm/bf16fp32/",)
        else:
            family = "simt_gemm"
            references = ("kernels/gemm/bf16fp32/",)
            notes.append("HipKittens quick-start kernels are BF16/FP32-accumulate; this SIMT/f32 shape needs custom lowering")

        return HipKittensMatch(
            family=family,
            operand_dtype=a_spec.dtype,
            accumulator_dtype=acc_spec.dtype,
            tile=tile,
            reference_paths=references,
            notes=tuple(notes),
            op_counts=dict(sorted(op_counts.items())),
        )

    def _configured_root(self) -> Path | None:
        configured = os.environ.get(_HIPKITTENS_ENV)
        if not configured:
            return None
        path = Path(configured).expanduser().resolve()
        if (path / "include" / "kittens.cuh").exists():
            return path
        return None

    def _branch_hint(self, target: AMDTarget) -> str:
        if target.arch == "gfx942":
            return "cdna3"
        return "main"

    def _render_cpp(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        match: HipKittensMatch,
        root: Path | None,
    ) -> str:
        manifest = {
            "backend": self.name,
            "entry_point": ir.name,
            "family": match.family,
            "target_arch": target.arch,
            "wave_size": target.wave_size,
            "hipkittens_branch_hint": self._branch_hint(target),
            "configured_hipkittens_root": str(root) if root is not None else None,
            "operand_dtype": match.operand_dtype,
            "accumulator_dtype": match.accumulator_dtype,
            "tile": list(match.tile),
            "reference_paths": list(match.reference_paths),
            "op_counts": match.op_counts,
            "notes": list(match.notes),
        }
        manifest_text = "\n".join(f"// {line}" for line in json.dumps(manifest, indent=2, sort_keys=True).splitlines())
        root_note = (
            f"// Build hint: hipcc -I{root / 'include'} <generated.cpp>\n"
            if root is not None
            else f"// Build hint: set {_HIPKITTENS_ENV} to a HipKittens checkout, then compile with -I${_HIPKITTENS_ENV}/include\n"
        )
        reference_lines = "\n".join(f"//   - {path}" for path in match.reference_paths)
        note_lines = "\n".join(f"//   - {note}" for note in match.notes) or "//   - none"
        argument_lines = ",\n    ".join(self._argument_decl(argument) for argument in ir.arguments)
        tensor_summary = "\n".join(self._tensor_summary(argument) for argument in ir.arguments)
        return (
            f"{manifest_text}\n"
            f"{root_note}"
            "#include <cstdint>\n"
            "#include <hip/hip_runtime.h>\n"
            "#include \"kittens.cuh\"\n\n"
            "using namespace kittens;\n\n"
            "namespace baybridge::hipkittens_ref {\n\n"
            f"// Matched kernel family: {match.family}\n"
            f"// Suggested HipKittens references:\n{reference_lines}\n"
            f"// Lowering notes:\n{note_lines}\n\n"
            f"__global__ void {ir.name}_reference(\n    {argument_lines}\n) {{\n"
            "    // Reference-only bridge. Baybridge still executes through its existing runtime or executable backends.\n"
            "    // This source captures the kernel family match and the likely HipKittens landing zone.\n"
            f"    // Target arch: {target.arch} (branch hint: {self._branch_hint(target)})\n"
            f"    // Operand dtype: {match.operand_dtype}, accumulator dtype: {match.accumulator_dtype}\n"
            f"    // Tile: ({match.tile[0]}, {match.tile[1]}, {match.tile[2]})\n"
            f"{tensor_summary}\n"
            "}\n\n"
            "} // namespace baybridge::hipkittens_ref\n"
        )

    def _argument_decl(self, argument: KernelArgument) -> str:
        spec = argument.spec
        if isinstance(spec, TensorSpec):
            return f"{self._cpp_type(spec.dtype)}* {argument.name}"
        return f"{self._cpp_type(spec.dtype)} {argument.name}"

    def _tensor_summary(self, argument: KernelArgument) -> str:
        spec = argument.spec
        if isinstance(spec, TensorSpec):
            return (
                f"    // tensor {argument.name}: shape={spec.shape}, dtype={spec.dtype}, "
                f"address_space={spec.address_space.value}"
            )
        return f"    // scalar {argument.name}: dtype={spec.dtype}"

    def _cpp_type(self, dtype: str) -> str:
        table = {
            "i1": "bool",
            "i8": "std::int8_t",
            "i32": "std::int32_t",
            "i64": "std::int64_t",
            "index": "std::int64_t",
            "f16": "half",
            "bf16": "hip_bfloat16",
            "f32": "float",
        }
        try:
            return table[dtype]
        except KeyError as exc:
            raise BackendNotImplementedError(f"hipkittens_ref does not support dtype '{dtype}'") from exc
