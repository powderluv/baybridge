from __future__ import annotations

from dataclasses import dataclass

from ..backend import LoweredModule
from ..diagnostics import CompilationError
from ..ir import Operation, PortableKernelIR, TensorSpec
from ..target import NvidiaTarget


@dataclass(frozen=True)
class _IndexedCopy1D:
    src_name: str
    dst_name: str
    dtype: str
    extent: int


@dataclass(frozen=True)
class _IndexedBinary1D:
    lhs_name: str
    rhs_name: str
    dst_name: str
    dtype: str
    extent: int
    op: str


class PtxBridge:
    _SUPPORTED_DTYPES = {"f32": 4, "i32": 4}

    def supports(self, ir: PortableKernelIR, target: NvidiaTarget) -> bool:
        try:
            self.lower(ir, target, backend_name="ptx_ref")
        except CompilationError:
            return False
        return True

    def lower(self, ir: PortableKernelIR, target: NvidiaTarget, *, backend_name: str) -> LoweredModule:
        match = self._match(ir)
        if isinstance(match, _IndexedCopy1D):
            text = self._render_indexed_copy(ir, target, match, backend_name=backend_name)
        else:
            text = self._render_indexed_binary(ir, target, match, backend_name=backend_name)
        return LoweredModule(
            backend_name=backend_name,
            entry_point=ir.name,
            dialect="ptx",
            text=text,
        )

    def _match(self, ir: PortableKernelIR) -> _IndexedCopy1D | _IndexedBinary1D:
        copy_match = self._match_indexed_copy_1d(ir)
        if copy_match is not None:
            return copy_match
        binary_match = self._match_indexed_binary_1d(ir)
        if binary_match is not None:
            return binary_match
        raise CompilationError(
            "ptx_ref currently supports only canonical indexed rank-1 dense copy and pointwise f32/i32 add/sub/mul/div kernels"
        )

    def _match_indexed_copy_1d(self, ir: PortableKernelIR) -> _IndexedCopy1D | None:
        if len(ir.arguments) != 2 or len(ir.operations) != 13:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype != dst_spec.dtype or src_spec.shape != dst_spec.shape:
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        load_op = ir.operations[11]
        store_op = ir.operations[12]
        if load_op.op != "load" or load_op.inputs != (src_arg.name, idx_name):
            return None
        if load_op.attrs.get("rank") != 1:
            return None
        if store_op.op != "store" or store_op.inputs != (load_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _IndexedCopy1D(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
        )

    def _match_indexed_binary_1d(self, ir: PortableKernelIR) -> _IndexedBinary1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 15:
            return None
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        lhs_spec = self._require_rank1_tensor(lhs_arg.spec)
        rhs_spec = self._require_rank1_tensor(rhs_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if lhs_spec is None or rhs_spec is None or dst_spec is None:
            return None
        if lhs_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != dst_spec.dtype:
            return None
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        lhs_load = ir.operations[11]
        rhs_load = ir.operations[12]
        binary_op = ir.operations[13]
        store_op = ir.operations[14]
        if lhs_load.op != "load" or lhs_load.inputs != (lhs_arg.name, idx_name):
            return None
        if rhs_load.op != "load" or rhs_load.inputs != (rhs_arg.name, idx_name):
            return None
        if lhs_load.attrs.get("rank") != 1 or rhs_load.attrs.get("rank") != 1:
            return None
        if binary_op.op not in {"add", "sub", "mul", "div"}:
            return None
        if binary_op.inputs != (lhs_load.outputs[0], rhs_load.outputs[0]):
            return None
        if store_op.op != "store" or store_op.inputs != (binary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _IndexedBinary1D(
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            extent=lhs_spec.shape[0],
            op=binary_op.op,
        )

    def _match_canonical_index_prefix(self, operations: tuple[Operation, ...]) -> str | None:
        if len(operations) != 11:
            return None
        if [op.op for op in operations[:3]] != ["thread_idx", "thread_idx", "thread_idx"]:
            return None
        if [op.attrs.get("axis") for op in operations[:3]] != ["x", "y", "z"]:
            return None
        if [op.op for op in operations[3:6]] != ["block_idx", "block_idx", "block_idx"]:
            return None
        if [op.attrs.get("axis") for op in operations[3:6]] != ["x", "y", "z"]:
            return None
        if [op.op for op in operations[6:9]] != ["block_dim", "block_dim", "block_dim"]:
            return None
        if [op.attrs.get("axis") for op in operations[6:9]] != ["x", "y", "z"]:
            return None
        thread_x = operations[0].outputs[0]
        block_x = operations[3].outputs[0]
        block_dim_x = operations[6].outputs[0]
        mul_op = operations[9]
        add_op = operations[10]
        if mul_op.op != "mul" or mul_op.inputs != (block_x, block_dim_x):
            return None
        if add_op.op != "add" or add_op.inputs != (mul_op.outputs[0], thread_x):
            return None
        return add_op.outputs[0]

    def _require_rank1_tensor(self, spec: object) -> TensorSpec | None:
        if not isinstance(spec, TensorSpec):
            return None
        layout = spec.resolved_layout()
        if len(spec.shape) != 1:
            return None
        if layout.stride != (1,):
            return None
        if spec.address_space.value != "global":
            return None
        return spec

    def _render_indexed_copy(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _IndexedCopy1D,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
        ]
        body.extend(self._declarations(dtype=match.dtype, rhs_param=False))
        body.extend(self._common_index_lines(match.extent, param_names=(f"{match.src_name}_param", f"{match.dst_name}_param")))
        body.extend(self._load_store_lines(dtype=match.dtype, lhs_ptr="%rd4", dst_ptr="%rd5"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_indexed_binary(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _IndexedBinary1D,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.lhs_name}_param,",
            f"    .param .u64 {match.rhs_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
        ]
        body.extend(self._declarations(dtype=match.dtype, rhs_param=True))
        body.extend(
            self._common_index_lines(
                match.extent,
                param_names=(f"{match.lhs_name}_param", f"{match.rhs_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._binary_lines(dtype=match.dtype, op=match.op))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _ptx_prologue(self, ir: PortableKernelIR, target: NvidiaTarget, *, backend_name: str) -> str:
        return (
            f"// baybridge.{backend_name}\n"
            f"// target: {target.sm}\n"
            f"// ptx_version: {target.ptx_version}\n"
            f"// warp_size: {target.warp_size}\n"
            f"// launch: grid={ir.launch.grid}, block={ir.launch.block}, shared_mem_bytes={ir.launch.shared_mem_bytes}\n"
            f".version {target.ptx_version}\n"
            f".target {target.sm}\n"
            ".address_size 64\n\n"
        )

    def _declarations(self, *, dtype: str, rhs_param: bool) -> list[str]:
        lines = [
            "    .reg .pred %p<2>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<6>;" if not rhs_param else "    .reg .b64 %rd<8>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<4>;")
        return lines

    def _common_index_lines(self, extent: int, *, param_names: tuple[str, ...]) -> list[str]:
        rhs_param = len(param_names) == 3
        lines = [
            f"    ld.param.u64 %rd1, [{param_names[0]}];",
            f"    ld.param.u64 %rd2, [{param_names[1]}];",
        ]
        if rhs_param:
            lines.append(f"    ld.param.u64 %rd3, [{param_names[2]}];")
        lines.extend(
            [
                "    mov.u32 %r1, %tid.x;",
                "    mov.u32 %r2, %ctaid.x;",
                "    mov.u32 %r3, %ntid.x;",
                "    mad.lo.u32 %r4, %r2, %r3, %r1;",
                f"    setp.ge.u32 %p1, %r4, {extent};",
                "    @%p1 bra L_done;",
                "    mul.wide.u32 %rd3, %r4, 4;" if not rhs_param else "    mul.wide.u32 %rd4, %r4, 4;",
            ]
        )
        if rhs_param:
            lines.extend(
                [
                    "    add.s64 %rd5, %rd1, %rd4;",
                    "    add.s64 %rd6, %rd2, %rd4;",
                    "    add.s64 %rd7, %rd3, %rd4;",
                ]
            )
        else:
            lines.extend(
                [
                    "    add.s64 %rd4, %rd1, %rd3;",
                    "    add.s64 %rd5, %rd2, %rd3;",
                ]
            )
        return lines

    def _load_store_lines(self, *, dtype: str, lhs_ptr: str, dst_ptr: str) -> list[str]:
        if dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{lhs_ptr}];",
                f"    st.global.f32 [{dst_ptr}], %f1;",
            ]
        if dtype == "i32":
            return [
                f"    ld.global.s32 %r5, [{lhs_ptr}];",
                f"    st.global.s32 [{dst_ptr}], %r5;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _binary_lines(self, *, dtype: str, op: str) -> list[str]:
        if dtype == "f32":
            instr = {
                "add": "add.rn.f32",
                "sub": "sub.rn.f32",
                "mul": "mul.rn.f32",
                "div": "div.rn.f32",
            }[op]
            return [
                "    ld.global.f32 %f1, [%rd5];",
                "    ld.global.f32 %f2, [%rd6];",
                f"    {instr} %f3, %f1, %f2;",
                "    st.global.f32 [%rd7], %f3;",
            ]
        if dtype == "i32":
            instr = {
                "add": "add.s32",
                "sub": "sub.s32",
                "mul": "mul.lo.s32",
                "div": "div.s32",
            }[op]
            return [
                "    ld.global.s32 %r5, [%rd5];",
                "    ld.global.s32 %r1, [%rd6];",
                f"    {instr} %r5, %r5, %r1;",
                "    st.global.s32 [%rd7], %r5;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")
