from __future__ import annotations

from dataclasses import dataclass, replace
import struct

from ..backend import LoweredModule
from ..diagnostics import CompilationError
from ..ir import Operation, PortableKernelIR, ScalarSpec, TensorSpec
from ..target import NvidiaTarget


@dataclass(frozen=True)
class _CopyMatch1D:
    mode: str
    src_name: str
    dst_name: str
    dtype: str
    extent: int


@dataclass(frozen=True)
class _BinaryMatch1D:
    mode: str
    lhs_name: str
    rhs_name: str
    dst_name: str
    dtype: str
    extent: int
    op: str


@dataclass(frozen=True)
class _UnaryMatch1D:
    mode: str
    src_name: str
    dst_name: str
    dtype: str
    extent: int
    op: str


@dataclass(frozen=True)
class _ScalarBroadcastMatch1D:
    mode: str
    scalar_mode: str
    src_name: str
    scalar_name: str
    dst_name: str
    dtype: str
    extent: int
    op: str


@dataclass(frozen=True)
class _ReduceMatch1D:
    src_name: str
    dst_name: str
    dtype: str
    extent: int
    op: str
    init_value: float | int


@dataclass(frozen=True)
class _ReduceBundle2DMatch:
    src_name: str
    dst_scalar_name: str
    dst_rows_name: str
    dst_cols_name: str
    dtype: str
    rows: int
    cols: int
    op: str
    scalar_init: float | int
    rows_init: float | int
    cols_init: float | int
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorFactoryBundle2DMatch:
    dst_zero_name: str
    dst_one_name: str
    dst_full_name: str
    dtype: str
    rows: int
    cols: int
    full_value: float | int


class PtxBridge:
    _SUPPORTED_DTYPES = {"f32": 4, "i32": 4}
    _SUPPORTED_BINARY_OPS = {"add", "sub", "mul", "div"}
    _SUPPORTED_UNARY_OPS = {"math_sqrt", "math_rsqrt"}

    def supports(self, ir: PortableKernelIR, target: NvidiaTarget) -> bool:
        try:
            self.lower(ir, target, backend_name="ptx_ref")
        except CompilationError:
            return False
        return True

    def lower(self, ir: PortableKernelIR, target: NvidiaTarget, *, backend_name: str) -> LoweredModule:
        match = self._match(ir)
        if isinstance(match, _CopyMatch1D):
            text = self._render_copy(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _BinaryMatch1D):
            text = self._render_binary(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _UnaryMatch1D):
            text = self._render_unary(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _ReduceMatch1D):
            text = self._render_reduce(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _ReduceBundle2DMatch):
            if match.parallel:
                text = self._render_parallel_reduce_bundle_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_reduce_bundle_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorFactoryBundle2DMatch):
            text = self._render_tensor_factory_bundle_2d(ir, target, match, backend_name=backend_name)
        else:
            text = self._render_scalar_broadcast(ir, target, match, backend_name=backend_name)
        return LoweredModule(
            backend_name=backend_name,
            entry_point=ir.name,
            dialect="ptx",
            text=text,
        )

    def _match(
        self,
        ir: PortableKernelIR,
    ) -> _CopyMatch1D | _BinaryMatch1D | _UnaryMatch1D | _ScalarBroadcastMatch1D | _ReduceMatch1D | _ReduceBundle2DMatch | _TensorFactoryBundle2DMatch:
        for matcher in (
            self._match_indexed_copy_1d,
            self._match_indexed_binary_1d,
            self._match_indexed_unary_1d,
            self._match_indexed_scalar_broadcast_1d,
            self._match_indexed_tensor_scalar_broadcast_1d,
            self._match_direct_copy_1d,
            self._match_direct_binary_1d,
            self._match_direct_unary_1d,
            self._match_direct_scalar_broadcast_1d,
            self._match_direct_tensor_scalar_broadcast_1d,
            self._match_reduce_1d_to_scalar,
            self._match_parallel_reduce_bundle_2d,
            self._match_reduce_bundle_2d,
            self._match_tensor_factory_bundle_2d,
        ):
            match = matcher(ir)
            if match is not None:
                return match
        raise CompilationError(
            "ptx_ref currently supports only exact rank-1 dense copy, pointwise f32/i32 add/sub/mul/div, "
            "exact f32 sqrt/rsqrt unary kernels, exact rank-1 scalar-broadcast kernels, "
            "exact rank-1 scalar reductions to dst[0], the exact 2D f32/i32 reduction bundle families, "
            "and the exact 2D f32/i32 tensor-factory bundle family"
        )

    def _match_indexed_copy_1d(self, ir: PortableKernelIR) -> _CopyMatch1D | None:
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
        return _CopyMatch1D(
            mode="indexed",
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
        )

    def _match_indexed_binary_1d(self, ir: PortableKernelIR) -> _BinaryMatch1D | None:
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
        if binary_op.op not in self._SUPPORTED_BINARY_OPS:
            return None
        if binary_op.inputs != (lhs_load.outputs[0], rhs_load.outputs[0]):
            return None
        if store_op.op != "store" or store_op.inputs != (binary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _BinaryMatch1D(
            mode="indexed",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            extent=lhs_spec.shape[0],
            op=binary_op.op,
        )

    def _match_direct_copy_1d(self, ir: PortableKernelIR) -> _CopyMatch1D | None:
        if len(ir.arguments) != 2 or len(ir.operations) != 5:
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
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        load_op = ir.operations[3]
        store_op = ir.operations[4]
        if load_op.op != "load" or load_op.inputs != (src_arg.name, idx_name):
            return None
        if load_op.attrs.get("rank") != 1:
            return None
        if store_op.op != "store" or store_op.inputs != (load_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _CopyMatch1D(
            mode="direct",
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
        )

    def _match_direct_binary_1d(self, ir: PortableKernelIR) -> _BinaryMatch1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 7:
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
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        lhs_load = ir.operations[3]
        rhs_load = ir.operations[4]
        binary_op = ir.operations[5]
        store_op = ir.operations[6]
        if lhs_load.op != "load" or lhs_load.inputs != (lhs_arg.name, idx_name):
            return None
        if rhs_load.op != "load" or rhs_load.inputs != (rhs_arg.name, idx_name):
            return None
        if lhs_load.attrs.get("rank") != 1 or rhs_load.attrs.get("rank") != 1:
            return None
        if binary_op.op not in self._SUPPORTED_BINARY_OPS:
            return None
        if binary_op.inputs != (lhs_load.outputs[0], rhs_load.outputs[0]):
            return None
        if store_op.op != "store" or store_op.inputs != (binary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _BinaryMatch1D(
            mode="direct",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            extent=lhs_spec.shape[0],
            op=binary_op.op,
        )

    def _match_indexed_unary_1d(self, ir: PortableKernelIR) -> _UnaryMatch1D | None:
        if len(ir.arguments) != 2 or len(ir.operations) != 14:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype != "f32" or dst_spec.dtype != "f32":
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        load_op = ir.operations[11]
        unary_op = ir.operations[12]
        store_op = ir.operations[13]
        if load_op.op != "load" or load_op.inputs != (src_arg.name, idx_name):
            return None
        if load_op.attrs.get("rank") != 1:
            return None
        if unary_op.op not in self._SUPPORTED_UNARY_OPS:
            return None
        if unary_op.inputs != (load_op.outputs[0],):
            return None
        if store_op.op != "store" or store_op.inputs != (unary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _UnaryMatch1D(
            mode="indexed",
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=unary_op.op,
        )

    def _match_direct_unary_1d(self, ir: PortableKernelIR) -> _UnaryMatch1D | None:
        if len(ir.arguments) != 2 or len(ir.operations) != 6:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype != "f32" or dst_spec.dtype != "f32":
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        load_op = ir.operations[3]
        unary_op = ir.operations[4]
        store_op = ir.operations[5]
        if load_op.op != "load" or load_op.inputs != (src_arg.name, idx_name):
            return None
        if load_op.attrs.get("rank") != 1:
            return None
        if unary_op.op not in self._SUPPORTED_UNARY_OPS:
            return None
        if unary_op.inputs != (load_op.outputs[0],):
            return None
        if store_op.op != "store" or store_op.inputs != (unary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _UnaryMatch1D(
            mode="direct",
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=unary_op.op,
        )

    def _match_direct_scalar_broadcast_1d(self, ir: PortableKernelIR) -> _ScalarBroadcastMatch1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 6:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None or not isinstance(scalar_arg.spec, ScalarSpec):
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if src_spec.dtype != scalar_arg.spec.dtype or src_spec.dtype != dst_spec.dtype:
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        src_load = ir.operations[3]
        binary_op = ir.operations[4]
        store_op = ir.operations[5]
        if src_load.op != "load" or src_load.inputs != (src_arg.name, idx_name):
            return None
        if src_load.attrs.get("rank") != 1:
            return None
        if binary_op.op not in self._SUPPORTED_BINARY_OPS:
            return None
        if binary_op.inputs != (src_load.outputs[0], scalar_arg.name):
            return None
        if store_op.op != "store" or store_op.inputs != (binary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarBroadcastMatch1D(
            mode="direct",
            scalar_mode="param",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=binary_op.op,
        )

    def _match_direct_tensor_scalar_broadcast_1d(self, ir: PortableKernelIR) -> _ScalarBroadcastMatch1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 8:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        scalar_spec = self._require_rank1_tensor(scalar_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or scalar_spec is None or dst_spec is None:
            return None
        if scalar_spec.shape != (1,):
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if src_spec.dtype != scalar_spec.dtype or src_spec.dtype != dst_spec.dtype:
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        src_load = ir.operations[3]
        cst_op = ir.operations[4]
        scalar_load = ir.operations[5]
        binary_op = ir.operations[6]
        store_op = ir.operations[7]
        if src_load.op != "load" or src_load.inputs != (src_arg.name, idx_name):
            return None
        if src_load.attrs.get("rank") != 1:
            return None
        if cst_op.op != "constant" or cst_op.attrs.get("value") != 0:
            return None
        zero_name = cst_op.outputs[0]
        if scalar_load.op != "load" or scalar_load.inputs != (scalar_arg.name, zero_name):
            return None
        if scalar_load.attrs.get("rank") != 1:
            return None
        if binary_op.op not in self._SUPPORTED_BINARY_OPS:
            return None
        if binary_op.inputs != (src_load.outputs[0], scalar_load.outputs[0]):
            return None
        if store_op.op != "store" or store_op.inputs != (binary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarBroadcastMatch1D(
            mode="direct",
            scalar_mode="tensor",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=binary_op.op,
        )

    def _match_reduce_1d_to_scalar(self, ir: PortableKernelIR) -> _ReduceMatch1D | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block[1:] != (1, 1):
            return None
        if len(ir.arguments) != 2 or len(ir.operations) != 4:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if dst_spec.dtype != src_spec.dtype or dst_spec.shape != (1,):
            return None
        const_init, reduce_op, const_index, store_op = ir.operations
        if const_init.op != "constant" or const_index.op != "constant":
            return None
        if reduce_op.op not in {"reduce_add", "reduce_mul", "reduce_max", "reduce_min"}:
            return None
        if reduce_op.inputs != (src_arg.name, const_init.outputs[0]):
            return None
        if reduce_op.attrs.get("reduction_profile") != 0:
            return None
        if const_index.attrs.get("value") != 0:
            return None
        if store_op.op != "store" or store_op.inputs != (reduce_op.outputs[0], dst_arg.name, const_index.outputs[0]):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        init_result = const_init.attrs.get("result", {})
        if init_result.get("kind") != "scalar" or init_result.get("dtype") != src_spec.dtype:
            return None
        init_value = const_init.attrs.get("value")
        if src_spec.dtype == "f32":
            init_value = float(init_value)
        else:
            init_value = int(init_value)
        return _ReduceMatch1D(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=reduce_op.op,
            init_value=init_value,
        )

    def _match_reduce_bundle_2d(self, ir: PortableKernelIR) -> _ReduceBundle2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 4 or len(ir.operations) != 12:
            return None
        src_arg, dst_scalar_arg, dst_rows_arg, dst_cols_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_scalar_spec = self._require_rank1_tensor(dst_scalar_arg.spec)
        dst_rows_spec = self._require_rank1_tensor(dst_rows_arg.spec)
        dst_cols_spec = self._require_rank1_tensor(dst_cols_arg.spec)
        if src_spec is None or dst_scalar_spec is None or dst_rows_spec is None or dst_cols_spec is None:
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        rows, cols = src_spec.shape
        if dst_scalar_spec.dtype != src_spec.dtype or dst_scalar_spec.shape != (1,):
            return None
        if dst_rows_spec.dtype != src_spec.dtype or dst_rows_spec.shape != (rows,):
            return None
        if dst_cols_spec.dtype != src_spec.dtype or dst_cols_spec.shape != (cols,):
            return None
        ops = ir.operations
        op_name = ops[3].op
        if op_name not in {"reduce_add", "reduce_mul", "reduce_max", "reduce_min"}:
            return None
        if [op.op for op in ops] != [
            "make_tensor",
            "copy",
            "constant",
            op_name,
            "constant",
            "store",
            "constant",
            op_name,
            "copy",
            "constant",
            op_name,
            "copy",
        ]:
            return None
        if tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
            return None
        if tuple(ops[3].inputs) != (ops[0].outputs[0], ops[2].outputs[0]):
            return None
        if ops[3].attrs.get("reduction_profile") != 0:
            return None
        if ops[4].attrs.get("value") != 0:
            return None
        if tuple(ops[5].inputs) != (ops[3].outputs[0], dst_scalar_arg.name, ops[4].outputs[0]):
            return None
        if ops[5].attrs.get("rank") != 1:
            return None
        if tuple(ops[7].inputs) != (ops[0].outputs[0], ops[6].outputs[0]):
            return None
        if ops[7].attrs.get("reduction_profile") != [None, 1]:
            return None
        if tuple(ops[8].inputs) != (ops[7].outputs[0], dst_rows_arg.name):
            return None
        if tuple(ops[10].inputs) != (ops[0].outputs[0], ops[9].outputs[0]):
            return None
        if ops[10].attrs.get("reduction_profile") != [1, None]:
            return None
        if tuple(ops[11].inputs) != (ops[10].outputs[0], dst_cols_arg.name):
            return None
        return _ReduceBundle2DMatch(
            src_name=src_arg.name,
            dst_scalar_name=dst_scalar_arg.name,
            dst_rows_name=dst_rows_arg.name,
            dst_cols_name=dst_cols_arg.name,
            dtype=src_spec.dtype,
            rows=rows,
            cols=cols,
            op=op_name,
            scalar_init=self._reduce_bundle_init_value(src_spec.dtype, ops[2].attrs["value"]),
            rows_init=self._reduce_bundle_init_value(src_spec.dtype, ops[6].attrs["value"]),
            cols_init=self._reduce_bundle_init_value(src_spec.dtype, ops[9].attrs["value"]),
        )

    def _match_parallel_reduce_bundle_2d(self, ir: PortableKernelIR) -> _ReduceBundle2DMatch | None:
        block_x, block_y, block_z = ir.launch.block
        if ir.launch.grid != (1, 1, 1) or block_y != 1 or block_z != 1:
            return None
        if block_x <= 1 or block_x > 1024 or (block_x & (block_x - 1)) != 0:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, block=(1, 1, 1)))
        match = self._match_reduce_bundle_2d(serial_ir)
        if match is None or match.op not in {"reduce_add", "reduce_mul", "reduce_max", "reduce_min"}:
            return None
        if block_x < max(match.rows, match.cols):
            return None
        return _ReduceBundle2DMatch(
            src_name=match.src_name,
            dst_scalar_name=match.dst_scalar_name,
            dst_rows_name=match.dst_rows_name,
            dst_cols_name=match.dst_cols_name,
            dtype=match.dtype,
            rows=match.rows,
            cols=match.cols,
            op=match.op,
            scalar_init=match.scalar_init,
            rows_init=match.rows_init,
            cols_init=match.cols_init,
            parallel=True,
            block_x=block_x,
        )

    def _match_tensor_factory_bundle_2d(self, ir: PortableKernelIR) -> _TensorFactoryBundle2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3 or len(ir.operations) != 12:
            return None
        dst_zero_arg, dst_one_arg, dst_full_arg = ir.arguments
        dst_zero_spec = self._require_rank2_tensor(dst_zero_arg.spec)
        dst_one_spec = self._require_rank2_tensor(dst_one_arg.spec)
        dst_full_spec = self._require_rank2_tensor(dst_full_arg.spec)
        if dst_zero_spec is None or dst_one_spec is None or dst_full_spec is None:
            return None
        if dst_zero_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if dst_zero_spec.dtype != dst_one_spec.dtype or dst_zero_spec.dtype != dst_full_spec.dtype:
            return None
        if dst_zero_spec.shape != dst_one_spec.shape or dst_zero_spec.shape != dst_full_spec.shape:
            return None
        ops = ir.operations
        expected = [
            "make_tensor",
            "constant",
            "fill",
            "copy",
            "make_tensor",
            "constant",
            "fill",
            "copy",
            "make_tensor",
            "constant",
            "fill",
            "copy",
        ]
        if [op.op for op in ops] != expected:
            return None
        if ops[1].attrs.get("value") != 0 or ops[5].attrs.get("value") != 1:
            return None
        groups = (
            (ops[0], ops[1], ops[2], ops[3], dst_zero_arg.name),
            (ops[4], ops[5], ops[6], ops[7], dst_one_arg.name),
            (ops[8], ops[9], ops[10], ops[11], dst_full_arg.name),
        )
        for make_tensor, constant, fill, copy, dst_name in groups:
            if make_tensor.attrs.get("shape") != list(dst_zero_spec.shape):
                return None
            if make_tensor.attrs.get("dtype") != dst_zero_spec.dtype:
                return None
            if make_tensor.attrs.get("address_space") != "register":
                return None
            if constant.attrs.get("result", {}).get("dtype") != dst_zero_spec.dtype:
                return None
            if tuple(fill.inputs) != (make_tensor.outputs[0], constant.outputs[0]):
                return None
            if tuple(copy.inputs) != (make_tensor.outputs[0], dst_name):
                return None
        return _TensorFactoryBundle2DMatch(
            dst_zero_name=dst_zero_arg.name,
            dst_one_name=dst_one_arg.name,
            dst_full_name=dst_full_arg.name,
            dtype=dst_zero_spec.dtype,
            rows=dst_zero_spec.shape[0],
            cols=dst_zero_spec.shape[1],
            full_value=self._reduce_bundle_init_value(dst_zero_spec.dtype, ops[9].attrs["value"]),
        )

    def _match_indexed_scalar_broadcast_1d(self, ir: PortableKernelIR) -> _ScalarBroadcastMatch1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 14:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None or not isinstance(scalar_arg.spec, ScalarSpec):
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if src_spec.dtype != scalar_arg.spec.dtype or src_spec.dtype != dst_spec.dtype:
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        src_load = ir.operations[11]
        binary_op = ir.operations[12]
        store_op = ir.operations[13]
        if src_load.op != "load" or src_load.inputs != (src_arg.name, idx_name):
            return None
        if src_load.attrs.get("rank") != 1:
            return None
        if binary_op.op not in self._SUPPORTED_BINARY_OPS:
            return None
        if binary_op.inputs != (src_load.outputs[0], scalar_arg.name):
            return None
        if store_op.op != "store" or store_op.inputs != (binary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarBroadcastMatch1D(
            mode="indexed",
            scalar_mode="param",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=binary_op.op,
        )

    def _match_indexed_tensor_scalar_broadcast_1d(self, ir: PortableKernelIR) -> _ScalarBroadcastMatch1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 16:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        scalar_spec = self._require_rank1_tensor(scalar_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or scalar_spec is None or dst_spec is None:
            return None
        if scalar_spec.shape != (1,):
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if src_spec.dtype != scalar_spec.dtype or src_spec.dtype != dst_spec.dtype:
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        src_load = ir.operations[11]
        cst_op = ir.operations[12]
        scalar_load = ir.operations[13]
        binary_op = ir.operations[14]
        store_op = ir.operations[15]
        if src_load.op != "load" or src_load.inputs != (src_arg.name, idx_name):
            return None
        if src_load.attrs.get("rank") != 1:
            return None
        if cst_op.op != "constant" or cst_op.attrs.get("value") != 0:
            return None
        zero_name = cst_op.outputs[0]
        if scalar_load.op != "load" or scalar_load.inputs != (scalar_arg.name, zero_name):
            return None
        if scalar_load.attrs.get("rank") != 1:
            return None
        if binary_op.op not in self._SUPPORTED_BINARY_OPS:
            return None
        if binary_op.inputs != (src_load.outputs[0], scalar_load.outputs[0]):
            return None
        if store_op.op != "store" or store_op.inputs != (binary_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarBroadcastMatch1D(
            mode="indexed",
            scalar_mode="tensor",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
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

    def _match_direct_index_prefix(self, operations: tuple[Operation, ...]) -> str | None:
        if len(operations) != 3:
            return None
        if [op.op for op in operations] != ["thread_idx", "thread_idx", "thread_idx"]:
            return None
        if [op.attrs.get("axis") for op in operations] != ["x", "y", "z"]:
            return None
        return operations[0].outputs[0]

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

    def _require_rank2_tensor(self, spec: object) -> TensorSpec | None:
        if not isinstance(spec, TensorSpec):
            return None
        layout = spec.resolved_layout()
        if len(spec.shape) != 2:
            return None
        rows, cols = spec.shape
        if layout.stride != (cols, 1):
            return None
        if spec.address_space.value != "global":
            return None
        return spec

    def _render_copy(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _CopyMatch1D,
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
        body.extend(self._copy_declarations(match.dtype, mode=match.mode))
        body.extend(self._index_lines(match.mode, match.extent, param_names=(f"{match.src_name}_param", f"{match.dst_name}_param")))
        body.extend(self._copy_lines(match.dtype, lhs_ptr="%rd4", dst_ptr="%rd5"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_binary(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _BinaryMatch1D,
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
        body.extend(self._binary_declarations(match.dtype, mode=match.mode))
        body.extend(
            self._index_lines(
                match.mode,
                match.extent,
                param_names=(f"{match.lhs_name}_param", f"{match.rhs_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._binary_lines(match.dtype, match.op, lhs_ptr="%rd5", rhs_ptr="%rd6", dst_ptr="%rd7"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_unary(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _UnaryMatch1D,
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
        body.extend(self._unary_declarations(mode=match.mode))
        body.extend(self._index_lines(match.mode, match.extent, param_names=(f"{match.src_name}_param", f"{match.dst_name}_param")))
        body.extend(self._unary_lines(match.op, src_ptr="%rd4", dst_ptr="%rd5"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_reduce(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _ReduceMatch1D,
        *,
        backend_name: str,
    ) -> str:
        if self._is_parallel_reduce_launch(ir, match):
            return self._render_parallel_reduce(ir, target, match, backend_name=backend_name)
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
        ]
        body.extend(self._reduce_declarations(match.dtype))
        body.extend(self._reduce_lines(match))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_reduce(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _ReduceMatch1D,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        instr = self._reduce_bundle_instr(match.dtype, match.op)
        if match.dtype == "f32":
            shared_decl = f"    .shared .align 4 .f32 smem[{block_x}];"
            reg_decl = "    .reg .f32 %f<4>;"
            init_line = self._reduce_init_lines(match.dtype, match.init_value)[0]
            load_global = "    ld.global.f32 %f2, [%rd4];"
            store_shared = "    st.shared.f32 [%rd7], %f1;"
            load_shared = "    ld.shared.f32 %f3, [%rd7];"
            reduce_line = f"    {instr} %f1, %f1, %f2;"
            reduce_peer_line = f"    {instr} %f1, %f1, %f3;"
            store_shared_partial = "    st.shared.f32 [%rd6], %f1;"
            store_result = "    st.global.f32 [%rd2], %f3;"
        elif match.dtype == "i32":
            shared_decl = f"    .shared .align 4 .s32 smem[{block_x}];"
            reg_decl = ""
            init_line = f"    mov.s32 %r4, {int(match.init_value)};"
            load_global = "    ld.global.s32 %r5, [%rd4];"
            store_shared = "    st.shared.s32 [%rd7], %r4;"
            load_shared = "    ld.shared.s32 %r5, [%rd7];"
            reduce_line = f"    {instr} %r4, %r4, %r5;"
            reduce_peer_line = f"    {instr} %r4, %r4, %r5;"
            store_shared_partial = "    st.shared.s32 [%rd6], %r4;"
            store_result = "    st.global.s32 [%rd2], %r5;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            shared_decl,
            "    .reg .pred %p<4>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<8>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            init_line,
            "    mov.u32 %r2, %r1;",
            "L_reduce_load:",
            f"    setp.ge.u32 %p1, %r2, {match.extent};",
            "    @%p1 bra L_reduce_store_partial;",
            "    mul.wide.u32 %rd3, %r2, 4;",
            "    add.s64 %rd4, %rd1, %rd3;",
            load_global,
            reduce_line,
            f"    add.s32 %r2, %r2, {block_x};",
            "    bra.uni L_reduce_load;",
            "L_reduce_store_partial:",
            "    mov.u64 %rd5, smem;",
            "    mul.wide.u32 %rd6, %r1, 4;",
            "    add.s64 %rd7, %rd5, %rd6;",
            store_shared,
            "    bar.sync 0;",
        ]
        body = [line for line in body if line]
        stride = block_x // 2
        while stride >= 1:
            body.extend(
                [
                    f"    setp.ge.u32 %p2, %r1, {stride};",
                    f"    @%p2 bra L_skip_stride_{stride};",
                    f"    add.s32 %r3, %r1, {stride};",
                    "    mul.wide.u32 %rd7, %r3, 4;",
                    "    add.s64 %rd7, %rd5, %rd7;",
                    load_shared,
                    reduce_peer_line,
                    "    mul.wide.u32 %rd6, %r1, 4;",
                    "    add.s64 %rd6, %rd5, %rd6;",
                    store_shared_partial,
                    f"L_skip_stride_{stride}:",
                    "    bar.sync 0;",
                ]
            )
            stride //= 2
        body.extend(
            [
                "    setp.ne.u32 %p3, %r1, 0;",
                "    @%p3 bra L_done;",
                "    ld.shared." + ("f32 %f3" if match.dtype == "f32" else "s32 %r5") + ", [%rd5];",
                store_result,
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        return ptx + "\n".join(body) + "\n"

    def _render_reduce_bundle_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _ReduceBundle2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        reduce_instr = self._reduce_bundle_instr(match.dtype, match.op)
        if match.dtype == "f32":
            acc_reg = "%f1"
            value_reg = "%f2"
            index_reg = "%r3"
            reg_decl = "    .reg .f32 %f<3>;"
            load_line = lambda ptr: f"    ld.global.f32 {value_reg}, [{ptr}];"
            store_line = lambda ptr: f"    st.global.f32 [{ptr}], {acc_reg};"
        elif match.dtype == "i32":
            acc_reg = "%r3"
            value_reg = "%r4"
            index_reg = "%r6"
            reg_decl = ""
            load_line = lambda ptr: f"    ld.global.s32 {value_reg}, [{ptr}];"
            store_line = lambda ptr: f"    st.global.s32 [{ptr}], {acc_reg};"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_scalar_name}_param,",
            f"    .param .u64 {match.dst_rows_name}_param,",
            f"    .param .u64 {match.dst_cols_name}_param",
            ")",
            "{",
        ]
        body.extend(
            [
                "    .reg .pred %p<3>;",
                "    .reg .b32 %r<7>;",
                "    .reg .b64 %rd<7>;",
                reg_decl,
                f"    ld.param.u64 %rd1, [{match.src_name}_param];",
                f"    ld.param.u64 %rd2, [{match.dst_scalar_name}_param];",
                f"    ld.param.u64 %rd3, [{match.dst_rows_name}_param];",
                f"    ld.param.u64 %rd4, [{match.dst_cols_name}_param];",
            ]
        )
        body = [line for line in body if line]
        body.extend(self._reduce_init_lines(match.dtype, match.scalar_init))
        body.extend(
            [
                "    mov.u32 %r1, 0;",
                "L_scalar_rows:",
                f"    setp.ge.u32 %p1, %r1, {match.rows};",
                "    @%p1 bra L_store_scalar;",
                "    mov.u32 %r2, 0;",
                "L_scalar_cols:",
                f"    setp.ge.u32 %p2, %r2, {match.cols};",
                "    @%p2 bra L_next_scalar_row;",
                f"    mad.lo.u32 {index_reg}, %r1, {match.cols}, %r2;",
                f"    mul.wide.u32 %rd5, {index_reg}, 4;",
                "    add.s64 %rd6, %rd1, %rd5;",
                load_line("%rd6"),
                f"    {reduce_instr} {acc_reg}, {acc_reg}, {value_reg};",
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_scalar_cols;",
                "L_next_scalar_row:",
                "    add.s32 %r1, %r1, 1;",
                "    bra.uni L_scalar_rows;",
                "L_store_scalar:",
                store_line("%rd2"),
            ]
        )
        body.extend(
            [
                "    mov.u32 %r1, 0;",
                "L_rows_outer:",
                f"    setp.ge.u32 %p1, %r1, {match.rows};",
                "    @%p1 bra L_cols_init;",
                *self._reduce_init_lines(match.dtype, match.rows_init),
                "    mov.u32 %r2, 0;",
                "L_rows_inner:",
                f"    setp.ge.u32 %p2, %r2, {match.cols};",
                "    @%p2 bra L_store_row;",
                f"    mad.lo.u32 {index_reg}, %r1, {match.cols}, %r2;",
                f"    mul.wide.u32 %rd5, {index_reg}, 4;",
                "    add.s64 %rd6, %rd1, %rd5;",
                load_line("%rd6"),
                f"    {reduce_instr} {acc_reg}, {acc_reg}, {value_reg};",
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_rows_inner;",
                "L_store_row:",
                "    mul.wide.u32 %rd5, %r1, 4;",
                "    add.s64 %rd6, %rd3, %rd5;",
                store_line("%rd6"),
                "    add.s32 %r1, %r1, 1;",
                "    bra.uni L_rows_outer;",
                "L_cols_init:",
                "    mov.u32 %r2, 0;",
                "L_cols_outer:",
                f"    setp.ge.u32 %p1, %r2, {match.cols};",
                "    @%p1 bra L_done;",
                *self._reduce_init_lines(match.dtype, match.cols_init),
                "    mov.u32 %r1, 0;",
                "L_cols_inner:",
                f"    setp.ge.u32 %p2, %r1, {match.rows};",
                "    @%p2 bra L_store_col;",
                f"    mad.lo.u32 {index_reg}, %r1, {match.cols}, %r2;",
                f"    mul.wide.u32 %rd5, {index_reg}, 4;",
                "    add.s64 %rd6, %rd1, %rd5;",
                load_line("%rd6"),
                f"    {reduce_instr} {acc_reg}, {acc_reg}, {value_reg};",
                "    add.s32 %r1, %r1, 1;",
                "    bra.uni L_cols_inner;",
                "L_store_col:",
                "    mul.wide.u32 %rd5, %r2, 4;",
                "    add.s64 %rd6, %rd4, %rd5;",
                store_line("%rd6"),
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_cols_outer;",
            ]
        )
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_reduce_bundle_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _ReduceBundle2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = match.block_x
        reduce_instr = self._reduce_bundle_instr(match.dtype, match.op)
        if match.dtype == "f32":
            shared_decl = f"    .shared .align 4 .f32 smem[{block_x}];"
            reg_decl = "    .reg .f32 %f<4>;"
            scalar_init = self._reduce_init_lines(match.dtype, match.scalar_init)[0]
            row_init = self._reduce_init_lines(match.dtype, match.rows_init)[0]
            col_init = self._reduce_init_lines(match.dtype, match.cols_init)[0]
            load_line = lambda ptr: f"    ld.global.f32 %f2, [{ptr}];"
            store_acc_line = lambda ptr: f"    st.global.f32 [{ptr}], %f1;"
            store_scalar_line = "    st.global.f32 [%rd2], %f3;"
            shared_store_line = "    st.shared.f32 [%rd7], %f1;"
            shared_load_peer_line = "    ld.shared.f32 %f3, [%rd7];"
            shared_load_root_line = "    ld.shared.f32 %f3, [%rd5];"
            scalar_acc_reg = "%f1"
            value_reg = "%f2"
            peer_reg = "%f3"
            index_reg = "%r6"
        elif match.dtype == "i32":
            shared_decl = f"    .shared .align 4 .s32 smem[{block_x}];"
            reg_decl = ""
            scalar_init = f"    mov.s32 %r4, {int(match.scalar_init)};"
            row_init = f"    mov.s32 %r4, {int(match.rows_init)};"
            col_init = f"    mov.s32 %r4, {int(match.cols_init)};"
            load_line = lambda ptr: f"    ld.global.s32 %r5, [{ptr}];"
            store_acc_line = lambda ptr: f"    st.global.s32 [{ptr}], %r4;"
            store_scalar_line = "    st.global.s32 [%rd2], %r5;"
            shared_store_line = "    st.shared.s32 [%rd7], %r4;"
            shared_load_peer_line = "    ld.shared.s32 %r5, [%rd7];"
            shared_load_root_line = "    ld.shared.s32 %r5, [%rd5];"
            scalar_acc_reg = "%r4"
            value_reg = "%r5"
            peer_reg = "%r5"
            index_reg = "%r6"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_scalar_name}_param,",
            f"    .param .u64 {match.dst_rows_name}_param,",
            f"    .param .u64 {match.dst_cols_name}_param",
            ")",
            "{",
            shared_decl,
            "    .reg .pred %p<4>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<8>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_scalar_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_rows_name}_param];",
            f"    ld.param.u64 %rd4, [{match.dst_cols_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            scalar_init,
            "    mov.u32 %r2, %r1;",
            "L_scalar_load:",
            f"    setp.ge.u32 %p1, %r2, {match.rows * match.cols};",
            "    @%p1 bra L_scalar_store_partial;",
            "    mul.wide.u32 %rd5, %r2, 4;",
            "    add.s64 %rd6, %rd1, %rd5;",
            load_line("%rd6"),
            f"    {reduce_instr} {scalar_acc_reg}, {scalar_acc_reg}, {value_reg};",
            f"    add.s32 %r2, %r2, {block_x};",
            "    bra.uni L_scalar_load;",
            "L_scalar_store_partial:",
            "    mov.u64 %rd5, smem;",
            "    mul.wide.u32 %rd6, %r1, 4;",
            "    add.s64 %rd7, %rd5, %rd6;",
            shared_store_line,
            "    bar.sync 0;",
        ]
        body = [line for line in body if line]
        stride = block_x // 2
        while stride >= 1:
            body.extend(
                [
                    f"    setp.ge.u32 %p2, %r1, {stride};",
                    f"    @%p2 bra L_skip_scalar_stride_{stride};",
                    f"    add.s32 %r3, %r1, {stride};",
                    "    mul.wide.u32 %rd7, %r3, 4;",
                    "    add.s64 %rd7, %rd5, %rd7;",
                    shared_load_peer_line,
                    f"    {reduce_instr} {scalar_acc_reg}, {scalar_acc_reg}, {peer_reg};",
                    "    mul.wide.u32 %rd6, %r1, 4;",
                    "    add.s64 %rd6, %rd5, %rd6;",
                    shared_store_line.replace("%rd7", "%rd6"),
                    f"L_skip_scalar_stride_{stride}:",
                    "    bar.sync 0;",
                ]
            )
            stride //= 2
        body.extend(
            [
                "    setp.ne.u32 %p3, %r1, 0;",
                "    @%p3 bra L_rows_phase;",
                shared_load_root_line,
                store_scalar_line,
                "L_rows_phase:",
                f"    setp.ge.u32 %p1, %r1, {match.rows};",
                "    @%p1 bra L_cols_phase;",
                row_init,
                "    mov.u32 %r2, 0;",
                "L_rows_inner:",
                f"    setp.ge.u32 %p2, %r2, {match.cols};",
                "    @%p2 bra L_store_row;",
                f"    mad.lo.u32 {index_reg}, %r1, {match.cols}, %r2;",
                f"    mul.wide.u32 %rd6, {index_reg}, 4;",
                "    add.s64 %rd7, %rd1, %rd6;",
                load_line("%rd7"),
                f"    {reduce_instr} {scalar_acc_reg}, {scalar_acc_reg}, {value_reg};",
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_rows_inner;",
                "L_store_row:",
                "    mul.wide.u32 %rd6, %r1, 4;",
                "    add.s64 %rd7, %rd3, %rd6;",
                store_acc_line("%rd7"),
                "L_cols_phase:",
                f"    setp.ge.u32 %p1, %r1, {match.cols};",
                "    @%p1 bra L_done;",
                col_init,
                "    mov.u32 %r2, 0;",
                "L_cols_inner:",
                f"    setp.ge.u32 %p2, %r2, {match.rows};",
                "    @%p2 bra L_store_col;",
                f"    mad.lo.u32 {index_reg}, %r2, {match.cols}, %r1;",
                f"    mul.wide.u32 %rd6, {index_reg}, 4;",
                "    add.s64 %rd7, %rd1, %rd6;",
                load_line("%rd7"),
                f"    {reduce_instr} {scalar_acc_reg}, {scalar_acc_reg}, {value_reg};",
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_cols_inner;",
                "L_store_col:",
                "    mul.wide.u32 %rd6, %r1, 4;",
                "    add.s64 %rd7, %rd4, %rd6;",
                store_acc_line("%rd7"),
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_factory_bundle_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorFactoryBundle2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        total = match.rows * match.cols
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<2>;"
            zero_init = f"    mov.f32 %f1, {self._float_immediate(0.0)};"
            one_init = f"    mov.f32 %f1, {self._float_immediate(1.0)};"
            full_init = f"    mov.f32 %f1, {self._float_immediate(float(match.full_value))};"
            store_line = "    st.global.f32 [%rd5], %f1;"
        elif match.dtype == "i32":
            reg_decl = ""
            zero_init = "    mov.s32 %r3, 0;"
            one_init = "    mov.s32 %r3, 1;"
            full_init = f"    mov.s32 %r3, {int(match.full_value)};"
            store_line = "    st.global.s32 [%rd5], %r3;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        def section(label_prefix: str, dst_ptr: str, init_line: str) -> list[str]:
            return [
                init_line,
                "    mov.u32 %r1, 0;",
                f"L_{label_prefix}_loop:",
                f"    setp.ge.u32 %p1, %r1, {total};",
                f"    @%p1 bra L_{label_prefix}_done;",
                "    mul.wide.u32 %rd4, %r1, 4;",
                f"    add.s64 %rd5, {dst_ptr}, %rd4;",
                store_line,
                "    add.s32 %r1, %r1, 1;",
                f"    bra.uni L_{label_prefix}_loop;",
                f"L_{label_prefix}_done:",
            ]

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.dst_zero_name}_param,",
            f"    .param .u64 {match.dst_one_name}_param,",
            f"    .param .u64 {match.dst_full_name}_param",
            ")",
            "{",
            "    .reg .pred %p<2>;",
            "    .reg .b32 %r<4>;",
            "    .reg .b64 %rd<6>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.dst_zero_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_one_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_full_name}_param];",
        ]
        body = [line for line in body if line]
        body.extend(section("zero", "%rd1", zero_init))
        body.extend(section("one", "%rd2", one_init))
        body.extend(section("full", "%rd3", full_init))
        body.extend(["    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_scalar_broadcast(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _ScalarBroadcastMatch1D,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
        else:
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
        body = [f".visible .entry {ir.name}(", *params[:-1], params[-1], ")", "{"]
        body.extend(self._scalar_broadcast_declarations(match.dtype, mode=match.mode, scalar_mode=match.scalar_mode))
        body.extend(
            self._scalar_broadcast_index_lines(
                match.mode,
                match.extent,
                dtype=match.dtype,
                scalar_mode=match.scalar_mode,
                param_names=(f"{match.src_name}_param", f"{match.scalar_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._scalar_broadcast_lines(match.dtype, match.op, scalar_mode=match.scalar_mode))
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

    def _copy_declarations(self, dtype: str, *, mode: str) -> list[str]:
        lines = [
            "    .reg .pred %p<2>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<6>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<2>;")
        return lines

    def _binary_declarations(self, dtype: str, *, mode: str) -> list[str]:
        lines = [
            "    .reg .pred %p<2>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<8>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<4>;")
        return lines

    def _unary_declarations(self, *, mode: str) -> list[str]:
        del mode
        return [
            "    .reg .pred %p<2>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<6>;",
            "    .reg .f32 %f<3>;",
        ]

    def _reduce_declarations(self, dtype: str) -> list[str]:
        lines = [
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<5>;",
            "    .reg .b64 %rd<5>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<3>;")
        return lines

    def _scalar_broadcast_declarations(self, dtype: str, *, mode: str, scalar_mode: str) -> list[str]:
        del mode, scalar_mode
        lines = [
            "    .reg .pred %p<2>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<8>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<4>;")
        return lines

    def _index_lines(self, mode: str, extent: int, *, param_names: tuple[str, ...]) -> list[str]:
        if mode == "indexed":
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
                f"    setp.ge.u32 %p1, %r1, {extent};",
                "    @%p1 bra L_done;",
                "    mul.wide.u32 %rd3, %r1, 4;" if not rhs_param else "    mul.wide.u32 %rd4, %r1, 4;",
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

    def _scalar_broadcast_index_lines(
        self,
        mode: str,
        extent: int,
        *,
        dtype: str,
        scalar_mode: str,
        param_names: tuple[str, str, str],
    ) -> list[str]:
        if mode == "indexed":
            lines = [
                f"    ld.param.u64 %rd1, [{param_names[0]}];",
                "    mov.u32 %r1, %tid.x;",
                "    mov.u32 %r2, %ctaid.x;",
                "    mov.u32 %r3, %ntid.x;",
                "    mad.lo.u32 %r4, %r2, %r3, %r1;",
                f"    setp.ge.u32 %p1, %r4, {extent};",
                "    @%p1 bra L_done;",
                "    mul.wide.u32 %rd3, %r4, 4;",
                "    add.s64 %rd4, %rd1, %rd3;",
            ]
            if scalar_mode == "param":
                if dtype == "f32":
                    lines.append(f"    ld.param.f32 %f2, [{param_names[1]}];")
                else:
                    lines.append(f"    ld.param.s32 %r5, [{param_names[1]}];")
                lines.extend(
                    [
                        f"    ld.param.u64 %rd2, [{param_names[2]}];",
                        "    add.s64 %rd5, %rd2, %rd3;",
                    ]
                )
                return lines
            lines.extend(
                [
                    f"    ld.param.u64 %rd2, [{param_names[1]}];",
                    f"    ld.param.u64 %rd6, [{param_names[2]}];",
                    "    add.s64 %rd5, %rd6, %rd3;",
                ]
            )
            return lines
        lines = [
            f"    ld.param.u64 %rd1, [{param_names[0]}];",
            "    mov.u32 %r1, %tid.x;",
            f"    setp.ge.u32 %p1, %r1, {extent};",
            "    @%p1 bra L_done;",
            "    mul.wide.u32 %rd3, %r1, 4;",
            "    add.s64 %rd4, %rd1, %rd3;",
        ]
        if scalar_mode == "param":
            if dtype == "f32":
                lines.extend(
                    [
                        f"    ld.param.f32 %f2, [{param_names[1]}];",
                        f"    ld.param.u64 %rd2, [{param_names[2]}];",
                        "    add.s64 %rd5, %rd2, %rd3;",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"    ld.param.s32 %r3, [{param_names[1]}];",
                        f"    ld.param.u64 %rd2, [{param_names[2]}];",
                        "    add.s64 %rd5, %rd2, %rd3;",
                    ]
                )
            return lines
        lines.extend(
            [
                f"    ld.param.u64 %rd2, [{param_names[1]}];",
                f"    ld.param.u64 %rd6, [{param_names[2]}];",
                "    add.s64 %rd5, %rd6, %rd3;",
            ]
        )
        return lines

    def _copy_lines(self, dtype: str, *, lhs_ptr: str, dst_ptr: str) -> list[str]:
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

    def _binary_lines(self, dtype: str, op: str, *, lhs_ptr: str, rhs_ptr: str, dst_ptr: str) -> list[str]:
        if dtype == "f32":
            instr = {
                "add": "add.rn.f32",
                "sub": "sub.rn.f32",
                "mul": "mul.rn.f32",
                "div": "div.rn.f32",
            }[op]
            return [
                f"    ld.global.f32 %f1, [{lhs_ptr}];",
                f"    ld.global.f32 %f2, [{rhs_ptr}];",
                f"    {instr} %f3, %f1, %f2;",
                f"    st.global.f32 [{dst_ptr}], %f3;",
            ]
        if dtype == "i32":
            instr = {
                "add": "add.s32",
                "sub": "sub.s32",
                "mul": "mul.lo.s32",
                "div": "div.s32",
            }[op]
            return [
                f"    ld.global.s32 %r5, [{lhs_ptr}];",
                f"    ld.global.s32 %r2, [{rhs_ptr}];",
                f"    {instr} %r5, %r5, %r2;",
                f"    st.global.s32 [{dst_ptr}], %r5;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _unary_lines(self, op: str, *, src_ptr: str, dst_ptr: str) -> list[str]:
        instr = {
            "math_sqrt": "sqrt.rn.f32",
            "math_rsqrt": "rsqrt.approx.f32",
        }[op]
        return [
            f"    ld.global.f32 %f1, [{src_ptr}];",
            f"    {instr} %f2, %f1;",
            f"    st.global.f32 [{dst_ptr}], %f2;",
        ]

    def _reduce_lines(self, match: _ReduceMatch1D) -> list[str]:
        lines = [
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    setp.ne.u32 %p1, %r1, 0;",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
        ]
        lines.extend(self._reduce_init_lines(match.dtype, match.init_value))
        lines.extend(
            [
                "L_reduce:",
                f"    setp.ge.u32 %p2, %r2, {match.extent};",
                "    @%p2 bra L_store;",
                "    mul.wide.u32 %rd3, %r2, 4;",
                "    add.s64 %rd4, %rd1, %rd3;",
            ]
        )
        lines.extend(self._reduce_step_lines(match.dtype, match.op))
        lines.extend(
            [
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_reduce;",
                "L_store:",
            ]
        )
        lines.extend(self._reduce_store_lines(match.dtype))
        return lines

    def _scalar_broadcast_lines(self, dtype: str, op: str, *, scalar_mode: str) -> list[str]:
        if dtype == "f32":
            instr = {
                "add": "add.rn.f32",
                "sub": "sub.rn.f32",
                "mul": "mul.rn.f32",
                "div": "div.rn.f32",
            }[op]
            lines = ["    ld.global.f32 %f1, [%rd4];"]
            if scalar_mode == "tensor":
                lines.append("    ld.global.f32 %f2, [%rd2];")
            lines.extend(
                [
                    f"    {instr} %f3, %f1, %f2;",
                    "    st.global.f32 [%rd5], %f3;",
                ]
            )
            return lines
        if dtype == "i32":
            instr = {
                "add": "add.s32",
                "sub": "sub.s32",
                "mul": "mul.lo.s32",
                "div": "div.s32",
            }[op]
            lhs_reg = "%r2" if scalar_mode == "tensor" else "%r4"
            rhs_reg = "%r3" if scalar_mode == "tensor" else "%r5"
            lines = [f"    ld.global.s32 {lhs_reg}, [%rd4];"]
            if scalar_mode == "tensor":
                lines.append(f"    ld.global.s32 {rhs_reg}, [%rd2];")
            lines.extend(
                [
                    f"    {instr} {lhs_reg}, {lhs_reg}, {rhs_reg};",
                    f"    st.global.s32 [%rd5], {lhs_reg};",
                ]
            )
            return lines
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _reduce_init_lines(self, dtype: str, value: float | int) -> list[str]:
        if dtype == "f32":
            return [f"    mov.f32 %f1, {self._float_immediate(float(value))};"]
        if dtype == "i32":
            return [f"    mov.s32 %r3, {int(value)};"]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _reduce_step_lines(self, dtype: str, op: str) -> list[str]:
        if dtype == "f32":
            instr = {
                "reduce_add": "add.rn.f32",
                "reduce_mul": "mul.rn.f32",
                "reduce_max": "max.f32",
                "reduce_min": "min.f32",
            }[op]
            return [
                "    ld.global.f32 %f2, [%rd4];",
                f"    {instr} %f1, %f1, %f2;",
            ]
        if dtype == "i32":
            instr = {
                "reduce_add": "add.s32",
                "reduce_mul": "mul.lo.s32",
                "reduce_max": "max.s32",
                "reduce_min": "min.s32",
            }[op]
            return [
                "    ld.global.s32 %r4, [%rd4];",
                f"    {instr} %r3, %r3, %r4;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _reduce_store_lines(self, dtype: str) -> list[str]:
        if dtype == "f32":
            return ["    st.global.f32 [%rd2], %f1;"]
        if dtype == "i32":
            return ["    st.global.s32 [%rd2], %r3;"]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _is_parallel_reduce_launch(self, ir: PortableKernelIR, match: _ReduceMatch1D) -> bool:
        block_x, block_y, block_z = ir.launch.block
        return (
            match.op in {"reduce_add", "reduce_mul", "reduce_max", "reduce_min"}
            and ir.launch.grid == (1, 1, 1)
            and block_y == 1
            and block_z == 1
            and block_x > 1
            and block_x <= 1024
            and (block_x & (block_x - 1)) == 0
        )

    def _reduce_bundle_instr(self, dtype: str, op: str) -> str:
        if dtype == "f32":
            return {
                "reduce_add": "add.rn.f32",
                "reduce_mul": "mul.rn.f32",
                "reduce_max": "max.f32",
                "reduce_min": "min.f32",
            }[op]
        if dtype == "i32":
            return {
                "reduce_add": "add.s32",
                "reduce_mul": "mul.lo.s32",
                "reduce_max": "max.s32",
                "reduce_min": "min.s32",
            }[op]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _reduce_bundle_init_value(self, dtype: str, value: float | int) -> float | int:
        if dtype == "f32":
            return float(value)
        if dtype == "i32":
            return int(value)
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _float_immediate(self, value: float) -> str:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return f"0f{bits:08X}"

    def _scalar_param_type(self, dtype: str) -> str:
        table = {
            "f32": ".f32",
            "i32": ".s32",
        }
        try:
            return table[dtype]
        except KeyError as exc:
            raise CompilationError(f"unsupported PTX scalar dtype '{dtype}'") from exc
