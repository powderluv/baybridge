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
class _CompareMatch1D:
    mode: str
    lhs_name: str
    rhs_name: str
    dst_name: str
    dtype: str
    extent: int
    op: str


@dataclass(frozen=True)
class _SelectMatch1D:
    mode: str
    pred_name: str
    true_name: str
    false_name: str
    dst_name: str
    dtype: str
    extent: int


@dataclass(frozen=True)
class _ScalarSelectMatch1D:
    mode: str
    scalar_mode: str
    pred_name: str
    tensor_name: str
    scalar_name: str
    dst_name: str
    dtype: str
    extent: int
    tensor_branch: str


@dataclass(frozen=True)
class _ScalarCompareBroadcastMatch1D:
    mode: str
    scalar_mode: str
    src_name: str
    scalar_name: str
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
class _TensorBinary2DMatch:
    mode: str
    lhs_name: str
    rhs_name: str
    dst_name: str
    dtype: str
    lhs_rows: int
    lhs_cols: int
    rhs_rows: int
    rhs_cols: int
    dst_rows: int
    dst_cols: int
    op: str
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorCompare2DMatch:
    mode: str
    lhs_name: str
    rhs_name: str
    dst_name: str
    dtype: str
    lhs_rows: int
    lhs_cols: int
    rhs_rows: int
    rhs_cols: int
    dst_rows: int
    dst_cols: int
    op: str
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorScalarCompare2DMatch:
    scalar_mode: str
    src_name: str
    scalar_name: str
    dst_name: str
    dtype: str
    rows: int
    cols: int
    op: str
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorSelect2DMatch:
    mode: str
    pred_name: str
    true_name: str
    false_name: str
    dst_name: str
    dtype: str
    true_rows: int
    true_cols: int
    false_rows: int
    false_cols: int
    dst_rows: int
    dst_cols: int
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorScalarSelect2DMatch:
    scalar_mode: str
    tensor_branch: str
    pred_name: str
    tensor_name: str
    scalar_name: str
    dst_name: str
    dtype: str
    tensor_rows: int
    tensor_cols: int
    rows: int
    cols: int
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorCopy2DMatch:
    src_name: str
    dst_name: str
    dtype: str
    rows: int
    cols: int
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _CopyReduceMatch1D:
    mode: str
    src_name: str
    dst_name: str
    dtype: str
    extent: int
    reduction: str


@dataclass(frozen=True)
class _TensorCopyReduce2DMatch:
    src_name: str
    dst_name: str
    dtype: str
    rows: int
    cols: int
    reduction: str
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorScalarBroadcast2DMatch:
    scalar_mode: str
    src_name: str
    scalar_name: str
    dst_name: str
    dtype: str
    rows: int
    cols: int
    op: str
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _TensorUnary2DMatch:
    src_name: str
    dst_name: str
    dtype: str
    rows: int
    cols: int
    op: str
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _ReduceMatch1D:
    src_name: str
    dst_name: str
    dtype: str
    extent: int
    op: str
    init_value: float | int


@dataclass(frozen=True)
class _TensorReduce2DMatch:
    src_name: str
    dst_name: str
    dtype: str
    rows: int
    cols: int
    op: str
    init_value: float | int
    reduction_profile: tuple[int | None, int | None]
    parallel: bool = False
    block_x: int = 1


@dataclass(frozen=True)
class _RowColReduceBundle2DMatch:
    src_name: str
    dst_rows_name: str
    dst_cols_name: str
    dtype: str
    rows: int
    cols: int
    op: str
    rows_init: float | int
    cols_init: float | int
    parallel: bool = False
    block_x: int = 1


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
    parallel: bool = False
    block_x: int = 1


class PtxBridge:
    _SUPPORTED_DTYPES = {"f32": 4, "i32": 4}
    _SUPPORTED_ELEMENTWISE_DTYPES = {"f32": 4, "i32": 4, "i1": 1}
    _SUPPORTED_BINARY_OPS = {"add", "sub", "mul", "div", "max", "min", "bitand", "bitor", "bitxor"}
    _SUPPORTED_UNARY_OPS = {
        "neg",
        "abs",
        "math_atan",
        "math_erf",
        "math_sqrt",
        "math_rsqrt",
        "math_sin",
        "math_cos",
        "math_exp",
        "math_exp2",
        "math_log",
        "math_log2",
        "math_log10",
        "bitnot",
    }

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
        elif isinstance(match, _CompareMatch1D):
            text = self._render_compare(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _SelectMatch1D):
            text = self._render_select(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _ScalarSelectMatch1D):
            text = self._render_scalar_select(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _ScalarCompareBroadcastMatch1D):
            text = self._render_scalar_compare(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _UnaryMatch1D):
            text = self._render_unary(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _ScalarBroadcastMatch1D):
            text = self._render_scalar_broadcast(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorBinary2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_binary_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_binary_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorCompare2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_compare_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_compare_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorScalarCompare2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_scalar_compare_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_scalar_compare_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorSelect2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_select_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_select_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorScalarSelect2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_scalar_select_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_scalar_select_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorCopy2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_copy_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_copy_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _CopyReduceMatch1D):
            text = self._render_copy_reduce(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorCopyReduce2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_copy_reduce_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_copy_reduce_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorScalarBroadcast2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_scalar_broadcast_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_scalar_broadcast_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorUnary2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_unary_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_unary_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _ReduceMatch1D):
            text = self._render_reduce(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorReduce2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_reduce_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_reduce_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _RowColReduceBundle2DMatch):
            if match.parallel:
                text = self._render_parallel_row_col_reduce_bundle_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_row_col_reduce_bundle_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _ReduceBundle2DMatch):
            if match.parallel:
                text = self._render_parallel_reduce_bundle_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_reduce_bundle_2d(ir, target, match, backend_name=backend_name)
        elif isinstance(match, _TensorFactoryBundle2DMatch):
            if match.parallel:
                text = self._render_parallel_tensor_factory_bundle_2d(ir, target, match, backend_name=backend_name)
            else:
                text = self._render_tensor_factory_bundle_2d(ir, target, match, backend_name=backend_name)
        else:
            raise AssertionError(f"unsupported PTX match type: {type(match)!r}")
        return LoweredModule(
            backend_name=backend_name,
            entry_point=ir.name,
            dialect="ptx",
            text=text,
        )

    def _match(
        self,
        ir: PortableKernelIR,
    ) -> _CopyMatch1D | _BinaryMatch1D | _CompareMatch1D | _SelectMatch1D | _ScalarSelectMatch1D | _ScalarCompareBroadcastMatch1D | _UnaryMatch1D | _ScalarBroadcastMatch1D | _TensorBinary2DMatch | _TensorCompare2DMatch | _TensorScalarCompare2DMatch | _TensorSelect2DMatch | _TensorScalarSelect2DMatch | _TensorCopy2DMatch | _CopyReduceMatch1D | _TensorCopyReduce2DMatch | _TensorScalarBroadcast2DMatch | _TensorUnary2DMatch | _ReduceMatch1D | _TensorReduce2DMatch | _RowColReduceBundle2DMatch | _ReduceBundle2DMatch | _TensorFactoryBundle2DMatch:
        for matcher in (
            self._match_indexed_copy_1d,
            self._match_indexed_binary_1d,
            self._match_indexed_compare_1d,
            self._match_indexed_select_1d,
            self._match_indexed_scalar_select_1d,
            self._match_indexed_scalar_compare_1d,
            self._match_indexed_tensor_scalar_compare_1d,
            self._match_indexed_unary_1d,
            self._match_indexed_scalar_broadcast_1d,
            self._match_indexed_tensor_scalar_broadcast_1d,
            self._match_direct_copy_1d,
            self._match_direct_binary_1d,
            self._match_direct_compare_1d,
            self._match_direct_select_1d,
            self._match_direct_scalar_select_1d,
            self._match_direct_scalar_compare_1d,
            self._match_direct_tensor_scalar_compare_1d,
            self._match_direct_unary_1d,
            self._match_direct_scalar_broadcast_1d,
            self._match_direct_tensor_scalar_broadcast_1d,
            self._match_copy_reduce_1d,
            self._match_parallel_tensor_copy_2d,
            self._match_parallel_tensor_copy_reduce_2d,
            self._match_parallel_tensor_scalar_broadcast_2d,
            self._match_parallel_tensor_binary_2d,
            self._match_parallel_tensor_compare_2d,
            self._match_parallel_tensor_scalar_compare_2d,
            self._match_parallel_tensor_select_2d,
            self._match_parallel_tensor_scalar_select_2d,
            self._match_parallel_tensor_unary_2d,
            self._match_parallel_tensor_factory_bundle_2d,
            self._match_parallel_tensor_reduce_2d,
            self._match_tensor_scalar_compare_2d,
            self._match_tensor_compare_2d,
            self._match_tensor_scalar_select_2d,
            self._match_tensor_select_2d,
            self._match_broadcast_tensor_select_2d,
            self._match_tensor_copy_2d,
            self._match_tensor_copy_reduce_2d,
            self._match_tensor_scalar_broadcast_2d,
            self._match_tensor_binary_2d,
            self._match_broadcast_tensor_binary_2d,
            self._match_broadcast_tensor_compare_2d,
            self._match_tensor_unary_2d,
            self._match_reduce_1d_to_scalar,
            self._match_tensor_reduce_2d,
            self._match_parallel_row_col_reduce_bundle_2d,
            self._match_row_col_reduce_bundle_2d,
            self._match_parallel_reduce_bundle_2d,
            self._match_reduce_bundle_2d,
            self._match_tensor_factory_bundle_2d,
        ):
            match = matcher(ir)
            if match is not None:
                return match
        raise CompilationError(
            "ptx_ref currently supports only exact rank-1 dense copy, pointwise f32/i32 add/sub/mul/div/max/min plus "
            "rank-1 i32 bitand/bitor/bitxor and rank-1 i1 copy/bitand/bitor/bitxor, exact rank-1 f32/i32 neg/abs kernels, exact rank-1 i32/i1 bitnot kernels, exact rank-1 f32/i32 compare-to-i1 kernels, exact rank-1 f32/i32/i1 select kernels from i1 predicate tensors including exact scalar-branch forms, exact rank-1 scalar-broadcast compare-to-i1 kernels, exact f32 round/floor/ceil/trunc/sqrt/rsqrt/sin/cos/acos/asin/atan/exp/exp2/log/log2/log10/erf unary kernels, exact rank-1 scalar-broadcast kernels "
            "including rank-1 i32 bitand/bitor/bitxor, "
            "exact rank-1 scalar reductions to dst[0], exact rank-1 and 2D copy_reduce families, exact 2D f32/i32/i1 copy, compare-to-i1, select, scalar-select, scalar-broadcast, scalar-broadcast compare-to-i1, tensor-binary, and broadcast-binary families where i1 is limited to copy and bitand/bitor/bitxor tensor binaries, exact 2D f32/i32 max/min tensor-binary and scalar-broadcast families, exact 2D f32 round/floor/ceil/trunc/sqrt/rsqrt/sin/cos/acos/asin/atan/exp/exp2/log/log2/log10/erf unary families, exact 2D rowwise/columnwise reduction families, the exact 2D f32/i32 row+column reduction bundle families, the exact 2D f32/i32 reduction bundle families, "
            "their exact parallel 2D row-tiled variants, and the exact 2D f32/i32 tensor-factory bundle family, including its exact parallel row-tiled form; 2D scalar-broadcast accepts either scalar kernel params or rank-1 extent-1 tensors"
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
        if not self._supports_elementwise_dtype(src_spec.dtype):
            return None
        if not self._has_rank1_indexed_launch(ir, src_spec.shape[0]):
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
        if not self._supports_elementwise_dtype(lhs_spec.dtype):
            return None
        if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != dst_spec.dtype:
            return None
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_indexed_launch(ir, lhs_spec.shape[0]):
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
        if not self._supports_binary(lhs_spec.dtype, binary_op.op):
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

    def _match_indexed_compare_1d(self, ir: PortableKernelIR) -> _CompareMatch1D | None:
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
        if lhs_spec.dtype != rhs_spec.dtype or dst_spec.dtype != "i1":
            return None
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_indexed_launch(ir, lhs_spec.shape[0]):
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        lhs_load = ir.operations[11]
        rhs_load = ir.operations[12]
        compare_op = ir.operations[13]
        store_op = ir.operations[14]
        if lhs_load.op != "load" or lhs_load.inputs != (lhs_arg.name, idx_name):
            return None
        if rhs_load.op != "load" or rhs_load.inputs != (rhs_arg.name, idx_name):
            return None
        if lhs_load.attrs.get("rank") != 1 or rhs_load.attrs.get("rank") != 1:
            return None
        if not self._supports_compare(lhs_spec.dtype, compare_op.op):
            return None
        if compare_op.inputs != (lhs_load.outputs[0], rhs_load.outputs[0]):
            return None
        if compare_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if store_op.op != "store" or store_op.inputs != (compare_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _CompareMatch1D(
            mode="indexed",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            extent=lhs_spec.shape[0],
            op=compare_op.op,
        )

    def _match_indexed_select_1d(self, ir: PortableKernelIR) -> _SelectMatch1D | None:
        if len(ir.arguments) != 4 or len(ir.operations) != 16:
            return None
        pred_arg, true_arg, false_arg, dst_arg = ir.arguments
        pred_spec = self._require_rank1_tensor(pred_arg.spec)
        true_spec = self._require_rank1_tensor(true_arg.spec)
        false_spec = self._require_rank1_tensor(false_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if pred_spec is None or true_spec is None or false_spec is None or dst_spec is None:
            return None
        if pred_spec.dtype != "i1":
            return None
        if not self._supports_elementwise_dtype(true_spec.dtype):
            return None
        if true_spec.dtype != false_spec.dtype or true_spec.dtype != dst_spec.dtype:
            return None
        if pred_spec.shape != true_spec.shape or pred_spec.shape != false_spec.shape or pred_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_indexed_launch(ir, pred_spec.shape[0]):
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        pred_load = ir.operations[11]
        true_load = ir.operations[12]
        false_load = ir.operations[13]
        select_op = ir.operations[14]
        store_op = ir.operations[15]
        if pred_load.op != "load" or pred_load.inputs != (pred_arg.name, idx_name):
            return None
        if true_load.op != "load" or true_load.inputs != (true_arg.name, idx_name):
            return None
        if false_load.op != "load" or false_load.inputs != (false_arg.name, idx_name):
            return None
        if pred_load.attrs.get("rank") != 1 or true_load.attrs.get("rank") != 1 or false_load.attrs.get("rank") != 1:
            return None
        if pred_load.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if select_op.op != "select":
            return None
        if select_op.inputs != (pred_load.outputs[0], true_load.outputs[0], false_load.outputs[0]):
            return None
        if select_op.attrs.get("result", {}).get("dtype") != true_spec.dtype:
            return None
        if store_op.op != "store" or store_op.inputs != (select_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _SelectMatch1D(
            mode="indexed",
            pred_name=pred_arg.name,
            true_name=true_arg.name,
            false_name=false_arg.name,
            dst_name=dst_arg.name,
            dtype=true_spec.dtype,
            extent=true_spec.shape[0],
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
        if not self._supports_elementwise_dtype(src_spec.dtype):
            return None
        if not self._has_rank1_direct_launch(ir, src_spec.shape[0]):
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
        if not self._supports_elementwise_dtype(lhs_spec.dtype):
            return None
        if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != dst_spec.dtype:
            return None
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_direct_launch(ir, lhs_spec.shape[0]):
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
        if not self._supports_binary(lhs_spec.dtype, binary_op.op):
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

    def _match_direct_compare_1d(self, ir: PortableKernelIR) -> _CompareMatch1D | None:
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
        if lhs_spec.dtype != rhs_spec.dtype or dst_spec.dtype != "i1":
            return None
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_direct_launch(ir, lhs_spec.shape[0]):
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        lhs_load = ir.operations[3]
        rhs_load = ir.operations[4]
        compare_op = ir.operations[5]
        store_op = ir.operations[6]
        if lhs_load.op != "load" or lhs_load.inputs != (lhs_arg.name, idx_name):
            return None
        if rhs_load.op != "load" or rhs_load.inputs != (rhs_arg.name, idx_name):
            return None
        if lhs_load.attrs.get("rank") != 1 or rhs_load.attrs.get("rank") != 1:
            return None
        if not self._supports_compare(lhs_spec.dtype, compare_op.op):
            return None
        if compare_op.inputs != (lhs_load.outputs[0], rhs_load.outputs[0]):
            return None
        if compare_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if store_op.op != "store" or store_op.inputs != (compare_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _CompareMatch1D(
            mode="direct",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            extent=lhs_spec.shape[0],
            op=compare_op.op,
        )

    def _match_direct_select_1d(self, ir: PortableKernelIR) -> _SelectMatch1D | None:
        if len(ir.arguments) != 4 or len(ir.operations) != 8:
            return None
        pred_arg, true_arg, false_arg, dst_arg = ir.arguments
        pred_spec = self._require_rank1_tensor(pred_arg.spec)
        true_spec = self._require_rank1_tensor(true_arg.spec)
        false_spec = self._require_rank1_tensor(false_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if pred_spec is None or true_spec is None or false_spec is None or dst_spec is None:
            return None
        if pred_spec.dtype != "i1":
            return None
        if not self._supports_elementwise_dtype(true_spec.dtype):
            return None
        if true_spec.dtype != false_spec.dtype or true_spec.dtype != dst_spec.dtype:
            return None
        if pred_spec.shape != true_spec.shape or pred_spec.shape != false_spec.shape or pred_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_direct_launch(ir, pred_spec.shape[0]):
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        pred_load = ir.operations[3]
        true_load = ir.operations[4]
        false_load = ir.operations[5]
        select_op = ir.operations[6]
        store_op = ir.operations[7]
        if pred_load.op != "load" or pred_load.inputs != (pred_arg.name, idx_name):
            return None
        if true_load.op != "load" or true_load.inputs != (true_arg.name, idx_name):
            return None
        if false_load.op != "load" or false_load.inputs != (false_arg.name, idx_name):
            return None
        if pred_load.attrs.get("rank") != 1 or true_load.attrs.get("rank") != 1 or false_load.attrs.get("rank") != 1:
            return None
        if pred_load.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if select_op.op != "select":
            return None
        if select_op.inputs != (pred_load.outputs[0], true_load.outputs[0], false_load.outputs[0]):
            return None
        if select_op.attrs.get("result", {}).get("dtype") != true_spec.dtype:
            return None
        if store_op.op != "store" or store_op.inputs != (select_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _SelectMatch1D(
            mode="direct",
            pred_name=pred_arg.name,
            true_name=true_arg.name,
            false_name=false_arg.name,
            dst_name=dst_arg.name,
            dtype=true_spec.dtype,
            extent=true_spec.shape[0],
        )

    def _match_indexed_scalar_select_1d(self, ir: PortableKernelIR) -> _ScalarSelectMatch1D | None:
        if len(ir.arguments) != 4:
            return None
        pred_arg, tensor_arg, scalar_arg, dst_arg = ir.arguments
        pred_spec = self._require_rank1_tensor(pred_arg.spec)
        tensor_spec = self._require_rank1_tensor(tensor_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if pred_spec is None or tensor_spec is None or dst_spec is None:
            return None
        if pred_spec.dtype != "i1":
            return None
        if not self._supports_elementwise_dtype(tensor_spec.dtype):
            return None
        if tensor_spec.dtype != dst_spec.dtype or pred_spec.shape != tensor_spec.shape or pred_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_indexed_launch(ir, pred_spec.shape[0]):
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        pred_load = ir.operations[11]
        if pred_load.op != "load" or pred_load.inputs != (pred_arg.name, idx_name):
            return None
        if pred_load.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if pred_load.attrs.get("rank") != 1:
            return None

        scalar_mode: str
        scalar_value_name: str
        tensor_value_name: str
        select_op_index: int
        store_op_index: int
        if isinstance(scalar_arg.spec, ScalarSpec):
            if tensor_spec.dtype != scalar_arg.spec.dtype or len(ir.operations) != 15:
                return None
            tensor_load = ir.operations[12]
            if tensor_load.op != "load" or tensor_load.inputs != (tensor_arg.name, idx_name):
                return None
            if tensor_load.attrs.get("rank") != 1:
                return None
            scalar_mode = "param"
            scalar_value_name = scalar_arg.name
            tensor_value_name = tensor_load.outputs[0]
            select_op_index = 13
            store_op_index = 14
        else:
            scalar_spec = self._require_extent1_tensor(scalar_arg.spec)
            if scalar_spec is None or scalar_spec.dtype != tensor_spec.dtype or len(ir.operations) != 17:
                return None
            remaining_ops = ir.operations[12:15]
            zero_ops = [
                op
                for op in remaining_ops
                if op.op == "constant"
                and op.attrs.get("value") == 0
                and op.attrs.get("result", {}).get("dtype") == "index"
            ]
            if len(zero_ops) != 1:
                return None
            zero_op = zero_ops[0]
            scalar_loads = [op for op in remaining_ops if op.op == "load" and op.inputs == (scalar_arg.name, zero_op.outputs[0])]
            tensor_loads = [op for op in remaining_ops if op.op == "load" and op.inputs == (tensor_arg.name, idx_name)]
            if len(scalar_loads) != 1 or len(tensor_loads) != 1:
                return None
            scalar_load = scalar_loads[0]
            tensor_load = tensor_loads[0]
            if scalar_load.attrs.get("rank") != 1:
                return None
            if scalar_load.attrs.get("result", {}).get("dtype") != tensor_spec.dtype:
                return None
            if tensor_load.attrs.get("rank") != 1:
                return None
            scalar_mode = "tensor"
            scalar_value_name = scalar_load.outputs[0]
            tensor_value_name = tensor_load.outputs[0]
            select_op_index = 15
            store_op_index = 16

        select_op = ir.operations[select_op_index]
        store_op = ir.operations[store_op_index]
        if select_op.op != "select":
            return None
        tensor_branch: str
        if select_op.inputs == (pred_load.outputs[0], tensor_value_name, scalar_value_name):
            tensor_branch = "true"
        elif select_op.inputs == (pred_load.outputs[0], scalar_value_name, tensor_value_name):
            tensor_branch = "false"
        else:
            return None
        if select_op.attrs.get("result", {}).get("dtype") != tensor_spec.dtype:
            return None
        if store_op.op != "store" or store_op.inputs != (select_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarSelectMatch1D(
            mode="indexed",
            scalar_mode=scalar_mode,
            pred_name=pred_arg.name,
            tensor_name=tensor_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=tensor_spec.dtype,
            extent=tensor_spec.shape[0],
            tensor_branch=tensor_branch,
        )

    def _match_direct_scalar_select_1d(self, ir: PortableKernelIR) -> _ScalarSelectMatch1D | None:
        if len(ir.arguments) != 4:
            return None
        pred_arg, tensor_arg, scalar_arg, dst_arg = ir.arguments
        pred_spec = self._require_rank1_tensor(pred_arg.spec)
        tensor_spec = self._require_rank1_tensor(tensor_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if pred_spec is None or tensor_spec is None or dst_spec is None:
            return None
        if pred_spec.dtype != "i1":
            return None
        if not self._supports_elementwise_dtype(tensor_spec.dtype):
            return None
        if tensor_spec.dtype != dst_spec.dtype or pred_spec.shape != tensor_spec.shape or pred_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_direct_launch(ir, pred_spec.shape[0]):
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        pred_load = ir.operations[3]
        if pred_load.op != "load" or pred_load.inputs != (pred_arg.name, idx_name):
            return None
        if pred_load.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if pred_load.attrs.get("rank") != 1:
            return None

        scalar_mode: str
        scalar_value_name: str
        tensor_value_name: str
        select_op_index: int
        store_op_index: int
        if isinstance(scalar_arg.spec, ScalarSpec):
            if tensor_spec.dtype != scalar_arg.spec.dtype or len(ir.operations) != 7:
                return None
            tensor_load = ir.operations[4]
            if tensor_load.op != "load" or tensor_load.inputs != (tensor_arg.name, idx_name):
                return None
            if tensor_load.attrs.get("rank") != 1:
                return None
            scalar_mode = "param"
            scalar_value_name = scalar_arg.name
            tensor_value_name = tensor_load.outputs[0]
            select_op_index = 5
            store_op_index = 6
        else:
            scalar_spec = self._require_extent1_tensor(scalar_arg.spec)
            if scalar_spec is None or scalar_spec.dtype != tensor_spec.dtype or len(ir.operations) != 9:
                return None
            remaining_ops = ir.operations[4:7]
            zero_ops = [
                op
                for op in remaining_ops
                if op.op == "constant"
                and op.attrs.get("value") == 0
                and op.attrs.get("result", {}).get("dtype") == "index"
            ]
            if len(zero_ops) != 1:
                return None
            zero_op = zero_ops[0]
            scalar_loads = [op for op in remaining_ops if op.op == "load" and op.inputs == (scalar_arg.name, zero_op.outputs[0])]
            tensor_loads = [op for op in remaining_ops if op.op == "load" and op.inputs == (tensor_arg.name, idx_name)]
            if len(scalar_loads) != 1 or len(tensor_loads) != 1:
                return None
            scalar_load = scalar_loads[0]
            tensor_load = tensor_loads[0]
            if scalar_load.attrs.get("rank") != 1:
                return None
            if scalar_load.attrs.get("result", {}).get("dtype") != tensor_spec.dtype:
                return None
            if tensor_load.attrs.get("rank") != 1:
                return None
            scalar_mode = "tensor"
            scalar_value_name = scalar_load.outputs[0]
            tensor_value_name = tensor_load.outputs[0]
            select_op_index = 7
            store_op_index = 8

        select_op = ir.operations[select_op_index]
        store_op = ir.operations[store_op_index]
        if select_op.op != "select":
            return None
        tensor_branch: str
        if select_op.inputs == (pred_load.outputs[0], tensor_value_name, scalar_value_name):
            tensor_branch = "true"
        elif select_op.inputs == (pred_load.outputs[0], scalar_value_name, tensor_value_name):
            tensor_branch = "false"
        else:
            return None
        if select_op.attrs.get("result", {}).get("dtype") != tensor_spec.dtype:
            return None
        if store_op.op != "store" or store_op.inputs != (select_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarSelectMatch1D(
            mode="direct",
            scalar_mode=scalar_mode,
            pred_name=pred_arg.name,
            tensor_name=tensor_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=tensor_spec.dtype,
            extent=tensor_spec.shape[0],
            tensor_branch=tensor_branch,
        )

    def _match_direct_scalar_compare_1d(self, ir: PortableKernelIR) -> _ScalarCompareBroadcastMatch1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 6:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None or not isinstance(scalar_arg.spec, ScalarSpec):
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if src_spec.dtype != scalar_arg.spec.dtype or dst_spec.dtype != "i1":
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_direct_launch(ir, src_spec.shape[0]):
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        src_load = ir.operations[3]
        compare_op = ir.operations[4]
        store_op = ir.operations[5]
        if src_load.op != "load" or src_load.inputs != (src_arg.name, idx_name):
            return None
        if src_load.attrs.get("rank") != 1:
            return None
        if not self._supports_compare(src_spec.dtype, compare_op.op):
            return None
        if compare_op.inputs != (src_load.outputs[0], scalar_arg.name):
            return None
        if compare_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if store_op.op != "store" or store_op.inputs != (compare_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarCompareBroadcastMatch1D(
            mode="direct",
            scalar_mode="param",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=compare_op.op,
        )

    def _match_direct_tensor_scalar_compare_1d(self, ir: PortableKernelIR) -> _ScalarCompareBroadcastMatch1D | None:
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
        if src_spec.dtype != scalar_spec.dtype or dst_spec.dtype != "i1":
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_direct_launch(ir, src_spec.shape[0]):
            return None
        idx_name = self._match_direct_index_prefix(ir.operations[:3])
        if idx_name is None:
            return None
        src_load = ir.operations[3]
        cst_op = ir.operations[4]
        scalar_load = ir.operations[5]
        compare_op = ir.operations[6]
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
        if not self._supports_compare(src_spec.dtype, compare_op.op):
            return None
        if compare_op.inputs != (src_load.outputs[0], scalar_load.outputs[0]):
            return None
        if compare_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if store_op.op != "store" or store_op.inputs != (compare_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarCompareBroadcastMatch1D(
            mode="direct",
            scalar_mode="tensor",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=compare_op.op,
        )

    def _match_indexed_unary_1d(self, ir: PortableKernelIR) -> _UnaryMatch1D | None:
        if len(ir.arguments) != 2 or len(ir.operations) != 14:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_indexed_launch(ir, src_spec.shape[0]):
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
        if not self._supports_rank1_unary(src_spec.dtype, unary_op.op):
            return None
        if unary_op.inputs != (load_op.outputs[0],):
            return None
        if dst_spec.dtype != src_spec.dtype:
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
        if src_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_direct_launch(ir, src_spec.shape[0]):
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
        if not self._supports_rank1_unary(src_spec.dtype, unary_op.op):
            return None
        if unary_op.inputs != (load_op.outputs[0],):
            return None
        if dst_spec.dtype != src_spec.dtype:
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
        if not self._has_rank1_direct_launch(ir, src_spec.shape[0]):
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
        if not self._supports_binary(src_spec.dtype, binary_op.op):
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
        if not self._has_rank1_direct_launch(ir, src_spec.shape[0]):
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
        if not self._supports_binary(src_spec.dtype, binary_op.op):
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

    def _match_copy_reduce_1d(self, ir: PortableKernelIR) -> _CopyReduceMatch1D | None:
        if len(ir.arguments) != 2 or len(ir.operations) != 1:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype != dst_spec.dtype or src_spec.shape != dst_spec.shape:
            return None
        op = ir.operations[0]
        reduction = str(op.attrs.get("reduction", "")).lower()
        if op.op != "copy_reduce" or tuple(op.inputs) != (src_arg.name, dst_arg.name):
            return None
        if not self._supports_copy_reduce(src_spec.dtype, reduction):
            return None
        grid_x, grid_y, grid_z = ir.launch.grid
        block_x, block_y, block_z = ir.launch.block
        if grid_y != 1 or grid_z != 1 or block_y != 1 or block_z != 1:
            return None
        if grid_x == 1 and block_x == 1:
            mode = "serial"
        elif grid_x == 1 and block_x >= src_spec.shape[0]:
            mode = "direct"
        elif grid_x >= 1 and block_x >= 1 and grid_x * block_x >= src_spec.shape[0]:
            mode = "indexed"
        else:
            return None
        return _CopyReduceMatch1D(
            mode=mode,
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            reduction=reduction,
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
        if not self._supports_reduce(src_spec.dtype, reduce_op.op):
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

    def _match_tensor_reduce_2d(self, ir: PortableKernelIR) -> _TensorReduce2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 2 or len(ir.operations) != 5:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES or dst_spec.dtype != src_spec.dtype:
            return None
        ops = ir.operations
        op_name = ops[3].op
        if not self._supports_reduce(src_spec.dtype, op_name):
            return None
        if [op.op for op in ops] != ["make_tensor", "copy", "constant", op_name, "copy"]:
            return None
        if tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
            return None
        if tuple(ops[3].inputs) != (ops[0].outputs[0], ops[2].outputs[0]):
            return None
        reduction_profile = ops[3].attrs.get("reduction_profile")
        if reduction_profile == [None, 1]:
            reduction_profile_tuple = (None, 1)
            if dst_spec.shape != (src_spec.shape[0],):
                return None
        elif reduction_profile == [1, None]:
            reduction_profile_tuple = (1, None)
            if dst_spec.shape != (src_spec.shape[1],):
                return None
        else:
            return None
        if tuple(ops[4].inputs) != (ops[3].outputs[0], dst_arg.name):
            return None
        return _TensorReduce2DMatch(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            rows=src_spec.shape[0],
            cols=src_spec.shape[1],
            op=op_name,
            init_value=self._reduce_bundle_init_value(src_spec.dtype, ops[2].attrs["value"]),
            reduction_profile=reduction_profile_tuple,
        )

    def _match_parallel_tensor_reduce_2d(self, ir: PortableKernelIR) -> _TensorReduce2DMatch | None:
        grid_x, grid_y, grid_z = ir.launch.grid
        block_x, block_y, block_z = ir.launch.block
        if grid_x < 1 or grid_y != 1 or grid_z != 1 or block_y != 1 or block_z != 1:
            return None
        if block_x <= 1 or block_x > 1024 or (block_x & (block_x - 1)) != 0:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_reduce_2d(serial_ir)
        if match is None:
            return None
        output_extent = match.rows if match.reduction_profile == (None, 1) else match.cols
        if grid_x * block_x < output_extent:
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_row_col_reduce_bundle_2d(self, ir: PortableKernelIR) -> _RowColReduceBundle2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3 or len(ir.operations) != 8:
            return None
        src_arg, dst_rows_arg, dst_cols_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_rows_spec = self._require_rank1_tensor(dst_rows_arg.spec)
        dst_cols_spec = self._require_rank1_tensor(dst_cols_arg.spec)
        if src_spec is None or dst_rows_spec is None or dst_cols_spec is None:
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        rows, cols = src_spec.shape
        if dst_rows_spec.dtype != src_spec.dtype or dst_rows_spec.shape != (rows,):
            return None
        if dst_cols_spec.dtype != src_spec.dtype or dst_cols_spec.shape != (cols,):
            return None
        ops = ir.operations
        op_name = ops[3].op
        if not self._supports_reduce(src_spec.dtype, op_name):
            return None
        if [op.op for op in ops] != [
            "make_tensor",
            "copy",
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
        if ops[3].attrs.get("reduction_profile") != [None, 1]:
            return None
        if tuple(ops[4].inputs) != (ops[3].outputs[0], dst_rows_arg.name):
            return None
        if tuple(ops[6].inputs) != (ops[0].outputs[0], ops[5].outputs[0]):
            return None
        if ops[6].attrs.get("reduction_profile") != [1, None]:
            return None
        if tuple(ops[7].inputs) != (ops[6].outputs[0], dst_cols_arg.name):
            return None
        return _RowColReduceBundle2DMatch(
            src_name=src_arg.name,
            dst_rows_name=dst_rows_arg.name,
            dst_cols_name=dst_cols_arg.name,
            dtype=src_spec.dtype,
            rows=rows,
            cols=cols,
            op=op_name,
            rows_init=self._reduce_bundle_init_value(src_spec.dtype, ops[2].attrs["value"]),
            cols_init=self._reduce_bundle_init_value(src_spec.dtype, ops[5].attrs["value"]),
        )

    def _match_parallel_row_col_reduce_bundle_2d(self, ir: PortableKernelIR) -> _RowColReduceBundle2DMatch | None:
        grid_x, grid_y, grid_z = ir.launch.grid
        block_x, block_y, block_z = ir.launch.block
        if grid_x < 1 or grid_y != 1 or grid_z != 1 or block_y != 1 or block_z != 1:
            return None
        if block_x <= 1 or block_x > 1024 or (block_x & (block_x - 1)) != 0:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_row_col_reduce_bundle_2d(serial_ir)
        if match is None or grid_x * block_x < max(match.rows, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

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
        if not self._supports_reduce(src_spec.dtype, op_name):
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
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_reduce_bundle_2d(serial_ir)
        if match is None or not self._supports_reduce(match.dtype, match.op):
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

    def _match_parallel_tensor_binary_2d(self, ir: PortableKernelIR) -> _TensorBinary2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_binary_2d(serial_ir)
        if match is None:
            match = self._match_broadcast_tensor_binary_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.dst_cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_compare_2d(self, ir: PortableKernelIR) -> _TensorCompare2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_compare_2d(serial_ir)
        if match is None:
            match = self._match_broadcast_tensor_compare_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.dst_cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_scalar_compare_2d(self, ir: PortableKernelIR) -> _TensorScalarCompare2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_scalar_compare_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_select_2d(self, ir: PortableKernelIR) -> _TensorSelect2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_select_2d(serial_ir)
        if match is None:
            match = self._match_broadcast_tensor_select_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.dst_cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_scalar_select_2d(self, ir: PortableKernelIR) -> _TensorScalarSelect2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_scalar_select_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_copy_2d(self, ir: PortableKernelIR) -> _TensorCopy2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_copy_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_copy_reduce_2d(self, ir: PortableKernelIR) -> _TensorCopyReduce2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_copy_reduce_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_scalar_broadcast_2d(self, ir: PortableKernelIR) -> _TensorScalarBroadcast2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_scalar_broadcast_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_unary_2d(self, ir: PortableKernelIR) -> _TensorUnary2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_unary_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_parallel_tensor_factory_bundle_2d(self, ir: PortableKernelIR) -> _TensorFactoryBundle2DMatch | None:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return None
        serial_ir = replace(ir, launch=replace(ir.launch, grid=(1, 1, 1), block=(1, 1, 1)))
        match = self._match_tensor_factory_bundle_2d(serial_ir)
        if match is None or not self._supports_parallel_tensor_2d_launch(ir, match.cols):
            return None
        return replace(match, parallel=True, block_x=block_x)

    def _match_tensor_binary_2d(self, ir: PortableKernelIR) -> _TensorBinary2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3 or len(ir.operations) != 6:
            return None
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        lhs_spec = self._require_rank2_tensor(lhs_arg.spec)
        rhs_spec = self._require_rank2_tensor(rhs_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if lhs_spec is None or rhs_spec is None or dst_spec is None:
            return None
        if not self._supports_elementwise_dtype(lhs_spec.dtype):
            return None
        if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != dst_spec.dtype:
            return None
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            return None
        ops = ir.operations
        if [op.op for op in ops[:4]] != ["make_tensor", "copy", "make_tensor", "copy"]:
            return None
        if ops[5].op != "copy":
            return None
        if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (lhs_arg.name, ops[0].outputs[0]):
            return None
        if ops[2].op != "make_tensor" or tuple(ops[3].inputs) != (rhs_arg.name, ops[2].outputs[0]):
            return None
        tensor_op = ops[4]
        if tensor_op.op not in {
            "tensor_add",
            "tensor_sub",
            "tensor_mul",
            "tensor_div",
            "tensor_max",
            "tensor_min",
            "tensor_bitand",
            "tensor_bitor",
            "tensor_bitxor",
            "math_atan2",
        }:
            return None
        if tuple(tensor_op.inputs) != (ops[0].outputs[0], ops[2].outputs[0]):
            return None
        if tuple(ops[5].inputs) != (tensor_op.outputs[0], dst_arg.name):
            return None
        return _TensorBinary2DMatch(
            mode="dense",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            lhs_rows=lhs_spec.shape[0],
            lhs_cols=lhs_spec.shape[1],
            rhs_rows=rhs_spec.shape[0],
            rhs_cols=rhs_spec.shape[1],
            dst_rows=dst_spec.shape[0],
            dst_cols=dst_spec.shape[1],
            op=tensor_op.op.removeprefix("tensor_"),
        )

    def _match_tensor_compare_2d(self, ir: PortableKernelIR) -> _TensorCompare2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3 or len(ir.operations) != 6:
            return None
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        lhs_spec = self._require_rank2_tensor(lhs_arg.spec)
        rhs_spec = self._require_rank2_tensor(rhs_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if lhs_spec is None or rhs_spec is None or dst_spec is None:
            return None
        if lhs_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if lhs_spec.dtype != rhs_spec.dtype or dst_spec.dtype != "i1":
            return None
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            return None
        ops = ir.operations
        if [op.op for op in ops[:4]] != ["make_tensor", "copy", "make_tensor", "copy"]:
            return None
        if ops[5].op != "copy":
            return None
        if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (lhs_arg.name, ops[0].outputs[0]):
            return None
        if ops[2].op != "make_tensor" or tuple(ops[3].inputs) != (rhs_arg.name, ops[2].outputs[0]):
            return None
        tensor_op = ops[4]
        if not self._supports_compare(lhs_spec.dtype, tensor_op.op):
            return None
        if tuple(tensor_op.inputs) != (ops[0].outputs[0], ops[2].outputs[0]):
            return None
        if tensor_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if tuple(ops[5].inputs) != (tensor_op.outputs[0], dst_arg.name):
            return None
        return _TensorCompare2DMatch(
            mode="dense",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            lhs_rows=lhs_spec.shape[0],
            lhs_cols=lhs_spec.shape[1],
            rhs_rows=rhs_spec.shape[0],
            rhs_cols=rhs_spec.shape[1],
            dst_rows=dst_spec.shape[0],
            dst_cols=dst_spec.shape[1],
            op=tensor_op.op,
        )

    def _match_tensor_copy_2d(self, ir: PortableKernelIR) -> _TensorCopy2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 2:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if not self._supports_elementwise_dtype(src_spec.dtype):
            return None
        if src_spec.dtype != dst_spec.dtype or src_spec.shape != dst_spec.shape:
            return None
        ops = ir.operations
        if len(ops) == 1:
            if ops[0].op != "copy" or tuple(ops[0].inputs) != (src_arg.name, dst_arg.name):
                return None
        elif len(ops) == 3:
            if ops[0].op != "make_tensor":
                return None
            if tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
                return None
            if ops[2].op != "copy" or tuple(ops[2].inputs) != (ops[0].outputs[0], dst_arg.name):
                return None
        else:
            return None
        return _TensorCopy2DMatch(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            rows=src_spec.shape[0],
            cols=src_spec.shape[1],
        )

    def _match_tensor_copy_reduce_2d(self, ir: PortableKernelIR) -> _TensorCopyReduce2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 2 or len(ir.operations) != 1:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype != dst_spec.dtype or src_spec.shape != dst_spec.shape:
            return None
        op = ir.operations[0]
        reduction = str(op.attrs.get("reduction", "")).lower()
        if op.op != "copy_reduce" or tuple(op.inputs) != (src_arg.name, dst_arg.name):
            return None
        if not self._supports_copy_reduce(src_spec.dtype, reduction):
            return None
        return _TensorCopyReduce2DMatch(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            rows=src_spec.shape[0],
            cols=src_spec.shape[1],
            reduction=reduction,
        )

    def _match_tensor_scalar_broadcast_2d(self, ir: PortableKernelIR) -> _TensorScalarBroadcast2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if src_spec.dtype != dst_spec.dtype:
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        ops = ir.operations
        scalar_mode: str
        scalar_value_name: str
        tensor_op_index: int
        copy_index: int
        if isinstance(scalar_arg.spec, ScalarSpec):
            if len(ops) != 4:
                return None
            if src_spec.dtype != scalar_arg.spec.dtype:
                return None
            if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
                return None
            scalar_mode = "param"
            scalar_value_name = scalar_arg.name
            tensor_op_index = 2
            copy_index = 3
        else:
            scalar_spec = self._require_extent1_tensor(scalar_arg.spec)
            if scalar_spec is None or src_spec.dtype != scalar_spec.dtype:
                return None
            if len(ops) != 6:
                return None
            if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
                return None
            if (
                ops[2].op != "constant"
                or ops[2].attrs.get("value") != 0
                or ops[2].attrs.get("result", {}).get("dtype") != "index"
            ):
                return None
            if ops[3].op != "load" or tuple(ops[3].inputs) != (scalar_arg.name, ops[2].outputs[0]):
                return None
            if ops[3].attrs.get("rank") != 1:
                return None
            if ops[3].attrs.get("result", {}).get("dtype") != src_spec.dtype:
                return None
            scalar_mode = "tensor"
            scalar_value_name = ops[3].outputs[0]
            tensor_op_index = 4
            copy_index = 5
        tensor_op = ops[tensor_op_index]
        if tensor_op.op not in {
            "tensor_add",
            "tensor_sub",
            "tensor_mul",
            "tensor_div",
            "tensor_max",
            "tensor_min",
            "tensor_bitand",
            "tensor_bitor",
            "tensor_bitxor",
            "math_atan2",
        }:
            return None
        if tuple(tensor_op.inputs) != (ops[0].outputs[0], scalar_value_name):
            return None
        if ops[copy_index].op != "copy" or tuple(ops[copy_index].inputs) != (tensor_op.outputs[0], dst_arg.name):
            return None
        return _TensorScalarBroadcast2DMatch(
            scalar_mode=scalar_mode,
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            rows=src_spec.shape[0],
            cols=src_spec.shape[1],
            op=tensor_op.op.removeprefix("tensor_"),
        )

    def _match_tensor_scalar_compare_2d(self, ir: PortableKernelIR) -> _TensorScalarCompare2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if dst_spec.dtype != "i1" or src_spec.shape != dst_spec.shape:
            return None
        ops = ir.operations
        scalar_mode: str
        scalar_value_name: str
        tensor_op_index: int
        copy_index: int
        if isinstance(scalar_arg.spec, ScalarSpec):
            if len(ops) != 4:
                return None
            if src_spec.dtype != scalar_arg.spec.dtype:
                return None
            if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
                return None
            scalar_mode = "param"
            scalar_value_name = scalar_arg.name
            tensor_op_index = 2
            copy_index = 3
        else:
            scalar_spec = self._require_extent1_tensor(scalar_arg.spec)
            if scalar_spec is None or src_spec.dtype != scalar_spec.dtype:
                return None
            if len(ops) != 6:
                return None
            if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
                return None
            if (
                ops[2].op != "constant"
                or ops[2].attrs.get("value") != 0
                or ops[2].attrs.get("result", {}).get("dtype") != "index"
            ):
                return None
            if ops[3].op != "load" or tuple(ops[3].inputs) != (scalar_arg.name, ops[2].outputs[0]):
                return None
            if ops[3].attrs.get("rank") != 1:
                return None
            if ops[3].attrs.get("result", {}).get("dtype") != src_spec.dtype:
                return None
            scalar_mode = "tensor"
            scalar_value_name = ops[3].outputs[0]
            tensor_op_index = 4
            copy_index = 5
        tensor_op = ops[tensor_op_index]
        if not self._supports_compare(src_spec.dtype, tensor_op.op):
            return None
        if tuple(tensor_op.inputs) != (ops[0].outputs[0], scalar_value_name):
            return None
        if tensor_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if ops[copy_index].op != "copy" or tuple(ops[copy_index].inputs) != (tensor_op.outputs[0], dst_arg.name):
            return None
        return _TensorScalarCompare2DMatch(
            scalar_mode=scalar_mode,
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            rows=src_spec.shape[0],
            cols=src_spec.shape[1],
            op=tensor_op.op,
        )

    def _match_tensor_select_2d(self, ir: PortableKernelIR) -> _TensorSelect2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 4 or len(ir.operations) != 8:
            return None
        pred_arg, true_arg, false_arg, dst_arg = ir.arguments
        pred_spec = self._require_rank2_tensor(pred_arg.spec)
        true_spec = self._require_rank2_tensor(true_arg.spec)
        false_spec = self._require_rank2_tensor(false_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if pred_spec is None or true_spec is None or false_spec is None or dst_spec is None:
            return None
        if pred_spec.dtype != "i1":
            return None
        if not self._supports_elementwise_dtype(true_spec.dtype):
            return None
        if true_spec.dtype != false_spec.dtype or true_spec.dtype != dst_spec.dtype:
            return None
        if pred_spec.shape != dst_spec.shape or true_spec.shape != dst_spec.shape or false_spec.shape != dst_spec.shape:
            return None
        ops = ir.operations
        if [op.op for op in ops[:6]] != ["make_tensor", "copy", "make_tensor", "copy", "make_tensor", "copy"]:
            return None
        if tuple(ops[1].inputs) != (pred_arg.name, ops[0].outputs[0]):
            return None
        if tuple(ops[3].inputs) != (true_arg.name, ops[2].outputs[0]):
            return None
        if tuple(ops[5].inputs) != (false_arg.name, ops[4].outputs[0]):
            return None
        select_op = ops[6]
        if select_op.op != "tensor_select":
            return None
        if tuple(select_op.inputs) != (ops[0].outputs[0], ops[2].outputs[0], ops[4].outputs[0]):
            return None
        if select_op.attrs.get("result", {}).get("dtype") != true_spec.dtype:
            return None
        if ops[7].op != "copy" or tuple(ops[7].inputs) != (select_op.outputs[0], dst_arg.name):
            return None
        return _TensorSelect2DMatch(
            mode="dense",
            pred_name=pred_arg.name,
            true_name=true_arg.name,
            false_name=false_arg.name,
            dst_name=dst_arg.name,
            dtype=true_spec.dtype,
            true_rows=true_spec.shape[0],
            true_cols=true_spec.shape[1],
            false_rows=false_spec.shape[0],
            false_cols=false_spec.shape[1],
            dst_rows=dst_spec.shape[0],
            dst_cols=dst_spec.shape[1],
        )

    def _match_tensor_scalar_select_2d(self, ir: PortableKernelIR) -> _TensorScalarSelect2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 4:
            return None
        pred_arg, tensor_arg, scalar_arg, dst_arg = ir.arguments
        pred_spec = self._require_rank2_tensor(pred_arg.spec)
        tensor_spec = self._require_rank2_tensor(tensor_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if pred_spec is None or tensor_spec is None or dst_spec is None:
            return None
        if pred_spec.dtype != "i1":
            return None
        if not self._supports_elementwise_dtype(tensor_spec.dtype):
            return None
        if tensor_spec.dtype != dst_spec.dtype or pred_spec.shape != dst_spec.shape:
            return None
        if not self._is_broadcast_compatible(tensor_spec.shape, dst_spec.shape):
            return None
        ops = ir.operations
        if len(ops) < 6:
            return None
        if ops[0].op != "make_tensor" or ops[1].op != "copy":
            return None
        if tuple(ops[1].inputs) != (pred_arg.name, ops[0].outputs[0]):
            return None
        if ops[-1].op != "copy":
            return None
        middle_ops = ops[2:-2]
        tensor_value_name: str | None = None
        scalar_mode: str
        scalar_value_name: str | None
        if isinstance(scalar_arg.spec, ScalarSpec):
            if tensor_spec.dtype != scalar_arg.spec.dtype:
                return None
            scalar_mode = "param"
            scalar_value_name = scalar_arg.name
        else:
            scalar_spec = self._require_extent1_tensor(scalar_arg.spec)
            if scalar_spec is None or scalar_spec.dtype != tensor_spec.dtype:
                return None
            scalar_mode = "tensor"
            scalar_value_name = None
        cursor = 0
        while cursor < len(middle_ops):
            op = middle_ops[cursor]
            if op.op == "make_tensor":
                if tensor_value_name is not None or cursor + 1 >= len(middle_ops):
                    return None
                copy_op = middle_ops[cursor + 1]
                if copy_op.op != "copy" or tuple(copy_op.inputs) != (tensor_arg.name, op.outputs[0]):
                    return None
                tensor_value_name = op.outputs[0]
                cursor += 2
                continue
            if op.op == "broadcast_to":
                if tensor_value_name is None or op.inputs != (tensor_value_name,):
                    return None
                if op.attrs.get("target_shape") != list(dst_spec.shape):
                    return None
                tensor_value_name = op.outputs[0]
                cursor += 1
                continue
            if op.op == "constant":
                if scalar_mode != "tensor" or scalar_value_name is not None or cursor + 1 >= len(middle_ops):
                    return None
                if op.attrs.get("value") != 0 or op.attrs.get("result", {}).get("dtype") != "index":
                    return None
                load_op = middle_ops[cursor + 1]
                if load_op.op != "load" or tuple(load_op.inputs) != (scalar_arg.name, op.outputs[0]):
                    return None
                if load_op.attrs.get("rank") != 1:
                    return None
                if load_op.attrs.get("result", {}).get("dtype") != tensor_spec.dtype:
                    return None
                scalar_value_name = load_op.outputs[0]
                cursor += 2
                continue
            return None
        if tensor_value_name is None or scalar_value_name is None:
            return None
        select_op = ops[-2]
        if select_op.op != "tensor_select":
            return None
        if select_op.inputs == (ops[0].outputs[0], tensor_value_name, scalar_value_name):
            tensor_branch = "true"
        elif select_op.inputs == (ops[0].outputs[0], scalar_value_name, tensor_value_name):
            tensor_branch = "false"
        else:
            return None
        if select_op.attrs.get("result", {}).get("dtype") != tensor_spec.dtype:
            return None
        if tuple(ops[-1].inputs) != (select_op.outputs[0], dst_arg.name):
            return None
        return _TensorScalarSelect2DMatch(
            scalar_mode=scalar_mode,
            tensor_branch=tensor_branch,
            pred_name=pred_arg.name,
            tensor_name=tensor_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=tensor_spec.dtype,
            tensor_rows=tensor_spec.shape[0],
            tensor_cols=tensor_spec.shape[1],
            rows=dst_spec.shape[0],
            cols=dst_spec.shape[1],
        )

    def _match_broadcast_tensor_binary_2d(self, ir: PortableKernelIR) -> _TensorBinary2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3 or len(ir.operations) != 8:
            return None
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        lhs_spec = self._require_rank2_tensor(lhs_arg.spec)
        rhs_spec = self._require_rank2_tensor(rhs_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if lhs_spec is None or rhs_spec is None or dst_spec is None:
            return None
        if not self._supports_elementwise_dtype(lhs_spec.dtype):
            return None
        if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != dst_spec.dtype:
            return None
        if not self._is_broadcast_compatible(lhs_spec.shape, dst_spec.shape):
            return None
        if not self._is_broadcast_compatible(rhs_spec.shape, dst_spec.shape):
            return None
        ops = ir.operations
        if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (lhs_arg.name, ops[0].outputs[0]):
            return None
        if ops[2].op != "make_tensor" or tuple(ops[3].inputs) != (rhs_arg.name, ops[2].outputs[0]):
            return None
        if ops[4].op != "broadcast_to" or ops[4].inputs != (ops[0].outputs[0],):
            return None
        if ops[5].op != "broadcast_to" or ops[5].inputs != (ops[2].outputs[0],):
            return None
        if ops[4].attrs.get("target_shape") != list(dst_spec.shape):
            return None
        if ops[5].attrs.get("target_shape") != list(dst_spec.shape):
            return None
        tensor_op = ops[6]
        if tensor_op.op not in {
            "tensor_add",
            "tensor_sub",
            "tensor_mul",
            "tensor_div",
            "tensor_max",
            "tensor_min",
            "tensor_bitand",
            "tensor_bitor",
            "tensor_bitxor",
            "math_atan2",
        }:
            return None
        if tuple(tensor_op.inputs) != (ops[4].outputs[0], ops[5].outputs[0]):
            return None
        if tuple(ops[7].inputs) != (tensor_op.outputs[0], dst_arg.name):
            return None
        return _TensorBinary2DMatch(
            mode="broadcast",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            lhs_rows=lhs_spec.shape[0],
            lhs_cols=lhs_spec.shape[1],
            rhs_rows=rhs_spec.shape[0],
            rhs_cols=rhs_spec.shape[1],
            dst_rows=dst_spec.shape[0],
            dst_cols=dst_spec.shape[1],
            op=tensor_op.op.removeprefix("tensor_"),
        )

    def _match_broadcast_tensor_compare_2d(self, ir: PortableKernelIR) -> _TensorCompare2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 3 or len(ir.operations) != 8:
            return None
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        lhs_spec = self._require_rank2_tensor(lhs_arg.spec)
        rhs_spec = self._require_rank2_tensor(rhs_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if lhs_spec is None or rhs_spec is None or dst_spec is None:
            return None
        if lhs_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if lhs_spec.dtype != rhs_spec.dtype or dst_spec.dtype != "i1":
            return None
        if not self._is_broadcast_compatible(lhs_spec.shape, dst_spec.shape):
            return None
        if not self._is_broadcast_compatible(rhs_spec.shape, dst_spec.shape):
            return None
        ops = ir.operations
        if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (lhs_arg.name, ops[0].outputs[0]):
            return None
        if ops[2].op != "make_tensor" or tuple(ops[3].inputs) != (rhs_arg.name, ops[2].outputs[0]):
            return None
        if ops[4].op != "broadcast_to" or ops[4].inputs != (ops[0].outputs[0],):
            return None
        if ops[5].op != "broadcast_to" or ops[5].inputs != (ops[2].outputs[0],):
            return None
        if ops[4].attrs.get("target_shape") != list(dst_spec.shape):
            return None
        if ops[5].attrs.get("target_shape") != list(dst_spec.shape):
            return None
        tensor_op = ops[6]
        if not self._supports_compare(lhs_spec.dtype, tensor_op.op):
            return None
        if tuple(tensor_op.inputs) != (ops[4].outputs[0], ops[5].outputs[0]):
            return None
        if tensor_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if tuple(ops[7].inputs) != (tensor_op.outputs[0], dst_arg.name):
            return None
        return _TensorCompare2DMatch(
            mode="broadcast",
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            lhs_rows=lhs_spec.shape[0],
            lhs_cols=lhs_spec.shape[1],
            rhs_rows=rhs_spec.shape[0],
            rhs_cols=rhs_spec.shape[1],
            dst_rows=dst_spec.shape[0],
            dst_cols=dst_spec.shape[1],
            op=tensor_op.op,
        )

    def _match_broadcast_tensor_select_2d(self, ir: PortableKernelIR) -> _TensorSelect2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 4:
            return None
        pred_arg, true_arg, false_arg, dst_arg = ir.arguments
        pred_spec = self._require_rank2_tensor(pred_arg.spec)
        true_spec = self._require_rank2_tensor(true_arg.spec)
        false_spec = self._require_rank2_tensor(false_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if pred_spec is None or true_spec is None or false_spec is None or dst_spec is None:
            return None
        if pred_spec.dtype != "i1":
            return None
        if not self._supports_elementwise_dtype(true_spec.dtype):
            return None
        if true_spec.dtype != false_spec.dtype or true_spec.dtype != dst_spec.dtype:
            return None
        if pred_spec.shape != dst_spec.shape:
            return None
        if not self._is_broadcast_compatible(true_spec.shape, dst_spec.shape):
            return None
        if not self._is_broadcast_compatible(false_spec.shape, dst_spec.shape):
            return None
        if true_spec.shape == dst_spec.shape and false_spec.shape == dst_spec.shape:
            return None
        ops = ir.operations
        if len(ops) < 9 or len(ops) > 10:
            return None
        if [op.op for op in ops[:6]] != ["make_tensor", "copy", "make_tensor", "copy", "make_tensor", "copy"]:
            return None
        if tuple(ops[1].inputs) != (pred_arg.name, ops[0].outputs[0]):
            return None
        if tuple(ops[3].inputs) != (true_arg.name, ops[2].outputs[0]):
            return None
        if tuple(ops[5].inputs) != (false_arg.name, ops[4].outputs[0]):
            return None
        true_value_name = ops[2].outputs[0]
        false_value_name = ops[4].outputs[0]
        cursor = 6
        if true_spec.shape != dst_spec.shape:
            if ops[cursor].op != "broadcast_to" or ops[cursor].inputs != (true_value_name,):
                return None
            if ops[cursor].attrs.get("target_shape") != list(dst_spec.shape):
                return None
            true_value_name = ops[cursor].outputs[0]
            cursor += 1
        if false_spec.shape != dst_spec.shape:
            if cursor >= len(ops) or ops[cursor].op != "broadcast_to" or ops[cursor].inputs != (false_value_name,):
                return None
            if ops[cursor].attrs.get("target_shape") != list(dst_spec.shape):
                return None
            false_value_name = ops[cursor].outputs[0]
            cursor += 1
        if len(ops) != cursor + 2:
            return None
        select_op = ops[cursor]
        if select_op.op != "tensor_select":
            return None
        if tuple(select_op.inputs) != (ops[0].outputs[0], true_value_name, false_value_name):
            return None
        if select_op.attrs.get("result", {}).get("dtype") != true_spec.dtype:
            return None
        if ops[cursor + 1].op != "copy" or tuple(ops[cursor + 1].inputs) != (select_op.outputs[0], dst_arg.name):
            return None
        return _TensorSelect2DMatch(
            mode="broadcast",
            pred_name=pred_arg.name,
            true_name=true_arg.name,
            false_name=false_arg.name,
            dst_name=dst_arg.name,
            dtype=true_spec.dtype,
            true_rows=true_spec.shape[0],
            true_cols=true_spec.shape[1],
            false_rows=false_spec.shape[0],
            false_cols=false_spec.shape[1],
            dst_rows=dst_spec.shape[0],
            dst_cols=dst_spec.shape[1],
        )

    def _match_tensor_unary_2d(self, ir: PortableKernelIR) -> _TensorUnary2DMatch | None:
        if ir.launch.grid != (1, 1, 1) or ir.launch.block != (1, 1, 1):
            return None
        if len(ir.arguments) != 2 or len(ir.operations) != 4:
            return None
        src_arg, dst_arg = ir.arguments
        src_spec = self._require_rank2_tensor(src_arg.spec)
        dst_spec = self._require_rank2_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None:
            return None
        if src_spec.dtype != dst_spec.dtype:
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        ops = ir.operations
        if ops[0].op != "make_tensor" or tuple(ops[1].inputs) != (src_arg.name, ops[0].outputs[0]):
            return None
        unary_op = ops[2]
        normalized_op = self._normalize_tensor_unary_op(unary_op.op)
        if normalized_op is None or not self._supports_rank1_unary(src_spec.dtype, normalized_op):
            return None
        if tuple(unary_op.inputs) != (ops[0].outputs[0],):
            return None
        if ops[3].op != "copy" or tuple(ops[3].inputs) != (unary_op.outputs[0], dst_arg.name):
            return None
        return _TensorUnary2DMatch(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            rows=src_spec.shape[0],
            cols=src_spec.shape[1],
            op=normalized_op,
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
        if not self._has_rank1_indexed_launch(ir, src_spec.shape[0]):
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
        if not self._supports_binary(src_spec.dtype, binary_op.op):
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
        if not self._has_rank1_indexed_launch(ir, src_spec.shape[0]):
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
        if not self._supports_binary(src_spec.dtype, binary_op.op):
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

    def _match_indexed_scalar_compare_1d(self, ir: PortableKernelIR) -> _ScalarCompareBroadcastMatch1D | None:
        if len(ir.arguments) != 3 or len(ir.operations) != 14:
            return None
        src_arg, scalar_arg, dst_arg = ir.arguments
        src_spec = self._require_rank1_tensor(src_arg.spec)
        dst_spec = self._require_rank1_tensor(dst_arg.spec)
        if src_spec is None or dst_spec is None or not isinstance(scalar_arg.spec, ScalarSpec):
            return None
        if src_spec.dtype not in self._SUPPORTED_DTYPES:
            return None
        if src_spec.dtype != scalar_arg.spec.dtype or dst_spec.dtype != "i1":
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_indexed_launch(ir, src_spec.shape[0]):
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        src_load = ir.operations[11]
        compare_op = ir.operations[12]
        store_op = ir.operations[13]
        if src_load.op != "load" or src_load.inputs != (src_arg.name, idx_name):
            return None
        if src_load.attrs.get("rank") != 1:
            return None
        if not self._supports_compare(src_spec.dtype, compare_op.op):
            return None
        if compare_op.inputs != (src_load.outputs[0], scalar_arg.name):
            return None
        if compare_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if store_op.op != "store" or store_op.inputs != (compare_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarCompareBroadcastMatch1D(
            mode="indexed",
            scalar_mode="param",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=compare_op.op,
        )

    def _match_indexed_tensor_scalar_compare_1d(self, ir: PortableKernelIR) -> _ScalarCompareBroadcastMatch1D | None:
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
        if src_spec.dtype != scalar_spec.dtype or dst_spec.dtype != "i1":
            return None
        if src_spec.shape != dst_spec.shape:
            return None
        if not self._has_rank1_indexed_launch(ir, src_spec.shape[0]):
            return None
        idx_name = self._match_canonical_index_prefix(ir.operations[:11])
        if idx_name is None:
            return None
        src_load = ir.operations[11]
        cst_op = ir.operations[12]
        scalar_load = ir.operations[13]
        compare_op = ir.operations[14]
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
        if not self._supports_compare(src_spec.dtype, compare_op.op):
            return None
        if compare_op.inputs != (src_load.outputs[0], scalar_load.outputs[0]):
            return None
        if compare_op.attrs.get("result", {}).get("dtype") != "i1":
            return None
        if store_op.op != "store" or store_op.inputs != (compare_op.outputs[0], dst_arg.name, idx_name):
            return None
        if store_op.attrs.get("rank") != 1:
            return None
        return _ScalarCompareBroadcastMatch1D(
            mode="indexed",
            scalar_mode="tensor",
            src_name=src_arg.name,
            scalar_name=scalar_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            extent=src_spec.shape[0],
            op=compare_op.op,
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

    def _has_rank1_indexed_launch(self, ir: PortableKernelIR, extent: int) -> bool:
        grid_x, grid_y, grid_z = ir.launch.grid
        block_x, block_y, block_z = ir.launch.block
        return (
            grid_x >= 1
            and grid_y == 1
            and grid_z == 1
            and block_x >= 1
            and block_y == 1
            and block_z == 1
            and grid_x * block_x >= extent
        )

    def _has_rank1_direct_launch(self, ir: PortableKernelIR, extent: int) -> bool:
        grid_x, grid_y, grid_z = ir.launch.grid
        block_x, block_y, block_z = ir.launch.block
        return (
            grid_x == 1
            and grid_y == 1
            and grid_z == 1
            and block_x >= extent
            and block_y == 1
            and block_z == 1
        )

    def _parallel_tensor_2d_block_x(self, ir: PortableKernelIR) -> int | None:
        block_x, block_y, block_z = ir.launch.block
        grid_x, grid_y, grid_z = ir.launch.grid
        if grid_x < 1 or grid_y < 1 or grid_z != 1:
            return None
        if block_y != 1 or block_z != 1:
            return None
        if block_x <= 1 or block_x > 1024 or (block_x & (block_x - 1)) != 0:
            return None
        return block_x

    def _supports_parallel_tensor_2d_launch(self, ir: PortableKernelIR, cols: int) -> bool:
        block_x = self._parallel_tensor_2d_block_x(ir)
        if block_x is None:
            return False
        return ir.launch.grid[0] * block_x >= cols

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
        body.extend(
            self._index_lines(
                match.mode,
                match.extent,
                dtype=match.dtype,
                param_names=(f"{match.src_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._copy_lines(match.dtype, lhs_ptr="%rd4", dst_ptr="%rd5"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_copy_reduce(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _CopyReduceMatch1D,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        if match.mode != "serial":
            body = [
                f".visible .entry {ir.name}(",
                f"    .param .u64 {match.src_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
                ")",
                "{",
            ]
            body.extend(self._copy_reduce_declarations(match.dtype, mode=match.mode))
            body.extend(
                self._index_lines(
                    match.mode,
                    match.extent,
                    dtype=match.dtype,
                    param_names=(f"{match.src_name}_param", f"{match.dst_name}_param"),
                )
            )
            body.extend(self._copy_reduce_lines(match.dtype, match.reduction, src_ptr="%rd4", dst_ptr="%rd5"))
            body.extend(["L_done:", "    ret;", "}"])
            return ptx + "\n".join(body) + "\n"
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<5>;",
            "    .reg .b64 %rd<6>;",
            "    .reg .f32 %f<4>;" if match.dtype == "f32" else "",
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    setp.ne.u32 %p1, %r1, 0;",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_copy_reduce:",
            f"    setp.ge.u32 %p2, %r2, {match.extent};",
            "    @%p2 bra L_done;",
            "    mul.wide.u32 %rd3, %r2, 4;",
            "    add.s64 %rd4, %rd1, %rd3;",
            "    add.s64 %rd5, %rd2, %rd3;",
            *self._copy_reduce_lines(match.dtype, match.reduction, src_ptr="%rd4", dst_ptr="%rd5"),
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_copy_reduce;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
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
        body.extend(self._binary_declarations(match.dtype, mode=match.mode, op=match.op))
        body.extend(
            self._index_lines(
                match.mode,
                match.extent,
                dtype=match.dtype,
                param_names=(f"{match.lhs_name}_param", f"{match.rhs_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._binary_lines(match.dtype, match.op, lhs_ptr="%rd5", rhs_ptr="%rd6", dst_ptr="%rd7"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_compare(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _CompareMatch1D,
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
        body.extend(self._compare_declarations(match.dtype))
        body.extend(
            self._compare_index_lines(
                match.mode,
                match.extent,
                param_names=(f"{match.lhs_name}_param", f"{match.rhs_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._compare_lines(match.dtype, match.op, lhs_ptr="%rd5", rhs_ptr="%rd6", dst_ptr="%rd7"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_select(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _SelectMatch1D,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.pred_name}_param,",
            f"    .param .u64 {match.true_name}_param,",
            f"    .param .u64 {match.false_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
        ]
        body.extend(self._select_declarations(match.dtype))
        body.extend(
            self._select_index_lines(
                match.mode,
                match.extent,
                dtype=match.dtype,
                param_names=(
                    f"{match.pred_name}_param",
                    f"{match.true_name}_param",
                    f"{match.false_name}_param",
                    f"{match.dst_name}_param",
                ),
            )
        )
        body.extend(self._select_lines(match.dtype, pred_ptr="%rd6", true_ptr="%rd8", false_ptr="%rd9", dst_ptr="%rd10"))
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_scalar_select(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _ScalarSelectMatch1D,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.pred_name}_param,",
                f"    .param .u64 {match.tensor_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
        else:
            params = [
                f"    .param .u64 {match.pred_name}_param,",
                f"    .param .u64 {match.tensor_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
        body = [f".visible .entry {ir.name}(", *params[:-1], params[-1], ")", "{"]
        body.extend(self._scalar_select_declarations(match.dtype, scalar_mode=match.scalar_mode))
        body.extend(
            self._scalar_select_index_lines(
                match.mode,
                match.extent,
                dtype=match.dtype,
                scalar_mode=match.scalar_mode,
                param_names=(
                    f"{match.pred_name}_param",
                    f"{match.tensor_name}_param",
                    f"{match.scalar_name}_param",
                    f"{match.dst_name}_param",
                ),
            )
        )
        body.extend(
            self._scalar_select_lines(
                match.dtype,
                scalar_mode=match.scalar_mode,
                tensor_branch=match.tensor_branch,
            )
        )
        body.extend(["L_done:", "    ret;", "}"])
        return ptx + "\n".join(body) + "\n"

    def _render_scalar_compare(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _ScalarCompareBroadcastMatch1D,
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
        body.extend(self._compare_declarations(match.dtype))
        body.extend(
            self._scalar_compare_index_lines(
                match.mode,
                match.extent,
                dtype=match.dtype,
                scalar_mode=match.scalar_mode,
                param_names=(f"{match.src_name}_param", f"{match.scalar_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._scalar_compare_lines(match.dtype, match.op, scalar_mode=match.scalar_mode))
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
        body.extend(self._unary_declarations(match.dtype, mode=match.mode, op=match.op))
        body.extend(
            self._index_lines(
                match.mode,
                match.extent,
                dtype=match.dtype,
                param_names=(f"{match.src_name}_param", f"{match.dst_name}_param"),
            )
        )
        body.extend(self._unary_lines(match.dtype, match.op, src_ptr="%rd4", dst_ptr="%rd5"))
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

    def _render_tensor_reduce_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorReduce2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        reduce_instr = self._reduce_bundle_instr(match.dtype, match.op)
        reduce_rows = match.reduction_profile == (None, 1)
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
            index_reg = "%r5"
            reg_decl = ""
            load_line = lambda ptr: f"    ld.global.s32 {value_reg}, [{ptr}];"
            store_line = lambda ptr: f"    st.global.s32 [{ptr}], {acc_reg};"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<5>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.rows if reduce_rows else match.cols};",
            "    @%p1 bra L_done;",
            *self._reduce_init_lines(match.dtype, match.init_value),
            "    mov.u32 %r2, 0;",
            "L_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.cols if reduce_rows else match.rows};",
            "    @%p2 bra L_store;",
            (
                f"    mad.lo.u32 {index_reg}, %r1, {match.cols}, %r2;"
                if reduce_rows
                else f"    mad.lo.u32 {index_reg}, %r2, {match.cols}, %r1;"
            ),
            f"    mul.wide.u32 %rd3, {index_reg}, 4;",
            "    add.s64 %rd4, %rd1, %rd3;",
            load_line("%rd4"),
            f"    {reduce_instr} {acc_reg}, {acc_reg}, {value_reg};",
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_inner;",
            "L_store:",
            "    mul.wide.u32 %rd3, %r1, 4;",
            "    add.s64 %rd4, %rd2, %rd3;",
            store_line("%rd4"),
            "    add.s32 %r1, %r1, 1;",
            "    bra.uni L_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_reduce_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorReduce2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        reduce_instr = self._reduce_bundle_instr(match.dtype, match.op)
        reduce_rows = match.reduction_profile == (None, 1)
        output_extent = match.rows if reduce_rows else match.cols
        reduction_extent = match.cols if reduce_rows else match.rows
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
            index_reg = "%r5"
            reg_decl = ""
            load_line = lambda ptr: f"    ld.global.s32 {value_reg}, [{ptr}];"
            store_line = lambda ptr: f"    st.global.s32 [{ptr}], {acc_reg};"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")
        output_index_lines = (
            [
                "    mov.u32 %r1, %tid.x;",
            ]
            if ir.launch.grid == (1, 1, 1)
            else [
                "    mov.u32 %r0, %tid.x;",
                "    mov.u32 %r1, %ctaid.x;",
                f"    mad.lo.u32 %r1, %r1, {match.block_x}, %r0;",
            ]
        )
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<5>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            *output_index_lines,
            f"    setp.ge.u32 %p1, %r1, {output_extent};",
            "    @%p1 bra L_done;",
            *self._reduce_init_lines(match.dtype, match.init_value),
            "    mov.u32 %r2, 0;",
            "L_inner:",
            f"    setp.ge.u32 %p2, %r2, {reduction_extent};",
            "    @%p2 bra L_store;",
            (
                f"    mad.lo.u32 {index_reg}, %r1, {match.cols}, %r2;"
                if reduce_rows
                else f"    mad.lo.u32 {index_reg}, %r2, {match.cols}, %r1;"
            ),
            f"    mul.wide.u32 %rd3, {index_reg}, 4;",
            "    add.s64 %rd4, %rd1, %rd3;",
            load_line("%rd4"),
            f"    {reduce_instr} {acc_reg}, {acc_reg}, {value_reg};",
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_inner;",
            "L_store:",
            "    mul.wide.u32 %rd3, %r1, 4;",
            "    add.s64 %rd4, %rd2, %rd3;",
            store_line("%rd4"),
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_row_col_reduce_bundle_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _RowColReduceBundle2DMatch,
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
            f"    .param .u64 {match.dst_rows_name}_param,",
            f"    .param .u64 {match.dst_cols_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<7>;",
            "    .reg .b64 %rd<7>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_rows_name}_param];",
            f"    ld.param.u64 %rd4, [{match.dst_cols_name}_param];",
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
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_row_col_reduce_bundle_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _RowColReduceBundle2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        reduce_instr = self._reduce_bundle_instr(match.dtype, match.op)
        if match.dtype == "f32":
            acc_reg = "%f1"
            value_reg = "%f2"
            index_reg = "%r6"
            reg_decl = "    .reg .f32 %f<3>;"
            load_line = lambda ptr: f"    ld.global.f32 {value_reg}, [{ptr}];"
            store_line = lambda ptr: f"    st.global.f32 [{ptr}], {acc_reg};"
        elif match.dtype == "i32":
            acc_reg = "%r3"
            value_reg = "%r4"
            index_reg = "%r5"
            reg_decl = ""
            load_line = lambda ptr: f"    ld.global.s32 {value_reg}, [{ptr}];"
            store_line = lambda ptr: f"    st.global.s32 [{ptr}], {acc_reg};"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")
        output_index_lines = (
            [
                "    mov.u32 %r1, %tid.x;",
            ]
            if ir.launch.grid == (1, 1, 1)
            else [
                "    mov.u32 %r0, %tid.x;",
                "    mov.u32 %r1, %ctaid.x;",
                f"    mad.lo.u32 %r1, %r1, {match.block_x}, %r0;",
            ]
        )
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_rows_name}_param,",
            f"    .param .u64 {match.dst_cols_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<7>;",
            "    .reg .b64 %rd<7>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_rows_name}_param];",
            f"    ld.param.u64 %rd4, [{match.dst_cols_name}_param];",
            *output_index_lines,
            f"    setp.ge.u32 %p1, %r1, {match.rows};",
            "    @%p1 bra L_cols_phase;",
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
            "L_cols_phase:",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            *self._reduce_init_lines(match.dtype, match.cols_init),
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_store_col;",
            f"    mad.lo.u32 {index_reg}, %r2, {match.cols}, %r1;",
            f"    mul.wide.u32 %rd5, {index_reg}, 4;",
            "    add.s64 %rd6, %rd1, %rd5;",
            load_line("%rd6"),
            f"    {reduce_instr} {acc_reg}, {acc_reg}, {value_reg};",
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_cols_inner;",
            "L_store_col:",
            "    mul.wide.u32 %rd5, %r1, 4;",
            "    add.s64 %rd6, %rd4, %rd5;",
            store_line("%rd6"),
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
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

    def _render_parallel_tensor_factory_bundle_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorFactoryBundle2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<4>;"
            zero_init = f"    mov.f32 %f1, {self._float_immediate(0.0)};"
            one_init = f"    mov.f32 %f2, {self._float_immediate(1.0)};"
            full_init = f"    mov.f32 %f3, {self._float_immediate(float(match.full_value))};"
            zero_store = "    st.global.f32 [%rd5], %f1;"
            one_store = "    st.global.f32 [%rd6], %f2;"
            full_store = "    st.global.f32 [%rd7], %f3;"
        elif match.dtype == "i32":
            reg_decl = ""
            zero_init = "    mov.s32 %r8, 0;"
            one_init = "    mov.s32 %r9, 1;"
            full_init = f"    mov.s32 %r10, {int(match.full_value)};"
            zero_store = "    st.global.s32 [%rd5], %r8;"
            one_store = "    st.global.s32 [%rd6], %r9;"
            full_store = "    st.global.s32 [%rd7], %r10;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.dst_zero_name}_param,",
            f"    .param .u64 {match.dst_one_name}_param,",
            f"    .param .u64 {match.dst_full_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<11>;",
            "    .reg .b64 %rd<8>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.dst_zero_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_one_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_full_name}_param];",
            zero_init,
            one_init,
            full_init,
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r7, %r2, {match.cols}, %r1;",
            "    mul.wide.u32 %rd4, %r7, 4;",
            "    add.s64 %rd5, %rd1, %rd4;",
            "    add.s64 %rd6, %rd2, %rd4;",
            "    add.s64 %rd7, %rd3, %rd4;",
            zero_store,
            one_store,
            full_store,
            f"    add.s32 %r2, %r2, {grid_y};",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_binary_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorBinary2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        element_size = self._element_size(match.dtype)
        lhs_offset_line = (
            "    cvt.u64.u32 %rd4, %r8;"
            if element_size == 1
            else f"    mul.wide.u32 %rd4, %r8, {element_size};"
        )
        rhs_offset_line = (
            "    cvt.u64.u32 %rd4, %r9;"
            if element_size == 1
            else f"    mul.wide.u32 %rd4, %r9, {element_size};"
        )
        dst_offset_line = (
            "    cvt.u64.u32 %rd4, %r10;"
            if element_size == 1
            else f"    mul.wide.u32 %rd4, %r10, {element_size};"
        )
        if match.dtype == "f32":
            reg_decl = self._binary_f32_reg_decl(match.op)
            load_lhs = "    ld.global.f32 %f1, [%rd5];"
            load_rhs = "    ld.global.f32 %f2, [%rd6];"
            store_dst = "    st.global.f32 [%rd7], %f3;"
            if match.op == "math_atan2":
                op_lines = self._atan2_f32_core_lines(y_reg="%f1", x_reg="%f2", result_reg="%f3")
            else:
                instr = self._binary_instr(match.dtype, match.op)
                op_lines = [f"    {instr} %f3, %f1, %f2;"]
        elif match.dtype == "i32":
            reg_decl = ""
            load_lhs = "    ld.global.s32 %r11, [%rd5];"
            load_rhs = "    ld.global.s32 %r12, [%rd6];"
            store_dst = "    st.global.s32 [%rd7], %r11;"
            instr = self._binary_instr(match.dtype, match.op)
            op_lines = [f"    {instr} %r11, %r11, %r12;"]
        elif match.dtype == "i1":
            reg_decl = ""
            load_lhs = "    ld.global.u8 %r11, [%rd5];"
            load_rhs = "    ld.global.u8 %r12, [%rd6];"
            store_dst = "    st.global.u8 [%rd7], %r11;"
            instr = self._binary_instr(match.dtype, match.op)
            op_lines = [f"    {instr} %r11, %r11, %r12;"]
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.lhs_name}_param,",
            f"    .param .u64 {match.rhs_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<4>;" if match.op == "math_atan2" else "    .reg .pred %p<3>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<8>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.lhs_name}_param];",
            f"    ld.param.u64 %rd2, [{match.rhs_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.dst_cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.dst_rows};",
            "    @%p2 bra L_done;",
            "    mov.u32 %r3, %r2;",
            "    mov.u32 %r4, %r1;",
        ]
        if match.lhs_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.lhs_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r8, %r3, {match.lhs_cols}, %r4;",
                "    mov.u32 %r5, %r2;",
                "    mov.u32 %r6, %r1;",
            ]
        )
        if match.rhs_rows == 1:
            body.append("    mov.u32 %r5, 0;")
        if match.rhs_cols == 1:
            body.append("    mov.u32 %r6, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r9, %r5, {match.rhs_cols}, %r6;",
                f"    mad.lo.u32 %r10, %r2, {match.dst_cols}, %r1;",
                lhs_offset_line,
                "    add.s64 %rd5, %rd1, %rd4;",
                rhs_offset_line,
                "    add.s64 %rd6, %rd2, %rd4;",
                dst_offset_line,
                "    add.s64 %rd7, %rd3, %rd4;",
                load_lhs,
                load_rhs,
                *op_lines,
                store_dst,
                f"    add.s32 %r2, %r2, {grid_y};",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_copy_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorCopy2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        element_size = self._element_size(match.dtype)
        offset_line = (
            "    cvt.u64.u32 %rd3, %r7;"
            if element_size == 1
            else f"    mul.wide.u32 %rd3, %r7, {element_size};"
        )
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<2>;"
            load_src = "    ld.global.f32 %f1, [%rd4];"
            store_dst = "    st.global.f32 [%rd5], %f1;"
        elif match.dtype == "i32":
            reg_decl = ""
            load_src = "    ld.global.s32 %r7, [%rd4];"
            store_dst = "    st.global.s32 [%rd5], %r7;"
        elif match.dtype == "i1":
            reg_decl = ""
            load_src = "    ld.global.u8 %r7, [%rd4];"
            store_dst = "    st.global.u8 [%rd5], %r7;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<6>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r7, %r2, {match.cols}, %r1;",
            offset_line,
            "    add.s64 %rd4, %rd1, %rd3;",
            "    add.s64 %rd5, %rd2, %rd3;",
            load_src,
            store_dst,
            f"    add.s32 %r2, %r2, {grid_y};",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_copy_reduce_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorCopyReduce2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<6>;",
            "    .reg .f32 %f<4>;" if match.dtype == "f32" else "",
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r7, %r2, {match.cols}, %r1;",
            "    mul.wide.u32 %rd3, %r7, 4;",
            "    add.s64 %rd4, %rd1, %rd3;",
            "    add.s64 %rd5, %rd2, %rd3;",
            *self._copy_reduce_lines(match.dtype, match.reduction, src_ptr="%rd4", dst_ptr="%rd5"),
            f"    add.s32 %r2, %r2, {grid_y};",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_unary_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorUnary2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        element_size = self._element_size(match.dtype)
        offset_line = (
            f"    mul.wide.u32 %rd3, %r8, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd3, %r8;"
        )
        reg_decl = self._unary_f32_reg_decl(match.op) if match.dtype == "f32" else ""
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<4>;" if match.op in {"math_acos", "math_asin"} else "    .reg .pred %p<3>;",
            "    .reg .b32 %r<9>;",
            "    .reg .b64 %rd<6>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r8, %r2, {match.cols}, %r1;",
            offset_line,
            "    add.s64 %rd4, %rd1, %rd3;",
            "    add.s64 %rd5, %rd2, %rd3;",
            *self._unary_lines(match.dtype, match.op, src_ptr="%rd4", dst_ptr="%rd5"),
            f"    add.s32 %r2, %r2, {grid_y};",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_scalar_broadcast_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorScalarBroadcast2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        if match.dtype == "f32":
            reg_decl = self._binary_f32_reg_decl(match.op)
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.f32 %f2, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.f32 %f2, [%rd2];"
            if match.op == "math_atan2":
                op_lines = self._atan2_f32_core_lines(y_reg="%f1", x_reg="%f2", result_reg="%f3")
            else:
                instr = self._binary_instr(match.dtype, match.op)
                op_lines = [f"    {instr} %f3, %f1, %f2;"]
        elif match.dtype == "i32":
            reg_decl = ""
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.s32 %r3, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.s32 %r3, [%rd2];"
            instr = self._binary_instr(match.dtype, match.op)
            op_lines = [f"    {instr} %r7, %r7, %r3;"]
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<6>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd2, [{match.dst_name}_param];"
            src_ptr_reg = "%rd1"
            dst_ptr_reg = "%rd2"
            offset_reg = "%rd3"
            src_addr_reg = "%rd4"
            dst_addr_reg = "%rd5"
        else:
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<7>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd3, [{match.dst_name}_param];"
            src_ptr_reg = "%rd1"
            dst_ptr_reg = "%rd3"
            offset_reg = "%rd4"
            src_addr_reg = "%rd5"
            dst_addr_reg = "%rd6"

        body = [
            f".visible .entry {ir.name}(",
            *params,
            ")",
            "{",
            "    .reg .pred %p<4>;" if match.op == "math_atan2" else "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            b64_decl,
            reg_decl,
            src_ptr_load,
            *([f"    ld.param.u64 %rd2, [{match.scalar_name}_param];"] if match.scalar_mode == "tensor" else []),
            scalar_load,
            dst_ptr_load,
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r7, %r2, {match.cols}, %r1;",
            f"    mul.wide.u32 {offset_reg}, %r7, 4;",
            f"    add.s64 {src_addr_reg}, {src_ptr_reg}, {offset_reg};",
            f"    add.s64 {dst_addr_reg}, {dst_ptr_reg}, {offset_reg};",
            f"    ld.global.{self._scalar_param_type(match.dtype).removeprefix('.')} {'%f1' if match.dtype == 'f32' else '%r7'}, [{src_addr_reg}];",
            *op_lines,
            f"    st.global.{self._scalar_param_type(match.dtype).removeprefix('.')} [{dst_addr_reg}], {'%f3' if match.dtype == 'f32' else '%r7'};",
            f"    add.s32 %r2, %r2, {grid_y};",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_binary_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorBinary2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        element_size = self._element_size(match.dtype)
        lhs_offset_line = (
            "    cvt.u64.u32 %rd4, %r8;"
            if element_size == 1
            else f"    mul.wide.u32 %rd4, %r8, {element_size};"
        )
        rhs_offset_line = (
            "    cvt.u64.u32 %rd4, %r9;"
            if element_size == 1
            else f"    mul.wide.u32 %rd4, %r9, {element_size};"
        )
        dst_offset_line = (
            "    cvt.u64.u32 %rd4, %r10;"
            if element_size == 1
            else f"    mul.wide.u32 %rd4, %r10, {element_size};"
        )
        if match.dtype == "f32":
            reg_decl = self._binary_f32_reg_decl(match.op)
            load_lhs = "    ld.global.f32 %f1, [%rd5];"
            load_rhs = "    ld.global.f32 %f2, [%rd6];"
            store_dst = "    st.global.f32 [%rd7], %f3;"
            if match.op == "math_atan2":
                op_lines = self._atan2_f32_core_lines(y_reg="%f1", x_reg="%f2", result_reg="%f3")
            else:
                instr = self._binary_instr(match.dtype, match.op)
                op_lines = [f"    {instr} %f3, %f1, %f2;"]
        elif match.dtype == "i32":
            reg_decl = ""
            load_lhs = "    ld.global.s32 %r11, [%rd5];"
            load_rhs = "    ld.global.s32 %r12, [%rd6];"
            store_dst = "    st.global.s32 [%rd7], %r11;"
            instr = self._binary_instr(match.dtype, match.op)
            op_lines = [f"    {instr} %r11, %r11, %r12;"]
        elif match.dtype == "i1":
            reg_decl = ""
            load_lhs = "    ld.global.u8 %r11, [%rd5];"
            load_rhs = "    ld.global.u8 %r12, [%rd6];"
            store_dst = "    st.global.u8 [%rd7], %r11;"
            instr = self._binary_instr(match.dtype, match.op)
            op_lines = [f"    {instr} %r11, %r11, %r12;"]
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.lhs_name}_param,",
            f"    .param .u64 {match.rhs_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<4>;" if match.op == "math_atan2" else "    .reg .pred %p<3>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<8>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.lhs_name}_param];",
            f"    ld.param.u64 %rd2, [{match.rhs_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.dst_rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.dst_cols};",
            "    @%p2 bra L_next_row;",
            f"    mov.u32 %r3, %r1;",
            f"    mov.u32 %r4, %r2;",
        ]
        body = [line for line in body if line]
        if match.lhs_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.lhs_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r8, %r3, {match.lhs_cols}, %r4;",
                f"    mov.u32 %r5, %r1;",
                f"    mov.u32 %r6, %r2;",
            ]
        )
        if match.rhs_rows == 1:
            body.append("    mov.u32 %r5, 0;")
        if match.rhs_cols == 1:
            body.append("    mov.u32 %r6, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r9, %r5, {match.rhs_cols}, %r6;",
                f"    mad.lo.u32 %r10, %r1, {match.dst_cols}, %r2;",
                lhs_offset_line,
                "    add.s64 %rd5, %rd1, %rd4;",
                rhs_offset_line,
                "    add.s64 %rd6, %rd2, %rd4;",
                dst_offset_line,
                "    add.s64 %rd7, %rd3, %rd4;",
                load_lhs,
                load_rhs,
                *op_lines,
                store_dst,
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_cols_inner;",
                "L_next_row:",
                "    add.s32 %r1, %r1, 1;",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_copy_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorCopy2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        element_size = self._element_size(match.dtype)
        offset_line = (
            "    cvt.u64.u32 %rd3, %r7;"
            if element_size == 1
            else f"    mul.wide.u32 %rd3, %r7, {element_size};"
        )
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<2>;"
            load_src = "    ld.global.f32 %f1, [%rd4];"
            store_dst = "    st.global.f32 [%rd5], %f1;"
        elif match.dtype == "i32":
            reg_decl = ""
            load_src = "    ld.global.s32 %r7, [%rd4];"
            store_dst = "    st.global.s32 [%rd5], %r7;"
        elif match.dtype == "i1":
            reg_decl = ""
            load_src = "    ld.global.u8 %r7, [%rd4];"
            store_dst = "    st.global.u8 [%rd5], %r7;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<6>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.cols};",
            "    @%p2 bra L_next_row;",
            f"    mad.lo.u32 %r7, %r1, {match.cols}, %r2;",
            offset_line,
            "    add.s64 %rd4, %rd1, %rd3;",
            "    add.s64 %rd5, %rd2, %rd3;",
            load_src,
            store_dst,
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_cols_inner;",
            "L_next_row:",
            "    add.s32 %r1, %r1, 1;",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_copy_reduce_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorCopyReduce2DMatch,
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
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<6>;",
            "    .reg .f32 %f<4>;" if match.dtype == "f32" else "",
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.cols};",
            "    @%p2 bra L_next_row;",
            f"    mad.lo.u32 %r7, %r1, {match.cols}, %r2;",
            "    mul.wide.u32 %rd3, %r7, 4;",
            "    add.s64 %rd4, %rd1, %rd3;",
            "    add.s64 %rd5, %rd2, %rd3;",
            *self._copy_reduce_lines(match.dtype, match.reduction, src_ptr="%rd4", dst_ptr="%rd5"),
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_cols_inner;",
            "L_next_row:",
            "    add.s32 %r1, %r1, 1;",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_unary_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorUnary2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        element_size = self._element_size(match.dtype)
        offset_line = (
            f"    mul.wide.u32 %rd3, %r8, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd3, %r8;"
        )
        reg_decl = self._unary_f32_reg_decl(match.op) if match.dtype == "f32" else ""
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.src_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<4>;" if match.op in {"math_acos", "math_asin"} else "    .reg .pred %p<3>;",
            "    .reg .b32 %r<9>;",
            "    .reg .b64 %rd<6>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.src_name}_param];",
            f"    ld.param.u64 %rd2, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.cols};",
            "    @%p2 bra L_next_row;",
            f"    mad.lo.u32 %r8, %r1, {match.cols}, %r2;",
            offset_line,
            "    add.s64 %rd4, %rd1, %rd3;",
            "    add.s64 %rd5, %rd2, %rd3;",
            *self._unary_lines(match.dtype, match.op, src_ptr="%rd4", dst_ptr="%rd5"),
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_cols_inner;",
            "L_next_row:",
            "    add.s32 %r1, %r1, 1;",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_scalar_broadcast_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorScalarBroadcast2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        if match.dtype == "f32":
            reg_decl = self._binary_f32_reg_decl(match.op)
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.f32 %f2, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.f32 %f2, [%rd2];"
            if match.op == "math_atan2":
                op_lines = self._atan2_f32_core_lines(y_reg="%f1", x_reg="%f2", result_reg="%f3")
            else:
                instr = self._binary_instr(match.dtype, match.op)
                op_lines = [f"    {instr} %f3, %f1, %f2;"]
        elif match.dtype == "i32":
            reg_decl = ""
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.s32 %r3, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.s32 %r3, [%rd2];"
            instr = self._binary_instr(match.dtype, match.op)
            op_lines = [f"    {instr} %r7, %r7, %r3;"]
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<6>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd2, [{match.dst_name}_param];"
            src_ptr_reg = "%rd1"
            dst_ptr_reg = "%rd2"
            offset_reg = "%rd3"
            src_addr_reg = "%rd4"
            dst_addr_reg = "%rd5"
        else:
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<7>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd3, [{match.dst_name}_param];"
            src_ptr_reg = "%rd1"
            dst_ptr_reg = "%rd3"
            offset_reg = "%rd4"
            src_addr_reg = "%rd5"
            dst_addr_reg = "%rd6"

        body = [
            f".visible .entry {ir.name}(",
            *params,
            ")",
            "{",
            "    .reg .pred %p<4>;" if match.op == "math_atan2" else "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            b64_decl,
            reg_decl,
            src_ptr_load,
            *([f"    ld.param.u64 %rd2, [{match.scalar_name}_param];"] if match.scalar_mode == "tensor" else []),
            scalar_load,
            dst_ptr_load,
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.cols};",
            "    @%p2 bra L_next_row;",
            f"    mad.lo.u32 %r7, %r1, {match.cols}, %r2;",
            f"    mul.wide.u32 {offset_reg}, %r7, 4;",
            f"    add.s64 {src_addr_reg}, {src_ptr_reg}, {offset_reg};",
            f"    add.s64 {dst_addr_reg}, {dst_ptr_reg}, {offset_reg};",
            f"    ld.global.{self._scalar_param_type(match.dtype).removeprefix('.')} {'%f1' if match.dtype == 'f32' else '%r7'}, [{src_addr_reg}];",
            *op_lines,
            f"    st.global.{self._scalar_param_type(match.dtype).removeprefix('.')} [{dst_addr_reg}], {'%f3' if match.dtype == 'f32' else '%r7'};",
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_cols_inner;",
            "L_next_row:",
            "    add.s32 %r1, %r1, 1;",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
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
        body.extend(
            self._scalar_broadcast_declarations(
                match.dtype,
                mode=match.mode,
                scalar_mode=match.scalar_mode,
                op=match.op,
            )
        )
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

    def _render_tensor_compare_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorCompare2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<3>;"
            load_lhs = "    ld.global.f32 %f1, [%rd5];"
            load_rhs = "    ld.global.f32 %f2, [%rd6];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p2, %f1, %f2;"
            store_dst = "    st.global.u8 [%rd7], %r11;"
        elif match.dtype == "i32":
            reg_decl = ""
            load_lhs = "    ld.global.s32 %r11, [%rd5];"
            load_rhs = "    ld.global.s32 %r12, [%rd6];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p2, %r11, %r12;"
            store_dst = "    st.global.u8 [%rd7], %r11;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.lhs_name}_param,",
            f"    .param .u64 {match.rhs_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<8>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.lhs_name}_param];",
            f"    ld.param.u64 %rd2, [{match.rhs_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.dst_rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.dst_cols};",
            "    @%p2 bra L_next_row;",
            "    mov.u32 %r3, %r1;",
            "    mov.u32 %r4, %r2;",
        ]
        if match.lhs_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.lhs_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r8, %r3, {match.lhs_cols}, %r4;",
                "    mov.u32 %r5, %r1;",
                "    mov.u32 %r6, %r2;",
            ]
        )
        if match.rhs_rows == 1:
            body.append("    mov.u32 %r5, 0;")
        if match.rhs_cols == 1:
            body.append("    mov.u32 %r6, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r9, %r5, {match.rhs_cols}, %r6;",
                f"    mad.lo.u32 %r10, %r1, {match.dst_cols}, %r2;",
                "    mul.wide.u32 %rd4, %r8, 4;",
                "    add.s64 %rd5, %rd1, %rd4;",
                "    mul.wide.u32 %rd4, %r9, 4;",
                "    add.s64 %rd6, %rd2, %rd4;",
                load_lhs,
                load_rhs,
                compare_line,
                "    selp.u32 %r11, 1, 0, %p2;",
                "    cvt.u64.u32 %rd7, %r10;",
                "    add.s64 %rd7, %rd3, %rd7;",
                store_dst,
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_cols_inner;",
                "L_next_row:",
                "    add.s32 %r1, %r1, 1;",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_compare_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorCompare2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<3>;"
            load_lhs = "    ld.global.f32 %f1, [%rd5];"
            load_rhs = "    ld.global.f32 %f2, [%rd6];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p3, %f1, %f2;"
            store_dst = "    st.global.u8 [%rd7], %r11;"
        elif match.dtype == "i32":
            reg_decl = ""
            load_lhs = "    ld.global.s32 %r11, [%rd5];"
            load_rhs = "    ld.global.s32 %r12, [%rd6];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p3, %r11, %r12;"
            store_dst = "    st.global.u8 [%rd7], %r11;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.lhs_name}_param,",
            f"    .param .u64 {match.rhs_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<4>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<8>;",
            reg_decl,
            f"    ld.param.u64 %rd1, [{match.lhs_name}_param];",
            f"    ld.param.u64 %rd2, [{match.rhs_name}_param];",
            f"    ld.param.u64 %rd3, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.dst_cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.dst_rows};",
            "    @%p2 bra L_done;",
            "    mov.u32 %r3, %r2;",
            "    mov.u32 %r4, %r1;",
        ]
        if match.lhs_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.lhs_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r8, %r3, {match.lhs_cols}, %r4;",
                "    mov.u32 %r5, %r2;",
                "    mov.u32 %r6, %r1;",
            ]
        )
        if match.rhs_rows == 1:
            body.append("    mov.u32 %r5, 0;")
        if match.rhs_cols == 1:
            body.append("    mov.u32 %r6, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r9, %r5, {match.rhs_cols}, %r6;",
                f"    mad.lo.u32 %r10, %r2, {match.dst_cols}, %r1;",
                "    mul.wide.u32 %rd4, %r8, 4;",
                "    add.s64 %rd5, %rd1, %rd4;",
                "    mul.wide.u32 %rd4, %r9, 4;",
                "    add.s64 %rd6, %rd2, %rd4;",
                load_lhs,
                load_rhs,
                compare_line,
                "    selp.u32 %r11, 1, 0, %p3;",
                "    cvt.u64.u32 %rd7, %r10;",
                "    add.s64 %rd7, %rd3, %rd7;",
                store_dst,
                f"    add.s32 %r2, %r2, {grid_y};",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_scalar_compare_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorScalarCompare2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<3>;"
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.f32 %f2, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.f32 %f2, [%rd2];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p3, %f1, %f2;"
        elif match.dtype == "i32":
            reg_decl = ""
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.s32 %r3, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.s32 %r3, [%rd2];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p3, %r5, %r3;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<6>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd2, [{match.dst_name}_param];"
            offset_reg = "%rd3"
            src_addr_reg = "%rd4"
            dst_ptr_reg = "%rd2"
            dst_addr_reg = "%rd5"
        else:
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<7>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd3, [{match.dst_name}_param];"
            offset_reg = "%rd4"
            src_addr_reg = "%rd5"
            dst_ptr_reg = "%rd3"
            dst_addr_reg = "%rd6"
        if match.dtype == "f32":
            load_src = f"    ld.global.f32 %f1, [{src_addr_reg}];"
        else:
            load_src = f"    ld.global.s32 %r5, [{src_addr_reg}];"

        body = [
            f".visible .entry {ir.name}(",
            *params,
            ")",
            "{",
            "    .reg .pred %p<4>;",
            "    .reg .b32 %r<8>;",
            b64_decl,
            reg_decl,
            src_ptr_load,
            *([f"    ld.param.u64 %rd2, [{match.scalar_name}_param];"] if match.scalar_mode == "tensor" else []),
            scalar_load,
            dst_ptr_load,
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.cols};",
            "    @%p2 bra L_next_row;",
            f"    mad.lo.u32 %r7, %r1, {match.cols}, %r2;",
            f"    mul.wide.u32 {offset_reg}, %r7, 4;",
            f"    add.s64 {src_addr_reg}, %rd1, {offset_reg};",
            load_src,
            compare_line,
            "    selp.u32 %r6, 1, 0, %p3;",
            f"    cvt.u64.u32 {dst_addr_reg}, %r7;",
            f"    add.s64 {dst_addr_reg}, {dst_ptr_reg}, {dst_addr_reg};",
            f"    st.global.u8 [{dst_addr_reg}], %r6;",
            "    add.s32 %r2, %r2, 1;",
            "    bra.uni L_cols_inner;",
            "L_next_row:",
            "    add.s32 %r1, %r1, 1;",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_scalar_compare_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorScalarCompare2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        if match.dtype == "f32":
            reg_decl = "    .reg .f32 %f<3>;"
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.f32 %f2, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.f32 %f2, [%rd2];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p3, %f1, %f2;"
        elif match.dtype == "i32":
            reg_decl = ""
            if match.scalar_mode == "param":
                scalar_load = f"    ld.param.s32 %r3, [{match.scalar_name}_param];"
            else:
                scalar_load = "    ld.global.s32 %r3, [%rd2];"
            compare_line = f"    {self._compare_instr(match.dtype, match.op)} %p3, %r5, %r3;"
        else:
            raise CompilationError(f"unsupported PTX dtype '{match.dtype}'")

        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<6>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd2, [{match.dst_name}_param];"
            offset_reg = "%rd3"
            src_addr_reg = "%rd4"
            dst_ptr_reg = "%rd2"
            dst_addr_reg = "%rd5"
        else:
            params = [
                f"    .param .u64 {match.src_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            b64_decl = "    .reg .b64 %rd<7>;"
            src_ptr_load = f"    ld.param.u64 %rd1, [{match.src_name}_param];"
            dst_ptr_load = f"    ld.param.u64 %rd3, [{match.dst_name}_param];"
            offset_reg = "%rd4"
            src_addr_reg = "%rd5"
            dst_ptr_reg = "%rd3"
            dst_addr_reg = "%rd6"
        if match.dtype == "f32":
            load_src = f"    ld.global.f32 %f1, [{src_addr_reg}];"
        else:
            load_src = f"    ld.global.s32 %r5, [{src_addr_reg}];"

        body = [
            f".visible .entry {ir.name}(",
            *params,
            ")",
            "{",
            "    .reg .pred %p<4>;",
            "    .reg .b32 %r<8>;",
            b64_decl,
            reg_decl,
            src_ptr_load,
            *([f"    ld.param.u64 %rd2, [{match.scalar_name}_param];"] if match.scalar_mode == "tensor" else []),
            scalar_load,
            dst_ptr_load,
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r7, %r2, {match.cols}, %r1;",
            f"    mul.wide.u32 {offset_reg}, %r7, 4;",
            f"    add.s64 {src_addr_reg}, %rd1, {offset_reg};",
            load_src,
            compare_line,
            "    selp.u32 %r6, 1, 0, %p3;",
            f"    cvt.u64.u32 {dst_addr_reg}, %r7;",
            f"    add.s64 {dst_addr_reg}, {dst_ptr_reg}, {dst_addr_reg};",
            f"    st.global.u8 [{dst_addr_reg}], %r6;",
            f"    add.s32 %r2, %r2, {grid_y};",
            "    bra.uni L_rows_outer;",
            "L_done:",
            "    ret;",
            "}",
        ]
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_select_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorSelect2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        element_size = self._element_size(match.dtype)
        true_offset = (
            "    mul.wide.u32 %rd6, %r11, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd6, %r11;"
        )
        false_offset = (
            "    mul.wide.u32 %rd7, %r12, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd7, %r12;"
        )
        dst_offset = (
            "    mul.wide.u32 %rd8, %r10, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd8, %r10;"
        )
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.pred_name}_param,",
            f"    .param .u64 {match.true_name}_param,",
            f"    .param .u64 {match.false_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<9>;",
            *([] if match.dtype != "f32" else ["    .reg .f32 %f<4>;"]),
            f"    ld.param.u64 %rd1, [{match.pred_name}_param];",
            f"    ld.param.u64 %rd2, [{match.true_name}_param];",
            f"    ld.param.u64 %rd3, [{match.false_name}_param];",
            f"    ld.param.u64 %rd4, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.dst_rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.dst_cols};",
            "    @%p2 bra L_next_row;",
            f"    mad.lo.u32 %r10, %r1, {match.dst_cols}, %r2;",
            "    cvt.u64.u32 %rd5, %r10;",
            "    add.s64 %rd5, %rd1, %rd5;",
            "    mov.u32 %r3, %r1;",
            "    mov.u32 %r4, %r2;",
        ]
        if match.true_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.true_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r11, %r3, {match.true_cols}, %r4;",
                true_offset,
                "    add.s64 %rd6, %rd2, %rd6;",
                "    mov.u32 %r5, %r1;",
                "    mov.u32 %r6, %r2;",
            ]
        )
        if match.false_rows == 1:
            body.append("    mov.u32 %r5, 0;")
        if match.false_cols == 1:
            body.append("    mov.u32 %r6, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r12, %r5, {match.false_cols}, %r6;",
                false_offset,
                "    add.s64 %rd7, %rd3, %rd7;",
                dst_offset,
                "    add.s64 %rd8, %rd4, %rd8;",
            ]
        )
        body.extend(self._select_lines(match.dtype, pred_ptr="%rd5", true_ptr="%rd6", false_ptr="%rd7", dst_ptr="%rd8"))
        body.extend(
            [
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_cols_inner;",
                "L_next_row:",
                "    add.s32 %r1, %r1, 1;",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_select_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorSelect2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        element_size = self._element_size(match.dtype)
        true_offset = (
            "    mul.wide.u32 %rd6, %r11, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd6, %r11;"
        )
        false_offset = (
            "    mul.wide.u32 %rd7, %r12, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd7, %r12;"
        )
        dst_offset = (
            "    mul.wide.u32 %rd8, %r10, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd8, %r10;"
        )
        body = [
            f".visible .entry {ir.name}(",
            f"    .param .u64 {match.pred_name}_param,",
            f"    .param .u64 {match.true_name}_param,",
            f"    .param .u64 {match.false_name}_param,",
            f"    .param .u64 {match.dst_name}_param",
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<9>;",
            *([] if match.dtype != "f32" else ["    .reg .f32 %f<4>;"]),
            f"    ld.param.u64 %rd1, [{match.pred_name}_param];",
            f"    ld.param.u64 %rd2, [{match.true_name}_param];",
            f"    ld.param.u64 %rd3, [{match.false_name}_param];",
            f"    ld.param.u64 %rd4, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.dst_cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.dst_rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r10, %r2, {match.dst_cols}, %r1;",
            "    cvt.u64.u32 %rd5, %r10;",
            "    add.s64 %rd5, %rd1, %rd5;",
            "    mov.u32 %r3, %r2;",
            "    mov.u32 %r4, %r1;",
        ]
        if match.true_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.true_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r11, %r3, {match.true_cols}, %r4;",
                true_offset,
                "    add.s64 %rd6, %rd2, %rd6;",
                "    mov.u32 %r5, %r2;",
                "    mov.u32 %r6, %r1;",
            ]
        )
        if match.false_rows == 1:
            body.append("    mov.u32 %r5, 0;")
        if match.false_cols == 1:
            body.append("    mov.u32 %r6, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r12, %r5, {match.false_cols}, %r6;",
                false_offset,
                "    add.s64 %rd7, %rd3, %rd7;",
                dst_offset,
                "    add.s64 %rd8, %rd4, %rd8;",
            ]
        )
        body.extend(self._select_lines(match.dtype, pred_ptr="%rd5", true_ptr="%rd6", false_ptr="%rd7", dst_ptr="%rd8"))
        body.extend(
            [
                f"    add.s32 %r2, %r2, {grid_y};",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_tensor_scalar_select_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorScalarSelect2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        element_size = self._element_size(match.dtype)
        tensor_offset = (
            "    mul.wide.u32 %rd6, %r11, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd6, %r11;"
        )
        dst_offset = (
            "    mul.wide.u32 %rd8, %r10, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd8, %r10;"
        )
        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.pred_name}_param,",
                f"    .param .u64 {match.tensor_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            scalar_load = {
                "f32": f"    ld.param.f32 %f2, [{match.scalar_name}_param];",
                "i32": f"    ld.param.s32 %r7, [{match.scalar_name}_param];",
                "i1": f"    ld.param.u8 %r7, [{match.scalar_name}_param];",
            }[match.dtype]
            scalar_ptr_load: list[str] = []
        else:
            params = [
                f"    .param .u64 {match.pred_name}_param,",
                f"    .param .u64 {match.tensor_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            scalar_load = ""
            scalar_ptr_load = [f"    ld.param.u64 %rd7, [{match.scalar_name}_param];"]
        body = [
            f".visible .entry {ir.name}(",
            *params,
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<9>;",
            *([] if match.dtype != "f32" else ["    .reg .f32 %f<4>;"]),
            f"    ld.param.u64 %rd1, [{match.pred_name}_param];",
            f"    ld.param.u64 %rd2, [{match.tensor_name}_param];",
            *scalar_ptr_load,
            scalar_load,
            f"    ld.param.u64 %rd3, [{match.dst_name}_param];",
            "    mov.u32 %r1, 0;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p1, %r1, {match.rows};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, 0;",
            "L_cols_inner:",
            f"    setp.ge.u32 %p2, %r2, {match.cols};",
            "    @%p2 bra L_next_row;",
            f"    mad.lo.u32 %r10, %r1, {match.cols}, %r2;",
            "    cvt.u64.u32 %rd4, %r10;",
            "    add.s64 %rd4, %rd1, %rd4;",
            "    mov.u32 %r3, %r1;",
            "    mov.u32 %r4, %r2;",
        ]
        if match.tensor_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.tensor_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r11, %r3, {match.tensor_cols}, %r4;",
                tensor_offset,
                "    add.s64 %rd6, %rd2, %rd6;",
                dst_offset,
                "    add.s64 %rd8, %rd3, %rd8;",
            ]
        )
        body.extend(
            self._scalar_select_lines(
                match.dtype,
                scalar_mode=match.scalar_mode,
                tensor_branch=match.tensor_branch,
            )
        )
        body.extend(
            [
                "    add.s32 %r2, %r2, 1;",
                "    bra.uni L_cols_inner;",
                "L_next_row:",
                "    add.s32 %r1, %r1, 1;",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        body = [line for line in body if line]
        return ptx + "\n".join(body) + "\n"

    def _render_parallel_tensor_scalar_select_2d(
        self,
        ir: PortableKernelIR,
        target: NvidiaTarget,
        match: _TensorScalarSelect2DMatch,
        *,
        backend_name: str,
    ) -> str:
        ptx = self._ptx_prologue(ir, target, backend_name=backend_name)
        block_x = ir.launch.block[0]
        grid_y = ir.launch.grid[1]
        element_size = self._element_size(match.dtype)
        tensor_offset = (
            "    mul.wide.u32 %rd6, %r11, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd6, %r11;"
        )
        dst_offset = (
            "    mul.wide.u32 %rd8, %r10, 4;"
            if element_size != 1
            else "    cvt.u64.u32 %rd8, %r10;"
        )
        if match.scalar_mode == "param":
            params = [
                f"    .param .u64 {match.pred_name}_param,",
                f"    .param .u64 {match.tensor_name}_param,",
                f"    .param {self._scalar_param_type(match.dtype)} {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            scalar_load = {
                "f32": f"    ld.param.f32 %f2, [{match.scalar_name}_param];",
                "i32": f"    ld.param.s32 %r7, [{match.scalar_name}_param];",
                "i1": f"    ld.param.u8 %r7, [{match.scalar_name}_param];",
            }[match.dtype]
            scalar_ptr_load: list[str] = []
        else:
            params = [
                f"    .param .u64 {match.pred_name}_param,",
                f"    .param .u64 {match.tensor_name}_param,",
                f"    .param .u64 {match.scalar_name}_param,",
                f"    .param .u64 {match.dst_name}_param",
            ]
            scalar_load = ""
            scalar_ptr_load = [f"    ld.param.u64 %rd7, [{match.scalar_name}_param];"]
        body = [
            f".visible .entry {ir.name}(",
            *params,
            ")",
            "{",
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<13>;",
            "    .reg .b64 %rd<9>;",
            *([] if match.dtype != "f32" else ["    .reg .f32 %f<4>;"]),
            f"    ld.param.u64 %rd1, [{match.pred_name}_param];",
            f"    ld.param.u64 %rd2, [{match.tensor_name}_param];",
            *scalar_ptr_load,
            scalar_load,
            f"    ld.param.u64 %rd3, [{match.dst_name}_param];",
            "    mov.u32 %r1, %tid.x;",
            "    mov.u32 %r2, %ctaid.x;",
            f"    mad.lo.u32 %r1, %r2, {block_x}, %r1;",
            f"    setp.ge.u32 %p1, %r1, {match.cols};",
            "    @%p1 bra L_done;",
            "    mov.u32 %r2, %ctaid.y;",
            "L_rows_outer:",
            f"    setp.ge.u32 %p2, %r2, {match.rows};",
            "    @%p2 bra L_done;",
            f"    mad.lo.u32 %r10, %r2, {match.cols}, %r1;",
            "    cvt.u64.u32 %rd4, %r10;",
            "    add.s64 %rd4, %rd1, %rd4;",
            "    mov.u32 %r3, %r2;",
            "    mov.u32 %r4, %r1;",
        ]
        if match.tensor_rows == 1:
            body.append("    mov.u32 %r3, 0;")
        if match.tensor_cols == 1:
            body.append("    mov.u32 %r4, 0;")
        body.extend(
            [
                f"    mad.lo.u32 %r11, %r3, {match.tensor_cols}, %r4;",
                tensor_offset,
                "    add.s64 %rd6, %rd2, %rd6;",
                dst_offset,
                "    add.s64 %rd8, %rd3, %rd8;",
            ]
        )
        body.extend(
            self._scalar_select_lines(
                match.dtype,
                scalar_mode=match.scalar_mode,
                tensor_branch=match.tensor_branch,
            )
        )
        body.extend(
            [
                f"    add.s32 %r2, %r2, {grid_y};",
                "    bra.uni L_rows_outer;",
                "L_done:",
                "    ret;",
                "}",
            ]
        )
        body = [line for line in body if line]
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

    def _binary_declarations(self, dtype: str, *, mode: str, op: str) -> list[str]:
        lines = [
            self._binary_pred_decl(op),
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<8>;",
        ]
        if dtype == "f32":
            lines.append(self._binary_f32_reg_decl(op))
        return lines

    def _compare_declarations(self, dtype: str) -> list[str]:
        lines = [
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<8>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<3>;")
        return lines

    def _select_declarations(self, dtype: str) -> list[str]:
        lines = [
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<11>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<4>;")
        return lines

    def _scalar_select_declarations(self, dtype: str, *, scalar_mode: str) -> list[str]:
        del scalar_mode
        lines = [
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<8>;",
            "    .reg .b64 %rd<9>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<4>;")
        return lines

    def _unary_declarations(self, dtype: str, *, mode: str, op: str) -> list[str]:
        del mode
        lines = [
            self._unary_pred_decl(op),
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<6>;",
        ]
        if dtype == "f32":
            lines.append(self._unary_f32_reg_decl(op))
        return lines

    def _reduce_declarations(self, dtype: str) -> list[str]:
        lines = [
            "    .reg .pred %p<3>;",
            "    .reg .b32 %r<5>;",
            "    .reg .b64 %rd<5>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<3>;")
        return lines

    def _scalar_broadcast_declarations(self, dtype: str, *, mode: str, scalar_mode: str, op: str) -> list[str]:
        del mode, scalar_mode
        lines = [
            self._binary_pred_decl(op),
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<8>;",
        ]
        if dtype == "f32":
            lines.append(self._binary_f32_reg_decl(op))
        return lines

    def _index_lines(self, mode: str, extent: int, *, dtype: str, param_names: tuple[str, ...]) -> list[str]:
        element_size = self._element_size(dtype)
        offset_line = (
            f"    mul.wide.u32 %rd3, %r4, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd3, %r4;"
        )
        rhs_offset_line = (
            f"    mul.wide.u32 %rd4, %r4, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd4, %r4;"
        )
        direct_offset_line = (
            f"    mul.wide.u32 %rd3, %r1, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd3, %r1;"
        )
        direct_rhs_offset_line = (
            f"    mul.wide.u32 %rd4, %r1, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd4, %r1;"
        )
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
                    offset_line if not rhs_param else rhs_offset_line,
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
                direct_offset_line if not rhs_param else direct_rhs_offset_line,
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
                        f"    ld.param.s32 %r5, [{param_names[1]}];",
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

    def _compare_index_lines(self, mode: str, extent: int, *, param_names: tuple[str, str, str]) -> list[str]:
        if mode == "indexed":
            return [
                f"    ld.param.u64 %rd1, [{param_names[0]}];",
                f"    ld.param.u64 %rd2, [{param_names[1]}];",
                f"    ld.param.u64 %rd3, [{param_names[2]}];",
                "    mov.u32 %r1, %tid.x;",
                "    mov.u32 %r2, %ctaid.x;",
                "    mov.u32 %r3, %ntid.x;",
                "    mad.lo.u32 %r4, %r2, %r3, %r1;",
                f"    setp.ge.u32 %p1, %r4, {extent};",
                "    @%p1 bra L_done;",
                "    mul.wide.u32 %rd4, %r4, 4;",
                "    add.s64 %rd5, %rd1, %rd4;",
                "    add.s64 %rd6, %rd2, %rd4;",
                "    cvt.u64.u32 %rd7, %r4;",
                "    add.s64 %rd7, %rd3, %rd7;",
            ]
        return [
            f"    ld.param.u64 %rd1, [{param_names[0]}];",
            f"    ld.param.u64 %rd2, [{param_names[1]}];",
            f"    ld.param.u64 %rd3, [{param_names[2]}];",
            "    mov.u32 %r1, %tid.x;",
            f"    setp.ge.u32 %p1, %r1, {extent};",
            "    @%p1 bra L_done;",
            "    mul.wide.u32 %rd4, %r1, 4;",
            "    add.s64 %rd5, %rd1, %rd4;",
            "    add.s64 %rd6, %rd2, %rd4;",
            "    cvt.u64.u32 %rd7, %r1;",
            "    add.s64 %rd7, %rd3, %rd7;",
        ]

    def _scalar_compare_index_lines(
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
                    lines.append(f"    ld.param.s32 %r3, [{param_names[1]}];")
                lines.extend(
                    [
                        f"    ld.param.u64 %rd2, [{param_names[2]}];",
                        "    cvt.u64.u32 %rd5, %r4;",
                        "    add.s64 %rd5, %rd2, %rd5;",
                    ]
                )
                return lines
            lines.extend(
                [
                    f"    ld.param.u64 %rd2, [{param_names[1]}];",
                    f"    ld.param.u64 %rd6, [{param_names[2]}];",
                    "    cvt.u64.u32 %rd5, %r4;",
                    "    add.s64 %rd5, %rd6, %rd5;",
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
                        "    cvt.u64.u32 %rd5, %r1;",
                        "    add.s64 %rd5, %rd2, %rd5;",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"    ld.param.s32 %r3, [{param_names[1]}];",
                        f"    ld.param.u64 %rd2, [{param_names[2]}];",
                        "    cvt.u64.u32 %rd5, %r1;",
                        "    add.s64 %rd5, %rd2, %rd5;",
                    ]
                )
            return lines
        lines.extend(
            [
                f"    ld.param.u64 %rd2, [{param_names[1]}];",
                f"    ld.param.u64 %rd6, [{param_names[2]}];",
                "    cvt.u64.u32 %rd5, %r1;",
                "    add.s64 %rd5, %rd6, %rd5;",
            ]
        )
        return lines

    def _select_index_lines(
        self,
        mode: str,
        extent: int,
        *,
        dtype: str,
        param_names: tuple[str, str, str, str],
    ) -> list[str]:
        element_size = self._element_size(dtype)
        indexed_value_offset = (
            f"    mul.wide.u32 %rd7, %r4, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd7, %r4;"
        )
        direct_value_offset = (
            f"    mul.wide.u32 %rd7, %r1, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd7, %r1;"
        )
        if mode == "indexed":
            return [
                f"    ld.param.u64 %rd1, [{param_names[0]}];",
                f"    ld.param.u64 %rd2, [{param_names[1]}];",
                f"    ld.param.u64 %rd3, [{param_names[2]}];",
                f"    ld.param.u64 %rd4, [{param_names[3]}];",
                "    mov.u32 %r1, %tid.x;",
                "    mov.u32 %r2, %ctaid.x;",
                "    mov.u32 %r3, %ntid.x;",
                "    mad.lo.u32 %r4, %r2, %r3, %r1;",
                f"    setp.ge.u32 %p1, %r4, {extent};",
                "    @%p1 bra L_done;",
                "    cvt.u64.u32 %rd5, %r4;",
                "    add.s64 %rd6, %rd1, %rd5;",
                indexed_value_offset,
                "    add.s64 %rd8, %rd2, %rd7;",
                "    add.s64 %rd9, %rd3, %rd7;",
                "    add.s64 %rd10, %rd4, %rd7;",
            ]
        return [
            f"    ld.param.u64 %rd1, [{param_names[0]}];",
            f"    ld.param.u64 %rd2, [{param_names[1]}];",
            f"    ld.param.u64 %rd3, [{param_names[2]}];",
            f"    ld.param.u64 %rd4, [{param_names[3]}];",
            "    mov.u32 %r1, %tid.x;",
            f"    setp.ge.u32 %p1, %r1, {extent};",
            "    @%p1 bra L_done;",
            "    cvt.u64.u32 %rd5, %r1;",
            "    add.s64 %rd6, %rd1, %rd5;",
            direct_value_offset,
            "    add.s64 %rd8, %rd2, %rd7;",
            "    add.s64 %rd9, %rd3, %rd7;",
            "    add.s64 %rd10, %rd4, %rd7;",
        ]

    def _scalar_select_index_lines(
        self,
        mode: str,
        extent: int,
        *,
        dtype: str,
        scalar_mode: str,
        param_names: tuple[str, str, str, str],
    ) -> list[str]:
        element_size = self._element_size(dtype)
        value_offset_indexed = (
            f"    mul.wide.u32 %rd5, %r4, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd5, %r4;"
        )
        value_offset_direct = (
            f"    mul.wide.u32 %rd5, %r1, {element_size};"
            if element_size != 1
            else "    cvt.u64.u32 %rd5, %r1;"
        )
        lines: list[str]
        if mode == "indexed":
            lines = [
                f"    ld.param.u64 %rd1, [{param_names[0]}];",
                f"    ld.param.u64 %rd2, [{param_names[1]}];",
                "    mov.u32 %r1, %tid.x;",
                "    mov.u32 %r2, %ctaid.x;",
                "    mov.u32 %r3, %ntid.x;",
                "    mad.lo.u32 %r4, %r2, %r3, %r1;",
                f"    setp.ge.u32 %p1, %r4, {extent};",
                "    @%p1 bra L_done;",
                "    cvt.u64.u32 %rd3, %r4;",
                "    add.s64 %rd4, %rd1, %rd3;",
                value_offset_indexed,
                "    add.s64 %rd6, %rd2, %rd5;",
            ]
            if scalar_mode == "param":
                if dtype == "f32":
                    lines.append(f"    ld.param.f32 %f2, [{param_names[2]}];")
                elif dtype == "i1":
                    lines.append(f"    ld.param.u8 %r7, [{param_names[2]}];")
                else:
                    lines.append(f"    ld.param.s32 %r7, [{param_names[2]}];")
                lines.extend(
                    [
                        f"    ld.param.u64 %rd7, [{param_names[3]}];",
                        "    add.s64 %rd8, %rd7, %rd5;",
                    ]
                )
                return lines
            lines.extend(
                [
                    f"    ld.param.u64 %rd7, [{param_names[2]}];",
                    f"    ld.param.u64 %rd8, [{param_names[3]}];",
                    "    add.s64 %rd8, %rd8, %rd5;",
                ]
            )
            return lines
        lines = [
            f"    ld.param.u64 %rd1, [{param_names[0]}];",
            f"    ld.param.u64 %rd2, [{param_names[1]}];",
            "    mov.u32 %r1, %tid.x;",
            f"    setp.ge.u32 %p1, %r1, {extent};",
            "    @%p1 bra L_done;",
            "    cvt.u64.u32 %rd3, %r1;",
            "    add.s64 %rd4, %rd1, %rd3;",
            value_offset_direct,
            "    add.s64 %rd6, %rd2, %rd5;",
        ]
        if scalar_mode == "param":
            if dtype == "f32":
                lines.extend(
                    [
                        f"    ld.param.f32 %f2, [{param_names[2]}];",
                        f"    ld.param.u64 %rd7, [{param_names[3]}];",
                        "    add.s64 %rd8, %rd7, %rd5;",
                    ]
                )
            elif dtype == "i1":
                lines.extend(
                    [
                        f"    ld.param.u8 %r7, [{param_names[2]}];",
                        f"    ld.param.u64 %rd7, [{param_names[3]}];",
                        "    add.s64 %rd8, %rd7, %rd5;",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"    ld.param.s32 %r7, [{param_names[2]}];",
                        f"    ld.param.u64 %rd7, [{param_names[3]}];",
                        "    add.s64 %rd8, %rd7, %rd5;",
                    ]
                )
            return lines
        lines.extend(
            [
                f"    ld.param.u64 %rd7, [{param_names[2]}];",
                f"    ld.param.u64 %rd8, [{param_names[3]}];",
                "    add.s64 %rd8, %rd8, %rd5;",
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
        if dtype == "i1":
            return [
                f"    ld.global.u8 %r5, [{lhs_ptr}];",
                f"    st.global.u8 [{dst_ptr}], %r5;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _copy_reduce_lines(self, dtype: str, reduction: str, *, src_ptr: str, dst_ptr: str) -> list[str]:
        instr = self._copy_reduce_instr(dtype, reduction)
        if dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{dst_ptr}];",
                f"    ld.global.f32 %f2, [{src_ptr}];",
                f"    {instr} %f3, %f1, %f2;",
                f"    st.global.f32 [{dst_ptr}], %f3;",
            ]
        if dtype == "i32":
            return [
                f"    ld.global.s32 %r3, [{dst_ptr}];",
                f"    ld.global.s32 %r4, [{src_ptr}];",
                f"    {instr} %r3, %r3, %r4;",
                f"    st.global.s32 [{dst_ptr}], %r3;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _copy_reduce_declarations(self, dtype: str, *, mode: str) -> list[str]:
        del mode
        lines = [
            "    .reg .pred %p<2>;",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<6>;",
        ]
        if dtype == "f32":
            lines.append("    .reg .f32 %f<4>;")
        return lines

    def _binary_lines(self, dtype: str, op: str, *, lhs_ptr: str, rhs_ptr: str, dst_ptr: str) -> list[str]:
        if dtype == "f32":
            if op == "math_atan2":
                return [
                    f"    ld.global.f32 %f1, [{lhs_ptr}];",
                    f"    ld.global.f32 %f2, [{rhs_ptr}];",
                    *self._atan2_f32_core_lines(y_reg="%f1", x_reg="%f2", result_reg="%f3"),
                    f"    st.global.f32 [{dst_ptr}], %f3;",
                ]
            instr = self._binary_instr(dtype, op)
            return [
                f"    ld.global.f32 %f1, [{lhs_ptr}];",
                f"    ld.global.f32 %f2, [{rhs_ptr}];",
                f"    {instr} %f3, %f1, %f2;",
                f"    st.global.f32 [{dst_ptr}], %f3;",
            ]
        if dtype == "i32":
            instr = self._binary_instr(dtype, op)
            return [
                f"    ld.global.s32 %r5, [{lhs_ptr}];",
                f"    ld.global.s32 %r2, [{rhs_ptr}];",
                f"    {instr} %r5, %r5, %r2;",
                f"    st.global.s32 [{dst_ptr}], %r5;",
            ]
        if dtype == "i1":
            instr = self._binary_instr(dtype, op)
            return [
                f"    ld.global.u8 %r5, [{lhs_ptr}];",
                f"    ld.global.u8 %r2, [{rhs_ptr}];",
                f"    {instr} %r5, %r5, %r2;",
                f"    st.global.u8 [{dst_ptr}], %r5;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _select_lines(self, dtype: str, *, pred_ptr: str, true_ptr: str, false_ptr: str, dst_ptr: str) -> list[str]:
        if dtype == "f32":
            return [
                f"    ld.global.u8 %r5, [{pred_ptr}];",
                "    setp.ne.u32 %p2, %r5, 0;",
                f"    ld.global.f32 %f1, [{true_ptr}];",
                f"    ld.global.f32 %f2, [{false_ptr}];",
                "    selp.f32 %f3, %f1, %f2, %p2;",
                f"    st.global.f32 [{dst_ptr}], %f3;",
            ]
        if dtype == "i32":
            return [
                f"    ld.global.u8 %r5, [{pred_ptr}];",
                "    setp.ne.u32 %p2, %r5, 0;",
                f"    ld.global.s32 %r6, [{true_ptr}];",
                f"    ld.global.s32 %r7, [{false_ptr}];",
                "    selp.b32 %r6, %r6, %r7, %p2;",
                f"    st.global.s32 [{dst_ptr}], %r6;",
            ]
        if dtype == "i1":
            return [
                f"    ld.global.u8 %r5, [{pred_ptr}];",
                "    setp.ne.u32 %p2, %r5, 0;",
                f"    ld.global.u8 %r6, [{true_ptr}];",
                f"    ld.global.u8 %r7, [{false_ptr}];",
                "    selp.b32 %r6, %r6, %r7, %p2;",
                f"    st.global.u8 [{dst_ptr}], %r6;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _scalar_select_lines(self, dtype: str, *, scalar_mode: str, tensor_branch: str) -> list[str]:
        if dtype == "f32":
            lines = [
                "    ld.global.u8 %r5, [%rd4];",
                "    setp.ne.u32 %p2, %r5, 0;",
                "    ld.global.f32 %f1, [%rd6];",
            ]
            if scalar_mode == "tensor":
                lines.append("    ld.global.f32 %f2, [%rd7];")
            if tensor_branch == "true":
                lines.append("    selp.f32 %f3, %f1, %f2, %p2;")
            else:
                lines.append("    selp.f32 %f3, %f2, %f1, %p2;")
            lines.append("    st.global.f32 [%rd8], %f3;")
            return lines
        if dtype == "i32":
            lines = [
                "    ld.global.u8 %r5, [%rd4];",
                "    setp.ne.u32 %p2, %r5, 0;",
                "    ld.global.s32 %r6, [%rd6];",
            ]
            if scalar_mode == "tensor":
                lines.append("    ld.global.s32 %r7, [%rd7];")
            if tensor_branch == "true":
                lines.append("    selp.b32 %r6, %r6, %r7, %p2;")
            else:
                lines.append("    selp.b32 %r6, %r7, %r6, %p2;")
            lines.append("    st.global.s32 [%rd8], %r6;")
            return lines
        if dtype == "i1":
            lines = [
                "    ld.global.u8 %r5, [%rd4];",
                "    setp.ne.u32 %p2, %r5, 0;",
                "    ld.global.u8 %r6, [%rd6];",
            ]
            if scalar_mode == "tensor":
                lines.append("    ld.global.u8 %r7, [%rd7];")
            if tensor_branch == "true":
                lines.append("    selp.b32 %r6, %r6, %r7, %p2;")
            else:
                lines.append("    selp.b32 %r6, %r7, %r6, %p2;")
            lines.append("    st.global.u8 [%rd8], %r6;")
            return lines
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _unary_lines(self, dtype: str, op: str, *, src_ptr: str, dst_ptr: str) -> list[str]:
        if op == "neg" and dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    neg.f32 %f2, %f1;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op == "neg" and dtype == "i32":
            return [
                f"    ld.global.s32 %r5, [{src_ptr}];",
                "    neg.s32 %r5, %r5;",
                f"    st.global.s32 [{dst_ptr}], %r5;",
            ]
        if op == "abs" and dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    abs.f32 %f2, %f1;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op == "abs" and dtype == "i32":
            return [
                f"    ld.global.s32 %r5, [{src_ptr}];",
                "    abs.s32 %r5, %r5;",
                f"    st.global.s32 [{dst_ptr}], %r5;",
            ]
        if op == "math_erf" and dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    abs.f32 %f2, %f1;",
                f"    mov.f32 %f3, {self._float_immediate(0.3275911)};",
                "    mul.rn.f32 %f3, %f2, %f3;",
                f"    mov.f32 %f4, {self._float_immediate(1.0)};",
                "    add.rn.f32 %f3, %f3, %f4;",
                "    rcp.approx.f32 %f3, %f3;",
                f"    mov.f32 %f4, {self._float_immediate(1.061405429)};",
                "    mul.rn.f32 %f4, %f4, %f3;",
                f"    mov.f32 %f5, {self._float_immediate(-1.453152027)};",
                "    add.rn.f32 %f4, %f4, %f5;",
                "    mul.rn.f32 %f4, %f4, %f3;",
                f"    mov.f32 %f5, {self._float_immediate(1.421413741)};",
                "    add.rn.f32 %f4, %f4, %f5;",
                "    mul.rn.f32 %f4, %f4, %f3;",
                f"    mov.f32 %f5, {self._float_immediate(-0.284496736)};",
                "    add.rn.f32 %f4, %f4, %f5;",
                "    mul.rn.f32 %f4, %f4, %f3;",
                f"    mov.f32 %f5, {self._float_immediate(0.254829592)};",
                "    add.rn.f32 %f4, %f4, %f5;",
                "    mul.rn.f32 %f4, %f4, %f3;",
                "    mul.rn.f32 %f5, %f2, %f2;",
                "    neg.f32 %f5, %f5;",
                f"    mov.f32 %f6, {self._float_immediate(1.4426950408889634)};",
                "    mul.rn.f32 %f5, %f5, %f6;",
                "    ex2.approx.f32 %f5, %f5;",
                "    mul.rn.f32 %f4, %f4, %f5;",
                f"    mov.f32 %f6, {self._float_immediate(1.0)};",
                "    sub.rn.f32 %f4, %f6, %f4;",
                "    neg.f32 %f5, %f4;",
                f"    mov.f32 %f6, {self._float_immediate(0.0)};",
                "    setp.lt.f32 %p0, %f1, %f6;",
                "    selp.f32 %f2, %f5, %f4, %p0;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op == "math_acos" and dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    mul.rn.f32 %f2, %f1, %f1;",
                f"    mov.f32 %f3, {self._float_immediate(1.0)};",
                "    sub.rn.f32 %f2, %f3, %f2;",
                f"    mov.f32 %f3, {self._float_immediate(0.0)};",
                "    max.f32 %f2, %f2, %f3;",
                "    sqrt.rn.f32 %f2, %f2;",
                *self._atan2_f32_core_lines(y_reg="%f2", x_reg="%f1", result_reg="%f3"),
                f"    st.global.f32 [{dst_ptr}], %f3;",
            ]
        if op == "math_asin" and dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    mul.rn.f32 %f2, %f1, %f1;",
                f"    mov.f32 %f3, {self._float_immediate(1.0)};",
                "    sub.rn.f32 %f2, %f3, %f2;",
                f"    mov.f32 %f3, {self._float_immediate(0.0)};",
                "    max.f32 %f2, %f2, %f3;",
                "    sqrt.rn.f32 %f2, %f2;",
                *self._atan2_f32_core_lines(y_reg="%f1", x_reg="%f2", result_reg="%f3"),
                f"    st.global.f32 [{dst_ptr}], %f3;",
            ]
        if op == "math_atan" and dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    abs.f32 %f2, %f1;",
                f"    mov.f32 %f3, {self._float_immediate(1.0)};",
                "    setp.gt.f32 %p0, %f2, %f3;",
                "    rcp.approx.f32 %f4, %f2;",
                "    selp.f32 %f2, %f4, %f2, %p0;",
                "    sub.rn.f32 %f3, %f3, %f2;",
                f"    mov.f32 %f4, {self._float_immediate(0.273)};",
                "    mul.rn.f32 %f3, %f3, %f4;",
                f"    mov.f32 %f4, {self._float_immediate(0.7853981633974483)};",
                "    add.rn.f32 %f3, %f4, %f3;",
                "    mul.rn.f32 %f3, %f2, %f3;",
                f"    mov.f32 %f4, {self._float_immediate(1.5707963267948966)};",
                "    sub.rn.f32 %f5, %f4, %f3;",
                "    selp.f32 %f3, %f5, %f3, %p0;",
                f"    mov.f32 %f4, {self._float_immediate(0.0)};",
                "    setp.lt.f32 %p0, %f1, %f4;",
                "    neg.f32 %f5, %f3;",
                "    selp.f32 %f2, %f5, %f3, %p0;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op == "math_exp":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                f"    mov.f32 %f2, {self._float_immediate(1.4426950408889634)};",
                "    mul.rn.f32 %f1, %f1, %f2;",
                "    ex2.approx.f32 %f2, %f1;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op == "math_log":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    lg2.approx.f32 %f2, %f1;",
                f"    mov.f32 %f1, {self._float_immediate(0.6931471805599453)};",
                "    mul.rn.f32 %f2, %f2, %f1;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op == "math_log10":
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                "    lg2.approx.f32 %f2, %f1;",
                f"    mov.f32 %f1, {self._float_immediate(0.3010299956639812)};",
                "    mul.rn.f32 %f2, %f2, %f1;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op in {"math_round", "math_floor", "math_ceil", "math_trunc", "math_sqrt", "math_rsqrt", "math_sin", "math_cos", "math_exp2", "math_log2"}:
            instr = {
                "math_round": "cvt.rni.f32.f32",
                "math_floor": "cvt.rmi.f32.f32",
                "math_ceil": "cvt.rpi.f32.f32",
                "math_trunc": "cvt.rzi.f32.f32",
                "math_sqrt": "sqrt.rn.f32",
                "math_rsqrt": "rsqrt.approx.f32",
                "math_sin": "sin.approx.f32",
                "math_cos": "cos.approx.f32",
                "math_exp2": "ex2.approx.f32",
                "math_log2": "lg2.approx.f32",
            }[op]
            return [
                f"    ld.global.f32 %f1, [{src_ptr}];",
                f"    {instr} %f2, %f1;",
                f"    st.global.f32 [{dst_ptr}], %f2;",
            ]
        if op == "bitnot" and dtype == "i32":
            return [
                f"    ld.global.s32 %r5, [{src_ptr}];",
                "    not.b32 %r5, %r5;",
                f"    st.global.s32 [{dst_ptr}], %r5;",
            ]
        if op == "bitnot" and dtype == "i1":
            return [
                f"    ld.global.u8 %r5, [{src_ptr}];",
                "    xor.b32 %r5, %r5, 1;",
                f"    st.global.u8 [{dst_ptr}], %r5;",
            ]
        raise CompilationError(f"unsupported PTX unary op '{op}'")

    def _atan2_f32_core_lines(self, *, y_reg: str, x_reg: str, result_reg: str) -> list[str]:
        return [
            f"    abs.f32 %f4, {y_reg};",
            f"    abs.f32 %f5, {x_reg};",
            f"    mov.f32 %f12, {self._float_immediate(0.0)};",
            f"    mov.f32 %f7, {self._float_immediate(1.0)};",
            "    setp.eq.f32 %p1, %f5, %f12;",
            "    selp.f32 %f6, %f7, %f5, %p1;",
            "    rcp.approx.f32 %f8, %f6;",
            "    mul.rn.f32 %f6, %f4, %f8;",
            "    setp.gt.f32 %p0, %f6, %f7;",
            "    rcp.approx.f32 %f8, %f6;",
            "    selp.f32 %f6, %f8, %f6, %p0;",
            "    sub.rn.f32 %f8, %f7, %f6;",
            f"    mov.f32 %f9, {self._float_immediate(0.273)};",
            "    mul.rn.f32 %f8, %f8, %f9;",
            f"    mov.f32 %f9, {self._float_immediate(0.7853981633974483)};",
            "    add.rn.f32 %f8, %f8, %f9;",
            f"    mul.rn.f32 {result_reg}, %f6, %f8;",
            f"    mov.f32 %f10, {self._float_immediate(1.5707963267948966)};",
            f"    sub.rn.f32 %f8, %f10, {result_reg};",
            f"    selp.f32 {result_reg}, %f8, {result_reg}, %p0;",
            f"    mov.f32 %f11, {self._float_immediate(3.141592653589793)};",
            f"    sub.rn.f32 %f8, %f11, {result_reg};",
            f"    setp.lt.f32 %p2, {x_reg}, %f12;",
            f"    selp.f32 {result_reg}, %f8, {result_reg}, %p2;",
            f"    selp.f32 {result_reg}, %f10, {result_reg}, %p1;",
            f"    neg.f32 %f8, {result_reg};",
            f"    setp.lt.f32 %p3, {y_reg}, %f12;",
            f"    selp.f32 {result_reg}, %f8, {result_reg}, %p3;",
            f"    setp.eq.f32 %p2, {y_reg}, %f12;",
            "    and.pred %p0, %p1, %p2;",
            f"    selp.f32 {result_reg}, %f12, {result_reg}, %p0;",
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
            if op == "math_atan2":
                lines = ["    ld.global.f32 %f1, [%rd4];"]
                if scalar_mode == "tensor":
                    lines.append("    ld.global.f32 %f2, [%rd2];")
                lines.extend(
                    [
                        *self._atan2_f32_core_lines(y_reg="%f1", x_reg="%f2", result_reg="%f3"),
                        "    st.global.f32 [%rd5], %f3;",
                    ]
                )
                return lines
            instr = self._binary_instr(dtype, op)
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
            instr = self._binary_instr(dtype, op)
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

    def _compare_lines(self, dtype: str, op: str, *, lhs_ptr: str, rhs_ptr: str, dst_ptr: str) -> list[str]:
        instr = self._compare_instr(dtype, op)
        if dtype == "f32":
            return [
                f"    ld.global.f32 %f1, [{lhs_ptr}];",
                f"    ld.global.f32 %f2, [{rhs_ptr}];",
                f"    {instr} %p2, %f1, %f2;",
                "    selp.u32 %r5, 1, 0, %p2;",
                f"    st.global.u8 [{dst_ptr}], %r5;",
            ]
        if dtype == "i32":
            return [
                f"    ld.global.s32 %r5, [{lhs_ptr}];",
                f"    ld.global.s32 %r2, [{rhs_ptr}];",
                f"    {instr} %p2, %r5, %r2;",
                "    selp.u32 %r5, 1, 0, %p2;",
                f"    st.global.u8 [{dst_ptr}], %r5;",
            ]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _scalar_compare_lines(self, dtype: str, op: str, *, scalar_mode: str) -> list[str]:
        instr = self._compare_instr(dtype, op)
        if dtype == "f32":
            lines = ["    ld.global.f32 %f1, [%rd4];"]
            if scalar_mode == "tensor":
                lines.append("    ld.global.f32 %f2, [%rd2];")
            lines.extend(
                [
                    f"    {instr} %p2, %f1, %f2;",
                    "    selp.u32 %r5, 1, 0, %p2;",
                    "    st.global.u8 [%rd5], %r5;",
                ]
            )
            return lines
        if dtype == "i32":
            lines = ["    ld.global.s32 %r2, [%rd4];"]
            if scalar_mode == "tensor":
                lines.append("    ld.global.s32 %r3, [%rd2];")
            lines.extend(
                [
                    f"    {instr} %p2, %r2, %r3;",
                    "    selp.u32 %r5, 1, 0, %p2;",
                    "    st.global.u8 [%rd5], %r5;",
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
            instr = self._reduce_bundle_instr(dtype, op)
            return [
                "    ld.global.f32 %f2, [%rd4];",
                f"    {instr} %f1, %f1, %f2;",
            ]
        if dtype == "i32":
            instr = self._reduce_bundle_instr(dtype, op)
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
            self._supports_parallel_reduce(match.dtype, match.op)
            and ir.launch.grid == (1, 1, 1)
            and block_y == 1
            and block_z == 1
            and block_x > 1
            and block_x <= 1024
            and (block_x & (block_x - 1)) == 0
        )

    def _supports_reduce(self, dtype: str, op: str) -> bool:
        try:
            self._reduce_bundle_instr(dtype, op)
        except CompilationError:
            return False
        return True

    def _supports_parallel_reduce(self, dtype: str, op: str) -> bool:
        if op in {"reduce_add", "reduce_mul", "reduce_max", "reduce_min"}:
            return self._supports_reduce(dtype, op)
        if dtype == "i32" and op in {"reduce_and", "reduce_or", "reduce_xor"}:
            return True
        return False

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
                "reduce_and": "and.b32",
                "reduce_or": "or.b32",
                "reduce_xor": "xor.b32",
            }[op]
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _reduce_bundle_init_value(self, dtype: str, value: float | int) -> float | int:
        if dtype == "f32":
            return float(value)
        if dtype == "i32":
            return int(value)
        raise CompilationError(f"unsupported PTX dtype '{dtype}'")

    def _supports_elementwise_dtype(self, dtype: str) -> bool:
        return dtype in self._SUPPORTED_ELEMENTWISE_DTYPES

    def _element_size(self, dtype: str) -> int:
        try:
            return self._SUPPORTED_ELEMENTWISE_DTYPES[dtype]
        except KeyError as exc:
            raise CompilationError(f"unsupported PTX elementwise dtype '{dtype}'") from exc

    def _supports_rank1_unary(self, dtype: str, op: str) -> bool:
        if op in {"neg", "abs"}:
            return dtype in {"f32", "i32"}
        if op in {"math_round", "math_floor", "math_ceil", "math_trunc", "math_sqrt", "math_rsqrt", "math_sin", "math_cos", "math_acos", "math_asin", "math_atan", "math_exp", "math_exp2", "math_log", "math_log2", "math_log10", "math_erf"}:
            return dtype == "f32"
        if op == "bitnot":
            return dtype in {"i32", "i1"}
        return False

    def _normalize_tensor_unary_op(self, op: str) -> str | None:
        if op == "tensor_neg":
            return "neg"
        if op == "tensor_abs":
            return "abs"
        if op in {"math_round", "math_floor", "math_ceil", "math_trunc", "math_sqrt", "math_rsqrt", "math_sin", "math_cos", "math_acos", "math_asin", "math_atan", "math_exp", "math_exp2", "math_log", "math_log2", "math_log10", "math_erf"}:
            return op
        if op == "tensor_bitnot":
            return "bitnot"
        return None

    def _unary_f32_reg_decl(self, op: str) -> str:
        if op in {"math_acos", "math_asin"}:
            return "    .reg .f32 %f<13>;"
        if op == "math_erf":
            return "    .reg .f32 %f<7>;"
        if op == "math_atan":
            return "    .reg .f32 %f<6>;"
        return "    .reg .f32 %f<3>;"

    def _unary_pred_decl(self, op: str) -> str:
        return "    .reg .pred %p<4>;" if op in {"math_acos", "math_asin"} else "    .reg .pred %p<2>;"

    def _binary_pred_decl(self, op: str) -> str:
        return "    .reg .pred %p<4>;" if op == "math_atan2" else "    .reg .pred %p<2>;"

    def _binary_f32_reg_decl(self, op: str) -> str:
        return "    .reg .f32 %f<13>;" if op == "math_atan2" else "    .reg .f32 %f<4>;"

    def _supports_binary(self, dtype: str, op: str) -> bool:
        if dtype == "f32" and op == "math_atan2":
            return True
        try:
            self._binary_instr(dtype, op)
        except CompilationError:
            return False
        return True

    def _supports_compare(self, dtype: str, op: str) -> bool:
        try:
            self._compare_instr(dtype, op)
        except CompilationError:
            return False
        return True

    def _binary_instr(self, dtype: str, op: str) -> str:
        if dtype == "f32":
            table = {
                "add": "add.rn.f32",
                "sub": "sub.rn.f32",
                "mul": "mul.rn.f32",
                "div": "div.rn.f32",
                "max": "max.f32",
                "min": "min.f32",
            }
        elif dtype == "i32":
            table = {
                "add": "add.s32",
                "sub": "sub.s32",
                "mul": "mul.lo.s32",
                "div": "div.s32",
                "max": "max.s32",
                "min": "min.s32",
                "bitand": "and.b32",
                "bitor": "or.b32",
                "bitxor": "xor.b32",
            }
        elif dtype == "i1":
            table = {
                "and": "and.b32",
                "or": "or.b32",
                "xor": "xor.b32",
                "bitand": "and.b32",
                "bitor": "or.b32",
                "bitxor": "xor.b32",
            }
        else:
            raise CompilationError(f"unsupported PTX dtype '{dtype}'")
        try:
            return table[op]
        except KeyError as exc:
            raise CompilationError(
                f"unsupported PTX binary op '{op}' for dtype '{dtype}'"
            ) from exc

    def _compare_instr(self, dtype: str, op: str) -> str:
        if dtype == "f32":
            table = {
                "cmp_lt": "setp.lt.f32",
                "cmp_le": "setp.le.f32",
                "cmp_gt": "setp.gt.f32",
                "cmp_ge": "setp.ge.f32",
                "cmp_eq": "setp.eq.f32",
                "cmp_ne": "setp.ne.f32",
            }
        elif dtype == "i32":
            table = {
                "cmp_lt": "setp.lt.s32",
                "cmp_le": "setp.le.s32",
                "cmp_gt": "setp.gt.s32",
                "cmp_ge": "setp.ge.s32",
                "cmp_eq": "setp.eq.s32",
                "cmp_ne": "setp.ne.s32",
            }
        else:
            raise CompilationError(f"unsupported PTX dtype '{dtype}'")
        try:
            return table[op]
        except KeyError as exc:
            raise CompilationError(
                f"unsupported PTX compare op '{op}' for dtype '{dtype}'"
            ) from exc

    def _supports_copy_reduce(self, dtype: str, reduction: str) -> bool:
        try:
            self._copy_reduce_instr(dtype, reduction)
        except CompilationError:
            return False
        return True

    def _copy_reduce_instr(self, dtype: str, reduction: str) -> str:
        if dtype == "f32":
            table = {
                "add": "add.rn.f32",
                "max": "max.f32",
                "min": "min.f32",
            }
        elif dtype == "i32":
            table = {
                "add": "add.s32",
                "max": "max.s32",
                "min": "min.s32",
                "and": "and.b32",
                "or": "or.b32",
                "xor": "xor.b32",
            }
        else:
            raise CompilationError(f"unsupported PTX dtype '{dtype}'")
        try:
            return table[reduction]
        except KeyError as exc:
            raise CompilationError(
                f"unsupported PTX copy_reduce reduction '{reduction}' for dtype '{dtype}'"
            ) from exc

    def _float_immediate(self, value: float) -> str:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return f"0f{bits:08X}"

    def _is_broadcast_compatible(self, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> bool:
        return all(src_dim == dst_dim or src_dim == 1 for src_dim, dst_dim in zip(src_shape, dst_shape))

    def _require_extent1_tensor(self, spec: object) -> TensorSpec | None:
        if not isinstance(spec, TensorSpec) or spec.shape != (1,):
            return None
        return spec

    def _scalar_param_type(self, dtype: str) -> str:
        table = {
            "f32": ".f32",
            "i1": ".u8",
            "i32": ".s32",
        }
        try:
            return table[dtype]
        except KeyError as exc:
            raise CompilationError(f"unsupported PTX scalar dtype '{dtype}'") from exc
