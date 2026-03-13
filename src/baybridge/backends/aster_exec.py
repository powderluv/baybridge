from __future__ import annotations

import ctypes
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..hip_runtime import load_hip_library
from ..ir import PortableKernelIR, TensorSpec
from ..runtime import RuntimeTensor, TensorHandle, tensor
from ..target import AMDTarget
from .aster_bridge import AsterBridge

_COPY_1D_TEMPLATE = """// baybridge.aster_exec
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!tensor_position_descriptor_1d = !aster_utils.struct<ptr: !sx2, pos: index, stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!m2reg_param_1d_vx4 = !aster_utils.struct<i: index, memref: memref<?x!vx4>>

amdgcn.module @mod target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  func.func private @distributed_index_1d() -> index
  func.func private @grid_stride_1d() -> index
  func.func private @store_to_global_dwordx4_wait(!vx4, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dwordx4_wait(!tensor_position_descriptor_2d) -> !vx4

  func.func private @global_load_body(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %m2reg_param: !m2reg_param_1d_vx4
  ) {{
    %idx, %memref = aster_utils.struct_extract %m2reg_param["i", "memref"] : !m2reg_param_1d_vx4 -> index, memref<?x!vx4>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %loaded = func.call @load_from_global_dwordx4_wait(%pos_desc_2d) : (!tensor_position_descriptor_2d) -> !vx4
    memref.store %loaded, %memref[%idx] : memref<?x!vx4>
    return
  }}

  func.func private @global_store_body(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %m2reg_param: !m2reg_param_1d_vx4
  ) {{
    %idx, %memref = aster_utils.struct_extract %m2reg_param["i", "memref"] : !m2reg_param_1d_vx4 -> index, memref<?x!vx4>
    %value = memref.load %memref[%idx] : memref<?x!vx4>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dwordx4_wait(%value, %pos_desc_2d) : (!vx4, !tensor_position_descriptor_2d) -> ()
    return
  }}

  func.func private @copy_loop(
    %num_elements: index,
    %src_global: !sx2,
    %dst_global: !sx2
  ) {{
    %memref = memref.alloca(%num_elements) : memref<?x!vx4>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %base_index = func.call @distributed_index_1d() : () -> index
    %grid_stride = func.call @grid_stride_1d() : () -> index
    %grid_stride_bytes = affine.apply affine_map<(s)[elt] -> (s * elt)>(%grid_stride)[%c16]

    scf.for %i = %c0 to %num_elements step %c1 {{
      %elem_index = affine.apply affine_map<(base, i)[stride] -> (base + i * stride)>
        (%base_index, %i)[%grid_stride]
      %src_pos_desc = aster_utils.struct_create(%src_global, %elem_index, %grid_stride_bytes, %c16) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %dst_pos_desc = aster_utils.struct_create(%dst_global, %elem_index, %grid_stride_bytes, %c16) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %m2reg_param = aster_utils.struct_create(%i, %memref) : (index, memref<?x!vx4>) -> !m2reg_param_1d_vx4
      func.call @global_load_body(%src_pos_desc, %m2reg_param)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, !m2reg_param_1d_vx4) -> ()
      func.call @global_store_body(%dst_pos_desc, %m2reg_param)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, !m2reg_param_1d_vx4) -> ()
    }} {{sched.dims = array<i64: {vector_chunks}>}}

    return
  }}

  amdgcn.kernel @{kernel_name} arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {{shared_memory_size = 0 : i32}} {{
    %src_ptr_s = amdgcn.load_arg 0 : !sx2
    %dst_ptr_s = amdgcn.load_arg 1 : !sx2
    %src_ptr, %dst_ptr = lsir.assume_noalias %src_ptr_s, %dst_ptr_s : (!sx2, !sx2) -> (!sx2, !sx2)
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %num_elements = arith.constant {vector_chunks} : index
    func.call @copy_loop(%num_elements, %src_ptr, %dst_ptr) : (index, !sx2, !sx2) -> ()
    amdgcn.end_kernel
  }}
}}
"""

_BINARY_1D_TEMPLATE = """// baybridge.aster_exec
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!tensor_position_descriptor_1d = !aster_utils.struct<ptr: !sx2, pos: index, stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!m2reg_param_1d_vx4 = !aster_utils.struct<i: index, memref: memref<?x!vx4>>

amdgcn.module @mod target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  func.func private @distributed_index_1d() -> index
  func.func private @grid_stride_1d() -> index
  func.func private @store_to_global_dwordx4_wait(!vx4, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dwordx4_wait(!tensor_position_descriptor_2d) -> !vx4

  func.func private @global_load_body(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %m2reg_param: !m2reg_param_1d_vx4
  ) {{
    %idx, %memref = aster_utils.struct_extract %m2reg_param["i", "memref"] : !m2reg_param_1d_vx4 -> index, memref<?x!vx4>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %loaded = func.call @load_from_global_dwordx4_wait(%pos_desc_2d) : (!tensor_position_descriptor_2d) -> !vx4
    memref.store %loaded, %memref[%idx] : memref<?x!vx4>
    return
  }}

  func.func private @global_store_body(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %m2reg_param: !m2reg_param_1d_vx4
  ) {{
    %idx, %memref = aster_utils.struct_extract %m2reg_param["i", "memref"] : !m2reg_param_1d_vx4 -> index, memref<?x!vx4>
    %value = memref.load %memref[%idx] : memref<?x!vx4>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dwordx4_wait(%value, %pos_desc_2d) : (!vx4, !tensor_position_descriptor_2d) -> ()
    return
  }}

  func.func private @vector_{binary_name}_body(
    %idx: index,
    %lhs_memref: memref<?x!vx4>,
    %rhs_memref: memref<?x!vx4>,
    %dst_memref: memref<?x!vx4>
  ) {{
    %lhs = memref.load %lhs_memref[%idx] : memref<?x!vx4>
    %rhs = memref.load %rhs_memref[%idx] : memref<?x!vx4>
    %lhs0, %lhs1, %lhs2, %lhs3 = amdgcn.split_register_range %lhs : !vx4
    %rhs0, %rhs1, %rhs2, %rhs3 = amdgcn.split_register_range %rhs : !vx4
    %tmp0 = amdgcn.alloca : !amdgcn.vgpr
    %tmp1 = amdgcn.alloca : !amdgcn.vgpr
    %tmp2 = amdgcn.alloca : !amdgcn.vgpr
    %tmp3 = amdgcn.alloca : !amdgcn.vgpr
    %result0 = lsir.{lsir_op} {value_type} %tmp0, %lhs0, %rhs0 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %result1 = lsir.{lsir_op} {value_type} %tmp1, %lhs1, %rhs1 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %result2 = lsir.{lsir_op} {value_type} %tmp2, %lhs2, %rhs2 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %result3 = lsir.{lsir_op} {value_type} %tmp3, %lhs3, %rhs3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %result = amdgcn.make_register_range %result0, %result1, %result2, %result3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    memref.store %result, %dst_memref[%idx] : memref<?x!vx4>
    return
  }}

  func.func private @{binary_name}_loop(
    %num_elements: index,
    %lhs_global: !sx2,
    %rhs_global: !sx2,
    %dst_global: !sx2
  ) {{
    %lhs_memref = memref.alloca(%num_elements) : memref<?x!vx4>
    %rhs_memref = memref.alloca(%num_elements) : memref<?x!vx4>
    %dst_memref = memref.alloca(%num_elements) : memref<?x!vx4>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %base_index = func.call @distributed_index_1d() : () -> index
    %grid_stride = func.call @grid_stride_1d() : () -> index
    %grid_stride_bytes = affine.apply affine_map<(s)[elt] -> (s * elt)>(%grid_stride)[%c16]

    scf.for %i = %c0 to %num_elements step %c1 {{
      %elem_index = affine.apply affine_map<(base, i)[stride] -> (base + i * stride)>
        (%base_index, %i)[%grid_stride]
      %lhs_pos_desc = aster_utils.struct_create(%lhs_global, %elem_index, %grid_stride_bytes, %c16) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %rhs_pos_desc = aster_utils.struct_create(%rhs_global, %elem_index, %grid_stride_bytes, %c16) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %dst_pos_desc = aster_utils.struct_create(%dst_global, %elem_index, %grid_stride_bytes, %c16) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %lhs_param = aster_utils.struct_create(%i, %lhs_memref) : (index, memref<?x!vx4>) -> !m2reg_param_1d_vx4
      %rhs_param = aster_utils.struct_create(%i, %rhs_memref) : (index, memref<?x!vx4>) -> !m2reg_param_1d_vx4
      %dst_param = aster_utils.struct_create(%i, %dst_memref) : (index, memref<?x!vx4>) -> !m2reg_param_1d_vx4
      func.call @global_load_body(%lhs_pos_desc, %lhs_param)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, !m2reg_param_1d_vx4) -> ()
      func.call @global_load_body(%rhs_pos_desc, %rhs_param)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, !m2reg_param_1d_vx4) -> ()
      func.call @vector_{binary_name}_body(%i, %lhs_memref, %rhs_memref, %dst_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (index, memref<?x!vx4>, memref<?x!vx4>, memref<?x!vx4>) -> ()
      func.call @global_store_body(%dst_pos_desc, %dst_param)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, !m2reg_param_1d_vx4) -> ()
    }} {{sched.dims = array<i64: {vector_chunks}>}}

    return
  }}

  amdgcn.kernel @{kernel_name} arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {{shared_memory_size = 0 : i32}} {{
    %lhs_ptr_s = amdgcn.load_arg 0 : !sx2
    %rhs_ptr_s = amdgcn.load_arg 1 : !sx2
    %dst_ptr_s = amdgcn.load_arg 2 : !sx2
    %lhs_ptr, %rhs_ptr, %dst_ptr = lsir.assume_noalias %lhs_ptr_s, %rhs_ptr_s, %dst_ptr_s : (!sx2, !sx2, !sx2) -> (!sx2, !sx2, !sx2)
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %num_elements = arith.constant {vector_chunks} : index
    func.call @{binary_name}_loop(%num_elements, %lhs_ptr, %rhs_ptr, %dst_ptr) : (index, !sx2, !sx2, !sx2) -> ()
    amdgcn.end_kernel
  }}
}}
"""


@dataclass(frozen=True)
class _Copy1DMatch:
    src_name: str
    dst_name: str
    dtype: str
    shape: tuple[int, ...]
    vector_chunks: int
    elements_per_chunk: int


@dataclass(frozen=True)
class _Binary1DMatch:
    lhs_name: str
    rhs_name: str
    dst_name: str
    dtype: str
    shape: tuple[int, ...]
    vector_chunks: int
    op: str
    binary_name: str
    lsir_op: str
    value_type: str


class AsterExecBackend:
    name = "aster_exec"
    artifact_extension = ".aster.mlir"

    def __init__(self) -> None:
        self._bridge = AsterBridge()

    def available(self, target: AMDTarget | None = None) -> bool:
        del target
        environment = self._bridge.environment()
        return environment.python_package_root is not None and self._bridge.configured_root() is not None

    def supports_inputs(self, values: tuple[Any, ...]) -> bool:
        if not values:
            return False
        return all(isinstance(value, (RuntimeTensor, TensorHandle)) for value in values)

    def supports_auto_selection(self, ir: PortableKernelIR, target: AMDTarget, values: tuple[Any, ...]) -> bool:
        if not self.available(target):
            return False
        if not self.supports_inputs(values):
            return False
        return self.supports(ir, target)

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        try:
            self._match_kernel(ir, target)
        except BackendNotImplementedError:
            return False
        return True

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        match = self._match_kernel(ir, target)
        if isinstance(match, _Copy1DMatch):
            text = self._render_copy_1d(match, ir.name, target)
        else:
            text = self._render_binary_1d(match, ir.name, target)
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="aster_exec_mlir",
            text=text,
        )

    def build_launcher(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        source_path: Path,
    ):
        match = self._match_kernel(ir, target)
        configured_root = self._bridge.configured_root()
        if configured_root is None:
            raise BackendNotImplementedError(
                "aster_exec requires BAYBRIDGE_ASTER_ROOT to point at an ASTER source checkout"
            )
        hsaco_path = source_path.with_suffix(".hsaco")
        state: dict[str, Any] = {}

        def launcher(*args: Any, **kwargs: Any) -> None:
            stream = kwargs.pop("stream", None)
            if stream is not None:
                raise TypeError("aster_exec does not support stream= yet")
            if kwargs:
                raise TypeError("aster_exec launcher only supports positional arguments")
            if len(args) != len(ir.arguments):
                raise TypeError(f"{ir.name} expects {len(ir.arguments)} arguments, got {len(args)}")
            modules = self._load_aster_modules()
            if not hsaco_path.exists():
                self._compile_to_hsaco(
                    modules,
                    source_path,
                    hsaco_path,
                    target,
                    configured_root,
                    ir.name,
                )
            if isinstance(match, _Copy1DMatch):
                input_values = [
                    self._as_runtime_tensor(args[0], expected_dtype=match.dtype, argument_name=match.src_name),
                ]
                dst_owner, dst_value = self._prepare_output(
                    args[1],
                    expected_dtype=match.dtype,
                    argument_name=match.dst_name,
                )
            else:
                input_values = [
                    self._as_runtime_tensor(args[0], expected_dtype=match.dtype, argument_name=match.lhs_name),
                    self._as_runtime_tensor(args[1], expected_dtype=match.dtype, argument_name=match.rhs_name),
                ]
                dst_owner, dst_value = self._prepare_output(
                    args[2],
                    expected_dtype=match.dtype,
                    argument_name=match.dst_name,
                )
            try:
                numpy = importlib.import_module("numpy")
            except ModuleNotFoundError:
                numpy = None
            if numpy is None:
                dst_array = dst_value.tolist()
                input_arrays = [input_value.tolist() for input_value in input_values]
            else:
                dtype = match.dtype
                input_arrays = [
                    numpy.array(input_value.tolist(), dtype=self._numpy_dtype(dtype)) for input_value in input_values
                ]
                dst_array = numpy.array(dst_value.tolist(), dtype=self._numpy_dtype(dtype))
            modules["hip"].execute_hsaco(
                str(hsaco_path),
                ir.name,
                input_arrays,
                [dst_array],
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
                num_iterations=1,
            )
            copied_data = dst_array.tolist() if hasattr(dst_array, "tolist") else dst_array
            dtype = match.dtype
            copied = tensor(copied_data, dtype=dtype)
            self._copy_back(dst_owner, copied)

        return launcher

    def emit_bundle(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        lowered_path: Path,
    ) -> Path:
        return self._bridge.write_repro_bundle(ir, target, lowered_module, lowered_path)

    def _match_kernel(self, ir: PortableKernelIR, target: AMDTarget) -> _Copy1DMatch | _Binary1DMatch:
        if len(ir.arguments) == 2:
            return self._match_copy_1d(ir, target)
        if len(ir.arguments) == 3:
            return self._match_binary_1d(ir, target)
        try:
            return self._match_copy_1d(ir, target)
        except BackendNotImplementedError:
            return self._match_binary_1d(ir, target)

    def _match_copy_1d(self, ir: PortableKernelIR, target: AMDTarget) -> _Copy1DMatch:
        if target.arch not in {"gfx942", "gfx950"}:
            raise BackendNotImplementedError(f"aster_exec does not support target arch '{target.arch}'")
        if target.wave_size != 64:
            raise BackendNotImplementedError("aster_exec currently requires wave_size=64")
        if len(ir.arguments) != 2:
            raise BackendNotImplementedError("aster_exec currently supports exactly two tensor arguments")
        if len(ir.operations) != 1 or ir.operations[0].op != "copy":
            raise BackendNotImplementedError("aster_exec currently supports a single copy operation only")
        src_arg, dst_arg = ir.arguments
        if not isinstance(src_arg.spec, TensorSpec) or not isinstance(dst_arg.spec, TensorSpec):
            raise BackendNotImplementedError("aster_exec currently supports tensor arguments only")
        src_spec = src_arg.spec
        dst_spec = dst_arg.spec
        if src_spec.address_space.value != "global" or dst_spec.address_space.value != "global":
            raise BackendNotImplementedError("aster_exec currently requires global-memory tensors")
        if src_spec.shape != dst_spec.shape or src_spec.dtype != dst_spec.dtype:
            raise BackendNotImplementedError("aster_exec requires matching source and destination tensor specs")
        if len(src_spec.shape) != 1:
            raise BackendNotImplementedError("aster_exec currently supports 1D tensors only")
        if src_spec.dtype not in {"f32", "i32", "f16"}:
            raise BackendNotImplementedError("aster_exec currently supports f32, i32, and f16 copies only")
        if src_spec.resolved_layout().stride != (1,) or dst_spec.resolved_layout().stride != (1,):
            raise BackendNotImplementedError("aster_exec currently requires contiguous 1D tensors")
        element_count = src_spec.shape[0]
        elements_per_chunk = 16 // self._dtype_size_bytes(src_spec.dtype)
        if element_count <= 0 or element_count % elements_per_chunk != 0:
            raise BackendNotImplementedError(
                "aster_exec requires a 1D element count aligned to a 16-byte transfer chunk"
            )
        return _Copy1DMatch(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            shape=src_spec.shape,
            vector_chunks=element_count // elements_per_chunk,
            elements_per_chunk=elements_per_chunk,
        )

    def _render_copy_1d(self, match: _Copy1DMatch, kernel_name: str, target: AMDTarget) -> str:
        return _COPY_1D_TEMPLATE.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            vector_chunks=match.vector_chunks,
        )

    def _match_binary_1d(self, ir: PortableKernelIR, target: AMDTarget) -> _Binary1DMatch:
        if target.arch not in {"gfx942", "gfx950"}:
            raise BackendNotImplementedError(f"aster_exec does not support target arch '{target.arch}'")
        if target.wave_size != 64:
            raise BackendNotImplementedError("aster_exec currently requires wave_size=64")
        if len(ir.arguments) != 3:
            raise BackendNotImplementedError("aster_exec currently supports exactly three tensor arguments for pointwise binary ops")
        op_names = [operation.op for operation in ir.operations]
        supported_ops = {
            "tensor_add": ("add", "addf"),
            "tensor_sub": ("sub", "subf"),
            "tensor_mul": ("mul", "mulf"),
            "tensor_div": ("div", "divf"),
        }
        if op_names[:4] != ["make_tensor", "copy", "make_tensor", "copy"] or op_names[-1:] != ["copy"]:
            raise BackendNotImplementedError(
                "aster_exec currently supports a single supported tensor binary pipeline only"
            )
        binary_op = op_names[4] if len(op_names) == 6 else None
        if binary_op not in supported_ops:
            raise BackendNotImplementedError(
                "aster_exec currently supports a single supported tensor binary pipeline only"
            )
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            raise BackendNotImplementedError("aster_exec currently supports tensor arguments only")
        lhs_spec = lhs_arg.spec
        rhs_spec = rhs_arg.spec
        dst_spec = dst_arg.spec
        if lhs_spec.address_space.value != "global" or rhs_spec.address_space.value != "global" or dst_spec.address_space.value != "global":
            raise BackendNotImplementedError("aster_exec add currently requires global-memory tensors")
        if lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            raise BackendNotImplementedError("aster_exec add requires matching tensor shapes")
        if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != dst_spec.dtype:
            raise BackendNotImplementedError("aster_exec pointwise binary ops require matching tensor dtypes")
        if len(lhs_spec.shape) != 1:
            raise BackendNotImplementedError("aster_exec add currently supports 1D tensors only")
        if (
            lhs_spec.resolved_layout().stride != (1,)
            or rhs_spec.resolved_layout().stride != (1,)
            or dst_spec.resolved_layout().stride != (1,)
        ):
            raise BackendNotImplementedError("aster_exec add currently requires contiguous 1D tensors")
        element_count = lhs_spec.shape[0]
        if element_count <= 0 or element_count % 4 != 0:
            raise BackendNotImplementedError("aster_exec pointwise binary ops require a 1D element count that is a positive multiple of 4")
        dtype_support = {
            "f32": {
                "tensor_add": ("add", "addf", "f32"),
                "tensor_sub": ("sub", "subf", "f32"),
                "tensor_mul": ("mul", "mulf", "f32"),
                "tensor_div": ("div", "divf", "f32"),
            },
            "i32": {
                "tensor_add": ("add", "addi", "i32"),
                "tensor_sub": ("sub", "subi", "i32"),
                "tensor_mul": ("mul", "muli", "i32"),
                "tensor_div": ("div", "divsi", "i32"),
            },
        }
        supported_for_dtype = dtype_support.get(lhs_spec.dtype)
        if supported_for_dtype is None or binary_op not in supported_for_dtype:
            raise BackendNotImplementedError(
                "aster_exec currently supports f32 and i32 add/sub/mul/div only"
            )
        binary_name, lsir_op, value_type = supported_for_dtype[binary_op]
        return _Binary1DMatch(
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            shape=lhs_spec.shape,
            vector_chunks=element_count // 4,
            op=binary_op,
            binary_name=binary_name,
            lsir_op=lsir_op,
            value_type=value_type,
        )

    def _render_binary_1d(self, match: _Binary1DMatch, kernel_name: str, target: AMDTarget) -> str:
        return _BINARY_1D_TEMPLATE.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            vector_chunks=match.vector_chunks,
            binary_name=match.binary_name,
            lsir_op=match.lsir_op,
            value_type=match.value_type,
        )

    def _target_isa(self, target: AMDTarget) -> str:
        if target.arch == "gfx950":
            return "cdna4"
        if target.arch in {"gfx940", "gfx942"}:
            return "cdna3"
        raise BackendNotImplementedError(f"aster_exec does not know the ASTER ISA for '{target.arch}'")

    def _load_aster_modules(self) -> dict[str, Any]:
        environment = self._bridge.environment()
        python_package_root = environment.python_package_root
        if python_package_root is None:
            raise BackendNotImplementedError("aster_exec requires an importable ASTER python package")
        load_hip_library(global_scope=True)
        package_parent = str(Path(python_package_root).parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)
        return {
            "ir": importlib.import_module("aster.ir"),
            "utils": importlib.import_module("aster.utils"),
            "testing": importlib.import_module("aster.testing"),
            "pipelines": importlib.import_module("aster.pass_pipelines"),
            "hip": importlib.import_module("aster.hip"),
        }

    def _compile_to_hsaco(
        self,
        modules: dict[str, Any],
        source_path: Path,
        hsaco_path: Path,
        target: AMDTarget,
        configured_root: Path,
        kernel_name: str,
    ) -> None:
        library_root = configured_root / "mlir_kernels" / "library" / "common"
        library_paths = [
            str(library_root / "register-init.mlir"),
            str(library_root / "indexing.mlir"),
            str(library_root / "copies.mlir"),
        ]
        with modules["ir"].Context() as ctx:
            asm, _ = modules["testing"].compile_mlir_file_to_asm(
                str(source_path),
                kernel_name,
                modules["pipelines"].get_pass_pipeline("default"),
                ctx,
                library_paths=library_paths,
            )
        compiled = modules["utils"].assemble_to_hsaco(
            asm,
            target=target.arch,
            wavefront_size=target.wave_size,
            output_path=str(hsaco_path),
        )
        if compiled is None:
            raise RuntimeError("ASTER failed to assemble the Baybridge kernel to HSACO")

    def _as_runtime_tensor(self, value: Any, *, expected_dtype: str, argument_name: str) -> RuntimeTensor:
        if isinstance(value, RuntimeTensor):
            tensor_value = value
        elif isinstance(value, TensorHandle):
            tensor_value = value.to_runtime_tensor()
        else:
            raise TypeError(
                f"aster_exec expects RuntimeTensor or TensorHandle values for argument '{argument_name}', got {type(value).__name__}"
            )
        if tensor_value.dtype != expected_dtype:
            raise TypeError(
                f"aster_exec expected dtype '{expected_dtype}' for argument '{argument_name}', got '{tensor_value.dtype}'"
            )
        return tensor_value

    def _prepare_output(
        self,
        value: Any,
        *,
        expected_dtype: str,
        argument_name: str,
    ) -> tuple[RuntimeTensor | TensorHandle, RuntimeTensor]:
        if isinstance(value, RuntimeTensor):
            return value, value
        if isinstance(value, TensorHandle):
            return value, value.to_runtime_tensor()
        raise TypeError(
            f"aster_exec expects RuntimeTensor or TensorHandle values for argument '{argument_name}', got {type(value).__name__}"
        )

    def _copy_back(self, owner: RuntimeTensor | TensorHandle, result: RuntimeTensor) -> None:
        if isinstance(owner, RuntimeTensor):
            owner.store(result)
            return
        owner.copy_from_runtime_tensor(result)

    def _numpy_dtype(self, dtype: str) -> str:
        if dtype == "f32":
            return "float32"
        if dtype == "f16":
            return "float16"
        if dtype == "i32":
            return "int32"
        raise BackendNotImplementedError(f"aster_exec does not support dtype '{dtype}'")

    def _dtype_size_bytes(self, dtype: str) -> int:
        if dtype in {"f32", "i32"}:
            return 4
        if dtype == "f16":
            return 2
        raise BackendNotImplementedError(f"aster_exec does not support dtype '{dtype}'")
