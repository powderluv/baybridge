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

_SCALAR_COPY_TEMPLATE = """// baybridge.aster_exec
!s = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!v = !amdgcn.vgpr
!tensor_position_descriptor_1d = !aster_utils.struct<ptr: !sx2, pos: index, stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

amdgcn.module @mod target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  func.func private @distributed_index_1d() -> index
  func.func private @grid_stride_1d() -> index
  func.func private @store_to_global_dword_wait(!v, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dword_wait(!tensor_position_descriptor_2d) -> !v

  func.func private @global_load_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %loaded = func.call @load_from_global_dword_wait(%pos_desc_2d) : (!tensor_position_descriptor_2d) -> !v
    memref.store %loaded, %memref[%idx] : memref<?x!v>
    return
  }}

  func.func private @global_store_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %value = memref.load %memref[%idx] : memref<?x!v>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dword_wait(%value, %pos_desc_2d) : (!v, !tensor_position_descriptor_2d) -> ()
    return
  }}

  func.func private @copy_scalar_loop(
    %num_elements: index,
    %src_global: !sx2,
    %dst_global: !sx2
  ) {{
    %memref = memref.alloca(%num_elements) : memref<?x!v>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %base_index = func.call @distributed_index_1d() : () -> index
    %grid_stride = func.call @grid_stride_1d() : () -> index
    %grid_stride_bytes = affine.apply affine_map<(s)[elt] -> (s * elt)>(%grid_stride)[%c4]

    scf.for %i = %c0 to %num_elements step %c1 {{
      %elem_index = affine.apply affine_map<(base, i)[stride] -> (base + i * stride)>
        (%base_index, %i)[%grid_stride]
      %src_pos_desc = aster_utils.struct_create(%src_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %dst_pos_desc = aster_utils.struct_create(%dst_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      func.call @global_load_body_scalar(%src_pos_desc, %i, %memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
      func.call @global_store_body_scalar(%dst_pos_desc, %i, %memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
    }} {{sched.dims = array<i64: {element_count}>}}

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
    %num_elements = arith.constant {element_count} : index
    func.call @copy_scalar_loop(%num_elements, %src_ptr, %dst_ptr) : (index, !sx2, !sx2) -> ()
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
{op_line0}
{op_line1}
{op_line2}
{op_line3}
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

_SCALAR_BINARY_TEMPLATE = """// baybridge.aster_exec
!sx2 = !amdgcn.sgpr<[? + 2]>
!v = !amdgcn.vgpr
!tensor_position_descriptor_1d = !aster_utils.struct<ptr: !sx2, pos: index, stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

amdgcn.module @mod target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  func.func private @distributed_index_1d() -> index
  func.func private @grid_stride_1d() -> index
  func.func private @store_to_global_dword_wait(!v, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dword_wait(!tensor_position_descriptor_2d) -> !v

  func.func private @global_load_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %loaded = func.call @load_from_global_dword_wait(%pos_desc_2d) : (!tensor_position_descriptor_2d) -> !v
    memref.store %loaded, %memref[%idx] : memref<?x!v>
    return
  }}

  func.func private @global_store_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %value = memref.load %memref[%idx] : memref<?x!v>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dword_wait(%value, %pos_desc_2d) : (!v, !tensor_position_descriptor_2d) -> ()
    return
  }}

  func.func private @scalar_binary_body(
    %idx: index,
    %lhs_memref: memref<?x!v>,
    %rhs_memref: memref<?x!v>,
    %dst_memref: memref<?x!v>
  ) {{
    %lhs = memref.load %lhs_memref[%idx] : memref<?x!v>
    %rhs = memref.load %rhs_memref[%idx] : memref<?x!v>
    %tmp = amdgcn.alloca : !amdgcn.vgpr
{op_line}
    memref.store %result, %dst_memref[%idx] : memref<?x!v>
    return
  }}

  func.func private @binary_scalar_loop(
    %num_elements: index,
    %lhs_global: !sx2,
    %rhs_global: !sx2,
    %dst_global: !sx2
  ) {{
    %lhs_memref = memref.alloca(%num_elements) : memref<?x!v>
    %rhs_memref = memref.alloca(%num_elements) : memref<?x!v>
    %dst_memref = memref.alloca(%num_elements) : memref<?x!v>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %base_index = func.call @distributed_index_1d() : () -> index
    %grid_stride = func.call @grid_stride_1d() : () -> index
    %grid_stride_bytes = affine.apply affine_map<(s)[elt] -> (s * elt)>(%grid_stride)[%c4]

    scf.for %i = %c0 to %num_elements step %c1 {{
      %elem_index = affine.apply affine_map<(base, i)[stride] -> (base + i * stride)>
        (%base_index, %i)[%grid_stride]
      %lhs_pos_desc = aster_utils.struct_create(%lhs_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %rhs_pos_desc = aster_utils.struct_create(%rhs_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %dst_pos_desc = aster_utils.struct_create(%dst_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      func.call @global_load_body_scalar(%lhs_pos_desc, %i, %lhs_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
      func.call @global_load_body_scalar(%rhs_pos_desc, %i, %rhs_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
      func.call @scalar_binary_body(%i, %lhs_memref, %rhs_memref, %dst_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (index, memref<?x!v>, memref<?x!v>, memref<?x!v>) -> ()
      func.call @global_store_body_scalar(%dst_pos_desc, %i, %dst_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
    }} {{sched.dims = array<i64: {element_count}>}}

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
    %num_elements = arith.constant {element_count} : index
    func.call @binary_scalar_loop(%num_elements, %lhs_ptr, %rhs_ptr, %dst_ptr) : (index, !sx2, !sx2, !sx2) -> ()
    amdgcn.end_kernel
  }}
}}
"""

_SCALAR_BROADCAST_BINARY_TEMPLATE = """// baybridge.aster_exec
!sx2 = !amdgcn.sgpr<[? + 2]>
!v = !amdgcn.vgpr
!tensor_position_descriptor_1d = !aster_utils.struct<ptr: !sx2, pos: index, stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

amdgcn.module @mod target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  func.func private @distributed_index_1d() -> index
  func.func private @grid_stride_1d() -> index
  func.func private @store_to_global_dword_wait(!v, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dword_wait(!tensor_position_descriptor_2d) -> !v

  func.func private @global_load_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %loaded = func.call @load_from_global_dword_wait(%pos_desc_2d) : (!tensor_position_descriptor_2d) -> !v
    memref.store %loaded, %memref[%idx] : memref<?x!v>
    return
  }}

  func.func private @global_store_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %value = memref.load %memref[%idx] : memref<?x!v>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dword_wait(%value, %pos_desc_2d) : (!v, !tensor_position_descriptor_2d) -> ()
    return
  }}

  func.func private @scalar_broadcast_binary_body(
    %idx: index,
    %lhs_memref: memref<?x!v>,
    %rhs_scalar_memref: memref<?x!v>,
    %dst_memref: memref<?x!v>
  ) {{
    %c0 = arith.constant 0 : index
    %lhs = memref.load %lhs_memref[%idx] : memref<?x!v>
    %rhs = memref.load %rhs_scalar_memref[%c0] : memref<?x!v>
    %tmp = amdgcn.alloca : !amdgcn.vgpr
{op_line}
    memref.store %result, %dst_memref[%idx] : memref<?x!v>
    return
  }}

  func.func private @binary_broadcast_scalar_loop(
    %num_elements: index,
    %lhs_global: !sx2,
    %rhs_global: !sx2,
    %dst_global: !sx2
  ) {{
    %lhs_memref = memref.alloca(%num_elements) : memref<?x!v>
    %rhs_scalar_memref = memref.alloca(%num_elements) : memref<?x!v>
    %dst_memref = memref.alloca(%num_elements) : memref<?x!v>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %base_index = func.call @distributed_index_1d() : () -> index
    %grid_stride = func.call @grid_stride_1d() : () -> index
    %grid_stride_bytes = affine.apply affine_map<(s)[elt] -> (s * elt)>(%grid_stride)[%c4]
    %rhs_pos_desc = aster_utils.struct_create(%rhs_global, %c0, %c0, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
    func.call @global_load_body_scalar(%rhs_pos_desc, %c0, %rhs_scalar_memref)
      {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
      : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()

    scf.for %i = %c0 to %num_elements step %c1 {{
      %elem_index = affine.apply affine_map<(base, i)[stride] -> (base + i * stride)>
        (%base_index, %i)[%grid_stride]
      %lhs_pos_desc = aster_utils.struct_create(%lhs_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %dst_pos_desc = aster_utils.struct_create(%dst_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      func.call @global_load_body_scalar(%lhs_pos_desc, %i, %lhs_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
      func.call @scalar_broadcast_binary_body(%i, %lhs_memref, %rhs_scalar_memref, %dst_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (index, memref<?x!v>, memref<?x!v>, memref<?x!v>) -> ()
      func.call @global_store_body_scalar(%dst_pos_desc, %i, %dst_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
    }} {{sched.dims = array<i64: {element_count}>}}

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
    %num_elements = arith.constant {element_count} : index
    func.call @binary_broadcast_scalar_loop(%num_elements, %lhs_ptr, %rhs_ptr, %dst_ptr) : (index, !sx2, !sx2, !sx2) -> ()
    amdgcn.end_kernel
  }}
}}
"""

_SCALAR_LHS_BROADCAST_BINARY_TEMPLATE = """// baybridge.aster_exec
!sx2 = !amdgcn.sgpr<[? + 2]>
!v = !amdgcn.vgpr
!tensor_position_descriptor_1d = !aster_utils.struct<ptr: !sx2, pos: index, stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

amdgcn.module @mod target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  func.func private @distributed_index_1d() -> index
  func.func private @grid_stride_1d() -> index
  func.func private @store_to_global_dword_wait(!v, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dword_wait(!tensor_position_descriptor_2d) -> !v

  func.func private @global_load_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %loaded = func.call @load_from_global_dword_wait(%pos_desc_2d) : (!tensor_position_descriptor_2d) -> !v
    memref.store %loaded, %memref[%idx] : memref<?x!v>
    return
  }}

  func.func private @global_store_body_scalar(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %idx: index,
    %memref: memref<?x!v>
  ) {{
    %value = memref.load %memref[%idx] : memref<?x!v>
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dword_wait(%value, %pos_desc_2d) : (!v, !tensor_position_descriptor_2d) -> ()
    return
  }}

  func.func private @scalar_lhs_broadcast_binary_body(
    %idx: index,
    %lhs_scalar_memref: memref<?x!v>,
    %rhs_memref: memref<?x!v>,
    %dst_memref: memref<?x!v>
  ) {{
    %c0 = arith.constant 0 : index
    %lhs = memref.load %lhs_scalar_memref[%c0] : memref<?x!v>
    %rhs = memref.load %rhs_memref[%idx] : memref<?x!v>
    %tmp = amdgcn.alloca : !amdgcn.vgpr
{op_line}
    memref.store %result, %dst_memref[%idx] : memref<?x!v>
    return
  }}

  func.func private @binary_lhs_broadcast_scalar_loop(
    %num_elements: index,
    %lhs_global: !sx2,
    %rhs_global: !sx2,
    %dst_global: !sx2
  ) {{
    %lhs_scalar_memref = memref.alloca(%num_elements) : memref<?x!v>
    %rhs_memref = memref.alloca(%num_elements) : memref<?x!v>
    %dst_memref = memref.alloca(%num_elements) : memref<?x!v>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %base_index = func.call @distributed_index_1d() : () -> index
    %grid_stride = func.call @grid_stride_1d() : () -> index
    %grid_stride_bytes = affine.apply affine_map<(s)[elt] -> (s * elt)>(%grid_stride)[%c4]
    %lhs_pos_desc = aster_utils.struct_create(%lhs_global, %c0, %c0, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
    func.call @global_load_body_scalar(%lhs_pos_desc, %c0, %lhs_scalar_memref)
      {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
      : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()

    scf.for %i = %c0 to %num_elements step %c1 {{
      %elem_index = affine.apply affine_map<(base, i)[stride] -> (base + i * stride)>
        (%base_index, %i)[%grid_stride]
      %rhs_pos_desc = aster_utils.struct_create(%rhs_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %dst_pos_desc = aster_utils.struct_create(%dst_global, %elem_index, %grid_stride_bytes, %c4) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      func.call @global_load_body_scalar(%rhs_pos_desc, %i, %rhs_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
      func.call @scalar_lhs_broadcast_binary_body(%i, %lhs_scalar_memref, %rhs_memref, %dst_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (index, memref<?x!v>, memref<?x!v>, memref<?x!v>) -> ()
      func.call @global_store_body_scalar(%dst_pos_desc, %i, %dst_memref)
        {{sched.delay = 0 : i32, sched.rate = 1 : i32}}
        : (!tensor_position_descriptor_1d, index, memref<?x!v>) -> ()
    }} {{sched.dims = array<i64: {element_count}>}}

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
    %num_elements = arith.constant {element_count} : index
    func.call @binary_lhs_broadcast_scalar_loop(%num_elements, %lhs_ptr, %rhs_ptr, %dst_ptr) : (index, !sx2, !sx2, !sx2) -> ()
    amdgcn.end_kernel
  }}
}}
"""

_MFMA_16X16X16_TEMPLATE = """// baybridge.aster_exec
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @mod target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  func.func private @alloc_vgpr() -> !amdgcn.vgpr
  func.func private @alloc_vgprx2() -> (!amdgcn.vgpr<[? + 2]>)
  func.func private @init_vgprx4(%cst: i32) -> (!amdgcn.vgpr<[? + 4]>)

  amdgcn.kernel @{kernel_name} arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {{shared_memory_size = 1024 : i32}} {{
    %a_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %b_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %c_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %threadidx_x = amdgcn.thread_id x : !amdgcn.vgpr<0>

    %a_reg_range = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %b_reg_range = func.call @alloc_vgprx2() : () -> (!amdgcn.vgpr<[? + 2]>)
    %c0 = arith.constant 0 : i32
    %c_reg_range = func.call @init_vgprx4(%c0) : (i32) -> (!amdgcn.vgpr<[? + 4]>)

    %offset_a = amdgcn.alloca : !amdgcn.vgpr
    %c3 = arith.constant 3 : i32
    %thread_offset_f16 = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_a ins %c3, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32

    %loaded_a, %tok_load_a = amdgcn.load global_load_dwordx2 dest %a_reg_range addr %a_ptr offset d(%thread_offset_f16) + c(%c0_i32) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>
    %loaded_b, %tok_load_b = amdgcn.load global_load_dwordx2 dest %b_reg_range addr %b_ptr offset d(%thread_offset_f16) + c(%c0_i32) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %tok_ds_a = amdgcn.store ds_write_b64 data %loaded_a addr %thread_offset_f16 offset c(%c0_i32) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
    %tok_ds_b = amdgcn.store ds_write_b64 data %loaded_b addr %thread_offset_f16 offset c(%c512_i32) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %loaded_a_from_lds, %tok_lds_a = amdgcn.load ds_read_b64 dest %a_reg_range addr %thread_offset_f16 offset c(%c0_i32) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
    %loaded_b_from_lds, %tok_lds_b = amdgcn.load ds_read_b64 dest %b_reg_range addr %thread_offset_f16 offset c(%c512_i32) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c_mfma_result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<{mfma_inst}> %c_reg_range, %loaded_a_from_lds, %loaded_b_from_lds, %c_reg_range
      : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr<[? + 4]> -> !amdgcn.vgpr<[? + 4]>

    %c4 = arith.constant 4 : i32
    %thread_offset_f32 = amdgcn.vop2 v_lshlrev_b32_e32 outs %offset_a ins %c4, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>
    %tok_store_c = amdgcn.store global_store_dwordx4 data %c_mfma_result addr %c_ptr offset d(%thread_offset_f32) + c(%c0_i32) : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
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
class _CopyScalarMatch:
    src_name: str
    dst_name: str
    dtype: str
    shape: tuple[int, ...]
    element_count: int


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
    op_kind: str
    op_name: str
    value_type: str | None


@dataclass(frozen=True)
class _BinaryScalarMatch:
    lhs_name: str
    rhs_name: str
    dst_name: str
    dtype: str
    shape: tuple[int, ...]
    element_count: int
    op_kind: str
    op_name: str
    value_type: str | None
    rhs_broadcast: bool = False
    lhs_broadcast: bool = False


@dataclass(frozen=True)
class _Mfma16x16x16Match:
    lhs_name: str
    rhs_name: str
    dst_name: str
    lhs_dtype: str
    rhs_dtype: str
    dst_dtype: str
    tile: tuple[int, int, int]
    mfma_inst: str


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
        elif isinstance(match, _CopyScalarMatch):
            text = self._render_copy_scalar(match, ir.name, target)
        elif isinstance(match, _Binary1DMatch):
            text = self._render_binary_1d(match, ir.name, target)
        elif isinstance(match, _Mfma16x16x16Match):
            text = self._render_mfma_16x16x16(match, ir.name, target)
        else:
            text = self._render_binary_scalar(match, ir.name, target)
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
            if isinstance(match, (_Copy1DMatch, _CopyScalarMatch)):
                input_values = [
                    self._as_runtime_tensor(args[0], expected_dtype=match.dtype, argument_name=match.src_name),
                ]
                dst_owner, dst_value = self._prepare_output(
                    args[1],
                    expected_dtype=match.dtype,
                    argument_name=match.dst_name,
                )
                input_dtypes = [match.dtype]
                dst_dtype = match.dtype
                block_dim = (1, 1, 1)
            elif isinstance(match, _Mfma16x16x16Match):
                input_values = [
                    self._as_runtime_tensor(args[0], expected_dtype=match.lhs_dtype, argument_name=match.lhs_name),
                    self._as_runtime_tensor(args[1], expected_dtype=match.rhs_dtype, argument_name=match.rhs_name),
                ]
                dst_owner, dst_value = self._prepare_output(
                    args[2],
                    expected_dtype=match.dst_dtype,
                    argument_name=match.dst_name,
                )
                input_dtypes = [match.lhs_dtype, match.rhs_dtype]
                dst_dtype = match.dst_dtype
                block_dim = (64, 1, 1)
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
                input_dtypes = [match.dtype, match.dtype]
                dst_dtype = match.dtype
                block_dim = (1, 1, 1)
            try:
                numpy = importlib.import_module("numpy")
            except ModuleNotFoundError:
                numpy = None
            if numpy is None:
                dst_array = dst_value.tolist()
                input_arrays = [input_value.tolist() for input_value in input_values]
            else:
                input_arrays = [
                    self._numpy_array_for_dtype(numpy, input_value, dtype)
                    for input_value, dtype in zip(input_values, input_dtypes, strict=True)
                ]
                dst_array = self._numpy_array_for_dtype(numpy, dst_value, dst_dtype)
            modules["hip"].execute_hsaco(
                str(hsaco_path),
                ir.name,
                input_arrays,
                [dst_array],
                grid_dim=(1, 1, 1),
                block_dim=block_dim,
                num_iterations=1,
            )
            copied_data = dst_array.tolist() if hasattr(dst_array, "tolist") else dst_array
            copied = tensor(copied_data, dtype=dst_dtype)
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

    def _match_kernel(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
    ) -> _Copy1DMatch | _CopyScalarMatch | _Binary1DMatch | _BinaryScalarMatch | _Mfma16x16x16Match:
        if len(ir.arguments) == 2:
            return self._match_copy_1d(ir, target)
        if len(ir.arguments) == 3:
            last_error: BackendNotImplementedError | None = None
            for matcher in (
                self._match_mfma_16x16x16,
                self._match_fragment_mfma_16x16x16_copyout,
                self._match_binary_1d,
            ):
                try:
                    return matcher(ir, target)
                except BackendNotImplementedError as exc:
                    last_error = exc
                    continue
            if last_error is not None:
                raise last_error
            raise BackendNotImplementedError("aster_exec could not match this three-argument kernel")
        try:
            return self._match_copy_1d(ir, target)
        except BackendNotImplementedError:
            return self._match_binary_1d(ir, target)

    def _match_mfma_16x16x16_specs(
        self,
        lhs_arg: Any,
        rhs_arg: Any,
        dst_arg: Any,
        target: AMDTarget,
    ) -> _Mfma16x16x16Match:
        if target.arch not in {"gfx942", "gfx950"}:
            raise BackendNotImplementedError(f"aster_exec does not support target arch '{target.arch}'")
        if target.wave_size != 64:
            raise BackendNotImplementedError("aster_exec currently requires wave_size=64")
        lhs_spec = lhs_arg.spec
        rhs_spec = rhs_arg.spec
        dst_spec = dst_arg.spec
        if lhs_spec.address_space.value != "global" or rhs_spec.address_space.value != "global" or dst_spec.address_space.value != "global":
            raise BackendNotImplementedError("aster_exec MFMA currently requires global-memory tensors")
        if lhs_spec.dtype != rhs_spec.dtype or dst_spec.dtype != "f32":
            raise BackendNotImplementedError("aster_exec MFMA currently requires matching operand dtypes and f32 accumulation")
        if lhs_spec.dtype not in {"f16", "bf16"}:
            raise BackendNotImplementedError("aster_exec MFMA currently supports f16/f16 -> f32 and bf16/bf16 -> f32 only")
        if lhs_spec.shape != (16, 16) or rhs_spec.shape != (16, 16) or dst_spec.shape != (16, 16):
            raise BackendNotImplementedError("aster_exec MFMA currently supports exact 16x16x16 tiles only")
        if (
            not self._is_dense_contiguous(lhs_spec)
            or not self._is_dense_contiguous(rhs_spec)
            or not self._is_dense_contiguous(dst_spec)
        ):
            raise BackendNotImplementedError("aster_exec MFMA currently requires dense contiguous tensors")
        mfma_inst = {
            "f16": "v_mfma_f32_16x16x16_f16",
            "bf16": "v_mfma_f32_16x16x16_bf16",
        }[lhs_spec.dtype]
        return _Mfma16x16x16Match(
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            lhs_dtype=lhs_spec.dtype,
            rhs_dtype=rhs_spec.dtype,
            dst_dtype=dst_spec.dtype,
            tile=(16, 16, 16),
            mfma_inst=mfma_inst,
        )

    def _match_mfma_16x16x16(self, ir: PortableKernelIR, target: AMDTarget) -> _Mfma16x16x16Match:
        if len(ir.arguments) != 3:
            raise BackendNotImplementedError("aster_exec MFMA currently supports exactly three tensor arguments")
        if len(ir.operations) != 1 or ir.operations[0].op != "mma":
            raise BackendNotImplementedError("aster_exec MFMA currently supports a single mma operation only")
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            raise BackendNotImplementedError("aster_exec MFMA currently supports tensor arguments only")
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        match = self._match_mfma_16x16x16_specs(lhs_arg, rhs_arg, dst_arg, target)
        operation = ir.operations[0]
        tile = tuple(int(dim) for dim in operation.attrs.get("tile") or ())
        if tile != match.tile:
            raise BackendNotImplementedError("aster_exec MFMA currently supports tile=(16,16,16) only")
        if operation.inputs != (lhs_arg.name, rhs_arg.name, dst_arg.name):
            raise BackendNotImplementedError("aster_exec MFMA currently requires mma inputs to match the tensor argument order")
        if operation.attrs.get("transpose_a") or operation.attrs.get("transpose_b"):
            raise BackendNotImplementedError("aster_exec MFMA currently supports only non-transposed operands")
        if operation.attrs.get("accumulate", True) is not True:
            raise BackendNotImplementedError("aster_exec MFMA currently requires accumulate=True")
        return match

    def _match_fragment_mfma_16x16x16_copyout(self, ir: PortableKernelIR, target: AMDTarget) -> _Mfma16x16x16Match:
        if len(ir.arguments) != 3:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently supports exactly three tensor arguments")
        if not all(isinstance(argument.spec, TensorSpec) for argument in ir.arguments):
            raise BackendNotImplementedError("aster_exec fragment MFMA currently supports tensor arguments only")
        lhs_arg, rhs_arg, dst_arg = ir.arguments
        match = self._match_mfma_16x16x16_specs(lhs_arg, rhs_arg, dst_arg, target)
        if not ir.operations or ir.operations[-1].op != "copy":
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires a final copy to the destination tensor")
        copy_op = ir.operations[-1]
        if len(copy_op.inputs) != 2 or copy_op.inputs[1] != dst_arg.name:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires copying the accumulator tensor directly to the destination")
        acc_name = copy_op.inputs[0]
        make_tensor_ops = [operation for operation in ir.operations if operation.op == "make_tensor" and operation.outputs == (acc_name,)]
        if len(make_tensor_ops) != 1:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires a single explicit accumulator tensor")
        acc_op = make_tensor_ops[0]
        if acc_op.attrs.get("dtype") != "f32" or acc_op.attrs.get("address_space") != "register":
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires an f32 register accumulator")
        mma_ops = [operation for operation in ir.operations if operation.op == "mma"]
        if len(mma_ops) != 1:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires exactly one mma op")
        mma_op = mma_ops[0]
        if tuple(int(dim) for dim in mma_op.attrs.get("tile") or ()) != match.tile:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently supports tile=(16,16,16) only")
        if mma_op.attrs.get("transpose_a") or mma_op.attrs.get("transpose_b"):
            raise BackendNotImplementedError("aster_exec fragment MFMA currently supports only non-transposed operands")
        if mma_op.attrs.get("accumulate", True) is not True:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires accumulate=True")
        lhs_fragment_name, rhs_fragment_name, mma_acc_name = mma_op.inputs
        if mma_acc_name != acc_name:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires the mma accumulator to be the explicit register tensor")
        fragment_ops = {operation.outputs[0]: operation for operation in ir.operations if operation.op == "fragment" and operation.outputs}
        lhs_fragment = fragment_ops.get(lhs_fragment_name)
        rhs_fragment = fragment_ops.get(rhs_fragment_name)
        if lhs_fragment is None or rhs_fragment is None:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires fragment views for both operands")
        if lhs_fragment.attrs.get("role") != "a" or rhs_fragment.attrs.get("role") != "b":
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires a/b fragment roles")
        if tuple(int(dim) for dim in lhs_fragment.attrs.get("tile") or ()) != match.tile:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires the lhs fragment tile to match the supported MFMA tile")
        if tuple(int(dim) for dim in rhs_fragment.attrs.get("tile") or ()) != match.tile:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires the rhs fragment tile to match the supported MFMA tile")
        if lhs_fragment.attrs.get("llvm_intrinsic") != {
            "f16": "llvm.amdgcn.mfma.f32.16x16x16f16",
            "bf16": "llvm.amdgcn.mfma.f32.16x16x16bf16",
        }[match.lhs_dtype]:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires a matching lhs fragment intrinsic")
        if rhs_fragment.attrs.get("llvm_intrinsic") != lhs_fragment.attrs.get("llvm_intrinsic"):
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires matching fragment intrinsics")
        partition_ops = {operation.outputs[0]: operation for operation in ir.operations if operation.op == "partition" and operation.outputs}
        lhs_partition = partition_ops.get(lhs_fragment.inputs[0])
        rhs_partition = partition_ops.get(rhs_fragment.inputs[0])
        if lhs_partition is None or rhs_partition is None:
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires partitioned global tiles feeding both fragments")
        if lhs_partition.inputs[:1] != (lhs_arg.name,) or rhs_partition.inputs[:1] != (rhs_arg.name,):
            raise BackendNotImplementedError("aster_exec fragment MFMA currently requires partitions to come directly from the tensor arguments")
        for partition_op, expected_dtype in ((lhs_partition, match.lhs_dtype), (rhs_partition, match.rhs_dtype)):
            result_spec = partition_op.attrs.get("result", {})
            layout = result_spec.get("layout", {})
            if tuple(result_spec.get("shape", ())) != (16, 16):
                raise BackendNotImplementedError("aster_exec fragment MFMA currently requires 16x16 partition tiles")
            if result_spec.get("dtype") != expected_dtype or result_spec.get("address_space") != "global":
                raise BackendNotImplementedError("aster_exec fragment MFMA currently requires global operand tiles with matching dtypes")
            if tuple(layout.get("stride", ())) != (16, 1):
                raise BackendNotImplementedError("aster_exec fragment MFMA currently requires dense contiguous partition tiles")
            if tuple(int(dim) for dim in partition_op.attrs.get("tile") or ()) != (16, 16):
                raise BackendNotImplementedError("aster_exec fragment MFMA currently requires tile=(16,16) partitioning")
            if partition_op.attrs.get("policy") != "blocked":
                raise BackendNotImplementedError("aster_exec fragment MFMA currently requires blocked partition policy")
        return _Mfma16x16x16Match(
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            lhs_dtype=match.lhs_dtype,
            rhs_dtype=match.rhs_dtype,
            dst_dtype=match.dst_dtype,
            tile=match.tile,
            mfma_inst=match.mfma_inst,
        )

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
        if src_spec.dtype not in {"f32", "i32", "f16"}:
            raise BackendNotImplementedError("aster_exec currently supports f32, i32, and f16 copies only")
        if not self._is_dense_contiguous(src_spec) or not self._is_dense_contiguous(dst_spec):
            raise BackendNotImplementedError("aster_exec currently requires dense contiguous tensors")
        element_count = self._total_elements(src_spec.shape)
        elements_per_chunk = 16 // self._dtype_size_bytes(src_spec.dtype)
        if element_count <= 0:
            raise BackendNotImplementedError("aster_exec requires a positive contiguous element count")
        if element_count % elements_per_chunk == 0:
            return _Copy1DMatch(
                src_name=src_arg.name,
                dst_name=dst_arg.name,
                dtype=src_spec.dtype,
                shape=src_spec.shape,
                vector_chunks=element_count // elements_per_chunk,
                elements_per_chunk=elements_per_chunk,
            )
        if src_spec.dtype == "f16" and element_count % 2 == 0:
            return _CopyScalarMatch(
                src_name=src_arg.name,
                dst_name=dst_arg.name,
                dtype=src_spec.dtype,
                shape=src_spec.shape,
                element_count=element_count // 2,
            )
        if src_spec.dtype in {"f32", "i32"}:
            return _CopyScalarMatch(
                src_name=src_arg.name,
                dst_name=dst_arg.name,
                dtype=src_spec.dtype,
                shape=src_spec.shape,
                element_count=element_count,
            )
        raise BackendNotImplementedError(
            "aster_exec requires a contiguous element count aligned to a supported transfer chunk for this dtype"
        )

    def _render_copy_1d(self, match: _Copy1DMatch, kernel_name: str, target: AMDTarget) -> str:
        return _COPY_1D_TEMPLATE.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            vector_chunks=match.vector_chunks,
        )

    def _render_copy_scalar(self, match: _CopyScalarMatch, kernel_name: str, target: AMDTarget) -> str:
        return _SCALAR_COPY_TEMPLATE.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            element_count=match.element_count,
        )

    def _match_binary_1d(self, ir: PortableKernelIR, target: AMDTarget) -> _Binary1DMatch | _BinaryScalarMatch:
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
        if op_names[-1:] != ["copy"]:
            raise BackendNotImplementedError(
                "aster_exec currently supports a single supported tensor binary pipeline only"
            )
        binary_op = None
        rhs_broadcast = False
        lhs_broadcast = False
        if len(op_names) == 6 and op_names[:4] == ["make_tensor", "copy", "make_tensor", "copy"]:
            binary_op = op_names[4]
        elif (
            len(op_names) == 7
            and op_names[:4] == ["make_tensor", "copy", "make_tensor", "copy"]
            and op_names[4] == "broadcast_to"
        ):
            binary_op = op_names[5]
            broadcast_op = ir.operations[4]
            binary_ir_op = ir.operations[5]
            lhs_loaded = ir.operations[0].outputs[0]
            rhs_loaded = ir.operations[2].outputs[0]
            if broadcast_op.inputs == (rhs_loaded,) and binary_ir_op.inputs == (lhs_loaded, broadcast_op.outputs[0]):
                rhs_broadcast = True
        elif (
            len(op_names) == 7
            and op_names[:3] == ["make_tensor", "copy", "broadcast_to"]
            and op_names[3:5] == ["make_tensor", "copy"]
        ):
            binary_op = op_names[5]
            broadcast_op = ir.operations[2]
            binary_ir_op = ir.operations[5]
            lhs_loaded = ir.operations[0].outputs[0]
            rhs_loaded = ir.operations[3].outputs[0]
            if broadcast_op.inputs == (lhs_loaded,) and binary_ir_op.inputs == (broadcast_op.outputs[0], rhs_loaded):
                lhs_broadcast = True
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
            raise BackendNotImplementedError("aster_exec pointwise binary ops currently require global-memory tensors")
        if rhs_broadcast:
            if lhs_spec.shape != dst_spec.shape:
                raise BackendNotImplementedError(
                    "aster_exec broadcasted pointwise binary ops require the dense source and destination tensor shapes to match"
                )
        elif lhs_broadcast:
            if rhs_spec.shape != dst_spec.shape:
                raise BackendNotImplementedError(
                    "aster_exec broadcasted pointwise binary ops require the dense source and destination tensor shapes to match"
                )
        elif lhs_spec.shape != rhs_spec.shape or lhs_spec.shape != dst_spec.shape:
            raise BackendNotImplementedError("aster_exec pointwise binary ops require matching tensor shapes")
        if lhs_spec.dtype != rhs_spec.dtype or lhs_spec.dtype != dst_spec.dtype:
            raise BackendNotImplementedError("aster_exec pointwise binary ops require matching tensor dtypes")
        if (
            not self._is_dense_contiguous(lhs_spec)
            or not self._is_dense_contiguous(rhs_spec)
            or not self._is_dense_contiguous(dst_spec)
        ):
            raise BackendNotImplementedError("aster_exec pointwise binary ops currently require dense contiguous tensors")
        dense_shape = lhs_spec.shape if not lhs_broadcast else rhs_spec.shape
        element_count = self._total_elements(dense_shape)
        if element_count <= 0:
            raise BackendNotImplementedError("aster_exec pointwise binary ops require a positive contiguous element count")
        dtype_support = {
            "f32": {
                "tensor_add": ("add", "vop2", "v_add_f32", None),
                "tensor_sub": ("sub", "vop2", "v_sub_f32", None),
                "tensor_mul": ("mul", "vop2", "v_mul_f32", None),
            },
            "i32": {
                "tensor_add": ("add", "vop2", "v_add_u32", None),
                "tensor_sub": ("sub", "vop2", "v_sub_u32", None),
                "tensor_mul": ("mul", "vop3", "v_mul_lo_u32", None),
            },
        }
        supported_for_dtype = dtype_support.get(lhs_spec.dtype)
        if supported_for_dtype is None or binary_op not in supported_for_dtype:
            raise BackendNotImplementedError(
                "aster_exec currently supports f32 and i32 add/sub/mul only"
            )
        if rhs_broadcast or lhs_broadcast:
            broadcast_op = ir.operations[4] if rhs_broadcast else ir.operations[2]
            broadcast_source_spec = rhs_spec if rhs_broadcast else lhs_spec
            if not self._is_dense_contiguous(broadcast_source_spec) or self._total_elements(broadcast_source_spec.shape) != 1:
                raise BackendNotImplementedError(
                    "aster_exec currently supports broadcast only from a dense single-element tensor argument"
                )
            result_spec = broadcast_op.attrs.get("result", {})
            if tuple(result_spec.get("shape", ())) != dense_shape:
                raise BackendNotImplementedError(
                    "aster_exec currently requires the broadcast target shape to match the dense destination shape"
                )
            broadcast_layout = result_spec.get("layout", {})
            if any(stride != 0 for stride in tuple(broadcast_layout.get("stride", ()))):
                raise BackendNotImplementedError(
                    "aster_exec currently requires a scalar broadcast layout with zero strides"
                )
        binary_name, op_kind, op_name, value_type = supported_for_dtype[binary_op]
        if lhs_spec.dtype in {"f32", "i32"} and element_count % 4 != 0:
            return _BinaryScalarMatch(
                lhs_name=lhs_arg.name,
                rhs_name=rhs_arg.name,
                dst_name=dst_arg.name,
                dtype=lhs_spec.dtype,
                shape=dense_shape,
                element_count=element_count,
                op_kind=op_kind,
                op_name=op_name,
                value_type=value_type,
                rhs_broadcast=rhs_broadcast,
                lhs_broadcast=lhs_broadcast,
            )
        if element_count % 4 != 0:
            raise BackendNotImplementedError(
                "aster_exec pointwise binary ops require a contiguous element count aligned to the vector path for this dtype"
            )
        if rhs_broadcast or lhs_broadcast:
            return _BinaryScalarMatch(
                lhs_name=lhs_arg.name,
                rhs_name=rhs_arg.name,
                dst_name=dst_arg.name,
                dtype=lhs_spec.dtype,
                shape=dense_shape,
                element_count=element_count,
                op_kind=op_kind,
                op_name=op_name,
                value_type=value_type,
                rhs_broadcast=rhs_broadcast,
                lhs_broadcast=lhs_broadcast,
            )
        return _Binary1DMatch(
            lhs_name=lhs_arg.name,
            rhs_name=rhs_arg.name,
            dst_name=dst_arg.name,
            dtype=lhs_spec.dtype,
            shape=lhs_spec.shape,
            vector_chunks=element_count // 4,
            op=binary_op,
            binary_name=binary_name,
            op_kind=op_kind,
            op_name=op_name,
            value_type=value_type,
        )

    def _render_binary_1d(self, match: _Binary1DMatch, kernel_name: str, target: AMDTarget) -> str:
        op_lines = [
            self._render_binary_vgpr_op(match, f"%result{index}", f"%tmp{index}", f"%lhs{index}", f"%rhs{index}")
            for index in range(4)
        ]
        return _BINARY_1D_TEMPLATE.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            vector_chunks=match.vector_chunks,
            binary_name=match.binary_name,
            op_line0=op_lines[0],
            op_line1=op_lines[1],
            op_line2=op_lines[2],
            op_line3=op_lines[3],
        )

    def _render_binary_scalar(self, match: _BinaryScalarMatch, kernel_name: str, target: AMDTarget) -> str:
        if match.rhs_broadcast:
            template = _SCALAR_BROADCAST_BINARY_TEMPLATE
        elif match.lhs_broadcast:
            template = _SCALAR_LHS_BROADCAST_BINARY_TEMPLATE
        else:
            template = _SCALAR_BINARY_TEMPLATE
        return template.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            element_count=match.element_count,
            op_line=self._render_binary_vgpr_op(match, "%result", "%tmp", "%lhs", "%rhs"),
        )

    def _render_mfma_16x16x16(self, match: _Mfma16x16x16Match, kernel_name: str, target: AMDTarget) -> str:
        return _MFMA_16X16X16_TEMPLATE.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            mfma_inst=match.mfma_inst,
        )

    def _render_binary_vgpr_op(
        self,
        match: _Binary1DMatch | _BinaryScalarMatch,
        result_name: str,
        temp_name: str,
        lhs_name: str,
        rhs_name: str,
    ) -> str:
        if match.op_kind == "lsir":
            return (
                f"    {result_name} = lsir.{match.op_name} {match.value_type} "
                f"{temp_name}, {lhs_name}, {rhs_name} : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr"
            )
        return (
            f"    {result_name} = amdgcn.{match.op_kind} {match.op_name} "
            f"outs {temp_name} ins {lhs_name}, {rhs_name} : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr"
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
        package_root = Path(python_package_root)
        package_parent = str(package_root.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)
        mlir_lib_root = package_root / "_mlir_libs"
        if mlir_lib_root.exists():
            preload_names = [
                "libMLIRPythonSupport-mlir.so",
                "libnanobind-mlir.so",
                "libASTER.so.23.0git",
                "libASTER.so",
            ]
            for name in preload_names:
                candidate = mlir_lib_root / name
                if candidate.exists():
                    ctypes.CDLL(str(candidate), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
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

    def _numpy_array_for_dtype(self, numpy: Any, value: RuntimeTensor, dtype: str) -> Any:
        if dtype == "bf16":
            float_array = numpy.array(value.tolist(), dtype="float32")
            return (float_array.view(numpy.uint32) >> 16).astype(numpy.uint16)
        return numpy.array(value.tolist(), dtype=self._numpy_dtype(dtype))

    def _dtype_size_bytes(self, dtype: str) -> int:
        if dtype in {"f32", "i32"}:
            return 4
        if dtype == "f16":
            return 2
        raise BackendNotImplementedError(f"aster_exec does not support dtype '{dtype}'")

    def _total_elements(self, shape: tuple[int, ...]) -> int:
        total = 1
        for extent in shape:
            total *= extent
        return total

    def _is_dense_contiguous(self, spec: TensorSpec) -> bool:
        layout = spec.resolved_layout()
        if len(layout.stride) != len(spec.shape):
            return False
        expected: list[int] = [0] * len(spec.shape)
        running = 1
        for index in range(len(spec.shape) - 1, -1, -1):
            expected[index] = running
            running *= spec.shape[index]
        return tuple(expected) == tuple(layout.stride)
