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


@dataclass(frozen=True)
class _Copy1DMatch:
    src_name: str
    dst_name: str
    dtype: str
    shape: tuple[int, ...]
    vector_chunks: int


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
            self._match_copy_1d(ir, target)
        except BackendNotImplementedError:
            return False
        return True

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        match = self._match_copy_1d(ir, target)
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="aster_exec_mlir",
            text=self._render_copy_1d(match, ir.name, target),
        )

    def build_launcher(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        source_path: Path,
    ):
        match = self._match_copy_1d(ir, target)
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
            src_value = self._as_runtime_tensor(args[0], expected_dtype=match.dtype, argument_name=match.src_name)
            dst_owner, dst_value = self._prepare_output(args[1], expected_dtype=match.dtype, argument_name=match.dst_name)
            try:
                numpy = importlib.import_module("numpy")
            except ModuleNotFoundError:
                numpy = None
            if numpy is None:
                src_array = src_value.tolist()
                dst_array = dst_value.tolist()
            else:
                src_array = numpy.array(src_value.tolist(), dtype=self._numpy_dtype(match.dtype))
                dst_array = numpy.array(dst_value.tolist(), dtype=self._numpy_dtype(match.dtype))
            modules["hip"].execute_hsaco(
                str(hsaco_path),
                ir.name,
                [src_array],
                [dst_array],
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
                num_iterations=1,
            )
            copied_data = dst_array.tolist() if hasattr(dst_array, "tolist") else dst_array
            copied = tensor(copied_data, dtype=match.dtype)
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
        if src_spec.dtype not in {"f32", "i32"}:
            raise BackendNotImplementedError("aster_exec currently supports f32 and i32 copies only")
        if src_spec.resolved_layout().stride != (1,) or dst_spec.resolved_layout().stride != (1,):
            raise BackendNotImplementedError("aster_exec currently requires contiguous 1D tensors")
        element_count = src_spec.shape[0]
        if element_count <= 0 or element_count % 4 != 0:
            raise BackendNotImplementedError("aster_exec requires a 1D element count that is a positive multiple of 4")
        return _Copy1DMatch(
            src_name=src_arg.name,
            dst_name=dst_arg.name,
            dtype=src_spec.dtype,
            shape=src_spec.shape,
            vector_chunks=element_count // 4,
        )

    def _render_copy_1d(self, match: _Copy1DMatch, kernel_name: str, target: AMDTarget) -> str:
        return _COPY_1D_TEMPLATE.format(
            target=target.arch,
            isa=self._target_isa(target),
            kernel_name=kernel_name,
            vector_chunks=match.vector_chunks,
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
        if dtype == "i32":
            return "int32"
        raise BackendNotImplementedError(f"aster_exec does not support dtype '{dtype}'")
