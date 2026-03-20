from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from glob import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..hip_runtime import HipRuntime, load_hip_library, require_hipcc, scalar_ctype
from ..ir import KernelArgument, PortableKernelIR, TensorSpec
from ..runtime import RuntimeTensor, TensorHandle
from ..target import AMDTarget

_HIPKITTENS_ENV = "BAYBRIDGE_HIPKITTENS_ROOT"


@dataclass(frozen=True)
class HipKittensExecDescriptor:
    family: str
    operand_dtype: str
    accumulator_dtype: str
    transpose_a: bool
    transpose_b: bool
    a_tile_shape: tuple[int, int]
    b_tile_shape: tuple[int, int]
    c_tile_shape: tuple[int, int]
    a_rt_shape: str
    b_rt_shape: str
    c_rt_shape: str
    a_layout: str
    b_layout: str
    c_layout: str
    gl_operand_dtype: str
    gl_accumulator_dtype: str
    rt_operand_alias: str
    rt_accumulator_alias: str
    host_operand_pointer: str
    host_accumulator_pointer: str
    mma_statement: str
    reference_path: str


@dataclass(frozen=True)
class HipKittensExecMatch:
    descriptor: HipKittensExecDescriptor
    a_name: str
    b_name: str
    c_name: str
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    c_shape: tuple[int, int]
    grid: tuple[int, int, int]
    k_tiles: int


@dataclass(frozen=True)
class HipKittensLayerNormExecMatch:
    x_name: str
    residual_name: str
    out_name: str
    out_resid_name: str
    weight_name: str
    bias_name: str
    shape: tuple[int, int, int]
    epsilon: float
    reference_path: str


@dataclass(frozen=True)
class HipKittensRmsNormExecMatch:
    x_name: str
    out_name: str
    gamma_name: str
    shape: tuple[int, int, int]
    epsilon: float
    reference_path: str


@dataclass(frozen=True)
class HipKittensAttentionExecMatch:
    q_name: str
    k_name: str
    v_name: str
    out_name: str
    lse_name: str
    q_shape: tuple[int, int, int, int]
    kv_shape: tuple[int, int, int, int]
    lse_shape: tuple[int, int, int, int]
    causal: bool
    reference_path: str


_DESCRIPTORS = (
    HipKittensExecDescriptor(
        family="bf16_gemm_32x16x32",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=False,
        a_tile_shape=(32, 16),
        b_tile_shape=(16, 32),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AB(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="bf16_gemm_AtB_32x16x32",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=False,
        a_tile_shape=(16, 32),
        b_tile_shape=(16, 32),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtB(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="bf16_gemm_ABt_32x16x32",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=True,
        a_tile_shape=(32, 16),
        b_tile_shape=(32, 16),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_ABt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="bf16_gemm_AtBt_32x16x32",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=True,
        a_tile_shape=(16, 32),
        b_tile_shape=(32, 16),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtBt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="bf16_gemm_16x32x16",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=False,
        a_tile_shape=(16, 32),
        b_tile_shape=(32, 16),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AB(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="bf16_gemm_AtB_16x32x16",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=False,
        a_tile_shape=(32, 16),
        b_tile_shape=(32, 16),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtB(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="bf16_gemm_ABt_16x32x16",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=True,
        a_tile_shape=(16, 32),
        b_tile_shape=(16, 32),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_ABt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="bf16_gemm_AtBt_16x32x16",
        operand_dtype="bf16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=True,
        a_tile_shape=(32, 16),
        b_tile_shape=(16, 32),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="bf16",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_bf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtBt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_32x16x32",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=False,
        a_tile_shape=(32, 16),
        b_tile_shape=(16, 32),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mfma323216(c_accum.tiles[0][0].data, a_tile.tiles[0][0].data, b_tile.tiles[0][0].data, c_accum.tiles[0][0].data);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_AtB_32x16x32",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=False,
        a_tile_shape=(16, 32),
        b_tile_shape=(16, 32),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtB(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_ABt_32x16x32",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=True,
        a_tile_shape=(32, 16),
        b_tile_shape=(32, 16),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_ABt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_AtBt_32x16x32",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=True,
        a_tile_shape=(16, 32),
        b_tile_shape=(32, 16),
        c_tile_shape=(32, 32),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_32x32",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtBt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_16x32x16",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=False,
        a_tile_shape=(16, 32),
        b_tile_shape=(32, 16),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mfma161632(c_accum.tiles[0][0].data, a_tile.tiles[0][0].data, b_tile.tiles[0][0].data, c_accum.tiles[0][0].data);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_AtB_16x32x16",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=False,
        a_tile_shape=(32, 16),
        b_tile_shape=(32, 16),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_32x16",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::col",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtB(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_ABt_16x32x16",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=False,
        transpose_b=True,
        a_tile_shape=(16, 32),
        b_tile_shape=(16, 32),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_16x32",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::row",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_ABt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
    HipKittensExecDescriptor(
        family="f16_gemm_AtBt_16x32x16",
        operand_dtype="f16",
        accumulator_dtype="f32",
        transpose_a=True,
        transpose_b=True,
        a_tile_shape=(32, 16),
        b_tile_shape=(16, 32),
        c_tile_shape=(16, 16),
        a_rt_shape="ducks::rt_shape::rt_32x16",
        b_rt_shape="ducks::rt_shape::rt_16x32",
        c_rt_shape="ducks::rt_shape::rt_16x16",
        a_layout="ducks::rt_layout::col",
        b_layout="ducks::rt_layout::row",
        c_layout="ducks::rt_layout::col",
        gl_operand_dtype="half",
        gl_accumulator_dtype="float",
        rt_operand_alias="rt_hf",
        rt_accumulator_alias="rt_fl",
        host_operand_pointer="const std::uint16_t*",
        host_accumulator_pointer="float*",
        mma_statement="mma_AtBt(c_accum, a_tile, b_tile, c_accum);",
        reference_path="include/ops/warp/register/tile/mma.cuh",
    ),
)


class HipKittensExecBackend:
    name = "hipkittens_exec"
    artifact_extension = ".hipkittens.cpp"

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        try:
            self._match(ir, target)
        except BackendNotImplementedError:
            return False
        return True

    def available(self, target: AMDTarget | None = None) -> bool:
        if self._configured_root() is None:
            return False
        if target is None:
            return True
        return self._toolchain_ready(target)

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        match = self._match(ir, target)
        root = self._configured_root()
        text = self._render_cpp(ir.name, target, match, root)
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="hipkittens_exec_cpp",
            text=text,
        )

    def build_launcher(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        source_path: Path,
    ):
        self._match(ir, target)
        root = self._require_root()
        shared_path = source_path.with_suffix("").with_suffix(".so")
        state: dict[str, Any] = {}

        def launcher(*args: Any, **kwargs: Any) -> None:
            stream = kwargs.pop("stream", None)
            del stream
            if kwargs:
                raise TypeError("hipkittens_exec launcher only supports positional arguments and an optional stream=")
            if len(args) != len(ir.arguments):
                raise TypeError(f"{ir.name} expects {len(ir.arguments)} arguments, got {len(args)}")
            if not shared_path.exists():
                self._compile_shared_object(source_path, shared_path, target, lowered_module.text, root)
            function = state.get("function")
            if function is None:
                load_hip_library(global_scope=True)
                library = ctypes.CDLL(str(shared_path))
                function = getattr(library, f"launch_{ir.name}")
                function.argtypes = self._launcher_argtypes(ir.arguments)
                function.restype = ctypes.c_int
                state["library"] = library
                state["function"] = function
            self._launch(function, ir, args)

        return launcher

    def _compile_shared_object(
        self,
        source_path: Path,
        shared_path: Path,
        target: AMDTarget,
        source_text: str,
        root: Path,
    ) -> None:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source_text, encoding="utf-8")
        hipcc = require_hipcc()
        command = [
            hipcc,
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"--offload-arch={target.arch}",
            self._arch_define(target),
            f"-I{root / 'include'}",
        ]
        command.extend(self._toolchain_include_args(target))
        command.extend(
            [
                str(source_path),
                "-o",
                str(shared_path),
            ]
        )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"hipcc failed for backend {self.name} with exit code {exc.returncode}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

    def _launch(self, function: Any, ir: PortableKernelIR, args: tuple[Any, ...]) -> None:
        hip = HipRuntime()
        tensor_allocations = []
        c_args: list[Any] = []
        try:
            for argument, value in zip(ir.arguments, args):
                if not isinstance(argument.spec, TensorSpec):
                    ctype = scalar_ctype(argument.spec.dtype)
                    c_args.append(ctype(value))
                    continue
                if isinstance(value, RuntimeTensor):
                    allocation = hip.upload_tensor(value)
                    tensor_allocations.append(allocation)
                    c_args.append(allocation.ptr)
                    continue
                if isinstance(value, TensorHandle):
                    data_ptr = value.data_ptr()
                    if not data_ptr:
                        raise TypeError(
                            f"hipkittens_exec tensor handle for argument '{argument.name}' does not expose a usable data_ptr()"
                        )
                    c_args.append(ctypes.c_void_p(data_ptr))
                    continue
                raise TypeError(
                    f"hipkittens_exec expects RuntimeTensor or TensorHandle values for tensor argument '{argument.name}', got {type(value).__name__}"
                )
            status = function(*c_args)
            if status != 0:
                raise RuntimeError(f"launch_{ir.name} returned HIP status {status}")
            for allocation in tensor_allocations:
                allocation.copy_back(hip)
        finally:
            for allocation in tensor_allocations:
                allocation.free(hip)

    def _launcher_argtypes(self, arguments: tuple[KernelArgument, ...]) -> list[Any]:
        argtypes: list[Any] = []
        for argument in arguments:
            if isinstance(argument.spec, TensorSpec):
                argtypes.append(ctypes.c_void_p)
            else:
                argtypes.append(scalar_ctype(argument.spec.dtype))
        return argtypes

    def _match(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
    ) -> HipKittensExecMatch | HipKittensLayerNormExecMatch | HipKittensRmsNormExecMatch | HipKittensAttentionExecMatch:
        if target.arch not in {"gfx950", "gfx942"}:
            raise BackendNotImplementedError(
                f"hipkittens_exec currently supports gfx950 and gfx942 only, got target arch '{target.arch}'"
            )
        if len(ir.operations) != 1:
            raise BackendNotImplementedError(
                "hipkittens_exec currently supports a single explicit HipKittens op or a single pure GEMM mma op"
            )
        operation = ir.operations[0]
        if operation.op == "mma":
            return self._match_gemm(ir, target)
        if operation.op == "layernorm":
            return self._match_layernorm(ir, target, operation)
        if operation.op == "rmsnorm":
            return self._match_rmsnorm(ir, target, operation)
        if operation.op == "attention":
            return self._match_attention(ir, target, operation)
        raise BackendNotImplementedError(
            "hipkittens_exec currently supports exact GEMM, layernorm, rmsnorm, and fused attention ops only"
        )

    def _match_gemm(self, ir: PortableKernelIR, target: AMDTarget) -> HipKittensExecMatch:
        mma_ops = [operation for operation in ir.operations if operation.op == "mma"]
        if len(mma_ops) != 1:
            raise BackendNotImplementedError("hipkittens_exec currently supports exactly one mma op")
        mma = mma_ops[0]
        a_name, b_name, c_name = mma.inputs
        transpose_a = bool(mma.attrs.get("transpose_a", False))
        transpose_b = bool(mma.attrs.get("transpose_b", False))
        specs = {argument.name: argument.spec for argument in ir.arguments if isinstance(argument.spec, TensorSpec)}
        try:
            a_spec = specs[a_name]
            b_spec = specs[b_name]
            c_spec = specs[c_name]
        except KeyError as exc:
            raise BackendNotImplementedError("hipkittens_exec expects mma operands to be direct tensor arguments") from exc
        if a_spec.dtype != b_spec.dtype:
            raise BackendNotImplementedError(
                f"hipkittens_exec requires matching operand dtypes, got {a_spec.dtype} and {b_spec.dtype}"
            )
        if target.arch == "gfx950" and a_spec.dtype == "f16" and (transpose_a or transpose_b):
            raise BackendNotImplementedError(
                "hipkittens_exec transposed f16 GEMM is not supported by the current HipKittens gfx950 MMA templates; "
                "use hipkittens_ref"
            )
        logical_m = a_spec.shape[1] if transpose_a else a_spec.shape[0]
        logical_k = a_spec.shape[0] if transpose_a else a_spec.shape[1]
        logical_kb = b_spec.shape[1] if transpose_b else b_spec.shape[0]
        logical_n = b_spec.shape[0] if transpose_b else b_spec.shape[1]
        if logical_kb != logical_k or c_spec.shape != (logical_m, logical_n):
            raise BackendNotImplementedError(
                f"hipkittens_exec requires GEMM-compatible shapes, got {a_spec.shape} x {b_spec.shape} -> {c_spec.shape}"
            )
        candidates: list[HipKittensExecMatch] = []
        for descriptor in _DESCRIPTORS:
            if not self._descriptor_supported_on_target(descriptor, target):
                continue
            if descriptor.transpose_a != transpose_a or descriptor.transpose_b != transpose_b:
                continue
            if a_spec.dtype != descriptor.operand_dtype or c_spec.dtype != descriptor.accumulator_dtype:
                continue
            tile_m, tile_n = descriptor.c_tile_shape
            tile_k = descriptor.a_tile_shape[0] if descriptor.transpose_a else descriptor.a_tile_shape[1]
            b_k = descriptor.b_tile_shape[1] if descriptor.transpose_b else descriptor.b_tile_shape[0]
            if tile_k != b_k:
                continue
            if logical_m % tile_m != 0:
                continue
            if logical_n % tile_n != 0:
                continue
            if logical_k % tile_k != 0:
                continue
            candidates.append(
                HipKittensExecMatch(
                    descriptor=descriptor,
                    a_name=a_name,
                    b_name=b_name,
                    c_name=c_name,
                    a_shape=a_spec.shape,
                    b_shape=b_spec.shape,
                    c_shape=c_spec.shape,
                    grid=(logical_n // tile_n, logical_m // tile_m, 1),
                    k_tiles=logical_k // tile_k,
                )
            )
        if candidates:
            return min(
                candidates,
                key=lambda match: (
                    match.grid[0] * match.grid[1] * match.k_tiles,
                    -(match.descriptor.a_tile_shape[0] * match.descriptor.b_tile_shape[1] * match.descriptor.a_tile_shape[1]),
                ),
            )
        raise BackendNotImplementedError(
            "hipkittens_exec only supports GEMM shapes composed from supported HipKittens tiles "
            + ", ".join(
                f"{d.operand_dtype}:{d.a_tile_shape} x {d.b_tile_shape} -> {d.accumulator_dtype}:{d.c_tile_shape}"
                for d in _DESCRIPTORS
                if self._descriptor_supported_on_target(d, target)
            )
        )

    def _match_layernorm(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        operation: Any,
    ) -> HipKittensLayerNormExecMatch:
        if target.arch not in {"gfx950", "gfx942"}:
            raise BackendNotImplementedError("hipkittens_exec layernorm is only wired for gfx950 and gfx942")
        if len(operation.inputs) != 6:
            raise BackendNotImplementedError("hipkittens_exec layernorm expects x, residual, out, out_resid, weight, and bias")
        x_name, residual_name, out_name, out_resid_name, weight_name, bias_name = operation.inputs
        specs = {argument.name: argument.spec for argument in ir.arguments if isinstance(argument.spec, TensorSpec)}
        try:
            x_spec = specs[x_name]
            residual_spec = specs[residual_name]
            out_spec = specs[out_name]
            out_resid_spec = specs[out_resid_name]
            weight_spec = specs[weight_name]
            bias_spec = specs[bias_name]
        except KeyError as exc:
            raise BackendNotImplementedError("hipkittens_exec layernorm expects direct tensor arguments") from exc
        if any(spec.dtype != "bf16" for spec in (x_spec, residual_spec, out_spec, out_resid_spec, weight_spec, bias_spec)):
            raise BackendNotImplementedError("hipkittens_exec layernorm currently requires BF16 tensors")
        if len(x_spec.shape) != 3 or residual_spec.shape != x_spec.shape or out_spec.shape != x_spec.shape or out_resid_spec.shape != x_spec.shape:
            raise BackendNotImplementedError(
                f"hipkittens_exec layernorm requires matching rank-3 x/residual/out/out_resid tensors, got {x_spec.shape}"
            )
        batch, seqlen, hidden = x_spec.shape
        if weight_spec.shape != (hidden,) or bias_spec.shape != (hidden,):
            raise BackendNotImplementedError(
                f"hipkittens_exec layernorm requires weight and bias shapes {(hidden,)}, got {weight_spec.shape} and {bias_spec.shape}"
            )
        if seqlen % 4 != 0:
            raise BackendNotImplementedError(
                f"hipkittens_exec layernorm requires the sequence dimension to be divisible by 4, got {seqlen}"
            )
        epsilon = float(operation.attrs.get("epsilon", 1e-5))
        return HipKittensLayerNormExecMatch(
            x_name=x_name,
            residual_name=residual_name,
            out_name=out_name,
            out_resid_name=out_resid_name,
            weight_name=weight_name,
            bias_name=bias_name,
            shape=(batch, seqlen, hidden),
            epsilon=epsilon,
            reference_path="kernels/layernorm/kernel.cpp",
        )

    def _match_rmsnorm(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        operation: Any,
    ) -> HipKittensRmsNormExecMatch:
        if target.arch == "gfx950":
            raise BackendNotImplementedError(
                "hipkittens_exec rmsnorm remains gfx942-only; current HipKittens gfx950 headers do not compile the "
                "generated rmsnorm kernel, so use hipkittens_ref"
            )
        if target.arch != "gfx942":
            raise BackendNotImplementedError("hipkittens_exec rmsnorm is currently wired only for gfx942/cdna3")
        if len(operation.inputs) != 3:
            raise BackendNotImplementedError("hipkittens_exec rmsnorm expects x, out, and gamma tensors")
        x_name, out_name, gamma_name = operation.inputs
        specs = {argument.name: argument.spec for argument in ir.arguments if isinstance(argument.spec, TensorSpec)}
        try:
            x_spec = specs[x_name]
            out_spec = specs[out_name]
            gamma_spec = specs[gamma_name]
        except KeyError as exc:
            raise BackendNotImplementedError("hipkittens_exec rmsnorm expects direct tensor arguments") from exc
        if any(spec.dtype != "bf16" for spec in (x_spec, out_spec, gamma_spec)):
            raise BackendNotImplementedError("hipkittens_exec rmsnorm currently requires BF16 tensors")
        if len(x_spec.shape) != 3 or out_spec.shape != x_spec.shape or gamma_spec.shape != x_spec.shape:
            raise BackendNotImplementedError(
                f"hipkittens_exec rmsnorm requires matching rank-3 x/out/gamma tensors, got {x_spec.shape}, {out_spec.shape}, {gamma_spec.shape}"
            )
        batch, seqlen, hidden = x_spec.shape
        if seqlen % 4 != 0:
            raise BackendNotImplementedError(
                f"hipkittens_exec rmsnorm requires the sequence dimension to be divisible by 4, got {seqlen}"
            )
        epsilon = float(operation.attrs.get("epsilon", 1e-5))
        return HipKittensRmsNormExecMatch(
            x_name=x_name,
            out_name=out_name,
            gamma_name=gamma_name,
            shape=(batch, seqlen, hidden),
            epsilon=epsilon,
            reference_path="kernels/rmsnorm/kernel.cpp",
        )

    def _match_attention(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        operation: Any,
    ) -> HipKittensAttentionExecMatch:
        root = self._configured_root()
        if target.arch not in {"gfx950", "gfx942"}:
            raise BackendNotImplementedError("hipkittens_exec fused attention is currently wired only for gfx950 and gfx942")
        if len(operation.inputs) != 5:
            raise BackendNotImplementedError("hipkittens_exec attention expects q, k, v, out, and lse tensors")
        q_name, k_name, v_name, out_name, lse_name = operation.inputs
        specs = {argument.name: argument.spec for argument in ir.arguments if isinstance(argument.spec, TensorSpec)}
        try:
            q_spec = specs[q_name]
            k_spec = specs[k_name]
            v_spec = specs[v_name]
            out_spec = specs[out_name]
            lse_spec = specs[lse_name]
        except KeyError as exc:
            raise BackendNotImplementedError("hipkittens_exec attention expects direct tensor arguments") from exc
        if any(spec.dtype != "bf16" for spec in (q_spec, k_spec, v_spec, out_spec)) or lse_spec.dtype != "f32":
            raise BackendNotImplementedError("hipkittens_exec attention requires BF16 q/k/v/out tensors and an F32 lse tensor")
        if len(q_spec.shape) != 4 or len(k_spec.shape) != 4 or len(v_spec.shape) != 4 or len(out_spec.shape) != 4:
            raise BackendNotImplementedError("hipkittens_exec attention requires rank-4 q, k, v, and out tensors")
        if q_spec.shape != out_spec.shape or k_spec.shape != v_spec.shape:
            raise BackendNotImplementedError("hipkittens_exec attention requires q/out and k/v shape pairs to match")
        batch, seqlen, heads, head_dim = q_spec.shape
        kv_batch, kv_seqlen, kv_heads, kv_dim = k_spec.shape
        if (kv_batch, kv_seqlen, kv_dim) != (batch, seqlen, head_dim):
            raise BackendNotImplementedError("hipkittens_exec attention requires compatible q/k/v shapes")
        if lse_spec.shape != (batch, heads, 1, seqlen):
            raise BackendNotImplementedError(
                f"hipkittens_exec attention requires lse shape {(batch, heads, 1, seqlen)}, got {lse_spec.shape}"
            )
        if head_dim != 128:
            raise BackendNotImplementedError(f"hipkittens_exec attention currently requires head_dim=128, got {head_dim}")
        if heads % kv_heads != 0:
            raise BackendNotImplementedError("hipkittens_exec attention requires query heads to be divisible by kv heads")
        if seqlen % 64 != 0 or seqlen < 256:
            raise BackendNotImplementedError(
                f"hipkittens_exec attention currently requires sequence length divisible by 64 and >= 256, got {seqlen}"
            )
        causal = bool(operation.attrs.get("causal", False))
        if target.arch == "gfx950":
            relative_path = "kernels/attn/gqa_causal/kernel.cpp" if causal else "kernels/attn/gqa/kernel.cpp"
            if root is not None and not (root / relative_path).exists():
                raise BackendNotImplementedError(
                    f"hipkittens_exec attention requires {relative_path} in the configured HipKittens root"
                )
        else:
            relative_path = "generated:gfx942_attention"
        return HipKittensAttentionExecMatch(
            q_name=q_name,
            k_name=k_name,
            v_name=v_name,
            out_name=out_name,
            lse_name=lse_name,
            q_shape=q_spec.shape,
            kv_shape=k_spec.shape,
            lse_shape=lse_spec.shape,
            causal=causal,
            reference_path=relative_path,
        )

    def _descriptor_supported_on_target(
        self,
        descriptor: HipKittensExecDescriptor,
        target: AMDTarget,
    ) -> bool:
        if descriptor.operand_dtype == "f16" and (descriptor.transpose_a or descriptor.transpose_b):
            return False
        if target.arch == "gfx942" and descriptor.operand_dtype == "f16":
            return False
        return True

    def _configured_root(self) -> Path | None:
        configured = os.environ.get(_HIPKITTENS_ENV)
        if not configured:
            return None
        path = Path(configured).expanduser().resolve()
        if (path / "include" / "kittens.cuh").exists():
            return path
        return None

    def _require_root(self) -> Path:
        root = self._configured_root()
        if root is None:
            raise BackendNotImplementedError(
                f"set {_HIPKITTENS_ENV} to a HipKittens checkout before launching hipkittens_exec kernels"
            )
        return root

    def _toolchain_ready(self, target: AMDTarget) -> bool:
        if target.arch != "gfx942":
            return True
        for include_dir in self._include_search_roots():
            if (include_dir / "hip_bf16.h").exists() or (include_dir / "hip" / "hip_bf16.h").exists():
                return True
        return False

    def _toolchain_include_args(self, target: AMDTarget) -> list[str]:
        if target.arch != "gfx942":
            return []
        include_args: list[str] = []
        seen: set[Path] = set()
        for include_dir in self._include_search_roots():
            candidates = [include_dir]
            if (include_dir / "hip" / "hip_bf16.h").exists():
                candidates.append(include_dir / "hip")
            if not any((candidate / "hip_bf16.h").exists() or (candidate / "hip" / "hip_bf16.h").exists() for candidate in candidates):
                continue
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                include_args.append(f"-I{candidate}")
        return include_args

    def _include_search_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        rocm_env = os.environ.get("ROCM_PATH")
        if rocm_env:
            roots.append(Path(rocm_env).expanduser() / "include")
        prefix = Path(sys.prefix)
        roots.append(prefix / "include")
        for candidate in glob(str(prefix / "lib" / "python*" / "site-packages" / "_rocm_sdk_*" / "include")):
            roots.append(Path(candidate))
        for candidate in ("/opt/rocm/include", "/usr/include", "/usr/local/include"):
            roots.append(Path(candidate))
        for candidate in sorted(glob("/opt/rocm-*")):
            roots.append(Path(candidate) / "include")
        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            deduped.append(root)
        return tuple(deduped)

    def _arch_define(self, target: AMDTarget) -> str:
        if target.arch == "gfx950":
            return "-DKITTENS_CDNA4"
        if target.arch == "gfx942":
            return "-DKITTENS_CDNA3"
        raise BackendNotImplementedError(f"hipkittens_exec does not support target arch '{target.arch}'")

    def _render_cpp(
        self,
        entry_point: str,
        target: AMDTarget,
        match: HipKittensExecMatch | HipKittensLayerNormExecMatch | HipKittensRmsNormExecMatch | HipKittensAttentionExecMatch,
        root: Path | None,
    ) -> str:
        if isinstance(match, HipKittensLayerNormExecMatch):
            return self._render_layernorm_cpp(entry_point, target, match, root)
        if isinstance(match, HipKittensRmsNormExecMatch):
            return self._render_rmsnorm_cpp(entry_point, target, match, root)
        if isinstance(match, HipKittensAttentionExecMatch):
            return self._render_attention_cpp(entry_point, target, match, root)
        return self._render_gemm_cpp(entry_point, target, match, root)

    def _render_gemm_cpp(
        self,
        entry_point: str,
        target: AMDTarget,
        match: HipKittensExecMatch,
        root: Path | None,
    ) -> str:
        descriptor = match.descriptor
        root_hint = (
            f"// HipKittens root: {root}\n" if root is not None else f"// Set {_HIPKITTENS_ENV} to a HipKittens checkout before launch\n"
        )
        a_load_coords = self._load_coords(descriptor.transpose_a, descriptor.transpose_b, operand="a")
        b_load_coords = self._load_coords(descriptor.transpose_a, descriptor.transpose_b, operand="b")
        return (
            "// Baybridge HipKittens executable backend\n"
            f"// family: {descriptor.family}\n"
            f"// target: {target.arch}\n"
            f"// reference: {descriptor.reference_path}\n"
            f"// full shapes: A={match.a_shape}, B={match.b_shape}, C={match.c_shape}\n"
            f"// tile grid: {match.grid}\n"
            f"// k_tiles: {match.k_tiles}\n"
            f"{root_hint}"
            "#include <cstdint>\n"
            "#include <hip/hip_runtime.h>\n"
            "#include \"kittens.cuh\"\n\n"
            "using namespace kittens;\n\n"
            f"using GLA = gl<{descriptor.gl_operand_dtype}, -1, -1, -1, -1>;\n"
            f"using GLB = gl<{descriptor.gl_operand_dtype}, -1, -1, -1, -1>;\n"
            f"using GLC = gl<{descriptor.gl_accumulator_dtype}, -1, -1, -1, -1>;\n\n"
            f"extern \"C\" __global__ void {entry_point}(GLA {match.a_name}, GLB {match.b_name}, GLC {match.c_name}) {{\n"
            "  const int tile_row = static_cast<int>(blockIdx.y);\n"
            "  const int tile_col = static_cast<int>(blockIdx.x);\n"
            f"  {self._render_rt_type(descriptor.rt_operand_alias, descriptor.a_tile_shape[0], descriptor.a_tile_shape[1], descriptor.a_layout, descriptor.a_rt_shape, target)} a_tile;\n"
            f"  {self._render_rt_type(descriptor.rt_operand_alias, descriptor.b_tile_shape[0], descriptor.b_tile_shape[1], descriptor.b_layout, descriptor.b_rt_shape, target)} b_tile;\n"
            f"  {self._render_rt_type(descriptor.rt_accumulator_alias, descriptor.c_tile_shape[0], descriptor.c_tile_shape[1], descriptor.c_layout, descriptor.c_rt_shape, target)} c_accum;\n"
            "  zero(c_accum);\n"
            f"  for (int k_tile = 0; k_tile < {match.k_tiles}; ++k_tile) {{\n"
            f"    load(a_tile, {match.a_name}, {a_load_coords});\n"
            f"    load(b_tile, {match.b_name}, {b_load_coords});\n"
            f"    {descriptor.mma_statement}\n"
            "  }\n"
            f"  store({match.c_name}, c_accum, {{0, 0, tile_row, tile_col}});\n"
            "}\n\n"
            f"extern \"C\" int launch_{entry_point}({descriptor.host_operand_pointer} {match.a_name}, {descriptor.host_operand_pointer} {match.b_name}, {descriptor.host_accumulator_pointer} {match.c_name}) {{\n"
            f"  auto gl_a = make_gl<GLA>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.a_name})), 1, 1, {match.a_shape[0]}, {match.a_shape[1]});\n"
            f"  auto gl_b = make_gl<GLB>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.b_name})), 1, 1, {match.b_shape[0]}, {match.b_shape[1]});\n"
            f"  auto gl_c = make_gl<GLC>(reinterpret_cast<uint64_t>({match.c_name}), 1, 1, {match.c_shape[0]}, {match.c_shape[1]});\n"
            f"  {entry_point}<<<dim3({match.grid[0]}, {match.grid[1]}, {match.grid[2]}), dim3(kittens::WARP_THREADS, 1, 1)>>>(gl_a, gl_b, gl_c);\n"
            "  return static_cast<int>(hipDeviceSynchronize());\n"
            "}\n"
        )

    def _render_layernorm_cpp(
        self,
        entry_point: str,
        target: AMDTarget,
        match: HipKittensLayerNormExecMatch,
        root: Path | None,
    ) -> str:
        if root is None:
            raise BackendNotImplementedError(
                f"set {_HIPKITTENS_ENV} to a HipKittens checkout before launching hipkittens_exec layernorm kernels"
            )
        source_path = root / match.reference_path
        if not source_path.exists():
            raise BackendNotImplementedError(f"hipkittens_exec could not find the layernorm kernel source at {source_path}")
        batch, seqlen, hidden = match.shape
        source_text = source_path.read_text(encoding="utf-8")
        source_text = source_text.replace('#include "pyutils/pyutils.cuh"\n', "")
        source_text = source_text.replace("#include <rocrand_kernel.h>\n", "")
        source_text = self._strip_pybind_module(source_text)
        dropout_start = source_text.find("template<kittens::ducks::rv::all T>")
        norm_globals_marker = "template<int _d_model> struct norm_globals"
        norm_globals_start = source_text.find(norm_globals_marker)
        if dropout_start != -1 and norm_globals_start != -1 and dropout_start < norm_globals_start:
            source_text = source_text[:dropout_start] + source_text[norm_globals_start:]
        source_text = source_text.replace("constexpr int B = 16;", f"constexpr int B = {batch};")
        source_text = source_text.replace("constexpr int H = 16;", "constexpr int H = 1;")
        source_text = source_text.replace("constexpr int N = 4096;", f"constexpr int N = {seqlen};")
        source_text = source_text.replace("constexpr int HEAD_D = 128;", f"constexpr int HEAD_D = {hidden};")
        source_text = source_text.replace("constexpr float DROPOUT_P = 0.01;", "constexpr float DROPOUT_P = 0.0;")
        source_text = source_text.replace(
            "__float2bfloat16(1e-05f)",
            f"__float2bfloat16({match.epsilon:.9g}f)",
        )
        preamble = (
            "// Baybridge HipKittens executable backend\n"
            "// family: layernorm\n"
            f"// target: {target.arch}\n"
            f"// reference: {match.reference_path}\n"
            f"// full shape: x=residual=out=out_resid={match.shape}, weight=bias=({hidden},)\n"
            f"// HipKittens root: {root}\n\n"
        )
        wrapper = (
            "\nextern \"C\" int launch_"
            f"{entry_point}(const std::uint16_t* {match.x_name}, const std::uint16_t* {match.residual_name}, "
            f"std::uint16_t* {match.out_name}, std::uint16_t* {match.out_resid_name}, "
            f"const std::uint16_t* {match.weight_name}, const std::uint16_t* {match.bias_name}) {{\n"
            f"    auto gl_x = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.x_name})), 1, B, N, D);\n"
            f"    auto gl_residual = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.residual_name})), 1, B, N, D);\n"
            f"    auto gl_out = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>({match.out_name}), 1, B, N, D);\n"
            f"    auto gl_out_resid = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>({match.out_resid_name}), 1, B, N, D);\n"
            f"    auto gl_weight = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.weight_name})), 1, 1, 1, D);\n"
            f"    auto gl_bias = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.bias_name})), 1, 1, 1, D);\n"
            "    norm_globals<D> g{gl_x, gl_residual, gl_out, gl_out_resid, gl_weight, gl_bias};\n"
            "    dispatch_micro<D>(g);\n"
            "    return static_cast<int>(hipDeviceSynchronize());\n"
            "}\n"
        )
        return preamble + source_text + wrapper

    def _render_rmsnorm_cpp(
        self,
        entry_point: str,
        target: AMDTarget,
        match: HipKittensRmsNormExecMatch,
        root: Path | None,
    ) -> str:
        batch, seqlen, hidden = match.shape
        root_hint = (
            f"// HipKittens root: {root}\n" if root is not None else f"// Set {_HIPKITTENS_ENV} to a HipKittens checkout before launch\n"
        )
        return (
            "// Baybridge HipKittens executable backend\n"
            "// family: rmsnorm\n"
            f"// target: {target.arch}\n"
            f"// reference: {match.reference_path}\n"
            f"// full shape: x=out=gamma={match.shape}\n"
            f"{root_hint}"
            "#include <cstdint>\n"
            "#include <hip/hip_runtime.h>\n"
            "#include \"kittens.cuh\"\n\n"
            "#define NUM_WORKERS (4)\n"
            "#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)\n\n"
            "using namespace kittens;\n\n"
            f"constexpr int B = {batch};\n"
            f"constexpr int N = {seqlen};\n"
            f"constexpr int D = {hidden};\n"
            f"constexpr float EPSILON = {match.epsilon:.9g}f;\n\n"
            "template<int _D> struct rmsnorm_globals {\n"
            "    using x_gl = gl<bf16, -1, -1, -1, -1>;\n"
            "    using o_gl = gl<bf16, -1, -1, -1, -1>;\n"
            "    using gamma_gl = gl<bf16, -1, -1, -1, -1>;\n"
            "    x_gl x;\n"
            "    o_gl o;\n"
            "    gamma_gl gamma;\n"
            "    float epsilon;\n"
            "    const int n_per_tile = NUM_WORKERS;\n"
            "    const int n_tile_size = N / n_per_tile;\n"
            "    dim3 grid() { return dim3(n_tile_size, B, 1); }\n"
            "    dim3 block() { return dim3(NUM_THREADS); }\n"
            "    size_t dynamic_shared_memory() { return 0; }\n"
            "};\n\n"
            "template<int _D>\n"
            "__global__ void rmsnorm_hk(const rmsnorm_globals<_D> g) {\n"
            "    auto warpid = kittens::warpid();\n"
            "    const int batch = blockIdx.y;\n"
            "    const int seq_start = blockIdx.x * g.n_per_tile;\n"
            "    const int seq_idx = seq_start + warpid;\n"
            "    rv<bf16, _D> x_reg, gamma_reg, squared_reg;\n"
            "    load(x_reg, g.x, {0, batch, seq_idx, 0});\n"
            "    load(gamma_reg, g.gamma, {0, batch, seq_idx, 0});\n"
            "    asm volatile(\"s_waitcnt vmcnt(0)\");\n"
            "    mul(squared_reg, x_reg, x_reg);\n"
            "    bf16 x_var;\n"
            "    sum(x_var, squared_reg);\n"
            "    float var_f32 = __bfloat162float(x_var) / float(_D);\n"
            "    float inv_rms_f32 = rsqrtf(var_f32 + g.epsilon);\n"
            "    bf16 inv_rms = __float2bfloat16(inv_rms_f32);\n"
            "    mul(x_reg, x_reg, inv_rms);\n"
            "    mul(x_reg, x_reg, gamma_reg);\n"
            "    store(g.o, x_reg, {0, batch, seq_idx, 0});\n"
            "}\n\n"
            "extern \"C\" int launch_"
            f"{entry_point}(const std::uint16_t* {match.x_name}, std::uint16_t* {match.out_name}, const std::uint16_t* {match.gamma_name}) {{\n"
            f"    auto gl_x = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.x_name})), 1, B, N, D);\n"
            f"    auto gl_out = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>({match.out_name}), 1, B, N, D);\n"
            f"    auto gl_gamma = make_gl<gl<bf16, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.gamma_name})), 1, B, N, D);\n"
            "    rmsnorm_globals<D> g{gl_x, gl_out, gl_gamma, EPSILON};\n"
            "    rmsnorm_hk<D><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);\n"
            "    return static_cast<int>(hipDeviceSynchronize());\n"
            "}\n"
        )

    def _render_attention_cpp(
        self,
        entry_point: str,
        target: AMDTarget,
        match: HipKittensAttentionExecMatch,
        root: Path | None,
    ) -> str:
        if target.arch == "gfx942":
            return self._render_attention_cpp_gfx942(entry_point, target, match, root)
        if root is None:
            raise BackendNotImplementedError(
                f"set {_HIPKITTENS_ENV} to a HipKittens checkout before launching hipkittens_exec attention kernels"
            )
        source_path = root / match.reference_path
        if not source_path.exists():
            raise BackendNotImplementedError(f"hipkittens_exec could not find the attention kernel source at {source_path}")
        batch, seqlen, heads, head_dim = match.q_shape
        _, _, kv_heads, _ = match.kv_shape
        source_text = source_path.read_text(encoding="utf-8")
        source_text = source_text.replace('#include "pyutils/pyutils.cuh"\n', "")
        source_text = self._strip_pybind_module(source_text)
        preamble = (
            "// Baybridge HipKittens executable backend\n"
            "// family: attention\n"
            f"// target: {target.arch}\n"
            f"// reference: {match.reference_path}\n"
            f"// q/out shape: {match.q_shape}\n"
            f"// k/v shape: {match.kv_shape}\n"
            f"// lse shape: {match.lse_shape}\n"
            f"// HipKittens root: {root}\n"
            f"#define ATTN_B {batch}\n"
            f"#define ATTN_H {heads}\n"
            f"#define ATTN_H_KV {kv_heads}\n"
            f"#define ATTN_N {seqlen}\n"
            f"#define ATTN_D {head_dim}\n\n"
        )
        wrapper = (
            "\nextern \"C\" int launch_"
            f"{entry_point}(const std::uint16_t* {match.q_name}, const std::uint16_t* {match.k_name}, const std::uint16_t* {match.v_name}, "
            f"std::uint16_t* {match.out_name}, float* {match.lse_name}) {{\n"
            f"    auto gl_q = make_gl<_gl_QKVO>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.q_name})), ATTN_B, ATTN_N, ATTN_H, ATTN_D);\n"
            f"    auto gl_k = make_gl<_gl_QKVO>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.k_name})), ATTN_B, ATTN_N, ATTN_H_KV, ATTN_D);\n"
            f"    auto gl_v = make_gl<_gl_QKVO>(reinterpret_cast<uint64_t>(const_cast<std::uint16_t*>({match.v_name})), ATTN_B, ATTN_N, ATTN_H_KV, ATTN_D);\n"
            f"    auto gl_o = make_gl<_gl_QKVO>(reinterpret_cast<uint64_t>({match.out_name}), ATTN_B, ATTN_N, ATTN_H, ATTN_D);\n"
            f"    auto gl_l = make_gl<gl<float, -1, -1, -1, -1>>(reinterpret_cast<uint64_t>({match.lse_name}), ATTN_B, ATTN_H, 1, ATTN_N);\n"
            "    attn_globals<ATTN_D> g{gl_q, gl_k, gl_v, gl_o, gl_l, nullptr};\n"
            "    dispatch_micro<ATTN_D>(g);\n"
            "    return static_cast<int>(hipDeviceSynchronize());\n"
            "}\n"
        )
        return preamble + source_text + wrapper

    def _render_attention_cpp_gfx942(
        self,
        entry_point: str,
        target: AMDTarget,
        match: HipKittensAttentionExecMatch,
        root: Path | None,
    ) -> str:
        batch, seqlen, heads, head_dim = match.q_shape
        _, _, kv_heads, _ = match.kv_shape
        causal_literal = "true" if match.causal else "false"
        root_hint = (
            f"// HipKittens root: {root}\n" if root is not None else f"// Set {_HIPKITTENS_ENV} to a HipKittens checkout before launch\n"
        )
        return (
            "// Baybridge HipKittens executable backend\n"
            "// family: attention\n"
            f"// target: {target.arch}\n"
            "// reference: generated:gfx942_attention\n"
            f"// q/out shape: {match.q_shape}\n"
            f"// k/v shape: {match.kv_shape}\n"
            f"// lse shape: {match.lse_shape}\n"
            f"{root_hint}"
            "#include <cstdint>\n"
            "#include <cmath>\n"
            "#include <hip/hip_runtime.h>\n"
            "#include <hip/hip_bfloat16.h>\n\n"
            f"constexpr int ATTN_B = {batch};\n"
            f"constexpr int ATTN_N = {seqlen};\n"
            f"constexpr int ATTN_H = {heads};\n"
            f"constexpr int ATTN_H_KV = {kv_heads};\n"
            f"constexpr int ATTN_D = {head_dim};\n"
            f"constexpr bool ATTN_CAUSAL = {causal_literal};\n"
            "constexpr int ATTN_GROUP_SIZE = ATTN_H / ATTN_H_KV;\n\n"
            "__device__ inline size_t q_offset(int b, int n, int h, int d) {\n"
            "    return static_cast<size_t>((((b * ATTN_N) + n) * ATTN_H + h) * ATTN_D + d);\n"
            "}\n"
            "__device__ inline size_t kv_offset(int b, int n, int h, int d) {\n"
            "    return static_cast<size_t>((((b * ATTN_N) + n) * ATTN_H_KV + h) * ATTN_D + d);\n"
            "}\n"
            "__device__ inline size_t lse_offset(int b, int h, int n) {\n"
            "    return static_cast<size_t>(((b * ATTN_H) + h) * ATTN_N + n);\n"
            "}\n\n"
            "__global__ void baybridge_attention_gfx942(\n"
            "    const hip_bfloat16* q,\n"
            "    const hip_bfloat16* k,\n"
            "    const hip_bfloat16* v,\n"
            "    hip_bfloat16* out,\n"
            "    float* lse) {\n"
            "    const int query_idx = blockIdx.x;\n"
            "    const int head = blockIdx.y;\n"
            "    const int batch = blockIdx.z;\n"
            "    if (threadIdx.x != 0) {\n"
            "        return;\n"
            "    }\n"
            "    const int kv_head = head / ATTN_GROUP_SIZE;\n"
            "    const float scale = 1.0f / sqrtf(static_cast<float>(ATTN_D));\n"
            "    float scores[ATTN_N];\n"
            "    float max_score = -INFINITY;\n"
            "    for (int key_idx = 0; key_idx < ATTN_N; ++key_idx) {\n"
            "        float score = 0.0f;\n"
            "        if (!ATTN_CAUSAL || key_idx <= query_idx) {\n"
            "            for (int dim_idx = 0; dim_idx < ATTN_D; ++dim_idx) {\n"
            "                const float qv = static_cast<float>(q[q_offset(batch, query_idx, head, dim_idx)]);\n"
            "                const float kv = static_cast<float>(k[kv_offset(batch, key_idx, kv_head, dim_idx)]);\n"
            "                score += qv * kv;\n"
            "            }\n"
            "            score *= scale;\n"
            "        } else {\n"
            "            score = -INFINITY;\n"
            "        }\n"
            "        scores[key_idx] = score;\n"
            "        if (score > max_score) {\n"
            "            max_score = score;\n"
            "        }\n"
            "    }\n"
            "    float denom = 0.0f;\n"
            "    for (int key_idx = 0; key_idx < ATTN_N; ++key_idx) {\n"
            "        const float weight = isinf(scores[key_idx]) && scores[key_idx] < 0.0f ? 0.0f : expf(scores[key_idx] - max_score);\n"
            "        scores[key_idx] = weight;\n"
            "        denom += weight;\n"
            "    }\n"
            "    lse[lse_offset(batch, head, query_idx)] = logf(denom) + max_score;\n"
            "    for (int dim_idx = 0; dim_idx < ATTN_D; ++dim_idx) {\n"
            "        float acc = 0.0f;\n"
            "        for (int key_idx = 0; key_idx < ATTN_N; ++key_idx) {\n"
            "            if (scores[key_idx] == 0.0f) {\n"
            "                continue;\n"
            "            }\n"
            "            const float vv = static_cast<float>(v[kv_offset(batch, key_idx, kv_head, dim_idx)]);\n"
            "            acc += (scores[key_idx] / denom) * vv;\n"
            "        }\n"
            "        out[q_offset(batch, query_idx, head, dim_idx)] = hip_bfloat16(acc);\n"
            "    }\n"
            "}\n\n"
            "extern \"C\" int launch_"
            f"{entry_point}(const std::uint16_t* {match.q_name}, const std::uint16_t* {match.k_name}, const std::uint16_t* {match.v_name}, "
            f"std::uint16_t* {match.out_name}, float* {match.lse_name}) {{\n"
            f"    auto* q_ptr = reinterpret_cast<const hip_bfloat16*>({match.q_name});\n"
            f"    auto* k_ptr = reinterpret_cast<const hip_bfloat16*>({match.k_name});\n"
            f"    auto* v_ptr = reinterpret_cast<const hip_bfloat16*>({match.v_name});\n"
            f"    auto* out_ptr = reinterpret_cast<hip_bfloat16*>({match.out_name});\n"
            f"    auto* lse_ptr = reinterpret_cast<float*>({match.lse_name});\n"
            "    baybridge_attention_gfx942<<<dim3(ATTN_N, ATTN_H, ATTN_B), dim3(1, 1, 1)>>>(q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr);\n"
            "    return static_cast<int>(hipDeviceSynchronize());\n"
            "}\n"
        )

    def _render_rt_type(
        self,
        alias: str,
        rows: int,
        cols: int,
        layout: str,
        rt_shape: str,
        target: AMDTarget,
    ) -> str:
        if target.arch == "gfx942":
            return f"{alias}<{rows}, {cols}, {layout}>"
        return f"{alias}<{rows}, {cols}, {layout}, {rt_shape}>"

    def _load_coords(self, transpose_a: bool, transpose_b: bool, *, operand: str) -> str:
        if operand == "a":
            if transpose_a:
                return "{0, 0, k_tile, tile_row}"
            return "{0, 0, tile_row, k_tile}"
        if operand == "b":
            if transpose_b:
                return "{0, 0, tile_col, k_tile}"
            return "{0, 0, k_tile, tile_col}"
        raise ValueError(f"unknown operand '{operand}'")

    def _strip_pybind_module(self, source_text: str) -> str:
        marker = "PYBIND11_MODULE("
        index = source_text.find(marker)
        if index == -1:
            return source_text
        return source_text[:index].rstrip() + "\n"
