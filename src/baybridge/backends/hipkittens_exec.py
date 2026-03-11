from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..hip_runtime import HipRuntime, require_hipcc, scalar_ctype
from ..ir import KernelArgument, PortableKernelIR, TensorSpec
from ..runtime import RuntimeTensor
from ..target import AMDTarget

_HIPKITTENS_ENV = "BAYBRIDGE_HIPKITTENS_ROOT"


@dataclass(frozen=True)
class HipKittensExecDescriptor:
    family: str
    operand_dtype: str
    accumulator_dtype: str
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


_DESCRIPTORS = (
    HipKittensExecDescriptor(
        family="bf16_gemm_32x16x32",
        operand_dtype="bf16",
        accumulator_dtype="f32",
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
        family="bf16_gemm_16x32x16",
        operand_dtype="bf16",
        accumulator_dtype="f32",
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
        family="f16_gemm_32x16x32",
        operand_dtype="f16",
        accumulator_dtype="f32",
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
        family="f16_gemm_16x32x16",
        operand_dtype="f16",
        accumulator_dtype="f32",
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

    def available(self) -> bool:
        return self._configured_root() is not None

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
            if kwargs:
                raise TypeError("hipkittens_exec launcher only supports positional arguments")
            if len(args) != len(ir.arguments):
                raise TypeError(f"{ir.name} expects {len(ir.arguments)} arguments, got {len(args)}")
            if not shared_path.exists():
                self._compile_shared_object(source_path, shared_path, target, lowered_module.text, root)
            function = state.get("function")
            if function is None:
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
            str(source_path),
            "-o",
            str(shared_path),
        ]
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
                if not isinstance(value, RuntimeTensor):
                    raise TypeError(
                        f"hipkittens_exec expects RuntimeTensor values for tensor argument '{argument.name}', got {type(value).__name__}"
                    )
                allocation = hip.upload_tensor(value)
                tensor_allocations.append(allocation)
                c_args.append(allocation.ptr)
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

    def _match(self, ir: PortableKernelIR, target: AMDTarget) -> HipKittensExecMatch:
        if target.arch != "gfx950":
            raise BackendNotImplementedError(
                f"hipkittens_exec currently supports gfx950 only, got target arch '{target.arch}'"
            )
        mma_ops = [operation for operation in ir.operations if operation.op == "mma"]
        if len(mma_ops) != 1:
            raise BackendNotImplementedError("hipkittens_exec currently supports exactly one mma op")
        if len(ir.operations) != 1:
            raise BackendNotImplementedError("hipkittens_exec currently supports pure GEMM kernels without extra ops")
        mma = mma_ops[0]
        a_name, b_name, c_name = mma.inputs
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
        if a_spec.shape[1] != b_spec.shape[0] or c_spec.shape != (a_spec.shape[0], b_spec.shape[1]):
            raise BackendNotImplementedError(
                f"hipkittens_exec requires GEMM-compatible shapes, got {a_spec.shape} x {b_spec.shape} -> {c_spec.shape}"
            )
        candidates: list[HipKittensExecMatch] = []
        for descriptor in _DESCRIPTORS:
            if a_spec.dtype != descriptor.operand_dtype or c_spec.dtype != descriptor.accumulator_dtype:
                continue
            tile_m, tile_k = descriptor.a_tile_shape
            b_k, tile_n = descriptor.b_tile_shape
            if tile_k != b_k:
                continue
            if a_spec.shape[0] % tile_m != 0:
                continue
            if b_spec.shape[1] % tile_n != 0:
                continue
            if a_spec.shape[1] % tile_k != 0:
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
                    grid=(b_spec.shape[1] // tile_n, a_spec.shape[0] // tile_m, 1),
                    k_tiles=a_spec.shape[1] // tile_k,
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
            )
        )

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

    def _arch_define(self, target: AMDTarget) -> str:
        if target.arch == "gfx950":
            return "-DKITTENS_CDNA4"
        raise BackendNotImplementedError(f"hipkittens_exec does not support target arch '{target.arch}'")

    def _render_cpp(
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
        tile_m, tile_k = descriptor.a_tile_shape
        _, tile_n = descriptor.b_tile_shape
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
            f"  {descriptor.rt_operand_alias}<{tile_m}, {tile_k}, {descriptor.a_layout}, {descriptor.a_rt_shape}> a_tile;\n"
            f"  {descriptor.rt_operand_alias}<{tile_k}, {tile_n}, {descriptor.b_layout}, {descriptor.b_rt_shape}> b_tile;\n"
            f"  {descriptor.rt_accumulator_alias}<{descriptor.c_tile_shape[0]}, {descriptor.c_tile_shape[1]}, {descriptor.c_layout}, {descriptor.c_rt_shape}> c_accum;\n"
            "  zero(c_accum);\n"
            f"  for (int k_tile = 0; k_tile < {match.k_tiles}; ++k_tile) {{\n"
            f"    load(a_tile, {match.a_name}, {{0, 0, tile_row, k_tile}});\n"
            f"    load(b_tile, {match.b_name}, {{0, 0, k_tile, tile_col}});\n"
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
