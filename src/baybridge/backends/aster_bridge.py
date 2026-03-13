from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..backend import LoweredModule
from ..ir import PortableKernelIR
from ..target import AMDTarget
from .gpu_mlir import GpuMlirBackend

_ASTER_ROOT_ENV = "BAYBRIDGE_ASTER_ROOT"


@dataclass(frozen=True)
class AsterEnvironment:
    configured_root: str | None
    aster_opt: str | None
    aster_translate: str | None
    python_package_root: str | None
    runtime_module_available: bool
    ready: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class AsterMatch:
    family: str
    reference_paths: tuple[str, ...]
    notes: tuple[str, ...]
    op_counts: dict[str, int]


class AsterBridge:
    def __init__(self) -> None:
        self._gpu_mlir = GpuMlirBackend()

    def lower_mlir(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        lowered = self._gpu_mlir.lower(ir, target)
        return LoweredModule(
            backend_name=lowered.backend_name,
            entry_point=lowered.entry_point,
            dialect=lowered.dialect,
            text=self._annotate_module(lowered.text, target),
        )

    def environment(self) -> AsterEnvironment:
        configured_root = self.configured_root()
        python_package_root = self.python_package_root()
        aster_opt = self.tool_path("aster-opt")
        aster_translate = self.tool_path("aster-translate")
        runtime_module_available = self._find_spec("aster.hip") is not None
        notes: list[str] = []
        if configured_root is None:
            notes.append(f"set {_ASTER_ROOT_ENV} to an ASTER checkout or install root to improve tool discovery")
        if aster_opt is None:
            notes.append("aster-opt was not found")
        if aster_translate is None:
            notes.append("aster-translate was not found")
        if python_package_root is None:
            notes.append("importable 'aster' python package was not found")
        ready = aster_opt is not None and aster_translate is not None
        return AsterEnvironment(
            configured_root=str(configured_root) if configured_root is not None else None,
            aster_opt=aster_opt,
            aster_translate=aster_translate,
            python_package_root=str(python_package_root) if python_package_root is not None else None,
            runtime_module_available=runtime_module_available,
            ready=ready,
            notes=tuple(notes),
        )

    def analyze(self, ir: PortableKernelIR) -> AsterMatch:
        op_counts = Counter(operation.op for operation in ir.operations)
        notes: list[str] = []
        if op_counts.get("copy_async"):
            notes.append("async copy operations are present")
        if op_counts.get("barrier"):
            notes.append("barrier operations are present")
        if op_counts.get("make_tensor"):
            notes.append("explicit tensor allocation or staging is present")
        if op_counts.get("local_tile") or op_counts.get("partition"):
            notes.append("tiled view operations are present")
        if op_counts.get("mma"):
            return AsterMatch(
                family="mfma_gemm",
                reference_paths=(
                    "test/integration/mfma-e2e.mlir",
                    "test/integration/mfma-32x32-e2e.mlir",
                    "mlir_kernels/gemm_sched_1wave_dword4_mxnxk_16x16x16_f16f16f32.mlir",
                ),
                notes=tuple(notes),
                op_counts=dict(sorted(op_counts.items())),
            )
        if any(name.startswith("reduce_") for name in op_counts):
            return AsterMatch(
                family="elementwise_reduce",
                reference_paths=(
                    "mlir_kernels/test/unit/test_loops.mlir",
                    "mlir_kernels/test/unit/test_indexing.mlir",
                ),
                notes=tuple(notes),
                op_counts=dict(sorted(op_counts.items())),
            )
        if op_counts.get("barrier") or any(
            operation.attrs.get("address_space") == "shared" for operation in ir.operations if operation.op == "make_tensor"
        ):
            return AsterMatch(
                family="shared_staging",
                reference_paths=(
                    "test/integration/g2s-load-lds-e2e.mlir",
                    "mlir_kernels/test/unit/test_global_to_lds_and_back.mlir",
                ),
                notes=tuple(notes),
                op_counts=dict(sorted(op_counts.items())),
            )
        return AsterMatch(
            family="elementwise",
            reference_paths=(
                "mlir_kernels/test/test_copy_1d.py",
                "mlir_kernels/test/unit/test_global_load_wave.mlir",
            ),
            notes=tuple(notes),
            op_counts=dict(sorted(op_counts.items())),
        )

    def configured_root(self) -> Path | None:
        configured = os.environ.get(_ASTER_ROOT_ENV)
        if not configured:
            return None
        return Path(configured).expanduser().resolve()

    def python_package_root(self) -> Path | None:
        configured_root = self.configured_root()
        if configured_root is not None:
            for relative in (
                (".aster", "python_packages", "aster"),
                ("build", "python_packages", "aster"),
                ("python", "aster"),
            ):
                candidate = configured_root.joinpath(*relative)
                if candidate.exists():
                    return candidate
        spec = self._find_spec("aster")
        if spec is None:
            return None
        locations = list(spec.submodule_search_locations or [])
        if locations:
            return Path(locations[0]).resolve()
        if spec.origin:
            return Path(spec.origin).resolve().parent
        return None

    def tool_path(self, name: str) -> str | None:
        configured_root = self.configured_root()
        if configured_root is not None:
            for relative in (
                ("build", "bin", name),
                (".aster", "bin", name),
                ("bin", name),
            ):
                candidate = configured_root.joinpath(*relative)
                if candidate.exists():
                    return str(candidate)
        executable = Path(sys.executable)
        for base in (executable.parent, executable.parent.parent):
            candidate = base / name
            if candidate.exists():
                return str(candidate)
        return shutil.which(name)

    def _find_spec(self, name: str):
        try:
            return importlib.util.find_spec(name)
        except ModuleNotFoundError:
            return None

    def write_repro_bundle(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        lowered_path: Path,
    ) -> Path:
        environment = self.environment()
        match = self.analyze(ir)
        bundle_dir = self._bundle_root(lowered_path)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        kernel_mlir_path = bundle_dir / "kernel.mlir"
        run_script_path = bundle_dir / "repro.sh"
        manifest_path = bundle_dir / "manifest.json"

        kernel_mlir_path.write_text(lowered_module.text, encoding="utf-8")
        run_script_path.write_text(
            self._render_repro_script(match, environment, target),
            encoding="utf-8",
        )
        run_script_path.chmod(0o755)
        manifest = {
            "entry_point": ir.name,
            "family": match.family,
            "target": target.to_dict(),
            "configured_root": environment.configured_root,
            "python_package_root": environment.python_package_root,
            "aster_opt": environment.aster_opt,
            "aster_translate": environment.aster_translate,
            "runtime_module_available": environment.runtime_module_available,
            "notes": list(environment.notes),
            "op_counts": match.op_counts,
            "reference_paths": list(match.reference_paths),
            "kernel_mlir": str(kernel_mlir_path),
            "repro_script": str(run_script_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return bundle_dir

    def _annotate_module(self, text: str, target: AMDTarget) -> str:
        attrs = f'aster.target = "{target.arch}", aster.wave_size = {target.wave_size}, '
        if text.startswith("module attributes {"):
            return text.replace("module attributes {", "module attributes {" + attrs, 1)
        if text.startswith("module {"):
            return text.replace("module {", "module attributes {" + attrs + "} {", 1)
        return text

    def _bundle_root(self, lowered_path: Path) -> Path:
        stem = lowered_path.name
        while "." in stem:
            stem = stem.rsplit(".", 1)[0]
        return lowered_path.parent / f"{stem}.aster_repro"

    def _render_repro_script(
        self,
        match: AsterMatch,
        environment: AsterEnvironment,
        target: AMDTarget,
    ) -> str:
        aster_opt = environment.aster_opt or "aster-opt"
        aster_translate = environment.aster_translate or "aster-translate"
        reference_paths = "\n".join(f"echo '  - {path}'" for path in match.reference_paths)
        return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
ASTER_OPT="${{ASTER_OPT:-{aster_opt}}}"
ASTER_TRANSLATE="${{ASTER_TRANSLATE:-{aster_translate}}}"

echo "Baybridge ASTER bundle"
echo "  family: {match.family}"
echo "  target: {target.arch}"
echo "  kernel_mlir: $ROOT/kernel.mlir"
echo "Baybridge currently emits GPU MLIR reference input, not ASTER AMDGCN IR."
echo "Use this bundle to derive the next lowering step against ASTER."
echo "Relevant upstream ASTER references:"
{reference_paths if reference_paths else "true"}
echo
echo "Suggested next step after adapting kernel.mlir to ASTER AMDGCN IR:"
echo "  $ASTER_OPT \\\"$ROOT/kernel.mlir\\\" --verify-roundtrip"
echo "  $ASTER_TRANSLATE \\\"$ROOT/kernel.mlir\\\" --mlir-to-asm > \\\"$ROOT/kernel.s\\\""
"""
