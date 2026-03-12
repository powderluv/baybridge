from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..backend import LoweredModule
from ..diagnostics import BackendNotImplementedError
from ..ir import PortableKernelIR
from ..target import AMDTarget
from .gpu_mlir import GpuMlirBackend

_WAVEASM_ROOT_ENV = "BAYBRIDGE_WAVEASM_ROOT"


@dataclass(frozen=True)
class WaveAsmEnvironment:
    configured_root: str | None
    waveasm_translate: str | None
    clang_driver: str | None
    ld_lld: str | None
    ready: bool
    notes: tuple[str, ...]


class WaveAsmBridge:
    def __init__(self) -> None:
        self._gpu_mlir = GpuMlirBackend()

    def lower_mlir(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        lowered = self._gpu_mlir.lower(ir, target)
        return LoweredModule(
            backend_name=lowered.backend_name,
            entry_point=lowered.entry_point,
            dialect=lowered.dialect,
            text=self._sanitize_mlir_for_waveasm(lowered.text),
        )

    def environment(self) -> WaveAsmEnvironment:
        configured_root = self.configured_root()
        waveasm_translate = self.tool_path("waveasm-translate")
        clang_driver = self.tool_path("clang++") or self.tool_path("clang")
        ld_lld = self.tool_path("ld.lld")
        notes: list[str] = []
        if configured_root is None:
            notes.append(f"set {_WAVEASM_ROOT_ENV} to a Wave checkout to improve tool discovery")
        if waveasm_translate is None:
            notes.append("waveasm-translate was not found")
        if clang_driver is None:
            notes.append("clang++/clang was not found")
        ready = waveasm_translate is not None and clang_driver is not None
        return WaveAsmEnvironment(
            configured_root=str(configured_root) if configured_root is not None else None,
            waveasm_translate=waveasm_translate,
            clang_driver=clang_driver,
            ld_lld=ld_lld,
            ready=ready,
            notes=tuple(notes),
        )

    def configured_root(self) -> Path | None:
        configured = os.environ.get(_WAVEASM_ROOT_ENV)
        if not configured:
            return None
        return Path(configured).expanduser()

    def _sanitize_mlir_for_waveasm(self, text: str) -> str:
        sanitized = re.sub(r"^module attributes \{[^}]*\} \{", "module {", text, count=1, flags=re.MULTILINE)
        sanitized = re.sub(
            r"(\s*gpu\.module @[^ ]+) attributes \{[^}]*\} \{",
            r"\1 {",
            sanitized,
            flags=re.MULTILINE,
        )
        sanitized = re.sub(
            r"(gpu\.func @[^(]+\([^)]*\)) kernel attributes \{[^}]*\} \{",
            r"\1 kernel {",
            sanitized,
            flags=re.MULTILINE,
        )
        sanitized = re.sub(
            r"memref<([^,>]+(?:x[^,>]+)*), strided<\[[^]]+\](?:, offset: [^>]+)?>, 1>",
            r"memref<\1>",
            sanitized,
        )
        sanitized = re.sub(
            r"memref<([^,>]+(?:x[^,>]+)*), strided<\[[^]]+\](?:, offset: [^>]+)?>, 3>",
            r"memref<\1, #gpu.address_space<workgroup>>",
            sanitized,
        )
        sanitized = re.sub(
            r"memref<([^,>]+(?:x[^,>]+)*), strided<\[[^]]+\](?:, offset: [^>]+)?>, 5>",
            r"memref<\1, #gpu.address_space<private>>",
            sanitized,
        )
        return sanitized

    def tool_path(self, name: str) -> str | None:
        configured_root = self.configured_root()
        if configured_root is not None:
            for relative in (
                ("build", "bin", name),
                ("build-wave", "bin", name),
                ("waveasm", "build", "bin", name),
                ("llvm-project", "build", "bin", name),
                ("llvm-install", "bin", name),
            ):
                candidate = configured_root.joinpath(*relative)
                if candidate.exists():
                    return str(candidate)
            for relative in (
                ("llvm-project", "mlir_install", "bin", name),
                ("..", "llvm-project", "mlir_install", "bin", name),
                ("..", "llvm-project", "build", "bin", name),
            ):
                candidate = configured_root.joinpath(*relative).resolve()
                if candidate.exists():
                    return str(candidate)
        executable = Path(sys.executable)
        for base in (executable.parent.parent, executable.parent.parent.parent):
            for candidate in base.glob(f"lib/python*/site-packages/_rocm_sdk_*/lib/llvm/bin/{name}"):
                if candidate.exists():
                    return str(candidate)
        return shutil.which(name)

    def compile_to_hsaco(
        self,
        source_path: Path,
        hsaco_path: Path,
        target: AMDTarget,
        mlir_text: str,
        *,
        workgroup_size: tuple[int, int, int] | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        environment = self.environment()
        if not environment.ready:
            details = "; ".join(environment.notes) or "WaveASM toolchain is not ready"
            raise BackendNotImplementedError(
                f"waveasm_exec requires waveasm-translate and clang++/clang; {details}"
            )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(mlir_text, encoding="utf-8")
        waveasm_ir_path = source_path.with_suffix(".waveasm.mlir")
        asm_path = source_path.with_suffix(".s")
        object_path = source_path.with_suffix(".o")
        assert environment.waveasm_translate is not None
        assert environment.clang_driver is not None
        mlir_translate_command = [
            environment.waveasm_translate,
            f"--target={target.arch}",
        ]
        mlir_translate_command.append(str(source_path))
        waveasm_ir_text = self._run(
            mlir_translate_command,
            op="waveasm-translate (mlir->waveasm)",
        )
        waveasm_ir_path.write_text(waveasm_ir_text, encoding="utf-8")
        assembly_command = [
            environment.waveasm_translate,
            f"--target={target.arch}",
            "--mlir-cse",
            "--waveasm-scoped-cse",
            "--waveasm-peephole",
            "--waveasm-scale-pack-elimination",
            "--loop-invariant-code-motion",
            "--waveasm-m0-redundancy-elim",
            "--waveasm-buffer-load-strength-reduction",
            "--waveasm-memory-offset-opt",
            "--canonicalize",
            "--waveasm-scoped-cse",
            "--waveasm-loop-address-promotion",
            "--waveasm-linear-scan=max-vgprs=512 max-agprs=512",
            "--waveasm-insert-waitcnt=ticketed-waitcnt=true",
            f"--waveasm-hazard-mitigation=target={target.arch}",
            "--emit-assembly",
        ]
        assembly_command.append(str(waveasm_ir_path))
        asm_text = self._run(
            assembly_command,
            op="waveasm-translate (waveasm->assembly)",
        )
        asm_path.write_text(asm_text, encoding="utf-8")
        self._run(
            [
                environment.clang_driver,
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                "-mcode-object-version=5",
                f"-mcpu={target.arch}",
                f"-mwavefrontsize{target.wave_size}",
                "-c",
                str(asm_path),
                "-o",
                str(object_path),
            ],
            op="clang assemble",
            extra_env=self._command_env(environment),
        )
        if environment.ld_lld is not None:
            self._run(
                [
                    environment.ld_lld,
                    "--no-undefined",
                    "-shared",
                    "-o",
                    str(hsaco_path),
                    str(object_path),
                ],
                op="ld.lld link",
            )
        else:
            self._run(
                [
                    environment.clang_driver,
                    "-target",
                    "amdgcn-amd-amdhsa",
                    "-Xlinker",
                    "--build-id=sha1",
                    "-o",
                    str(hsaco_path),
                    str(object_path),
                ],
                op="clang link",
                extra_env=self._command_env(environment),
            )
        return waveasm_ir_path, asm_path, object_path, hsaco_path

    def _command_env(self, environment: WaveAsmEnvironment) -> dict[str, str] | None:
        if environment.ld_lld is None:
            return None
        env = os.environ.copy()
        linker_dir = str(Path(environment.ld_lld).resolve().parent)
        current_path = env.get("PATH", "")
        env["PATH"] = linker_dir if not current_path else linker_dir + os.pathsep + current_path
        return env

    def _run(self, command: list[str], *, op: str, extra_env: dict[str, str] | None = None) -> str:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, env=extra_env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{op} failed with exit code {exc.returncode}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc
        return result.stdout
