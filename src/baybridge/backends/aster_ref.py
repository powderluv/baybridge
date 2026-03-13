import json
from pathlib import Path

from ..backend import LoweredModule
from ..ir import PortableKernelIR
from ..target import AMDTarget
from .aster_bridge import AsterBridge


class AsterRefBackend:
    name = "aster_ref"

    def __init__(self) -> None:
        self._bridge = AsterBridge()

    def available(self, target: AMDTarget | None = None) -> bool:
        del target
        environment = self._bridge.environment()
        return (
            environment.configured_root is not None
            or environment.aster_opt is not None
            or environment.aster_translate is not None
            or environment.python_package_root is not None
        )

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        try:
            self._bridge.lower_mlir(ir, target)
            self._bridge.analyze(ir)
        except Exception:
            return False
        return True

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        lowered = self._bridge.lower_mlir(ir, target)
        environment = self._bridge.environment()
        match = self._bridge.analyze(ir)
        header_lines = [
            "// baybridge.aster_ref",
            f"// target: {target.arch}",
            f"// wave_size: {target.wave_size}",
            f"// family: {match.family}",
        ]
        if environment.configured_root is not None:
            header_lines.append(f"// configured_root: {environment.configured_root}")
        else:
            header_lines.append("// configured_root: unset (BAYBRIDGE_ASTER_ROOT)")
        if environment.python_package_root is not None:
            header_lines.append(f"// python_package_root: {environment.python_package_root}")
        else:
            header_lines.append("// python_package_root: not found")
        if environment.aster_opt is not None:
            header_lines.append(f"// aster_opt: {environment.aster_opt}")
        else:
            header_lines.append("// aster_opt: not found on PATH")
        if environment.aster_translate is not None:
            header_lines.append(f"// aster_translate: {environment.aster_translate}")
        else:
            header_lines.append("// aster_translate: not found on PATH")
        if environment.runtime_module_available:
            header_lines.append("// runtime_module: available via python package")
        else:
            header_lines.append("// runtime_module: not importable")
        header_lines.append("// reference_paths:")
        header_lines.extend(f"//   - {path}" for path in match.reference_paths)
        if match.notes:
            header_lines.append("// notes:")
            header_lines.extend(f"//   - {note}" for note in match.notes)
        header_lines.extend(
            [
                "// boundary:",
                "//   Baybridge currently emits GPU MLIR reference input, not ASTER AMDGCN IR.",
                "//   The next integration slice is a dedicated Baybridge->ASTER lowering pass.",
                "// suggested_pipeline:",
                '//   aster-opt kernel.mlir --verify-roundtrip',
                '//   aster-translate kernel.mlir --mlir-to-asm > kernel.s',
            ]
        )
        manifest = {
            "backend": self.name,
            "entry_point": ir.name,
            "family": match.family,
            "target_arch": target.arch,
            "wave_size": target.wave_size,
            "configured_aster_root": environment.configured_root,
            "python_package_root": environment.python_package_root,
            "aster_opt": environment.aster_opt,
            "aster_translate": environment.aster_translate,
            "runtime_module_available": environment.runtime_module_available,
            "op_counts": match.op_counts,
            "reference_paths": list(match.reference_paths),
            "notes": list(match.notes),
        }
        manifest_lines = [f"// {line}" for line in json.dumps(manifest, indent=2, sort_keys=True).splitlines()]
        text = "\n".join(header_lines + manifest_lines) + "\n" + lowered.text
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="aster_mlir",
            text=text,
        )

    def emit_bundle(
        self,
        ir: PortableKernelIR,
        target: AMDTarget,
        lowered_module: LoweredModule,
        lowered_path: Path,
    ) -> Path:
        return self._bridge.write_repro_bundle(ir, target, lowered_module, lowered_path)
