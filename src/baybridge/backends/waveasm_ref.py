from ..backend import LoweredModule
from ..ir import PortableKernelIR
from ..target import AMDTarget
from .waveasm_bridge import WaveAsmBridge


class WaveAsmRefBackend:
    name = "waveasm_ref"

    def __init__(self) -> None:
        self._bridge = WaveAsmBridge()

    def available(self, target: AMDTarget | None = None) -> bool:
        del target
        environment = self._bridge.environment()
        return environment.configured_root is not None or environment.waveasm_translate is not None

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        try:
            self._bridge.lower_mlir(ir, target)
        except Exception:
            return False
        return True

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        lowered = self._bridge.lower_mlir(ir, target)
        environment = self._bridge.environment()
        module_text = self._annotate_module(lowered.text, target)
        header_lines = [
            "// baybridge.waveasm_ref",
            f"// target: {target.arch}",
            f"// wave_size: {target.wave_size}",
        ]
        if environment.configured_root is not None:
            header_lines.append(f"// configured_root: {environment.configured_root}")
        else:
            header_lines.append("// configured_root: unset (BAYBRIDGE_WAVEASM_ROOT)")
        if environment.waveasm_translate is not None:
            header_lines.append(f"// waveasm_translate: {environment.waveasm_translate}")
        else:
            header_lines.append("// waveasm_translate: not found on PATH")
        if environment.clang_driver is not None:
            header_lines.append(f"// clang_driver: {environment.clang_driver}")
        else:
            header_lines.append("// clang_driver: not found on PATH")
        header_lines.extend(
            [
                "// suggested_pipeline:",
                (
                    f"//   waveasm-translate --target={target.arch} --mlir-cse --waveasm-scoped-cse "
                    f"--waveasm-peephole --waveasm-scale-pack-elimination --loop-invariant-code-motion "
                    f"--waveasm-m0-redundancy-elim --waveasm-buffer-load-strength-reduction "
                    f"--waveasm-memory-offset-opt --canonicalize --waveasm-scoped-cse "
                    f"--waveasm-loop-address-promotion --waveasm-linear-scan=max-vgprs=512 max-agprs=512 "
                    f"--waveasm-insert-waitcnt=ticketed-waitcnt=true "
                    f"--waveasm-hazard-mitigation=target={target.arch} --emit-assembly "
                    f"--workgroup-size-x={ir.launch.block[0]} --workgroup-size-y={ir.launch.block[1]} "
                    f"--workgroup-size-z={ir.launch.block[2]} kernel.mlir > kernel.s"
                ),
                (
                    f"//   clang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=5 "
                    f"-mcpu={target.arch} -mwavefrontsize{target.wave_size} -c kernel.s -o kernel.o"
                ),
                "//   ld.lld --no-undefined -shared -o kernel.hsaco kernel.o",
            ]
        )
        text = "\n".join(header_lines) + "\n" + module_text
        return LoweredModule(
            backend_name=self.name,
            entry_point=ir.name,
            dialect="waveasm_mlir",
            text=text,
        )

    def _annotate_module(self, text: str, target: AMDTarget) -> str:
        attrs = f'waveasm.target = "{target.arch}", waveasm.wave_size = {target.wave_size}, '
        if text.startswith("module attributes {"):
            return text.replace("module attributes {", "module attributes {" + attrs, 1)
        if text.startswith("module {"):
            return text.replace("module {", "module attributes {" + attrs + "} {", 1)
        return text
