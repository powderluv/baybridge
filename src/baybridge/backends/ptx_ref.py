from __future__ import annotations

from ..backend import LoweredModule
from ..target import NvidiaTarget
from ..ir import PortableKernelIR
from .ptx_bridge import PtxBridge


class PtxRefBackend:
    name = "ptx_ref"
    artifact_extension = ".ptx"

    def __init__(self) -> None:
        self._bridge = PtxBridge()

    def supports(self, ir: PortableKernelIR, target: NvidiaTarget) -> bool:
        return self._bridge.supports(ir, target)

    def lower(self, ir: PortableKernelIR, target: NvidiaTarget) -> LoweredModule:
        return self._bridge.lower(ir, target, backend_name=self.name)
