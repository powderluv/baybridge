from __future__ import annotations

from ..backend import LoweredModule
from ..target import AMDTarget
from ..ir import PortableKernelIR
from .flydsl_bridge import FlyDslBridge


class FlyDslRefBackend:
    name = "flydsl_ref"
    artifact_extension = ".fly.mlir"

    def __init__(self) -> None:
        self._bridge = FlyDslBridge()

    def available(self, target: AMDTarget | None = None) -> bool:
        del target
        return self._bridge.reference_available()

    def supports(self, ir: PortableKernelIR, target: AMDTarget) -> bool:
        return self._bridge.supports(ir, target)

    def lower(self, ir: PortableKernelIR, target: AMDTarget) -> LoweredModule:
        return self._bridge.lower(ir, target, backend_name=self.name)
