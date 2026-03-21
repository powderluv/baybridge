from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ir import PortableKernelIR
from .target import AMDTarget, NvidiaTarget


@dataclass(frozen=True)
class LoweredModule:
    backend_name: str
    entry_point: str
    dialect: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "entry_point": self.entry_point,
            "dialect": self.dialect,
            "text": self.text,
        }


class Backend(Protocol):
    name: str

    def lower(self, ir: PortableKernelIR, target: AMDTarget | NvidiaTarget) -> LoweredModule: ...
