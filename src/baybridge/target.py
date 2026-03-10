from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AMDTarget:
    arch: str = "gfx942"
    wave_size: int = 64
    rocm_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "wave_size": self.wave_size,
            "rocm_version": self.rocm_version,
        }
