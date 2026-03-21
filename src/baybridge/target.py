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
            "vendor": "amd",
        }


@dataclass(frozen=True)
class NvidiaTarget:
    sm: str = "sm_80"
    ptx_version: str = "8.0"
    warp_size: int = 32
    driver_version: str | None = None

    @property
    def arch(self) -> str:
        return self.sm

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "sm": self.sm,
            "ptx_version": self.ptx_version,
            "warp_size": self.warp_size,
            "driver_version": self.driver_version,
            "vendor": "nvidia",
        }
