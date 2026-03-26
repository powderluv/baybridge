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
    sm: str | int = "sm_80"
    ptx_version: str = "8.0"
    warp_size: int = 32
    driver_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sm", self._normalize_sm(self.sm))

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

    @staticmethod
    def _normalize_sm(value: str | int) -> str:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("NvidiaTarget(sm=...) expects a positive SM value")
            return f"sm_{value}"
        text = str(value).strip().lower()
        if not text:
            raise ValueError("NvidiaTarget(sm=...) expects a non-empty SM value")
        suffix = text[3:] if text.startswith("sm_") else text
        if not suffix or not all(ch.isalnum() for ch in suffix):
            raise ValueError("NvidiaTarget(sm=...) expects values like 80, '80', or 'sm_80'")
        return f"sm_{suffix}"
