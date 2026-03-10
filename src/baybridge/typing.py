from __future__ import annotations

Boolean = bool
Int32 = int
Int = int


class Constexpr:
    def __class_getitem__(cls, item):
        return cls
