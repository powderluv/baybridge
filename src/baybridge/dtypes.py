from __future__ import annotations


class ElementType(str):
    @property
    def width(self) -> int:
        table = {
            "i1": 1,
            "i8": 8,
            "i32": 32,
            "i64": 64,
            "f16": 16,
            "bf16": 16,
            "f32": 32,
            "index": 64,
        }
        try:
            return table[str(self)]
        except KeyError as exc:
            raise ValueError(f"baybridge element type '{self}' does not define a bit width") from exc


def element_type(dtype: str) -> ElementType:
    return ElementType(dtype)
