from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MFMADescriptor:
    tile: tuple[int, int, int]
    operand_dtype: str
    accumulator_dtype: str
    variant_name: str
    llvm_intrinsic: str
    wave_size: int = 64
    lane_shape: tuple[int, int] = (4, 16)

    def operand_shape(self, role: str) -> tuple[int, int]:
        m, n, k = self.tile
        table = {
            "a": (m, k),
            "b": (k, n),
            "acc": (m, n),
        }
        try:
            return table[role]
        except KeyError as exc:
            raise ValueError(f"unsupported MFMA fragment role '{role}'") from exc


MFMA_DESCRIPTORS: tuple[MFMADescriptor, ...] = (
    MFMADescriptor(
        tile=(16, 16, 16),
        operand_dtype="f16",
        accumulator_dtype="f32",
        variant_name="mfma_f32_16x16x16f16",
        llvm_intrinsic="llvm.amdgcn.mfma.f32.16x16x16f16",
    ),
    MFMADescriptor(
        tile=(16, 16, 16),
        operand_dtype="bf16",
        accumulator_dtype="f32",
        variant_name="mfma_f32_16x16x16bf16",
        llvm_intrinsic="llvm.amdgcn.mfma.f32.16x16x16bf16",
    ),
    MFMADescriptor(
        tile=(16, 16, 16),
        operand_dtype="f16",
        accumulator_dtype="f16",
        variant_name="mfma_f16_16x16x16f16",
        llvm_intrinsic="llvm.amdgcn.mfma.f16.16x16x16f16",
    ),
    MFMADescriptor(
        tile=(16, 16, 16),
        operand_dtype="bf16",
        accumulator_dtype="bf16",
        variant_name="mfma_bf16_16x16x16bf16",
        llvm_intrinsic="llvm.amdgcn.mfma.bf16.16x16x16bf16",
    ),
    MFMADescriptor(
        tile=(16, 16, 4),
        operand_dtype="f32",
        accumulator_dtype="f32",
        variant_name="mfma_f32_16x16x4f32",
        llvm_intrinsic="llvm.amdgcn.mfma.f32.16x16x4f32",
    ),
    MFMADescriptor(
        tile=(16, 16, 32),
        operand_dtype="i8",
        accumulator_dtype="i32",
        variant_name="mfma_i32_16x16x32i8",
        llvm_intrinsic="llvm.amdgcn.mfma.i32.16x16x32.i8",
    ),
)


def resolve_mfma_descriptor(
    tile: tuple[int, int, int],
    operand_dtype: str,
    accumulator_dtype: str,
    *,
    wave_size: int = 64,
) -> MFMADescriptor:
    for descriptor in MFMA_DESCRIPTORS:
        if (
            descriptor.tile == tile
            and descriptor.operand_dtype == operand_dtype
            and descriptor.accumulator_dtype == accumulator_dtype
            and descriptor.wave_size == wave_size
        ):
            return descriptor
    supported = ", ".join(
        f"{descriptor.variant_name}[tile={descriptor.tile}, operand_dtype={descriptor.operand_dtype}, accumulator_dtype={descriptor.accumulator_dtype}, wave_size={descriptor.wave_size}]"
        for descriptor in MFMA_DESCRIPTORS
    )
    raise ValueError(
        "unsupported MFMA descriptor for "
        f"tile={tile}, operand_dtype={operand_dtype}, accumulator_dtype={accumulator_dtype}, wave_size={wave_size}. "
        f"Supported descriptors: {supported}"
    )
