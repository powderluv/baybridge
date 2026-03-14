from .diagnostics import UnsupportedOperationError
from .frontend import nvgpu as _nvgpu

cpasync = _nvgpu.cpasync
warp = _nvgpu.warp
warpgroup = _nvgpu.warpgroup
wgmma = _nvgpu.wgmma
tcgen05 = _nvgpu.tcgen05
mbarrier = _nvgpu.mbarrier
CopyUniversalOp = _nvgpu.CopyUniversalOp
MmaUniversalOp = _nvgpu.MmaUniversalOp
make_tiled_tma_atom_A = _nvgpu.make_tiled_tma_atom_A
make_tiled_tma_atom_B = _nvgpu.make_tiled_tma_atom_B

__all__ = [
    'CopyUniversalOp',
    'MmaUniversalOp',
    'cpasync',
    'mbarrier',
    'make_tiled_tma_atom_A',
    'make_tiled_tma_atom_B',
    'tcgen05',
    'warp',
    'warpgroup',
    'wgmma',
]


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    raise UnsupportedOperationError(
        f"baybridge.nvgpu.{name} is NVIDIA-specific and not implemented in the AMD port"
    )
