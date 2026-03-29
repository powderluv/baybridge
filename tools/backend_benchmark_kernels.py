from __future__ import annotations

import ctypes

import baybridge as bb
from baybridge.hip_runtime import pack_tensor_value
from baybridge.nvgpu import cpasync, tcgen05

try:
    import torch
except Exception:
    torch = None

try:
    from baybridge.cuda_driver import CudaDriver, CUdeviceptr
except Exception:
    CudaDriver = None
    CUdeviceptr = None

POINTWISE_N = 65536
ASTER_POINTWISE_N = 4096
FLYDSL_BLOCK = 256
FLYDSL_GRID = POINTWISE_N // FLYDSL_BLOCK
PTX_BLOCK = 256
PTX_GRID = POINTWISE_N // PTX_BLOCK
FLYDSL_MICRO_N = 4096
FLYDSL_MICRO_ROWS = 64
FLYDSL_MICRO_COLS = 64
FLYDSL_SHARED_N = 256
PTX_ROW_TILE_BLOCK = 32
PTX_ROW_TILE_GRID_X = FLYDSL_MICRO_COLS // PTX_ROW_TILE_BLOCK
PTX_ROW_TILE_GRID_Y = 4


class _CudaBenchmarkTensor:
    def __init__(self, driver, ptr: int, shape: tuple[int, ...], dtype: str, stride: tuple[int, ...]) -> None:
        self._driver = driver
        self._ptr = int(ptr)
        self.shape = shape
        self.dtype = dtype
        self._stride = stride
        self._freed = False

    def __dlpack__(self):
        return "cuda-capsule"

    def __dlpack_device__(self):
        return (2, 0)

    def data_ptr(self):
        return self._ptr

    def stride(self):
        return self._stride

    def _free(self) -> None:
        if self._freed:
            return
        self._freed = True
        try:
            self._driver.mem_free(CUdeviceptr(self._ptr))
        except Exception:
            pass

    def __del__(self) -> None:
        self._free()


@bb.kernel
def dense_copy_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel
def dense_add_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + other.load())


@bb.kernel
def dense_sub_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() - other.load())


@bb.kernel
def dense_mul_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() * other.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_add_2d_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_add_2d_f16_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_max_2d_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.maximum(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_add_2d_i32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_bitand_2d_i32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() & rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_copy_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_copy_2d_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_copy_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def broadcast_add_2d_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def broadcast_add_2d_i32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def broadcast_min_2d_i32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.minimum(lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_sqrt_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_rsqrt_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.rsqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_cos_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.cos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_log_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_round_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_trunc_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.trunc(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_erf_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_atan_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_asin_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_acos_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_abs_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_abs_2d_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_abs_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_bitnot_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_scalar_add_2d_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_scalar_add_2d_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_scalar_bitand_2d_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_scalar_bitor_2d_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() | alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_tensor_scalar_add_2d_f32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_tensor_scalar_add_2d_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_tensor_scalar_bitand_2d_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() & alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_tensor_scalar_bitor_2d_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def dense_select_2d_f32_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def broadcast_select_2d_i32_kernel(pred: bb.Tensor, lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_add_2d_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_add_2d_f16_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_add_2d_i32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_copy_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_copy_2d_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_copy_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_broadcast_add_2d_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_broadcast_add_2d_i32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_broadcast_bitor_2d_i32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() | rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_sqrt_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.sqrt(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_log2_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log2(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_round_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_floor_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.floor(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_exp_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_erf_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_atan_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_asin_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_acos_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_abs_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_abs_2d_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_abs_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_bitnot_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_scalar_add_2d_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_scalar_max_2d_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.maximum(src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_scalar_add_2d_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() + alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_scalar_bitand_2d_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() & alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_scalar_bitor_2d_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    dst.store(src.load() | alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_tensor_scalar_add_2d_f32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_tensor_scalar_add_2d_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_tensor_scalar_bitand_2d_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() & alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_tensor_scalar_bitor_2d_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() | alpha[0])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_dense_scalar_select_2d_f32_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), src.load(), alpha))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_tensor_factory_2d_f32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_tensor_factory_2d_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_copy_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_copy_2d_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_exp2_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.exp2(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_log10_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.log10(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_round_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.round(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_ceil_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.ceil(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_erf_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.erf(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_atan_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.atan(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_asin_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.asin(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_acos_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.acos(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_atan2_2d_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.math.atan2(lhs.load(), rhs.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_add_2d_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_add_2d_f16_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_tensor_scalar_min_2d_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.minimum(src.load(), alpha[0]))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_tensor_scalar_add_2d_f32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + alpha[0])


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_tensor_scalar_select_2d_i32_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    dst.store(bb.where(pred.load(), alpha[0], src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_bitnot_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(~src.load())


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_abs_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_abs_2d_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_dense_abs_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_copy_reduce_add_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(
    launch=bb.LaunchConfig(
        grid=(PTX_ROW_TILE_GRID_X, PTX_ROW_TILE_GRID_Y, 1),
        block=(PTX_ROW_TILE_BLOCK, 1, 1),
    )
)
def multiblock_tensor_factory_2d_f32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_copy_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_copy_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_copy_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(FLYDSL_GRID, 1, 1), block=(FLYDSL_BLOCK, 1, 1)))
def indexed_add_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_add_f16_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(FLYDSL_GRID, 1, 1), block=(FLYDSL_BLOCK, 1, 1)))
def indexed_sub_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] - other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(FLYDSL_GRID, 1, 1), block=(FLYDSL_BLOCK, 1, 1)))
def indexed_mul_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] * other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(FLYDSL_GRID, 1, 1), block=(FLYDSL_BLOCK, 1, 1)))
def indexed_div_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] / other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_add_i32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_bitand_i32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] & other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_bitor_i32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] | other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_bitxor_i32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] ^ other[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_bitnot_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = ~src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_scalar_broadcast_add_f32_kernel(src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_scalar_broadcast_add_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_scalar_broadcast_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] | alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_tensor_scalar_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] & alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_tensor_scalar_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] | alpha[0]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_select_scalar_f32_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.where(pred[idx], src[idx], alpha)


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_select_tensor_scalar_i32_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.where(pred[idx], alpha[0], src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_sqrt_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.sqrt(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_rsqrt_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.rsqrt(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_sin_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.sin(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_exp_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.exp(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_erf_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.erf(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_round_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.round(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_floor_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.floor(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_atan_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.atan(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_asin_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.asin(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_acos_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.acos(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_atan2_f32_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.math.atan2(lhs[idx], rhs[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_neg_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = -src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_neg_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = -src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_abs_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = abs(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_abs_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = abs(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_abs_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = abs(src[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_max_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.maximum(src[idx], other[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_min_i32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = bb.minimum(src[idx], other[idx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_add_f16_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + other[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_sqrt_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.sqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_rsqrt_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.rsqrt(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_exp2_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.exp2(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_log10_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.log10(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_erf_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.math.erf(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_round_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.round(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_ceil_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.ceil(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_atan_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.atan(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_asin_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.asin(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_acos_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.acos(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_neg_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = -src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_neg_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = -src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_neg_f16_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = -src[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_abs_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = abs(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_abs_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = abs(src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_bitand_i32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] & other[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_bitor_i32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] | other[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_scalar_broadcast_bitor_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] | alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_scalar_broadcast_bitand_i32_kernel(src: bb.Tensor, alpha: bb.Int32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] & alpha


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def direct_select_scalar_f32_kernel(pred: bb.Tensor, src: bb.Tensor, alpha: bb.Float32, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.where(pred[tidx], alpha, src[tidx])


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_add_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def copy_reduce_add_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def copy_reduce_max_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def copy_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_copy_reduce_add_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_copy_reduce_max_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_copy_reduce_xor_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.XOR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(PTX_GRID, 1, 1), block=(PTX_BLOCK, 1, 1)))
def indexed_copy_reduce_or_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(PTX_BLOCK, 1, 1)))
def parallel_reduce_add_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(PTX_BLOCK, 1, 1)))
def parallel_reduce_add_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst[0] = src.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def copy_reduce_add_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def copy_reduce_max_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def copy_reduce_or_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_copy_reduce_add_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_copy_reduce_max_2d_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.MAX, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_copy_reduce_or_2d_i32_kernel(src: bb.Tensor, dst: bb.Tensor):
    atom = bb.make_copy_atom(
        cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.OR, tcgen05.CtaGroup.ONE),
        src.element_type,
    )
    bb.copy(atom, src, dst)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_add_2d_bundle_f32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_add_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_mul_2d_bundle_f32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_max_2d_bundle_f32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_min_2d_bundle_f32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_mul_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_and_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_or_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_xor_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_max_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_min_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def tensor_factory_2d_f32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def tensor_factory_2d_i32_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_add_2d_bundle_f32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_add_2d_bundle_i32_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_scalar[0] = src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=0)
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_add_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_add_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_mul_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_max_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_min_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_mul_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_max_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_rows_min_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_add_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_add_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_mul_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_max_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_min_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_mul_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_and_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_or_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_xor_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_max_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_rows_min_2d_i32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_add_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_add_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_mul_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_max_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_min_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_mul_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_max_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def reduce_cols_min_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_add_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_add_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_mul_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_max_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_min_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_mul_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MUL, 1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_and_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.AND, -1, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_or_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.OR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_xor_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.XOR, 0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_max_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MAX, -99, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def parallel_reduce_cols_min_2d_i32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.MIN, 999, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(32, 1, 1)))
def multiblock_reduce_rows_add_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(32, 1, 1)))
def multiblock_reduce_cols_add_2d_f32_kernel(src: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(32, 1, 1)))
def multiblock_reduce_add_rowcol_2d_f32_kernel(src: bb.Tensor, dst_rows: bb.Tensor, dst_cols: bb.Tensor):
    src_vec = src.load()
    dst_rows.store(src_vec.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))
    dst_cols.store(src_vec.reduce(bb.ReductionOp.ADD, 1.0, reduction_profile=(1, None)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_unary_sin_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    values = src.load()
    dst.store(bb.math.sin(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_unary_rsqrt_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    values = src.load()
    dst.store(bb.math.rsqrt(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_broadcast_add_2d_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst: bb.Tensor):
    dst.store(lhs.load() + rhs.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_reduce_add_2d_kernel(src: bb.Tensor, dst_scalar: bb.Tensor, dst_rows: bb.Tensor):
    loaded = src.load()
    dst_scalar[0] = loaded.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0)
    dst_rows.store(loaded.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_unary_math_2d_kernel(
    src: bb.Tensor,
    dst_exp: bb.Tensor,
    dst_log: bb.Tensor,
    dst_cos: bb.Tensor,
    dst_erf: bb.Tensor,
):
    values = src.load()
    dst_exp.store(bb.math.exp(values))
    dst_log.store(bb.math.log(values))
    dst_cos.store(bb.math.cos(values))
    dst_erf.store(bb.math.erf(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(FLYDSL_SHARED_N, 1, 1)))
def flydsl_shared_stage_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor(
        "smem",
        shape=(FLYDSL_SHARED_N,),
        dtype="f32",
        address_space=bb.AddressSpace.SHARED,
    )
    smem[tidx] = src[tidx]
    bb.barrier()
    dst[tidx] = smem[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def flydsl_tensor_factory_2d_kernel(dst_zero: bb.Tensor, dst_one: bb.Tensor, dst_full: bb.Tensor):
    dst_zero.store(bb.zeros_like(dst_zero))
    dst_one.store(bb.ones_like(dst_one))
    dst_full.store(bb.full_like(dst_full, 7.0))


@bb.kernel
def hipkittens_bf16_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def hipkittens_f16_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def aster_mfma_f16_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def aster_mfma_bf16_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def aster_mfma_fp8_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def aster_mfma_bf8_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def aster_mfma_fp8_bf8_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def aster_mfma_bf8_fp8_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


def _vector_f32(n: int, *, scale: float = 1.0, offset: float = 0.0) -> list[float]:
    return [scale * float(index % 251) + offset for index in range(n)]


def _vector_i32(n: int, *, scale: int = 1, offset: int = 0) -> list[int]:
    return [scale * int(index % 251) + offset for index in range(n)]


def _matrix_f32(rows: int, cols: int, *, scale: float = 1.0, offset: float = 0.0) -> list[list[float]]:
    return [
        [scale * float((row * cols + col) % 251) + offset for col in range(cols)]
        for row in range(rows)
    ]


def _matrix_i32(rows: int, cols: int, *, scale: int = 1, offset: int = 0) -> list[list[int]]:
    return [
        [scale * int((row * cols + col) % 251) + offset for col in range(cols)]
        for row in range(rows)
    ]


def _maybe_torch_tensor(values, *, shape, dtype: str, backend_name: str):
    if backend_name != "flydsl_exec" or torch is None:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        return None
    torch_dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]
    flat = torch.tensor(values, dtype=torch_dtype, device="cuda")
    return flat.reshape(shape)


def _make_f32_vector_args(n: int, *, backend_name: str):
    src_values = _vector_f32(n, scale=0.5, offset=1.0)
    other_values = _vector_f32(n, scale=0.25, offset=2.0)
    src = _maybe_torch_tensor(src_values, shape=(n,), dtype="f32", backend_name=backend_name)
    other = _maybe_torch_tensor(other_values, shape=(n,), dtype="f32", backend_name=backend_name)
    dst = _maybe_torch_tensor([0.0 for _ in range(n)], shape=(n,), dtype="f32", backend_name=backend_name)
    if src is not None and other is not None and dst is not None:
        return src, other, dst
    return (
        bb.tensor(src_values, dtype="f32"),
        bb.tensor(other_values, dtype="f32"),
        bb.zeros((n,), dtype="f32"),
    )


def _make_f16_vector_args(n: int, *, backend_name: str):
    src_values = [0.5 * float(index) for index in range(n)]
    other_values = [0.25 * float(index + 1) for index in range(n)]
    src = _maybe_torch_tensor(src_values, shape=(n,), dtype="f16", backend_name=backend_name)
    other = _maybe_torch_tensor(other_values, shape=(n,), dtype="f16", backend_name=backend_name)
    dst = _maybe_torch_tensor([0.0 for _ in range(n)], shape=(n,), dtype="f16", backend_name=backend_name)
    if src is not None and other is not None and dst is not None:
        return src, other, dst
    return (
        bb.tensor(src_values, dtype="f16"),
        bb.tensor(other_values, dtype="f16"),
        bb.zeros((n,), dtype="f16"),
    )


def _maybe_cuda_handle_vector(
    values: list[float] | list[int],
    *,
    dtype: str,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
) -> object | None:
    if CudaDriver is None:
        return None
    try:
        driver = CudaDriver()
    except Exception:
        return None
    if driver.device_count() < 1:
        return None
    ctype = {
        "f32": ctypes.c_float,
        "f16": ctypes.c_uint16,
        "i1": ctypes.c_bool,
        "i32": ctypes.c_int32,
    }[dtype]
    wrapper_dtype = {
        "f32": "torch.float32",
        "f16": "torch.float16",
        "i1": "torch.bool",
        "i32": "torch.int32",
    }[dtype]
    host_values = [pack_tensor_value(value, dtype) for value in values] if dtype == "f16" else values
    host = (ctype * len(host_values))(*host_values)
    byte_size = ctypes.sizeof(host)
    ptr = driver.mem_alloc(byte_size)
    driver.memcpy_htod(ptr, ctypes.cast(host, ctypes.c_void_p), byte_size)
    if shape is None:
        shape = (len(values),)
    if stride is None:
        stride = (1,) if len(shape) == 1 else (shape[1], 1)
    wrapper = _CudaBenchmarkTensor(driver, int(ptr.value), shape, wrapper_dtype, stride)
    return bb.from_dlpack(wrapper)


def _maybe_cuda_dlpack_vector(
    values: list[float] | list[int],
    *,
    dtype: str,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
) -> object | None:
    if CudaDriver is None:
        return None
    try:
        driver = CudaDriver()
    except Exception:
        return None
    if driver.device_count() < 1:
        return None
    ctype = {
        "f32": ctypes.c_float,
        "f16": ctypes.c_uint16,
        "i1": ctypes.c_bool,
        "i32": ctypes.c_int32,
    }[dtype]
    wrapper_dtype = {
        "f32": "torch.float32",
        "f16": "torch.float16",
        "i1": "torch.bool",
        "i32": "torch.int32",
    }[dtype]
    host_values = [pack_tensor_value(value, dtype) for value in values] if dtype == "f16" else values
    host = (ctype * len(host_values))(*host_values)
    byte_size = ctypes.sizeof(host)
    ptr = driver.mem_alloc(byte_size)
    driver.memcpy_htod(ptr, ctypes.cast(host, ctypes.c_void_p), byte_size)
    if shape is None:
        shape = (len(values),)
    if stride is None:
        stride = (1,) if len(shape) == 1 else (shape[1], 1)
    return _CudaBenchmarkTensor(driver, int(ptr.value), shape, wrapper_dtype, stride)


def dense_copy_f32_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    src = _maybe_torch_tensor(src_values, shape=(POINTWISE_N,), dtype="f32", backend_name=backend_name or "")
    dst = _maybe_torch_tensor([0.0 for _ in range(POINTWISE_N)], shape=(POINTWISE_N,), dtype="f32", backend_name=backend_name or "")
    if src is None or dst is None:
        src = bb.tensor(src_values, dtype="f32")
        dst = bb.zeros((POINTWISE_N,), dtype="f32")
    return {"args": (src, dst), "result_indices": ()}


def indexed_copy_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((POINTWISE_N,), dtype="f32"),
        ),
        "result_indices": (),
    }


def indexed_copy_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((POINTWISE_N,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(POINTWISE_N)], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def indexed_copy_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_copy_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def indexed_copy_f16_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f16"),
            bb.zeros((POINTWISE_N,), dtype="f16"),
        ),
        "result_indices": (),
    }


def indexed_copy_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    compile_args = (
        bb.tensor(src_values, dtype="f16"),
        bb.zeros((POINTWISE_N,), dtype="f16"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f16")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(POINTWISE_N)], dtype="f16")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def indexed_copy_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    compile_args = (
        bb.tensor(src_values, dtype="f16"),
        bb.zeros((POINTWISE_N,), dtype="f16"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="f16")
    dst = _maybe_cuda_dlpack_vector([0.0 for _ in range(POINTWISE_N)], dtype="f16")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def dense_add_f32_args(*, backend_name=None, **_kwargs):
    src, other, dst = _make_f32_vector_args(POINTWISE_N, backend_name=backend_name or "")
    return {"args": (src, other, dst), "result_indices": ()}


def _tensor_copy_2d_payload(
    *,
    dtype: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if dtype == "f32":
        matrix_factory = _matrix_f32
        zero_fill = 0.0
        scale = 0.5
        offset = 1.0
    elif dtype == "i32":
        matrix_factory = _matrix_i32
        zero_fill = 0
        scale = 3
        offset = 7
    elif dtype == "f16":
        matrix_factory = _matrix_f32
        zero_fill = 0.0
        scale = 0.5
        offset = 1.0
    else:
        raise ValueError(f"unsupported tensor-copy 2D dtype: {dtype}")
    src_values = matrix_factory(rows, cols, scale=scale, offset=offset)
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.zeros((rows, cols), dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("tensor-copy 2D payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src_flat = [value for row in src_values for value in row]
    src = vector_factory(src_flat, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    dst = vector_factory(
        [zero_fill for _ in range(rows * cols)],
        dtype=dtype,
        shape=(rows, cols),
        stride=(cols, 1),
    )
    if src is None or dst is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def dense_copy_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name)


def dense_copy_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def dense_copy_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def dense_copy_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name)


def dense_copy_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def dense_copy_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def dense_copy_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="i32", backend_name=backend_name)


def dense_copy_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_copy_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name)


def parallel_dense_copy_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_copy_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_copy_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name)


def parallel_dense_copy_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_copy_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_copy_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="i32", backend_name=backend_name)


def parallel_dense_copy_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_copy_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name)


def multiblock_dense_copy_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_copy_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_copy_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name)


def multiblock_dense_copy_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_copy_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_copy_2d_payload(dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def _tensor_binary_2d_payload(
    *,
    mode: str,
    dtype: str,
    op: str = "add",
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("tensor-binary 2D payload cannot request both cuda_handle and cuda_dlpack")
    if dtype == "f32":
        if op not in {"add", "max", "atan2"}:
            raise ValueError(f"unsupported tensor-binary 2D op for f32: {op}")
        matrix_factory = _matrix_f32
        zero_fill = 0.0
        lhs_scale = 0.5
        lhs_offset = 1.0
        rhs_scale = 0.25
        rhs_offset = 2.0
    elif dtype == "f16":
        if op != "add":
            raise ValueError(f"unsupported tensor-binary 2D op for f16: {op}")
        matrix_factory = _matrix_f32
        zero_fill = 0.0
        lhs_scale = 0.5
        lhs_offset = 0.5
        rhs_scale = 0.25
        rhs_offset = 0.25
    elif dtype == "i32":
        if op not in {"add", "min", "bitand", "bitor"}:
            raise ValueError(f"unsupported tensor-binary 2D op for i32: {op}")
        matrix_factory = _matrix_i32
        zero_fill = 0
        lhs_scale = 3
        lhs_offset = 7
        rhs_scale = 5
        rhs_offset = 11
    else:
        raise ValueError(f"unsupported tensor-binary 2D dtype: {dtype}")
    if dtype == "f32" and op == "atan2":
        lhs_values = [
            [-2.0 + 0.25 * float((row * cols + col) % 16) for col in range(cols)]
            for row in range(rows)
        ]
        rhs_values = [
            [float(((row * cols + col) % 5) - 2) for col in range(cols)]
            for row in range(rows)
        ]
    if mode == "dense":
        if op != "atan2":
            lhs_values = matrix_factory(rows, cols, scale=lhs_scale, offset=lhs_offset)
            rhs_values = matrix_factory(rows, cols, scale=rhs_scale, offset=rhs_offset)
        lhs_shape = (rows, cols)
        rhs_shape = (rows, cols)
        lhs_stride = (cols, 1)
        rhs_stride = (cols, 1)
    elif mode == "broadcast":
        lhs_values = matrix_factory(rows, 1, scale=lhs_scale, offset=lhs_offset)
        rhs_values = matrix_factory(1, cols, scale=rhs_scale, offset=rhs_offset)
        lhs_shape = (rows, 1)
        rhs_shape = (1, cols)
        lhs_stride = (1, 1)
        rhs_stride = (cols, 1)
    else:
        raise ValueError(f"unsupported tensor-binary 2D mode: {mode}")
    compile_args = (
        bb.tensor(lhs_values, dtype=dtype),
        bb.tensor(rhs_values, dtype=dtype),
        bb.zeros((rows, cols), dtype=dtype),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    lhs_flat = [value for row in lhs_values for value in row]
    rhs_flat = [value for row in rhs_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    lhs = vector_factory(lhs_flat, dtype=dtype, shape=lhs_shape, stride=lhs_stride)
    rhs = vector_factory(rhs_flat, dtype=dtype, shape=rhs_shape, stride=rhs_stride)
    dst = vector_factory(
        [zero_fill for _ in range(rows * cols)],
        dtype=dtype,
        shape=(rows, cols),
        stride=(cols, 1),
    )
    if lhs is None or rhs is None or dst is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (lhs, rhs, dst),
        "result_indices": (),
    }


def dense_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", backend_name=backend_name)


def dense_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def dense_add_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name)


def dense_add_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def dense_add_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def dense_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", op="max", backend_name=backend_name)


def dense_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def dense_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def dense_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="i32", backend_name=backend_name)


def dense_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def dense_bitand_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="i32", op="bitand", backend_name=backend_name)


def dense_bitand_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="i32", op="bitand", backend_name=backend_name, use_cuda_handle=True)


def dense_bitand_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="i32", op="bitand", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", backend_name=backend_name)


def parallel_dense_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_add_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name)


def parallel_dense_add_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_add_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    lhs_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
    rhs_values = _matrix_f32(rows, cols, scale=0.25, offset=2.0)
    compile_args = (
        bb.tensor(lhs_values, dtype="f32"),
        bb.tensor(rhs_values, dtype="f32"),
        bb.zeros((rows, cols), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    lhs = _maybe_cuda_dlpack_vector(
        [value for row in lhs_values for value in row],
        dtype="f32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    rhs = _maybe_cuda_dlpack_vector(
        [value for row in rhs_values for value in row],
        dtype="f32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    dst = _maybe_cuda_dlpack_vector(
        [0.0 for _ in range(rows * cols)],
        dtype="f32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    if lhs is None or rhs is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (lhs, rhs, dst),
        "result_indices": (),
    }


def parallel_dense_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="i32", backend_name=backend_name)


def parallel_dense_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_add_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    lhs_values = _matrix_i32(rows, cols, scale=3, offset=7)
    rhs_values = _matrix_i32(rows, cols, scale=5, offset=11)
    compile_args = (
        bb.tensor(lhs_values, dtype="i32"),
        bb.tensor(rhs_values, dtype="i32"),
        bb.zeros((rows, cols), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    lhs = _maybe_cuda_dlpack_vector(
        [value for row in lhs_values for value in row],
        dtype="i32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    rhs = _maybe_cuda_dlpack_vector(
        [value for row in rhs_values for value in row],
        dtype="i32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    dst = _maybe_cuda_dlpack_vector(
        [0 for _ in range(rows * cols)],
        dtype="i32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    if lhs is None or rhs is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (lhs, rhs, dst),
        "result_indices": (),
    }


def multiblock_dense_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", backend_name=backend_name)


def multiblock_dense_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_add_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name)


def multiblock_dense_add_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_add_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    lhs_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
    rhs_values = _matrix_f32(rows, cols, scale=0.25, offset=2.0)
    compile_args = (
        bb.tensor(lhs_values, dtype="f32"),
        bb.tensor(rhs_values, dtype="f32"),
        bb.zeros((rows, cols), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    lhs = _maybe_cuda_dlpack_vector(
        [value for row in lhs_values for value in row],
        dtype="f32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    rhs = _maybe_cuda_dlpack_vector(
        [value for row in rhs_values for value in row],
        dtype="f32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    dst = _maybe_cuda_dlpack_vector(
        [0.0 for _ in range(rows * cols)],
        dtype="f32",
        shape=(rows, cols),
        stride=(cols, 1),
    )
    if lhs is None or rhs is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (lhs, rhs, dst),
        "result_indices": (),
    }


def broadcast_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="f32", backend_name=backend_name)


def broadcast_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def broadcast_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", backend_name=backend_name)


def broadcast_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def broadcast_min_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", op="min", backend_name=backend_name)


def broadcast_min_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def broadcast_min_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_broadcast_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="f32", backend_name=backend_name)


def parallel_broadcast_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_broadcast_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", backend_name=backend_name)


def parallel_broadcast_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def parallel_broadcast_bitor_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", op="bitor", backend_name=backend_name)


def parallel_broadcast_bitor_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", op="bitor", backend_name=backend_name, use_cuda_handle=True)


def parallel_broadcast_bitor_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="broadcast", dtype="i32", op="bitor", backend_name=backend_name, use_cuda_dlpack=True)


def _tensor_unary_2d_payload(
    *,
    dtype: str,
    op: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("use_cuda_handle and use_cuda_dlpack are mutually exclusive")
    if dtype == "f32":
        if op not in {"abs", "round", "floor", "ceil", "trunc", "sqrt", "rsqrt", "sin", "cos", "acos", "asin", "atan", "exp", "exp2", "log", "log2", "log10", "erf"}:
            raise ValueError(f"unsupported tensor-unary 2D op for f32: {op}")
        if op == "abs":
            src_values = _matrix_f32(rows, cols, scale=1.0, offset=-32.0)
        elif op in {"round", "floor", "ceil", "trunc"}:
            src_values = _matrix_f32(rows, cols, scale=0.5, offset=-32.25)
        elif op in {"sin", "cos"}:
            src_values = [
                [0.125 * float((row * cols + col) % 16) for col in range(cols)]
                for row in range(rows)
            ]
        elif op in {"acos", "asin"}:
            src_values = [
                [-0.875 + 0.125 * float((row * cols + col) % 15) for col in range(cols)]
                for row in range(rows)
            ]
        elif op == "atan":
            src_values = [
                [-2.0 + 0.25 * float((row * cols + col) % 16) for col in range(cols)]
                for row in range(rows)
            ]
        elif op == "erf":
            src_values = [
                [-2.0 + 0.25 * float((row * cols + col) % 16) for col in range(cols)]
                for row in range(rows)
            ]
        elif op == "exp":
            src_values = [
                [0.125 * float((row * cols + col) % 16) for col in range(cols)]
                for row in range(rows)
            ]
        elif op == "exp2":
            src_values = [
                [0.25 * float((row * cols + col) % 8) for col in range(cols)]
                for row in range(rows)
            ]
        elif op == "log":
            src_values = [
                [float(1 << ((row * cols + col) % 8)) for col in range(cols)]
                for row in range(rows)
            ]
        elif op == "log2":
            src_values = [
                [float(1 << ((row * cols + col) % 8)) for col in range(cols)]
                for row in range(rows)
            ]
        elif op == "log10":
            src_values = [
                [float(10 ** ((row * cols + col) % 4)) for col in range(cols)]
                for row in range(rows)
            ]
        else:
            src_values = _matrix_f32(rows, cols, scale=1.0, offset=4.0)
        zero_fill = 0.0
    elif dtype == "f16":
        if op != "abs":
            raise ValueError(f"unsupported tensor-unary 2D op for f16: {op}")
        src_values = _matrix_f32(rows, cols, scale=0.5, offset=-16.0)
        zero_fill = 0.0
    elif dtype == "i32":
        if op not in {"abs", "bitnot"}:
            raise ValueError(f"unsupported tensor-unary 2D op for i32: {op}")
        if op == "abs":
            src_values = _matrix_i32(rows, cols, scale=3, offset=-97)
        else:
            src_values = _matrix_i32(rows, cols, scale=3, offset=1)
        zero_fill = 0
    else:
        raise ValueError(f"unsupported tensor-unary 2D dtype: {dtype}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.zeros((rows, cols), dtype=dtype),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    flat_values = [value for row in src_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(flat_values, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    dst = vector_factory([zero_fill for _ in range(rows * cols)], dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    if src is None or dst is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def dense_sqrt_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="sqrt", backend_name=backend_name)


def dense_cos_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="cos", backend_name=backend_name)


def dense_cos_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="cos", backend_name=backend_name, use_cuda_handle=True)


def dense_cos_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="cos", backend_name=backend_name, use_cuda_dlpack=True)


def dense_log_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log", backend_name=backend_name)


def dense_log_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log", backend_name=backend_name, use_cuda_handle=True)


def dense_log_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log", backend_name=backend_name, use_cuda_dlpack=True)


def dense_erf_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name)


def dense_erf_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name, use_cuda_handle=True)


def dense_erf_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name, use_cuda_dlpack=True)


def dense_atan_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name)


def dense_round_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name)


def dense_round_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name, use_cuda_handle=True)


def dense_round_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name, use_cuda_dlpack=True)


def dense_trunc_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="trunc", backend_name=backend_name)


def dense_trunc_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="trunc", backend_name=backend_name, use_cuda_handle=True)


def dense_trunc_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="trunc", backend_name=backend_name, use_cuda_dlpack=True)


def dense_asin_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name)


def dense_acos_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name)


def dense_asin_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name, use_cuda_handle=True)


def dense_acos_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name, use_cuda_handle=True)


def dense_asin_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name, use_cuda_dlpack=True)


def dense_acos_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name, use_cuda_dlpack=True)


def dense_atan_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name, use_cuda_handle=True)


def dense_atan_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name, use_cuda_dlpack=True)


def dense_abs_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name)


def dense_abs_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name, use_cuda_handle=True)


def dense_abs_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def dense_abs_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name)


def dense_abs_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name, use_cuda_handle=True)


def dense_abs_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def dense_sqrt_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="sqrt", backend_name=backend_name, use_cuda_handle=True)


def dense_rsqrt_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="rsqrt", backend_name=backend_name)


def dense_rsqrt_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="rsqrt", backend_name=backend_name, use_cuda_handle=True)


def dense_abs_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name)


def dense_abs_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name, use_cuda_handle=True)


def dense_abs_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def dense_bitnot_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name)


def dense_bitnot_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name, use_cuda_handle=True)


def dense_bitnot_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_sqrt_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="sqrt", backend_name=backend_name)


def parallel_dense_log2_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log2", backend_name=backend_name)


def parallel_dense_log2_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log2", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_log2_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log2", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_exp_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="exp", backend_name=backend_name)


def parallel_dense_exp_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="exp", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_exp_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="exp", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_erf_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name)


def parallel_dense_erf_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_erf_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_atan_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name)


def parallel_dense_round_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name)


def parallel_dense_round_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_round_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_floor_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="floor", backend_name=backend_name)


def parallel_dense_floor_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="floor", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_floor_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="floor", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_asin_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name)


def parallel_dense_acos_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name)


def parallel_dense_asin_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_acos_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_asin_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_acos_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_atan_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_atan_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_abs_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name)


def parallel_dense_abs_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_abs_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_abs_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name)


def parallel_dense_abs_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_abs_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_sqrt_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="sqrt", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_abs_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name)


def parallel_dense_abs_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_abs_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_bitnot_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name)


def parallel_dense_bitnot_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_bitnot_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_bitnot_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name)


def multiblock_dense_exp2_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="exp2", backend_name=backend_name)


def multiblock_dense_exp2_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="exp2", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_exp2_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="exp2", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_log10_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log10", backend_name=backend_name)


def multiblock_dense_log10_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log10", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_log10_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="log10", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_erf_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name)


def multiblock_dense_erf_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_erf_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="erf", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_atan_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name)


def multiblock_dense_round_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name)


def multiblock_dense_round_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_round_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="round", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_ceil_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="ceil", backend_name=backend_name)


def multiblock_dense_ceil_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="ceil", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_ceil_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="ceil", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_asin_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name)


def multiblock_dense_acos_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name)


def multiblock_dense_asin_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_acos_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_asin_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="asin", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_acos_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="acos", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_atan_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_atan_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="atan", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_atan2_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", op="atan2", backend_name=backend_name)


def multiblock_dense_atan2_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", op="atan2", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_atan2_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_binary_2d_payload(mode="dense", dtype="f32", op="atan2", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_abs_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name)


def multiblock_dense_abs_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_abs_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f32", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_abs_2d_f16_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name)


def multiblock_dense_abs_2d_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_abs_2d_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="f16", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_abs_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name)


def multiblock_dense_abs_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_abs_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="abs", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_bitnot_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_bitnot_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_unary_2d_payload(dtype="i32", op="bitnot", backend_name=backend_name, use_cuda_dlpack=True)


def _tensor_scalar_broadcast_2d_payload(
    *,
    dtype: str,
    op: str = "add",
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if dtype == "f32":
        matrix_factory = _matrix_f32
        alpha_compile = bb.Float32(1.5)
        alpha_run = 1.5
        zero_fill = 0.0
        scale = 0.5
        offset = 1.0
        if op not in {"add", "max"}:
            raise ValueError(f"unsupported tensor-scalar-broadcast 2D op for f32: {op}")
    elif dtype == "i32":
        matrix_factory = _matrix_i32
        zero_fill = 0
        scale = 3
        offset = 7
        if op == "add":
            alpha_compile = bb.Int32(7)
            alpha_run = 7
        elif op == "min":
            alpha_compile = bb.Int32(17)
            alpha_run = 17
        elif op == "bitand":
            alpha_compile = bb.Int32(5)
            alpha_run = 5
        elif op == "bitor":
            alpha_compile = bb.Int32(24)
            alpha_run = 24
        else:
            raise ValueError(f"unsupported tensor-scalar-broadcast 2D op for i32: {op}")
    else:
        raise ValueError(f"unsupported tensor-scalar-broadcast 2D dtype: {dtype}")
    src_values = matrix_factory(rows, cols, scale=scale, offset=offset)
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        alpha_compile,
        bb.zeros((rows, cols), dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("tensor-scalar-broadcast payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    flat_values = [value for row in src_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(flat_values, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    dst = vector_factory(
        [zero_fill for _ in range(rows * cols)],
        dtype=dtype,
        shape=(rows, cols),
        stride=(cols, 1),
    )
    if src is None or dst is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, alpha_run, dst),
        "result_indices": (),
    }


def _tensor_source_scalar_broadcast_2d_payload(
    *,
    dtype: str,
    op: str = "add",
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if dtype == "f32":
        matrix_factory = _matrix_f32
        alpha_values = [1.5]
        zero_fill = 0.0
        scale = 0.5
        offset = 1.0
        if op not in {"add", "max"}:
            raise ValueError(f"unsupported tensor-source scalar-broadcast 2D op for f32: {op}")
    elif dtype == "i32":
        matrix_factory = _matrix_i32
        zero_fill = 0
        scale = 3
        offset = 7
        if op == "add":
            alpha_values = [7]
        elif op == "min":
            alpha_values = [17]
        elif op == "bitand":
            alpha_values = [5]
        elif op == "bitor":
            alpha_values = [24]
        else:
            raise ValueError(f"unsupported tensor-source scalar-broadcast 2D op for i32: {op}")
    else:
        raise ValueError(f"unsupported tensor-source scalar-broadcast 2D dtype: {dtype}")
    src_values = matrix_factory(rows, cols, scale=scale, offset=offset)
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.tensor(alpha_values, dtype=dtype),
        bb.zeros((rows, cols), dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("tensor-source scalar-broadcast payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    flat_values = [value for row in src_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(flat_values, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    alpha = vector_factory(alpha_values, dtype=dtype, shape=(1,), stride=(1,))
    dst = vector_factory([zero_fill for _ in range(rows * cols)], dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    if src is None or alpha is None or dst is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, alpha, dst),
        "result_indices": (),
    }


def _select_2d_payload(
    *,
    mode: str,
    dtype: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("select 2D payload cannot request both cuda_handle and cuda_dlpack")
    pred_values = [[bool((row + col) % 2) for col in range(cols)] for row in range(rows)]
    pred_flat = [value for row in pred_values for value in row]
    if mode == "dense":
        if dtype != "f32":
            raise ValueError(f"unsupported dense select 2D dtype: {dtype}")
        lhs_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
        rhs_values = _matrix_f32(rows, cols, scale=0.25, offset=-2.0)
        zero_fill = 0.0
        compile_args = (
            bb.tensor(pred_values, dtype="i1"),
            bb.tensor(lhs_values, dtype="f32"),
            bb.tensor(rhs_values, dtype="f32"),
            bb.zeros((rows, cols), dtype="f32"),
        )
        run_shape_specs = (
            (lhs_values, "f32", (rows, cols), (cols, 1)),
            (rhs_values, "f32", (rows, cols), (cols, 1)),
        )
        scalar_run = None
    elif mode == "broadcast":
        if dtype != "i32":
            raise ValueError(f"unsupported broadcast select 2D dtype: {dtype}")
        lhs_values = _matrix_i32(rows, 1, scale=3, offset=7)
        rhs_values = _matrix_i32(1, cols, scale=5, offset=11)
        zero_fill = 0
        compile_args = (
            bb.tensor(pred_values, dtype="i1"),
            bb.tensor(lhs_values, dtype="i32"),
            bb.tensor(rhs_values, dtype="i32"),
            bb.zeros((rows, cols), dtype="i32"),
        )
        run_shape_specs = (
            (lhs_values, "i32", (rows, 1), (1, 1)),
            (rhs_values, "i32", (1, cols), (cols, 1)),
        )
        scalar_run = None
    elif mode == "scalar_param":
        if dtype != "f32":
            raise ValueError(f"unsupported scalar-param select 2D dtype: {dtype}")
        src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
        zero_fill = 0.0
        compile_args = (
            bb.tensor(pred_values, dtype="i1"),
            bb.tensor(src_values, dtype="f32"),
            bb.Float32(1.5),
            bb.zeros((rows, cols), dtype="f32"),
        )
        run_shape_specs = (
            (src_values, "f32", (rows, cols), (cols, 1)),
        )
        scalar_run = 1.5
    elif mode == "tensor_scalar":
        if dtype != "i32":
            raise ValueError(f"unsupported tensor-source scalar select 2D dtype: {dtype}")
        src_values = _matrix_i32(rows, cols, scale=3, offset=7)
        zero_fill = 0
        compile_args = (
            bb.tensor(pred_values, dtype="i1"),
            bb.tensor(src_values, dtype="i32"),
            bb.tensor([17], dtype="i32"),
            bb.zeros((rows, cols), dtype="i32"),
        )
        run_shape_specs = (
            (src_values, "i32", (rows, cols), (cols, 1)),
            ([17], "i32", (1,), (1,)),
        )
        scalar_run = None
    else:
        raise ValueError(f"unsupported select 2D mode: {mode}")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    pred = vector_factory(pred_flat, dtype="i1", shape=(rows, cols), stride=(cols, 1))
    if pred is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    run_args: list[object] = [pred]
    for values, value_dtype, shape, stride in run_shape_specs:
        flat_values = [value for row in values for value in row] if shape != (1,) else values
        arg = vector_factory(flat_values, dtype=value_dtype, shape=shape, stride=stride)
        if arg is None:
            return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
        run_args.append(arg)
    if scalar_run is not None:
        run_args.append(scalar_run)
    dst = vector_factory(
        [zero_fill for _ in range(rows * cols)],
        dtype=dtype,
        shape=(rows, cols),
        stride=(cols, 1),
    )
    if dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    run_args.append(dst)
    return {
        "compile_args": compile_args,
        "run_args": tuple(run_args),
        "result_indices": (),
    }


def dense_scalar_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name)


def dense_scalar_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_scalar_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="f32", op="max", backend_name=backend_name)


def parallel_dense_scalar_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_scalar_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def dense_scalar_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name)


def dense_scalar_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def dense_scalar_bitand_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="i32", op="bitand", backend_name=backend_name)


def dense_scalar_bitand_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(
        dtype="i32", op="bitand", backend_name=backend_name, use_cuda_handle=True
    )


def dense_scalar_bitand_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(
        dtype="i32", op="bitand", backend_name=backend_name, use_cuda_dlpack=True
    )


def parallel_dense_scalar_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name)


def parallel_dense_scalar_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_scalar_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name)


def parallel_dense_scalar_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_scalar_bitand_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(dtype="i32", op="bitand", backend_name=backend_name)


def parallel_dense_scalar_bitand_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(
        dtype="i32", op="bitand", backend_name=backend_name, use_cuda_handle=True
    )


def parallel_dense_scalar_bitand_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_scalar_broadcast_2d_payload(
        dtype="i32", op="bitand", backend_name=backend_name, use_cuda_dlpack=True
    )


def dense_tensor_scalar_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name)


def dense_tensor_scalar_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def dense_tensor_scalar_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def dense_tensor_scalar_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name)


def dense_tensor_scalar_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def dense_tensor_scalar_add_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def dense_tensor_scalar_bitor_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", op="bitor", backend_name=backend_name)


def dense_tensor_scalar_bitor_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(
        dtype="i32", op="bitor", backend_name=backend_name, use_cuda_handle=True
    )


def dense_tensor_scalar_bitor_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(
        dtype="i32", op="bitor", backend_name=backend_name, use_cuda_dlpack=True
    )


def parallel_dense_tensor_scalar_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name)


def parallel_dense_tensor_scalar_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_tensor_scalar_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_tensor_scalar_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name)


def parallel_dense_tensor_scalar_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_tensor_scalar_add_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_tensor_scalar_bitor_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", op="bitor", backend_name=backend_name)


def parallel_dense_tensor_scalar_bitor_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(
        dtype="i32", op="bitor", backend_name=backend_name, use_cuda_handle=True
    )


def parallel_dense_tensor_scalar_bitor_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(
        dtype="i32", op="bitor", backend_name=backend_name, use_cuda_dlpack=True
    )


def multiblock_dense_tensor_scalar_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name)


def multiblock_dense_tensor_scalar_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_tensor_scalar_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_tensor_scalar_min_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", op="min", backend_name=backend_name)


def multiblock_dense_tensor_scalar_min_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_tensor_scalar_min_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_source_scalar_broadcast_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def dense_select_2d_f32_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="dense", dtype="f32", backend_name=backend_name)


def dense_select_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="dense", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def dense_select_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="dense", dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def broadcast_select_2d_i32_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="broadcast", dtype="i32", backend_name=backend_name)


def broadcast_select_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="broadcast", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def broadcast_select_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="broadcast", dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_dense_scalar_select_2d_f32_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="scalar_param", dtype="f32", backend_name=backend_name)


def parallel_dense_scalar_select_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="scalar_param", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_dense_scalar_select_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="scalar_param", dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_dense_tensor_scalar_select_2d_i32_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="tensor_scalar", dtype="i32", backend_name=backend_name)


def multiblock_dense_tensor_scalar_select_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="tensor_scalar", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def multiblock_dense_tensor_scalar_select_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _select_2d_payload(mode="tensor_scalar", dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_scalar_broadcast_add_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.Float32(1.5),
            bb.zeros((POINTWISE_N,), dtype="f32"),
        ),
        "result_indices": (),
    }


def indexed_scalar_broadcast_add_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.Float32(1.5),
        bb.zeros((POINTWISE_N,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(POINTWISE_N)], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 1.5, dst),
        "result_indices": (),
    }


def indexed_add_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor(other_values, dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_add_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    other = _maybe_cuda_handle_vector(other_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_add_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    other = _maybe_cuda_dlpack_vector(other_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_bitand_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor(other_values, dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_bitand_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    other = _maybe_cuda_handle_vector(other_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_bitand_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    other = _maybe_cuda_dlpack_vector(other_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_bitor_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor(other_values, dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_bitor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    other = _maybe_cuda_handle_vector(other_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_bitor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    other = _maybe_cuda_dlpack_vector(other_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_bitxor_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor(other_values, dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_bitxor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    other = _maybe_cuda_handle_vector(other_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_bitxor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(POINTWISE_N, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    other = _maybe_cuda_dlpack_vector(other_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_bitnot_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_bitnot_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def indexed_bitnot_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def indexed_scalar_broadcast_add_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.Int32(7),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_scalar_broadcast_add_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.Int32(7),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 7, dst),
        "result_indices": (),
    }


def indexed_scalar_broadcast_bitor_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.Int32(24),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_scalar_broadcast_bitor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.Int32(24),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 24, dst),
        "result_indices": (),
    }


def indexed_scalar_broadcast_bitor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.Int32(24),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 24, dst),
        "result_indices": (),
    }


def indexed_tensor_scalar_bitand_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor([11], dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_tensor_scalar_bitand_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor([11], dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    alpha = _maybe_cuda_handle_vector([11], dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or alpha is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, alpha, dst),
        "result_indices": (),
    }


def indexed_tensor_scalar_bitand_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor([11], dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    alpha = _maybe_cuda_dlpack_vector([11], dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or alpha is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, alpha, dst),
        "result_indices": (),
    }


def indexed_tensor_scalar_bitor_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor([11], dtype="i32"),
            bb.zeros((POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_tensor_scalar_bitor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor([11], dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    alpha = _maybe_cuda_handle_vector([11], dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or alpha is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, alpha, dst),
        "result_indices": (),
    }


def indexed_tensor_scalar_bitor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor([11], dtype="i32"),
        bb.zeros((POINTWISE_N,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    alpha = _maybe_cuda_dlpack_vector([11], dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(POINTWISE_N)], dtype="i32")
    if src is None or alpha is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, alpha, dst),
        "result_indices": (),
    }


def _select_1d_payload(
    *,
    mode: str,
    scalar_mode: str,
    dtype: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("select 1D payload cannot request both cuda_handle and cuda_dlpack")
    if mode == "indexed":
        n = POINTWISE_N
        scalar_value = 3.5 if dtype == "f32" else 17
    elif mode == "direct":
        n = 128
        scalar_value = 5.5 if dtype == "f32" else 23
    else:
        raise ValueError(f"unsupported select 1D mode: {mode}")
    pred_values = [bool(index % 2) for index in range(n)]
    if dtype == "f32":
        src_values = _vector_f32(n, scale=0.5, offset=1.0)
        zero_values = [0.0 for _ in range(n)]
        scalar_arg = bb.Float32(float(scalar_value))
    elif dtype == "i32":
        src_values = _vector_i32(n, scale=3, offset=7)
        zero_values = [0 for _ in range(n)]
        scalar_arg = bb.tensor([int(scalar_value)], dtype="i32") if scalar_mode == "tensor" else bb.Int32(int(scalar_value))
    else:
        raise ValueError(f"unsupported select 1D dtype: {dtype}")
    compile_args = (
        bb.tensor(pred_values, dtype="i1"),
        bb.tensor(src_values, dtype=dtype),
        scalar_arg,
        bb.zeros((n,), dtype=dtype),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    pred = vector_factory(pred_values, dtype="i1")
    src = vector_factory(src_values, dtype=dtype)
    if scalar_mode == "tensor":
        scalar = vector_factory([scalar_value], dtype=dtype)
    else:
        scalar = scalar_value
    dst = vector_factory(zero_values, dtype=dtype)
    if pred is None or src is None or scalar is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (pred, src, scalar, dst),
        "result_indices": (),
    }


def indexed_select_scalar_f32_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(mode="indexed", scalar_mode="param", dtype="f32", backend_name=backend_name)


def indexed_select_scalar_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(
        mode="indexed",
        scalar_mode="param",
        dtype="f32",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def indexed_select_scalar_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(
        mode="indexed",
        scalar_mode="param",
        dtype="f32",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def indexed_select_tensor_scalar_i32_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(mode="indexed", scalar_mode="tensor", dtype="i32", backend_name=backend_name)


def indexed_select_tensor_scalar_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(
        mode="indexed",
        scalar_mode="tensor",
        dtype="i32",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def indexed_select_tensor_scalar_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(
        mode="indexed",
        scalar_mode="tensor",
        dtype="i32",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def _add_1d_payload(
    *,
    mode: str,
    dtype: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("add 1D payload cannot request both cuda_handle and cuda_dlpack")
    if mode == "indexed":
        n = POINTWISE_N
    elif mode == "direct":
        n = 128
    else:
        raise ValueError(f"unsupported add 1D mode: {mode}")
    if dtype != "f16":
        raise ValueError(f"unsupported add 1D dtype: {dtype}")
    src_values = [0.5 * float(index) for index in range(n)]
    other_values = [0.25 * float(index + 1) for index in range(n)]
    zero_values = [0.0 for _ in range(n)]
    compile_args = (
        bb.tensor(src_values, dtype="f16"),
        bb.tensor(other_values, dtype="f16"),
        bb.zeros((n,), dtype="f16"),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(src_values, dtype="f16")
    other = vector_factory(other_values, dtype="f16")
    dst = vector_factory(zero_values, dtype="f16")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def _neg_1d_payload(
    *,
    mode: str,
    dtype: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("neg 1D payload cannot request both cuda_handle and cuda_dlpack")
    if mode == "indexed":
        n = POINTWISE_N
    elif mode == "direct":
        n = 128
    else:
        raise ValueError(f"unsupported neg 1D mode: {mode}")
    if dtype == "f32":
        src_values = [float(index) - 64.0 for index in range(n)]
        zero_values = [0.0 for _ in range(n)]
    elif dtype == "f16":
        src_values = [0.5 * float(index - 64) for index in range(n)]
        zero_values = [0.0 for _ in range(n)]
    elif dtype == "i32":
        src_values = [index - 64 for index in range(n)]
        zero_values = [0 for _ in range(n)]
    else:
        raise ValueError(f"unsupported neg 1D dtype: {dtype}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.zeros((n,), dtype=dtype),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(src_values, dtype=dtype)
    dst = vector_factory(zero_values, dtype=dtype)
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def _abs_1d_payload(
    *,
    mode: str,
    dtype: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("abs 1D payload cannot request both cuda_handle and cuda_dlpack")
    if mode == "indexed":
        n = POINTWISE_N
    elif mode == "direct":
        n = 128
    else:
        raise ValueError(f"unsupported abs 1D mode: {mode}")
    if dtype == "f32":
        src_values = [float(index) - 64.0 for index in range(n)]
        zero_values = [0.0 for _ in range(n)]
    elif dtype == "f16":
        src_values = [0.5 * float(index - 64) for index in range(n)]
        zero_values = [0.0 for _ in range(n)]
    elif dtype == "i32":
        src_values = [index - 64 for index in range(n)]
        zero_values = [0 for _ in range(n)]
    else:
        raise ValueError(f"unsupported abs 1D dtype: {dtype}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.zeros((n,), dtype=dtype),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(src_values, dtype=dtype)
    dst = vector_factory(zero_values, dtype=dtype)
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def _extrema_1d_payload(
    *,
    mode: str,
    dtype: str,
    op: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("extrema 1D payload cannot request both cuda_handle and cuda_dlpack")
    if mode != "indexed":
        raise ValueError(f"unsupported extrema 1D mode: {mode}")
    n = POINTWISE_N
    if dtype == "f32" and op == "max":
        lhs_values = _vector_f32(n, scale=0.5, offset=1.0)
        rhs_values = _vector_f32(n, scale=0.25, offset=-2.0)
        zero_values = [0.0 for _ in range(n)]
    elif dtype == "i32" and op == "min":
        lhs_values = _vector_i32(n, scale=3, offset=7)
        rhs_values = _vector_i32(n, scale=5, offset=-11)
        zero_values = [0 for _ in range(n)]
    else:
        raise ValueError(f"unsupported extrema 1D payload combination: dtype={dtype}, op={op}")
    compile_args = (
        bb.tensor(lhs_values, dtype=dtype),
        bb.tensor(rhs_values, dtype=dtype),
        bb.zeros((n,), dtype=dtype),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    lhs = vector_factory(lhs_values, dtype=dtype)
    rhs = vector_factory(rhs_values, dtype=dtype)
    dst = vector_factory(zero_values, dtype=dtype)
    if lhs is None or rhs is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (lhs, rhs, dst),
        "result_indices": (),
    }


def _atan2_1d_payload(
    *,
    mode: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("atan2 1D payload cannot request both cuda_handle and cuda_dlpack")
    if mode != "indexed":
        raise ValueError(f"unsupported atan2 1D mode: {mode}")
    n = POINTWISE_N
    lhs_values = [-2.0 + 0.25 * float(index % 16) for index in range(n)]
    rhs_values = [float((index % 5) - 2) for index in range(n)]
    compile_args = (
        bb.tensor(lhs_values, dtype="f32"),
        bb.tensor(rhs_values, dtype="f32"),
        bb.zeros((n,), dtype="f32"),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    lhs = vector_factory(lhs_values, dtype="f32")
    rhs = vector_factory(rhs_values, dtype="f32")
    dst = vector_factory([0.0 for _ in range(n)], dtype="f32")
    if lhs is None or rhs is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (lhs, rhs, dst),
        "result_indices": (),
    }


def _native_math_1d_payload(
    *,
    mode: str,
    op: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("native math 1D payload cannot request both cuda_handle and cuda_dlpack")
    if mode == "indexed":
        n = POINTWISE_N
    elif mode == "direct":
        n = 128
    else:
        raise ValueError(f"unsupported native math 1D mode: {mode}")
    if op == "sin":
        src_values = [0.125 * float(index % 16) for index in range(n)]
    elif op in {"round", "floor", "ceil", "trunc"}:
        src_values = [0.5 * float((index % 16) - 8) + 0.25 for index in range(n)]
    elif op in {"acos", "asin"}:
        src_values = [-0.875 + 0.125 * float(index % 15) for index in range(n)]
    elif op == "atan":
        src_values = [0.25 * float((index % 16) - 8) for index in range(n)]
    elif op == "erf":
        src_values = [0.25 * float((index % 16) - 8) for index in range(n)]
    elif op == "exp":
        src_values = [0.125 * float(index % 8) for index in range(n)]
    elif op == "exp2":
        src_values = [0.25 * float(index % 8) for index in range(n)]
    elif op == "log":
        src_values = [float(1 << (index % 8)) for index in range(n)]
    elif op == "log10":
        src_values = [float(10 ** (index % 4)) for index in range(n)]
    else:
        raise ValueError(f"unsupported native math 1D op: {op}")
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((n,), dtype="f32"),
    )
    if not use_cuda_handle and not use_cuda_dlpack:
        return {"args": compile_args, "result_indices": ()}
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(src_values, dtype="f32")
    dst = vector_factory([0.0 for _ in range(n)], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def indexed_sin_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="sin", backend_name=backend_name)


def indexed_sin_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="sin", backend_name=backend_name, use_cuda_handle=True)


def indexed_sin_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="sin", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_exp_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="exp", backend_name=backend_name)


def indexed_exp_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="exp", backend_name=backend_name, use_cuda_handle=True)


def indexed_exp_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="exp", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_atan_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="atan", backend_name=backend_name)


def indexed_round_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="round", backend_name=backend_name)


def indexed_round_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="round", backend_name=backend_name, use_cuda_handle=True)


def indexed_round_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="round", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_floor_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="floor", backend_name=backend_name)


def indexed_floor_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="floor", backend_name=backend_name, use_cuda_handle=True)


def indexed_floor_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="floor", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_asin_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="asin", backend_name=backend_name)


def indexed_acos_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="acos", backend_name=backend_name)


def indexed_asin_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="asin", backend_name=backend_name, use_cuda_handle=True)


def indexed_acos_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="acos", backend_name=backend_name, use_cuda_handle=True)


def indexed_asin_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="asin", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_acos_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="acos", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_atan_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="atan", backend_name=backend_name, use_cuda_handle=True)


def indexed_atan_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="atan", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_atan2_f32_args(*, backend_name=None, **_kwargs):
    return _atan2_1d_payload(mode="indexed", backend_name=backend_name)


def indexed_atan2_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _atan2_1d_payload(mode="indexed", backend_name=backend_name, use_cuda_handle=True)


def indexed_atan2_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _atan2_1d_payload(mode="indexed", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_erf_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="erf", backend_name=backend_name)


def indexed_erf_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="erf", backend_name=backend_name, use_cuda_handle=True)


def indexed_erf_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="indexed", op="erf", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_abs_f32_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="f32", backend_name=backend_name)


def indexed_abs_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def indexed_abs_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_abs_f16_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="f16", backend_name=backend_name)


def indexed_abs_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def indexed_abs_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_abs_i32_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="i32", backend_name=backend_name)


def indexed_abs_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def indexed_abs_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="indexed", dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_max_f32_args(*, backend_name=None, **_kwargs):
    return _extrema_1d_payload(mode="indexed", dtype="f32", op="max", backend_name=backend_name)


def indexed_max_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _extrema_1d_payload(mode="indexed", dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def indexed_max_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _extrema_1d_payload(mode="indexed", dtype="f32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_min_i32_args(*, backend_name=None, **_kwargs):
    return _extrema_1d_payload(mode="indexed", dtype="i32", op="min", backend_name=backend_name)


def indexed_min_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _extrema_1d_payload(mode="indexed", dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def indexed_min_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _extrema_1d_payload(mode="indexed", dtype="i32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def direct_add_f16_args(*, backend_name=None, **_kwargs):
    return _add_1d_payload(mode="direct", dtype="f16", backend_name=backend_name)


def direct_add_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _add_1d_payload(mode="direct", dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def direct_add_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _add_1d_payload(mode="direct", dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def direct_abs_f32_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="direct", dtype="f32", backend_name=backend_name)


def direct_abs_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="direct", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def direct_abs_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="direct", dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def direct_abs_i32_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="direct", dtype="i32", backend_name=backend_name)


def direct_abs_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="direct", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def direct_abs_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _abs_1d_payload(mode="direct", dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def direct_exp2_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="exp2", backend_name=backend_name)


def direct_exp2_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="exp2", backend_name=backend_name, use_cuda_handle=True)


def direct_exp2_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="exp2", backend_name=backend_name, use_cuda_dlpack=True)


def direct_atan_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="atan", backend_name=backend_name)


def direct_round_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="round", backend_name=backend_name)


def direct_round_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="round", backend_name=backend_name, use_cuda_handle=True)


def direct_round_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="round", backend_name=backend_name, use_cuda_dlpack=True)


def direct_ceil_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="ceil", backend_name=backend_name)


def direct_ceil_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="ceil", backend_name=backend_name, use_cuda_handle=True)


def direct_ceil_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="ceil", backend_name=backend_name, use_cuda_dlpack=True)


def direct_asin_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="asin", backend_name=backend_name)


def direct_acos_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="acos", backend_name=backend_name)


def direct_asin_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="asin", backend_name=backend_name, use_cuda_handle=True)


def direct_acos_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="acos", backend_name=backend_name, use_cuda_handle=True)


def direct_asin_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="asin", backend_name=backend_name, use_cuda_dlpack=True)


def direct_acos_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="acos", backend_name=backend_name, use_cuda_dlpack=True)


def direct_atan_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="atan", backend_name=backend_name, use_cuda_handle=True)


def direct_atan_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="atan", backend_name=backend_name, use_cuda_dlpack=True)


def direct_log10_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="log10", backend_name=backend_name)


def direct_log10_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="log10", backend_name=backend_name, use_cuda_handle=True)


def direct_log10_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="log10", backend_name=backend_name, use_cuda_dlpack=True)


def direct_erf_f32_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="erf", backend_name=backend_name)


def direct_erf_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="erf", backend_name=backend_name, use_cuda_handle=True)


def direct_erf_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _native_math_1d_payload(mode="direct", op="erf", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_neg_f32_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="indexed", dtype="f32", backend_name=backend_name)


def indexed_neg_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="indexed", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def indexed_neg_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="indexed", dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_neg_i32_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="indexed", dtype="i32", backend_name=backend_name)


def indexed_neg_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="indexed", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def indexed_neg_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="indexed", dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def direct_neg_f32_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="f32", backend_name=backend_name)


def direct_neg_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def direct_neg_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def direct_neg_i32_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="i32", backend_name=backend_name)


def direct_neg_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def direct_neg_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def direct_neg_f16_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="f16", backend_name=backend_name)


def direct_neg_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def direct_neg_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _neg_1d_payload(mode="direct", dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_sqrt_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = [4.0 + float(index % 7) for index in range(POINTWISE_N)]
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((POINTWISE_N,), dtype="f32"),
        ),
        "result_indices": (),
    }


def indexed_sqrt_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = [4.0 + float(index % 7) for index in range(POINTWISE_N)]
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((POINTWISE_N,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(POINTWISE_N)], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def indexed_rsqrt_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = [4.0 + float(index % 7) for index in range(POINTWISE_N)]
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((POINTWISE_N,), dtype="f32"),
        ),
        "result_indices": (),
    }


def indexed_rsqrt_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = [4.0 + float(index % 7) for index in range(POINTWISE_N)]
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((POINTWISE_N,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(POINTWISE_N)], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def direct_sqrt_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    n = 128
    src_values = [4.0 + float(index % 7) for index in range(n)]
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((n,), dtype="f32"),
        ),
        "result_indices": (),
    }


def direct_sqrt_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = [4.0 + float(index % 7) for index in range(n)]
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((n,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(n)], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def direct_rsqrt_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    n = 128
    src_values = [4.0 + float(index % 7) for index in range(n)]
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((n,), dtype="f32"),
        ),
        "result_indices": (),
    }


def direct_rsqrt_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = [4.0 + float(index % 7) for index in range(n)]
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((n,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(n)], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def direct_bitand_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    other_values = _vector_i32(n, scale=5, offset=11)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor(other_values, dtype="i32"),
            bb.zeros((n,), dtype="i32"),
        ),
        "result_indices": (),
    }


def direct_bitand_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    other_values = _vector_i32(n, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    other = _maybe_cuda_handle_vector(other_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(n)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def direct_bitand_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    other_values = _vector_i32(n, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    other = _maybe_cuda_dlpack_vector(other_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(n)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def direct_bitor_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    other_values = _vector_i32(n, scale=5, offset=11)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor(other_values, dtype="i32"),
            bb.zeros((n,), dtype="i32"),
        ),
        "result_indices": (),
    }


def direct_bitor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    other_values = _vector_i32(n, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    other = _maybe_cuda_handle_vector(other_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(n)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def direct_bitor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    other_values = _vector_i32(n, scale=5, offset=11)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.tensor(other_values, dtype="i32"),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    other = _maybe_cuda_dlpack_vector(other_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(n)], dtype="i32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def direct_scalar_broadcast_bitor_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.Int32(24),
            bb.zeros((n,), dtype="i32"),
        ),
        "result_indices": (),
    }


def direct_scalar_broadcast_bitor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.Int32(24),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(n)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 24, dst),
        "result_indices": (),
    }


def direct_scalar_broadcast_bitor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.Int32(24),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(n)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 24, dst),
        "result_indices": (),
    }


def direct_scalar_broadcast_bitand_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.Int32(11),
            bb.zeros((n,), dtype="i32"),
        ),
        "result_indices": (),
    }


def direct_scalar_broadcast_bitand_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.Int32(11),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0 for _ in range(n)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 11, dst),
        "result_indices": (),
    }


def direct_scalar_broadcast_bitand_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    n = 128
    src_values = _vector_i32(n, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.Int32(11),
        bb.zeros((n,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="i32")
    dst = _maybe_cuda_dlpack_vector([0 for _ in range(n)], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, 11, dst),
        "result_indices": (),
    }


def direct_select_scalar_f32_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(mode="direct", scalar_mode="param", dtype="f32", backend_name=backend_name)


def direct_select_scalar_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(
        mode="direct",
        scalar_mode="param",
        dtype="f32",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def direct_select_scalar_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _select_1d_payload(
        mode="direct",
        scalar_mode="param",
        dtype="f32",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def reduce_add_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((1,), dtype="f32"),
        ),
        "result_indices": (),
    }


def reduce_add_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((1,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def _copy_reduce_1d_payload(
    *,
    dtype: str,
    reduction: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    if (dtype, reduction) == ("f32", "add"):
        src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
        dst_values = [0.0 for _ in range(POINTWISE_N)]
    elif (dtype, reduction) == ("f32", "max"):
        src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
        dst_values = _vector_f32(POINTWISE_N, scale=0.25, offset=0.5)
    elif (dtype, reduction) == ("i32", "xor"):
        src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
        dst_values = [0 for _ in range(POINTWISE_N)]
    elif (dtype, reduction) == ("i32", "or"):
        src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
        dst_values = [1 for _ in range(POINTWISE_N)]
    else:
        raise ValueError(f"unsupported copy_reduce 1D payload for dtype={dtype!r}, reduction={reduction!r}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.tensor(dst_values, dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("copy_reduce 1D payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(src_values, dtype=dtype)
    dst = vector_factory(dst_values, dtype=dtype)
    if src is None or dst is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def copy_reduce_add_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="f32", reduction="add", backend_name=backend_name)


def copy_reduce_add_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def copy_reduce_add_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def copy_reduce_max_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="f32", reduction="max", backend_name=backend_name)


def copy_reduce_max_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def copy_reduce_max_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def copy_reduce_xor_i32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="i32", reduction="xor", backend_name=backend_name)


def copy_reduce_xor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="xor",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def copy_reduce_xor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="xor",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def copy_reduce_or_i32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="i32", reduction="or", backend_name=backend_name)


def copy_reduce_or_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def copy_reduce_or_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def indexed_copy_reduce_add_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="f32", reduction="add", backend_name=backend_name)


def indexed_copy_reduce_add_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def indexed_copy_reduce_add_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def indexed_copy_reduce_max_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="f32", reduction="max", backend_name=backend_name)


def indexed_copy_reduce_max_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def indexed_copy_reduce_max_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def indexed_copy_reduce_xor_i32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="i32", reduction="xor", backend_name=backend_name)


def indexed_copy_reduce_xor_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="xor",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def indexed_copy_reduce_xor_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="xor",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def indexed_copy_reduce_or_i32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(dtype="i32", reduction="or", backend_name=backend_name)


def indexed_copy_reduce_or_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def indexed_copy_reduce_or_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_1d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def parallel_reduce_add_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((1,), dtype="f32"),
        ),
        "result_indices": (),
    }


def parallel_reduce_add_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((1,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0], dtype="f32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def parallel_reduce_add_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.zeros((1,), dtype="i32"),
        ),
        "result_indices": (),
    }


def parallel_reduce_add_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_i32(POINTWISE_N, scale=3, offset=7)
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.zeros((1,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="i32")
    dst = _maybe_cuda_handle_vector([0], dtype="i32")
    if src is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def _copy_reduce_2d_payload(
    *,
    dtype: str,
    reduction: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if (dtype, reduction) == ("f32", "add"):
        src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
        dst_values = [[0.0 for _ in range(cols)] for _ in range(rows)]
    elif (dtype, reduction) == ("f32", "max"):
        src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
        dst_values = _matrix_f32(rows, cols, scale=0.25, offset=0.5)
    elif (dtype, reduction) == ("i32", "or"):
        src_values = _matrix_i32(rows, cols, scale=3, offset=7)
        dst_values = [[1 for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError(f"unsupported copy_reduce 2D payload for dtype={dtype!r}, reduction={reduction!r}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.tensor(dst_values, dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("copy_reduce 2D payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    flat_src = [value for row in src_values for value in row]
    flat_dst = [value for row in dst_values for value in row]
    src = vector_factory(flat_src, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    dst = vector_factory(flat_dst, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    if src is None or dst is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst),
        "result_indices": (),
    }


def copy_reduce_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(dtype="f32", reduction="add", backend_name=backend_name)


def copy_reduce_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def copy_reduce_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def copy_reduce_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(dtype="f32", reduction="max", backend_name=backend_name)


def copy_reduce_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def copy_reduce_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def copy_reduce_or_2d_i32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(dtype="i32", reduction="or", backend_name=backend_name)


def copy_reduce_or_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def copy_reduce_or_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def parallel_copy_reduce_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(dtype="f32", reduction="add", backend_name=backend_name)


def parallel_copy_reduce_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def parallel_copy_reduce_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def parallel_copy_reduce_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(dtype="f32", reduction="max", backend_name=backend_name)


def parallel_copy_reduce_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def parallel_copy_reduce_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="max",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def parallel_copy_reduce_or_2d_i32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(dtype="i32", reduction="or", backend_name=backend_name)


def parallel_copy_reduce_or_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def parallel_copy_reduce_or_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="i32",
        reduction="or",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def multiblock_copy_reduce_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(dtype="f32", reduction="add", backend_name=backend_name)


def multiblock_copy_reduce_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_handle=True,
    )


def multiblock_copy_reduce_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _copy_reduce_2d_payload(
        dtype="f32",
        reduction="add",
        backend_name=backend_name,
        use_cuda_dlpack=True,
    )


def parallel_reduce_add_2d_bundle_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _matrix_f32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((1,), dtype="f32"),
            bb.zeros((FLYDSL_MICRO_ROWS,), dtype="f32"),
            bb.zeros((FLYDSL_MICRO_COLS,), dtype="f32"),
        ),
        "result_indices": (),
    }


def parallel_reduce_add_2d_bundle_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
    flat_values = [value for row in src_values for value in row]
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((1,), dtype="f32"),
        bb.zeros((rows,), dtype="f32"),
        bb.zeros((cols,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(flat_values, dtype="f32", shape=(rows, cols), stride=(cols, 1))
    dst_scalar = _maybe_cuda_handle_vector([0.0], dtype="f32")
    dst_rows = _maybe_cuda_handle_vector([0.0 for _ in range(rows)], dtype="f32")
    dst_cols = _maybe_cuda_handle_vector([0.0 for _ in range(cols)], dtype="f32")
    if src is None or dst_scalar is None or dst_rows is None or dst_cols is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_scalar, dst_rows, dst_cols),
        "result_indices": (),
    }


def parallel_reduce_add_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _matrix_i32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.zeros((1,), dtype="i32"),
            bb.zeros((FLYDSL_MICRO_ROWS,), dtype="i32"),
            bb.zeros((FLYDSL_MICRO_COLS,), dtype="i32"),
        ),
        "result_indices": (),
    }


def parallel_reduce_add_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    src_values = _matrix_i32(rows, cols, scale=3, offset=7)
    flat_values = [value for row in src_values for value in row]
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.zeros((1,), dtype="i32"),
        bb.zeros((rows,), dtype="i32"),
        bb.zeros((cols,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(flat_values, dtype="i32", shape=(rows, cols), stride=(cols, 1))
    dst_scalar = _maybe_cuda_handle_vector([0], dtype="i32")
    dst_rows = _maybe_cuda_handle_vector([0 for _ in range(rows)], dtype="i32")
    dst_cols = _maybe_cuda_handle_vector([0 for _ in range(cols)], dtype="i32")
    if src is None or dst_scalar is None or dst_rows is None or dst_cols is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_scalar, dst_rows, dst_cols),
        "result_indices": (),
    }


def _parallel_reduce_2d_bundle_payload(
    *,
    dtype: str,
    op: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if dtype == "f32":
        src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
        if op == "mul":
            scalar_init = [1.0]
            row_init_value = 1.0
            col_init_value = 1.0
        elif op == "max":
            scalar_init = [-99.0]
            row_init_value = -99.0
            col_init_value = -99.0
        elif op == "min":
            scalar_init = [999.0]
            row_init_value = 999.0
            col_init_value = 999.0
        else:
            raise ValueError(f"unsupported 2D parallel reduce op: {op}")
    elif dtype == "i32":
        src_values = _matrix_i32(rows, cols, scale=3, offset=7)
        if op == "mul":
            scalar_init = [1]
            row_init_value = 1
            col_init_value = 1
        elif op == "and":
            scalar_init = [-1]
            row_init_value = -1
            col_init_value = -1
        elif op == "or":
            scalar_init = [0]
            row_init_value = 0
            col_init_value = 0
        elif op == "xor":
            scalar_init = [0]
            row_init_value = 0
            col_init_value = 0
        elif op == "max":
            scalar_init = [-99]
            row_init_value = -99
            col_init_value = -99
        elif op == "min":
            scalar_init = [999]
            row_init_value = 999
            col_init_value = 999
        else:
            raise ValueError(f"unsupported 2D parallel reduce op: {op}")
    else:
        raise ValueError(f"unsupported 2D parallel reduce dtype: {dtype}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.tensor(scalar_init, dtype=dtype),
        bb.tensor([row_init_value for _ in range(rows)], dtype=dtype),
        bb.tensor([col_init_value for _ in range(cols)], dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("parallel 2D bundle payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    flat_values = [value for row in src_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(flat_values, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    dst_scalar = vector_factory(scalar_init, dtype=dtype)
    dst_rows = vector_factory([row_init_value for _ in range(rows)], dtype=dtype)
    dst_cols = vector_factory([col_init_value for _ in range(cols)], dtype=dtype)
    if src is None or dst_scalar is None or dst_rows is None or dst_cols is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_scalar, dst_rows, dst_cols),
        "result_indices": (),
    }


def parallel_reduce_mul_2d_bundle_f32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="f32", op="mul", backend_name=backend_name)


def parallel_reduce_mul_2d_bundle_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_max_2d_bundle_f32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="f32", op="max", backend_name=backend_name)


def parallel_reduce_max_2d_bundle_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_min_2d_bundle_f32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="f32", op="min", backend_name=backend_name)


def parallel_reduce_min_2d_bundle_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_mul_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="mul", backend_name=backend_name)


def parallel_reduce_mul_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_and_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="and", backend_name=backend_name)


def parallel_reduce_and_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="and", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_and_2d_bundle_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="and", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_or_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="or", backend_name=backend_name)


def parallel_reduce_or_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="or", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_or_2d_bundle_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="or", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_xor_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="xor", backend_name=backend_name)


def parallel_reduce_xor_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="xor", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_xor_2d_bundle_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="xor", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_max_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="max", backend_name=backend_name)


def parallel_reduce_max_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_min_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="min", backend_name=backend_name)


def parallel_reduce_min_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _parallel_reduce_2d_bundle_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def _tensor_factory_2d_payload(
    *,
    dtype: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    compile_args = (
        bb.zeros((rows, cols), dtype=dtype),
        bb.zeros((rows, cols), dtype=dtype),
        bb.zeros((rows, cols), dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("tensor-factory 2D payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    zero_fill = 0.0 if dtype == "f32" else 0
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    zero = vector_factory([zero_fill for _ in range(rows * cols)], dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    one = vector_factory([zero_fill for _ in range(rows * cols)], dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    full = vector_factory([zero_fill for _ in range(rows * cols)], dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    if zero is None or one is None or full is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (zero, one, full),
        "result_indices": (),
    }


def tensor_factory_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="f32", backend_name=backend_name)


def tensor_factory_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def tensor_factory_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="i32", backend_name=backend_name)


def tensor_factory_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def parallel_tensor_factory_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="f32", backend_name=backend_name)


def parallel_tensor_factory_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_tensor_factory_2d_i32_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="i32", backend_name=backend_name)


def parallel_tensor_factory_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def multiblock_tensor_factory_2d_f32_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="f32", backend_name=backend_name)


def multiblock_tensor_factory_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def multiblock_tensor_factory_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _tensor_factory_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_add_2d_bundle_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _matrix_f32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((1,), dtype="f32"),
            bb.zeros((FLYDSL_MICRO_ROWS,), dtype="f32"),
            bb.zeros((FLYDSL_MICRO_COLS,), dtype="f32"),
        ),
        "result_indices": (),
    }


def reduce_add_2d_bundle_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
    flat_values = [value for row in src_values for value in row]
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.zeros((1,), dtype="f32"),
        bb.zeros((rows,), dtype="f32"),
        bb.zeros((cols,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(flat_values, dtype="f32", shape=(rows, cols), stride=(cols, 1))
    dst_scalar = _maybe_cuda_handle_vector([0.0], dtype="f32")
    dst_rows = _maybe_cuda_handle_vector([0.0 for _ in range(rows)], dtype="f32")
    dst_cols = _maybe_cuda_handle_vector([0.0 for _ in range(cols)], dtype="f32")
    if src is None or dst_scalar is None or dst_rows is None or dst_cols is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_scalar, dst_rows, dst_cols),
        "result_indices": (),
    }


def reduce_add_2d_bundle_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _matrix_i32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.zeros((1,), dtype="i32"),
            bb.zeros((FLYDSL_MICRO_ROWS,), dtype="i32"),
            bb.zeros((FLYDSL_MICRO_COLS,), dtype="i32"),
        ),
        "result_indices": (),
    }


def reduce_add_2d_bundle_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    src_values = _matrix_i32(rows, cols, scale=3, offset=7)
    flat_values = [value for row in src_values for value in row]
    compile_args = (
        bb.tensor(src_values, dtype="i32"),
        bb.zeros((1,), dtype="i32"),
        bb.zeros((rows,), dtype="i32"),
        bb.zeros((cols,), dtype="i32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(flat_values, dtype="i32", shape=(rows, cols), stride=(cols, 1))
    dst_scalar = _maybe_cuda_handle_vector([0], dtype="i32")
    dst_rows = _maybe_cuda_handle_vector([0 for _ in range(rows)], dtype="i32")
    dst_cols = _maybe_cuda_handle_vector([0 for _ in range(cols)], dtype="i32")
    if src is None or dst_scalar is None or dst_rows is None or dst_cols is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_scalar, dst_rows, dst_cols),
        "result_indices": (),
    }


def _reduce_rows_2d_payload(
    *,
    dtype: str,
    op: str,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if dtype == "f32":
        src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
        if op == "add":
            init = 0.0
        elif op == "mul":
            init = 1.0
        elif op == "max":
            init = -99.0
        elif op == "min":
            init = 999.0
        else:
            raise ValueError(f"unsupported 2D row-reduce op: {op}")
    elif dtype == "i32":
        src_values = _matrix_i32(rows, cols, scale=3, offset=7)
        if op == "add":
            init = 0
        elif op == "mul":
            init = 1
        elif op == "and":
            init = -1
        elif op == "or":
            init = 0
        elif op == "xor":
            init = 0
        elif op == "max":
            init = -99
        elif op == "min":
            init = 999
        else:
            raise ValueError(f"unsupported 2D row-reduce op: {op}")
    else:
        raise ValueError(f"unsupported 2D row-reduce dtype: {dtype}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.tensor([init for _ in range(rows)], dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("row-reduce payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    flat_values = [value for row in src_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(flat_values, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    dst_rows = vector_factory([init for _ in range(rows)], dtype=dtype)
    if src is None or dst_rows is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_rows),
        "result_indices": (),
    }


def _reduce_cols_2d_payload(
    *,
    dtype: str,
    op: str = "add",
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    if dtype == "f32":
        src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
        if op == "add":
            init = 0.0
        elif op == "mul":
            init = 1.0
        elif op == "max":
            init = -99.0
        elif op == "min":
            init = 999.0
        else:
            raise ValueError(f"unsupported 2D col-reduce op: {op}")
    elif dtype == "i32":
        src_values = _matrix_i32(rows, cols, scale=3, offset=7)
        if op == "add":
            init = 0
        elif op == "mul":
            init = 1
        elif op == "and":
            init = -1
        elif op == "or":
            init = 0
        elif op == "xor":
            init = 0
        elif op == "max":
            init = -99
        elif op == "min":
            init = 999
        else:
            raise ValueError(f"unsupported 2D col-reduce op: {op}")
    else:
        raise ValueError(f"unsupported 2D col-reduce dtype: {dtype}")
    compile_args = (
        bb.tensor(src_values, dtype=dtype),
        bb.tensor([init for _ in range(cols)], dtype=dtype),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("col-reduce payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    flat_values = [value for row in src_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(flat_values, dtype=dtype, shape=(rows, cols), stride=(cols, 1))
    dst_cols = vector_factory([init for _ in range(cols)], dtype=dtype)
    if src is None or dst_cols is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_cols),
        "result_indices": (),
    }


def _rowcol_reduce_bundle_2d_payload(
    *,
    backend_name=None,
    use_cuda_handle: bool = False,
    use_cuda_dlpack: bool = False,
):
    rows = FLYDSL_MICRO_ROWS
    cols = FLYDSL_MICRO_COLS
    src_values = _matrix_f32(rows, cols, scale=0.5, offset=1.0)
    rows_init = [0.0 for _ in range(rows)]
    cols_init = [1.0 for _ in range(cols)]
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.tensor(rows_init, dtype="f32"),
        bb.tensor(cols_init, dtype="f32"),
    )
    if use_cuda_handle and use_cuda_dlpack:
        raise ValueError("rowcol-reduce payload cannot request both cuda_handle and cuda_dlpack")
    if not use_cuda_handle and not use_cuda_dlpack:
        return {
            "args": compile_args,
            "result_indices": (),
        }
    if backend_name != "ptx_exec":
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    flat_values = [value for row in src_values for value in row]
    vector_factory = _maybe_cuda_handle_vector if use_cuda_handle else _maybe_cuda_dlpack_vector
    src = vector_factory(flat_values, dtype="f32", shape=(rows, cols), stride=(cols, 1))
    dst_rows = vector_factory(rows_init, dtype="f32")
    dst_cols = vector_factory(cols_init, dtype="f32")
    if src is None or dst_rows is None or dst_cols is None:
        return {
            "compile_args": compile_args,
            "run_args": compile_args,
            "result_indices": (),
        }
    return {
        "compile_args": compile_args,
        "run_args": (src, dst_rows, dst_cols),
        "result_indices": (),
    }


def reduce_rows_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name)


def reduce_rows_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_rows_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="add", backend_name=backend_name)


def reduce_rows_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="add", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_add_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="add", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_rows_mul_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="mul", backend_name=backend_name)


def reduce_rows_mul_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_mul_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_rows_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="max", backend_name=backend_name)


def reduce_rows_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_rows_min_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="min", backend_name=backend_name)


def reduce_rows_min_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_min_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_rows_mul_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="mul", backend_name=backend_name)


def reduce_rows_mul_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_mul_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_rows_max_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="max", backend_name=backend_name)


def reduce_rows_max_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_max_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_rows_min_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="min", backend_name=backend_name)


def reduce_rows_min_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def reduce_rows_min_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name)


def parallel_reduce_rows_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="add", backend_name=backend_name)


def parallel_reduce_rows_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="add", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_add_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="add", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_mul_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="mul", backend_name=backend_name)


def parallel_reduce_rows_mul_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_mul_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="max", backend_name=backend_name)


def parallel_reduce_rows_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_min_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="min", backend_name=backend_name)


def parallel_reduce_rows_min_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_min_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_mul_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="mul", backend_name=backend_name)


def parallel_reduce_rows_mul_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_mul_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_and_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="and", backend_name=backend_name)


def parallel_reduce_rows_and_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="and", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_and_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="and", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_or_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="or", backend_name=backend_name)


def parallel_reduce_rows_or_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="or", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_or_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="or", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_xor_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="xor", backend_name=backend_name)


def parallel_reduce_rows_xor_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="xor", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_xor_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="xor", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_max_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="max", backend_name=backend_name)


def parallel_reduce_rows_max_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_max_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_rows_min_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="min", backend_name=backend_name)


def parallel_reduce_rows_min_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_rows_min_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", backend_name=backend_name)


def reduce_cols_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", backend_name=backend_name)


def reduce_cols_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_add_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_mul_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="mul", backend_name=backend_name)


def reduce_cols_mul_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_mul_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="max", backend_name=backend_name)


def reduce_cols_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_min_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="min", backend_name=backend_name)


def reduce_cols_min_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_min_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_mul_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="mul", backend_name=backend_name)


def reduce_cols_mul_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_mul_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_max_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="max", backend_name=backend_name)


def reduce_cols_max_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_max_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def reduce_cols_min_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="min", backend_name=backend_name)


def reduce_cols_min_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def reduce_cols_min_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", backend_name=backend_name)


def parallel_reduce_cols_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_add_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", backend_name=backend_name)


def parallel_reduce_cols_add_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_add_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_mul_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="mul", backend_name=backend_name)


def parallel_reduce_cols_mul_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_mul_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_max_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="max", backend_name=backend_name)


def parallel_reduce_cols_max_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_max_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_min_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="min", backend_name=backend_name)


def parallel_reduce_cols_min_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_min_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_mul_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="mul", backend_name=backend_name)


def parallel_reduce_cols_mul_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_mul_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="mul", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_and_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="and", backend_name=backend_name)


def parallel_reduce_cols_and_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="and", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_and_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="and", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_or_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="or", backend_name=backend_name)


def parallel_reduce_cols_or_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="or", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_or_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="or", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_xor_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="xor", backend_name=backend_name)


def parallel_reduce_cols_xor_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="xor", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_xor_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="xor", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_max_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="max", backend_name=backend_name)


def parallel_reduce_cols_max_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_max_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="max", backend_name=backend_name, use_cuda_dlpack=True)


def parallel_reduce_cols_min_2d_i32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="min", backend_name=backend_name)


def parallel_reduce_cols_min_2d_i32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_handle=True)


def parallel_reduce_cols_min_2d_i32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="i32", op="min", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_reduce_rows_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name)


def multiblock_reduce_rows_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_handle=True)


def multiblock_reduce_rows_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_rows_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_reduce_cols_add_2d_f32_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="add", backend_name=backend_name)


def multiblock_reduce_cols_add_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_handle=True)


def multiblock_reduce_cols_add_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _reduce_cols_2d_payload(dtype="f32", op="add", backend_name=backend_name, use_cuda_dlpack=True)


def multiblock_reduce_add_rowcol_2d_f32_args(*, backend_name=None, **_kwargs):
    return _rowcol_reduce_bundle_2d_payload(backend_name=backend_name)


def multiblock_reduce_add_rowcol_2d_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _rowcol_reduce_bundle_2d_payload(backend_name=backend_name, use_cuda_handle=True)


def multiblock_reduce_add_rowcol_2d_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _rowcol_reduce_bundle_2d_payload(backend_name=backend_name, use_cuda_dlpack=True)


def indexed_add_f32_cuda_handle_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    other_values = _vector_f32(POINTWISE_N, scale=0.25, offset=2.0)
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.tensor(other_values, dtype="f32"),
        bb.zeros((POINTWISE_N,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_handle_vector(src_values, dtype="f32")
    other = _maybe_cuda_handle_vector(other_values, dtype="f32")
    dst = _maybe_cuda_handle_vector([0.0 for _ in range(POINTWISE_N)], dtype="f32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def indexed_add_f32_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    other_values = _vector_f32(POINTWISE_N, scale=0.25, offset=2.0)
    compile_args = (
        bb.tensor(src_values, dtype="f32"),
        bb.tensor(other_values, dtype="f32"),
        bb.zeros((POINTWISE_N,), dtype="f32"),
    )
    if backend_name != "ptx_exec":
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    src = _maybe_cuda_dlpack_vector(src_values, dtype="f32")
    other = _maybe_cuda_dlpack_vector(other_values, dtype="f32")
    dst = _maybe_cuda_dlpack_vector([0.0 for _ in range(POINTWISE_N)], dtype="f32")
    if src is None or other is None or dst is None:
        return {"compile_args": compile_args, "run_args": compile_args, "result_indices": ()}
    return {
        "compile_args": compile_args,
        "run_args": (src, other, dst),
        "result_indices": (),
    }


def aster_dense_copy_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(ASTER_POINTWISE_N, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.zeros((ASTER_POINTWISE_N,), dtype="f32"),
        ),
        "result_indices": (),
    }


def aster_dense_add_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(ASTER_POINTWISE_N, scale=0.5, offset=1.0)
    other_values = _vector_f32(ASTER_POINTWISE_N, scale=0.25, offset=2.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.tensor(other_values, dtype="f32"),
            bb.zeros((ASTER_POINTWISE_N,), dtype="f32"),
        ),
        "result_indices": (),
    }


def aster_dense_copy_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(ASTER_POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.zeros((ASTER_POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def aster_dense_add_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(ASTER_POINTWISE_N, scale=3, offset=7)
    other_values = _vector_i32(ASTER_POINTWISE_N, scale=5, offset=11)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor(other_values, dtype="i32"),
            bb.zeros((ASTER_POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def aster_dense_copy_f16_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(ASTER_POINTWISE_N, scale=0.125, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f16"),
            bb.zeros((ASTER_POINTWISE_N,), dtype="f16"),
        ),
        "result_indices": (),
    }


def aster_broadcast_add_f32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_f32(ASTER_POINTWISE_N, scale=0.5, offset=1.0)
    return {
        "args": (
            bb.tensor(src_values, dtype="f32"),
            bb.tensor([3.5], dtype="f32"),
            bb.zeros((ASTER_POINTWISE_N,), dtype="f32"),
        ),
        "result_indices": (),
    }


def aster_broadcast_add_i32_args(*, backend_name=None, **_kwargs):
    del backend_name
    src_values = _vector_i32(ASTER_POINTWISE_N, scale=3, offset=7)
    return {
        "args": (
            bb.tensor(src_values, dtype="i32"),
            bb.tensor([13], dtype="i32"),
            bb.zeros((ASTER_POINTWISE_N,), dtype="i32"),
        ),
        "result_indices": (),
    }


def indexed_add_f32_args(*, backend_name=None, **_kwargs):
    src, other, dst = _make_f32_vector_args(POINTWISE_N, backend_name=backend_name or "")
    return {"args": (src, other, dst), "result_indices": ()}


def indexed_add_f16_args(*, backend_name=None, **_kwargs):
    return _add_1d_payload(mode="indexed", dtype="f16", backend_name=backend_name)


def indexed_add_f16_cuda_handle_args(*, backend_name=None, **_kwargs):
    return _add_1d_payload(mode="indexed", dtype="f16", backend_name=backend_name, use_cuda_handle=True)


def indexed_add_f16_cuda_dlpack_args(*, backend_name=None, **_kwargs):
    return _add_1d_payload(mode="indexed", dtype="f16", backend_name=backend_name, use_cuda_dlpack=True)


def indexed_sub_f32_args(*, backend_name=None, **_kwargs):
    src, other, dst = _make_f32_vector_args(POINTWISE_N, backend_name=backend_name or "")
    return {"args": (src, other, dst), "result_indices": ()}


def indexed_mul_f32_args(*, backend_name=None, **_kwargs):
    src, other, dst = _make_f32_vector_args(POINTWISE_N, backend_name=backend_name or "")
    return {"args": (src, other, dst), "result_indices": ()}


def indexed_div_f32_args(*, backend_name=None, **_kwargs):
    src, other, dst = _make_f32_vector_args(POINTWISE_N, backend_name=backend_name or "")
    return {"args": (src, other, dst), "result_indices": ()}


def flydsl_unary_sin_f32_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(FLYDSL_MICRO_N, scale=0.001, offset=0.25)
    src = _maybe_torch_tensor(src_values, shape=(FLYDSL_MICRO_N,), dtype="f32", backend_name=backend_name or "")
    dst = _maybe_torch_tensor([0.0 for _ in range(FLYDSL_MICRO_N)], shape=(FLYDSL_MICRO_N,), dtype="f32", backend_name=backend_name or "")
    if src is None or dst is None:
        src = bb.tensor(src_values, dtype="f32")
        dst = bb.zeros((FLYDSL_MICRO_N,), dtype="f32")
    return {"args": (src, dst), "result_indices": ()}


def flydsl_unary_rsqrt_f32_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(FLYDSL_MICRO_N, scale=0.001, offset=1.0)
    src = _maybe_torch_tensor(src_values, shape=(FLYDSL_MICRO_N,), dtype="f32", backend_name=backend_name or "")
    dst = _maybe_torch_tensor([0.0 for _ in range(FLYDSL_MICRO_N)], shape=(FLYDSL_MICRO_N,), dtype="f32", backend_name=backend_name or "")
    if src is None or dst is None:
        src = bb.tensor(src_values, dtype="f32")
        dst = bb.zeros((FLYDSL_MICRO_N,), dtype="f32")
    return {"args": (src, dst), "result_indices": ()}


def flydsl_broadcast_add_2d_args(*, backend_name=None, **_kwargs):
    lhs_values = _matrix_f32(FLYDSL_MICRO_ROWS, 1, scale=0.5, offset=1.0)
    rhs_values = _matrix_f32(1, FLYDSL_MICRO_COLS, scale=0.25, offset=2.0)
    dst_values = _matrix_f32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=0.0, offset=0.0)
    lhs = _maybe_torch_tensor(lhs_values, shape=(FLYDSL_MICRO_ROWS, 1), dtype="f32", backend_name=backend_name or "")
    rhs = _maybe_torch_tensor(rhs_values, shape=(1, FLYDSL_MICRO_COLS), dtype="f32", backend_name=backend_name or "")
    dst = _maybe_torch_tensor(
        dst_values,
        shape=(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS),
        dtype="f32",
        backend_name=backend_name or "",
    )
    if lhs is None or rhs is None or dst is None:
        lhs = bb.tensor(lhs_values, dtype="f32")
        rhs = bb.tensor(rhs_values, dtype="f32")
        dst = bb.zeros((FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS), dtype="f32")
    return {"args": (lhs, rhs, dst), "result_indices": ()}


def flydsl_reduce_add_2d_args(*, backend_name=None, **_kwargs):
    src_values = _matrix_f32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=0.125, offset=1.0)
    src = _maybe_torch_tensor(
        src_values,
        shape=(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS),
        dtype="f32",
        backend_name=backend_name or "",
    )
    dst_scalar = _maybe_torch_tensor([0.0], shape=(1,), dtype="f32", backend_name=backend_name or "")
    dst_rows = _maybe_torch_tensor(
        [0.0 for _ in range(FLYDSL_MICRO_ROWS)],
        shape=(FLYDSL_MICRO_ROWS,),
        dtype="f32",
        backend_name=backend_name or "",
    )
    if src is None or dst_scalar is None or dst_rows is None:
        src = bb.tensor(src_values, dtype="f32")
        dst_scalar = bb.zeros((1,), dtype="f32")
        dst_rows = bb.zeros((FLYDSL_MICRO_ROWS,), dtype="f32")
    return {"args": (src, dst_scalar, dst_rows), "result_indices": ()}


def flydsl_unary_math_2d_args(*, backend_name=None, **_kwargs):
    src_values = _matrix_f32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=0.001, offset=1.0)
    src = _maybe_torch_tensor(
        src_values,
        shape=(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS),
        dtype="f32",
        backend_name=backend_name or "",
    )
    dsts = [
        _maybe_torch_tensor(
            _matrix_f32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=0.0, offset=0.0),
            shape=(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS),
            dtype="f32",
            backend_name=backend_name or "",
        )
        for _ in range(4)
    ]
    if src is None or any(dst is None for dst in dsts):
        src = bb.tensor(src_values, dtype="f32")
        dsts = [bb.zeros((FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS), dtype="f32") for _ in range(4)]
    return {"args": (src, *dsts), "result_indices": ()}


def flydsl_shared_stage_f32_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(FLYDSL_SHARED_N, scale=0.125, offset=1.0)
    src = _maybe_torch_tensor(src_values, shape=(FLYDSL_SHARED_N,), dtype="f32", backend_name=backend_name or "")
    dst = _maybe_torch_tensor([0.0 for _ in range(FLYDSL_SHARED_N)], shape=(FLYDSL_SHARED_N,), dtype="f32", backend_name=backend_name or "")
    if src is None or dst is None:
        src = bb.tensor(src_values, dtype="f32")
        dst = bb.zeros((FLYDSL_SHARED_N,), dtype="f32")
    return {"args": (src, dst), "result_indices": ()}


def flydsl_tensor_factory_2d_args(*, backend_name=None, **_kwargs):
    dsts = [
        _maybe_torch_tensor(
            _matrix_f32(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS, scale=0.0, offset=0.0),
            shape=(FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS),
            dtype="f32",
            backend_name=backend_name or "",
        )
        for _ in range(3)
    ]
    if any(dst is None for dst in dsts):
        dsts = [bb.zeros((FLYDSL_MICRO_ROWS, FLYDSL_MICRO_COLS), dtype="f32") for _ in range(3)]
    return {"args": tuple(dsts), "result_indices": ()}


def hipkittens_bf16_gemm_args(**_kwargs):
    a = bb.tensor([[row * 16 + col + 1 for col in range(16)] for row in range(32)], dtype="bf16")
    b = bb.tensor([[1.0 if col == row else 0.0 for col in range(32)] for row in range(16)], dtype="bf16")
    c = bb.zeros((32, 32), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}


def hipkittens_f16_gemm_args(**_kwargs):
    a = bb.tensor([[row * 16 + col + 1 for col in range(16)] for row in range(32)], dtype="f16")
    b = bb.tensor([[1.0 if col == row else 0.0 for col in range(32)] for row in range(16)], dtype="f16")
    c = bb.zeros((32, 32), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}


def aster_mfma_f16_gemm_args(**_kwargs):
    a = bb.tensor([[row * 16 + col + 1 for col in range(16)] for row in range(16)], dtype="f16")
    b = bb.tensor([[1.0 if col == row else 0.0 for col in range(16)] for row in range(16)], dtype="f16")
    c = bb.zeros((16, 16), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}


def aster_mfma_bf16_gemm_args(**_kwargs):
    a = bb.tensor([[row * 16 + col + 1 for col in range(16)] for row in range(16)], dtype="bf16")
    b = bb.tensor([[1.0 if col == row else 0.0 for col in range(16)] for row in range(16)], dtype="bf16")
    c = bb.zeros((16, 16), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}


def aster_mfma_fp8_gemm_args(**_kwargs):
    a = bb.pack_fp8([[1.0 for _ in range(32)] for _ in range(16)])
    b = bb.pack_fp8([[1.0 for _ in range(16)] for _ in range(32)])
    c = bb.zeros((16, 16), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}


def aster_mfma_bf8_gemm_args(**_kwargs):
    a = bb.pack_bf8([[1.0 for _ in range(32)] for _ in range(16)])
    b = bb.pack_bf8([[1.0 for _ in range(16)] for _ in range(32)])
    c = bb.zeros((16, 16), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}


def aster_mfma_fp8_bf8_gemm_args(**_kwargs):
    a = bb.pack_fp8([[1.0 for _ in range(32)] for _ in range(16)])
    b = bb.pack_bf8([[1.0 for _ in range(16)] for _ in range(32)])
    c = bb.zeros((16, 16), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}


def aster_mfma_bf8_fp8_gemm_args(**_kwargs):
    a = bb.pack_bf8([[1.0 for _ in range(32)] for _ in range(16)])
    b = bb.pack_fp8([[1.0 for _ in range(16)] for _ in range(32)])
    c = bb.zeros((16, 16), dtype="f32")
    return {"args": (a, b, c), "result_indices": ()}
