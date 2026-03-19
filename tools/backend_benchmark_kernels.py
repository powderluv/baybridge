from __future__ import annotations

import baybridge as bb

try:
    import torch
except Exception:
    torch = None

POINTWISE_N = 65536
ASTER_POINTWISE_N = 4096
FLYDSL_BLOCK = 256
FLYDSL_GRID = POINTWISE_N // FLYDSL_BLOCK
FLYDSL_MICRO_N = 4096
FLYDSL_MICRO_ROWS = 64
FLYDSL_MICRO_COLS = 64
FLYDSL_SHARED_N = 256


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


@bb.kernel(launch=bb.LaunchConfig(grid=(FLYDSL_GRID, 1, 1), block=(FLYDSL_BLOCK, 1, 1)))
def indexed_add_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + other[idx]


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


def dense_copy_f32_args(*, backend_name=None, **_kwargs):
    src_values = _vector_f32(POINTWISE_N, scale=0.5, offset=1.0)
    src = _maybe_torch_tensor(src_values, shape=(POINTWISE_N,), dtype="f32", backend_name=backend_name or "")
    dst = _maybe_torch_tensor([0.0 for _ in range(POINTWISE_N)], shape=(POINTWISE_N,), dtype="f32", backend_name=backend_name or "")
    if src is None or dst is None:
        src = bb.tensor(src_values, dtype="f32")
        dst = bb.zeros((POINTWISE_N,), dtype="f32")
    return {"args": (src, dst), "result_indices": ()}


def dense_add_f32_args(*, backend_name=None, **_kwargs):
    src, other, dst = _make_f32_vector_args(POINTWISE_N, backend_name=backend_name or "")
    return {"args": (src, other, dst), "result_indices": ()}


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
