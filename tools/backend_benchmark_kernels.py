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


@bb.kernel
def dense_copy_f32_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.copy(src, dst)


@bb.kernel
def dense_add_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    dst.store(src.load() + other.load())


@bb.kernel(launch=bb.LaunchConfig(grid=(FLYDSL_GRID, 1, 1), block=(FLYDSL_BLOCK, 1, 1)))
def indexed_add_f32_kernel(src: bb.Tensor, other: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx] + other[idx]


@bb.kernel
def hipkittens_bf16_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel
def hipkittens_f16_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


def _vector_f32(n: int, *, scale: float = 1.0, offset: float = 0.0) -> list[float]:
    return [scale * float(index % 251) + offset for index in range(n)]


def _vector_i32(n: int, *, scale: int = 1, offset: int = 0) -> list[int]:
    return [scale * int(index % 251) + offset for index in range(n)]


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


def indexed_add_f32_args(*, backend_name=None, **_kwargs):
    src, other, dst = _make_f32_vector_args(POINTWISE_N, backend_name=backend_name or "")
    return {"args": (src, other, dst), "result_indices": ()}


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
