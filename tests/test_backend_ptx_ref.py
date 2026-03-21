from __future__ import annotations

from pathlib import Path

import pytest

import baybridge as bb
from baybridge.cuda_driver import CudaDriver, CudaDriverError


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(64, 1, 1)))
def ptx_indexed_copy_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    dst[idx] = src[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(64, 1, 1)))
def ptx_indexed_add_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] + b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(64, 1, 1)))
def ptx_indexed_mul_i32_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    idx = bidx * bdim + tidx
    c[idx] = a[idx] * b[idx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 1, 1)))
def ptx_unsupported_math_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = bb.acos(src[tidx])


def _nvidia_target() -> bb.NvidiaTarget:
    return bb.NvidiaTarget(sm="sm_80", ptx_version="8.0")


def test_ptx_ref_lowers_indexed_copy_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 256, dtype="f32")
    dst = bb.zeros((256,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    assert artifact.backend_name == "ptx_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "ptx"
    assert artifact.lowered_path is not None
    assert artifact.lowered_path.suffix == ".ptx"
    text = artifact.lowered_module.text
    assert ".version 8.0" in text
    assert ".target sm_80" in text
    assert ".visible .entry ptx_indexed_copy_kernel(" in text
    assert "ld.global.f32 %f1, [%rd4];" in text
    assert "st.global.f32 [%rd5], %f1;" in text


def test_ptx_ref_lowers_indexed_add_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1.0] * 256, dtype="f32")
    b = bb.tensor([2.0] * 256, dtype="f32")
    c = bb.zeros((256,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_add_kernel(" in text
    assert "ld.global.f32 %f1, [%rd5];" in text
    assert "ld.global.f32 %f2, [%rd6];" in text
    assert "add.rn.f32 %f3, %f1, %f2;" in text
    assert "st.global.f32 [%rd7], %f3;" in text


def test_ptx_ref_lowers_indexed_mul_i32_kernel(tmp_path: Path) -> None:
    a = bb.tensor([1] * 128, dtype="i32")
    b = bb.tensor([2] * 128, dtype="i32")
    c = bb.zeros((128,), dtype="i32")

    artifact = bb.compile(
        ptx_indexed_mul_i32_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    text = artifact.lowered_module.text
    assert ".visible .entry ptx_indexed_mul_i32_kernel(" in text
    assert "ld.global.s32 %r5, [%rd5];" in text
    assert "ld.global.s32 %r1, [%rd6];" in text
    assert "mul.lo.s32 %r5, %r5, %r1;" in text
    assert "st.global.s32 [%rd7], %r5;" in text


def test_compile_auto_prefers_ptx_ref_for_nvidia_target(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 256, dtype="f32")
    dst = bb.zeros((256,), dtype="f32")

    artifact = bb.compile(
        ptx_indexed_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
    )

    assert artifact.backend_name == "ptx_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "ptx"


def test_ptx_ref_rejects_unsupported_math_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 8, dtype="f32")
    dst = bb.zeros((8,), dtype="f32")

    with pytest.raises(bb.CompilationError, match="ptx_ref currently supports only canonical indexed rank-1 dense copy"):
        bb.compile(
            ptx_unsupported_math_kernel,
            src,
            dst,
            cache_dir=tmp_path,
            target=_nvidia_target(),
            backend="ptx_ref",
        )


def test_ptx_ref_generated_module_loads_in_cuda_driver_if_available(tmp_path: Path) -> None:
    src = bb.tensor([1.0] * 256, dtype="f32")
    dst = bb.zeros((256,), dtype="f32")
    artifact = bb.compile(
        ptx_indexed_copy_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        target=_nvidia_target(),
        backend="ptx_ref",
    )

    try:
        driver = CudaDriver()
    except (bb.BackendNotImplementedError, CudaDriverError) as exc:
        pytest.skip(f"CUDA driver not usable on this host: {exc}")

    module = driver.load_module_from_ptx(artifact.lowered_module.text)
    try:
        function = driver.function(module, artifact.lowered_module.entry_point)
        assert int(function.value or 0) != 0
    finally:
        driver.unload_module(module)
        driver.release_primary_context(0)
