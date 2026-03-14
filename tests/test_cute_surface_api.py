import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def top_level_math_kernel(
    src: bb.Tensor,
    dst_acos: bb.Tensor,
    dst_asin: bb.Tensor,
    dst_atan: bb.Tensor,
    dst_sqrt: bb.Tensor,
    dst_rsqrt: bb.Tensor,
):
    values = src.load()
    dst_acos.store(bb.acos(values))
    dst_asin.store(bb.asin(values))
    dst_atan.store(bb.atan(values))
    dst_sqrt.store(bb.sqrt(values))
    dst_rsqrt.store(bb.rsqrt(values))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def bool_reduce_kernel(src: bb.Tensor, out: bb.Tensor):
    bb.store(bb.Int32(bb.any_(src)), out, 0)
    bb.store(bb.Int32(bb.all_(src)), out, 1)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def prefetch_kernel(src: bb.Tensor, dst: bb.Tensor):
    bb.prefetch(src)
    dst[0] = src[0]


def test_top_level_math_wrappers_runtime_and_compile(tmp_path: Path) -> None:
    src = bb.tensor([0.25, 0.5, 0.75], dtype="f32")
    dst_acos = bb.zeros((3,), dtype="f32")
    dst_asin = bb.zeros((3,), dtype="f32")
    dst_atan = bb.zeros((3,), dtype="f32")
    dst_sqrt = bb.zeros((3,), dtype="f32")
    dst_rsqrt = bb.zeros((3,), dtype="f32")

    top_level_math_kernel(src, dst_acos, dst_asin, dst_atan, dst_sqrt, dst_rsqrt).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )

    assert dst_acos.tolist() == pytest.approx([math.acos(0.25), math.acos(0.5), math.acos(0.75)], rel=1e-6, abs=1e-6)
    assert dst_asin.tolist() == pytest.approx([math.asin(0.25), math.asin(0.5), math.asin(0.75)], rel=1e-6, abs=1e-6)
    assert dst_atan.tolist() == pytest.approx([math.atan(0.25), math.atan(0.5), math.atan(0.75)], rel=1e-6, abs=1e-6)
    assert dst_sqrt.tolist() == pytest.approx([math.sqrt(0.25), math.sqrt(0.5), math.sqrt(0.75)], rel=1e-6, abs=1e-6)
    assert dst_rsqrt.tolist() == pytest.approx(
        [1.0 / math.sqrt(0.25), 1.0 / math.sqrt(0.5), 1.0 / math.sqrt(0.75)],
        rel=1e-6,
        abs=1e-6,
    )

    artifact = bb.compile(
        top_level_math_kernel,
        src,
        dst_acos,
        dst_asin,
        dst_atan,
        dst_sqrt,
        dst_rsqrt,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )
    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]
    assert "math_acos" in ops
    assert "math_asin" in ops
    assert "math_atan" in ops
    assert "math_sqrt" in ops
    assert "math_rsqrt" in ops


def test_top_level_math_wrappers_run_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    src = bb.tensor([0.25, 0.5, 0.75], dtype="f32")
    dst_acos = bb.zeros((3,), dtype="f32")
    dst_asin = bb.zeros((3,), dtype="f32")
    dst_atan = bb.zeros((3,), dtype="f32")
    dst_sqrt = bb.zeros((3,), dtype="f32")
    dst_rsqrt = bb.zeros((3,), dtype="f32")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")

    artifact = bb.compile(
        top_level_math_kernel,
        src,
        dst_acos,
        dst_asin,
        dst_atan,
        dst_sqrt,
        dst_rsqrt,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(src, dst_acos, dst_asin, dst_atan, dst_sqrt, dst_rsqrt)

    assert dst_acos.tolist() == pytest.approx([math.acos(0.25), math.acos(0.5), math.acos(0.75)], rel=1e-5, abs=1e-5)
    assert dst_asin.tolist() == pytest.approx([math.asin(0.25), math.asin(0.5), math.asin(0.75)], rel=1e-5, abs=1e-5)
    assert dst_atan.tolist() == pytest.approx([math.atan(0.25), math.atan(0.5), math.atan(0.75)], rel=1e-5, abs=1e-5)
    assert dst_sqrt.tolist() == pytest.approx([math.sqrt(0.25), math.sqrt(0.5), math.sqrt(0.75)], rel=1e-5, abs=1e-5)
    assert dst_rsqrt.tolist() == pytest.approx(
        [1.0 / math.sqrt(0.25), 1.0 / math.sqrt(0.5), 1.0 / math.sqrt(0.75)],
        rel=1e-5,
        abs=1e-5,
    )


def test_boolean_reduce_and_python_tuple_helpers(tmp_path: Path) -> None:
    src = bb.tensor([True, False, True], dtype="i1")
    out = bb.zeros((2,), dtype="i32")
    bool_reduce_kernel(src, out).launch(grid=(1, 1, 1), block=(1, 1, 1))
    assert out.tolist() == [1, 0]

    artifact = bb.compile(bool_reduce_kernel, src, out, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]
    assert "reduce_max" in ops
    assert "reduce_min" in ops

    assert bb.repeat_as_tuple("x", 3) == ("x", "x", "x")
    assert bb.repeat(7, 1) == 7
    assert bb.repeat(7, 3) == (7, 7, 7)
    assert bb.repeat_like("v", ((1, 2), (3, 4))) == (("v", "v"), ("v", "v"))
    assert bb.tuple_cat((1, 2), 3, (4, 5)) == (1, 2, 3, 4, 5)
    assert bb.transform_apply((1, 2), (10, 20), f=lambda lhs, rhs: lhs + rhs, g=lambda *items: tuple(items)) == (11, 22)


def test_predicated_copy_and_prefetch_surface(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")
    bb.basic_copy_if(False, src, dst)
    assert dst.tolist() == [0.0, 0.0, 0.0, 0.0]

    bb.basic_copy_if(bb.Boolean(True), src, dst)
    assert dst.tolist() == [1.0, 2.0, 3.0, 4.0]

    with pytest.raises(bb.UnsupportedOperationError):
        @bb.kernel
        def traced_copy_if_kernel(inp: bb.Tensor, out: bb.Tensor):
            bb.basic_copy_if(bb.arch.elect_one(), inp, out)

        bb.compile(traced_copy_if_kernel, src, dst, cache_dir=tmp_path, backend="portable")

    prefetch_dst = bb.zeros((1,), dtype="f32")
    prefetch_kernel(src, prefetch_dst).launch(grid=(1, 1, 1), block=(1, 1, 1))
    assert prefetch_dst.tolist() == [1.0]
    artifact = bb.compile(prefetch_kernel, src, prefetch_dst, cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None


def test_tiled_atom_state_and_arch_barrier_helpers(tmp_path: Path) -> None:
    atom = bb.make_copy_atom(bb.nvgpu.CopyUniversalOp(), "f32")
    assert atom.get() == atom.op
    atom.set("cache_mode", "always")
    assert atom.get("cache_mode") == "always"
    updated_atom = atom.with_(cache_mode="global")
    assert updated_atom.get("cache_mode") == "global"
    assert atom.type == "f32"

    tiled_copy = bb.make_tiled_copy_tv(
        atom,
        bb.make_layout((1, 4), stride=(4, 1)),
        bb.make_layout((2, 1), stride=(1, 1)),
    )
    assert tiled_copy.layout_tv.tile_shape == (2, 4)
    assert tiled_copy.thr_layout.shape == (1, 4)
    assert tiled_copy.val_layout.shape == (2, 1)
    assert tiled_copy.num_threads == 4
    assert tiled_copy.num_values == 2
    assert tiled_copy.partition_shape_S() == (2, 4)
    assert tiled_copy.partition_shape_D() == (2, 4)

    sample = bb.zeros((2, 4), dtype="f32")
    assert tiled_copy.partition_S(sample, 0).shape == (2,)
    assert tiled_copy.partition_D(sample, 0).shape == (2,)
    slice_copy = tiled_copy.get_thread_slice(0)
    assert slice_copy.retile_S(sample) is sample
    assert slice_copy.retile_D(sample) is sample
    assert slice_copy.partition_fragment_S(sample) is sample
    assert slice_copy.partition_fragment_D(sample) is sample

    tiled_mma = bb.make_tiled_mma(SimpleNamespace(tile=(16, 16, 4), dtype="f32", accumulator_dtype="f32", wave_size=64))
    tiled_mma.set("epilogue", "bias")
    assert tiled_mma.get("epilogue") == "bias"
    updated_mma = tiled_mma.with_(epilogue="relu")
    assert updated_mma.get("epilogue") == "relu"
    assert tiled_mma.shape_mnk == (16, 16, 4)
    assert tiled_mma.type == "f32"
    frag_acc = tiled_mma.make_fragment_ACC((16, 16))
    assert frag_acc.shape == (16, 16)

    mbarrier = bb.MbarrierArray(1)[0]
    bb.arch.barrier()
    bb.arch.barrier_arrive()
    bb.arch.mbarrier_init(mbarrier, 64)
    bb.arch.mbarrier_init_fence(mbarrier, 64)
    bb.arch.mbarrier_arrive(mbarrier)
    bb.arch.mbarrier_expect_tx(mbarrier, 16)
    bb.arch.mbarrier_arrive_and_expect_tx(mbarrier, 16)
    bb.arch.mbarrier_wait(mbarrier)
    assert bb.arch.mbarrier_try_wait(mbarrier) is True
    assert bb.arch.mbarrier_test_wait(mbarrier) is True
    assert bb.arch.mbarrier_conditional_try_wait(mbarrier, True) is True
    bb.arch.fence_tma_store()
    assert mbarrier.data_ptr().memspace == bb.AddressSpace.SHARED
    assert callable(bb.lane_idx)
