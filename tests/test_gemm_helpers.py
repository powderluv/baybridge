import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def warp_idx_kernel(out: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    warp = bb.Int32(bb.arch.make_warp_uniform(bb.arch.warp_idx()))
    out[tidx] = warp


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def domain_offset_copy_kernel(inp: bb.Tensor, out: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    shifted = bb.domain_offset((2,), inp)
    out[tidx] = shifted[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def local_tile_copy_kernel(inp: bb.Tensor, out: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    tile = bb.local_tile(inp, tiler=(2, 4, 2), coord=(1, 0, None), proj=(1, None, 1))
    row = tidx // 2
    col = tidx % 2
    out[tidx] = tile[row, col, 0]


@bb.kernel
def traced_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


def test_tile_and_size_helpers() -> None:
    assert bb.ceil_div(10, 3) == 4
    assert bb.ceil_div((10, 9, 1), (4, 4, 1)) == (3, 3, 1)

    layout = bb.make_layout((8, 16), stride=(32, 1))
    assert bb.cosize(layout) == 240
    assert bb.size_in_bytes("f32", layout) == 960
    assert bb.size_in_bytes(bb.Float32, (8, 16)) == 512

    swizzle = bb.make_swizzle(2, 3, 3)
    assert swizzle((1, 2)) == (1, 2)

    composed = bb.make_composed_layout(swizzle, 0, bb.make_layout((8, 64), stride=(64, 1)))
    tiled = bb.tile_to_shape(composed, (64, 64, 2), (0, 1, 2))
    assert isinstance(tiled, bb.ComposedLayout)
    assert tiled.shape == (64, 64, 2)

    grouped_layout = bb.group_modes(bb.make_layout((2, 3, 4), stride=(12, 4, 1)), 0, 2)
    assert grouped_layout.shape == (6, 4)
    assert grouped_layout.stride == (4, 1)

    grouped_tensor = bb.group_modes(
        bb.tensor(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
            ],
            dtype="i32",
        ),
        0,
        2,
    )
    assert grouped_tensor.shape == (6, 4)
    assert grouped_tensor[4, 2] == 19


def test_copy_aliases_and_type_exports() -> None:
    src = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    dst = bb.zeros((2, 2), dtype="f32")
    bb.basic_copy(src, dst)
    assert dst.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    dst2 = bb.zeros((2, 2), dtype="f32")
    bb.autovec_copy(src, dst2, vector_bytes=16)
    assert dst2.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    atom = bb.make_copy_atom(bb.nvgpu.CopyUniversalOp(), "f32")
    tiled_copy = bb.make_tiled_copy_tv(
        atom,
        bb.make_layout((1, 4), stride=(4, 1)),
        bb.make_layout((2, 1), stride=(1, 1)),
    )
    assert isinstance(atom, bb.CopyAtom)
    assert isinstance(tiled_copy, bb.TiledCopy)


def test_domain_offset_runtime_and_compiled(tmp_path: Path) -> None:
    inp = bb.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype="f32")
    out = bb.zeros((4,), dtype="f32")

    domain_offset_copy_kernel(inp, out).launch(grid=(1, 1, 1), block=(4, 1, 1))
    assert out.tolist() == [30.0, 40.0, 50.0, 60.0]

    artifact = bb.compile(domain_offset_copy_kernel, inp, out, cache_dir=tmp_path, backend="hipcc_exec")
    assert artifact.lowered_module is not None
    assert "&inp[" in artifact.lowered_module.text


def test_local_tile_runtime_identity_and_compiled(tmp_path: Path) -> None:
    inp = bb.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        ],
        dtype="f32",
    )
    tile = bb.local_tile(inp, tiler=(2, 4, 2), coord=(1, 0, None), proj=(1, None, 1))
    assert tile.shape == (2, 2, 4)
    assert tile[0, 0, 0] == 17.0
    assert tile[1, 1, 3] == 32.0

    coords = bb.local_tile(bb.make_identity_tensor((4, 8)), tiler=(2, 4, 2), coord=(1, 0, None), proj=(1, None, 1))
    assert coords.shape == (2, 2, 4)
    assert coords[1, 1, 3] == (3, 7)

    out = bb.zeros((4,), dtype="f32")
    local_tile_copy_kernel(inp, out).launch(grid=(1, 1, 1), block=(4, 1, 1))
    assert out.tolist() == [17.0, 18.0, 25.0, 26.0]

    artifact = bb.compile(local_tile_copy_kernel, inp, out, cache_dir=tmp_path, backend="hipcc_exec")
    assert artifact.lowered_module is not None
    assert "&inp[" in artifact.lowered_module.text


def test_tiled_mma_and_gemm_helpers(tmp_path: Path) -> None:
    tiled_mma = bb.make_tiled_mma(
        SimpleNamespace(tile=(16, 16, 4), dtype="f32", accumulator_dtype="f32", wave_size=64)
    )
    atom = bb.make_copy_atom(bb.nvgpu.CopyUniversalOp(), "f32")
    copy_a = bb.make_tiled_copy_A(atom, tiled_mma)
    copy_b = bb.make_tiled_copy_B(atom, tiled_mma)
    copy_c = bb.make_tiled_copy_C(atom, tiled_mma)
    assert tiled_mma.shape == (16, 16, 4)
    assert copy_a.tv_layout.tile_shape == (16, 4)
    assert copy_b.tv_layout.tile_shape == (4, 16)
    assert copy_c.tv_layout.tile_shape == (16, 16)

    lhs = bb.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="f32")
    rhs = bb.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype="f32")
    out = bb.zeros((2, 2), dtype="f32")
    bb.gemm(lhs, rhs, out)
    assert out.tolist() == [[58.0, 64.0], [139.0, 154.0]]

    a = bb.zeros((16, 4), dtype="f32")
    b = bb.zeros((4, 16), dtype="f32")
    c = bb.zeros((16, 16), dtype="f32")
    artifact = bb.compile(traced_gemm_kernel, a, b, c, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    assert "amdgpu.mfma" in artifact.lowered_module.text


def test_tiled_copy_and_mma_object_model_helpers() -> None:
    atom = bb.make_copy_atom(bb.nvgpu.CopyUniversalOp(), "f32")
    tiled_copy = bb.make_tiled_copy_tv(
        atom,
        bb.make_layout((1, 4), stride=(4, 1)),
        bb.make_layout((2, 1), stride=(1, 1)),
    )
    thr_copy = tiled_copy.get_thread_slice(3)
    assert tiled_copy.get_thread_slice(3) == thr_copy
    assert tiled_copy.partition_shape_S((64, 64)) == (2, 4)
    assert tiled_copy.partition_shape_D((64, 64)) == (2, 4)
    assert thr_copy.partition_shape_S((64, 64)) == (2, 4)
    assert thr_copy.partition_shape_D((64, 64)) == (2, 4)

    fragment = bb.zeros((2, 4), dtype="f32")
    assert thr_copy.retile(fragment) is fragment

    tiled_mma = bb.make_tiled_mma(
        SimpleNamespace(tile=(16, 16, 4), dtype="f32", accumulator_dtype="f32", wave_size=64)
    )
    thr_mma = tiled_mma.get_thread_slice(7)
    assert tiled_mma.get_thread_slice(7) == thr_mma
    assert tiled_mma.partition_shape_A((128, 64)) == (16, 4)
    assert tiled_mma.partition_shape_B((128, 64)) == (4, 16)
    assert tiled_mma.partition_shape_C((128, 64)) == (16, 16)
    assert thr_mma.partition_shape_A((128, 64)) == (16, 4)
    assert thr_mma.partition_shape_B((128, 64)) == (4, 16)
    assert thr_mma.partition_shape_C((128, 64)) == (16, 16)

    a = bb.zeros((16, 4), dtype="f32")
    b = bb.zeros((4, 16), dtype="f32")
    c = bb.zeros((16, 16), dtype="f32")
    frag_a = thr_mma.make_fragment_A(a)
    frag_b = thr_mma.make_fragment_B(b)
    frag_c = thr_mma.make_fragment_C(c)
    assert frag_a.shape == (16, 4)
    assert frag_b.shape == (4, 16)
    assert frag_c.shape == (16, 16)


def test_warp_idx_kernel_runs_direct_and_compiled(tmp_path: Path) -> None:
    expected = [0] * 64 + [1] * 64

    direct_out = bb.zeros((128,), dtype="i32")
    warp_idx_kernel(direct_out).launch(grid=(1, 1, 1), block=(128, 1, 1))
    assert direct_out.tolist() == expected

    compiled_out = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(warp_idx_kernel, compiled_out, cache_dir=tmp_path, backend="hipcc_exec")
    assert artifact.lowered_module is not None
    assert "threadIdx.x" in artifact.lowered_module.text


def test_warp_idx_kernel_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    out = bb.zeros((128,), dtype="i32")
    artifact = bb.compile(
        warp_idx_kernel,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(out)
    assert out.tolist() == [0] * 64 + [1] * 64


def test_domain_offset_kernel_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    inp = bb.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    artifact = bb.compile(
        domain_offset_copy_kernel,
        inp,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(inp, out)
    assert out.tolist() == [30.0, 40.0, 50.0, 60.0]


def test_local_tile_kernel_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    inp = bb.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        ],
        dtype="f32",
    )
    out = bb.zeros((4,), dtype="f32")
    artifact = bb.compile(
        local_tile_copy_kernel,
        inp,
        out,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(inp, out)
    assert out.tolist() == [17.0, 18.0, 25.0, 26.0]
