from pathlib import Path

import baybridge as bb


def test_make_ordered_layout_and_select() -> None:
    layout = bb.make_ordered_layout((2, 3, 4), order=(2, 0, 1))

    assert layout.stride == (4, 8, 1)

    selected_shape = bb.select((2, 3, 4), mode=[1, 2])
    selected_layout = bb.select(layout, mode=[2, 0])

    assert selected_shape == (3, 4)
    assert selected_layout.shape == (4, 2)
    assert selected_layout.stride == (1, 4)


def test_make_layout_tv_returns_tiler_and_thread_value_layout() -> None:
    thr_layout = bb.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = bb.make_ordered_layout((4, 4), order=(1, 0))

    tiler, tv_layout = bb.make_layout_tv(thr_layout, val_layout)

    assert tiler == (16, 128)
    assert tv_layout.shape == (128, 16)
    assert bb.size(tv_layout, mode=[0]) == 128
    assert bb.size(tv_layout, mode=[1]) == 16


def test_zipped_divide_runtime_views_and_identity_tiles() -> None:
    tensor = bb.tensor([[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]], dtype="f32")
    tiled = bb.zipped_divide(tensor, (1, 2))

    assert tiled.shape == ((1, 2), (2, 3))
    assert bb.size(tiled, mode=[1]) == 6
    assert bb.product_each(tiled.shape[1]) == (2, 3, 1)

    block = tiled[((None, None), 1)]
    assert block.shape == (1, 2)
    assert block.tolist() == [[3, 4]]

    coords = bb.zipped_divide(bb.make_identity_tensor(tensor.shape), (1, 2))
    coord_block = coords[((None, None), 4)]
    assert coord_block[0, 0] == (1, 2)
    assert coord_block[0, 1] == (1, 3)


@bb.kernel(launch=bb.LaunchConfig(grid=(2, 1, 1), block=(4, 1, 1)))
def tiled_load_kernel(
    src: bb.TensorSpec(shape=(1, 8), dtype="f32"),
    dst: bb.TensorSpec(shape=(1, 8), dtype="f32"),
):
    tiled = bb.zipped_divide(src, (1, 4))
    block = tiled[((None, None), bb.block_idx("x"))]
    tidx, _, _ = bb.arch.thread_idx()
    value = block[0, tidx]
    dst[0, bb.block_idx("x") * 4 + tidx] = value


def test_zipped_divide_tracing_emits_partitioned_tile_access(tmp_path: Path) -> None:
    artifact = bb.compile(tiled_load_kernel, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]

    assert ops[:10] == [
        "block_idx",
        "constant",
        "floordiv",
        "constant",
        "mod",
        "constant",
        "mul",
        "constant",
        "mul",
        "partition",
    ]
    assert ops[10:14] == ["thread_idx", "thread_idx", "thread_idx", "constant"]
    assert ops[14:] == ["load", "block_idx", "constant", "mul", "add", "constant", "store"]
    assert artifact.ir.operations[9].attrs["tile"] == [1, 4]
    assert "memref.subview" in artifact.lowered_module.text
