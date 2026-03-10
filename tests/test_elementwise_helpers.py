from pathlib import Path

import baybridge as bb


def test_recast_layout_scales_contiguous_axis() -> None:
    byte_layout = bb.make_ordered_layout((16, 16), order=(1, 0))
    element_layout = bb.recast_layout(32, 8, byte_layout)

    assert element_layout.shape == (16, 4)
    assert element_layout.stride == (4, 1)


def test_element_type_is_string_compatible_and_exposes_width() -> None:
    runtime_tensor = bb.tensor([1.0, 2.0], dtype="f16")
    assert runtime_tensor.element_type == "f16"
    assert runtime_tensor.element_type.width == 16

    traced = bb.TensorSpec(shape=(4,), dtype="bf16")
    assert traced.dtype == "bf16"


def test_composition_remaps_tiled_rest_axes_runtime() -> None:
    tensor = bb.tensor([[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]], dtype="f32")
    tiled = bb.zipped_divide(tensor, (1, 2))
    remap = bb.make_ordered_layout(bb.select(tiled.shape[1], mode=[1, 0]), order=(1, 0))

    remapped = bb.composition(tiled, (None, remap))

    assert remapped.shape == ((1, 2), (3, 2))
    assert remapped[((None, None), 1)].tolist() == [[10, 20]]


def test_elem_less_where_and_print_tensor_runtime(capsys) -> None:
    assert bb.elem_less((1, 2), (2, 3)) is True
    assert bb.elem_less(4, 3) is False
    assert bb.where(True, 7, 9) == 7
    assert bb.where(False, 7, 9) == 9

    tmp = bb.tensor([1, -2, 3, -4], dtype="f32")
    masked = bb.where(tmp > 0, tmp, bb.full_like(tmp, 0.0))
    assert masked.tolist() == [1, 0.0, 3, 0.0]

    tensor = bb.tensor([[1, 2], [3, 4]], dtype="f32")
    bb.print_tensor(tensor)

    assert capsys.readouterr().out.splitlines() == ["[[1, 2], [3, 4]]"]


@bb.kernel
def tv_elementwise_add_kernel(g_a, g_b, g_c, tv_layout):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blk_a = g_a[blk_coord]
    blk_b = g_b[blk_coord]
    blk_c = g_c[blk_coord]

    tidfrg_a = bb.composition(blk_a, tv_layout)
    tidfrg_b = bb.composition(blk_b, tv_layout)
    tidfrg_c = bb.composition(blk_c, tv_layout)

    thr_coord = (tidx, bb.repeat_like(None, tidfrg_a[1]))
    thr_a = tidfrg_a[thr_coord]
    thr_b = tidfrg_b[thr_coord]
    thr_c = tidfrg_c[thr_coord]

    thr_c.store(thr_a.load() + thr_b.load())


def test_thread_value_composition_runtime_executes_vectorized_add() -> None:
    a = bb.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="f32")
    b = bb.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype="f32")
    c = bb.zeros((1, 8), dtype="f32")

    thr_layout = bb.make_ordered_layout((1, 4), order=(1, 0))
    val_layout = bb.make_ordered_layout((1, 1), order=(1, 0))
    tiler, tv_layout = bb.make_layout_tv(thr_layout, val_layout)

    g_a = bb.zipped_divide(a, tiler)
    g_b = bb.zipped_divide(b, tiler)
    g_c = bb.zipped_divide(c, tiler)

    tv_elementwise_add_kernel(g_a, g_b, g_c, tv_layout).launch(
        grid=(bb.size(g_c, mode=[1]), 1, 1),
        block=(bb.size(tv_layout, mode=[0]), 1, 1),
    )

    assert c.tolist() == [[11, 22, 33, 44, 55, 66, 77, 88]]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def helper_kernel(
    src: bb.TensorSpec(shape=(4,), dtype="f32"),
    dst: bb.TensorSpec(shape=(4,), dtype="f32"),
):
    reg = bb.make_rmem_tensor((4,), "f32")
    frag = bb.make_fragment_like(src)
    tidx, _, _ = bb.arch.thread_idx()
    value = bb.where(bb.elem_less(tidx, bb.size(src)), src[tidx], 0.0)
    reg[tidx] = value
    frag[tidx] = reg[tidx]
    dst[tidx] = frag[tidx]


def test_helper_kernel_traces_select_and_register_tensors(tmp_path: Path) -> None:
    artifact = bb.compile(helper_kernel, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]
    assert ops[:2] == ["make_tensor", "make_tensor"]
    assert "cmp_lt" in ops
    assert "select" in ops
    assert ops.count("store") == 3
    assert 'arith.select' in artifact.lowered_module.text
