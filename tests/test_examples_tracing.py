from pathlib import Path

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def indexed_add_kernel(
    g_a: bb.Tensor,
    g_b: bb.Tensor,
    g_c: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    _, n = g_a.shape
    col = tidx % n
    row = tidx // n
    g_c[row, col] = g_a[row, col] + g_b[row, col]


@bb.jit
def indexed_add_wrapper(
    m_a: bb.Tensor,
    m_b: bb.Tensor,
    m_c: bb.Tensor,
):
    indexed_add_kernel(m_a, m_b, m_c).launch(grid=(1, 1, 1), block=(4, 1, 1))


@bb.kernel
def vectorized_add_kernel(
    g_a: bb.Tensor,
    g_b: bb.Tensor,
    g_c: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    _, n = g_a.shape[1]
    ni = thread_idx % n
    mi = thread_idx // n
    g_c[(None, (mi, ni))] = g_a[(None, (mi, ni))].load() + g_b[(None, (mi, ni))].load()


@bb.jit
def vectorized_add_wrapper(
    m_a: bb.Tensor,
    m_b: bb.Tensor,
    m_c: bb.Tensor,
):
    g_a = bb.zipped_divide(m_a, (1, 4))
    g_b = bb.zipped_divide(m_b, (1, 4))
    g_c = bb.zipped_divide(m_c, (1, 4))
    vectorized_add_kernel(g_a, g_b, g_c).launch(grid=(1, 1, 1), block=(256, 1, 1))


@bb.kernel
def tiled_add_kernel(g_a, g_b, g_c, tv_layout):
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


@bb.jit
def tiled_add_wrapper(
    m_a: bb.Tensor,
    m_b: bb.Tensor,
    m_c: bb.Tensor,
):
    thr_layout = bb.make_ordered_layout((1, 4), order=(1, 0))
    val_layout = bb.make_ordered_layout((2, 1), order=(1, 0))
    tiler_mn, tv_layout = bb.make_layout_tv(thr_layout, val_layout)
    g_a = bb.zipped_divide(m_a, tiler_mn)
    g_b = bb.zipped_divide(m_b, tiler_mn)
    g_c = bb.zipped_divide(m_c, tiler_mn)
    remap_block = bb.make_ordered_layout(bb.select(g_a.shape[1], mode=[1, 0]), order=(1, 0))
    g_a = bb.composition(g_a, (None, remap_block))
    g_b = bb.composition(g_b, (None, remap_block))
    g_c = bb.composition(g_c, (None, remap_block))
    tiled_add_kernel(g_a, g_b, g_c, tv_layout).launch(grid=(4, 1, 1), block=(4, 1, 1))


def test_basic_indexed_kernel_traces_from_sample_args(tmp_path: Path) -> None:
    a = bb.tensor([[1, 2], [3, 4]], dtype="f32")
    b = bb.tensor([[10, 20], [30, 40]], dtype="f32")
    c = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(indexed_add_kernel, a, b, c, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    assert [operation.op for operation in artifact.ir.operations] == [
        "thread_idx",
        "thread_idx",
        "thread_idx",
        "constant",
        "mod",
        "constant",
        "floordiv",
        "load",
        "load",
        "add",
        "store",
    ]
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "memref.load %g_a[%floordiv_7, %mod_5]" in text
    assert "memref.load %g_b[%floordiv_7, %mod_5]" in text
    assert "memref.store %add_10, %g_c[%floordiv_7, %mod_5]" in text


def test_single_launch_wrapper_traces_to_launched_kernel(tmp_path: Path) -> None:
    a = bb.tensor([[1, 2], [3, 4]], dtype="f32")
    b = bb.tensor([[10, 20], [30, 40]], dtype="f32")
    c = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(indexed_add_wrapper, a, b, c, cache_dir=tmp_path, backend="portable")

    assert artifact.ir is not None
    assert artifact.ir.name == "indexed_add_kernel"
    assert artifact.ir.launch.grid == (1, 1, 1)
    assert artifact.ir.launch.block == (4, 1, 1)
    assert artifact.ir.metadata["compiled_from"] == "launch_wrapper"
    assert artifact.ir.metadata["wrapped_by"] == "indexed_add_wrapper"


def test_vectorized_launch_wrapper_traces_tile_copy_and_tensor_add(tmp_path: Path) -> None:
    a = bb.tensor([[float(index) for index in range(1024)]], dtype="f32")
    b = bb.tensor([[float(index + 10) for index in range(1024)]], dtype="f32")
    c = bb.zeros((1, 1024), dtype="f32")

    artifact = bb.compile(vectorized_add_wrapper, a, b, c, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    assert artifact.ir.name == "vectorized_add_kernel"
    assert artifact.ir.metadata["compiled_from"] == "launch_wrapper"
    ops = [operation.op for operation in artifact.ir.operations]
    assert "partition" in ops
    assert "copy" in ops
    assert "tensor_add" in ops
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "memref.subview %m_a" in text
    assert '"baybridge.tensor_add"' in text


def test_tiled_launch_wrapper_traces_thread_fragment_ops(tmp_path: Path) -> None:
    a = bb.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0],
        ],
        dtype="f32",
    )
    b = bb.tensor(
        [
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
            [800.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0],
            [8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2000.0, 1000.0],
        ],
        dtype="f32",
    )
    c = bb.zeros((4, 8), dtype="f32")

    artifact = bb.compile(tiled_add_wrapper, a, b, c, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    assert artifact.ir.name == "tiled_add_kernel"
    ops = [operation.op for operation in artifact.ir.operations]
    assert "thread_fragment_load" in ops
    assert "thread_fragment_store" in ops
    assert "tensor_add" in ops
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '"baybridge.thread_fragment_load"' in text
    assert '"baybridge.thread_fragment_store"' in text


def test_size_supports_tensors_layouts_and_modes() -> None:
    tensor = bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="f32")
    layout = bb.make_layout((2, 3))

    assert bb.size(tensor) == 6
    assert bb.size(tensor, mode=[1]) == 3
    assert bb.size(layout) == 6
    assert bb.size(layout, mode=[0]) == 2
