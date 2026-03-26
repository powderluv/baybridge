from pathlib import Path

import baybridge as bb


@bb.kernel
def hello_world_kernel():
    tidx, _, _ = bb.arch.thread_idx()
    if tidx == 0:
        bb.printf("Hello world")


@bb.jit
def hello_world():
    bb.printf("hello world")
    hello_world_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


@bb.kernel
def naive_elementwise_add_kernel(
    g_a: bb.Tensor,
    g_b: bb.Tensor,
    g_c: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()
    linear_idx = bidx * bdim + tidx
    m, n = g_a.shape
    if linear_idx < m * n:
        col = linear_idx % n
        row = linear_idx // n
        g_c[row, col] = g_a[row, col] + g_b[row, col]


@bb.jit
def naive_elementwise_add(
    m_a: bb.Tensor,
    m_b: bb.Tensor,
    m_c: bb.Tensor,
):
    threads_per_block = 4
    m, n = m_a.shape
    naive_elementwise_add_kernel(m_a, m_b, m_c).launch(
        grid=((m * n + threads_per_block - 1) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


@bb.kernel
def vectorized_elementwise_add_kernel(
    g_a: bb.Tensor,
    g_b: bb.Tensor,
    g_c: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()
    bdim, _, _ = bb.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = g_a.shape[1]
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = g_a[(None, (mi, ni))].load()
    b_val = g_b[(None, (mi, ni))].load()
    g_c[(None, (mi, ni))] = a_val + b_val


@bb.jit
def vectorized_elementwise_add(
    m_a: bb.Tensor,
    m_b: bb.Tensor,
    m_c: bb.Tensor,
):
    threads_per_block = 256

    g_a = bb.zipped_divide(m_a, (1, 4))
    g_b = bb.zipped_divide(m_b, (1, 4))
    g_c = bb.zipped_divide(m_c, (1, 4))

    vectorized_elementwise_add_kernel(g_a, g_b, g_c).launch(
        grid=(bb.size(g_c, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


@bb.kernel
def tiled_elementwise_add_kernel(g_a, g_b, g_c, tv_layout):
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
def tiled_elementwise_add(
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

    tiled_elementwise_add_kernel(g_a, g_b, g_c, tv_layout).launch(
        grid=(bb.size(g_c, mode=[1]), 1, 1),
        block=(bb.size(tv_layout, mode=[0]), 1, 1),
    )


@bb.kernel
def tensor_abs_kernel(src: bb.Tensor, dst: bb.Tensor):
    dst.store(abs(src.load()))


@bb.jit
def tensor_abs(src: bb.Tensor, dst: bb.Tensor):
    tensor_abs_kernel(src, dst).launch(grid=(1, 1, 1), block=(1, 1, 1))


@bb.kernel
def tensor_rounding_kernel(
    src: bb.Tensor,
    dst_round: bb.Tensor,
    dst_floor: bb.Tensor,
    dst_ceil: bb.Tensor,
    dst_trunc: bb.Tensor,
):
    values = src.load()
    dst_round.store(bb.round(values))
    dst_floor.store(bb.floor(values))
    dst_ceil.store(bb.ceil(values))
    dst_trunc.store(bb.trunc(values))


@bb.jit
def tensor_rounding(
    src: bb.Tensor,
    dst_round: bb.Tensor,
    dst_floor: bb.Tensor,
    dst_ceil: bb.Tensor,
    dst_trunc: bb.Tensor,
):
    tensor_rounding_kernel(src, dst_round, dst_floor, dst_ceil, dst_trunc).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


@bb.kernel
def tensor_extrema_kernel(lhs: bb.Tensor, rhs: bb.Tensor, dst_max: bb.Tensor, dst_min: bb.Tensor):
    lhs_values = lhs.load()
    rhs_values = rhs.load()
    dst_max.store(bb.maximum(lhs_values, rhs_values))
    dst_min.store(bb.minimum(lhs_values, rhs_values))


@bb.jit
def tensor_extrema(lhs: bb.Tensor, rhs: bb.Tensor, dst_max: bb.Tensor, dst_min: bb.Tensor):
    tensor_extrema_kernel(lhs, rhs, dst_max, dst_min).launch(grid=(1, 1, 1), block=(1, 1, 1))


def test_hello_world_example_runs_via_compile_fallback(tmp_path: Path, capsys) -> None:
    artifact = bb.compile(hello_world, cache_dir=tmp_path)
    assert artifact.ir is None
    assert artifact.lowered_module is None
    assert artifact.artifact_path.exists()

    artifact()

    assert capsys.readouterr().out.splitlines() == ["hello world", "Hello world"]


def test_naive_elementwise_add_example_runs_direct_and_compiled(tmp_path: Path) -> None:
    a = bb.tensor([[1, 2, 3], [4, 5, 6]], dtype="f32")
    b = bb.tensor([[10, 20, 30], [40, 50, 60]], dtype="f32")

    direct_out = bb.zeros((2, 3), dtype="f32")
    naive_elementwise_add(a, b, direct_out)
    assert direct_out.tolist() == [[11, 22, 33], [44, 55, 66]]

    compiled = bb.compile(naive_elementwise_add, a, b, direct_out, cache_dir=tmp_path)
    assert compiled.ir is None

    compiled_out = bb.zeros((2, 3), dtype="f32")
    compiled(a, b, compiled_out)
    assert compiled_out.tolist() == [[11, 22, 33], [44, 55, 66]]


def test_vectorized_elementwise_add_example_runs_direct_and_compiled(tmp_path: Path) -> None:
    row_count = 1
    col_count = 1024
    a = bb.tensor([[float(index) for index in range(col_count)] for _ in range(row_count)], dtype="f32")
    b = bb.tensor([[float(index + 1000) for index in range(col_count)] for _ in range(row_count)], dtype="f32")
    expected = [
        [lhs + rhs for lhs, rhs in zip(a_row, b_row)]
        for a_row, b_row in zip(a.tolist(), b.tolist())
    ]

    direct_out = bb.zeros((row_count, col_count), dtype="f32")
    vectorized_elementwise_add(a, b, direct_out)
    assert direct_out.tolist() == expected

    compiled_out = bb.zeros((row_count, col_count), dtype="f32")
    artifact = bb.compile(vectorized_elementwise_add, a, b, compiled_out, cache_dir=tmp_path)

    assert artifact.ir is not None
    assert artifact.ir.name == "vectorized_elementwise_add_kernel"
    assert artifact.lowered_module is not None
    assert artifact.artifact_path.exists()

    artifact(a, b, compiled_out)
    assert compiled_out.tolist() == expected


def test_tiled_elementwise_add_example_runs_direct_and_compiled(tmp_path: Path) -> None:
    a = bb.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [10, 20, 30, 40, 50, 60, 70, 80],
            [100, 200, 300, 400, 500, 600, 700, 800],
            [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        ],
        dtype="f32",
    )
    b = bb.tensor(
        [
            [8, 7, 6, 5, 4, 3, 2, 1],
            [80, 70, 60, 50, 40, 30, 20, 10],
            [800, 700, 600, 500, 400, 300, 200, 100],
            [8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000],
        ],
        dtype="f32",
    )
    expected = [
        [lhs + rhs for lhs, rhs in zip(a_row, b_row)]
        for a_row, b_row in zip(a.tolist(), b.tolist())
    ]

    direct_out = bb.zeros((4, 8), dtype="f32")
    tiled_elementwise_add(a, b, direct_out)
    assert direct_out.tolist() == expected

    compiled_out = bb.zeros((4, 8), dtype="f32")
    artifact = bb.compile(tiled_elementwise_add, a, b, compiled_out, cache_dir=tmp_path)

    assert artifact.ir is not None
    assert artifact.ir.name == "tiled_elementwise_add_kernel"
    assert artifact.lowered_module is not None
    assert artifact.artifact_path.exists()

    artifact(a, b, compiled_out)
    assert compiled_out.tolist() == expected


def test_tensor_abs_example_runs_direct_and_compiled(tmp_path: Path) -> None:
    src = bb.tensor([[1.0, -2.5], [-3.0, 4.25]], dtype="f32")

    direct_out = bb.zeros((2, 2), dtype="f32")
    tensor_abs(src, direct_out)
    assert direct_out.tolist() == [[1.0, 2.5], [3.0, 4.25]]

    compiled_out = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(tensor_abs, src, compiled_out, cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None
    assert artifact.ir.name == "tensor_abs_kernel"
    ops = [operation.op for operation in artifact.ir.operations]
    assert "tensor_abs" in ops

    artifact(src, compiled_out)
    assert compiled_out.tolist() == [[1.0, 2.5], [3.0, 4.25]]


def test_tensor_rounding_example_runs_direct_and_compiled(tmp_path: Path) -> None:
    src = bb.tensor([[-1.75, -0.25], [0.25, 1.75]], dtype="f32")

    direct_round = bb.zeros((2, 2), dtype="f32")
    direct_floor = bb.zeros((2, 2), dtype="f32")
    direct_ceil = bb.zeros((2, 2), dtype="f32")
    direct_trunc = bb.zeros((2, 2), dtype="f32")
    tensor_rounding(src, direct_round, direct_floor, direct_ceil, direct_trunc)
    assert direct_round.tolist() == [[-2.0, 0.0], [0.0, 2.0]]
    assert direct_floor.tolist() == [[-2.0, -1.0], [0.0, 1.0]]
    assert direct_ceil.tolist() == [[-1.0, 0.0], [1.0, 2.0]]
    assert direct_trunc.tolist() == [[-1.0, 0.0], [0.0, 1.0]]

    compiled_round = bb.zeros((2, 2), dtype="f32")
    compiled_floor = bb.zeros((2, 2), dtype="f32")
    compiled_ceil = bb.zeros((2, 2), dtype="f32")
    compiled_trunc = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(
        tensor_rounding,
        src,
        compiled_round,
        compiled_floor,
        compiled_ceil,
        compiled_trunc,
        cache_dir=tmp_path,
        backend="portable",
    )
    assert artifact.ir is not None
    assert artifact.ir.name == "tensor_rounding_kernel"
    ops = [operation.op for operation in artifact.ir.operations]
    assert "math_round" in ops
    assert "math_floor" in ops
    assert "math_ceil" in ops
    assert "math_trunc" in ops

    artifact(src, compiled_round, compiled_floor, compiled_ceil, compiled_trunc)
    assert compiled_round.tolist() == [[-2.0, 0.0], [0.0, 2.0]]
    assert compiled_floor.tolist() == [[-2.0, -1.0], [0.0, 1.0]]
    assert compiled_ceil.tolist() == [[-1.0, 0.0], [1.0, 2.0]]
    assert compiled_trunc.tolist() == [[-1.0, 0.0], [0.0, 1.0]]


def test_tensor_extrema_example_runs_direct_and_compiled(tmp_path: Path) -> None:
    lhs = bb.tensor([[1.0, -2.5], [7.0, 4.25]], dtype="f32")
    rhs = bb.tensor([[0.5, -3.0], [8.0, 2.0]], dtype="f32")

    direct_max = bb.zeros((2, 2), dtype="f32")
    direct_min = bb.zeros((2, 2), dtype="f32")
    tensor_extrema(lhs, rhs, direct_max, direct_min)
    assert direct_max.tolist() == [[1.0, -2.5], [8.0, 4.25]]
    assert direct_min.tolist() == [[0.5, -3.0], [7.0, 2.0]]

    compiled_max = bb.zeros((2, 2), dtype="f32")
    compiled_min = bb.zeros((2, 2), dtype="f32")
    artifact = bb.compile(tensor_extrema, lhs, rhs, compiled_max, compiled_min, cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None
    assert artifact.ir.name == "tensor_extrema_kernel"
    ops = [operation.op for operation in artifact.ir.operations]
    assert "tensor_max" in ops
    assert "tensor_min" in ops

    artifact(lhs, rhs, compiled_max, compiled_min)
    assert compiled_max.tolist() == [[1.0, -2.5], [8.0, 4.25]]
    assert compiled_min.tolist() == [[0.5, -3.0], [7.0, 2.0]]
