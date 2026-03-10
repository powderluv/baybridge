import operator
from pathlib import Path

import baybridge as bb


@bb.kernel
def elementwise_apply_like_kernel(op, m_inputs, m_c, c_c, shape, tv_layout):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()

    blk_crd = ((None, None), bidx)
    g_inputs = [tensor[blk_crd] for tensor in m_inputs]
    g_c = m_c[blk_crd]
    g_crd = c_c[blk_crd]

    tidfrg_inputs = [bb.composition(tensor, tv_layout) for tensor in g_inputs]
    tidfrg_c = bb.composition(g_c, tv_layout)
    tidfrg_crd = bb.composition(g_crd, tv_layout)

    thr_crd = (tidx, bb.repeat_like(None, tidfrg_inputs[0][1]))
    thr_inputs = [tensor[thr_crd] for tensor in tidfrg_inputs]
    thr_c = tidfrg_c[thr_crd]
    thr_crds = tidfrg_crd[thr_crd]

    frg_pred = bb.make_rmem_tensor(thr_crds.shape, "i1")
    for index in range(bb.size(frg_pred)):
        frg_pred[index] = bb.elem_less(thr_crds[index], shape)

    result = op(*[tensor.load() for tensor in thr_inputs])
    thr_c.store(result)


@bb.jit
def elementwise_apply_like(op, inputs, result):
    thr_layout = bb.make_ordered_layout((1, 4), order=(1, 0))
    val_layout = bb.make_ordered_layout((2, 1), order=(1, 0))
    tiler_mn, tv_layout = bb.make_layout_tv(thr_layout, val_layout)

    m_inputs = [bb.zipped_divide(input_tensor, tiler_mn) for input_tensor in inputs]
    m_c = bb.zipped_divide(result, tiler_mn)

    remap_block = bb.make_ordered_layout(bb.select(m_inputs[0].shape[1], mode=[1, 0]), order=(1, 0))
    m_inputs = [bb.composition(tensor, (None, remap_block)) for tensor in m_inputs]
    m_c = bb.composition(m_c, (None, remap_block))

    c_c = bb.zipped_divide(bb.make_identity_tensor(result.shape), tiler=tiler_mn)
    elementwise_apply_like_kernel(op, m_inputs, m_c, c_c, result.shape, tv_layout).launch(
        grid=(bb.size(m_c, mode=[1]), 1, 1),
        block=(bb.size(tv_layout, mode=[0]), 1, 1),
    )


def _input_tensors():
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
    expected = [
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
        [900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0],
        [9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0],
    ]
    return a, b, expected


def test_elementwise_apply_like_runs_direct_and_compiled(tmp_path: Path) -> None:
    a, b, expected = _input_tensors()

    direct_out = bb.zeros((4, 8), dtype="f32")
    elementwise_apply_like(operator.add, [a, b], direct_out)
    assert direct_out.tolist() == expected

    compiled_out = bb.zeros((4, 8), dtype="f32")
    artifact = bb.compile(elementwise_apply_like, operator.add, [a, b], compiled_out, cache_dir=tmp_path)

    assert artifact.ir is not None
    assert artifact.ir.name == "elementwise_apply_like_kernel"
    assert artifact.artifact_path.exists()

    artifact(operator.add, [a, b], compiled_out)
    assert compiled_out.tolist() == expected


def test_elementwise_apply_like_traces_coordinate_predicates(tmp_path: Path) -> None:
    a, b, _ = _input_tensors()
    c = bb.zeros((4, 8), dtype="f32")

    artifact = bb.compile(elementwise_apply_like, operator.add, [a, b], c, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    assert [argument.name for argument in artifact.ir.arguments] == ["inputs_0", "inputs_1", "result"]
    ops = [operation.op for operation in artifact.ir.operations]
    assert "thread_fragment_load" in ops
    assert "thread_fragment_store" in ops
    assert ops.count("cmp_lt") == 4
    assert ops.count("store") == 2
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '"baybridge.thread_fragment_load"' in text
    assert '"baybridge.thread_fragment_store"' in text
