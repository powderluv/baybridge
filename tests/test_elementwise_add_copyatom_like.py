import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel
def elementwise_add_copyatom_like_kernel(
    g_a: bb.Tensor,
    g_b: bb.Tensor,
    g_c: bb.Tensor,
    c_c: bb.Tensor,
    shape: bb.Shape,
    thr_layout: bb.Layout,
    val_layout: bb.Layout,
):
    tidx, _, _ = bb.arch.thread_idx()
    bidx, _, _ = bb.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blk_a = g_a[blk_coord]
    blk_b = g_b[blk_coord]
    blk_c = g_c[blk_coord]
    blk_crd = c_c[blk_coord]

    copy_atom_load = bb.make_copy_atom(bb.nvgpu.CopyUniversalOp(), g_a.element_type)
    copy_atom_store = bb.make_copy_atom(bb.nvgpu.CopyUniversalOp(), g_c.element_type)

    tiled_copy_a = bb.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_b = bb.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_c = bb.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_a = tiled_copy_a.get_slice(tidx)
    thr_copy_b = tiled_copy_b.get_slice(tidx)
    thr_copy_c = tiled_copy_c.get_slice(tidx)

    thr_a = thr_copy_a.partition_S(blk_a)
    thr_b = thr_copy_b.partition_S(blk_b)
    thr_c = thr_copy_c.partition_S(blk_c)
    thr_crd = thr_copy_c.partition_S(blk_crd)

    frg_a = bb.make_fragment_like(thr_a)
    frg_b = bb.make_rmem_tensor_like(thr_b)
    frg_c = bb.make_rmem_tensor_like(thr_c)
    frg_pred = bb.make_rmem_tensor(thr_crd.shape, "i1")

    for index in range(bb.size(frg_pred)):
        frg_pred[index] = bb.elem_less(thr_crd[index], shape)

    bb.copy(copy_atom_load, thr_a, frg_a, pred=frg_pred)
    bb.copy(copy_atom_load, thr_b, frg_b, pred=frg_pred)
    frg_c.store(frg_a.load() + frg_b.load())
    bb.copy(copy_atom_store, frg_c, thr_c, pred=frg_pred)


@bb.testing.autotune_jit(
    params_dict={"copy_bits": [64]},
    update_on_change=["shape"],
    warmup_iterations=1,
    iterations=1,
)
@bb.jit
def elementwise_add_copyatom_autotune_like(
    m_a: bb.Tensor,
    m_b: bb.Tensor,
    m_c: bb.Tensor,
    *,
    copy_bits: int = 64,
):
    elementwise_add_copyatom_like(m_a, m_b, m_c, copy_bits=copy_bits)


@bb.jit
def elementwise_add_copyatom_like(
    m_a: bb.Tensor,
    m_b: bb.Tensor,
    m_c: bb.Tensor,
    *,
    copy_bits: int = 64,
):
    dtype = m_a.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = bb.make_ordered_layout((1, 4), order=(1, 0))
    val_layout = bb.make_ordered_layout((2, vector_size), order=(1, 0))
    tiler_mn, tv_layout = bb.make_layout_tv(thr_layout, val_layout)

    g_a = bb.zipped_divide(m_a, tiler_mn)
    g_b = bb.zipped_divide(m_b, tiler_mn)
    g_c = bb.zipped_divide(m_c, tiler_mn)
    c_c = bb.zipped_divide(bb.make_identity_tensor(m_c.shape), tiler=tiler_mn)

    elementwise_add_copyatom_like_kernel(g_a, g_b, g_c, c_c, m_c.shape, thr_layout, val_layout).launch(
        grid=(bb.size(g_c, mode=[1]), 1, 1),
        block=(bb.size(tv_layout, mode=[0]), 1, 1),
    )


def _tensor_pair(rows: int, cols: int) -> tuple[bb.Tensor, bb.Tensor, list[list[float]]]:
    a = bb.tensor(
        [[float((row + 1) * (col + 1)) for col in range(cols)] for row in range(rows)],
        dtype="f32",
    )
    b = bb.tensor(
        [[float(((row + 1) * 100) + col) for col in range(cols)] for row in range(rows)],
        dtype="f32",
    )
    expected = [
        [float((row + 1) * (col + 1) + ((row + 1) * 100) + col) for col in range(cols)]
        for row in range(rows)
    ]
    return a, b, expected


def test_elementwise_add_copyatom_like_runs_direct_and_compiled(tmp_path: Path) -> None:
    a, b, expected = _tensor_pair(4, 8)

    direct_out = bb.zeros((4, 8), dtype="f32")
    elementwise_add_copyatom_like(a, b, direct_out)
    assert direct_out.tolist() == expected

    compiled_out = bb.zeros((4, 8), dtype="f32")
    artifact = bb.compile(elementwise_add_copyatom_like, a, b, compiled_out, cache_dir=tmp_path)

    assert artifact.ir is not None
    assert artifact.ir.name == "elementwise_add_copyatom_like_kernel"
    assert artifact.artifact_path.exists()

    artifact(a, b, compiled_out)
    assert compiled_out.tolist() == expected


def test_elementwise_add_copyatom_like_traces_copy_atoms(tmp_path: Path) -> None:
    a, b, _ = _tensor_pair(3, 10)
    c = bb.zeros((3, 10), dtype="f32")

    artifact = bb.compile(
        elementwise_add_copyatom_like,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="gpu_text",
    )

    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]
    assert ops.count("thread_fragment_load") == 2
    assert ops.count("thread_fragment_store") == 1
    assert "tensor_add" in ops
    assert artifact.lowered_module is not None
    assert '"baybridge.thread_fragment_load"' in artifact.lowered_module.text
    assert '"baybridge.thread_fragment_store"' in artifact.lowered_module.text


def test_elementwise_add_copyatom_autotune_like_compiles(tmp_path: Path) -> None:
    a, b, expected = _tensor_pair(4, 8)
    c = bb.zeros((4, 8), dtype="f32")

    artifact = bb.compile(elementwise_add_copyatom_autotune_like, a, b, c, cache_dir=tmp_path)

    assert artifact.ir is not None
    artifact(a, b, c)
    assert c.tolist() == expected


def test_elementwise_add_copyatom_like_runs_partial_tiles_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    a, b, expected = _tensor_pair(3, 10)
    c = bb.zeros((3, 10), dtype="f32")

    artifact = bb.compile(
        elementwise_add_copyatom_like,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, b, c)

    assert c.tolist() == expected
