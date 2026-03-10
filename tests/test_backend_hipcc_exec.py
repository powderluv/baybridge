import os
import operator
from pathlib import Path

import pytest

import baybridge as bb
from tests.test_elementwise_add_copyatom_like import elementwise_add_copyatom_like
from tests.test_elementwise_apply_like import elementwise_apply_like


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


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def indexed_add_half_kernel(
    g_a: bb.Tensor,
    g_b: bb.Tensor,
    g_c: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    g_c[tidx] = g_a[tidx] + g_b[tidx]


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def shared_stage_kernel(
    g_a: bb.Tensor,
    g_c: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    smem = bb.make_tensor("smem", shape=(4,), dtype="f32", address_space=bb.AddressSpace.SHARED)
    smem[tidx] = g_a[tidx]
    bb.barrier()
    g_c[tidx] = smem[tidx]


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


def test_hipcc_exec_backend_emits_hip_source(tmp_path: Path) -> None:
    a = bb.tensor([[1, 2], [3, 4]], dtype="f32")
    b = bb.tensor([[10, 20], [30, 40]], dtype="f32")
    c = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(indexed_add_kernel, a, b, c, cache_dir=tmp_path, backend="hipcc_exec")

    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hip_cpp"
    text = artifact.lowered_module.text
    assert 'extern "C" __global__ void indexed_add_kernel(float* g_a, float* g_b, float* g_c)' in text
    assert 'extern "C" int launch_indexed_add_kernel(' in text
    assert "const std::int64_t thread_idx_x_1 = static_cast<std::int64_t>(threadIdx.x);" in text
    assert "const float g_a_val_8 = g_a[(floordiv_7) * 2 + (mod_5) * 1];" in text
    assert "g_c[(floordiv_7) * 2 + (mod_5) * 1] = add_10;" in text
    assert artifact.lowered_path is not None
    assert artifact.lowered_path.suffixes[-2:] == [".hip", ".cpp"]


def test_hipcc_exec_backend_emits_half_and_bfloat_tensor_types(tmp_path: Path) -> None:
    half_a = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f16")
    half_b = bb.tensor([0.5, 1.5, 2.5, 3.5], dtype="f16")
    half_c = bb.zeros((4,), dtype="f16")
    half_artifact = bb.compile(indexed_add_half_kernel, half_a, half_b, half_c, cache_dir=tmp_path, backend="hipcc_exec")
    assert half_artifact.lowered_module is not None
    assert '__global__ void indexed_add_half_kernel(__half* g_a, __half* g_b, __half* g_c)' in half_artifact.lowered_module.text

    bf16_a = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="bf16")
    bf16_b = bb.tensor([0.5, 1.5, 2.5, 3.5], dtype="bf16")
    bf16_c = bb.zeros((4,), dtype="bf16")
    bf16_artifact = bb.compile(indexed_add_half_kernel, bf16_a, bf16_b, bf16_c, cache_dir=tmp_path, backend="hipcc_exec")
    assert bf16_artifact.lowered_module is not None
    assert 'hip_bfloat16* g_a' in bf16_artifact.lowered_module.text
    assert 'hip_bfloat16* g_b' in bf16_artifact.lowered_module.text
    assert 'hip_bfloat16* g_c' in bf16_artifact.lowered_module.text


def test_hipcc_exec_backend_emits_shared_memory_and_barrier(tmp_path: Path) -> None:
    a = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    c = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(shared_stage_kernel, a, c, cache_dir=tmp_path, backend="hipcc_exec")

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "__shared__ float smem[4];" in text
    assert "__syncthreads();" in text


def test_hipcc_exec_backend_emits_partition_copy_and_tensor_add(tmp_path: Path) -> None:
    a = bb.tensor([[float(index) for index in range(1024)]], dtype="f32")
    b = bb.tensor([[float(index + 1) for index in range(1024)]], dtype="f32")
    c = bb.zeros((1, 1024), dtype="f32")

    artifact = bb.compile(vectorized_add_wrapper, a, b, c, cache_dir=tmp_path, backend="hipcc_exec")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "float* m_a_part_" in text
    assert "float* m_b_part_" in text
    assert "float* m_c_part_" in text
    assert "float tensor_add_" in text
    assert "_load_" in text
    assert "for (std::int64_t idx_" in text


def test_hipcc_exec_backend_emits_thread_fragment_gather_and_scatter(tmp_path: Path) -> None:
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

    artifact = bb.compile(tiled_add_wrapper, a, b, c, cache_dir=tmp_path, backend="hipcc_exec")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "thread_idx_x_1" in text
    assert "tensor_add_" in text
    assert "thread_coord" in text
    assert "value_coord" in text


def test_hipcc_exec_backend_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    a = bb.tensor([[1, 2], [3, 4]], dtype="f32")
    b = bb.tensor([[10, 20], [30, 40]], dtype="f32")
    c = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(
        indexed_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, b, c)

    assert c.tolist() == [[11.0, 22.0], [33.0, 44.0]]


def test_hipcc_exec_backend_runs_shared_memory_kernel_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    a = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    c = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        shared_stage_kernel,
        a,
        c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, c)

    assert c.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_hipcc_exec_backend_runs_half_kernel_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    a = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f16")
    b = bb.tensor([0.5, 1.5, 2.5, 3.5], dtype="f16")
    c = bb.zeros((4,), dtype="f16")

    artifact = bb.compile(
        indexed_add_half_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, b, c)

    assert c.tolist() == pytest.approx([1.5, 3.5, 5.5, 7.5], rel=1e-3, abs=1e-3)


def test_hipcc_exec_backend_runs_vectorized_wrapper_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    a = bb.tensor([[float(index) for index in range(1024)]], dtype="f32")
    b = bb.tensor([[float(index + 1000) for index in range(1024)]], dtype="f32")
    c = bb.zeros((1, 1024), dtype="f32")

    artifact = bb.compile(
        vectorized_add_wrapper,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, b, c)

    assert c.tolist() == [[float((2 * index) + 1000) for index in range(1024)]]


def test_hipcc_exec_backend_runs_tiled_wrapper_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
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

    artifact = bb.compile(
        tiled_add_wrapper,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(a, b, c)

    assert c.tolist() == [
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
        [900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0],
        [9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0],
    ]


def test_hipcc_exec_backend_emits_predicated_copyatom_kernel(tmp_path: Path) -> None:
    a = bb.tensor([[float((row + 1) * (col + 1)) for col in range(10)] for row in range(3)], dtype="f32")
    b = bb.tensor([[float(((row + 1) * 100) + col) for col in range(10)] for row in range(3)], dtype="f32")
    c = bb.zeros((3, 10), dtype="f32")

    artifact = bb.compile(elementwise_add_copyatom_like, a, b, c, cache_dir=tmp_path, backend="hipcc_exec")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "? m_a_part_" in text
    assert "? m_b_part_" in text
    assert "if (rmem[" in text
    assert "thread_coord" in text
    assert "value_coord" in text


def test_hipcc_exec_backend_emits_elementwise_apply_like_kernel(tmp_path: Path) -> None:
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

    artifact = bb.compile(elementwise_apply_like, operator.add, [a, b], c, cache_dir=tmp_path, backend="hipcc_exec")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "bool rmem" in text
    assert "const bool cmp_lt_" in text
    assert "&&" in text
    assert "thread_coord" in text
    assert "value_coord" in text


def test_hipcc_exec_backend_runs_elementwise_apply_like_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
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

    artifact = bb.compile(
        elementwise_apply_like,
        operator.add,
        [a, b],
        c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(operator.add, [a, b], c)

    assert c.tolist() == [
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
        [900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0],
        [9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0, 9000.0],
    ]
