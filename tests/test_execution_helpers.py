from pathlib import Path

import baybridge as bb


@bb.jit(launch=bb.LaunchConfig(grid=(4, 4, 1), block=(64, 4, 1)))
def execution_partition_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    bb.partition_program(a, (16, 16))
    bb.partition_thread(a, (1, 4))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(128, 1, 1)))
def wave_partition_kernel(
    a: bb.TensorSpec(shape=(128,), dtype="f16"),
):
    bb.partition_wave(a, (32,))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(256, 1, 1)))
def wave_copy_kernel(
    src: bb.TensorSpec(shape=(256,), dtype="f16"),
    dst: bb.TensorSpec(shape=(256,), dtype="f16"),
):
    wave = bb.wave_id()
    idx = wave * 4 + bb.lane_id()
    in_bounds = idx < bb.dim(src, 0)
    value = bb.load(src, idx, predicate=in_bounds, else_value=0)
    bb.store(value, dst, idx, predicate=in_bounds)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))
def lane_coord_copy_kernel(
    src: bb.TensorSpec(shape=(64,), dtype="f16"),
    dst: bb.TensorSpec(shape=(64,), dtype="f16"),
):
    row, col = bb.lane_coords((4, 16))
    idx = row * 16 + col
    in_bounds = idx < bb.dim(src, 0)
    value = bb.load(src, idx, predicate=in_bounds, else_value=0)
    bb.store(value, dst, idx, predicate=in_bounds)


@bb.jit(launch=bb.LaunchConfig(grid=(4, 1, 1), block=(64, 1, 1)))
def fragment_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
    b: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    a_frag = bb.make_fragment_a(a, tile=(16, 16, 16))
    b_frag = bb.make_fragment_b(b, tile=(16, 16, 16))
    bb.mma(a_frag, b_frag, tile=(16, 16, 16))


def test_execution_partition_helpers_reach_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(execution_partition_kernel, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "program_id",
        "constant",
        "mul",
        "program_id",
        "constant",
        "mul",
        "partition",
        "thread_idx",
        "constant",
        "mul",
        "thread_idx",
        "constant",
        "mul",
        "partition",
    ]
    assert artifact.ir.operations[6].inputs == ("a", "mul_3", "mul_6")
    assert artifact.ir.operations[13].inputs == ("a", "mul_10", "mul_13")


def test_execution_partition_helpers_reach_gpu_text(tmp_path: Path) -> None:
    artifact = bb.compile(execution_partition_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "%a_part_7 = memref.subview %a[%mul_3, %mul_6] [16, 16] [1, 1]" in text
    assert "%a_part_14 = memref.subview %a[%mul_10, %mul_13] [1, 4] [1, 1]" in text


def test_wave_partition_reaches_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(wave_partition_kernel, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "thread_idx",
        "constant",
        "floordiv",
        "constant",
        "mul",
        "partition",
    ]
    assert artifact.ir.operations[5].inputs == ("a", "mul_5")


def test_wave_partition_reaches_gpu_text(tmp_path: Path) -> None:
    artifact = bb.compile(wave_partition_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "%floordiv_3 = arith.floordivsi %thread_idx_x_1, %cst_2 : index" in text
    assert "%a_part_6 = memref.subview %a[%mul_5] [32] [1]" in text


def test_wave_id_reaches_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(wave_copy_kernel, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "thread_idx",
        "constant",
        "floordiv",
        "constant",
        "mul",
        "lane_id",
        "add",
        "tensor_dim",
        "cmp_lt",
        "constant",
        "masked_load",
        "masked_store",
    ]


def test_wave_id_reaches_gpu_text(tmp_path: Path) -> None:
    artifact = bb.compile(wave_copy_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "%floordiv_3 = arith.floordivsi %thread_idx_x_1, %cst_2 : index" in text
    assert '"rocdl.lane_id"() : () -> index' in text
    assert "arith.cmpi slt %add_7, %src_dim_0_8 : index" in text


def test_lane_coords_reach_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(lane_coord_copy_kernel, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "lane_id",
        "constant",
        "floordiv",
        "constant",
        "mod",
        "constant",
        "mul",
        "add",
        "tensor_dim",
        "cmp_lt",
        "constant",
        "masked_load",
        "masked_store",
    ]


def test_lane_coords_reach_gpu_text(tmp_path: Path) -> None:
    artifact = bb.compile(lane_coord_copy_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "%floordiv_3 = arith.floordivsi %lane_id_1, %cst_2 : index" in text
    assert "%mod_5 = arith.remsi %lane_id_1, %cst_4 : index" in text
    assert "arith.cmpi slt %add_8, %src_dim_0_9 : index" in text


def test_fragment_helpers_reach_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(fragment_kernel, cache_dir=tmp_path, backend="portable")
    ops = [operation.op for operation in artifact.ir.operations]
    assert ops.count("partition") == 2
    assert ops.count("fragment") == 2
    assert ops[-2:] == ["make_tensor", "mma"]
    fragment_attrs = [operation.attrs for operation in artifact.ir.operations if operation.op == "fragment"]
    assert fragment_attrs[0]["role"] == "a"
    assert fragment_attrs[0]["variant"] == "mfma_f32_16x16x16f16"
    assert fragment_attrs[1]["role"] == "b"
