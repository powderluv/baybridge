import pytest

import baybridge as bb


def test_launch_config_rejects_bad_dim3() -> None:
    with pytest.raises(ValueError, match="exactly 3 dimensions"):
        bb.LaunchConfig(grid=(1, 1), block=(1, 1, 1))


def test_launch_config_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        bb.LaunchConfig(grid=(1, 1, 1), block=(64, 0, 1))


def test_tensor_spec_rejects_unknown_address_space() -> None:
    with pytest.raises(ValueError, match="unsupported address_space"):
        bb.TensorSpec(shape=(16,), dtype="f16", address_space="texture")


def test_layout_rejects_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="same rank"):
        bb.make_layout((4, 8), stride=(1,))


def test_layout_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="strides must be > 0"):
        bb.make_layout((4, 8), stride=(1, 0))


def test_make_ordered_layout_rejects_non_permutation_order() -> None:
    with pytest.raises(ValueError, match="permutation of the shape axes"):
        bb.make_ordered_layout((4, 8), order=(0, 0))


def test_recast_layout_rejects_non_integral_extent() -> None:
    layout = bb.make_ordered_layout((2, 5), order=(1, 0))

    with pytest.raises(ValueError, match="divisible by the target bit width"):
        bb.recast_layout(24, 8, layout)


def test_topology_builtin_rejects_bad_axis() -> None:
    with pytest.raises(ValueError, match="axis must be one of"):
        bb.program_id("w")


def test_wave_id_rejects_non_positive_wave_size() -> None:
    with pytest.raises(ValueError, match="wave_size must be > 0"):
        bb.wave_id(wave_size=0)


def test_lane_coords_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="non-empty shape"):
        bb.lane_coords(())


def test_make_identity_tensor_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="non-empty shape"):
        bb.make_identity_tensor(())


def test_wait_group_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="count must be >= 0"):
        bb.wait_group(count=-1)


def test_make_layout_tv_rejects_rank_mismatch() -> None:
    thr_layout = bb.make_layout((4, 32))
    val_layout = bb.make_layout((4, 4, 2))

    with pytest.raises(ValueError, match="same rank"):
        bb.make_layout_tv(thr_layout, val_layout)


def test_composition_rejects_non_layout_rest_mapping() -> None:
    tiled = bb.zipped_divide(bb.tensor([[1, 2, 3, 4]], dtype="f32"), (1, 2))

    with pytest.raises(TypeError, match="baybridge.Layout"):
        bb.composition(tiled, (None, (2, 2)))


@bb.kernel
def invalid_dim_axis(
    src: bb.TensorSpec(shape=(16,), dtype="f16"),
):
    bb.dim(src, 1)


def test_dim_rejects_axis_out_of_range(tmp_path) -> None:
    with pytest.raises(ValueError, match="out of range"):
        bb.compile(invalid_dim_axis, cache_dir=tmp_path, backend="portable")


@bb.kernel
def invalid_partition_rank(
    src: bb.TensorSpec(shape=(16, 16), dtype="f16"),
):
    bb.partition(src, (8,))


def test_partition_rejects_rank_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="tile rank"):
        bb.compile(invalid_partition_rank, cache_dir=tmp_path, backend="portable")


@bb.kernel
def invalid_partition_extent(
    src: bb.TensorSpec(shape=(16, 16), dtype="f16"),
):
    bb.partition(src, (32, 8))


def test_partition_rejects_oversized_tile(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot exceed the source tensor shape"):
        bb.compile(invalid_partition_extent, cache_dir=tmp_path, backend="portable")


@bb.kernel
def invalid_partition_program_axes(
    src: bb.TensorSpec(shape=(16, 16), dtype="f16"),
):
    bb.partition_program(src, (8, 8), axes=("x",))


def test_partition_program_rejects_axis_rank_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="axes must have length 2"):
        bb.compile(invalid_partition_program_axes, cache_dir=tmp_path, backend="portable")


@bb.kernel
def invalid_partition_wave_size(
    src: bb.TensorSpec(shape=(64,), dtype="f16"),
):
    bb.partition_wave(src, (16,), wave_size=0)


def test_partition_wave_rejects_non_positive_wave_size(tmp_path) -> None:
    with pytest.raises(ValueError, match="wave_size must be > 0"):
        bb.compile(invalid_partition_wave_size, cache_dir=tmp_path, backend="portable")


@bb.kernel
def invalid_partition_offset_extent(
    src: bb.TensorSpec(shape=(16, 16), dtype="f16"),
):
    bb.partition(src, (8, 8), offset=(12, 0))


def test_partition_rejects_static_offset_overflow(tmp_path) -> None:
    with pytest.raises(ValueError, match="must stay within the source tensor shape"):
        bb.compile(invalid_partition_offset_extent, cache_dir=tmp_path, backend="portable")


@bb.kernel
def rank_mismatch_load(
    src: bb.TensorSpec(shape=(16, 16), dtype="f16"),
):
    bb.load(src, 0)


def test_load_rejects_rank_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="load expects 2 indices"):
        bb.compile(rank_mismatch_load, cache_dir=tmp_path, backend="portable")


@bb.jit
def invalid_mma_operand_dtypes(
    a: bb.TensorSpec(shape=(16, 16), dtype="f16"),
    b: bb.TensorSpec(shape=(16, 16), dtype="bf16"),
):
    bb.mma(a, b, tile=(16, 16, 16))


def test_mma_rejects_mismatched_operand_dtypes(tmp_path) -> None:
    with pytest.raises(ValueError, match="matching operand dtypes"):
        bb.compile(invalid_mma_operand_dtypes, cache_dir=tmp_path, backend="portable")


@bb.jit
def invalid_fragment_tile(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    bb.make_fragment_a(a, tile=(8, 8, 4))


def test_fragment_helper_rejects_unsupported_tile(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported MFMA descriptor"):
        bb.compile(invalid_fragment_tile, cache_dir=tmp_path, backend="portable")


@bb.kernel
def invalid_dynamic_python_if(
    src: bb.TensorSpec(shape=(16,), dtype="f16"),
    dst: bb.TensorSpec(shape=(16,), dtype="f16"),
):
    if bb.thread_idx("x") == 0:
        bb.copy(src, dst)


def test_dynamic_python_if_rejects_traced_scalars(tmp_path) -> None:
    with pytest.raises(bb.CompilationError, match="dynamic Python control flow"):
        bb.compile(invalid_dynamic_python_if, cache_dir=tmp_path, backend="portable")
