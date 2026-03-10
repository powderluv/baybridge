from pathlib import Path

import baybridge as bb


@bb.jit(launch=bb.LaunchConfig(grid=(2, 2, 1), block=(128, 1, 1)))
def offset_tiled_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
    b: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    tile_m = bb.program_id("y") * 16
    tile_n = bb.program_id("x") * 16
    a_tile = bb.partition(a, (16, 16), offset=(tile_m, tile_n))
    b_tile = bb.partition(b, (16, 16), offset=(tile_m, tile_n))
    bb.mma(a_tile, b_tile, tile=(16, 16, 16))


def test_offset_partition_reaches_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(offset_tiled_kernel, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "program_id",
        "constant",
        "mul",
        "program_id",
        "constant",
        "mul",
        "partition",
        "partition",
        "make_tensor",
        "mma",
    ]
    assert artifact.ir.operations[6].inputs == ("a", "mul_3", "mul_6")
    assert artifact.ir.operations[7].inputs == ("b", "mul_3", "mul_6")


def test_offset_partition_reaches_gpu_text_backend(tmp_path: Path) -> None:
    artifact = bb.compile(offset_tiled_kernel, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "%a_part_7 = memref.subview %a[%mul_3, %mul_6] [16, 16] [1, 1]" in text
    assert "%b_part_8 = memref.subview %b[%mul_3, %mul_6] [16, 16] [1, 1]" in text
