from pathlib import Path

import baybridge as bb


COL_MAJOR_XOR4 = bb.make_layout((4, 8), stride=(1, 4), swizzle="xor4")


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(8, 4, 1)))
def layout_copy(
    src: bb.TensorSpec(shape=(4, 8), dtype="f16", layout=COL_MAJOR_XOR4),
    dst: bb.TensorSpec(shape=(4, 8), dtype="f16"),
):
    row = bb.thread_idx("y")
    col = bb.thread_idx("x")
    value = bb.load(src, (row, col))
    bb.store(value, dst, (row, col))


def test_layout_copy_reaches_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(layout_copy, cache_dir=tmp_path, backend="portable")
    assert artifact.ir.arguments[0].spec.to_dict()["layout"]["stride"] == [1, 4]
    assert artifact.ir.arguments[0].spec.to_dict()["layout"]["swizzle"] == "xor4"


def test_layout_copy_reaches_mlir_text_backend(tmp_path: Path) -> None:
    artifact = bb.compile(layout_copy, cache_dir=tmp_path, backend="mlir_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert '!baybridge.tensor<shape=[4, 8], dtype=f16, stride=[1, 4], swizzle="xor4", space=global>' in text
    assert '!baybridge.tensor<shape=[4, 8], dtype=f16, stride=[8, 1], swizzle=null, space=global>' in text


def test_layout_copy_reaches_gpu_text_backend(tmp_path: Path) -> None:
    artifact = bb.compile(layout_copy, cache_dir=tmp_path, backend="gpu_text")
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "memref<4x8xf16, strided<[1, 4], offset: 0>, 1>" in text
    assert "memref<4x8xf16, strided<[8, 1], offset: 0>, 1>" in text
    assert "memref.load %src[%thread_idx_y_1, %thread_idx_x_2]" in text
    assert "memref.store %src_val_3, %dst[%thread_idx_y_1, %thread_idx_x_2]" in text
