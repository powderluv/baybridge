import pytest

import baybridge as bb


@bb.kernel
def unsupported_kernel(
    src: bb.TensorSpec(shape=(16,), dtype="f16"),
):
    bb.nvgpu.cp_async(src, src)


@bb.kernel
def missing_annotation(src):
    bb.copy(src, src)


def test_nvgpu_namespace_is_rejected(tmp_path) -> None:
    with pytest.raises(bb.UnsupportedOperationError, match="NVIDIA-specific"):
        bb.compile(unsupported_kernel, cache_dir=tmp_path)


def test_missing_annotations_fail_early(tmp_path) -> None:
    with pytest.raises(bb.CompilationError, match="must be annotated"):
        bb.compile(missing_annotation, cache_dir=tmp_path)
