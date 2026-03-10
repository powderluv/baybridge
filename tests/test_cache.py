from pathlib import Path

import baybridge as bb


@bb.kernel
def cacheable(
    src: bb.TensorSpec(shape=(32,), dtype="f32"),
    dst: bb.TensorSpec(shape=(32,), dtype="f32"),
):
    bb.copy(src, dst)


def test_compile_hits_cache(tmp_path: Path) -> None:
    first = bb.compile(cacheable, cache_dir=tmp_path)
    second = bb.compile(cacheable, cache_dir=tmp_path)
    assert first.cache_key == second.cache_key
    assert second.from_cache is True
