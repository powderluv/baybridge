from pathlib import Path

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_elementwise_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    dst[tidx] = src[tidx] + 1.0


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def flydsl_tiled_kernel(src: bb.Tensor, dst: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    tile = bb.local_tile(src, tiler=(2, 4, 2), coord=(1, 0, None), proj=(1, None, 1))
    row = tidx // 2
    col = tidx % 2
    dst[tidx] = tile[row, col, 0]


@bb.kernel
def flydsl_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


def test_flydsl_ref_lowers_elementwise_kernel(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(flydsl_elementwise_kernel, src, dst, cache_dir=tmp_path, backend="flydsl_ref")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "flydsl_ref"
    assert "fly.kernel @flydsl_elementwise_kernel" in artifact.lowered_module.text
    assert "fly.thread_idx" in artifact.lowered_module.text
    assert "fly.load" in artifact.lowered_module.text
    assert "fly.store" in artifact.lowered_module.text
    assert '"family": "elementwise"' in artifact.lowered_module.text


def test_flydsl_ref_lowers_tiled_layout_kernel(tmp_path: Path) -> None:
    src = bb.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        ],
        dtype="f32",
    )
    dst = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(flydsl_tiled_kernel, src, dst, cache_dir=tmp_path, backend="flydsl_ref")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert '"family": "layout_tiled"' in artifact.lowered_module.text
    assert "fly.local_tile" in artifact.lowered_module.text


def test_flydsl_ref_lowers_gemm_kernel(tmp_path: Path) -> None:
    a = bb.tensor([[1.0] * 4 for _ in range(4)], dtype="f32")
    b = bb.tensor([[1.0] * 4 for _ in range(4)], dtype="f32")
    c = bb.zeros((4, 4), dtype="f32")

    artifact = bb.compile(flydsl_gemm_kernel, a, b, c, cache_dir=tmp_path, backend="flydsl_ref")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert '"family": "mfma_gemm"' in artifact.lowered_module.text
    assert "fly.mma" in artifact.lowered_module.text or "fly.gemm" in artifact.lowered_module.text


def test_flydsl_ref_respects_configured_root_hint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BAYBRIDGE_FLYDSL_ROOT", str(tmp_path))
    (tmp_path / "python").mkdir()
    (tmp_path / "README.md").write_text("FlyDSL", encoding="utf-8")

    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")
    artifact = bb.compile(flydsl_elementwise_kernel, src, dst, cache_dir=tmp_path / "cache", backend="flydsl_ref")

    assert artifact.lowered_module is not None
    assert f'"configured_flydsl_root": "{tmp_path.resolve()}"' in artifact.lowered_module.text
    assert "Build hint: use FlyDSL from" in artifact.lowered_module.text
