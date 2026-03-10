from pathlib import Path

import baybridge as bb


@bb.kernel

def simt_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel

def attention_like_kernel(q: bb.Tensor, k: bb.Tensor, scores: bb.Tensor, probs: bb.Tensor, row_sum: bb.Tensor):
    bb.gemm(q, k, scores)
    exp_scores = bb.math.exp2(scores.load())
    probs.store(exp_scores)
    row_sum.store(exp_scores.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


def test_hipkittens_ref_lowers_gemm_family(tmp_path: Path) -> None:
    a = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    b = bb.tensor([[5.0, 6.0], [7.0, 8.0]], dtype="f32")
    c = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(simt_gemm_kernel, a, b, c, cache_dir=tmp_path, backend="hipkittens_ref")

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_cpp"
    assert '#include "kittens.cuh"' in artifact.lowered_module.text
    assert 'using namespace kittens;' in artifact.lowered_module.text
    assert '"family": "simt_gemm"' in artifact.lowered_module.text
    assert 'kernels/gemm/bf16fp32/' in artifact.lowered_module.text
    assert (
        '-I$BAYBRIDGE_HIPKITTENS_ROOT/include' in artifact.lowered_module.text
        or "Build hint: hipcc -I" in artifact.lowered_module.text
    )

    artifact(a, b, c)
    assert c.tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_hipkittens_ref_lowers_attention_family(tmp_path: Path) -> None:
    q = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    k = bb.tensor([[1.0, 0.0], [0.0, 1.0]], dtype="f32")
    scores = bb.zeros((2, 2), dtype="f32")
    probs = bb.zeros((2, 2), dtype="f32")
    row_sum = bb.zeros((2,), dtype="f32")

    artifact = bb.compile(
        attention_like_kernel,
        q,
        k,
        scores,
        probs,
        row_sum,
        cache_dir=tmp_path,
        backend="hipkittens_ref",
    )

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert '"family": "attention"' in artifact.lowered_module.text
    assert 'kernels/attn/gqa/' in artifact.lowered_module.text
    assert 'softmax-style exp2 is present' in artifact.lowered_module.text



def test_hipkittens_ref_uses_configured_root_hint(tmp_path: Path, monkeypatch) -> None:
    hipkittens_root = tmp_path / "HipKittens"
    (hipkittens_root / "include").mkdir(parents=True)
    (hipkittens_root / "include" / "kittens.cuh").write_text("// stub\n", encoding="utf-8")
    monkeypatch.setenv("BAYBRIDGE_HIPKITTENS_ROOT", str(hipkittens_root))

    a = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    b = bb.tensor([[5.0, 6.0], [7.0, 8.0]], dtype="f32")
    c = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(simt_gemm_kernel, a, b, c, cache_dir=tmp_path, backend="hipkittens_ref")

    assert artifact.lowered_module is not None
    assert str(hipkittens_root / 'include') in artifact.lowered_module.text
