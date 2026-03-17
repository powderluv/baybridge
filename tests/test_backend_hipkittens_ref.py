from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.hipkittens_exec import HipKittensExecBackend


@bb.kernel

def simt_gemm_kernel(a: bb.Tensor, b: bb.Tensor, c: bb.Tensor):
    bb.gemm(a, b, c)


@bb.kernel

def attention_like_kernel(q: bb.Tensor, k: bb.Tensor, scores: bb.Tensor, probs: bb.Tensor, row_sum: bb.Tensor):
    bb.gemm(q, k, scores)
    exp_scores = bb.math.exp2(scores.load())
    probs.store(exp_scores)
    row_sum.store(exp_scores.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=(None, 1)))


@bb.kernel

def layernorm_like_kernel(
    x: bb.Tensor,
    residual: bb.Tensor,
    out: bb.Tensor,
    out_resid: bb.Tensor,
    weight: bb.Tensor,
    bias: bb.Tensor,
):
    mixed = x.load() + residual.load()
    out_resid.store(mixed)
    mean = mixed.reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0) / 4.0
    centered = mixed - mean
    var = (centered * centered).reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0) / 4.0
    denom = bb.math.sqrt(var + 1e-5)
    out.store((centered / denom) * weight.load() + bias.load())


@bb.kernel
def rmsnorm_like_kernel(
    x: bb.Tensor,
    out: bb.Tensor,
    weight: bb.Tensor,
):
    loaded = x.load()
    mean_square = (loaded * loaded).reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0) / 4.0
    denom = bb.math.sqrt(mean_square + 1e-5)
    out.store((loaded / denom) * weight.load())


@bb.kernel
def rmsnorm_rsqrt_like_kernel(
    x: bb.Tensor,
    out: bb.Tensor,
    weight: bb.Tensor,
):
    loaded = x.load()
    inv_rms = bb.math.rsqrt((loaded * loaded).reduce(bb.ReductionOp.ADD, 0.0, reduction_profile=0) / 4.0 + 1e-5)
    out.store((loaded * inv_rms) * weight.load())


def test_compile_auto_prefers_hipkittens_ref_for_attention_kernel(tmp_path: Path) -> None:
    q = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    k = bb.tensor([[1.0, 0.0], [0.0, 1.0]], dtype="f32")
    scores = bb.zeros((2, 2), dtype="f32")
    probs = bb.zeros((2, 2), dtype="f32")
    row_sum = bb.zeros((2,), dtype="f32")

    artifact = bb.compile(attention_like_kernel, q, k, scores, probs, row_sum, cache_dir=tmp_path)

    assert artifact.backend_name == "hipkittens_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_cpp"


def test_compile_auto_prefers_hipkittens_ref_for_layernorm_kernel(tmp_path: Path) -> None:
    x = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    residual = bb.tensor([0.5, 0.5, 0.5, 0.5], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    out_resid = bb.zeros((4,), dtype="f32")
    weight = bb.tensor([1.0, 1.0, 1.0, 1.0], dtype="f32")
    bias = bb.tensor([0.0, 0.0, 0.0, 0.0], dtype="f32")

    artifact = bb.compile(layernorm_like_kernel, x, residual, out, out_resid, weight, bias, cache_dir=tmp_path)

    assert artifact.backend_name == "hipkittens_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_cpp"


def test_compile_auto_prefers_hipkittens_ref_for_rmsnorm_kernel(tmp_path: Path) -> None:
    x = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    weight = bb.tensor([1.0, 1.0, 1.0, 1.0], dtype="f32")

    artifact = bb.compile(rmsnorm_like_kernel, x, out, weight, cache_dir=tmp_path)

    assert artifact.backend_name == "hipkittens_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_cpp"


def test_compile_auto_prefers_hipkittens_ref_for_rsqrt_rmsnorm_kernel(tmp_path: Path) -> None:
    x = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    weight = bb.tensor([1.0, 1.0, 1.0, 1.0], dtype="f32")

    artifact = bb.compile(rmsnorm_rsqrt_like_kernel, x, out, weight, cache_dir=tmp_path)

    assert artifact.backend_name == "hipkittens_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "hipkittens_cpp"


def test_compile_auto_prefers_hipkittens_ref_for_tensorop_family_when_exec_is_unavailable(tmp_path: Path) -> None:
    a = bb.tensor([[1.0] * 16 for _ in range(32)], dtype="bf16")
    b = bb.tensor([[1.0] * 32 for _ in range(16)], dtype="bf16")
    c = bb.zeros((32, 32), dtype="f32")

    artifact = bb.compile(simt_gemm_kernel, a, b, c, cache_dir=tmp_path, target=bb.AMDTarget(arch="gfx942"))

    if HipKittensExecBackend().available(bb.AMDTarget(arch="gfx942")):
        assert artifact.backend_name == "hipkittens_exec"
    else:
        assert artifact.backend_name == "hipkittens_ref"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect in {"hipkittens_cpp", "hipkittens_exec_cpp"}


def test_compile_keeps_default_backend_for_non_hipkittens_family(tmp_path: Path) -> None:
    a = bb.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
    b = bb.tensor([[5.0, 6.0], [7.0, 8.0]], dtype="f32")
    c = bb.zeros((2, 2), dtype="f32")

    artifact = bb.compile(simt_gemm_kernel, a, b, c, cache_dir=tmp_path)

    assert artifact.backend_name == "mlir_text"
    assert artifact.lowered_module is not None
    assert artifact.lowered_module.dialect == "baybridge"


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


def test_hipkittens_ref_lowers_layernorm_family(tmp_path: Path) -> None:
    x = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    residual = bb.tensor([0.5, 0.5, 0.5, 0.5], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    out_resid = bb.zeros((4,), dtype="f32")
    weight = bb.tensor([1.0, 1.0, 1.0, 1.0], dtype="f32")
    bias = bb.tensor([0.0, 0.0, 0.0, 0.0], dtype="f32")

    artifact = bb.compile(
        layernorm_like_kernel,
        x,
        residual,
        out,
        out_resid,
        weight,
        bias,
        cache_dir=tmp_path,
        backend="hipkittens_ref",
    )

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert '"family": "layernorm"' in artifact.lowered_module.text
    assert 'kernels/layernorm/' in artifact.lowered_module.text
    assert 'variance normalization pattern is present' in artifact.lowered_module.text
    assert 'mean-centering is present' in artifact.lowered_module.text

    artifact(x, residual, out, out_resid, weight, bias)
    assert out_resid.tolist() == pytest.approx([1.5, 2.5, 3.5, 4.5])
    assert out.tolist() == pytest.approx([-1.341635, -0.447212, 0.447212, 1.341635], rel=1e-5, abs=1e-5)


def test_hipkittens_ref_lowers_rmsnorm_family(tmp_path: Path) -> None:
    x = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    weight = bb.tensor([1.0, 1.0, 1.0, 1.0], dtype="f32")

    artifact = bb.compile(
        rmsnorm_like_kernel,
        x,
        out,
        weight,
        cache_dir=tmp_path,
        backend="hipkittens_ref",
    )

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert '"family": "rmsnorm"' in artifact.lowered_module.text
    assert 'root-mean-square normalization pattern is present' in artifact.lowered_module.text
    assert (
        'kernels/rmsnorm/' in artifact.lowered_module.text
        or 'kernels/layernorm/' in artifact.lowered_module.text
    )

    artifact(x, out, weight)
    assert out.tolist() == pytest.approx([0.365148, 0.730296, 1.095444, 1.460593], rel=1e-5, abs=1e-5)


def test_hipkittens_ref_lowers_rsqrt_rmsnorm_family(tmp_path: Path) -> None:
    x = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    weight = bb.tensor([1.0, 1.0, 1.0, 1.0], dtype="f32")

    artifact = bb.compile(
        rmsnorm_rsqrt_like_kernel,
        x,
        out,
        weight,
        cache_dir=tmp_path,
        backend="hipkittens_ref",
    )

    assert artifact.ir is not None
    assert artifact.lowered_module is not None
    assert '"family": "rmsnorm"' in artifact.lowered_module.text
    assert 'normalization uses reciprocal-sqrt' in artifact.lowered_module.text

    artifact(x, out, weight)
    assert out.tolist() == pytest.approx([0.365148, 0.730296, 1.095444, 1.460593], rel=1e-5, abs=1e-5)



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


def test_hipkittens_ref_filters_reference_paths_by_configured_root(tmp_path: Path, monkeypatch) -> None:
    hipkittens_root = tmp_path / "HipKittens"
    (hipkittens_root / "include").mkdir(parents=True)
    (hipkittens_root / "include" / "kittens.cuh").write_text("// stub\n", encoding="utf-8")
    (hipkittens_root / "kernels" / "layernorm").mkdir(parents=True)
    monkeypatch.setenv("BAYBRIDGE_HIPKITTENS_ROOT", str(hipkittens_root))

    x = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    out = bb.zeros((4,), dtype="f32")
    weight = bb.tensor([1.0, 1.0, 1.0, 1.0], dtype="f32")

    artifact = bb.compile(rmsnorm_like_kernel, x, out, weight, cache_dir=tmp_path, backend="hipkittens_ref")

    assert artifact.lowered_module is not None
    assert 'kernels/layernorm/' in artifact.lowered_module.text
    assert 'kernels/rmsnorm/' not in artifact.lowered_module.text
