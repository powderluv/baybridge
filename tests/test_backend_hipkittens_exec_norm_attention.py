import math
import os
from pathlib import Path

import pytest

import baybridge as bb
from baybridge.backends.hipkittens_exec import HipKittensExecBackend


@bb.kernel
def exact_layernorm_kernel(
    x: bb.Tensor,
    residual: bb.Tensor,
    out: bb.Tensor,
    out_resid: bb.Tensor,
    weight: bb.Tensor,
    bias: bb.Tensor,
):
    bb.layernorm(x, residual, out, out_resid, weight, bias, epsilon=1e-5)


@bb.kernel
def exact_rmsnorm_kernel(
    x: bb.Tensor,
    out: bb.Tensor,
    gamma: bb.Tensor,
):
    bb.rmsnorm(x, out, gamma, epsilon=1e-5)


@bb.kernel
def exact_attention_kernel(
    q: bb.Tensor,
    k: bb.Tensor,
    v: bb.Tensor,
    out: bb.Tensor,
    lse: bb.Tensor,
):
    bb.attention(q, k, v, out, lse, causal=False)


@bb.kernel
def exact_causal_attention_kernel(
    q: bb.Tensor,
    k: bb.Tensor,
    v: bb.Tensor,
    out: bb.Tensor,
    lse: bb.Tensor,
):
    bb.attention(q, k, v, out, lse, causal=True)


def _fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_root = tmp_path / "HipKittens"
    (fake_root / "include").mkdir(parents=True)
    (fake_root / "include" / "kittens.cuh").write_text("// stub\n", encoding="utf-8")
    (fake_root / "kernels" / "layernorm").mkdir(parents=True)
    (fake_root / "kernels" / "layernorm" / "kernel.cpp").write_text(
        '#include "kittens.cuh"\n'
        '#include "pyutils/pyutils.cuh"\n'
        "constexpr int B = 16;\n"
        "constexpr int H = 16;\n"
        "constexpr int N = 4096;\n"
        "constexpr int HEAD_D = 128;\n"
        "constexpr int D = HEAD_D * H;\n"
        "constexpr float DROPOUT_P = 0.01;\n"
        "template<int _d_model> struct norm_globals {};\n"
        "template<int D> __global__ void layernorm_tk(const norm_globals<D> g) {\n"
        "  auto x = __float2bfloat16(1e-05f);\n"
        "}\n"
        "template<int D> void dispatch_micro(norm_globals<D> g) {}\n"
        "PYBIND11_MODULE(tk_kernel, m) {}\n",
        encoding="utf-8",
    )
    (fake_root / "kernels" / "rmsnorm").mkdir(parents=True)
    (fake_root / "kernels" / "rmsnorm" / "kernel.cpp").write_text(
        '#include "kittens.cuh"\n'
        '#include "pyutils/pyutils.cuh"\n'
        "constexpr int B = 16;\n"
        "constexpr int H = 16;\n"
        "constexpr int N = 4096;\n"
        "constexpr int D = 128;\n"
        "template<int _N> struct rmsnorm_globals {};\n"
        "template<int D> __global__ void rmsnorm_hk(const rmsnorm_globals<D> g) {}\n"
        "template<int D> void dispatch_rmsnorm(rmsnorm_globals<D> g) {}\n"
        "PYBIND11_MODULE(rms_norm_kernel, m) {}\n",
        encoding="utf-8",
    )
    for relative in ("kernels/attn/gqa", "kernels/attn/gqa_causal"):
        source_dir = fake_root / relative
        source_dir.mkdir(parents=True)
        (source_dir / "kernel.cpp").write_text(
            '#include "kittens.cuh"\n'
            '#include "pyutils/pyutils.cuh"\n'
            "using namespace kittens;\n"
            "using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;\n"
            "template<int D> struct attn_globals {\n"
            "  _gl_QKVO Qg, Kg, Vg, Og;\n"
            "  gl<float, -1, -1, -1, -1> L_vec;\n"
            "  hipStream_t stream;\n"
            "  dim3 grid() { return dim3(1, 1, 1); }\n"
            "  dim3 block() { return dim3(1, 1, 1); }\n"
            "  size_t dynamic_shared_memory() { return 0; }\n"
            "};\n"
            "template<int D>\n"
            "void dispatch_micro(attn_globals<D>) {}\n"
            "PYBIND11_MODULE(tk_kernel, m) {}\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("BAYBRIDGE_HIPKITTENS_ROOT", str(fake_root))
    return fake_root


def _make_layernorm_inputs() -> tuple[bb.Tensor, bb.Tensor, bb.Tensor, bb.Tensor, bb.Tensor, bb.Tensor]:
    x = bb.tensor(
        [
            [[((batch * 4 + row) * 128 + col) % 17 / 8.0 for col in range(128)] for row in range(4)]
            for batch in range(2)
        ],
        dtype="bf16",
    )
    residual = bb.tensor(
        [
            [[((batch * 4 + row) * 128 + col) % 11 / 16.0 for col in range(128)] for row in range(4)]
            for batch in range(2)
        ],
        dtype="bf16",
    )
    out = bb.zeros((2, 4, 128), dtype="bf16")
    out_resid = bb.zeros((2, 4, 128), dtype="bf16")
    weight = bb.tensor([1.0 + ((col % 7) / 64.0) for col in range(128)], dtype="bf16")
    bias = bb.tensor([((col % 5) - 2) / 32.0 for col in range(128)], dtype="bf16")
    return x, residual, out, out_resid, weight, bias


def _make_rmsnorm_inputs() -> tuple[bb.Tensor, bb.Tensor, bb.Tensor]:
    x = bb.tensor(
        [
            [[((batch * 4 + row) * 128 + col) % 23 / 8.0 for col in range(128)] for row in range(4)]
            for batch in range(2)
        ],
        dtype="bf16",
    )
    out = bb.zeros((2, 4, 128), dtype="bf16")
    gamma = bb.tensor(
        [
            [[1.0 + ((batch + row + col) % 9) / 64.0 for col in range(128)] for row in range(4)]
            for batch in range(2)
        ],
        dtype="bf16",
    )
    return x, out, gamma


def _make_attention_inputs() -> tuple[bb.Tensor, bb.Tensor, bb.Tensor, bb.Tensor, bb.Tensor]:
    q = bb.zeros((1, 256, 8, 128), dtype="bf16")
    k = bb.zeros((1, 256, 2, 128), dtype="bf16")
    v = bb.zeros((1, 256, 2, 128), dtype="bf16")
    out = bb.zeros((1, 256, 8, 128), dtype="bf16")
    lse = bb.zeros((1, 8, 1, 256), dtype="f32")
    return q, k, v, out, lse


def _exec_backend_available(arch: str) -> bool:
    return HipKittensExecBackend().available(bb.AMDTarget(arch=arch))


def _flatten_nested(values):
    if isinstance(values, list):
        for item in values:
            yield from _flatten_nested(item)
        return
    yield values


def test_hipkittens_ref_lowers_explicit_layernorm_family(tmp_path: Path) -> None:
    artifact = bb.compile(exact_layernorm_kernel, *_make_layernorm_inputs(), cache_dir=tmp_path, backend="hipkittens_ref")

    assert artifact.lowered_module is not None
    assert '"family": "layernorm"' in artifact.lowered_module.text
    assert "explicit baybridge.layernorm op is present" in artifact.lowered_module.text


def test_hipkittens_ref_lowers_explicit_rmsnorm_family(tmp_path: Path) -> None:
    artifact = bb.compile(exact_rmsnorm_kernel, *_make_rmsnorm_inputs(), cache_dir=tmp_path, backend="hipkittens_ref")

    assert artifact.lowered_module is not None
    assert '"family": "rmsnorm"' in artifact.lowered_module.text
    assert "explicit baybridge.rmsnorm op is present" in artifact.lowered_module.text


def test_hipkittens_ref_lowers_explicit_attention_family(tmp_path: Path) -> None:
    artifact = bb.compile(exact_attention_kernel, *_make_attention_inputs(), cache_dir=tmp_path, backend="hipkittens_ref")

    assert artifact.lowered_module is not None
    assert '"family": "attention"' in artifact.lowered_module.text
    assert "explicit baybridge.attention op is present" in artifact.lowered_module.text


def test_hipkittens_exec_lowers_exact_layernorm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    artifact = bb.compile(
        exact_layernorm_kernel,
        *_make_layernorm_inputs(),
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch="gfx950"),
    )

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "// family: layernorm" in text
    assert "constexpr int B = 2;" in text
    assert "constexpr int N = 4;" in text
    assert "constexpr int HEAD_D = 128;" in text
    assert "constexpr int H = 1;" in text
    assert "layernorm_tk(const norm_globals<D> g)" in text
    assert "dispatch_micro<D>(g);" in text


def test_hipkittens_exec_lowers_exact_rmsnorm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    artifact = bb.compile(
        exact_rmsnorm_kernel,
        *_make_rmsnorm_inputs(),
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch="gfx942"),
    )

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "// family: rmsnorm" in text
    assert "constexpr int B = 2;" in text
    assert "constexpr int N = 4;" in text
    assert "constexpr int D = 128;" in text
    assert "rmsnorm_hk<D>" in text


def test_hipkittens_exec_lowers_exact_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)
    artifact = bb.compile(
        exact_attention_kernel,
        *_make_attention_inputs(),
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch="gfx950"),
    )

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "#define ATTN_B 1" in text
    assert "#define ATTN_H 8" in text
    assert "#define ATTN_H_KV 2" in text
    assert "#define ATTN_N 256" in text
    assert "#define ATTN_D 128" in text
    assert "dispatch_micro<ATTN_D>(g);" in text


def test_compile_auto_prefers_hipkittens_exec_for_exact_layernorm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)

    artifact = bb.compile(
        exact_layernorm_kernel,
        *_make_layernorm_inputs(),
        cache_dir=tmp_path,
        target=bb.AMDTarget(arch="gfx950"),
    )

    assert artifact.backend_name == "hipkittens_exec"


def test_compile_auto_prefers_hipkittens_exec_for_exact_rmsnorm_gfx942(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_root(tmp_path, monkeypatch)

    artifact = bb.compile(
        exact_rmsnorm_kernel,
        *_make_rmsnorm_inputs(),
        cache_dir=tmp_path,
        target=bb.AMDTarget(arch="gfx942"),
    )

    if _exec_backend_available("gfx942"):
        assert artifact.backend_name == "hipkittens_exec"
    else:
        assert artifact.backend_name == "hipkittens_ref"


def test_compile_auto_prefers_hipkittens_exec_for_exact_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)

    artifact = bb.compile(
        exact_attention_kernel,
        *_make_attention_inputs(),
        cache_dir=tmp_path,
        target=bb.AMDTarget(arch="gfx950"),
    )

    assert artifact.backend_name == "hipkittens_exec"


def test_compile_falls_back_to_hipkittens_ref_for_attention_on_gfx942(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_root(tmp_path, monkeypatch)

    artifact = bb.compile(
        exact_attention_kernel,
        *_make_attention_inputs(),
        cache_dir=tmp_path,
        target=bb.AMDTarget(arch="gfx942"),
    )

    assert artifact.backend_name == "hipkittens_ref"


def test_hipkittens_exec_runs_exact_layernorm(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HipKittens backend tests")
    if not os.environ.get("BAYBRIDGE_HIPKITTENS_ROOT"):
        pytest.skip("set BAYBRIDGE_HIPKITTENS_ROOT to a HipKittens checkout to run executable HipKittens backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    if not _exec_backend_available(target_arch):
        pytest.skip(f"hipkittens_exec is not toolchain-ready for target {target_arch}")

    ref_args = _make_layernorm_inputs()
    ref_artifact = bb.compile(exact_layernorm_kernel, *ref_args, cache_dir=tmp_path, backend="portable")
    ref_artifact(*ref_args)

    exec_args = _make_layernorm_inputs()
    exec_artifact = bb.compile(
        exact_layernorm_kernel,
        *exec_args,
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    exec_artifact(*exec_args)

    assert list(_flatten_nested(exec_args[2].tolist())) == pytest.approx(
        list(_flatten_nested(ref_args[2].tolist())),
        rel=5e-2,
        abs=5e-2,
    )
    assert list(_flatten_nested(exec_args[3].tolist())) == pytest.approx(
        list(_flatten_nested(ref_args[3].tolist())),
        rel=5e-2,
        abs=5e-2,
    )


def test_hipkittens_exec_runs_exact_rmsnorm(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HipKittens backend tests")
    if not os.environ.get("BAYBRIDGE_HIPKITTENS_ROOT"):
        pytest.skip("set BAYBRIDGE_HIPKITTENS_ROOT to a HipKittens checkout to run executable HipKittens backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    if target_arch != "gfx942":
        pytest.skip("hipkittens_exec rmsnorm is currently wired only for gfx942")
    if not _exec_backend_available(target_arch):
        pytest.skip(f"hipkittens_exec is not toolchain-ready for target {target_arch}")

    ref_args = _make_rmsnorm_inputs()
    ref_artifact = bb.compile(exact_rmsnorm_kernel, *ref_args, cache_dir=tmp_path, backend="portable")
    ref_artifact(*ref_args)

    exec_args = _make_rmsnorm_inputs()
    exec_artifact = bb.compile(
        exact_rmsnorm_kernel,
        *exec_args,
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    exec_artifact(*exec_args)

    assert list(_flatten_nested(exec_args[1].tolist())) == pytest.approx(
        list(_flatten_nested(ref_args[1].tolist())),
        rel=5e-2,
        abs=5e-2,
    )


def test_hipkittens_exec_runs_exact_attention(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HipKittens backend tests")
    if not os.environ.get("BAYBRIDGE_HIPKITTENS_ROOT"):
        pytest.skip("set BAYBRIDGE_HIPKITTENS_ROOT to a HipKittens checkout to run executable HipKittens backend tests")
    if not torch.cuda.is_available():
        pytest.skip("ROCm torch is required for the exact attention HipKittens test")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    if target_arch != "gfx950":
        pytest.skip("hipkittens_exec fused attention is currently wired only for gfx950")
    if not _exec_backend_available(target_arch):
        pytest.skip(f"hipkittens_exec is not toolchain-ready for target {target_arch}")

    torch.manual_seed(0)
    q = torch.randn((1, 256, 8, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((1, 256, 2, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((1, 256, 2, 128), dtype=torch.bfloat16, device="cuda")
    out = torch.zeros((1, 256, 8, 128), dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros((1, 8, 1, 256), dtype=torch.float32, device="cuda")

    qh = bb.from_dlpack(q)
    kh = bb.from_dlpack(k)
    vh = bb.from_dlpack(v)
    outh = bb.from_dlpack(out)
    lseh = bb.from_dlpack(lse)

    artifact = bb.compile(
        exact_attention_kernel,
        qh,
        kh,
        vh,
        outh,
        lseh,
        cache_dir=tmp_path,
        backend="hipkittens_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(qh, kh, vh, outh, lseh)

    group_size = 8 // 2
    scale = 1.0 / math.sqrt(128.0)
    k_expanded = k.repeat_interleave(group_size, dim=2)
    v_expanded = v.repeat_interleave(group_size, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k_expanded.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhqk,bkhd->bqhd", probs, v_expanded.float()).to(torch.bfloat16)
    ref_lse = torch.logsumexp(scores, dim=-1).unsqueeze(2)

    assert torch.allclose(out.float(), ref_out.float(), atol=3e-1, rtol=3e-1)
    assert torch.allclose(lse, ref_lse, atol=3e-1, rtol=3e-1)
