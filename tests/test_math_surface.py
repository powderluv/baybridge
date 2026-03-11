import math
import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def extended_math_kernel(
    src: bb.Tensor,
    other: bb.Tensor,
    dst_exp: bb.Tensor,
    dst_log: bb.Tensor,
    dst_log2: bb.Tensor,
    dst_log10: bb.Tensor,
    dst_cos: bb.Tensor,
    dst_erf: bb.Tensor,
    dst_atan2: bb.Tensor,
):
    values = src.load()
    other_values = other.load()
    dst_exp.store(bb.math.exp(values))
    dst_log.store(bb.math.log(values))
    dst_log2.store(bb.math.log2(values))
    dst_log10.store(bb.math.log10(values))
    dst_cos.store(bb.math.cos(values))
    dst_erf.store(bb.math.erf(values))
    dst_atan2.store(bb.math.atan2(values, other_values))


@bb.jit
def extended_math_wrapper(
    src: bb.Tensor,
    other: bb.Tensor,
    dst_exp: bb.Tensor,
    dst_log: bb.Tensor,
    dst_log2: bb.Tensor,
    dst_log10: bb.Tensor,
    dst_cos: bb.Tensor,
    dst_erf: bb.Tensor,
    dst_atan2: bb.Tensor,
):
    extended_math_kernel(
        src,
        other,
        dst_exp,
        dst_log,
        dst_log2,
        dst_log10,
        dst_cos,
        dst_erf,
        dst_atan2,
    ).launch(grid=(1, 1, 1), block=(1, 1, 1))


def test_extended_math_runtime_and_compile(tmp_path: Path) -> None:
    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    other = bb.tensor([1.0, 2.0, 8.0], dtype="f32")
    dst_exp = bb.zeros((3,), dtype="f32")
    dst_log = bb.zeros((3,), dtype="f32")
    dst_log2 = bb.zeros((3,), dtype="f32")
    dst_log10 = bb.zeros((3,), dtype="f32")
    dst_cos = bb.zeros((3,), dtype="f32")
    dst_erf = bb.zeros((3,), dtype="f32")
    dst_atan2 = bb.zeros((3,), dtype="f32")

    extended_math_wrapper(src, other, dst_exp, dst_log, dst_log2, dst_log10, dst_cos, dst_erf, dst_atan2)

    assert dst_exp.tolist() == pytest.approx([math.exp(1.0), math.exp(2.0), math.exp(4.0)], rel=1e-6, abs=1e-6)
    assert dst_log.tolist() == pytest.approx([math.log(1.0), math.log(2.0), math.log(4.0)], rel=1e-6, abs=1e-6)
    assert dst_log2.tolist() == pytest.approx([math.log2(1.0), math.log2(2.0), math.log2(4.0)], rel=1e-6, abs=1e-6)
    assert dst_log10.tolist() == pytest.approx([math.log10(1.0), math.log10(2.0), math.log10(4.0)], rel=1e-6, abs=1e-6)
    assert dst_cos.tolist() == pytest.approx([math.cos(1.0), math.cos(2.0), math.cos(4.0)], rel=1e-6, abs=1e-6)
    assert dst_erf.tolist() == pytest.approx([math.erf(1.0), math.erf(2.0), math.erf(4.0)], rel=1e-6, abs=1e-6)
    assert dst_atan2.tolist() == pytest.approx(
        [math.atan2(1.0, 1.0), math.atan2(2.0, 2.0), math.atan2(4.0, 8.0)],
        rel=1e-6,
        abs=1e-6,
    )

    artifact = bb.compile(
        extended_math_wrapper,
        src,
        other,
        dst_exp,
        dst_log,
        dst_log2,
        dst_log10,
        dst_cos,
        dst_erf,
        dst_atan2,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )
    assert artifact.ir is not None
    ops = [operation.op for operation in artifact.ir.operations]
    assert "math_exp" in ops
    assert "math_log" in ops
    assert "math_log2" in ops
    assert "math_log10" in ops
    assert "math_cos" in ops
    assert "math_erf" in ops
    assert "math_atan2" in ops


def test_extended_math_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    src = bb.tensor([1.0, 2.0, 4.0], dtype="f32")
    other = bb.tensor([1.0, 2.0, 8.0], dtype="f32")
    dst_exp = bb.zeros((3,), dtype="f32")
    dst_log = bb.zeros((3,), dtype="f32")
    dst_log2 = bb.zeros((3,), dtype="f32")
    dst_log10 = bb.zeros((3,), dtype="f32")
    dst_cos = bb.zeros((3,), dtype="f32")
    dst_erf = bb.zeros((3,), dtype="f32")
    dst_atan2 = bb.zeros((3,), dtype="f32")

    artifact = bb.compile(
        extended_math_wrapper,
        src,
        other,
        dst_exp,
        dst_log,
        dst_log2,
        dst_log10,
        dst_cos,
        dst_erf,
        dst_atan2,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )
    artifact(src, other, dst_exp, dst_log, dst_log2, dst_log10, dst_cos, dst_erf, dst_atan2)

    assert dst_exp.tolist() == pytest.approx([math.exp(1.0), math.exp(2.0), math.exp(4.0)], rel=1e-5, abs=1e-5)
    assert dst_log.tolist() == pytest.approx([math.log(1.0), math.log(2.0), math.log(4.0)], rel=1e-5, abs=1e-5)
    assert dst_log2.tolist() == pytest.approx([math.log2(1.0), math.log2(2.0), math.log2(4.0)], rel=1e-5, abs=1e-5)
    assert dst_log10.tolist() == pytest.approx([math.log10(1.0), math.log10(2.0), math.log10(4.0)], rel=1e-5, abs=1e-5)
    assert dst_cos.tolist() == pytest.approx([math.cos(1.0), math.cos(2.0), math.cos(4.0)], rel=1e-5, abs=1e-5)
    assert dst_erf.tolist() == pytest.approx([math.erf(1.0), math.erf(2.0), math.erf(4.0)], rel=1e-5, abs=1e-5)
    assert dst_atan2.tolist() == pytest.approx(
        [math.atan2(1.0, 1.0), math.atan2(2.0, 2.0), math.atan2(4.0, 8.0)],
        rel=1e-5,
        abs=1e-5,
    )
