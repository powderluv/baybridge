import math
import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.jit
def type_conversion_demo() -> None:
    x = bb.Int32(42)
    y = x.to(bb.Float32)
    bb.printf("Int32({}) => Float32({})", x, y)

    a = bb.Float32(3.14)
    b = a.to(bb.Int32)
    bb.printf("Float32({}) => Int32({})", a, b)

    c = bb.Int32(127)
    d = c.to(bb.Int8)
    bb.printf("Int32({}) => Int8({})", c, d)

    e = bb.Int32(300)
    f = e.to(bb.Int8)
    bb.printf("Int32({}) => Int8({}) (truncated due to range limitation)", e, f)


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def scalar_operator_kernel(out_i32: bb.Tensor, out_f32: bb.Tensor):
    a = bb.Int32(10)
    b = bb.Int32(3)
    x = bb.Float32(5.5)

    out_i32[0] = a + b
    out_i32[1] = a / b
    out_i32[2] = a & b
    out_i32[3] = -a
    out_i32[4] = ~a
    out_i32[5] = bb.Float32(3.14).to(bb.Int32)
    out_i32[6] = bb.Int32(300).to(bb.Int8).to(bb.Int32)

    out_f32[0] = x * 2
    out_f32[1] = a + x
    out_f32[2] = x / bb.Float32(2.0)
    out_f32[3] = bb.Int32(42).to(bb.Float32)


def test_typed_scalar_runtime_conversions(capsys) -> None:
    type_conversion_demo()

    assert capsys.readouterr().out.splitlines() == [
        "Int32(42) => Float32(42.0)",
        "Float32(3.14) => Int32(3)",
        "Int32(127) => Int8(127)",
        "Int32(300) => Int8(44) (truncated due to range limitation)",
    ]


def test_typed_scalar_runtime_operators() -> None:
    a = bb.Int32(10)
    b = bb.Int32(3)
    x = bb.Float32(5.5)

    assert isinstance(a, bb.RuntimeScalar)
    assert (a + b).dtype == "i32"
    assert int(a + b) == 13
    assert float(x * 2) == 11.0
    assert (a + x).dtype == "f32"
    assert math.isclose(float(a + x), 15.5)
    assert int(a / b) == 3
    assert math.isclose(float(x / bb.Float32(2.0)), 2.75)
    assert bool(a > b) is True
    assert int(a & b) == 2
    assert int(-a) == -10
    assert int(~a) == -11


def test_scalar_operator_kernel_lowers_to_hip_cpp(tmp_path: Path) -> None:
    out_i32 = bb.zeros((7,), dtype="i32")
    out_f32 = bb.zeros((4,), dtype="f32")

    artifact = bb.compile(
        scalar_operator_kernel,
        out_i32,
        out_f32,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )

    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "const std::int32_t div_" in text
    assert "const std::int32_t bitand_" in text
    assert "const std::int32_t neg_" in text
    assert "const std::int32_t bitnot_" in text
    assert "static_cast<std::int32_t>(" in text
    assert "static_cast<std::uint8_t>(" in text


def test_scalar_operator_kernel_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    out_i32 = bb.zeros((7,), dtype="i32")
    out_f32 = bb.zeros((4,), dtype="f32")
    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")

    artifact = bb.compile(
        scalar_operator_kernel,
        out_i32,
        out_f32,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(out_i32, out_f32)

    assert out_i32.tolist() == [13, 3, 2, -10, -11, 3, 44]
    assert out_f32.tolist() == pytest.approx([11.0, 15.5, 2.75, 42.0])
