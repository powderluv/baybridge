from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tool_module(name: str):
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    module_path = tools_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aster_benchmark_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    copy_payload = kernels.aster_dense_copy_f32_args()
    add_payload = kernels.aster_dense_add_f32_args()
    copy_i32_payload = kernels.aster_dense_copy_i32_args()
    add_i32_payload = kernels.aster_dense_add_i32_args()
    copy_f16_payload = kernels.aster_dense_copy_f16_args()
    broadcast_add_payload = kernels.aster_broadcast_add_f32_args()
    broadcast_add_i32_payload = kernels.aster_broadcast_add_i32_args()
    mfma_f16_payload = kernels.aster_mfma_f16_gemm_args()
    mfma_bf16_payload = kernels.aster_mfma_bf16_gemm_args()
    mfma_fp8_payload = kernels.aster_mfma_fp8_gemm_args()
    mfma_bf8_payload = kernels.aster_mfma_bf8_gemm_args()
    mfma_fp8_bf8_payload = kernels.aster_mfma_fp8_bf8_gemm_args()
    mfma_bf8_fp8_payload = kernels.aster_mfma_bf8_fp8_gemm_args()
    flydsl_sin_payload = kernels.flydsl_unary_sin_f32_args()
    flydsl_rsqrt_payload = kernels.flydsl_unary_rsqrt_f32_args()
    indexed_sub_payload = kernels.indexed_sub_f32_args()
    indexed_mul_payload = kernels.indexed_mul_f32_args()
    indexed_div_payload = kernels.indexed_div_f32_args()
    flydsl_broadcast_payload = kernels.flydsl_broadcast_add_2d_args()
    flydsl_reduce_payload = kernels.flydsl_reduce_add_2d_args()
    flydsl_unary_2d_payload = kernels.flydsl_unary_math_2d_args()
    flydsl_shared_payload = kernels.flydsl_shared_stage_f32_args()
    flydsl_factory_payload = kernels.flydsl_tensor_factory_2d_args()

    copy_args = copy_payload["args"]
    add_args = add_payload["args"]
    copy_i32_args = copy_i32_payload["args"]
    add_i32_args = add_i32_payload["args"]
    copy_f16_args = copy_f16_payload["args"]
    broadcast_add_args = broadcast_add_payload["args"]
    broadcast_add_i32_args = broadcast_add_i32_payload["args"]
    mfma_f16_args = mfma_f16_payload["args"]
    mfma_bf16_args = mfma_bf16_payload["args"]
    mfma_fp8_args = mfma_fp8_payload["args"]
    mfma_bf8_args = mfma_bf8_payload["args"]
    mfma_fp8_bf8_args = mfma_fp8_bf8_payload["args"]
    mfma_bf8_fp8_args = mfma_bf8_fp8_payload["args"]
    flydsl_sin_args = flydsl_sin_payload["args"]
    flydsl_rsqrt_args = flydsl_rsqrt_payload["args"]
    indexed_sub_args = indexed_sub_payload["args"]
    indexed_mul_args = indexed_mul_payload["args"]
    indexed_div_args = indexed_div_payload["args"]
    flydsl_broadcast_args = flydsl_broadcast_payload["args"]
    flydsl_reduce_args = flydsl_reduce_payload["args"]
    flydsl_unary_2d_args = flydsl_unary_2d_payload["args"]
    flydsl_shared_args = flydsl_shared_payload["args"]
    flydsl_factory_args = flydsl_factory_payload["args"]

    assert len(copy_args) == 2
    assert copy_args[0].shape == (kernels.ASTER_POINTWISE_N,)
    assert copy_args[1].shape == (kernels.ASTER_POINTWISE_N,)
    assert copy_payload["result_indices"] == ()

    assert len(add_args) == 3
    assert add_args[0].shape == (kernels.ASTER_POINTWISE_N,)
    assert add_args[1].shape == (kernels.ASTER_POINTWISE_N,)
    assert add_args[2].shape == (kernels.ASTER_POINTWISE_N,)
    assert add_payload["result_indices"] == ()

    assert len(copy_i32_args) == 2
    assert copy_i32_args[0].shape == (kernels.ASTER_POINTWISE_N,)
    assert copy_i32_args[1].shape == (kernels.ASTER_POINTWISE_N,)
    assert str(copy_i32_args[0].dtype) == "i32"
    assert str(copy_i32_args[1].dtype) == "i32"
    assert copy_i32_payload["result_indices"] == ()

    assert len(add_i32_args) == 3
    assert add_i32_args[0].shape == (kernels.ASTER_POINTWISE_N,)
    assert add_i32_args[1].shape == (kernels.ASTER_POINTWISE_N,)
    assert add_i32_args[2].shape == (kernels.ASTER_POINTWISE_N,)
    assert str(add_i32_args[0].dtype) == "i32"
    assert str(add_i32_args[1].dtype) == "i32"
    assert str(add_i32_args[2].dtype) == "i32"
    assert add_i32_payload["result_indices"] == ()

    assert len(copy_f16_args) == 2
    assert copy_f16_args[0].shape == (kernels.ASTER_POINTWISE_N,)
    assert copy_f16_args[1].shape == (kernels.ASTER_POINTWISE_N,)
    assert str(copy_f16_args[0].dtype) == "f16"
    assert str(copy_f16_args[1].dtype) == "f16"
    assert copy_f16_payload["result_indices"] == ()

    assert len(broadcast_add_args) == 3
    assert broadcast_add_args[0].shape == (kernels.ASTER_POINTWISE_N,)
    assert broadcast_add_args[1].shape == (1,)
    assert broadcast_add_args[2].shape == (kernels.ASTER_POINTWISE_N,)
    assert str(broadcast_add_args[0].dtype) == "f32"
    assert str(broadcast_add_args[1].dtype) == "f32"
    assert str(broadcast_add_args[2].dtype) == "f32"
    assert broadcast_add_payload["result_indices"] == ()

    assert len(broadcast_add_i32_args) == 3
    assert broadcast_add_i32_args[0].shape == (kernels.ASTER_POINTWISE_N,)
    assert broadcast_add_i32_args[1].shape == (1,)
    assert broadcast_add_i32_args[2].shape == (kernels.ASTER_POINTWISE_N,)
    assert str(broadcast_add_i32_args[0].dtype) == "i32"
    assert str(broadcast_add_i32_args[1].dtype) == "i32"
    assert str(broadcast_add_i32_args[2].dtype) == "i32"
    assert broadcast_add_i32_payload["result_indices"] == ()

    assert len(mfma_f16_args) == 3
    assert mfma_f16_args[0].shape == (16, 16)
    assert mfma_f16_args[1].shape == (16, 16)
    assert mfma_f16_args[2].shape == (16, 16)
    assert str(mfma_f16_args[0].dtype) == "f16"
    assert str(mfma_f16_args[1].dtype) == "f16"
    assert str(mfma_f16_args[2].dtype) == "f32"
    assert mfma_f16_payload["result_indices"] == ()

    assert len(mfma_bf16_args) == 3
    assert mfma_bf16_args[0].shape == (16, 16)
    assert mfma_bf16_args[1].shape == (16, 16)
    assert mfma_bf16_args[2].shape == (16, 16)
    assert str(mfma_bf16_args[0].dtype) == "bf16"
    assert str(mfma_bf16_args[1].dtype) == "bf16"
    assert str(mfma_bf16_args[2].dtype) == "f32"
    assert mfma_bf16_payload["result_indices"] == ()

    assert len(mfma_fp8_args) == 3
    assert mfma_fp8_args[0].shape == (16, 32)
    assert mfma_fp8_args[1].shape == (32, 16)
    assert mfma_fp8_args[2].shape == (16, 16)
    assert str(mfma_fp8_args[0].dtype) == "fp8"
    assert str(mfma_fp8_args[1].dtype) == "fp8"
    assert str(mfma_fp8_args[2].dtype) == "f32"
    assert mfma_fp8_payload["result_indices"] == ()

    assert len(mfma_bf8_args) == 3
    assert mfma_bf8_args[0].shape == (16, 32)
    assert mfma_bf8_args[1].shape == (32, 16)
    assert mfma_bf8_args[2].shape == (16, 16)
    assert str(mfma_bf8_args[0].dtype) == "bf8"
    assert str(mfma_bf8_args[1].dtype) == "bf8"
    assert str(mfma_bf8_args[2].dtype) == "f32"
    assert mfma_bf8_payload["result_indices"] == ()

    assert len(mfma_fp8_bf8_args) == 3
    assert mfma_fp8_bf8_args[0].shape == (16, 32)
    assert mfma_fp8_bf8_args[1].shape == (32, 16)
    assert mfma_fp8_bf8_args[2].shape == (16, 16)
    assert str(mfma_fp8_bf8_args[0].dtype) == "fp8"
    assert str(mfma_fp8_bf8_args[1].dtype) == "bf8"
    assert str(mfma_fp8_bf8_args[2].dtype) == "f32"
    assert mfma_fp8_bf8_payload["result_indices"] == ()

    assert len(mfma_bf8_fp8_args) == 3
    assert mfma_bf8_fp8_args[0].shape == (16, 32)
    assert mfma_bf8_fp8_args[1].shape == (32, 16)
    assert mfma_bf8_fp8_args[2].shape == (16, 16)
    assert str(mfma_bf8_fp8_args[0].dtype) == "bf8"
    assert str(mfma_bf8_fp8_args[1].dtype) == "fp8"
    assert str(mfma_bf8_fp8_args[2].dtype) == "f32"
    assert mfma_bf8_fp8_payload["result_indices"] == ()

    assert len(flydsl_sin_args) == 2
    assert flydsl_sin_args[0].shape == (kernels.FLYDSL_MICRO_N,)
    assert flydsl_sin_args[1].shape == (kernels.FLYDSL_MICRO_N,)
    assert flydsl_sin_payload["result_indices"] == ()

    assert len(flydsl_rsqrt_args) == 2
    assert flydsl_rsqrt_args[0].shape == (kernels.FLYDSL_MICRO_N,)
    assert flydsl_rsqrt_args[1].shape == (kernels.FLYDSL_MICRO_N,)
    assert flydsl_rsqrt_payload["result_indices"] == ()

    assert len(indexed_sub_args) == 3
    assert indexed_sub_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_sub_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_sub_args[2].shape == (kernels.POINTWISE_N,)
    assert indexed_sub_payload["result_indices"] == ()

    assert len(indexed_mul_args) == 3
    assert indexed_mul_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_mul_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_mul_args[2].shape == (kernels.POINTWISE_N,)
    assert indexed_mul_payload["result_indices"] == ()

    assert len(indexed_div_args) == 3
    assert indexed_div_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_div_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_div_args[2].shape == (kernels.POINTWISE_N,)
    assert indexed_div_payload["result_indices"] == ()

    assert len(flydsl_broadcast_args) == 3
    assert flydsl_broadcast_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert flydsl_broadcast_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert flydsl_broadcast_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert flydsl_broadcast_payload["result_indices"] == ()

    assert len(flydsl_reduce_args) == 3
    assert flydsl_reduce_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert flydsl_reduce_args[1].shape == (1,)
    assert flydsl_reduce_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert flydsl_reduce_payload["result_indices"] == ()

    assert len(flydsl_unary_2d_args) == 5
    for arg in flydsl_unary_2d_args:
        assert arg.shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(arg.dtype) == "f32"
    assert flydsl_unary_2d_payload["result_indices"] == ()

    assert len(flydsl_shared_args) == 2
    assert flydsl_shared_args[0].shape == (kernels.FLYDSL_SHARED_N,)
    assert flydsl_shared_args[1].shape == (kernels.FLYDSL_SHARED_N,)
    assert flydsl_shared_payload["result_indices"] == ()

    assert len(flydsl_factory_args) == 3
    for arg in flydsl_factory_args:
        assert arg.shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(arg.dtype) == "f32"
    assert flydsl_factory_payload["result_indices"] == ()


def test_compare_backends_uses_hip_synchronizer_for_aster_exec(monkeypatch) -> None:
    compare_backends = _load_tool_module("compare_backends")

    calls: list[str] = []

    class FakeHipRuntime:
        def synchronize(self) -> None:
            calls.append("sync")

    monkeypatch.setattr("baybridge.hip_runtime.HipRuntime", FakeHipRuntime)

    synchronize = compare_backends._make_execution_synchronizer("aster_exec", ())

    assert callable(synchronize)
    synchronize()
    assert calls == ["sync"]


def test_compare_backends_returns_no_synchronizer_for_non_exec_backend() -> None:
    compare_backends = _load_tool_module("compare_backends")

    assert compare_backends._make_execution_synchronizer("gpu_mlir", ()) is None


def test_compare_backends_summarizes_cold_and_warm_timings() -> None:
    compare_backends = _load_tool_module("compare_backends")

    summary = compare_backends._summarize_timings_ms([10.0, 4.0, 6.0, 8.0])

    assert summary["cold_ms"] == 10.0
    assert summary["warm_timings_ms"] == [4.0, 6.0, 8.0]
    assert summary["warm_median_ms"] == 6.0


def test_report_cold_warm_summarizes_successful_backend_result() -> None:
    report_cold_warm = _load_tool_module("report_cold_warm")

    payload = {
        "environment": {"target": "gfx950"},
        "results": [
            {
                "backend": "flydsl_exec",
                "resolved_backend": "flydsl_exec",
                "target": "gfx950",
                "status": "ok",
                "execute_status": "ok",
                "timings_ms": [100.0, 10.0, 12.0],
                "cold_ms": 100.0,
                "warm_median_ms": 11.0,
                "warm_timings_ms": [10.0, 12.0],
            }
        ],
    }

    summary = report_cold_warm._summarize_compare_payload(payload)

    assert summary["environment"] == {"target": "gfx950"}
    assert summary["results"] == [
        {
            "backend": "flydsl_exec",
            "resolved_backend": "flydsl_exec",
            "target": "gfx950",
            "status": "ok",
            "execute_status": "ok",
            "cold_ms": 100.0,
            "warm_median_ms": 11.0,
            "warm_timings_ms": [10.0, 12.0],
            "repeat": 3,
        }
    ]
    assert report_cold_warm._format_summary(summary) == "target=gfx950\nflydsl_exec cold=100.00 warm_median=11.00 repeat=3"


def test_report_cold_warm_adds_baseline_ratios() -> None:
    report_cold_warm = _load_tool_module("report_cold_warm")

    payload = {
        "results": [
            {
                "backend": "hipcc_exec",
                "resolved_backend": "hipcc_exec",
                "target": "gfx950",
                "status": "ok",
                "execute_status": "ok",
                "timings_ms": [40.0, 4.0, 6.0],
                "cold_ms": 40.0,
                "warm_median_ms": 5.0,
                "warm_timings_ms": [4.0, 6.0],
            },
            {
                "backend": "flydsl_exec",
                "resolved_backend": "flydsl_exec",
                "target": "gfx950",
                "status": "ok",
                "execute_status": "ok",
                "timings_ms": [100.0, 10.0, 12.0],
                "cold_ms": 100.0,
                "warm_median_ms": 11.0,
                "warm_timings_ms": [10.0, 12.0],
            },
        ]
    }

    summary = report_cold_warm._summarize_compare_payload(payload, baseline_backend="hipcc_exec")

    assert summary["baseline_backend"] == "hipcc_exec"
    assert summary["results"][0]["cold_ratio"] == 1.0
    assert summary["results"][0]["warm_median_ratio"] == 1.0
    assert summary["results"][1]["cold_ratio"] == 2.5
    assert summary["results"][1]["warm_median_ratio"] == 2.2
    assert report_cold_warm._format_summary(summary) == (
        "baseline=hipcc_exec\n"
        "hipcc_exec cold=40.00 warm_median=5.00 repeat=3 cold_ratio=1.00 warm_ratio=1.00\n"
        "flydsl_exec cold=100.00 warm_median=11.00 repeat=3 cold_ratio=2.50 warm_ratio=2.20"
    )


def test_report_cold_warm_requires_successful_baseline_backend() -> None:
    report_cold_warm = _load_tool_module("report_cold_warm")

    with pytest.raises(SystemExit, match="baseline backend 'hipcc_exec' was not found in successful results"):
        report_cold_warm._summarize_compare_payload(
            [{"backend": "flydsl_exec", "status": "ok", "execute_status": "ok", "cold_ms": 1.0, "warm_median_ms": 1.0}],
            baseline_backend="hipcc_exec",
        )


def test_report_cold_warm_formats_skipped_backend_result() -> None:
    report_cold_warm = _load_tool_module("report_cold_warm")

    summary = report_cold_warm._summarize_compare_payload(
        [
            {
                "backend": "flydsl_exec",
                "status": "ok",
                "execute_status": "skipped_unvalidated_real_flydsl_exec",
                "execute_note": "gate still enabled",
            }
        ]
    )

    assert summary["results"] == [
        {
            "backend": "flydsl_exec",
            "resolved_backend": None,
            "target": None,
            "status": "ok",
            "execute_status": "skipped_unvalidated_real_flydsl_exec",
            "execute_note": "gate still enabled",
        }
    ]
    assert report_cold_warm._format_summary(summary) == (
        "flydsl_exec skipped_unvalidated_real_flydsl_exec gate still enabled"
    )


def test_backend_benchmark_kernels_exports_sub_and_mul_microbench_kernels() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    assert callable(kernels.dense_sub_f32_kernel)
    assert callable(kernels.dense_mul_f32_kernel)
    assert callable(kernels.indexed_sub_f32_kernel)
    assert callable(kernels.indexed_mul_f32_kernel)
    assert callable(kernels.indexed_div_f32_kernel)
    assert callable(kernels.flydsl_unary_sin_f32_kernel)
    assert callable(kernels.flydsl_unary_rsqrt_f32_kernel)
    assert callable(kernels.flydsl_broadcast_add_2d_kernel)
    assert callable(kernels.flydsl_reduce_add_2d_kernel)
    assert callable(kernels.flydsl_unary_math_2d_kernel)
    assert callable(kernels.flydsl_shared_stage_f32_kernel)
    assert callable(kernels.flydsl_tensor_factory_2d_kernel)
    assert callable(kernels.aster_mfma_f16_gemm_kernel)
    assert callable(kernels.aster_mfma_bf16_gemm_kernel)
    assert callable(kernels.aster_mfma_fp8_gemm_kernel)
    assert callable(kernels.aster_mfma_bf8_gemm_kernel)
    assert callable(kernels.aster_mfma_fp8_bf8_gemm_kernel)
    assert callable(kernels.aster_mfma_bf8_fp8_gemm_kernel)
