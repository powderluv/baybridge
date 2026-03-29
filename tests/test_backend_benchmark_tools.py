from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import baybridge as bb


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
    indexed_copy_payload = kernels.indexed_copy_f32_args()
    indexed_copy_cuda_handle_payload = kernels.indexed_copy_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_copy_i32_payload = kernels.indexed_copy_i32_args()
    indexed_copy_i32_cuda_handle_payload = kernels.indexed_copy_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_add_cuda_handle_payload = kernels.indexed_add_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_add_cuda_dlpack_payload = kernels.indexed_add_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_add_i32_payload = kernels.indexed_add_i32_args()
    indexed_add_i32_cuda_handle_payload = kernels.indexed_add_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_add_i32_cuda_dlpack_payload = kernels.indexed_add_i32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_scalar_broadcast_payload = kernels.indexed_scalar_broadcast_add_f32_args()
    indexed_scalar_broadcast_cuda_handle_payload = kernels.indexed_scalar_broadcast_add_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_scalar_broadcast_i32_payload = kernels.indexed_scalar_broadcast_add_i32_args()
    indexed_scalar_broadcast_i32_cuda_handle_payload = kernels.indexed_scalar_broadcast_add_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_sqrt_payload = kernels.indexed_sqrt_f32_args()
    indexed_sqrt_cuda_handle_payload = kernels.indexed_sqrt_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_rsqrt_payload = kernels.indexed_rsqrt_f32_args()
    indexed_rsqrt_cuda_handle_payload = kernels.indexed_rsqrt_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_sqrt_payload = kernels.direct_sqrt_f32_args()
    direct_sqrt_cuda_handle_payload = kernels.direct_sqrt_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_rsqrt_payload = kernels.direct_rsqrt_f32_args()
    direct_rsqrt_cuda_handle_payload = kernels.direct_rsqrt_f32_cuda_handle_args(backend_name="ptx_exec")
    reduce_add_payload = kernels.reduce_add_f32_args()
    reduce_add_cuda_handle_payload = kernels.reduce_add_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_add_payload = kernels.parallel_reduce_add_f32_args()
    parallel_reduce_add_cuda_handle_payload = kernels.parallel_reduce_add_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_add_i32_payload = kernels.parallel_reduce_add_i32_args()
    parallel_reduce_add_i32_cuda_handle_payload = kernels.parallel_reduce_add_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_add_2d_bundle_payload = kernels.parallel_reduce_add_2d_bundle_f32_args()
    parallel_reduce_add_2d_bundle_cuda_handle_payload = kernels.parallel_reduce_add_2d_bundle_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_add_2d_bundle_i32_payload = kernels.parallel_reduce_add_2d_bundle_i32_args()
    parallel_reduce_add_2d_bundle_i32_cuda_handle_payload = kernels.parallel_reduce_add_2d_bundle_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_mul_2d_bundle_payload = kernels.parallel_reduce_mul_2d_bundle_f32_args()
    parallel_reduce_mul_2d_bundle_cuda_handle_payload = kernels.parallel_reduce_mul_2d_bundle_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_max_2d_bundle_payload = kernels.parallel_reduce_max_2d_bundle_f32_args()
    parallel_reduce_max_2d_bundle_cuda_handle_payload = kernels.parallel_reduce_max_2d_bundle_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_min_2d_bundle_payload = kernels.parallel_reduce_min_2d_bundle_f32_args()
    parallel_reduce_min_2d_bundle_cuda_handle_payload = kernels.parallel_reduce_min_2d_bundle_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_mul_2d_bundle_i32_payload = kernels.parallel_reduce_mul_2d_bundle_i32_args()
    parallel_reduce_mul_2d_bundle_i32_cuda_handle_payload = kernels.parallel_reduce_mul_2d_bundle_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_max_2d_bundle_i32_payload = kernels.parallel_reduce_max_2d_bundle_i32_args()
    parallel_reduce_max_2d_bundle_i32_cuda_handle_payload = kernels.parallel_reduce_max_2d_bundle_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_reduce_min_2d_bundle_i32_payload = kernels.parallel_reduce_min_2d_bundle_i32_args()
    parallel_reduce_min_2d_bundle_i32_cuda_handle_payload = kernels.parallel_reduce_min_2d_bundle_i32_cuda_handle_args(backend_name="ptx_exec")
    tensor_factory_2d_payload = kernels.tensor_factory_2d_f32_args()
    tensor_factory_2d_cuda_handle_payload = kernels.tensor_factory_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    tensor_factory_2d_i32_payload = kernels.tensor_factory_2d_i32_args()
    tensor_factory_2d_i32_cuda_handle_payload = kernels.tensor_factory_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    reduce_add_2d_bundle_payload = kernels.reduce_add_2d_bundle_f32_args()
    reduce_add_2d_bundle_cuda_handle_payload = kernels.reduce_add_2d_bundle_f32_cuda_handle_args(backend_name="ptx_exec")
    reduce_add_2d_bundle_i32_payload = kernels.reduce_add_2d_bundle_i32_args()
    reduce_add_2d_bundle_i32_cuda_handle_payload = kernels.reduce_add_2d_bundle_i32_cuda_handle_args(backend_name="ptx_exec")
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
    indexed_copy_args = indexed_copy_payload["args"]
    indexed_copy_cuda_handle_compile_args = indexed_copy_cuda_handle_payload["compile_args"]
    indexed_copy_cuda_handle_run_args = indexed_copy_cuda_handle_payload["run_args"]
    indexed_copy_i32_args = indexed_copy_i32_payload["args"]
    indexed_copy_i32_cuda_handle_compile_args = indexed_copy_i32_cuda_handle_payload["compile_args"]
    indexed_copy_i32_cuda_handle_run_args = indexed_copy_i32_cuda_handle_payload["run_args"]
    indexed_add_cuda_handle_compile_args = indexed_add_cuda_handle_payload["compile_args"]
    indexed_add_cuda_handle_run_args = indexed_add_cuda_handle_payload["run_args"]
    indexed_add_cuda_dlpack_compile_args = indexed_add_cuda_dlpack_payload["compile_args"]
    indexed_add_cuda_dlpack_run_args = indexed_add_cuda_dlpack_payload["run_args"]
    indexed_add_i32_args = indexed_add_i32_payload["args"]
    indexed_add_i32_cuda_handle_compile_args = indexed_add_i32_cuda_handle_payload["compile_args"]
    indexed_add_i32_cuda_handle_run_args = indexed_add_i32_cuda_handle_payload["run_args"]
    indexed_add_i32_cuda_dlpack_compile_args = indexed_add_i32_cuda_dlpack_payload["compile_args"]
    indexed_add_i32_cuda_dlpack_run_args = indexed_add_i32_cuda_dlpack_payload["run_args"]
    indexed_scalar_broadcast_args = indexed_scalar_broadcast_payload["args"]
    indexed_scalar_broadcast_cuda_handle_compile_args = indexed_scalar_broadcast_cuda_handle_payload["compile_args"]
    indexed_scalar_broadcast_cuda_handle_run_args = indexed_scalar_broadcast_cuda_handle_payload["run_args"]
    indexed_scalar_broadcast_i32_args = indexed_scalar_broadcast_i32_payload["args"]
    indexed_scalar_broadcast_i32_cuda_handle_compile_args = indexed_scalar_broadcast_i32_cuda_handle_payload["compile_args"]
    indexed_scalar_broadcast_i32_cuda_handle_run_args = indexed_scalar_broadcast_i32_cuda_handle_payload["run_args"]
    indexed_sqrt_args = indexed_sqrt_payload["args"]
    indexed_sqrt_cuda_handle_compile_args = indexed_sqrt_cuda_handle_payload["compile_args"]
    indexed_sqrt_cuda_handle_run_args = indexed_sqrt_cuda_handle_payload["run_args"]
    indexed_rsqrt_args = indexed_rsqrt_payload["args"]
    indexed_rsqrt_cuda_handle_compile_args = indexed_rsqrt_cuda_handle_payload["compile_args"]
    indexed_rsqrt_cuda_handle_run_args = indexed_rsqrt_cuda_handle_payload["run_args"]
    direct_sqrt_args = direct_sqrt_payload["args"]
    direct_sqrt_cuda_handle_compile_args = direct_sqrt_cuda_handle_payload["compile_args"]
    direct_sqrt_cuda_handle_run_args = direct_sqrt_cuda_handle_payload["run_args"]
    direct_rsqrt_args = direct_rsqrt_payload["args"]
    direct_rsqrt_cuda_handle_compile_args = direct_rsqrt_cuda_handle_payload["compile_args"]
    direct_rsqrt_cuda_handle_run_args = direct_rsqrt_cuda_handle_payload["run_args"]
    reduce_add_args = reduce_add_payload["args"]
    reduce_add_cuda_handle_compile_args = reduce_add_cuda_handle_payload["compile_args"]
    reduce_add_cuda_handle_run_args = reduce_add_cuda_handle_payload["run_args"]
    parallel_reduce_add_args = parallel_reduce_add_payload["args"]
    parallel_reduce_add_cuda_handle_compile_args = parallel_reduce_add_cuda_handle_payload["compile_args"]
    parallel_reduce_add_cuda_handle_run_args = parallel_reduce_add_cuda_handle_payload["run_args"]
    parallel_reduce_add_i32_args = parallel_reduce_add_i32_payload["args"]
    parallel_reduce_add_i32_cuda_handle_compile_args = parallel_reduce_add_i32_cuda_handle_payload["compile_args"]
    parallel_reduce_add_i32_cuda_handle_run_args = parallel_reduce_add_i32_cuda_handle_payload["run_args"]
    parallel_reduce_add_2d_bundle_args = parallel_reduce_add_2d_bundle_payload["args"]
    parallel_reduce_add_2d_bundle_cuda_handle_compile_args = parallel_reduce_add_2d_bundle_cuda_handle_payload["compile_args"]
    parallel_reduce_add_2d_bundle_cuda_handle_run_args = parallel_reduce_add_2d_bundle_cuda_handle_payload["run_args"]
    parallel_reduce_add_2d_bundle_i32_args = parallel_reduce_add_2d_bundle_i32_payload["args"]
    parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args = parallel_reduce_add_2d_bundle_i32_cuda_handle_payload["compile_args"]
    parallel_reduce_add_2d_bundle_i32_cuda_handle_run_args = parallel_reduce_add_2d_bundle_i32_cuda_handle_payload["run_args"]
    parallel_reduce_mul_2d_bundle_args = parallel_reduce_mul_2d_bundle_payload["args"]
    parallel_reduce_mul_2d_bundle_cuda_handle_compile_args = parallel_reduce_mul_2d_bundle_cuda_handle_payload["compile_args"]
    parallel_reduce_mul_2d_bundle_cuda_handle_run_args = parallel_reduce_mul_2d_bundle_cuda_handle_payload["run_args"]
    parallel_reduce_max_2d_bundle_args = parallel_reduce_max_2d_bundle_payload["args"]
    parallel_reduce_max_2d_bundle_cuda_handle_compile_args = parallel_reduce_max_2d_bundle_cuda_handle_payload["compile_args"]
    parallel_reduce_max_2d_bundle_cuda_handle_run_args = parallel_reduce_max_2d_bundle_cuda_handle_payload["run_args"]
    parallel_reduce_min_2d_bundle_args = parallel_reduce_min_2d_bundle_payload["args"]
    parallel_reduce_min_2d_bundle_cuda_handle_compile_args = parallel_reduce_min_2d_bundle_cuda_handle_payload["compile_args"]
    parallel_reduce_min_2d_bundle_cuda_handle_run_args = parallel_reduce_min_2d_bundle_cuda_handle_payload["run_args"]
    parallel_reduce_mul_2d_bundle_i32_args = parallel_reduce_mul_2d_bundle_i32_payload["args"]
    parallel_reduce_mul_2d_bundle_i32_cuda_handle_compile_args = parallel_reduce_mul_2d_bundle_i32_cuda_handle_payload["compile_args"]
    parallel_reduce_mul_2d_bundle_i32_cuda_handle_run_args = parallel_reduce_mul_2d_bundle_i32_cuda_handle_payload["run_args"]
    parallel_reduce_max_2d_bundle_i32_args = parallel_reduce_max_2d_bundle_i32_payload["args"]
    parallel_reduce_max_2d_bundle_i32_cuda_handle_compile_args = parallel_reduce_max_2d_bundle_i32_cuda_handle_payload["compile_args"]
    parallel_reduce_max_2d_bundle_i32_cuda_handle_run_args = parallel_reduce_max_2d_bundle_i32_cuda_handle_payload["run_args"]
    parallel_reduce_min_2d_bundle_i32_args = parallel_reduce_min_2d_bundle_i32_payload["args"]
    parallel_reduce_min_2d_bundle_i32_cuda_handle_compile_args = parallel_reduce_min_2d_bundle_i32_cuda_handle_payload["compile_args"]
    parallel_reduce_min_2d_bundle_i32_cuda_handle_run_args = parallel_reduce_min_2d_bundle_i32_cuda_handle_payload["run_args"]
    tensor_factory_2d_args = tensor_factory_2d_payload["args"]
    tensor_factory_2d_cuda_handle_compile_args = tensor_factory_2d_cuda_handle_payload["compile_args"]
    tensor_factory_2d_cuda_handle_run_args = tensor_factory_2d_cuda_handle_payload["run_args"]
    tensor_factory_2d_i32_args = tensor_factory_2d_i32_payload["args"]
    tensor_factory_2d_i32_cuda_handle_compile_args = tensor_factory_2d_i32_cuda_handle_payload["compile_args"]
    tensor_factory_2d_i32_cuda_handle_run_args = tensor_factory_2d_i32_cuda_handle_payload["run_args"]
    reduce_add_2d_bundle_args = reduce_add_2d_bundle_payload["args"]
    reduce_add_2d_bundle_cuda_handle_compile_args = reduce_add_2d_bundle_cuda_handle_payload["compile_args"]
    reduce_add_2d_bundle_cuda_handle_run_args = reduce_add_2d_bundle_cuda_handle_payload["run_args"]
    reduce_add_2d_bundle_i32_args = reduce_add_2d_bundle_i32_payload["args"]
    reduce_add_2d_bundle_i32_cuda_handle_compile_args = reduce_add_2d_bundle_i32_cuda_handle_payload["compile_args"]
    reduce_add_2d_bundle_i32_cuda_handle_run_args = reduce_add_2d_bundle_i32_cuda_handle_payload["run_args"]
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

    assert len(indexed_copy_args) == 2
    assert indexed_copy_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_copy_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_copy_payload["result_indices"] == ()

    assert len(indexed_copy_cuda_handle_compile_args) == 2
    assert indexed_copy_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_copy_cuda_handle_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert len(indexed_copy_cuda_handle_run_args) == 2
    assert indexed_copy_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_copy_i32_args) == 2
    assert indexed_copy_i32_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_copy_i32_args[1].shape == (kernels.POINTWISE_N,)
    assert str(indexed_copy_i32_args[0].dtype) == "i32"
    assert str(indexed_copy_i32_args[1].dtype) == "i32"
    assert indexed_copy_i32_payload["result_indices"] == ()

    assert len(indexed_copy_i32_cuda_handle_compile_args) == 2
    assert indexed_copy_i32_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_copy_i32_cuda_handle_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert str(indexed_copy_i32_cuda_handle_compile_args[0].dtype) == "i32"
    assert str(indexed_copy_i32_cuda_handle_compile_args[1].dtype) == "i32"
    assert len(indexed_copy_i32_cuda_handle_run_args) == 2
    assert indexed_copy_i32_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_add_cuda_handle_compile_args) == 3
    assert indexed_add_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_add_cuda_handle_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_add_cuda_handle_compile_args[2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_add_cuda_handle_run_args) == 3
    assert indexed_add_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_add_cuda_dlpack_compile_args) == 3
    assert indexed_add_cuda_dlpack_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_add_cuda_dlpack_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_add_cuda_dlpack_compile_args[2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_add_cuda_dlpack_run_args) == 3
    assert indexed_add_cuda_dlpack_payload["result_indices"] == ()

    assert len(indexed_add_i32_args) == 3
    assert indexed_add_i32_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_add_i32_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_add_i32_args[2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_add_i32_args[0].dtype) == "i32"
    assert str(indexed_add_i32_args[1].dtype) == "i32"
    assert str(indexed_add_i32_args[2].dtype) == "i32"
    assert indexed_add_i32_payload["result_indices"] == ()

    assert len(indexed_add_i32_cuda_handle_compile_args) == 3
    assert indexed_add_i32_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_add_i32_cuda_handle_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_add_i32_cuda_handle_compile_args[2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_add_i32_cuda_handle_compile_args[0].dtype) == "i32"
    assert str(indexed_add_i32_cuda_handle_compile_args[1].dtype) == "i32"
    assert str(indexed_add_i32_cuda_handle_compile_args[2].dtype) == "i32"
    assert len(indexed_add_i32_cuda_handle_run_args) == 3
    assert indexed_add_i32_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_add_i32_cuda_dlpack_compile_args) == 3
    assert indexed_add_i32_cuda_dlpack_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_add_i32_cuda_dlpack_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_add_i32_cuda_dlpack_compile_args[2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_add_i32_cuda_dlpack_compile_args[0].dtype) == "i32"
    assert str(indexed_add_i32_cuda_dlpack_compile_args[1].dtype) == "i32"
    assert str(indexed_add_i32_cuda_dlpack_compile_args[2].dtype) == "i32"
    assert len(indexed_add_i32_cuda_dlpack_run_args) == 3
    assert indexed_add_i32_cuda_dlpack_payload["result_indices"] == ()

    assert len(indexed_scalar_broadcast_args) == 3
    assert indexed_scalar_broadcast_args[0].shape == (kernels.POINTWISE_N,)
    assert float(indexed_scalar_broadcast_args[1]) == 1.5
    assert indexed_scalar_broadcast_args[2].shape == (kernels.POINTWISE_N,)
    assert indexed_scalar_broadcast_payload["result_indices"] == ()

    assert len(indexed_scalar_broadcast_cuda_handle_compile_args) == 3
    assert indexed_scalar_broadcast_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert float(indexed_scalar_broadcast_cuda_handle_compile_args[1]) == 1.5
    assert indexed_scalar_broadcast_cuda_handle_compile_args[2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_scalar_broadcast_cuda_handle_run_args) == 3
    assert indexed_scalar_broadcast_cuda_handle_run_args[1] == 1.5
    assert indexed_scalar_broadcast_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_scalar_broadcast_i32_args) == 3
    assert indexed_scalar_broadcast_i32_args[0].shape == (kernels.POINTWISE_N,)
    assert int(indexed_scalar_broadcast_i32_args[1]) == 7
    assert indexed_scalar_broadcast_i32_args[2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_scalar_broadcast_i32_args[0].dtype) == "i32"
    assert str(indexed_scalar_broadcast_i32_args[2].dtype) == "i32"
    assert indexed_scalar_broadcast_i32_payload["result_indices"] == ()

    assert len(indexed_scalar_broadcast_i32_cuda_handle_compile_args) == 3
    assert indexed_scalar_broadcast_i32_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert int(indexed_scalar_broadcast_i32_cuda_handle_compile_args[1]) == 7
    assert indexed_scalar_broadcast_i32_cuda_handle_compile_args[2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_scalar_broadcast_i32_cuda_handle_compile_args[0].dtype) == "i32"
    assert str(indexed_scalar_broadcast_i32_cuda_handle_compile_args[2].dtype) == "i32"
    assert len(indexed_scalar_broadcast_i32_cuda_handle_run_args) == 3
    assert indexed_scalar_broadcast_i32_cuda_handle_run_args[1] == 7
    assert indexed_scalar_broadcast_i32_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_sqrt_args) == 2
    assert indexed_sqrt_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_sqrt_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_sqrt_payload["result_indices"] == ()

    assert len(indexed_sqrt_cuda_handle_compile_args) == 2
    assert indexed_sqrt_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_sqrt_cuda_handle_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert len(indexed_sqrt_cuda_handle_run_args) == 2
    assert indexed_sqrt_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_rsqrt_args) == 2
    assert indexed_rsqrt_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_rsqrt_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_rsqrt_payload["result_indices"] == ()

    assert len(indexed_rsqrt_cuda_handle_compile_args) == 2
    assert indexed_rsqrt_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_rsqrt_cuda_handle_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert len(indexed_rsqrt_cuda_handle_run_args) == 2
    assert indexed_rsqrt_cuda_handle_payload["result_indices"] == ()

    assert len(direct_sqrt_args) == 2
    assert direct_sqrt_args[0].shape == (128,)
    assert direct_sqrt_args[1].shape == (128,)
    assert direct_sqrt_payload["result_indices"] == ()

    assert len(direct_sqrt_cuda_handle_compile_args) == 2
    assert direct_sqrt_cuda_handle_compile_args[0].shape == (128,)
    assert direct_sqrt_cuda_handle_compile_args[1].shape == (128,)
    assert len(direct_sqrt_cuda_handle_run_args) == 2
    assert direct_sqrt_cuda_handle_payload["result_indices"] == ()

    assert len(direct_rsqrt_args) == 2
    assert direct_rsqrt_args[0].shape == (128,)
    assert direct_rsqrt_args[1].shape == (128,)
    assert direct_rsqrt_payload["result_indices"] == ()

    assert len(direct_rsqrt_cuda_handle_compile_args) == 2
    assert direct_rsqrt_cuda_handle_compile_args[0].shape == (128,)
    assert direct_rsqrt_cuda_handle_compile_args[1].shape == (128,)
    assert len(direct_rsqrt_cuda_handle_run_args) == 2
    assert direct_rsqrt_cuda_handle_payload["result_indices"] == ()

    assert len(reduce_add_args) == 2
    assert reduce_add_args[0].shape == (kernels.POINTWISE_N,)
    assert reduce_add_args[1].shape == (1,)
    assert reduce_add_payload["result_indices"] == ()

    assert len(reduce_add_cuda_handle_compile_args) == 2
    assert reduce_add_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert reduce_add_cuda_handle_compile_args[1].shape == (1,)
    assert len(reduce_add_cuda_handle_run_args) == 2
    assert reduce_add_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_reduce_add_args) == 2
    assert parallel_reduce_add_args[0].shape == (kernels.POINTWISE_N,)
    assert parallel_reduce_add_args[1].shape == (1,)
    assert parallel_reduce_add_payload["result_indices"] == ()

    assert len(parallel_reduce_add_cuda_handle_compile_args) == 2
    assert parallel_reduce_add_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert parallel_reduce_add_cuda_handle_compile_args[1].shape == (1,)
    assert len(parallel_reduce_add_cuda_handle_run_args) == 2
    assert parallel_reduce_add_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_reduce_add_i32_args) == 2
    assert parallel_reduce_add_i32_args[0].shape == (kernels.POINTWISE_N,)
    assert parallel_reduce_add_i32_args[1].shape == (1,)
    assert str(parallel_reduce_add_i32_args[0].dtype) == "i32"
    assert str(parallel_reduce_add_i32_args[1].dtype) == "i32"
    assert parallel_reduce_add_i32_payload["result_indices"] == ()

    assert len(parallel_reduce_add_i32_cuda_handle_compile_args) == 2
    assert parallel_reduce_add_i32_cuda_handle_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert parallel_reduce_add_i32_cuda_handle_compile_args[1].shape == (1,)
    assert str(parallel_reduce_add_i32_cuda_handle_compile_args[0].dtype) == "i32"
    assert str(parallel_reduce_add_i32_cuda_handle_compile_args[1].dtype) == "i32"
    assert len(parallel_reduce_add_i32_cuda_handle_run_args) == 2
    assert parallel_reduce_add_i32_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_reduce_add_2d_bundle_args) == 4
    assert parallel_reduce_add_2d_bundle_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_reduce_add_2d_bundle_args[1].shape == (1,)
    assert parallel_reduce_add_2d_bundle_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_reduce_add_2d_bundle_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert parallel_reduce_add_2d_bundle_payload["result_indices"] == ()

    assert len(parallel_reduce_add_2d_bundle_cuda_handle_compile_args) == 4
    assert parallel_reduce_add_2d_bundle_cuda_handle_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_reduce_add_2d_bundle_cuda_handle_compile_args[1].shape == (1,)
    assert parallel_reduce_add_2d_bundle_cuda_handle_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_reduce_add_2d_bundle_cuda_handle_compile_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert len(parallel_reduce_add_2d_bundle_cuda_handle_run_args) == 4
    assert parallel_reduce_add_2d_bundle_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_reduce_add_2d_bundle_i32_args) == 4
    assert parallel_reduce_add_2d_bundle_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_reduce_add_2d_bundle_i32_args[1].shape == (1,)
    assert parallel_reduce_add_2d_bundle_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_reduce_add_2d_bundle_i32_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert str(parallel_reduce_add_2d_bundle_i32_args[0].dtype) == "i32"
    assert str(parallel_reduce_add_2d_bundle_i32_args[1].dtype) == "i32"
    assert str(parallel_reduce_add_2d_bundle_i32_args[2].dtype) == "i32"
    assert str(parallel_reduce_add_2d_bundle_i32_args[3].dtype) == "i32"
    assert parallel_reduce_add_2d_bundle_i32_payload["result_indices"] == ()

    assert len(parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args) == 4
    assert parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[1].shape == (1,)
    assert parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert str(parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[0].dtype) == "i32"
    assert str(parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[1].dtype) == "i32"
    assert str(parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[2].dtype) == "i32"
    assert str(parallel_reduce_add_2d_bundle_i32_cuda_handle_compile_args[3].dtype) == "i32"
    assert len(parallel_reduce_add_2d_bundle_i32_cuda_handle_run_args) == 4
    assert parallel_reduce_add_2d_bundle_i32_cuda_handle_payload["result_indices"] == ()

    def _assert_parallel_reduce_2d_bundle_payload(args, payload, *, dtype: str) -> None:
        assert len(args) == 4
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert args[1].shape == (1,)
        assert args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
        assert args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
        if dtype == "i32":
            assert str(args[0].dtype) == "i32"
            assert str(args[1].dtype) == "i32"
            assert str(args[2].dtype) == "i32"
            assert str(args[3].dtype) == "i32"
        assert payload["result_indices"] == ()

    def _assert_parallel_reduce_2d_bundle_cuda_handle_payload(compile_args, run_args, payload, *, dtype: str) -> None:
        _assert_parallel_reduce_2d_bundle_payload(compile_args, payload, dtype=dtype)
        assert len(run_args) == 4

    _assert_parallel_reduce_2d_bundle_payload(parallel_reduce_mul_2d_bundle_args, parallel_reduce_mul_2d_bundle_payload, dtype="f32")
    _assert_parallel_reduce_2d_bundle_cuda_handle_payload(
        parallel_reduce_mul_2d_bundle_cuda_handle_compile_args,
        parallel_reduce_mul_2d_bundle_cuda_handle_run_args,
        parallel_reduce_mul_2d_bundle_cuda_handle_payload,
        dtype="f32",
    )
    _assert_parallel_reduce_2d_bundle_payload(parallel_reduce_max_2d_bundle_args, parallel_reduce_max_2d_bundle_payload, dtype="f32")
    _assert_parallel_reduce_2d_bundle_cuda_handle_payload(
        parallel_reduce_max_2d_bundle_cuda_handle_compile_args,
        parallel_reduce_max_2d_bundle_cuda_handle_run_args,
        parallel_reduce_max_2d_bundle_cuda_handle_payload,
        dtype="f32",
    )
    _assert_parallel_reduce_2d_bundle_payload(parallel_reduce_min_2d_bundle_args, parallel_reduce_min_2d_bundle_payload, dtype="f32")
    _assert_parallel_reduce_2d_bundle_cuda_handle_payload(
        parallel_reduce_min_2d_bundle_cuda_handle_compile_args,
        parallel_reduce_min_2d_bundle_cuda_handle_run_args,
        parallel_reduce_min_2d_bundle_cuda_handle_payload,
        dtype="f32",
    )
    _assert_parallel_reduce_2d_bundle_payload(parallel_reduce_mul_2d_bundle_i32_args, parallel_reduce_mul_2d_bundle_i32_payload, dtype="i32")
    _assert_parallel_reduce_2d_bundle_cuda_handle_payload(
        parallel_reduce_mul_2d_bundle_i32_cuda_handle_compile_args,
        parallel_reduce_mul_2d_bundle_i32_cuda_handle_run_args,
        parallel_reduce_mul_2d_bundle_i32_cuda_handle_payload,
        dtype="i32",
    )
    _assert_parallel_reduce_2d_bundle_payload(parallel_reduce_max_2d_bundle_i32_args, parallel_reduce_max_2d_bundle_i32_payload, dtype="i32")
    _assert_parallel_reduce_2d_bundle_cuda_handle_payload(
        parallel_reduce_max_2d_bundle_i32_cuda_handle_compile_args,
        parallel_reduce_max_2d_bundle_i32_cuda_handle_run_args,
        parallel_reduce_max_2d_bundle_i32_cuda_handle_payload,
        dtype="i32",
    )
    _assert_parallel_reduce_2d_bundle_payload(parallel_reduce_min_2d_bundle_i32_args, parallel_reduce_min_2d_bundle_i32_payload, dtype="i32")
    _assert_parallel_reduce_2d_bundle_cuda_handle_payload(
        parallel_reduce_min_2d_bundle_i32_cuda_handle_compile_args,
        parallel_reduce_min_2d_bundle_i32_cuda_handle_run_args,
        parallel_reduce_min_2d_bundle_i32_cuda_handle_payload,
        dtype="i32",
    )

    def _assert_tensor_factory_payload(args, payload, *, dtype: str) -> None:
        assert len(args) == 3
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        if dtype == "i32":
            assert str(args[0].dtype) == "i32"
            assert str(args[1].dtype) == "i32"
            assert str(args[2].dtype) == "i32"
        assert payload["result_indices"] == ()

    def _assert_tensor_factory_cuda_handle_payload(compile_args, run_args, payload, *, dtype: str) -> None:
        _assert_tensor_factory_payload(compile_args, payload, dtype=dtype)
        assert len(run_args) == 3

    _assert_tensor_factory_payload(tensor_factory_2d_args, tensor_factory_2d_payload, dtype="f32")
    _assert_tensor_factory_cuda_handle_payload(
        tensor_factory_2d_cuda_handle_compile_args,
        tensor_factory_2d_cuda_handle_run_args,
        tensor_factory_2d_cuda_handle_payload,
        dtype="f32",
    )
    _assert_tensor_factory_payload(tensor_factory_2d_i32_args, tensor_factory_2d_i32_payload, dtype="i32")
    _assert_tensor_factory_cuda_handle_payload(
        tensor_factory_2d_i32_cuda_handle_compile_args,
        tensor_factory_2d_i32_cuda_handle_run_args,
        tensor_factory_2d_i32_cuda_handle_payload,
        dtype="i32",
    )

    assert len(reduce_add_2d_bundle_args) == 4
    assert reduce_add_2d_bundle_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert reduce_add_2d_bundle_args[1].shape == (1,)
    assert reduce_add_2d_bundle_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert reduce_add_2d_bundle_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert reduce_add_2d_bundle_payload["result_indices"] == ()

    assert len(reduce_add_2d_bundle_cuda_handle_compile_args) == 4
    assert reduce_add_2d_bundle_cuda_handle_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert reduce_add_2d_bundle_cuda_handle_compile_args[1].shape == (1,)
    assert reduce_add_2d_bundle_cuda_handle_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert reduce_add_2d_bundle_cuda_handle_compile_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert len(reduce_add_2d_bundle_cuda_handle_run_args) == 4
    assert reduce_add_2d_bundle_cuda_handle_payload["result_indices"] == ()

    assert len(reduce_add_2d_bundle_i32_args) == 4
    assert reduce_add_2d_bundle_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert reduce_add_2d_bundle_i32_args[1].shape == (1,)
    assert reduce_add_2d_bundle_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert reduce_add_2d_bundle_i32_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert str(reduce_add_2d_bundle_i32_args[0].dtype) == "i32"
    assert str(reduce_add_2d_bundle_i32_args[1].dtype) == "i32"
    assert str(reduce_add_2d_bundle_i32_args[2].dtype) == "i32"
    assert str(reduce_add_2d_bundle_i32_args[3].dtype) == "i32"
    assert reduce_add_2d_bundle_i32_payload["result_indices"] == ()

    assert len(reduce_add_2d_bundle_i32_cuda_handle_compile_args) == 4
    assert reduce_add_2d_bundle_i32_cuda_handle_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert reduce_add_2d_bundle_i32_cuda_handle_compile_args[1].shape == (1,)
    assert reduce_add_2d_bundle_i32_cuda_handle_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert reduce_add_2d_bundle_i32_cuda_handle_compile_args[3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert str(reduce_add_2d_bundle_i32_cuda_handle_compile_args[0].dtype) == "i32"
    assert str(reduce_add_2d_bundle_i32_cuda_handle_compile_args[1].dtype) == "i32"
    assert str(reduce_add_2d_bundle_i32_cuda_handle_compile_args[2].dtype) == "i32"
    assert str(reduce_add_2d_bundle_i32_cuda_handle_compile_args[3].dtype) == "i32"
    assert len(reduce_add_2d_bundle_i32_cuda_handle_run_args) == 4
    assert reduce_add_2d_bundle_i32_cuda_handle_payload["result_indices"] == ()

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


def test_compare_backends_treats_ptx_exec_as_executable() -> None:
    compare_backends = _load_tool_module("compare_backends")

    assert compare_backends._is_executable_backend("ptx_exec") is True


def test_compare_backends_resolves_numeric_nvidia_targets() -> None:
    compare_backends = _load_tool_module("compare_backends")

    int_like_target = compare_backends._resolve_cli_target(bb, "80")
    prefixed_target = compare_backends._resolve_cli_target(bb, "sm_90a")
    amd_target = compare_backends._resolve_cli_target(bb, "gfx950")

    assert isinstance(int_like_target, bb.NvidiaTarget)
    assert int_like_target.arch == "sm_80"
    assert isinstance(prefixed_target, bb.NvidiaTarget)
    assert prefixed_target.arch == "sm_90a"
    assert isinstance(amd_target, bb.AMDTarget)
    assert amd_target.arch == "gfx950"


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


def test_ptx_tensor_binary_2d_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    copy_payload = kernels.dense_copy_2d_f32_args()
    copy_cuda_handle_payload = kernels.dense_copy_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    copy_i32_payload = kernels.dense_copy_2d_i32_args()
    copy_i32_cuda_handle_payload = kernels.dense_copy_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_copy_payload = kernels.parallel_dense_copy_2d_f32_args()
    parallel_copy_cuda_handle_payload = kernels.parallel_dense_copy_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_copy_i32_payload = kernels.parallel_dense_copy_2d_i32_args()
    parallel_copy_i32_cuda_handle_payload = kernels.parallel_dense_copy_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    scalar_dense_payload = kernels.dense_scalar_add_2d_f32_args()
    scalar_dense_cuda_handle_payload = kernels.dense_scalar_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    scalar_dense_i32_payload = kernels.dense_scalar_add_2d_i32_args()
    scalar_dense_i32_cuda_handle_payload = kernels.dense_scalar_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    tensor_scalar_dense_payload = kernels.dense_tensor_scalar_add_2d_f32_args()
    tensor_scalar_dense_cuda_handle_payload = kernels.dense_tensor_scalar_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    tensor_scalar_dense_i32_payload = kernels.dense_tensor_scalar_add_2d_i32_args()
    tensor_scalar_dense_i32_cuda_handle_payload = kernels.dense_tensor_scalar_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_scalar_dense_payload = kernels.parallel_dense_scalar_add_2d_f32_args()
    parallel_scalar_dense_cuda_handle_payload = kernels.parallel_dense_scalar_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_scalar_dense_i32_payload = kernels.parallel_dense_scalar_add_2d_i32_args()
    parallel_scalar_dense_i32_cuda_handle_payload = kernels.parallel_dense_scalar_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_tensor_scalar_dense_payload = kernels.parallel_dense_tensor_scalar_add_2d_f32_args()
    parallel_tensor_scalar_dense_cuda_handle_payload = kernels.parallel_dense_tensor_scalar_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_tensor_scalar_dense_i32_payload = kernels.parallel_dense_tensor_scalar_add_2d_i32_args()
    parallel_tensor_scalar_dense_i32_cuda_handle_payload = kernels.parallel_dense_tensor_scalar_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_add_2d_f32_args()
    dense_cuda_handle_payload = kernels.dense_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_i32_payload = kernels.dense_add_2d_i32_args()
    dense_i32_cuda_handle_payload = kernels.dense_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_dense_payload = kernels.parallel_dense_add_2d_f32_args()
    parallel_dense_cuda_handle_payload = kernels.parallel_dense_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_dense_cuda_dlpack_payload = kernels.parallel_dense_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_dense_i32_payload = kernels.parallel_dense_add_2d_i32_args()
    parallel_dense_i32_cuda_handle_payload = kernels.parallel_dense_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_dense_i32_cuda_dlpack_payload = kernels.parallel_dense_add_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    broadcast_payload = kernels.broadcast_add_2d_f32_args()
    broadcast_cuda_handle_payload = kernels.broadcast_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    broadcast_i32_payload = kernels.broadcast_add_2d_i32_args()
    broadcast_i32_cuda_handle_payload = kernels.broadcast_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_broadcast_payload = kernels.parallel_broadcast_add_2d_f32_args()
    parallel_broadcast_cuda_handle_payload = kernels.parallel_broadcast_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_broadcast_i32_payload = kernels.parallel_broadcast_add_2d_i32_args()
    parallel_broadcast_i32_cuda_handle_payload = kernels.parallel_broadcast_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    sqrt_payload = kernels.dense_sqrt_2d_f32_args()
    sqrt_cuda_handle_payload = kernels.dense_sqrt_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_sqrt_payload = kernels.parallel_dense_sqrt_2d_f32_args()
    parallel_sqrt_cuda_handle_payload = kernels.parallel_dense_sqrt_2d_f32_cuda_handle_args(backend_name="ptx_exec")

    copy_args = copy_payload["args"]
    copy_cuda_compile_args = copy_cuda_handle_payload["compile_args"]
    copy_cuda_run_args = copy_cuda_handle_payload["run_args"]
    copy_i32_args = copy_i32_payload["args"]
    copy_i32_cuda_compile_args = copy_i32_cuda_handle_payload["compile_args"]
    copy_i32_cuda_run_args = copy_i32_cuda_handle_payload["run_args"]
    parallel_copy_args = parallel_copy_payload["args"]
    parallel_copy_cuda_compile_args = parallel_copy_cuda_handle_payload["compile_args"]
    parallel_copy_cuda_run_args = parallel_copy_cuda_handle_payload["run_args"]
    parallel_copy_i32_args = parallel_copy_i32_payload["args"]
    parallel_copy_i32_cuda_compile_args = parallel_copy_i32_cuda_handle_payload["compile_args"]
    parallel_copy_i32_cuda_run_args = parallel_copy_i32_cuda_handle_payload["run_args"]
    scalar_dense_args = scalar_dense_payload["args"]
    scalar_dense_cuda_compile_args = scalar_dense_cuda_handle_payload["compile_args"]
    scalar_dense_cuda_run_args = scalar_dense_cuda_handle_payload["run_args"]
    scalar_dense_i32_args = scalar_dense_i32_payload["args"]
    scalar_dense_i32_cuda_compile_args = scalar_dense_i32_cuda_handle_payload["compile_args"]
    scalar_dense_i32_cuda_run_args = scalar_dense_i32_cuda_handle_payload["run_args"]
    tensor_scalar_dense_args = tensor_scalar_dense_payload["args"]
    tensor_scalar_dense_cuda_compile_args = tensor_scalar_dense_cuda_handle_payload["compile_args"]
    tensor_scalar_dense_cuda_run_args = tensor_scalar_dense_cuda_handle_payload["run_args"]
    tensor_scalar_dense_i32_args = tensor_scalar_dense_i32_payload["args"]
    tensor_scalar_dense_i32_cuda_compile_args = tensor_scalar_dense_i32_cuda_handle_payload["compile_args"]
    tensor_scalar_dense_i32_cuda_run_args = tensor_scalar_dense_i32_cuda_handle_payload["run_args"]
    parallel_scalar_dense_args = parallel_scalar_dense_payload["args"]
    parallel_scalar_dense_cuda_compile_args = parallel_scalar_dense_cuda_handle_payload["compile_args"]
    parallel_scalar_dense_cuda_run_args = parallel_scalar_dense_cuda_handle_payload["run_args"]
    parallel_scalar_dense_i32_args = parallel_scalar_dense_i32_payload["args"]
    parallel_scalar_dense_i32_cuda_compile_args = parallel_scalar_dense_i32_cuda_handle_payload["compile_args"]
    parallel_scalar_dense_i32_cuda_run_args = parallel_scalar_dense_i32_cuda_handle_payload["run_args"]
    parallel_tensor_scalar_dense_args = parallel_tensor_scalar_dense_payload["args"]
    parallel_tensor_scalar_dense_cuda_compile_args = parallel_tensor_scalar_dense_cuda_handle_payload["compile_args"]
    parallel_tensor_scalar_dense_cuda_run_args = parallel_tensor_scalar_dense_cuda_handle_payload["run_args"]
    parallel_tensor_scalar_dense_i32_args = parallel_tensor_scalar_dense_i32_payload["args"]
    parallel_tensor_scalar_dense_i32_cuda_compile_args = parallel_tensor_scalar_dense_i32_cuda_handle_payload["compile_args"]
    parallel_tensor_scalar_dense_i32_cuda_run_args = parallel_tensor_scalar_dense_i32_cuda_handle_payload["run_args"]
    dense_args = dense_payload["args"]
    dense_cuda_compile_args = dense_cuda_handle_payload["compile_args"]
    dense_cuda_run_args = dense_cuda_handle_payload["run_args"]
    dense_i32_args = dense_i32_payload["args"]
    dense_i32_cuda_compile_args = dense_i32_cuda_handle_payload["compile_args"]
    dense_i32_cuda_run_args = dense_i32_cuda_handle_payload["run_args"]
    parallel_dense_args = parallel_dense_payload["args"]
    parallel_dense_cuda_compile_args = parallel_dense_cuda_handle_payload["compile_args"]
    parallel_dense_cuda_run_args = parallel_dense_cuda_handle_payload["run_args"]
    parallel_dense_cuda_dlpack_compile_args = parallel_dense_cuda_dlpack_payload["compile_args"]
    parallel_dense_cuda_dlpack_run_args = parallel_dense_cuda_dlpack_payload["run_args"]
    parallel_dense_i32_args = parallel_dense_i32_payload["args"]
    parallel_dense_i32_cuda_compile_args = parallel_dense_i32_cuda_handle_payload["compile_args"]
    parallel_dense_i32_cuda_run_args = parallel_dense_i32_cuda_handle_payload["run_args"]
    parallel_dense_i32_cuda_dlpack_compile_args = parallel_dense_i32_cuda_dlpack_payload["compile_args"]
    parallel_dense_i32_cuda_dlpack_run_args = parallel_dense_i32_cuda_dlpack_payload["run_args"]
    broadcast_args = broadcast_payload["args"]
    broadcast_cuda_compile_args = broadcast_cuda_handle_payload["compile_args"]
    broadcast_cuda_run_args = broadcast_cuda_handle_payload["run_args"]
    broadcast_i32_args = broadcast_i32_payload["args"]
    broadcast_i32_cuda_compile_args = broadcast_i32_cuda_handle_payload["compile_args"]
    broadcast_i32_cuda_run_args = broadcast_i32_cuda_handle_payload["run_args"]
    parallel_broadcast_args = parallel_broadcast_payload["args"]
    parallel_broadcast_cuda_compile_args = parallel_broadcast_cuda_handle_payload["compile_args"]
    parallel_broadcast_cuda_run_args = parallel_broadcast_cuda_handle_payload["run_args"]
    parallel_broadcast_i32_args = parallel_broadcast_i32_payload["args"]
    parallel_broadcast_i32_cuda_compile_args = parallel_broadcast_i32_cuda_handle_payload["compile_args"]
    parallel_broadcast_i32_cuda_run_args = parallel_broadcast_i32_cuda_handle_payload["run_args"]
    sqrt_args = sqrt_payload["args"]
    sqrt_cuda_compile_args = sqrt_cuda_handle_payload["compile_args"]
    sqrt_cuda_run_args = sqrt_cuda_handle_payload["run_args"]
    parallel_sqrt_args = parallel_sqrt_payload["args"]
    parallel_sqrt_cuda_compile_args = parallel_sqrt_cuda_handle_payload["compile_args"]
    parallel_sqrt_cuda_run_args = parallel_sqrt_cuda_handle_payload["run_args"]

    assert len(copy_args) == 2
    assert copy_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_payload["result_indices"] == ()

    assert len(copy_cuda_compile_args) == 2
    assert copy_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(copy_cuda_run_args) == 2
    assert copy_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_cuda_handle_payload["result_indices"] == ()

    assert len(copy_i32_args) == 2
    assert copy_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_i32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(copy_i32_args[0].dtype) == "i32"
    assert copy_i32_payload["result_indices"] == ()

    assert len(copy_i32_cuda_compile_args) == 2
    assert copy_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_i32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(copy_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(copy_i32_cuda_run_args) == 2
    assert copy_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_i32_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert copy_i32_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_copy_args) == 2
    assert parallel_copy_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_payload["result_indices"] == ()

    assert len(parallel_copy_cuda_compile_args) == 2
    assert parallel_copy_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_copy_cuda_run_args) == 2
    assert parallel_copy_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_copy_i32_args) == 2
    assert parallel_copy_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_i32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_copy_i32_args[0].dtype) == "i32"
    assert parallel_copy_i32_payload["result_indices"] == ()

    assert len(parallel_copy_i32_cuda_compile_args) == 2
    assert parallel_copy_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_i32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_copy_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(parallel_copy_i32_cuda_run_args) == 2
    assert parallel_copy_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_i32_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_copy_i32_cuda_handle_payload["result_indices"] == ()

    assert len(scalar_dense_args) == 3
    assert scalar_dense_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert float(scalar_dense_args[1]) == 1.5
    assert scalar_dense_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert scalar_dense_payload["result_indices"] == ()

    assert len(scalar_dense_cuda_compile_args) == 3
    assert scalar_dense_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert float(scalar_dense_cuda_compile_args[1]) == 1.5
    assert scalar_dense_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(scalar_dense_cuda_run_args) == 3
    assert scalar_dense_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert scalar_dense_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert scalar_dense_cuda_handle_payload["result_indices"] == ()

    assert len(scalar_dense_i32_args) == 3
    assert scalar_dense_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert int(scalar_dense_i32_args[1]) == 7
    assert scalar_dense_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(scalar_dense_i32_args[0].dtype) == "i32"
    assert scalar_dense_i32_payload["result_indices"] == ()

    assert len(scalar_dense_i32_cuda_compile_args) == 3
    assert scalar_dense_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert int(scalar_dense_i32_cuda_compile_args[1]) == 7
    assert scalar_dense_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(scalar_dense_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(scalar_dense_i32_cuda_run_args) == 3
    assert scalar_dense_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert scalar_dense_i32_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert scalar_dense_i32_cuda_handle_payload["result_indices"] == ()

    assert len(tensor_scalar_dense_args) == 3
    assert tensor_scalar_dense_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_args[1].shape == (1,)
    assert tensor_scalar_dense_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_i32_payload["result_indices"] == ()

    assert len(tensor_scalar_dense_cuda_compile_args) == 3
    assert tensor_scalar_dense_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_cuda_compile_args[1].shape == (1,)
    assert tensor_scalar_dense_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(tensor_scalar_dense_cuda_run_args) == 3
    assert tensor_scalar_dense_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_cuda_run_args[1].shape == (1,)
    assert tensor_scalar_dense_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_cuda_handle_payload["result_indices"] == ()

    assert len(tensor_scalar_dense_i32_args) == 3
    assert tensor_scalar_dense_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_i32_args[1].shape == (1,)
    assert tensor_scalar_dense_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(tensor_scalar_dense_i32_args[0].dtype) == "i32"
    assert tensor_scalar_dense_payload["result_indices"] == ()

    assert len(tensor_scalar_dense_i32_cuda_compile_args) == 3
    assert tensor_scalar_dense_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_i32_cuda_compile_args[1].shape == (1,)
    assert tensor_scalar_dense_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(tensor_scalar_dense_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(tensor_scalar_dense_i32_cuda_run_args) == 3
    assert tensor_scalar_dense_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_i32_cuda_run_args[1].shape == (1,)
    assert tensor_scalar_dense_i32_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert tensor_scalar_dense_i32_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_scalar_dense_args) == 3
    assert parallel_scalar_dense_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert float(parallel_scalar_dense_args[1]) == 1.5
    assert parallel_scalar_dense_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_scalar_dense_payload["result_indices"] == ()

    assert len(parallel_scalar_dense_cuda_compile_args) == 3
    assert parallel_scalar_dense_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert float(parallel_scalar_dense_cuda_compile_args[1]) == 1.5
    assert parallel_scalar_dense_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_scalar_dense_cuda_run_args) == 3
    assert parallel_scalar_dense_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_scalar_dense_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_scalar_dense_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_scalar_dense_i32_args) == 3
    assert parallel_scalar_dense_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert int(parallel_scalar_dense_i32_args[1]) == 7
    assert parallel_scalar_dense_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_scalar_dense_i32_args[0].dtype) == "i32"
    assert parallel_scalar_dense_i32_payload["result_indices"] == ()

    assert len(parallel_scalar_dense_i32_cuda_compile_args) == 3
    assert parallel_scalar_dense_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert int(parallel_scalar_dense_i32_cuda_compile_args[1]) == 7
    assert parallel_scalar_dense_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_scalar_dense_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(parallel_scalar_dense_i32_cuda_run_args) == 3
    assert parallel_scalar_dense_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_scalar_dense_i32_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_scalar_dense_i32_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_tensor_scalar_dense_args) == 3
    assert parallel_tensor_scalar_dense_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_args[1].shape == (1,)
    assert parallel_tensor_scalar_dense_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_i32_payload["result_indices"] == ()

    assert len(parallel_tensor_scalar_dense_cuda_compile_args) == 3
    assert parallel_tensor_scalar_dense_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_cuda_compile_args[1].shape == (1,)
    assert parallel_tensor_scalar_dense_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_tensor_scalar_dense_cuda_run_args) == 3
    assert parallel_tensor_scalar_dense_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_cuda_run_args[1].shape == (1,)
    assert parallel_tensor_scalar_dense_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_tensor_scalar_dense_i32_args) == 3
    assert parallel_tensor_scalar_dense_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_i32_args[1].shape == (1,)
    assert parallel_tensor_scalar_dense_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_tensor_scalar_dense_i32_args[0].dtype) == "i32"
    assert parallel_tensor_scalar_dense_payload["result_indices"] == ()

    assert len(parallel_tensor_scalar_dense_i32_cuda_compile_args) == 3
    assert parallel_tensor_scalar_dense_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_i32_cuda_compile_args[1].shape == (1,)
    assert parallel_tensor_scalar_dense_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_tensor_scalar_dense_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(parallel_tensor_scalar_dense_i32_cuda_run_args) == 3
    assert parallel_tensor_scalar_dense_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_i32_cuda_run_args[1].shape == (1,)
    assert parallel_tensor_scalar_dense_i32_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_tensor_scalar_dense_i32_cuda_handle_payload["result_indices"] == ()

    assert len(dense_args) == 3
    assert dense_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_payload["result_indices"] == ()

    assert len(dense_cuda_compile_args) == 3
    assert dense_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(dense_cuda_run_args) == 3
    assert dense_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_handle_payload["result_indices"] == ()

    assert len(dense_i32_args) == 3
    assert dense_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_i32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(dense_i32_args[0].dtype) == "i32"
    assert dense_i32_payload["result_indices"] == ()

    assert len(dense_i32_cuda_compile_args) == 3
    assert dense_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_i32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(dense_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(dense_i32_cuda_run_args) == 3
    assert dense_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_i32_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_dense_args) == 3
    assert parallel_dense_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_payload["result_indices"] == ()

    assert len(parallel_dense_cuda_compile_args) == 3
    assert parallel_dense_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_dense_cuda_run_args) == 3
    assert parallel_dense_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_dense_cuda_dlpack_compile_args) == 3
    assert parallel_dense_cuda_dlpack_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_dlpack_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_dlpack_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_dense_cuda_dlpack_run_args) == 3
    assert parallel_dense_cuda_dlpack_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_dlpack_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_dlpack_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_cuda_dlpack_payload["result_indices"] == ()

    assert len(parallel_dense_i32_args) == 3
    assert parallel_dense_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_dense_i32_args[0].dtype) == "i32"
    assert parallel_dense_i32_payload["result_indices"] == ()

    assert len(parallel_dense_i32_cuda_compile_args) == 3
    assert parallel_dense_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_dense_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(parallel_dense_i32_cuda_run_args) == 3
    assert parallel_dense_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_dense_i32_cuda_dlpack_compile_args) == 3
    assert parallel_dense_i32_cuda_dlpack_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_cuda_dlpack_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_cuda_dlpack_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_dense_i32_cuda_dlpack_compile_args[0].dtype) == "i32"
    assert len(parallel_dense_i32_cuda_dlpack_run_args) == 3
    assert parallel_dense_i32_cuda_dlpack_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_i32_cuda_dlpack_payload["result_indices"] == ()

    assert len(broadcast_args) == 3
    assert broadcast_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert broadcast_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert broadcast_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert broadcast_payload["result_indices"] == ()

    assert len(broadcast_cuda_compile_args) == 3
    assert broadcast_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert broadcast_cuda_compile_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert broadcast_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(broadcast_cuda_run_args) == 3
    assert broadcast_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert broadcast_cuda_run_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert broadcast_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert broadcast_cuda_handle_payload["result_indices"] == ()

    assert len(broadcast_i32_args) == 3
    assert broadcast_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert broadcast_i32_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert broadcast_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(broadcast_i32_args[0].dtype) == "i32"
    assert broadcast_i32_payload["result_indices"] == ()

    assert len(broadcast_i32_cuda_compile_args) == 3
    assert broadcast_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert broadcast_i32_cuda_compile_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert broadcast_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(broadcast_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(broadcast_i32_cuda_run_args) == 3
    assert broadcast_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert broadcast_i32_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_broadcast_args) == 3
    assert parallel_broadcast_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_payload["result_indices"] == ()

    assert len(parallel_broadcast_cuda_compile_args) == 3
    assert parallel_broadcast_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_cuda_compile_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_broadcast_cuda_run_args) == 3
    assert parallel_broadcast_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_cuda_run_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_broadcast_i32_args) == 3
    assert parallel_broadcast_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_i32_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_i32_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_broadcast_i32_args[0].dtype) == "i32"
    assert parallel_broadcast_i32_payload["result_indices"] == ()

    assert len(parallel_broadcast_i32_cuda_compile_args) == 3
    assert parallel_broadcast_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_i32_cuda_compile_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_i32_cuda_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_broadcast_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(parallel_broadcast_i32_cuda_run_args) == 3
    assert parallel_broadcast_i32_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_i32_cuda_run_args[1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_i32_cuda_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_i32_cuda_handle_payload["result_indices"] == ()

    assert len(sqrt_args) == 2
    assert sqrt_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert sqrt_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert sqrt_payload["result_indices"] == ()

    assert len(sqrt_cuda_compile_args) == 2
    assert sqrt_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert sqrt_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(sqrt_cuda_run_args) == 2
    assert sqrt_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert sqrt_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert sqrt_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_sqrt_args) == 2
    assert parallel_sqrt_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_sqrt_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_sqrt_payload["result_indices"] == ()

    assert len(parallel_sqrt_cuda_compile_args) == 2
    assert parallel_sqrt_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_sqrt_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_sqrt_cuda_run_args) == 2
    assert parallel_sqrt_cuda_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_sqrt_cuda_run_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_sqrt_cuda_handle_payload["result_indices"] == ()


def test_ptx_integer_scalar_broadcast_2d_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    dense_scalar_payload = kernels.dense_scalar_bitand_2d_i32_args()
    dense_scalar_cuda_handle_payload = kernels.dense_scalar_bitand_2d_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    dense_scalar_cuda_dlpack_payload = kernels.dense_scalar_bitand_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    dense_tensor_payload = kernels.dense_tensor_scalar_bitor_2d_i32_args()
    dense_tensor_cuda_handle_payload = kernels.dense_tensor_scalar_bitor_2d_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    dense_tensor_cuda_dlpack_payload = kernels.dense_tensor_scalar_bitor_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    parallel_scalar_payload = kernels.parallel_dense_scalar_bitand_2d_i32_args()
    parallel_scalar_cuda_handle_payload = kernels.parallel_dense_scalar_bitand_2d_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    parallel_scalar_cuda_dlpack_payload = kernels.parallel_dense_scalar_bitand_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    parallel_tensor_payload = kernels.parallel_dense_tensor_scalar_bitor_2d_i32_args()
    parallel_tensor_cuda_handle_payload = kernels.parallel_dense_tensor_scalar_bitor_2d_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    parallel_tensor_cuda_dlpack_payload = kernels.parallel_dense_tensor_scalar_bitor_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )

    for payload in (dense_scalar_payload, dense_tensor_payload, parallel_scalar_payload, parallel_tensor_payload):
        assert len(payload["args"]) == 3
        assert payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(payload["args"][0].dtype) == "i32"
        assert payload["args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(payload["args"][2].dtype) == "i32"
        assert payload["result_indices"] == ()

    assert dense_scalar_payload["args"][1] == bb.Int32(5)
    assert dense_tensor_payload["args"][1].shape == (1,)
    assert dense_tensor_payload["args"][1].tolist() == [24]
    assert parallel_scalar_payload["args"][1] == bb.Int32(5)
    assert parallel_tensor_payload["args"][1].shape == (1,)
    assert parallel_tensor_payload["args"][1].tolist() == [24]

    for payload in (
        dense_scalar_cuda_handle_payload,
        dense_scalar_cuda_dlpack_payload,
        dense_tensor_cuda_handle_payload,
        dense_tensor_cuda_dlpack_payload,
        parallel_scalar_cuda_handle_payload,
        parallel_scalar_cuda_dlpack_payload,
        parallel_tensor_cuda_handle_payload,
        parallel_tensor_cuda_dlpack_payload,
    ):
        assert len(payload["compile_args"]) == 3
        assert payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["compile_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()


def test_ptx_cuda_dlpack_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_add_payload = kernels.indexed_add_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_dense_add_payload = kernels.parallel_dense_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_tensor_scalar_f32_payload = kernels.dense_tensor_scalar_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_tensor_scalar_i32_payload = kernels.dense_tensor_scalar_add_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_dense_tensor_scalar_f32_payload = kernels.parallel_dense_tensor_scalar_add_2d_f32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    parallel_dense_tensor_scalar_i32_payload = kernels.parallel_dense_tensor_scalar_add_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )

    indexed_add_compile_args = indexed_add_payload["compile_args"]
    indexed_add_run_args = indexed_add_payload["run_args"]
    parallel_dense_add_compile_args = parallel_dense_add_payload["compile_args"]
    parallel_dense_add_run_args = parallel_dense_add_payload["run_args"]
    dense_tensor_scalar_f32_compile_args = dense_tensor_scalar_f32_payload["compile_args"]
    dense_tensor_scalar_f32_run_args = dense_tensor_scalar_f32_payload["run_args"]
    dense_tensor_scalar_i32_compile_args = dense_tensor_scalar_i32_payload["compile_args"]
    dense_tensor_scalar_i32_run_args = dense_tensor_scalar_i32_payload["run_args"]
    parallel_dense_tensor_scalar_f32_compile_args = parallel_dense_tensor_scalar_f32_payload["compile_args"]
    parallel_dense_tensor_scalar_f32_run_args = parallel_dense_tensor_scalar_f32_payload["run_args"]
    parallel_dense_tensor_scalar_i32_compile_args = parallel_dense_tensor_scalar_i32_payload["compile_args"]
    parallel_dense_tensor_scalar_i32_run_args = parallel_dense_tensor_scalar_i32_payload["run_args"]

    assert len(indexed_add_compile_args) == 3
    assert indexed_add_compile_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_add_compile_args[1].shape == (kernels.POINTWISE_N,)
    assert indexed_add_compile_args[2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_add_run_args) == 3
    assert indexed_add_payload["result_indices"] == ()

    assert len(parallel_dense_add_compile_args) == 3
    assert parallel_dense_add_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_add_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_add_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_dense_add_run_args) == 3
    assert parallel_dense_add_payload["result_indices"] == ()

    assert len(dense_tensor_scalar_f32_compile_args) == 3
    assert dense_tensor_scalar_f32_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_tensor_scalar_f32_compile_args[1].shape == (1,)
    assert dense_tensor_scalar_f32_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(dense_tensor_scalar_f32_run_args) == 3
    assert dense_tensor_scalar_f32_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_tensor_scalar_f32_run_args[1].shape == (1,)
    assert dense_tensor_scalar_f32_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_tensor_scalar_f32_payload["result_indices"] == ()

    assert len(dense_tensor_scalar_i32_compile_args) == 3
    assert dense_tensor_scalar_i32_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_tensor_scalar_i32_compile_args[1].shape == (1,)
    assert dense_tensor_scalar_i32_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(dense_tensor_scalar_i32_compile_args[0].dtype) == "i32"
    assert len(dense_tensor_scalar_i32_run_args) == 3
    assert dense_tensor_scalar_i32_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_tensor_scalar_i32_run_args[1].shape == (1,)
    assert dense_tensor_scalar_i32_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_tensor_scalar_i32_payload["result_indices"] == ()

    assert len(parallel_dense_tensor_scalar_f32_compile_args) == 3
    assert parallel_dense_tensor_scalar_f32_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_tensor_scalar_f32_compile_args[1].shape == (1,)
    assert parallel_dense_tensor_scalar_f32_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_dense_tensor_scalar_f32_run_args) == 3
    assert parallel_dense_tensor_scalar_f32_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_tensor_scalar_f32_run_args[1].shape == (1,)
    assert parallel_dense_tensor_scalar_f32_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_tensor_scalar_f32_payload["result_indices"] == ()

    assert len(parallel_dense_tensor_scalar_i32_compile_args) == 3
    assert parallel_dense_tensor_scalar_i32_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_tensor_scalar_i32_compile_args[1].shape == (1,)
    assert parallel_dense_tensor_scalar_i32_compile_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_dense_tensor_scalar_i32_compile_args[0].dtype) == "i32"
    assert len(parallel_dense_tensor_scalar_i32_run_args) == 3
    assert parallel_dense_tensor_scalar_i32_run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_tensor_scalar_i32_run_args[1].shape == (1,)
    assert parallel_dense_tensor_scalar_i32_run_args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_dense_tensor_scalar_i32_payload["result_indices"] == ()


def test_backend_benchmark_kernels_exports_sub_and_mul_microbench_kernels() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    assert callable(kernels.dense_sub_f32_kernel)
    assert callable(kernels.dense_mul_f32_kernel)
    assert callable(kernels.dense_copy_2d_f32_kernel)
    assert callable(kernels.dense_copy_2d_f32_args)
    assert callable(kernels.dense_copy_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_copy_2d_f32_kernel)
    assert callable(kernels.parallel_dense_copy_2d_f32_args)
    assert callable(kernels.parallel_dense_copy_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_copy_2d_i32_kernel)
    assert callable(kernels.parallel_dense_copy_2d_i32_args)
    assert callable(kernels.parallel_dense_copy_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_copy_2d_i32_kernel)
    assert callable(kernels.dense_copy_2d_i32_args)
    assert callable(kernels.dense_copy_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_scalar_add_2d_f32_kernel)
    assert callable(kernels.dense_scalar_add_2d_f32_args)
    assert callable(kernels.dense_scalar_add_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_tensor_scalar_add_2d_f32_kernel)
    assert callable(kernels.dense_tensor_scalar_add_2d_f32_args)
    assert callable(kernels.dense_tensor_scalar_add_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_tensor_scalar_add_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_scalar_add_2d_f32_kernel)
    assert callable(kernels.parallel_dense_scalar_add_2d_f32_args)
    assert callable(kernels.parallel_dense_scalar_add_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_f32_kernel)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_f32_args)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_scalar_add_2d_i32_kernel)
    assert callable(kernels.parallel_dense_scalar_add_2d_i32_args)
    assert callable(kernels.parallel_dense_scalar_add_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_dense_scalar_bitand_2d_i32_kernel)
    assert callable(kernels.parallel_dense_scalar_bitand_2d_i32_args)
    assert callable(kernels.parallel_dense_scalar_bitand_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_dense_scalar_bitand_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_i32_kernel)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_i32_args)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_dense_tensor_scalar_add_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_tensor_scalar_bitor_2d_i32_kernel)
    assert callable(kernels.parallel_dense_tensor_scalar_bitor_2d_i32_args)
    assert callable(kernels.parallel_dense_tensor_scalar_bitor_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_dense_tensor_scalar_bitor_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_scalar_select_2d_f32_kernel)
    assert callable(kernels.parallel_dense_scalar_select_2d_f32_args)
    assert callable(kernels.parallel_dense_scalar_select_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_scalar_select_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_scalar_add_2d_i32_kernel)
    assert callable(kernels.dense_scalar_add_2d_i32_args)
    assert callable(kernels.dense_scalar_add_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_scalar_bitand_2d_i32_kernel)
    assert callable(kernels.dense_scalar_bitand_2d_i32_args)
    assert callable(kernels.dense_scalar_bitand_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_scalar_bitand_2d_i32_cuda_dlpack_args)
    assert callable(kernels.dense_tensor_scalar_add_2d_i32_kernel)
    assert callable(kernels.dense_tensor_scalar_add_2d_i32_args)
    assert callable(kernels.dense_tensor_scalar_add_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_tensor_scalar_add_2d_i32_cuda_dlpack_args)
    assert callable(kernels.dense_tensor_scalar_bitor_2d_i32_kernel)
    assert callable(kernels.dense_tensor_scalar_bitor_2d_i32_args)
    assert callable(kernels.dense_tensor_scalar_bitor_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_tensor_scalar_bitor_2d_i32_cuda_dlpack_args)
    assert callable(kernels.dense_select_2d_f32_kernel)
    assert callable(kernels.dense_select_2d_f32_args)
    assert callable(kernels.dense_select_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_select_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_add_2d_f32_kernel)
    assert callable(kernels.dense_add_2d_f32_args)
    assert callable(kernels.dense_add_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_max_2d_f32_kernel)
    assert callable(kernels.dense_max_2d_f32_args)
    assert callable(kernels.dense_max_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_max_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_add_2d_f32_kernel)
    assert callable(kernels.parallel_dense_add_2d_f32_args)
    assert callable(kernels.parallel_dense_add_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_add_2d_i32_kernel)
    assert callable(kernels.dense_add_2d_i32_args)
    assert callable(kernels.dense_add_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_bitand_2d_i32_kernel)
    assert callable(kernels.dense_bitand_2d_i32_args)
    assert callable(kernels.dense_bitand_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_bitand_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_add_2d_i32_kernel)
    assert callable(kernels.parallel_dense_add_2d_i32_args)
    assert callable(kernels.parallel_dense_add_2d_i32_cuda_handle_args)
    assert callable(kernels.broadcast_add_2d_f32_kernel)
    assert callable(kernels.broadcast_add_2d_f32_args)
    assert callable(kernels.broadcast_add_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_broadcast_add_2d_f32_kernel)
    assert callable(kernels.parallel_broadcast_add_2d_f32_args)
    assert callable(kernels.parallel_broadcast_add_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_broadcast_add_2d_i32_kernel)
    assert callable(kernels.parallel_broadcast_add_2d_i32_args)
    assert callable(kernels.parallel_broadcast_add_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_broadcast_bitor_2d_i32_kernel)
    assert callable(kernels.parallel_broadcast_bitor_2d_i32_args)
    assert callable(kernels.parallel_broadcast_bitor_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_broadcast_bitor_2d_i32_cuda_dlpack_args)
    assert callable(kernels.broadcast_select_2d_i32_kernel)
    assert callable(kernels.broadcast_select_2d_i32_args)
    assert callable(kernels.broadcast_select_2d_i32_cuda_handle_args)
    assert callable(kernels.broadcast_select_2d_i32_cuda_dlpack_args)
    assert callable(kernels.broadcast_add_2d_i32_kernel)
    assert callable(kernels.broadcast_add_2d_i32_args)
    assert callable(kernels.broadcast_add_2d_i32_cuda_handle_args)
    assert callable(kernels.broadcast_min_2d_i32_kernel)
    assert callable(kernels.broadcast_min_2d_i32_args)
    assert callable(kernels.broadcast_min_2d_i32_cuda_handle_args)
    assert callable(kernels.broadcast_min_2d_i32_cuda_dlpack_args)
    assert callable(kernels.dense_sqrt_2d_f32_kernel)
    assert callable(kernels.dense_sqrt_2d_f32_args)
    assert callable(kernels.dense_sqrt_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_cos_2d_f32_kernel)
    assert callable(kernels.dense_cos_2d_f32_args)
    assert callable(kernels.dense_cos_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_cos_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_log_2d_f32_kernel)
    assert callable(kernels.dense_log_2d_f32_args)
    assert callable(kernels.dense_log_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_log_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_atan_2d_f32_kernel)
    assert callable(kernels.dense_atan_2d_f32_args)
    assert callable(kernels.dense_atan_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_atan_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_round_2d_f32_kernel)
    assert callable(kernels.dense_round_2d_f32_args)
    assert callable(kernels.dense_round_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_round_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_trunc_2d_f32_kernel)
    assert callable(kernels.dense_trunc_2d_f32_args)
    assert callable(kernels.dense_trunc_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_trunc_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_acos_2d_f32_kernel)
    assert callable(kernels.dense_acos_2d_f32_args)
    assert callable(kernels.dense_acos_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_acos_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_erf_2d_f32_kernel)
    assert callable(kernels.dense_erf_2d_f32_args)
    assert callable(kernels.dense_erf_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_erf_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_abs_2d_f32_kernel)
    assert callable(kernels.dense_abs_2d_f32_args)
    assert callable(kernels.dense_abs_2d_f32_cuda_handle_args)
    assert callable(kernels.dense_abs_2d_f32_cuda_dlpack_args)
    assert callable(kernels.dense_abs_2d_i32_kernel)
    assert callable(kernels.dense_abs_2d_i32_args)
    assert callable(kernels.dense_abs_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_abs_2d_i32_cuda_dlpack_args)
    assert callable(kernels.dense_bitnot_2d_i32_kernel)
    assert callable(kernels.dense_bitnot_2d_i32_args)
    assert callable(kernels.dense_bitnot_2d_i32_cuda_handle_args)
    assert callable(kernels.dense_bitnot_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_sqrt_2d_f32_kernel)
    assert callable(kernels.parallel_dense_sqrt_2d_f32_args)
    assert callable(kernels.parallel_dense_sqrt_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_log2_2d_f32_kernel)
    assert callable(kernels.parallel_dense_log2_2d_f32_args)
    assert callable(kernels.parallel_dense_log2_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_log2_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_exp_2d_f32_kernel)
    assert callable(kernels.parallel_dense_exp_2d_f32_args)
    assert callable(kernels.parallel_dense_exp_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_exp_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_atan_2d_f32_kernel)
    assert callable(kernels.parallel_dense_atan_2d_f32_args)
    assert callable(kernels.parallel_dense_atan_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_atan_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_round_2d_f32_kernel)
    assert callable(kernels.parallel_dense_round_2d_f32_args)
    assert callable(kernels.parallel_dense_round_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_round_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_floor_2d_f32_kernel)
    assert callable(kernels.parallel_dense_floor_2d_f32_args)
    assert callable(kernels.parallel_dense_floor_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_floor_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_acos_2d_f32_kernel)
    assert callable(kernels.parallel_dense_acos_2d_f32_args)
    assert callable(kernels.parallel_dense_acos_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_acos_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_erf_2d_f32_kernel)
    assert callable(kernels.parallel_dense_erf_2d_f32_args)
    assert callable(kernels.parallel_dense_erf_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_erf_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_scalar_max_2d_f32_kernel)
    assert callable(kernels.parallel_dense_scalar_max_2d_f32_args)
    assert callable(kernels.parallel_dense_scalar_max_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_scalar_max_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_abs_2d_f32_kernel)
    assert callable(kernels.parallel_dense_abs_2d_f32_args)
    assert callable(kernels.parallel_dense_abs_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_dense_abs_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_abs_2d_i32_kernel)
    assert callable(kernels.parallel_dense_abs_2d_i32_args)
    assert callable(kernels.parallel_dense_abs_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_dense_abs_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_dense_bitnot_2d_i32_kernel)
    assert callable(kernels.parallel_dense_bitnot_2d_i32_args)
    assert callable(kernels.parallel_dense_bitnot_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_dense_bitnot_2d_i32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_abs_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_abs_2d_f32_args)
    assert callable(kernels.multiblock_dense_abs_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_abs_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_exp2_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_exp2_2d_f32_args)
    assert callable(kernels.multiblock_dense_exp2_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_exp2_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_log10_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_log10_2d_f32_args)
    assert callable(kernels.multiblock_dense_log10_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_log10_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_atan_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_atan_2d_f32_args)
    assert callable(kernels.multiblock_dense_atan_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_atan_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_round_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_round_2d_f32_args)
    assert callable(kernels.multiblock_dense_round_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_round_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_ceil_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_ceil_2d_f32_args)
    assert callable(kernels.multiblock_dense_ceil_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_ceil_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_acos_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_acos_2d_f32_args)
    assert callable(kernels.multiblock_dense_acos_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_acos_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_atan2_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_atan2_2d_f32_args)
    assert callable(kernels.multiblock_dense_atan2_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_atan2_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_erf_2d_f32_kernel)
    assert callable(kernels.multiblock_dense_erf_2d_f32_args)
    assert callable(kernels.multiblock_dense_erf_2d_f32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_erf_2d_f32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_abs_2d_i32_kernel)
    assert callable(kernels.multiblock_dense_abs_2d_i32_args)
    assert callable(kernels.multiblock_dense_abs_2d_i32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_abs_2d_i32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_tensor_scalar_select_2d_i32_kernel)
    assert callable(kernels.multiblock_dense_tensor_scalar_select_2d_i32_args)
    assert callable(kernels.multiblock_dense_tensor_scalar_select_2d_i32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_tensor_scalar_select_2d_i32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_tensor_scalar_min_2d_i32_kernel)
    assert callable(kernels.multiblock_dense_tensor_scalar_min_2d_i32_args)
    assert callable(kernels.multiblock_dense_tensor_scalar_min_2d_i32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_tensor_scalar_min_2d_i32_cuda_dlpack_args)
    assert callable(kernels.multiblock_dense_bitnot_2d_i32_kernel)
    assert callable(kernels.multiblock_dense_bitnot_2d_i32_args)
    assert callable(kernels.multiblock_dense_bitnot_2d_i32_cuda_handle_args)
    assert callable(kernels.multiblock_dense_bitnot_2d_i32_cuda_dlpack_args)
    assert callable(kernels.dense_rsqrt_2d_f32_kernel)
    assert callable(kernels.dense_rsqrt_2d_f32_args)
    assert callable(kernels.dense_rsqrt_2d_f32_cuda_handle_args)
    assert callable(kernels.indexed_copy_f32_kernel)
    assert callable(kernels.indexed_copy_f32_cuda_handle_args)
    assert callable(kernels.indexed_copy_i32_kernel)
    assert callable(kernels.indexed_copy_i32_cuda_handle_args)
    assert callable(kernels.indexed_add_f32_cuda_handle_args)
    assert callable(kernels.indexed_add_i32_kernel)
    assert callable(kernels.indexed_add_i32_cuda_handle_args)
    assert callable(kernels.indexed_bitnot_i32_kernel)
    assert callable(kernels.indexed_bitnot_i32_args)
    assert callable(kernels.indexed_bitnot_i32_cuda_handle_args)
    assert callable(kernels.indexed_bitnot_i32_cuda_dlpack_args)
    assert callable(kernels.indexed_neg_f32_kernel)
    assert callable(kernels.indexed_neg_f32_args)
    assert callable(kernels.indexed_neg_f32_cuda_handle_args)
    assert callable(kernels.indexed_neg_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_neg_i32_kernel)
    assert callable(kernels.indexed_neg_i32_args)
    assert callable(kernels.indexed_neg_i32_cuda_handle_args)
    assert callable(kernels.indexed_neg_i32_cuda_dlpack_args)
    assert callable(kernels.indexed_abs_f32_kernel)
    assert callable(kernels.indexed_abs_f32_args)
    assert callable(kernels.indexed_abs_f32_cuda_handle_args)
    assert callable(kernels.indexed_abs_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_abs_i32_kernel)
    assert callable(kernels.indexed_abs_i32_args)
    assert callable(kernels.indexed_abs_i32_cuda_handle_args)
    assert callable(kernels.indexed_abs_i32_cuda_dlpack_args)
    assert callable(kernels.indexed_max_f32_kernel)
    assert callable(kernels.indexed_max_f32_args)
    assert callable(kernels.indexed_max_f32_cuda_handle_args)
    assert callable(kernels.indexed_max_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_min_i32_kernel)
    assert callable(kernels.indexed_min_i32_args)
    assert callable(kernels.indexed_min_i32_cuda_handle_args)
    assert callable(kernels.indexed_min_i32_cuda_dlpack_args)
    assert callable(kernels.indexed_scalar_broadcast_add_f32_kernel)
    assert callable(kernels.indexed_scalar_broadcast_add_f32_cuda_handle_args)
    assert callable(kernels.indexed_scalar_broadcast_add_i32_kernel)
    assert callable(kernels.indexed_scalar_broadcast_add_i32_cuda_handle_args)
    assert callable(kernels.indexed_sqrt_f32_kernel)
    assert callable(kernels.indexed_sin_f32_kernel)
    assert callable(kernels.indexed_sin_f32_args)
    assert callable(kernels.indexed_sin_f32_cuda_handle_args)
    assert callable(kernels.indexed_sin_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_exp_f32_kernel)
    assert callable(kernels.indexed_exp_f32_args)
    assert callable(kernels.indexed_exp_f32_cuda_handle_args)
    assert callable(kernels.indexed_exp_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_atan_f32_kernel)
    assert callable(kernels.indexed_atan_f32_args)
    assert callable(kernels.indexed_atan_f32_cuda_handle_args)
    assert callable(kernels.indexed_atan_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_round_f32_kernel)
    assert callable(kernels.indexed_round_f32_args)
    assert callable(kernels.indexed_round_f32_cuda_handle_args)
    assert callable(kernels.indexed_round_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_floor_f32_kernel)
    assert callable(kernels.indexed_floor_f32_args)
    assert callable(kernels.indexed_floor_f32_cuda_handle_args)
    assert callable(kernels.indexed_floor_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_acos_f32_kernel)
    assert callable(kernels.indexed_acos_f32_args)
    assert callable(kernels.indexed_acos_f32_cuda_handle_args)
    assert callable(kernels.indexed_acos_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_atan2_f32_kernel)
    assert callable(kernels.indexed_atan2_f32_args)
    assert callable(kernels.indexed_atan2_f32_cuda_handle_args)
    assert callable(kernels.indexed_atan2_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_erf_f32_kernel)
    assert callable(kernels.indexed_erf_f32_args)
    assert callable(kernels.indexed_erf_f32_cuda_handle_args)
    assert callable(kernels.indexed_erf_f32_cuda_dlpack_args)
    assert callable(kernels.indexed_rsqrt_f32_kernel)
    assert callable(kernels.direct_neg_f32_kernel)
    assert callable(kernels.direct_neg_f32_args)
    assert callable(kernels.direct_neg_f32_cuda_handle_args)
    assert callable(kernels.direct_neg_f32_cuda_dlpack_args)
    assert callable(kernels.direct_neg_i32_kernel)
    assert callable(kernels.direct_neg_i32_args)
    assert callable(kernels.direct_neg_i32_cuda_handle_args)
    assert callable(kernels.direct_neg_i32_cuda_dlpack_args)
    assert callable(kernels.direct_abs_f32_kernel)
    assert callable(kernels.direct_abs_f32_args)
    assert callable(kernels.direct_abs_f32_cuda_handle_args)
    assert callable(kernels.direct_abs_f32_cuda_dlpack_args)
    assert callable(kernels.direct_abs_i32_kernel)
    assert callable(kernels.direct_abs_i32_args)
    assert callable(kernels.direct_abs_i32_cuda_handle_args)
    assert callable(kernels.direct_abs_i32_cuda_dlpack_args)
    assert callable(kernels.direct_sqrt_f32_kernel)
    assert callable(kernels.direct_exp2_f32_kernel)
    assert callable(kernels.direct_exp2_f32_args)
    assert callable(kernels.direct_exp2_f32_cuda_handle_args)
    assert callable(kernels.direct_exp2_f32_cuda_dlpack_args)
    assert callable(kernels.direct_atan_f32_kernel)
    assert callable(kernels.direct_atan_f32_args)
    assert callable(kernels.direct_atan_f32_cuda_handle_args)
    assert callable(kernels.direct_atan_f32_cuda_dlpack_args)
    assert callable(kernels.direct_round_f32_kernel)
    assert callable(kernels.direct_round_f32_args)
    assert callable(kernels.direct_round_f32_cuda_handle_args)
    assert callable(kernels.direct_round_f32_cuda_dlpack_args)
    assert callable(kernels.direct_ceil_f32_kernel)
    assert callable(kernels.direct_ceil_f32_args)
    assert callable(kernels.direct_ceil_f32_cuda_handle_args)
    assert callable(kernels.direct_ceil_f32_cuda_dlpack_args)
    assert callable(kernels.direct_acos_f32_kernel)
    assert callable(kernels.direct_acos_f32_args)
    assert callable(kernels.direct_acos_f32_cuda_handle_args)
    assert callable(kernels.direct_acos_f32_cuda_dlpack_args)
    assert callable(kernels.direct_log10_f32_kernel)
    assert callable(kernels.direct_log10_f32_args)
    assert callable(kernels.direct_log10_f32_cuda_handle_args)
    assert callable(kernels.direct_log10_f32_cuda_dlpack_args)
    assert callable(kernels.direct_erf_f32_kernel)
    assert callable(kernels.direct_erf_f32_args)
    assert callable(kernels.direct_erf_f32_cuda_handle_args)
    assert callable(kernels.direct_erf_f32_cuda_dlpack_args)
    assert callable(kernels.direct_rsqrt_f32_kernel)
    assert callable(kernels.indexed_rsqrt_f32_cuda_handle_args)
    assert callable(kernels.direct_sqrt_f32_cuda_handle_args)
    assert callable(kernels.direct_rsqrt_f32_cuda_handle_args)
    assert callable(kernels.reduce_add_f32_kernel)
    assert callable(kernels.reduce_add_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_add_f32_kernel)
    assert callable(kernels.parallel_reduce_add_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_add_i32_kernel)
    assert callable(kernels.parallel_reduce_add_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_add_2d_bundle_f32_kernel)
    assert callable(kernels.parallel_reduce_add_2d_bundle_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_add_2d_bundle_i32_kernel)
    assert callable(kernels.parallel_reduce_add_2d_bundle_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_mul_2d_bundle_f32_kernel)
    assert callable(kernels.parallel_reduce_mul_2d_bundle_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_max_2d_bundle_f32_kernel)
    assert callable(kernels.parallel_reduce_max_2d_bundle_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_min_2d_bundle_f32_kernel)
    assert callable(kernels.parallel_reduce_min_2d_bundle_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_mul_2d_bundle_i32_kernel)
    assert callable(kernels.parallel_reduce_mul_2d_bundle_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_max_2d_bundle_i32_kernel)
    assert callable(kernels.parallel_reduce_max_2d_bundle_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_min_2d_bundle_i32_kernel)
    assert callable(kernels.parallel_reduce_min_2d_bundle_i32_cuda_handle_args)
    assert callable(kernels.tensor_factory_2d_f32_kernel)
    assert callable(kernels.tensor_factory_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_tensor_factory_2d_f32_kernel)
    assert callable(kernels.parallel_tensor_factory_2d_f32_args)
    assert callable(kernels.parallel_tensor_factory_2d_f32_cuda_handle_args)
    assert callable(kernels.tensor_factory_2d_i32_kernel)
    assert callable(kernels.tensor_factory_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_tensor_factory_2d_i32_kernel)
    assert callable(kernels.parallel_tensor_factory_2d_i32_args)
    assert callable(kernels.parallel_tensor_factory_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_add_2d_bundle_f32_kernel)
    assert callable(kernels.reduce_add_2d_bundle_f32_cuda_handle_args)
    assert callable(kernels.reduce_add_2d_bundle_i32_kernel)
    assert callable(kernels.reduce_add_2d_bundle_i32_cuda_handle_args)
    assert callable(kernels.reduce_rows_add_2d_f32_kernel)
    assert callable(kernels.reduce_rows_add_2d_f32_args)
    assert callable(kernels.reduce_rows_add_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_rows_add_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_rows_add_2d_i32_kernel)
    assert callable(kernels.reduce_rows_add_2d_i32_args)
    assert callable(kernels.reduce_rows_add_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_rows_add_2d_i32_cuda_dlpack_args)
    assert callable(kernels.reduce_rows_mul_2d_f32_kernel)
    assert callable(kernels.reduce_rows_mul_2d_f32_args)
    assert callable(kernels.reduce_rows_mul_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_rows_mul_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_rows_max_2d_f32_kernel)
    assert callable(kernels.reduce_rows_max_2d_f32_args)
    assert callable(kernels.reduce_rows_max_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_rows_max_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_rows_min_2d_f32_kernel)
    assert callable(kernels.reduce_rows_min_2d_f32_args)
    assert callable(kernels.reduce_rows_min_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_rows_min_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_rows_mul_2d_i32_kernel)
    assert callable(kernels.reduce_rows_mul_2d_i32_args)
    assert callable(kernels.reduce_rows_mul_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_rows_mul_2d_i32_cuda_dlpack_args)
    assert callable(kernels.reduce_rows_max_2d_i32_kernel)
    assert callable(kernels.reduce_rows_max_2d_i32_args)
    assert callable(kernels.reduce_rows_max_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_rows_max_2d_i32_cuda_dlpack_args)
    assert callable(kernels.reduce_rows_min_2d_i32_kernel)
    assert callable(kernels.reduce_rows_min_2d_i32_args)
    assert callable(kernels.reduce_rows_min_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_rows_min_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_add_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_rows_add_2d_f32_args)
    assert callable(kernels.parallel_reduce_rows_add_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_add_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_add_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_rows_add_2d_i32_args)
    assert callable(kernels.parallel_reduce_rows_add_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_add_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_mul_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_rows_mul_2d_f32_args)
    assert callable(kernels.parallel_reduce_rows_mul_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_mul_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_max_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_rows_max_2d_f32_args)
    assert callable(kernels.parallel_reduce_rows_max_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_max_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_min_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_rows_min_2d_f32_args)
    assert callable(kernels.parallel_reduce_rows_min_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_min_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_mul_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_rows_mul_2d_i32_args)
    assert callable(kernels.parallel_reduce_rows_mul_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_mul_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_max_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_rows_max_2d_i32_args)
    assert callable(kernels.parallel_reduce_rows_max_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_max_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_rows_min_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_rows_min_2d_i32_args)
    assert callable(kernels.parallel_reduce_rows_min_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_rows_min_2d_i32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_add_2d_f32_kernel)
    assert callable(kernels.reduce_cols_add_2d_f32_args)
    assert callable(kernels.reduce_cols_add_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_cols_add_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_add_2d_i32_kernel)
    assert callable(kernels.reduce_cols_add_2d_i32_args)
    assert callable(kernels.reduce_cols_add_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_cols_add_2d_i32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_mul_2d_f32_kernel)
    assert callable(kernels.reduce_cols_mul_2d_f32_args)
    assert callable(kernels.reduce_cols_mul_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_cols_mul_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_max_2d_f32_kernel)
    assert callable(kernels.reduce_cols_max_2d_f32_args)
    assert callable(kernels.reduce_cols_max_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_cols_max_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_min_2d_f32_kernel)
    assert callable(kernels.reduce_cols_min_2d_f32_args)
    assert callable(kernels.reduce_cols_min_2d_f32_cuda_handle_args)
    assert callable(kernels.reduce_cols_min_2d_f32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_mul_2d_i32_kernel)
    assert callable(kernels.reduce_cols_mul_2d_i32_args)
    assert callable(kernels.reduce_cols_mul_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_cols_mul_2d_i32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_max_2d_i32_kernel)
    assert callable(kernels.reduce_cols_max_2d_i32_args)
    assert callable(kernels.reduce_cols_max_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_cols_max_2d_i32_cuda_dlpack_args)
    assert callable(kernels.reduce_cols_min_2d_i32_kernel)
    assert callable(kernels.reduce_cols_min_2d_i32_args)
    assert callable(kernels.reduce_cols_min_2d_i32_cuda_handle_args)
    assert callable(kernels.reduce_cols_min_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_add_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_cols_add_2d_f32_args)
    assert callable(kernels.parallel_reduce_cols_add_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_add_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_add_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_cols_add_2d_i32_args)
    assert callable(kernels.parallel_reduce_cols_add_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_add_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_mul_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_cols_mul_2d_f32_args)
    assert callable(kernels.parallel_reduce_cols_mul_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_mul_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_max_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_cols_max_2d_f32_args)
    assert callable(kernels.parallel_reduce_cols_max_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_max_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_min_2d_f32_kernel)
    assert callable(kernels.parallel_reduce_cols_min_2d_f32_args)
    assert callable(kernels.parallel_reduce_cols_min_2d_f32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_min_2d_f32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_mul_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_cols_mul_2d_i32_args)
    assert callable(kernels.parallel_reduce_cols_mul_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_mul_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_max_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_cols_max_2d_i32_args)
    assert callable(kernels.parallel_reduce_cols_max_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_max_2d_i32_cuda_dlpack_args)
    assert callable(kernels.parallel_reduce_cols_min_2d_i32_kernel)
    assert callable(kernels.parallel_reduce_cols_min_2d_i32_args)
    assert callable(kernels.parallel_reduce_cols_min_2d_i32_cuda_handle_args)
    assert callable(kernels.parallel_reduce_cols_min_2d_i32_cuda_dlpack_args)
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


def test_ptx_parallel_tensor_factory_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    f32_payload = kernels.parallel_tensor_factory_2d_f32_args()
    f32_cuda_handle_payload = kernels.parallel_tensor_factory_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    i32_payload = kernels.parallel_tensor_factory_2d_i32_args()
    i32_cuda_handle_payload = kernels.parallel_tensor_factory_2d_i32_cuda_handle_args(backend_name="ptx_exec")

    f32_args = f32_payload["args"]
    f32_cuda_compile_args = f32_cuda_handle_payload["compile_args"]
    f32_cuda_run_args = f32_cuda_handle_payload["run_args"]
    i32_args = i32_payload["args"]
    i32_cuda_compile_args = i32_cuda_handle_payload["compile_args"]
    i32_cuda_run_args = i32_cuda_handle_payload["run_args"]

    assert len(f32_args) == 3
    assert f32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert f32_payload["result_indices"] == ()
    assert len(f32_cuda_compile_args) == 3
    assert len(f32_cuda_run_args) == 3
    assert f32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert f32_cuda_handle_payload["result_indices"] == ()

    assert len(i32_args) == 3
    assert i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(i32_args[0].dtype) == "i32"
    assert i32_payload["result_indices"] == ()
    assert len(i32_cuda_compile_args) == 3
    assert len(i32_cuda_run_args) == 3
    assert i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(i32_cuda_compile_args[0].dtype) == "i32"
    assert i32_cuda_handle_payload["result_indices"] == ()


def test_ptx_reduce_rows_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    f32_payload = kernels.reduce_rows_add_2d_f32_args()
    f32_cuda_handle_payload = kernels.reduce_rows_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    f32_cuda_dlpack_payload = kernels.reduce_rows_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    i32_payload = kernels.reduce_rows_add_2d_i32_args()
    i32_cuda_handle_payload = kernels.reduce_rows_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    i32_cuda_dlpack_payload = kernels.reduce_rows_add_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_mul_f32_payload = kernels.reduce_rows_mul_2d_f32_args()
    serial_mul_f32_cuda_dlpack_payload = kernels.reduce_rows_mul_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_mul_i32_payload = kernels.reduce_rows_mul_2d_i32_args()
    serial_mul_i32_cuda_dlpack_payload = kernels.reduce_rows_mul_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_max_f32_payload = kernels.reduce_rows_max_2d_f32_args()
    serial_max_f32_cuda_dlpack_payload = kernels.reduce_rows_max_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_max_i32_payload = kernels.reduce_rows_max_2d_i32_args()
    serial_max_i32_cuda_dlpack_payload = kernels.reduce_rows_max_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_min_f32_payload = kernels.reduce_rows_min_2d_f32_args()
    serial_min_f32_cuda_dlpack_payload = kernels.reduce_rows_min_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_min_i32_payload = kernels.reduce_rows_min_2d_i32_args()
    serial_min_i32_cuda_dlpack_payload = kernels.reduce_rows_min_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_f32_payload = kernels.parallel_reduce_rows_add_2d_f32_args()
    parallel_f32_cuda_handle_payload = kernels.parallel_reduce_rows_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_f32_cuda_dlpack_payload = kernels.parallel_reduce_rows_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_i32_payload = kernels.parallel_reduce_rows_add_2d_i32_args()
    parallel_i32_cuda_handle_payload = kernels.parallel_reduce_rows_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_i32_cuda_dlpack_payload = kernels.parallel_reduce_rows_add_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_mul_f32_payload = kernels.parallel_reduce_rows_mul_2d_f32_args()
    parallel_mul_f32_cuda_dlpack_payload = kernels.parallel_reduce_rows_mul_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_mul_i32_payload = kernels.parallel_reduce_rows_mul_2d_i32_args()
    parallel_mul_i32_cuda_dlpack_payload = kernels.parallel_reduce_rows_mul_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_max_f32_payload = kernels.parallel_reduce_rows_max_2d_f32_args()
    parallel_max_f32_cuda_dlpack_payload = kernels.parallel_reduce_rows_max_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_max_i32_payload = kernels.parallel_reduce_rows_max_2d_i32_args()
    parallel_max_i32_cuda_dlpack_payload = kernels.parallel_reduce_rows_max_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_min_f32_payload = kernels.parallel_reduce_rows_min_2d_f32_args()
    parallel_min_f32_cuda_dlpack_payload = kernels.parallel_reduce_rows_min_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_min_i32_payload = kernels.parallel_reduce_rows_min_2d_i32_args()
    parallel_min_i32_cuda_dlpack_payload = kernels.parallel_reduce_rows_min_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")

    f32_args = f32_payload["args"]
    f32_cuda_compile_args = f32_cuda_handle_payload["compile_args"]
    f32_cuda_run_args = f32_cuda_handle_payload["run_args"]
    f32_cuda_dlpack_compile_args = f32_cuda_dlpack_payload["compile_args"]
    f32_cuda_dlpack_run_args = f32_cuda_dlpack_payload["run_args"]
    i32_args = i32_payload["args"]
    i32_cuda_compile_args = i32_cuda_handle_payload["compile_args"]
    i32_cuda_run_args = i32_cuda_handle_payload["run_args"]
    i32_cuda_dlpack_compile_args = i32_cuda_dlpack_payload["compile_args"]
    i32_cuda_dlpack_run_args = i32_cuda_dlpack_payload["run_args"]
    parallel_f32_args = parallel_f32_payload["args"]
    parallel_f32_cuda_compile_args = parallel_f32_cuda_handle_payload["compile_args"]
    parallel_f32_cuda_run_args = parallel_f32_cuda_handle_payload["run_args"]
    parallel_f32_cuda_dlpack_compile_args = parallel_f32_cuda_dlpack_payload["compile_args"]
    parallel_f32_cuda_dlpack_run_args = parallel_f32_cuda_dlpack_payload["run_args"]
    parallel_i32_args = parallel_i32_payload["args"]
    parallel_i32_cuda_compile_args = parallel_i32_cuda_handle_payload["compile_args"]
    parallel_i32_cuda_run_args = parallel_i32_cuda_handle_payload["run_args"]
    parallel_i32_cuda_dlpack_compile_args = parallel_i32_cuda_dlpack_payload["compile_args"]
    parallel_i32_cuda_dlpack_run_args = parallel_i32_cuda_dlpack_payload["run_args"]

    assert len(f32_args) == 2
    assert f32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert f32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert f32_payload["result_indices"] == ()
    assert len(f32_cuda_compile_args) == 2
    assert f32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert f32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert len(f32_cuda_run_args) == 2
    assert f32_cuda_handle_payload["result_indices"] == ()
    assert len(f32_cuda_dlpack_compile_args) == 2
    assert f32_cuda_dlpack_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert f32_cuda_dlpack_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert len(f32_cuda_dlpack_run_args) == 2
    assert f32_cuda_dlpack_payload["result_indices"] == ()

    assert len(i32_args) == 2
    assert i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert i32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert str(i32_args[0].dtype) == "i32"
    assert i32_payload["result_indices"] == ()
    assert len(i32_cuda_compile_args) == 2
    assert i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert i32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert str(i32_cuda_compile_args[0].dtype) == "i32"
    assert len(i32_cuda_run_args) == 2
    assert i32_cuda_handle_payload["result_indices"] == ()
    assert len(i32_cuda_dlpack_compile_args) == 2
    assert i32_cuda_dlpack_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert i32_cuda_dlpack_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert str(i32_cuda_dlpack_compile_args[0].dtype) == "i32"
    assert len(i32_cuda_dlpack_run_args) == 2
    assert i32_cuda_dlpack_payload["result_indices"] == ()

    assert len(parallel_f32_args) == 2
    assert parallel_f32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_f32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_f32_payload["result_indices"] == ()
    assert len(parallel_f32_cuda_compile_args) == 2
    assert parallel_f32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_f32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert len(parallel_f32_cuda_run_args) == 2
    assert parallel_f32_cuda_handle_payload["result_indices"] == ()
    assert len(parallel_f32_cuda_dlpack_compile_args) == 2
    assert parallel_f32_cuda_dlpack_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_f32_cuda_dlpack_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert len(parallel_f32_cuda_dlpack_run_args) == 2
    assert parallel_f32_cuda_dlpack_payload["result_indices"] == ()

    assert len(parallel_i32_args) == 2
    assert parallel_i32_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_i32_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert str(parallel_i32_args[0].dtype) == "i32"
    assert parallel_i32_payload["result_indices"] == ()
    assert len(parallel_i32_cuda_compile_args) == 2
    assert parallel_i32_cuda_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_i32_cuda_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert str(parallel_i32_cuda_compile_args[0].dtype) == "i32"
    assert len(parallel_i32_cuda_run_args) == 2
    assert parallel_i32_cuda_handle_payload["result_indices"] == ()
    assert len(parallel_i32_cuda_dlpack_compile_args) == 2
    assert parallel_i32_cuda_dlpack_compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_i32_cuda_dlpack_compile_args[1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert str(parallel_i32_cuda_dlpack_compile_args[0].dtype) == "i32"
    assert len(parallel_i32_cuda_dlpack_run_args) == 2
    assert parallel_i32_cuda_dlpack_payload["result_indices"] == ()

    assert parallel_mul_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_max_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_min_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_mul_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_max_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert parallel_min_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert str(parallel_mul_i32_payload["args"][0].dtype) == "i32"
    assert len(parallel_mul_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_max_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_min_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_mul_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_max_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_min_i32_cuda_dlpack_payload["run_args"]) == 2
    assert serial_mul_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert serial_max_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert serial_min_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert serial_mul_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert serial_max_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert serial_min_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert len(serial_mul_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_max_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_min_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_mul_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_max_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_min_i32_cuda_dlpack_payload["run_args"]) == 2


def test_ptx_reduce_cols_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    f32_payload = kernels.reduce_cols_add_2d_f32_args()
    f32_cuda_handle_payload = kernels.reduce_cols_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    f32_cuda_dlpack_payload = kernels.reduce_cols_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    i32_payload = kernels.reduce_cols_add_2d_i32_args()
    i32_cuda_handle_payload = kernels.reduce_cols_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    i32_cuda_dlpack_payload = kernels.reduce_cols_add_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_mul_f32_payload = kernels.reduce_cols_mul_2d_f32_args()
    serial_mul_f32_cuda_dlpack_payload = kernels.reduce_cols_mul_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_mul_i32_payload = kernels.reduce_cols_mul_2d_i32_args()
    serial_mul_i32_cuda_dlpack_payload = kernels.reduce_cols_mul_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_max_f32_payload = kernels.reduce_cols_max_2d_f32_args()
    serial_max_f32_cuda_dlpack_payload = kernels.reduce_cols_max_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_max_i32_payload = kernels.reduce_cols_max_2d_i32_args()
    serial_max_i32_cuda_dlpack_payload = kernels.reduce_cols_max_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_min_f32_payload = kernels.reduce_cols_min_2d_f32_args()
    serial_min_f32_cuda_dlpack_payload = kernels.reduce_cols_min_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    serial_min_i32_payload = kernels.reduce_cols_min_2d_i32_args()
    serial_min_i32_cuda_dlpack_payload = kernels.reduce_cols_min_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_f32_payload = kernels.parallel_reduce_cols_add_2d_f32_args()
    parallel_f32_cuda_handle_payload = kernels.parallel_reduce_cols_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_f32_cuda_dlpack_payload = kernels.parallel_reduce_cols_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_i32_payload = kernels.parallel_reduce_cols_add_2d_i32_args()
    parallel_i32_cuda_handle_payload = kernels.parallel_reduce_cols_add_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_i32_cuda_dlpack_payload = kernels.parallel_reduce_cols_add_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_mul_f32_payload = kernels.parallel_reduce_cols_mul_2d_f32_args()
    parallel_mul_f32_cuda_dlpack_payload = kernels.parallel_reduce_cols_mul_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_mul_i32_payload = kernels.parallel_reduce_cols_mul_2d_i32_args()
    parallel_mul_i32_cuda_dlpack_payload = kernels.parallel_reduce_cols_mul_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_max_f32_payload = kernels.parallel_reduce_cols_max_2d_f32_args()
    parallel_max_f32_cuda_dlpack_payload = kernels.parallel_reduce_cols_max_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_max_i32_payload = kernels.parallel_reduce_cols_max_2d_i32_args()
    parallel_max_i32_cuda_dlpack_payload = kernels.parallel_reduce_cols_max_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_min_f32_payload = kernels.parallel_reduce_cols_min_2d_f32_args()
    parallel_min_f32_cuda_dlpack_payload = kernels.parallel_reduce_cols_min_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_min_i32_payload = kernels.parallel_reduce_cols_min_2d_i32_args()
    parallel_min_i32_cuda_dlpack_payload = kernels.parallel_reduce_cols_min_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")

    for payload in (
        f32_payload,
        i32_payload,
        parallel_f32_payload,
        parallel_i32_payload,
    ):
        assert len(payload["args"]) == 2
        assert payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
        assert payload["result_indices"] == ()

    for payload in (
        f32_cuda_handle_payload,
        f32_cuda_dlpack_payload,
        i32_cuda_handle_payload,
        i32_cuda_dlpack_payload,
        parallel_f32_cuda_handle_payload,
        parallel_f32_cuda_dlpack_payload,
        parallel_i32_cuda_handle_payload,
        parallel_i32_cuda_dlpack_payload,
    ):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["compile_args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    assert str(i32_payload["args"][0].dtype) == "i32"
    assert str(i32_cuda_handle_payload["compile_args"][0].dtype) == "i32"
    assert str(i32_cuda_dlpack_payload["compile_args"][0].dtype) == "i32"
    assert str(parallel_i32_payload["args"][0].dtype) == "i32"
    assert str(parallel_i32_cuda_handle_payload["compile_args"][0].dtype) == "i32"
    assert str(parallel_i32_cuda_dlpack_payload["compile_args"][0].dtype) == "i32"
    assert parallel_mul_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert parallel_max_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert parallel_min_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert parallel_mul_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert parallel_max_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert parallel_min_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert str(parallel_mul_i32_payload["args"][0].dtype) == "i32"
    assert len(parallel_mul_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_max_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_min_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_mul_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_max_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(parallel_min_i32_cuda_dlpack_payload["run_args"]) == 2
    assert serial_mul_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert serial_max_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert serial_min_f32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert serial_mul_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert serial_max_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert serial_min_i32_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert len(serial_mul_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_max_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_min_f32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_mul_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_max_i32_cuda_dlpack_payload["run_args"]) == 2
    assert len(serial_min_i32_cuda_dlpack_payload["run_args"]) == 2


def test_ptx_integer_bitwise_reduction_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    row_and_payload = kernels.parallel_reduce_rows_and_2d_i32_args()
    row_and_cuda_handle_payload = kernels.parallel_reduce_rows_and_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    row_and_cuda_dlpack_payload = kernels.parallel_reduce_rows_and_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    col_or_payload = kernels.parallel_reduce_cols_or_2d_i32_args()
    col_or_cuda_handle_payload = kernels.parallel_reduce_cols_or_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    col_or_cuda_dlpack_payload = kernels.parallel_reduce_cols_or_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    bundle_xor_payload = kernels.parallel_reduce_xor_2d_bundle_i32_args()
    bundle_xor_cuda_handle_payload = kernels.parallel_reduce_xor_2d_bundle_i32_cuda_handle_args(backend_name="ptx_exec")
    bundle_xor_cuda_dlpack_payload = kernels.parallel_reduce_xor_2d_bundle_i32_cuda_dlpack_args(backend_name="ptx_exec")

    for payload in (row_and_payload, col_or_payload):
        assert len(payload["args"]) == 2
        assert payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(payload["args"][0].dtype) == "i32"
        assert payload["result_indices"] == ()

    assert row_and_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert col_or_payload["args"][1].shape == (kernels.FLYDSL_MICRO_COLS,)

    for payload, expected_shape in (
        (row_and_cuda_handle_payload, (kernels.FLYDSL_MICRO_ROWS,)),
        (row_and_cuda_dlpack_payload, (kernels.FLYDSL_MICRO_ROWS,)),
        (col_or_cuda_handle_payload, (kernels.FLYDSL_MICRO_COLS,)),
        (col_or_cuda_dlpack_payload, (kernels.FLYDSL_MICRO_COLS,)),
    ):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["compile_args"][1].shape == expected_shape
        assert str(payload["compile_args"][0].dtype) == "i32"
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    assert len(bundle_xor_payload["args"]) == 4
    assert bundle_xor_payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert bundle_xor_payload["args"][1].shape == (1,)
    assert bundle_xor_payload["args"][2].shape == (kernels.FLYDSL_MICRO_ROWS,)
    assert bundle_xor_payload["args"][3].shape == (kernels.FLYDSL_MICRO_COLS,)
    assert str(bundle_xor_payload["args"][0].dtype) == "i32"
    assert bundle_xor_payload["result_indices"] == ()

    for payload in (bundle_xor_cuda_handle_payload, bundle_xor_cuda_dlpack_payload):
        assert len(payload["compile_args"]) == 4
        assert payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["compile_args"][1].shape == (1,)
        assert payload["compile_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS,)
        assert payload["compile_args"][3].shape == (kernels.FLYDSL_MICRO_COLS,)
        assert str(payload["compile_args"][0].dtype) == "i32"
        assert len(payload["run_args"]) == 4
        assert payload["result_indices"] == ()


def test_ptx_copy_reduce_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    rank1_f32_payload = kernels.copy_reduce_add_f32_args()
    rank1_f32_cuda_handle_payload = kernels.copy_reduce_add_f32_cuda_handle_args(backend_name="ptx_exec")
    rank1_f32_cuda_dlpack_payload = kernels.copy_reduce_add_f32_cuda_dlpack_args(backend_name="ptx_exec")
    rank1_f32_max_payload = kernels.copy_reduce_max_f32_args()
    rank1_f32_max_cuda_handle_payload = kernels.copy_reduce_max_f32_cuda_handle_args(backend_name="ptx_exec")
    rank1_f32_max_cuda_dlpack_payload = kernels.copy_reduce_max_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_rank1_f32_payload = kernels.indexed_copy_reduce_add_f32_args()
    indexed_rank1_f32_cuda_handle_payload = kernels.indexed_copy_reduce_add_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_rank1_f32_cuda_dlpack_payload = kernels.indexed_copy_reduce_add_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_rank1_f32_max_payload = kernels.indexed_copy_reduce_max_f32_args()
    indexed_rank1_f32_max_cuda_handle_payload = kernels.indexed_copy_reduce_max_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_rank1_f32_max_cuda_dlpack_payload = kernels.indexed_copy_reduce_max_f32_cuda_dlpack_args(backend_name="ptx_exec")
    rank1_i32_payload = kernels.copy_reduce_xor_i32_args()
    rank1_i32_cuda_handle_payload = kernels.copy_reduce_xor_i32_cuda_handle_args(backend_name="ptx_exec")
    rank1_i32_cuda_dlpack_payload = kernels.copy_reduce_xor_i32_cuda_dlpack_args(backend_name="ptx_exec")
    rank1_i32_or_payload = kernels.copy_reduce_or_i32_args()
    rank1_i32_or_cuda_handle_payload = kernels.copy_reduce_or_i32_cuda_handle_args(backend_name="ptx_exec")
    rank1_i32_or_cuda_dlpack_payload = kernels.copy_reduce_or_i32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_rank1_i32_payload = kernels.indexed_copy_reduce_xor_i32_args()
    indexed_rank1_i32_cuda_handle_payload = kernels.indexed_copy_reduce_xor_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_rank1_i32_cuda_dlpack_payload = kernels.indexed_copy_reduce_xor_i32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_rank1_i32_or_payload = kernels.indexed_copy_reduce_or_i32_args()
    indexed_rank1_i32_or_cuda_handle_payload = kernels.indexed_copy_reduce_or_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_rank1_i32_or_cuda_dlpack_payload = kernels.indexed_copy_reduce_or_i32_cuda_dlpack_args(backend_name="ptx_exec")
    tensor2d_payload = kernels.copy_reduce_add_2d_f32_args()
    tensor2d_cuda_handle_payload = kernels.copy_reduce_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    tensor2d_cuda_dlpack_payload = kernels.copy_reduce_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    tensor2d_max_payload = kernels.copy_reduce_max_2d_f32_args()
    tensor2d_max_cuda_handle_payload = kernels.copy_reduce_max_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    tensor2d_max_cuda_dlpack_payload = kernels.copy_reduce_max_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    tensor2d_or_i32_payload = kernels.copy_reduce_or_2d_i32_args()
    tensor2d_or_i32_cuda_handle_payload = kernels.copy_reduce_or_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    tensor2d_or_i32_cuda_dlpack_payload = kernels.copy_reduce_or_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_tensor2d_payload = kernels.parallel_copy_reduce_add_2d_f32_args()
    parallel_tensor2d_cuda_handle_payload = kernels.parallel_copy_reduce_add_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_tensor2d_cuda_dlpack_payload = kernels.parallel_copy_reduce_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_tensor2d_max_payload = kernels.parallel_copy_reduce_max_2d_f32_args()
    parallel_tensor2d_max_cuda_handle_payload = kernels.parallel_copy_reduce_max_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_tensor2d_max_cuda_dlpack_payload = kernels.parallel_copy_reduce_max_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_tensor2d_or_i32_payload = kernels.parallel_copy_reduce_or_2d_i32_args()
    parallel_tensor2d_or_i32_cuda_handle_payload = kernels.parallel_copy_reduce_or_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_tensor2d_or_i32_cuda_dlpack_payload = kernels.parallel_copy_reduce_or_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")

    for payload in (
        rank1_f32_payload,
        rank1_f32_max_payload,
        indexed_rank1_f32_payload,
        indexed_rank1_f32_max_payload,
        rank1_i32_payload,
        rank1_i32_or_payload,
        indexed_rank1_i32_payload,
        indexed_rank1_i32_or_payload,
    ):
        assert len(payload["args"]) == 2
        assert payload["args"][0].shape == (kernels.POINTWISE_N,)
        assert payload["args"][1].shape == (kernels.POINTWISE_N,)
        assert payload["result_indices"] == ()

    for payload in (
        rank1_f32_cuda_handle_payload,
        rank1_f32_cuda_dlpack_payload,
        rank1_f32_max_cuda_handle_payload,
        rank1_f32_max_cuda_dlpack_payload,
        indexed_rank1_f32_cuda_handle_payload,
        indexed_rank1_f32_cuda_dlpack_payload,
        indexed_rank1_f32_max_cuda_handle_payload,
        indexed_rank1_f32_max_cuda_dlpack_payload,
        rank1_i32_cuda_handle_payload,
        rank1_i32_cuda_dlpack_payload,
        rank1_i32_or_cuda_handle_payload,
        rank1_i32_or_cuda_dlpack_payload,
        indexed_rank1_i32_cuda_handle_payload,
        indexed_rank1_i32_cuda_dlpack_payload,
        indexed_rank1_i32_or_cuda_handle_payload,
        indexed_rank1_i32_or_cuda_dlpack_payload,
    ):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
        assert payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    assert str(rank1_i32_payload["args"][0].dtype) == "i32"
    assert str(rank1_i32_payload["args"][1].dtype) == "i32"
    assert str(rank1_i32_or_payload["args"][0].dtype) == "i32"
    assert str(rank1_i32_or_payload["args"][1].dtype) == "i32"
    assert str(indexed_rank1_i32_payload["args"][0].dtype) == "i32"
    assert str(indexed_rank1_i32_payload["args"][1].dtype) == "i32"
    assert str(indexed_rank1_i32_or_payload["args"][0].dtype) == "i32"
    assert str(indexed_rank1_i32_or_payload["args"][1].dtype) == "i32"
    assert str(rank1_i32_cuda_handle_payload["compile_args"][0].dtype) == "i32"
    assert str(rank1_i32_cuda_handle_payload["compile_args"][1].dtype) == "i32"
    assert str(rank1_i32_cuda_dlpack_payload["compile_args"][0].dtype) == "i32"
    assert str(rank1_i32_cuda_dlpack_payload["compile_args"][1].dtype) == "i32"
    assert str(rank1_i32_or_cuda_handle_payload["compile_args"][0].dtype) == "i32"
    assert str(rank1_i32_or_cuda_handle_payload["compile_args"][1].dtype) == "i32"
    assert str(rank1_i32_or_cuda_dlpack_payload["compile_args"][0].dtype) == "i32"
    assert str(rank1_i32_or_cuda_dlpack_payload["compile_args"][1].dtype) == "i32"
    assert str(indexed_rank1_i32_cuda_handle_payload["compile_args"][0].dtype) == "i32"
    assert str(indexed_rank1_i32_cuda_handle_payload["compile_args"][1].dtype) == "i32"
    assert str(indexed_rank1_i32_cuda_dlpack_payload["compile_args"][0].dtype) == "i32"
    assert str(indexed_rank1_i32_cuda_dlpack_payload["compile_args"][1].dtype) == "i32"
    assert str(indexed_rank1_i32_or_cuda_handle_payload["compile_args"][0].dtype) == "i32"
    assert str(indexed_rank1_i32_or_cuda_handle_payload["compile_args"][1].dtype) == "i32"
    assert str(indexed_rank1_i32_or_cuda_dlpack_payload["compile_args"][0].dtype) == "i32"
    assert str(indexed_rank1_i32_or_cuda_dlpack_payload["compile_args"][1].dtype) == "i32"

    for payload in (
        tensor2d_payload,
        tensor2d_max_payload,
        tensor2d_or_i32_payload,
        parallel_tensor2d_payload,
        parallel_tensor2d_max_payload,
        parallel_tensor2d_or_i32_payload,
    ):
        assert len(payload["args"]) == 2
        assert payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["result_indices"] == ()

    for payload in (
        tensor2d_cuda_handle_payload,
        tensor2d_cuda_dlpack_payload,
        tensor2d_max_cuda_handle_payload,
        tensor2d_max_cuda_dlpack_payload,
        tensor2d_or_i32_cuda_handle_payload,
        tensor2d_or_i32_cuda_dlpack_payload,
        parallel_tensor2d_cuda_handle_payload,
        parallel_tensor2d_cuda_dlpack_payload,
        parallel_tensor2d_max_cuda_handle_payload,
        parallel_tensor2d_max_cuda_dlpack_payload,
        parallel_tensor2d_or_i32_cuda_handle_payload,
        parallel_tensor2d_or_i32_cuda_dlpack_payload,
    ):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["compile_args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()


def test_ptx_multiblock_row_tiled_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    payloads = (
        kernels.multiblock_dense_copy_2d_f32_args(),
        kernels.multiblock_dense_add_2d_f32_args(),
        kernels.multiblock_dense_tensor_scalar_add_2d_f32_args(),
        kernels.multiblock_copy_reduce_add_2d_f32_args(),
        kernels.multiblock_tensor_factory_2d_f32_args(),
        kernels.multiblock_reduce_rows_add_2d_f32_args(),
        kernels.multiblock_reduce_cols_add_2d_f32_args(),
        kernels.multiblock_reduce_add_rowcol_2d_f32_args(),
    )
    cuda_handle_payloads = (
        kernels.multiblock_dense_copy_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
        kernels.multiblock_dense_add_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
        kernels.multiblock_dense_tensor_scalar_add_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
        kernels.multiblock_copy_reduce_add_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
        kernels.multiblock_tensor_factory_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
        kernels.multiblock_reduce_rows_add_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
        kernels.multiblock_reduce_cols_add_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
        kernels.multiblock_reduce_add_rowcol_2d_f32_cuda_handle_args(backend_name="ptx_exec"),
    )
    cuda_dlpack_payloads = (
        kernels.multiblock_dense_copy_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
        kernels.multiblock_dense_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
        kernels.multiblock_dense_tensor_scalar_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
        kernels.multiblock_copy_reduce_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
        kernels.multiblock_tensor_factory_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
        kernels.multiblock_reduce_rows_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
        kernels.multiblock_reduce_cols_add_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
        kernels.multiblock_reduce_add_rowcol_2d_f32_cuda_dlpack_args(backend_name="ptx_exec"),
    )

    for payload in payloads:
        args = payload["args"]
        assert payload["result_indices"] == ()
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        if len(args) > 1 and hasattr(args[1], "shape"):
            assert args[1].shape in {
                (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
                (1,),
                (kernels.FLYDSL_MICRO_ROWS,),
                (kernels.FLYDSL_MICRO_COLS,),
            }
        if len(args) > 2 and hasattr(args[2], "shape"):
            assert args[2].shape in {
                (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
                (kernels.FLYDSL_MICRO_COLS,),
            }

    for payload in (*cuda_handle_payloads, *cuda_dlpack_payloads):
        compile_args = payload["compile_args"]
        run_args = payload["run_args"]
        assert payload["result_indices"] == ()
        assert compile_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert len(run_args) == len(compile_args)


def test_ptx_integer_bitwise_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_bitand_i32_args()
    indexed_cuda_handle_payload = kernels.indexed_bitand_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_bitand_i32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_or_payload = kernels.indexed_bitor_i32_args()
    indexed_or_cuda_handle_payload = kernels.indexed_bitor_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_or_cuda_dlpack_payload = kernels.indexed_bitor_i32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_xor_payload = kernels.indexed_bitxor_i32_args()
    indexed_xor_cuda_handle_payload = kernels.indexed_bitxor_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_xor_cuda_dlpack_payload = kernels.indexed_bitxor_i32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_payload = kernels.direct_bitand_i32_args()
    direct_cuda_handle_payload = kernels.direct_bitand_i32_cuda_handle_args(backend_name="ptx_exec")
    direct_cuda_dlpack_payload = kernels.direct_bitand_i32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_or_payload = kernels.direct_bitor_i32_args()
    direct_or_cuda_handle_payload = kernels.direct_bitor_i32_cuda_handle_args(backend_name="ptx_exec")
    direct_or_cuda_dlpack_payload = kernels.direct_bitor_i32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_scalar_or_payload = kernels.direct_scalar_broadcast_bitor_i32_args()
    direct_scalar_or_cuda_handle_payload = kernels.direct_scalar_broadcast_bitor_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    direct_scalar_or_cuda_dlpack_payload = kernels.direct_scalar_broadcast_bitor_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    direct_scalar_and_payload = kernels.direct_scalar_broadcast_bitand_i32_args()
    direct_scalar_and_cuda_handle_payload = kernels.direct_scalar_broadcast_bitand_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    direct_scalar_and_cuda_dlpack_payload = kernels.direct_scalar_broadcast_bitand_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    indexed_scalar_payload = kernels.indexed_scalar_broadcast_bitor_i32_args()
    indexed_scalar_cuda_handle_payload = kernels.indexed_scalar_broadcast_bitor_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_scalar_cuda_dlpack_payload = kernels.indexed_scalar_broadcast_bitor_i32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_tensor_scalar_payload = kernels.indexed_tensor_scalar_bitand_i32_args()
    indexed_tensor_scalar_cuda_handle_payload = kernels.indexed_tensor_scalar_bitand_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    indexed_tensor_scalar_cuda_dlpack_payload = kernels.indexed_tensor_scalar_bitand_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    indexed_tensor_scalar_or_payload = kernels.indexed_tensor_scalar_bitor_i32_args()
    indexed_tensor_scalar_or_cuda_handle_payload = kernels.indexed_tensor_scalar_bitor_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    indexed_tensor_scalar_or_cuda_dlpack_payload = kernels.indexed_tensor_scalar_bitor_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )

    assert len(indexed_payload["args"]) == 3
    assert indexed_payload["args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_payload["args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_payload["args"][2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_payload["args"][0].dtype) == "i32"
    assert str(indexed_payload["args"][1].dtype) == "i32"
    assert str(indexed_payload["args"][2].dtype) == "i32"
    assert indexed_payload["result_indices"] == ()

    assert len(indexed_cuda_handle_payload["compile_args"]) == 3
    assert indexed_cuda_handle_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_cuda_handle_payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_cuda_handle_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_cuda_handle_payload["run_args"]) == 3
    assert indexed_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_cuda_dlpack_payload["compile_args"]) == 3
    assert indexed_cuda_dlpack_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_cuda_dlpack_payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_cuda_dlpack_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_cuda_dlpack_payload["run_args"]) == 3
    assert indexed_cuda_dlpack_payload["result_indices"] == ()

    assert len(indexed_or_payload["args"]) == 3
    assert indexed_or_payload["args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_or_payload["args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_or_payload["args"][2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_or_payload["args"][0].dtype) == "i32"
    assert str(indexed_or_payload["args"][1].dtype) == "i32"
    assert str(indexed_or_payload["args"][2].dtype) == "i32"
    assert indexed_or_payload["result_indices"] == ()

    assert len(indexed_or_cuda_handle_payload["compile_args"]) == 3
    assert indexed_or_cuda_handle_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_or_cuda_handle_payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_or_cuda_handle_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_or_cuda_handle_payload["run_args"]) == 3
    assert indexed_or_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_or_cuda_dlpack_payload["compile_args"]) == 3
    assert indexed_or_cuda_dlpack_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_or_cuda_dlpack_payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_or_cuda_dlpack_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_or_cuda_dlpack_payload["run_args"]) == 3
    assert indexed_or_cuda_dlpack_payload["result_indices"] == ()

    assert len(indexed_xor_payload["args"]) == 3
    assert indexed_xor_payload["args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_xor_payload["args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_xor_payload["args"][2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_xor_payload["args"][0].dtype) == "i32"
    assert str(indexed_xor_payload["args"][1].dtype) == "i32"
    assert str(indexed_xor_payload["args"][2].dtype) == "i32"
    assert indexed_xor_payload["result_indices"] == ()

    assert len(indexed_xor_cuda_handle_payload["compile_args"]) == 3
    assert indexed_xor_cuda_handle_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_xor_cuda_handle_payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_xor_cuda_handle_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_xor_cuda_handle_payload["run_args"]) == 3
    assert indexed_xor_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_xor_cuda_dlpack_payload["compile_args"]) == 3
    assert indexed_xor_cuda_dlpack_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_xor_cuda_dlpack_payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_xor_cuda_dlpack_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_xor_cuda_dlpack_payload["run_args"]) == 3
    assert indexed_xor_cuda_dlpack_payload["result_indices"] == ()

    assert len(direct_payload["args"]) == 3
    assert direct_payload["args"][0].shape == (128,)
    assert direct_payload["args"][1].shape == (128,)
    assert direct_payload["args"][2].shape == (128,)
    assert str(direct_payload["args"][0].dtype) == "i32"
    assert str(direct_payload["args"][1].dtype) == "i32"
    assert str(direct_payload["args"][2].dtype) == "i32"
    assert direct_payload["result_indices"] == ()

    assert len(direct_cuda_handle_payload["compile_args"]) == 3
    assert direct_cuda_handle_payload["compile_args"][0].shape == (128,)
    assert direct_cuda_handle_payload["compile_args"][1].shape == (128,)
    assert direct_cuda_handle_payload["compile_args"][2].shape == (128,)
    assert len(direct_cuda_handle_payload["run_args"]) == 3
    assert direct_cuda_handle_payload["result_indices"] == ()

    assert len(direct_cuda_dlpack_payload["compile_args"]) == 3
    assert direct_cuda_dlpack_payload["compile_args"][0].shape == (128,)
    assert direct_cuda_dlpack_payload["compile_args"][1].shape == (128,)
    assert direct_cuda_dlpack_payload["compile_args"][2].shape == (128,)
    assert len(direct_cuda_dlpack_payload["run_args"]) == 3
    assert direct_cuda_dlpack_payload["result_indices"] == ()

    assert len(direct_or_payload["args"]) == 3
    assert direct_or_payload["args"][0].shape == (128,)
    assert direct_or_payload["args"][1].shape == (128,)
    assert direct_or_payload["args"][2].shape == (128,)
    assert str(direct_or_payload["args"][0].dtype) == "i32"
    assert str(direct_or_payload["args"][1].dtype) == "i32"
    assert str(direct_or_payload["args"][2].dtype) == "i32"
    assert direct_or_payload["result_indices"] == ()

    assert len(direct_or_cuda_handle_payload["compile_args"]) == 3
    assert direct_or_cuda_handle_payload["compile_args"][0].shape == (128,)
    assert direct_or_cuda_handle_payload["compile_args"][1].shape == (128,)
    assert direct_or_cuda_handle_payload["compile_args"][2].shape == (128,)
    assert len(direct_or_cuda_handle_payload["run_args"]) == 3
    assert direct_or_cuda_handle_payload["result_indices"] == ()

    assert len(direct_or_cuda_dlpack_payload["compile_args"]) == 3
    assert direct_or_cuda_dlpack_payload["compile_args"][0].shape == (128,)
    assert direct_or_cuda_dlpack_payload["compile_args"][1].shape == (128,)
    assert direct_or_cuda_dlpack_payload["compile_args"][2].shape == (128,)
    assert len(direct_or_cuda_dlpack_payload["run_args"]) == 3
    assert direct_or_cuda_dlpack_payload["result_indices"] == ()

    assert len(direct_scalar_or_payload["args"]) == 3
    assert direct_scalar_or_payload["args"][0].shape == (128,)
    assert int(direct_scalar_or_payload["args"][1]) == 24
    assert direct_scalar_or_payload["args"][2].shape == (128,)
    assert str(direct_scalar_or_payload["args"][0].dtype) == "i32"
    assert str(direct_scalar_or_payload["args"][2].dtype) == "i32"
    assert direct_scalar_or_payload["result_indices"] == ()

    assert len(direct_scalar_or_cuda_handle_payload["compile_args"]) == 3
    assert direct_scalar_or_cuda_handle_payload["compile_args"][0].shape == (128,)
    assert int(direct_scalar_or_cuda_handle_payload["compile_args"][1]) == 24
    assert direct_scalar_or_cuda_handle_payload["compile_args"][2].shape == (128,)
    assert len(direct_scalar_or_cuda_handle_payload["run_args"]) == 3
    assert direct_scalar_or_cuda_handle_payload["run_args"][1] == 24
    assert direct_scalar_or_cuda_handle_payload["result_indices"] == ()

    assert len(direct_scalar_or_cuda_dlpack_payload["compile_args"]) == 3
    assert direct_scalar_or_cuda_dlpack_payload["compile_args"][0].shape == (128,)
    assert int(direct_scalar_or_cuda_dlpack_payload["compile_args"][1]) == 24
    assert direct_scalar_or_cuda_dlpack_payload["compile_args"][2].shape == (128,)
    assert len(direct_scalar_or_cuda_dlpack_payload["run_args"]) == 3
    assert direct_scalar_or_cuda_dlpack_payload["run_args"][1] == 24
    assert direct_scalar_or_cuda_dlpack_payload["result_indices"] == ()

    assert len(direct_scalar_and_payload["args"]) == 3
    assert direct_scalar_and_payload["args"][0].shape == (128,)
    assert int(direct_scalar_and_payload["args"][1]) == 11
    assert direct_scalar_and_payload["args"][2].shape == (128,)
    assert str(direct_scalar_and_payload["args"][0].dtype) == "i32"
    assert str(direct_scalar_and_payload["args"][2].dtype) == "i32"
    assert direct_scalar_and_payload["result_indices"] == ()

    assert len(direct_scalar_and_cuda_handle_payload["compile_args"]) == 3
    assert direct_scalar_and_cuda_handle_payload["compile_args"][0].shape == (128,)
    assert int(direct_scalar_and_cuda_handle_payload["compile_args"][1]) == 11
    assert direct_scalar_and_cuda_handle_payload["compile_args"][2].shape == (128,)
    assert len(direct_scalar_and_cuda_handle_payload["run_args"]) == 3
    assert direct_scalar_and_cuda_handle_payload["run_args"][1] == 11
    assert direct_scalar_and_cuda_handle_payload["result_indices"] == ()

    assert len(direct_scalar_and_cuda_dlpack_payload["compile_args"]) == 3
    assert direct_scalar_and_cuda_dlpack_payload["compile_args"][0].shape == (128,)
    assert int(direct_scalar_and_cuda_dlpack_payload["compile_args"][1]) == 11
    assert direct_scalar_and_cuda_dlpack_payload["compile_args"][2].shape == (128,)
    assert len(direct_scalar_and_cuda_dlpack_payload["run_args"]) == 3
    assert direct_scalar_and_cuda_dlpack_payload["run_args"][1] == 11


    assert len(indexed_scalar_payload["args"]) == 3
    assert indexed_scalar_payload["args"][0].shape == (kernels.POINTWISE_N,)
    assert int(indexed_scalar_payload["args"][1]) == 24
    assert indexed_scalar_payload["args"][2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_scalar_payload["args"][0].dtype) == "i32"
    assert str(indexed_scalar_payload["args"][2].dtype) == "i32"
    assert indexed_scalar_payload["result_indices"] == ()

    assert len(indexed_scalar_cuda_handle_payload["compile_args"]) == 3
    assert indexed_scalar_cuda_handle_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert int(indexed_scalar_cuda_handle_payload["compile_args"][1]) == 24
    assert indexed_scalar_cuda_handle_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_scalar_cuda_handle_payload["run_args"]) == 3
    assert indexed_scalar_cuda_handle_payload["run_args"][1] == 24
    assert indexed_scalar_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_scalar_cuda_dlpack_payload["compile_args"]) == 3
    assert indexed_scalar_cuda_dlpack_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert int(indexed_scalar_cuda_dlpack_payload["compile_args"][1]) == 24
    assert indexed_scalar_cuda_dlpack_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_scalar_cuda_dlpack_payload["run_args"]) == 3
    assert indexed_scalar_cuda_dlpack_payload["run_args"][1] == 24
    assert indexed_scalar_cuda_dlpack_payload["result_indices"] == ()

    assert len(indexed_tensor_scalar_payload["args"]) == 3
    assert indexed_tensor_scalar_payload["args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_tensor_scalar_payload["args"][1].shape == (1,)
    assert indexed_tensor_scalar_payload["args"][2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_tensor_scalar_payload["args"][0].dtype) == "i32"
    assert str(indexed_tensor_scalar_payload["args"][1].dtype) == "i32"
    assert str(indexed_tensor_scalar_payload["args"][2].dtype) == "i32"
    assert indexed_tensor_scalar_payload["result_indices"] == ()

    assert len(indexed_tensor_scalar_cuda_handle_payload["compile_args"]) == 3
    assert indexed_tensor_scalar_cuda_handle_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_tensor_scalar_cuda_handle_payload["compile_args"][1].shape == (1,)
    assert indexed_tensor_scalar_cuda_handle_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_tensor_scalar_cuda_handle_payload["run_args"]) == 3
    assert indexed_tensor_scalar_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_tensor_scalar_cuda_dlpack_payload["compile_args"]) == 3
    assert indexed_tensor_scalar_cuda_dlpack_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_tensor_scalar_cuda_dlpack_payload["compile_args"][1].shape == (1,)
    assert indexed_tensor_scalar_cuda_dlpack_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_tensor_scalar_cuda_dlpack_payload["run_args"]) == 3
    assert indexed_tensor_scalar_cuda_dlpack_payload["result_indices"] == ()

    assert len(indexed_tensor_scalar_or_payload["args"]) == 3
    assert indexed_tensor_scalar_or_payload["args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_tensor_scalar_or_payload["args"][1].shape == (1,)
    assert indexed_tensor_scalar_or_payload["args"][2].shape == (kernels.POINTWISE_N,)
    assert str(indexed_tensor_scalar_or_payload["args"][0].dtype) == "i32"
    assert str(indexed_tensor_scalar_or_payload["args"][1].dtype) == "i32"
    assert str(indexed_tensor_scalar_or_payload["args"][2].dtype) == "i32"
    assert indexed_tensor_scalar_or_payload["result_indices"] == ()

    assert len(indexed_tensor_scalar_or_cuda_handle_payload["compile_args"]) == 3
    assert indexed_tensor_scalar_or_cuda_handle_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_tensor_scalar_or_cuda_handle_payload["compile_args"][1].shape == (1,)
    assert indexed_tensor_scalar_or_cuda_handle_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_tensor_scalar_or_cuda_handle_payload["run_args"]) == 3
    assert indexed_tensor_scalar_or_cuda_handle_payload["result_indices"] == ()

    assert len(indexed_tensor_scalar_or_cuda_dlpack_payload["compile_args"]) == 3
    assert indexed_tensor_scalar_or_cuda_dlpack_payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
    assert indexed_tensor_scalar_or_cuda_dlpack_payload["compile_args"][1].shape == (1,)
    assert indexed_tensor_scalar_or_cuda_dlpack_payload["compile_args"][2].shape == (kernels.POINTWISE_N,)
    assert len(indexed_tensor_scalar_or_cuda_dlpack_payload["run_args"]) == 3
    assert indexed_tensor_scalar_or_cuda_dlpack_payload["result_indices"] == ()


def test_ptx_f16_copy_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_copy_f16_args()
    indexed_cuda_handle_payload = kernels.indexed_copy_f16_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_copy_f16_cuda_dlpack_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_copy_2d_f16_args()
    dense_cuda_handle_payload = kernels.dense_copy_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_copy_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_payload = kernels.parallel_dense_copy_2d_f16_args()
    parallel_cuda_handle_payload = kernels.parallel_dense_copy_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    parallel_cuda_dlpack_payload = kernels.parallel_dense_copy_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_copy_2d_f16_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_copy_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_copy_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")

    indexed_args = indexed_payload["args"]
    assert len(indexed_args) == 2
    assert indexed_args[0].shape == (kernels.POINTWISE_N,)
    assert indexed_args[1].shape == (kernels.POINTWISE_N,)
    assert str(indexed_args[0].dtype) == "f16"
    assert str(indexed_args[1].dtype) == "f16"
    assert indexed_payload["result_indices"] == ()

    for payload in (indexed_cuda_handle_payload, indexed_cuda_dlpack_payload):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
        assert payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
        assert str(payload["compile_args"][0].dtype) == "f16"
        assert str(payload["compile_args"][1].dtype) == "f16"
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    for payload in (dense_payload, parallel_payload, multiblock_payload):
        args = payload["args"]
        assert len(args) == 2
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f16"
        assert str(args[1].dtype) == "f16"
        assert payload["result_indices"] == ()

    for payload in (
        dense_cuda_handle_payload,
        dense_cuda_dlpack_payload,
        parallel_cuda_handle_payload,
        parallel_cuda_dlpack_payload,
        multiblock_cuda_handle_payload,
        multiblock_cuda_dlpack_payload,
    ):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["compile_args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(payload["compile_args"][0].dtype) == "f16"
        assert str(payload["compile_args"][1].dtype) == "f16"
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()


def test_ptx_f16_unary_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_abs_payload = kernels.indexed_abs_f16_args()
    indexed_abs_cuda_handle_payload = kernels.indexed_abs_f16_cuda_handle_args(backend_name="ptx_exec")
    indexed_abs_cuda_dlpack_payload = kernels.indexed_abs_f16_cuda_dlpack_args(backend_name="ptx_exec")
    direct_neg_payload = kernels.direct_neg_f16_args()
    direct_neg_cuda_handle_payload = kernels.direct_neg_f16_cuda_handle_args(backend_name="ptx_exec")
    direct_neg_cuda_dlpack_payload = kernels.direct_neg_f16_cuda_dlpack_args(backend_name="ptx_exec")
    dense_abs_payload = kernels.dense_abs_2d_f16_args()
    dense_abs_cuda_handle_payload = kernels.dense_abs_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    dense_abs_cuda_dlpack_payload = kernels.dense_abs_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_abs_payload = kernels.parallel_dense_abs_2d_f16_args()
    parallel_abs_cuda_handle_payload = kernels.parallel_dense_abs_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    parallel_abs_cuda_dlpack_payload = kernels.parallel_dense_abs_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_abs_payload = kernels.multiblock_dense_abs_2d_f16_args()
    multiblock_abs_cuda_handle_payload = kernels.multiblock_dense_abs_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    multiblock_abs_cuda_dlpack_payload = kernels.multiblock_dense_abs_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_rank1_payload(payload, *, n: int, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == "f16"
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == "f16"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    def _assert_2d_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f16"
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == "f16"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    _assert_rank1_payload(indexed_abs_payload, n=kernels.POINTWISE_N, staged=True)
    _assert_rank1_payload(indexed_abs_cuda_handle_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_rank1_payload(indexed_abs_cuda_dlpack_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_rank1_payload(direct_neg_payload, n=128, staged=True)
    _assert_rank1_payload(direct_neg_cuda_handle_payload, n=128, staged=False)
    _assert_rank1_payload(direct_neg_cuda_dlpack_payload, n=128, staged=False)

    _assert_2d_payload(dense_abs_payload, staged=True)
    _assert_2d_payload(dense_abs_cuda_handle_payload, staged=False)
    _assert_2d_payload(dense_abs_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(parallel_abs_payload, staged=True)
    _assert_2d_payload(parallel_abs_cuda_handle_payload, staged=False)
    _assert_2d_payload(parallel_abs_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(multiblock_abs_payload, staged=True)
    _assert_2d_payload(multiblock_abs_cuda_handle_payload, staged=False)
    _assert_2d_payload(multiblock_abs_cuda_dlpack_payload, staged=False)


def test_ptx_f16_add_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_add_f16_args()
    indexed_cuda_handle_payload = kernels.indexed_add_f16_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_add_f16_cuda_dlpack_args(backend_name="ptx_exec")
    direct_payload = kernels.direct_add_f16_args()
    direct_cuda_handle_payload = kernels.direct_add_f16_cuda_handle_args(backend_name="ptx_exec")
    direct_cuda_dlpack_payload = kernels.direct_add_f16_cuda_dlpack_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_add_2d_f16_args()
    dense_cuda_handle_payload = kernels.dense_add_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_add_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_payload = kernels.parallel_dense_add_2d_f16_args()
    parallel_cuda_handle_payload = kernels.parallel_dense_add_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    parallel_cuda_dlpack_payload = kernels.parallel_dense_add_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_add_2d_f16_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_add_2d_f16_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_add_2d_f16_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_rank1_payload(payload, *, n: int, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 3
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == "f16"
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == "f16"
        assert args[2].shape == (n,)
        assert str(args[2].dtype) == "f16"
        if not staged:
            assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()

    def _assert_2d_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 3
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f16"
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == "f16"
        assert args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[2].dtype) == "f16"
        if not staged:
            assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()

    _assert_rank1_payload(indexed_payload, n=kernels.POINTWISE_N, staged=True)
    _assert_rank1_payload(indexed_cuda_handle_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_rank1_payload(indexed_cuda_dlpack_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_rank1_payload(direct_payload, n=128, staged=True)
    _assert_rank1_payload(direct_cuda_handle_payload, n=128, staged=False)
    _assert_rank1_payload(direct_cuda_dlpack_payload, n=128, staged=False)

    _assert_2d_payload(dense_payload, staged=True)
    _assert_2d_payload(dense_cuda_handle_payload, staged=False)
    _assert_2d_payload(dense_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(parallel_payload, staged=True)
    _assert_2d_payload(parallel_cuda_handle_payload, staged=False)
    _assert_2d_payload(parallel_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(multiblock_payload, staged=True)
    _assert_2d_payload(multiblock_cuda_handle_payload, staged=False)
    _assert_2d_payload(multiblock_cuda_dlpack_payload, staged=False)


def test_ptx_bitnot_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_bitnot_i32_args()
    indexed_cuda_handle_payload = kernels.indexed_bitnot_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_bitnot_i32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_bitnot_2d_i32_args()
    dense_cuda_handle_payload = kernels.dense_bitnot_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_bitnot_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_payload = kernels.parallel_dense_bitnot_2d_i32_args()
    parallel_cuda_handle_payload = kernels.parallel_dense_bitnot_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_cuda_dlpack_payload = kernels.parallel_dense_bitnot_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_bitnot_2d_i32_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_bitnot_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_bitnot_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")

    assert len(indexed_payload["args"]) == 2
    assert indexed_payload["args"][0].shape == (kernels.POINTWISE_N,)
    assert str(indexed_payload["args"][0].dtype) == "i32"
    assert indexed_payload["args"][1].shape == (kernels.POINTWISE_N,)
    assert indexed_payload["result_indices"] == ()

    for payload in (indexed_cuda_handle_payload, indexed_cuda_dlpack_payload):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.POINTWISE_N,)
        assert str(payload["compile_args"][0].dtype) == "i32"
        assert payload["compile_args"][1].shape == (kernels.POINTWISE_N,)
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    for payload in (dense_payload, parallel_payload, multiblock_payload):
        assert len(payload["args"]) == 2
        assert payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(payload["args"][0].dtype) == "i32"
        assert payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert payload["result_indices"] == ()

    for payload in (
        dense_cuda_handle_payload,
        dense_cuda_dlpack_payload,
        parallel_cuda_handle_payload,
        parallel_cuda_dlpack_payload,
        multiblock_cuda_handle_payload,
        multiblock_cuda_dlpack_payload,
    ):
        assert len(payload["compile_args"]) == 2
        assert payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(payload["compile_args"][0].dtype) == "i32"
        assert payload["compile_args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()


def test_ptx_neg_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_f32_payload = kernels.indexed_neg_f32_args()
    indexed_f32_cuda_handle_payload = kernels.indexed_neg_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_f32_cuda_dlpack_payload = kernels.indexed_neg_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_i32_payload = kernels.indexed_neg_i32_args()
    indexed_i32_cuda_handle_payload = kernels.indexed_neg_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_i32_cuda_dlpack_payload = kernels.indexed_neg_i32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_f32_payload = kernels.direct_neg_f32_args()
    direct_f32_cuda_handle_payload = kernels.direct_neg_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_f32_cuda_dlpack_payload = kernels.direct_neg_f32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_i32_payload = kernels.direct_neg_i32_args()
    direct_i32_cuda_handle_payload = kernels.direct_neg_i32_cuda_handle_args(backend_name="ptx_exec")
    direct_i32_cuda_dlpack_payload = kernels.direct_neg_i32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_payload(payload, *, n: int, dtype: str, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == dtype
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == dtype
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    _assert_payload(indexed_f32_payload, n=kernels.POINTWISE_N, dtype="f32", staged=True)
    _assert_payload(indexed_f32_cuda_handle_payload, n=kernels.POINTWISE_N, dtype="f32", staged=False)
    _assert_payload(indexed_f32_cuda_dlpack_payload, n=kernels.POINTWISE_N, dtype="f32", staged=False)
    _assert_payload(indexed_i32_payload, n=kernels.POINTWISE_N, dtype="i32", staged=True)
    _assert_payload(indexed_i32_cuda_handle_payload, n=kernels.POINTWISE_N, dtype="i32", staged=False)
    _assert_payload(indexed_i32_cuda_dlpack_payload, n=kernels.POINTWISE_N, dtype="i32", staged=False)
    _assert_payload(direct_f32_payload, n=128, dtype="f32", staged=True)
    _assert_payload(direct_f32_cuda_handle_payload, n=128, dtype="f32", staged=False)
    _assert_payload(direct_f32_cuda_dlpack_payload, n=128, dtype="f32", staged=False)
    _assert_payload(direct_i32_payload, n=128, dtype="i32", staged=True)
    _assert_payload(direct_i32_cuda_handle_payload, n=128, dtype="i32", staged=False)
    _assert_payload(direct_i32_cuda_dlpack_payload, n=128, dtype="i32", staged=False)


def test_ptx_abs_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_f32_payload = kernels.indexed_abs_f32_args()
    indexed_f32_cuda_handle_payload = kernels.indexed_abs_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_f32_cuda_dlpack_payload = kernels.indexed_abs_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_i32_payload = kernels.indexed_abs_i32_args()
    indexed_i32_cuda_handle_payload = kernels.indexed_abs_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_i32_cuda_dlpack_payload = kernels.indexed_abs_i32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_f32_payload = kernels.direct_abs_f32_args()
    direct_f32_cuda_handle_payload = kernels.direct_abs_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_f32_cuda_dlpack_payload = kernels.direct_abs_f32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_i32_payload = kernels.direct_abs_i32_args()
    direct_i32_cuda_handle_payload = kernels.direct_abs_i32_cuda_handle_args(backend_name="ptx_exec")
    direct_i32_cuda_dlpack_payload = kernels.direct_abs_i32_cuda_dlpack_args(backend_name="ptx_exec")

    dense_f32_payload = kernels.dense_abs_2d_f32_args()
    dense_f32_cuda_handle_payload = kernels.dense_abs_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_f32_cuda_dlpack_payload = kernels.dense_abs_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_i32_payload = kernels.dense_abs_2d_i32_args()
    dense_i32_cuda_handle_payload = kernels.dense_abs_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    dense_i32_cuda_dlpack_payload = kernels.dense_abs_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_f32_payload = kernels.parallel_dense_abs_2d_f32_args()
    parallel_f32_cuda_handle_payload = kernels.parallel_dense_abs_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_f32_cuda_dlpack_payload = kernels.parallel_dense_abs_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_i32_payload = kernels.parallel_dense_abs_2d_i32_args()
    parallel_i32_cuda_handle_payload = kernels.parallel_dense_abs_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    parallel_i32_cuda_dlpack_payload = kernels.parallel_dense_abs_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_f32_payload = kernels.multiblock_dense_abs_2d_f32_args()
    multiblock_f32_cuda_handle_payload = kernels.multiblock_dense_abs_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_f32_cuda_dlpack_payload = kernels.multiblock_dense_abs_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_i32_payload = kernels.multiblock_dense_abs_2d_i32_args()
    multiblock_i32_cuda_handle_payload = kernels.multiblock_dense_abs_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_i32_cuda_dlpack_payload = kernels.multiblock_dense_abs_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_1d_payload(payload, *, n: int, dtype: str, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == dtype
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == dtype
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    def _assert_2d_payload(payload, *, dtype: str, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == dtype
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == dtype
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    _assert_1d_payload(indexed_f32_payload, n=kernels.POINTWISE_N, dtype="f32", staged=True)
    _assert_1d_payload(indexed_f32_cuda_handle_payload, n=kernels.POINTWISE_N, dtype="f32", staged=False)
    _assert_1d_payload(indexed_f32_cuda_dlpack_payload, n=kernels.POINTWISE_N, dtype="f32", staged=False)
    _assert_1d_payload(indexed_i32_payload, n=kernels.POINTWISE_N, dtype="i32", staged=True)
    _assert_1d_payload(indexed_i32_cuda_handle_payload, n=kernels.POINTWISE_N, dtype="i32", staged=False)
    _assert_1d_payload(indexed_i32_cuda_dlpack_payload, n=kernels.POINTWISE_N, dtype="i32", staged=False)
    _assert_1d_payload(direct_f32_payload, n=128, dtype="f32", staged=True)
    _assert_1d_payload(direct_f32_cuda_handle_payload, n=128, dtype="f32", staged=False)
    _assert_1d_payload(direct_f32_cuda_dlpack_payload, n=128, dtype="f32", staged=False)
    _assert_1d_payload(direct_i32_payload, n=128, dtype="i32", staged=True)
    _assert_1d_payload(direct_i32_cuda_handle_payload, n=128, dtype="i32", staged=False)
    _assert_1d_payload(direct_i32_cuda_dlpack_payload, n=128, dtype="i32", staged=False)

    for payload in (dense_f32_payload, parallel_f32_payload, multiblock_f32_payload):
        _assert_2d_payload(payload, dtype="f32", staged=True)
    for payload in (dense_i32_payload, parallel_i32_payload, multiblock_i32_payload):
        _assert_2d_payload(payload, dtype="i32", staged=True)
    for payload in (
        dense_f32_cuda_handle_payload,
        dense_f32_cuda_dlpack_payload,
        parallel_f32_cuda_handle_payload,
        parallel_f32_cuda_dlpack_payload,
        multiblock_f32_cuda_handle_payload,
        multiblock_f32_cuda_dlpack_payload,
    ):
        _assert_2d_payload(payload, dtype="f32", staged=False)
    for payload in (
        dense_i32_cuda_handle_payload,
        dense_i32_cuda_dlpack_payload,
        parallel_i32_cuda_handle_payload,
        parallel_i32_cuda_dlpack_payload,
        multiblock_i32_cuda_handle_payload,
        multiblock_i32_cuda_dlpack_payload,
    ):
        _assert_2d_payload(payload, dtype="i32", staged=False)


def test_ptx_extrema_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_max_payload = kernels.indexed_max_f32_args()
    indexed_max_cuda_handle_payload = kernels.indexed_max_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_max_cuda_dlpack_payload = kernels.indexed_max_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_min_payload = kernels.indexed_min_i32_args()
    indexed_min_cuda_handle_payload = kernels.indexed_min_i32_cuda_handle_args(backend_name="ptx_exec")
    indexed_min_cuda_dlpack_payload = kernels.indexed_min_i32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_max_payload = kernels.dense_max_2d_f32_args()
    dense_max_cuda_handle_payload = kernels.dense_max_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_max_cuda_dlpack_payload = kernels.dense_max_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    broadcast_min_payload = kernels.broadcast_min_2d_i32_args()
    broadcast_min_cuda_handle_payload = kernels.broadcast_min_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    broadcast_min_cuda_dlpack_payload = kernels.broadcast_min_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_scalar_max_payload = kernels.parallel_dense_scalar_max_2d_f32_args()
    parallel_scalar_max_cuda_handle_payload = kernels.parallel_dense_scalar_max_2d_f32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    parallel_scalar_max_cuda_dlpack_payload = kernels.parallel_dense_scalar_max_2d_f32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    multiblock_tensor_min_payload = kernels.multiblock_dense_tensor_scalar_min_2d_i32_args()
    multiblock_tensor_min_cuda_handle_payload = kernels.multiblock_dense_tensor_scalar_min_2d_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    multiblock_tensor_min_cuda_dlpack_payload = kernels.multiblock_dense_tensor_scalar_min_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )

    def _assert_1d_binary_payload(payload, *, dtype: str, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 3
        assert args[0].shape == (kernels.POINTWISE_N,)
        assert str(args[0].dtype) == dtype
        assert args[1].shape == (kernels.POINTWISE_N,)
        assert str(args[1].dtype) == dtype
        assert args[2].shape == (kernels.POINTWISE_N,)
        assert str(args[2].dtype) == dtype
        if not staged:
            assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()

    def _assert_2d_binary_payload(payload, *, lhs_shape: tuple[int, int], rhs_shape: tuple[int, int], dtype: str, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 3
        assert args[0].shape == lhs_shape
        assert str(args[0].dtype) == dtype
        assert args[1].shape == rhs_shape
        assert str(args[1].dtype) == dtype
        assert args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[2].dtype) == dtype
        if not staged:
            assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()

    def _assert_2d_scalar_payload(payload, *, dtype: str, scalar_is_tensor: bool, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 3
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == dtype
        if scalar_is_tensor:
            assert args[1].shape == (1,)
            assert str(args[1].dtype) == dtype
        else:
            assert str(args[1].dtype) == dtype
        assert args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[2].dtype) == dtype
        if not staged:
            assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()

    _assert_1d_binary_payload(indexed_max_payload, dtype="f32", staged=True)
    _assert_1d_binary_payload(indexed_max_cuda_handle_payload, dtype="f32", staged=False)
    _assert_1d_binary_payload(indexed_max_cuda_dlpack_payload, dtype="f32", staged=False)
    _assert_1d_binary_payload(indexed_min_payload, dtype="i32", staged=True)
    _assert_1d_binary_payload(indexed_min_cuda_handle_payload, dtype="i32", staged=False)
    _assert_1d_binary_payload(indexed_min_cuda_dlpack_payload, dtype="i32", staged=False)

    _assert_2d_binary_payload(
        dense_max_payload,
        lhs_shape=(kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
        rhs_shape=(kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
        dtype="f32",
        staged=True,
    )
    _assert_2d_binary_payload(
        dense_max_cuda_handle_payload,
        lhs_shape=(kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
        rhs_shape=(kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
        dtype="f32",
        staged=False,
    )
    _assert_2d_binary_payload(
        dense_max_cuda_dlpack_payload,
        lhs_shape=(kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
        rhs_shape=(kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS),
        dtype="f32",
        staged=False,
    )
    _assert_2d_binary_payload(
        broadcast_min_payload,
        lhs_shape=(kernels.FLYDSL_MICRO_ROWS, 1),
        rhs_shape=(1, kernels.FLYDSL_MICRO_COLS),
        dtype="i32",
        staged=True,
    )
    _assert_2d_binary_payload(
        broadcast_min_cuda_handle_payload,
        lhs_shape=(kernels.FLYDSL_MICRO_ROWS, 1),
        rhs_shape=(1, kernels.FLYDSL_MICRO_COLS),
        dtype="i32",
        staged=False,
    )
    _assert_2d_binary_payload(
        broadcast_min_cuda_dlpack_payload,
        lhs_shape=(kernels.FLYDSL_MICRO_ROWS, 1),
        rhs_shape=(1, kernels.FLYDSL_MICRO_COLS),
        dtype="i32",
        staged=False,
    )
    _assert_2d_scalar_payload(parallel_scalar_max_payload, dtype="f32", scalar_is_tensor=False, staged=True)
    _assert_2d_scalar_payload(parallel_scalar_max_cuda_handle_payload, dtype="f32", scalar_is_tensor=False, staged=False)
    _assert_2d_scalar_payload(parallel_scalar_max_cuda_dlpack_payload, dtype="f32", scalar_is_tensor=False, staged=False)
    _assert_2d_scalar_payload(multiblock_tensor_min_payload, dtype="i32", scalar_is_tensor=True, staged=True)
    _assert_2d_scalar_payload(
        multiblock_tensor_min_cuda_handle_payload, dtype="i32", scalar_is_tensor=True, staged=False
    )
    _assert_2d_scalar_payload(
        multiblock_tensor_min_cuda_dlpack_payload, dtype="i32", scalar_is_tensor=True, staged=False
    )


def test_ptx_native_math_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_atan_f32_args()
    indexed_cuda_handle_payload = kernels.indexed_atan_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_atan_f32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_payload = kernels.direct_atan_f32_args()
    direct_cuda_handle_payload = kernels.direct_atan_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_cuda_dlpack_payload = kernels.direct_atan_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_atan_2d_f32_args()
    dense_cuda_handle_payload = kernels.dense_atan_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_atan_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_payload = kernels.parallel_dense_atan_2d_f32_args()
    parallel_cuda_handle_payload = kernels.parallel_dense_atan_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_cuda_dlpack_payload = kernels.parallel_dense_atan_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_atan_2d_f32_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_atan_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_atan_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_1d_payload(payload, *, n: int, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    def _assert_2d_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    _assert_1d_payload(indexed_payload, n=kernels.POINTWISE_N, staged=True)
    _assert_1d_payload(indexed_cuda_handle_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(indexed_cuda_dlpack_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(direct_payload, n=128, staged=True)
    _assert_1d_payload(direct_cuda_handle_payload, n=128, staged=False)
    _assert_1d_payload(direct_cuda_dlpack_payload, n=128, staged=False)
    _assert_2d_payload(dense_payload, staged=True)
    _assert_2d_payload(dense_cuda_handle_payload, staged=False)
    _assert_2d_payload(dense_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(parallel_payload, staged=True)
    _assert_2d_payload(parallel_cuda_handle_payload, staged=False)
    _assert_2d_payload(parallel_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(multiblock_payload, staged=True)
    _assert_2d_payload(multiblock_cuda_handle_payload, staged=False)
    _assert_2d_payload(multiblock_cuda_dlpack_payload, staged=False)


def test_ptx_rounding_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_round_payload = kernels.indexed_round_f32_args()
    indexed_round_cuda_handle_payload = kernels.indexed_round_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_round_cuda_dlpack_payload = kernels.indexed_round_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_payload = kernels.indexed_floor_f32_args()
    indexed_cuda_handle_payload = kernels.indexed_floor_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_floor_f32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_round_payload = kernels.direct_round_f32_args()
    direct_round_cuda_handle_payload = kernels.direct_round_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_round_cuda_dlpack_payload = kernels.direct_round_f32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_payload = kernels.direct_ceil_f32_args()
    direct_cuda_handle_payload = kernels.direct_ceil_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_cuda_dlpack_payload = kernels.direct_ceil_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_round_payload = kernels.dense_round_2d_f32_args()
    dense_round_cuda_handle_payload = kernels.dense_round_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_round_cuda_dlpack_payload = kernels.dense_round_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_trunc_2d_f32_args()
    dense_cuda_handle_payload = kernels.dense_trunc_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_trunc_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_round_payload = kernels.parallel_dense_round_2d_f32_args()
    parallel_round_cuda_handle_payload = kernels.parallel_dense_round_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_round_cuda_dlpack_payload = kernels.parallel_dense_round_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_payload = kernels.parallel_dense_floor_2d_f32_args()
    parallel_cuda_handle_payload = kernels.parallel_dense_floor_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_cuda_dlpack_payload = kernels.parallel_dense_floor_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_round_payload = kernels.multiblock_dense_round_2d_f32_args()
    multiblock_round_cuda_handle_payload = kernels.multiblock_dense_round_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_round_cuda_dlpack_payload = kernels.multiblock_dense_round_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_ceil_2d_f32_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_ceil_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_ceil_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_1d_payload(payload, *, n: int, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    def _assert_2d_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    _assert_1d_payload(indexed_round_payload, n=kernels.POINTWISE_N, staged=True)
    _assert_1d_payload(indexed_round_cuda_handle_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(indexed_round_cuda_dlpack_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(indexed_payload, n=kernels.POINTWISE_N, staged=True)
    _assert_1d_payload(indexed_cuda_handle_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(indexed_cuda_dlpack_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(direct_round_payload, n=128, staged=True)
    _assert_1d_payload(direct_round_cuda_handle_payload, n=128, staged=False)
    _assert_1d_payload(direct_round_cuda_dlpack_payload, n=128, staged=False)
    _assert_1d_payload(direct_payload, n=128, staged=True)
    _assert_1d_payload(direct_cuda_handle_payload, n=128, staged=False)
    _assert_1d_payload(direct_cuda_dlpack_payload, n=128, staged=False)
    _assert_2d_payload(dense_round_payload, staged=True)
    _assert_2d_payload(dense_round_cuda_handle_payload, staged=False)
    _assert_2d_payload(dense_round_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(dense_payload, staged=True)
    _assert_2d_payload(dense_cuda_handle_payload, staged=False)
    _assert_2d_payload(dense_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(parallel_round_payload, staged=True)
    _assert_2d_payload(parallel_round_cuda_handle_payload, staged=False)
    _assert_2d_payload(parallel_round_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(parallel_payload, staged=True)
    _assert_2d_payload(parallel_cuda_handle_payload, staged=False)
    _assert_2d_payload(parallel_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(multiblock_round_payload, staged=True)
    _assert_2d_payload(multiblock_round_cuda_handle_payload, staged=False)
    _assert_2d_payload(multiblock_round_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(multiblock_payload, staged=True)
    _assert_2d_payload(multiblock_cuda_handle_payload, staged=False)
    _assert_2d_payload(multiblock_cuda_dlpack_payload, staged=False)


def test_ptx_asin_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_asin_f32_args()
    indexed_cuda_handle_payload = kernels.indexed_asin_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_asin_f32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_payload = kernels.direct_asin_f32_args()
    direct_cuda_handle_payload = kernels.direct_asin_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_cuda_dlpack_payload = kernels.direct_asin_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_asin_2d_f32_args()
    dense_cuda_handle_payload = kernels.dense_asin_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_asin_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_payload = kernels.parallel_dense_asin_2d_f32_args()
    parallel_cuda_handle_payload = kernels.parallel_dense_asin_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_cuda_dlpack_payload = kernels.parallel_dense_asin_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_asin_2d_f32_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_asin_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_asin_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_1d_payload(payload, *, n: int, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    def _assert_2d_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    _assert_1d_payload(indexed_payload, n=kernels.POINTWISE_N, staged=True)
    _assert_1d_payload(indexed_cuda_handle_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(indexed_cuda_dlpack_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(direct_payload, n=128, staged=True)
    _assert_1d_payload(direct_cuda_handle_payload, n=128, staged=False)
    _assert_1d_payload(direct_cuda_dlpack_payload, n=128, staged=False)
    _assert_2d_payload(dense_payload, staged=True)
    _assert_2d_payload(dense_cuda_handle_payload, staged=False)
    _assert_2d_payload(dense_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(parallel_payload, staged=True)
    _assert_2d_payload(parallel_cuda_handle_payload, staged=False)
    _assert_2d_payload(parallel_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(multiblock_payload, staged=True)
    _assert_2d_payload(multiblock_cuda_handle_payload, staged=False)
    _assert_2d_payload(multiblock_cuda_dlpack_payload, staged=False)


def test_ptx_acos_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_acos_f32_args()
    indexed_cuda_handle_payload = kernels.indexed_acos_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_acos_f32_cuda_dlpack_args(backend_name="ptx_exec")
    direct_payload = kernels.direct_acos_f32_args()
    direct_cuda_handle_payload = kernels.direct_acos_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_cuda_dlpack_payload = kernels.direct_acos_f32_cuda_dlpack_args(backend_name="ptx_exec")
    dense_payload = kernels.dense_acos_2d_f32_args()
    dense_cuda_handle_payload = kernels.dense_acos_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_acos_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_payload = kernels.parallel_dense_acos_2d_f32_args()
    parallel_cuda_handle_payload = kernels.parallel_dense_acos_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    parallel_cuda_dlpack_payload = kernels.parallel_dense_acos_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_acos_2d_f32_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_acos_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_acos_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_1d_payload(payload, *, n: int, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    def _assert_2d_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 2
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 2
        assert payload["result_indices"] == ()

    _assert_1d_payload(indexed_payload, n=kernels.POINTWISE_N, staged=True)
    _assert_1d_payload(indexed_cuda_handle_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(indexed_cuda_dlpack_payload, n=kernels.POINTWISE_N, staged=False)
    _assert_1d_payload(direct_payload, n=128, staged=True)
    _assert_1d_payload(direct_cuda_handle_payload, n=128, staged=False)
    _assert_1d_payload(direct_cuda_dlpack_payload, n=128, staged=False)
    _assert_2d_payload(dense_payload, staged=True)
    _assert_2d_payload(dense_cuda_handle_payload, staged=False)
    _assert_2d_payload(dense_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(parallel_payload, staged=True)
    _assert_2d_payload(parallel_cuda_handle_payload, staged=False)
    _assert_2d_payload(parallel_cuda_dlpack_payload, staged=False)
    _assert_2d_payload(multiblock_payload, staged=True)
    _assert_2d_payload(multiblock_cuda_handle_payload, staged=False)
    _assert_2d_payload(multiblock_cuda_dlpack_payload, staged=False)


def test_ptx_atan2_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_payload = kernels.indexed_atan2_f32_args()
    indexed_cuda_handle_payload = kernels.indexed_atan2_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_cuda_dlpack_payload = kernels.indexed_atan2_f32_cuda_dlpack_args(backend_name="ptx_exec")
    multiblock_payload = kernels.multiblock_dense_atan2_2d_f32_args()
    multiblock_cuda_handle_payload = kernels.multiblock_dense_atan2_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    multiblock_cuda_dlpack_payload = kernels.multiblock_dense_atan2_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_1d_binary_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 3
        assert args[0].shape == (kernels.POINTWISE_N,)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (kernels.POINTWISE_N,)
        assert str(args[1].dtype) == "f32"
        assert args[2].shape == (kernels.POINTWISE_N,)
        assert str(args[2].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()

    def _assert_2d_binary_payload(payload, *, staged: bool) -> None:
        key = "args" if staged else "compile_args"
        args = payload[key]
        assert len(args) == 3
        assert args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[0].dtype) == "f32"
        assert args[1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[1].dtype) == "f32"
        assert args[2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert str(args[2].dtype) == "f32"
        if not staged:
            assert len(payload["run_args"]) == 3
        assert payload["result_indices"] == ()

    _assert_1d_binary_payload(indexed_payload, staged=True)
    _assert_1d_binary_payload(indexed_cuda_handle_payload, staged=False)
    _assert_1d_binary_payload(indexed_cuda_dlpack_payload, staged=False)
    _assert_2d_binary_payload(multiblock_payload, staged=True)
    _assert_2d_binary_payload(multiblock_cuda_handle_payload, staged=False)
    _assert_2d_binary_payload(multiblock_cuda_dlpack_payload, staged=False)


def test_ptx_select_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    indexed_scalar_payload = kernels.indexed_select_scalar_f32_args()
    indexed_scalar_cuda_handle_payload = kernels.indexed_select_scalar_f32_cuda_handle_args(backend_name="ptx_exec")
    indexed_scalar_cuda_dlpack_payload = kernels.indexed_select_scalar_f32_cuda_dlpack_args(backend_name="ptx_exec")
    indexed_tensor_payload = kernels.indexed_select_tensor_scalar_i32_args()
    indexed_tensor_cuda_handle_payload = kernels.indexed_select_tensor_scalar_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    indexed_tensor_cuda_dlpack_payload = kernels.indexed_select_tensor_scalar_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    direct_scalar_payload = kernels.direct_select_scalar_f32_args()
    direct_scalar_cuda_handle_payload = kernels.direct_select_scalar_f32_cuda_handle_args(backend_name="ptx_exec")
    direct_scalar_cuda_dlpack_payload = kernels.direct_select_scalar_f32_cuda_dlpack_args(backend_name="ptx_exec")

    def _assert_select_payload(args, payload, *, n: int, dtype: str, scalar_mode: str, scalar_value) -> None:
        assert len(args) == 4
        assert args[0].shape == (n,)
        assert str(args[0].dtype) == "i1"
        assert args[1].shape == (n,)
        assert str(args[1].dtype) == dtype
        if scalar_mode == "param":
            assert float(args[2]) == float(scalar_value)
        else:
            assert args[2].shape == (1,)
            assert str(args[2].dtype) == dtype
        assert args[3].shape == (n,)
        assert str(args[3].dtype) == dtype
        assert payload["result_indices"] == ()

    def _assert_select_exec_payload(payload, *, n: int, dtype: str, scalar_mode: str, scalar_value) -> None:
        compile_args = payload["compile_args"]
        run_args = payload["run_args"]
        _assert_select_payload(
            compile_args,
            payload,
            n=n,
            dtype=dtype,
            scalar_mode=scalar_mode,
            scalar_value=scalar_value,
        )
        assert len(run_args) == 4
        assert run_args[0].shape == (n,)
        assert run_args[1].shape == (n,)
        if scalar_mode == "param":
            assert float(run_args[2]) == float(scalar_value)
        else:
            assert run_args[2].shape == (1,)
        assert run_args[3].shape == (n,)

    _assert_select_payload(
        indexed_scalar_payload["args"],
        indexed_scalar_payload,
        n=kernels.POINTWISE_N,
        dtype="f32",
        scalar_mode="param",
        scalar_value=3.5,
    )
    _assert_select_exec_payload(
        indexed_scalar_cuda_handle_payload,
        n=kernels.POINTWISE_N,
        dtype="f32",
        scalar_mode="param",
        scalar_value=3.5,
    )
    _assert_select_exec_payload(
        indexed_scalar_cuda_dlpack_payload,
        n=kernels.POINTWISE_N,
        dtype="f32",
        scalar_mode="param",
        scalar_value=3.5,
    )
    _assert_select_payload(
        indexed_tensor_payload["args"],
        indexed_tensor_payload,
        n=kernels.POINTWISE_N,
        dtype="i32",
        scalar_mode="tensor",
        scalar_value=17,
    )
    _assert_select_exec_payload(
        indexed_tensor_cuda_handle_payload,
        n=kernels.POINTWISE_N,
        dtype="i32",
        scalar_mode="tensor",
        scalar_value=17,
    )
    _assert_select_exec_payload(
        indexed_tensor_cuda_dlpack_payload,
        n=kernels.POINTWISE_N,
        dtype="i32",
        scalar_mode="tensor",
        scalar_value=17,
    )
    _assert_select_payload(
        direct_scalar_payload["args"],
        direct_scalar_payload,
        n=128,
        dtype="f32",
        scalar_mode="param",
        scalar_value=5.5,
    )
    _assert_select_exec_payload(
        direct_scalar_cuda_handle_payload,
        n=128,
        dtype="f32",
        scalar_mode="param",
        scalar_value=5.5,
    )
    _assert_select_exec_payload(
        direct_scalar_cuda_dlpack_payload,
        n=128,
        dtype="f32",
        scalar_mode="param",
        scalar_value=5.5,
    )


def test_ptx_select_2d_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    dense_payload = kernels.dense_select_2d_f32_args()
    dense_cuda_handle_payload = kernels.dense_select_2d_f32_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_select_2d_f32_cuda_dlpack_args(backend_name="ptx_exec")
    broadcast_payload = kernels.broadcast_select_2d_i32_args()
    broadcast_cuda_handle_payload = kernels.broadcast_select_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    broadcast_cuda_dlpack_payload = kernels.broadcast_select_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_scalar_payload = kernels.parallel_dense_scalar_select_2d_f32_args()
    parallel_scalar_cuda_handle_payload = kernels.parallel_dense_scalar_select_2d_f32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    parallel_scalar_cuda_dlpack_payload = kernels.parallel_dense_scalar_select_2d_f32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )
    multiblock_tensor_payload = kernels.multiblock_dense_tensor_scalar_select_2d_i32_args()
    multiblock_tensor_cuda_handle_payload = kernels.multiblock_dense_tensor_scalar_select_2d_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    multiblock_tensor_cuda_dlpack_payload = kernels.multiblock_dense_tensor_scalar_select_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )

    def _assert_select_2d_payload(args, payload, *, dtype: str, mode: str, scalar_value) -> None:
        rows = kernels.FLYDSL_MICRO_ROWS
        cols = kernels.FLYDSL_MICRO_COLS
        assert len(args) == 4
        assert args[0].shape == (rows, cols)
        assert str(args[0].dtype) == "i1"
        if mode == "dense":
            assert args[1].shape == (rows, cols)
            assert str(args[1].dtype) == dtype
            assert args[2].shape == (rows, cols)
            assert str(args[2].dtype) == dtype
        elif mode == "broadcast":
            assert args[1].shape == (rows, 1)
            assert str(args[1].dtype) == dtype
            assert args[2].shape == (1, cols)
            assert str(args[2].dtype) == dtype
        elif mode == "scalar_param":
            assert args[1].shape == (rows, cols)
            assert str(args[1].dtype) == dtype
            assert float(args[2]) == float(scalar_value)
        elif mode == "tensor_scalar":
            assert args[1].shape == (rows, cols)
            assert str(args[1].dtype) == dtype
            assert args[2].shape == (1,)
            assert str(args[2].dtype) == dtype
        else:
            raise AssertionError(f"unexpected 2D select mode: {mode}")
        assert args[3].shape == (rows, cols)
        assert str(args[3].dtype) == dtype
        assert payload["result_indices"] == ()

    def _assert_select_2d_exec_payload(payload, *, dtype: str, mode: str, scalar_value) -> None:
        compile_args = payload["compile_args"]
        run_args = payload["run_args"]
        _assert_select_2d_payload(compile_args, payload, dtype=dtype, mode=mode, scalar_value=scalar_value)
        assert len(run_args) == 4
        assert run_args[0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
        assert run_args[1].shape == compile_args[1].shape
        if mode in {"dense", "broadcast"}:
            assert run_args[2].shape == compile_args[2].shape
        elif mode == "scalar_param":
            assert float(run_args[2]) == float(scalar_value)
        else:
            assert run_args[2].shape == (1,)
        assert run_args[3].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)

    _assert_select_2d_payload(dense_payload["args"], dense_payload, dtype="f32", mode="dense", scalar_value=None)
    _assert_select_2d_exec_payload(
        dense_cuda_handle_payload,
        dtype="f32",
        mode="dense",
        scalar_value=None,
    )
    _assert_select_2d_exec_payload(
        dense_cuda_dlpack_payload,
        dtype="f32",
        mode="dense",
        scalar_value=None,
    )
    _assert_select_2d_payload(
        broadcast_payload["args"],
        broadcast_payload,
        dtype="i32",
        mode="broadcast",
        scalar_value=None,
    )
    _assert_select_2d_exec_payload(
        broadcast_cuda_handle_payload,
        dtype="i32",
        mode="broadcast",
        scalar_value=None,
    )
    _assert_select_2d_exec_payload(
        broadcast_cuda_dlpack_payload,
        dtype="i32",
        mode="broadcast",
        scalar_value=None,
    )
    _assert_select_2d_payload(
        parallel_scalar_payload["args"],
        parallel_scalar_payload,
        dtype="f32",
        mode="scalar_param",
        scalar_value=1.5,
    )
    _assert_select_2d_exec_payload(
        parallel_scalar_cuda_handle_payload,
        dtype="f32",
        mode="scalar_param",
        scalar_value=1.5,
    )
    _assert_select_2d_exec_payload(
        parallel_scalar_cuda_dlpack_payload,
        dtype="f32",
        mode="scalar_param",
        scalar_value=1.5,
    )
    _assert_select_2d_payload(
        multiblock_tensor_payload["args"],
        multiblock_tensor_payload,
        dtype="i32",
        mode="tensor_scalar",
        scalar_value=17,
    )
    _assert_select_2d_exec_payload(
        multiblock_tensor_cuda_handle_payload,
        dtype="i32",
        mode="tensor_scalar",
        scalar_value=17,
    )
    _assert_select_2d_exec_payload(
        multiblock_tensor_cuda_dlpack_payload,
        dtype="i32",
        mode="tensor_scalar",
        scalar_value=17,
    )


def test_ptx_integer_bitwise_tensor_binary_2d_sample_factories_return_expected_shapes() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    dense_payload = kernels.dense_bitand_2d_i32_args()
    dense_cuda_handle_payload = kernels.dense_bitand_2d_i32_cuda_handle_args(backend_name="ptx_exec")
    dense_cuda_dlpack_payload = kernels.dense_bitand_2d_i32_cuda_dlpack_args(backend_name="ptx_exec")
    parallel_broadcast_payload = kernels.parallel_broadcast_bitor_2d_i32_args()
    parallel_broadcast_cuda_handle_payload = kernels.parallel_broadcast_bitor_2d_i32_cuda_handle_args(
        backend_name="ptx_exec"
    )
    parallel_broadcast_cuda_dlpack_payload = kernels.parallel_broadcast_bitor_2d_i32_cuda_dlpack_args(
        backend_name="ptx_exec"
    )

    assert len(dense_payload["args"]) == 3
    assert dense_payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_payload["args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_payload["args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(dense_payload["args"][0].dtype) == "i32"
    assert dense_payload["result_indices"] == ()

    assert len(dense_cuda_handle_payload["compile_args"]) == 3
    assert dense_cuda_handle_payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_handle_payload["compile_args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_handle_payload["compile_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(dense_cuda_handle_payload["run_args"]) == 3
    assert dense_cuda_handle_payload["run_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_handle_payload["run_args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_handle_payload["run_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_handle_payload["result_indices"] == ()

    assert len(dense_cuda_dlpack_payload["compile_args"]) == 3
    assert dense_cuda_dlpack_payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_dlpack_payload["compile_args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_dlpack_payload["compile_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(dense_cuda_dlpack_payload["run_args"]) == 3
    assert dense_cuda_dlpack_payload["run_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_dlpack_payload["run_args"][1].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_dlpack_payload["run_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert dense_cuda_dlpack_payload["result_indices"] == ()

    assert len(parallel_broadcast_payload["args"]) == 3
    assert parallel_broadcast_payload["args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_payload["args"][1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_payload["args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert str(parallel_broadcast_payload["args"][0].dtype) == "i32"
    assert parallel_broadcast_payload["result_indices"] == ()

    assert len(parallel_broadcast_cuda_handle_payload["compile_args"]) == 3
    assert parallel_broadcast_cuda_handle_payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_cuda_handle_payload["compile_args"][1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_handle_payload["compile_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_broadcast_cuda_handle_payload["run_args"]) == 3
    assert parallel_broadcast_cuda_handle_payload["run_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_cuda_handle_payload["run_args"][1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_handle_payload["run_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_handle_payload["result_indices"] == ()

    assert len(parallel_broadcast_cuda_dlpack_payload["compile_args"]) == 3
    assert parallel_broadcast_cuda_dlpack_payload["compile_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_cuda_dlpack_payload["compile_args"][1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_dlpack_payload["compile_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert len(parallel_broadcast_cuda_dlpack_payload["run_args"]) == 3
    assert parallel_broadcast_cuda_dlpack_payload["run_args"][0].shape == (kernels.FLYDSL_MICRO_ROWS, 1)
    assert parallel_broadcast_cuda_dlpack_payload["run_args"][1].shape == (1, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_dlpack_payload["run_args"][2].shape == (kernels.FLYDSL_MICRO_ROWS, kernels.FLYDSL_MICRO_COLS)
    assert parallel_broadcast_cuda_dlpack_payload["result_indices"] == ()
