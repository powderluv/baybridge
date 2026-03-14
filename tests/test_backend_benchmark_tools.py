from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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

    copy_args = copy_payload["args"]
    add_args = add_payload["args"]
    copy_i32_args = copy_i32_payload["args"]
    add_i32_args = add_i32_payload["args"]
    copy_f16_args = copy_f16_payload["args"]
    broadcast_add_args = broadcast_add_payload["args"]
    broadcast_add_i32_args = broadcast_add_i32_payload["args"]

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


def test_backend_benchmark_kernels_exports_sub_and_mul_microbench_kernels() -> None:
    kernels = _load_tool_module("backend_benchmark_kernels")

    assert callable(kernels.dense_sub_f32_kernel)
    assert callable(kernels.dense_mul_f32_kernel)
