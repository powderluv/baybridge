import os
from pathlib import Path

import pytest

import baybridge as bb
from cuda.bindings import driver as cuda_driver
from cuda.bindings import runtime as cuda_runtime


@bb.jit
def opaque_stream_kernel(stream: cuda_driver.CUstream):
    del stream
    bb.printf("opaque stream accepted")


def test_cuda_driver_shim_exposes_custream_and_flags() -> None:
    stream = cuda_driver.CUstream(17)
    assert isinstance(stream, int)
    assert int(stream) == 17
    assert cuda_driver.CUstream_flags.CU_STREAM_DEFAULT == 0
    assert cuda_driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL == 0
    assert int(cuda_driver.CUgraph(3)) == 3
    assert int(cuda_driver.CUgraphExec(5)) == 5


def test_pointer_compatibility_fields_exist() -> None:
    ptr = bb.make_ptr("f32", 4096, bb.AddressSpace.GLOBAL)
    assert ptr._pointer == 4096
    assert ptr.memspace == bb.AddressSpace.GLOBAL


def test_compile_accepts_opaque_custream_argument(tmp_path: Path) -> None:
    artifact = bb.compile(opaque_stream_kernel, cuda_driver.CUstream(0), cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None
    assert artifact.ir.name == "opaque_stream_kernel"


def test_cuda_graph_shim_exposes_capture_flow() -> None:
    stream = cuda_driver.CUstream(11)
    begin_status, = cuda_runtime.cudaStreamBeginCapture(stream)
    assert begin_status == cuda_runtime.cudaError_t.cudaSuccess
    end_status, graph = cuda_runtime.cudaStreamEndCapture(stream)
    assert end_status == cuda_runtime.cudaError_t.cudaSuccess
    assert isinstance(graph, cuda_driver.CUgraph)
    instantiate_status, graph_exec = cuda_runtime.cudaGraphInstantiate(graph)
    assert instantiate_status == cuda_runtime.cudaError_t.cudaSuccess
    assert isinstance(graph_exec, cuda_driver.CUgraphExec)
    launch_status, = cuda_runtime.cudaGraphLaunch(graph_exec, stream)
    assert launch_status == cuda_runtime.cudaError_t.cudaSuccess


def test_cuda_runtime_shim_runs_on_amd_hardware() -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    status, ptr = cuda_runtime.cudaMalloc(64)
    assert status == cuda_runtime.cudaError_t.cudaSuccess
    assert ptr != 0
    try:
        memset_status, = cuda_runtime.cudaMemset(ptr, 0, 64)
        assert memset_status == cuda_runtime.cudaError_t.cudaSuccess
        sync_status, = cuda_runtime.cudaDeviceSynchronize()
        assert sync_status == cuda_runtime.cudaError_t.cudaSuccess
    finally:
        free_status, = cuda_runtime.cudaFree(ptr)
        assert free_status == cuda_runtime.cudaError_t.cudaSuccess


def test_hipcc_exec_artifact_accepts_stream_kwarg_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    @bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
    def stream_kwarg_kernel(src: bb.Tensor, dst: bb.Tensor):
        tidx, _, _ = bb.arch.thread_idx()
        dst[tidx] = src[tidx] + 1.0

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    src = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    dst = bb.zeros((4,), dtype="f32")
    artifact = bb.compile(
        stream_kwarg_kernel,
        src,
        dst,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(src, dst, stream=cuda_driver.CUstream(0))
    assert dst.tolist() == [2.0, 3.0, 4.0, 5.0]
