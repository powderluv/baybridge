from pathlib import Path

import baybridge as bb


@bb.kernel
def copy_kernel(
    src: bb.TensorSpec(shape=(128,), dtype="f16"),
    dst: bb.TensorSpec(shape=(128,), dtype="f16"),
):
    bb.copy(src, dst, vector_bytes=16)


@bb.jit(launch=bb.LaunchConfig(grid=(80, 1, 1), block=(256, 1, 1), shared_mem_bytes=4096))
def tiled_kernel(
    a: bb.TensorSpec(shape=(64, 64), dtype="f16"),
    b: bb.TensorSpec(shape=(64, 64), dtype="f16"),
):
    a_tile = bb.partition(a, (16, 16))
    b_tile = bb.partition(b, (16, 16))
    bb.mma(a_tile, b_tile, tile=(16, 16, 16))
    bb.barrier()


@bb.kernel(launch=bb.LaunchConfig(grid=(120, 3, 1), block=(256, 1, 1)))
def topology_kernel(
    src: bb.TensorSpec(shape=(128,), dtype="f16"),
    dst: bb.TensorSpec(shape=(128,), dtype="f16"),
):
    bb.program_id("x")
    bb.block_idx("y")
    bb.thread_idx("x")
    bb.block_dim("x")
    bb.grid_dim("y")
    bb.lane_id()
    bb.copy(src, dst)


def test_compile_records_ir(tmp_path: Path) -> None:
    artifact = bb.compile(copy_kernel, cache_dir=tmp_path)
    assert artifact.ir.name == "copy_kernel"
    assert len(artifact.ir.arguments) == 2
    assert [operation.op for operation in artifact.ir.operations] == ["copy"]
    assert artifact.ir.operations[0].attrs["vector_bytes"] == 16
    assert artifact.target.arch in {"gfx942", "gfx950"}
    assert artifact.backend_name == "mlir_text"
    assert artifact.ir.launch.block == (1, 1, 1)
    assert artifact.artifact_path.exists()
    assert artifact.lowered_path is not None and artifact.lowered_path.exists()
    assert artifact.lowered_module is not None
    assert artifact.from_cache is False


def test_jit_kernel_traces_multiple_ops(tmp_path: Path) -> None:
    artifact = bb.compile(tiled_kernel, cache_dir=tmp_path)
    assert [operation.op for operation in artifact.ir.operations] == ["partition", "partition", "make_tensor", "mma", "barrier"]
    assert artifact.ir.metadata["frontend"] == "jit"
    assert artifact.ir.launch.grid == (80, 1, 1)
    assert artifact.ir.launch.block == (256, 1, 1)
    assert artifact.ir.launch.shared_mem_bytes == 4096
    assert artifact.ir.operations[0].attrs["result"]["shape"] == [16, 16]
    assert artifact.ir.operations[1].attrs["result"]["shape"] == [16, 16]
    assert artifact.ir.operations[2].attrs["dtype"] == "f32"
    assert artifact.lowered_module is not None


def test_topology_builtins_enter_portable_ir(tmp_path: Path) -> None:
    artifact = bb.compile(topology_kernel, cache_dir=tmp_path, backend="portable")
    assert [operation.op for operation in artifact.ir.operations] == [
        "program_id",
        "block_idx",
        "thread_idx",
        "block_dim",
        "grid_dim",
        "lane_id",
        "copy",
    ]
    axes = [operation.attrs.get("axis") for operation in artifact.ir.operations[:-1]]
    assert axes == ["x", "y", "x", "x", "y", None]
