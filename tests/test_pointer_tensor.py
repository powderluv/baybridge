import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.jit
def fill_tensor_from_ptr(ptr: bb.Pointer):
    layout = bb.make_layout((4,), stride=(1,))
    tensor = bb.make_tensor(ptr, layout)
    tensor.fill(7.0)


def test_pointer_backed_tensor_runs_direct_and_compiled(tmp_path: Path) -> None:
    storage = bb.zeros((4,), dtype="f32")
    ptr = bb.make_ptr("f32", storage, assumed_align=16)

    fill_tensor_from_ptr(ptr)
    assert storage.tolist() == [7.0, 7.0, 7.0, 7.0]

    storage = bb.zeros((4,), dtype="f32")
    ptr = bb.make_ptr("f32", storage, assumed_align=16)
    artifact = bb.compile(fill_tensor_from_ptr, ptr, cache_dir=tmp_path, backend="gpu_text")

    assert artifact.ir is not None
    assert [argument.name for argument in artifact.ir.arguments] == ["ptr_tensor"]
    ops = [operation.op for operation in artifact.ir.operations]
    assert "pointer_tensor" in ops
    assert "fill" in ops

    artifact(ptr)
    assert storage.tolist() == [7.0, 7.0, 7.0, 7.0]


def test_pointer_backed_tensor_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    storage = bb.zeros((4,), dtype="f32")
    ptr = bb.make_ptr("f32", storage, assumed_align=16)

    artifact = bb.compile(
        fill_tensor_from_ptr,
        ptr,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(ptr)

    assert storage.tolist() == [7.0, 7.0, 7.0, 7.0]
