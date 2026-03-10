import os
from pathlib import Path

import pytest

import baybridge as bb


@bb.struct
class SharedData:
    values: bb.struct.MemRange["f32", 64]
    counter: "i32"
    flag: "i8"


@bb.struct
class Complex:
    real: "f32"
    imag: "f32"


@bb.struct
class SharedStorage:
    a: bb.struct.MemRange["f32", 32]
    b: "i64"
    c: Complex
    x: bb.struct.Align[bb.struct.MemRange["f32", 32], 128]
    y: bb.struct.Align["i32", 8]
    z: bb.struct.Align[Complex, 16]


@bb.kernel
def dynamic_smem_size_kernel():
    allocator = bb.SmemAllocator()
    allocator.allocate(SharedData)
    allocator.allocate(512, byte_alignment=64)
    allocator.allocate_array(element_type="i32", num_elems=128)
    allocator.allocate_tensor(
        element_type="f16",
        layout=bb.make_layout((32, 16)),
        byte_alignment=16,
    )


@bb.jit
def dynamic_smem_size_wrapper():
    kernel = dynamic_smem_size_kernel()
    kernel.launch(grid=(1, 1, 1), block=(1, 1, 1))


@bb.kernel
def smem_allocator_kernel(
    const_a: float,
    dst_a: bb.Tensor,
    const_b: float,
    dst_b: bb.Tensor,
    const_c: float,
    dst_c: bb.Tensor,
):
    allocator = bb.SmemAllocator()
    int_ptr = allocator.allocate("i32")
    assert int_ptr.dtype == "i32"
    struct_in_smem = allocator.allocate(SharedStorage)
    section_in_smem = allocator.allocate(64, byte_alignment=128)
    allocator.allocate_array(element_type="i64", num_elems=14)
    tensor_in_smem = allocator.allocate_tensor(
        element_type="f32",
        layout=bb.make_layout((16, 2)),
        byte_alignment=32,
    )

    a_tensor = struct_in_smem.a.get_tensor(bb.make_layout((8, 4)))
    a_tensor.fill(const_a)
    dst_a.store(a_tensor.load())

    sec_ptr = bb.recast_ptr(section_in_smem, dtype="f32")
    sec_tensor = bb.make_tensor(sec_ptr, bb.make_layout((8, 2)))
    sec_tensor.fill(const_b)
    dst_b.store(sec_tensor.load())

    tensor_in_smem.fill(const_c)
    dst_c.store(tensor_in_smem.load())


@bb.jit
def smem_allocator_wrapper(
    const_a: float,
    dst_a: bb.Tensor,
    const_b: float,
    dst_b: bb.Tensor,
    const_c: float,
    dst_c: bb.Tensor,
):
    kernel = smem_allocator_kernel(const_a, dst_a, const_b, dst_b, const_c, dst_c)
    kernel.launch(grid=(1, 1, 1), block=(1, 1, 1))


def test_struct_layout_metadata() -> None:
    shared_data = SharedData.__baybridge_struct__
    assert shared_data.size_bytes == 264
    assert shared_data.alignment_bytes == 4
    assert [field.offset_bytes for field in shared_data.fields] == [0, 256, 260]

    shared_storage = SharedStorage.__baybridge_struct__
    assert shared_storage.size_bytes == 512
    assert shared_storage.alignment_bytes == 128
    assert [field.offset_bytes for field in shared_storage.fields] == [0, 128, 136, 256, 384, 400]


def test_dynamic_smem_size_example_infers_shared_memory() -> None:
    launch = dynamic_smem_size_kernel()
    assert launch.smem_usage() == 2368


def test_dynamic_smem_size_example_compiles(tmp_path: Path) -> None:
    artifact = bb.compile(
        dynamic_smem_size_wrapper,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )

    assert artifact.ir is not None
    assert artifact.ir.launch.shared_mem_bytes == 2368
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "reinterpret_cast<std::int32_t*>(baybridge_dynamic_smem + 256)" in text
    assert "reinterpret_cast<std::int8_t*>(baybridge_dynamic_smem + 320)" in text
    assert "reinterpret_cast<__half*>(baybridge_dynamic_smem + 1344)" in text


def test_smem_allocator_kernel_runs_via_reference_runtime() -> None:
    dst_a = bb.zeros((8, 4), dtype="f32")
    dst_b = bb.zeros((8, 2), dtype="f32")
    dst_c = bb.zeros((16, 2), dtype="f32")

    smem_allocator_wrapper(0.5, dst_a, 1.0, dst_b, 2.0, dst_c)

    assert dst_a.tolist() == [[0.5] * 4 for _ in range(8)]
    assert dst_b.tolist() == [[1.0] * 2 for _ in range(8)]
    assert dst_c.tolist() == [[2.0] * 2 for _ in range(16)]


def test_smem_allocator_kernel_compiles_and_reports_smem(tmp_path: Path) -> None:
    dst_a = bb.zeros((8, 4), dtype="f32")
    dst_b = bb.zeros((8, 2), dtype="f32")
    dst_c = bb.zeros((16, 2), dtype="f32")

    artifact = bb.compile(
        smem_allocator_wrapper,
        0.5,
        dst_a,
        1.0,
        dst_b,
        2.0,
        dst_c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
    )

    assert artifact.ir is not None
    assert artifact.ir.launch.shared_mem_bytes == 960
    assert artifact.lowered_module is not None
    text = artifact.lowered_module.text
    assert "reinterpret_cast<std::int32_t*>(baybridge_dynamic_smem + 0)" in text
    assert "reinterpret_cast<float*>(baybridge_dynamic_smem + 128)" in text
    assert "reinterpret_cast<std::int8_t*>(baybridge_dynamic_smem + 640)" in text
    assert "reinterpret_cast<float*>(baybridge_dynamic_smem + 832)" in text


def test_smem_allocator_kernel_runs_on_amd_hardware(tmp_path: Path) -> None:
    if os.environ.get("BAYBRIDGE_RUN_EXEC_TESTS") != "1":
        pytest.skip("set BAYBRIDGE_RUN_EXEC_TESTS=1 to run executable HIP backend tests")

    target_arch = os.environ.get("BAYBRIDGE_EXEC_ARCH", "gfx950")
    dst_a = bb.zeros((8, 4), dtype="f32")
    dst_b = bb.zeros((8, 2), dtype="f32")
    dst_c = bb.zeros((16, 2), dtype="f32")

    artifact = bb.compile(
        smem_allocator_wrapper,
        0.5,
        dst_a,
        1.0,
        dst_b,
        2.0,
        dst_c,
        cache_dir=tmp_path,
        backend="hipcc_exec",
        target=bb.AMDTarget(arch=target_arch),
    )

    artifact(0.5, dst_a, 1.0, dst_b, 2.0, dst_c)

    assert dst_a.tolist() == [[0.5] * 4 for _ in range(8)]
    assert dst_b.tolist() == [[1.0] * 2 for _ in range(8)]
    assert dst_c.tolist() == [[2.0] * 2 for _ in range(16)]
