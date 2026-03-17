import pytest

import baybridge as bb


class FakeDLPackTensor:
    shape = (4, 8)
    dtype = "float16"

    def stride(self):
        return (8, 1)

    def data_ptr(self):
        return 4096

    def tolist(self):
        return [[float(col) for col in range(8)] for _ in range(4)]

    def __dlpack__(self):
        return "capsule"

    def __dlpack_device__(self):
        return (2, 7)


class FakeFp8DLPackTensor:
    shape = (4, 8)
    dtype = "float8_e4m3fnuz"

    def stride(self):
        return (8, 1)

    def data_ptr(self):
        return 8192

    def tolist(self):
        return [[0x40 for _ in range(8)] for _ in range(4)]

    def __dlpack__(self):
        return "fp8-capsule"

    def __dlpack_device__(self):
        return (2, 3)


def test_from_dlpack_extracts_metadata() -> None:
    handle = bb.from_dlpack(FakeDLPackTensor())
    assert handle.capsule == "capsule"
    assert handle.shape == (4, 8)
    assert handle.dtype == "f16"
    assert handle.device_type == 2
    assert handle.device_id == 7
    assert handle.stride == (8, 1)
    assert handle.data_ptr() == 4096


def test_from_dlpack_normalizes_fp8_dtype_alias() -> None:
    handle = bb.from_dlpack(FakeFp8DLPackTensor())
    assert handle.dtype == "fp8"
    assert handle.shape == (4, 8)
    assert handle.data_ptr() == 8192


def test_from_dlpack_supports_alignment_and_dynamic_layout_marking() -> None:
    handle = bb.from_dlpack(FakeDLPackTensor(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    assert handle.assumed_align == 16
    assert handle.dynamic_layout_leading_dim == 1
    assert handle.to_runtime_tensor().shape == (4, 8)


def test_pack_fp8_scalar_reference_values() -> None:
    assert bb.pack_fp8(0.0) == 0x00
    assert bb.pack_fp8(1.0) == 0x40
    assert bb.pack_fp8(1.5) == 0x44
    assert bb.pack_fp8(2.0) == 0x48


def test_pack_bf8_scalar_reference_values() -> None:
    assert bb.pack_bf8(0.0) == 0x00
    assert bb.pack_bf8(1.0) == 0x40
    assert bb.pack_bf8(1.5) == 0x42
    assert bb.pack_bf8(2.0) == 0x44


def test_pack_and_unpack_fp8_tensor_roundtrip_known_values() -> None:
    packed = bb.pack_fp8([[0.0, 1.0], [1.5, 2.0]])
    assert packed.shape == (2, 2)
    assert packed.dtype == "fp8"
    assert packed.tolist() == [[0x00, 0x40], [0x44, 0x48]]

    unpacked = bb.unpack_fp8(packed)
    assert unpacked.dtype == "f32"
    assert unpacked.tolist()[0] == pytest.approx([0.0, 1.0])
    assert unpacked.tolist()[1] == pytest.approx([1.5, 2.0])


def test_pack_and_unpack_bf8_tensor_roundtrip_known_values() -> None:
    packed = bb.pack_bf8([[0.0, 1.0], [1.5, 2.0]])
    assert packed.shape == (2, 2)
    assert packed.dtype == "bf8"
    assert packed.tolist() == [[0x00, 0x40], [0x42, 0x44]]

    unpacked = bb.unpack_bf8(packed)
    assert unpacked.dtype == "f32"
    assert unpacked.tolist()[0] == pytest.approx([0.0, 1.0])
    assert unpacked.tolist()[1] == pytest.approx([1.5, 2.0])
