import baybridge as bb


class FakeDLPackTensor:
    shape = (4, 8)
    dtype = "float16"

    def __dlpack__(self):
        return "capsule"

    def __dlpack_device__(self):
        return (2, 7)


def test_from_dlpack_extracts_metadata() -> None:
    handle = bb.from_dlpack(FakeDLPackTensor())
    assert handle.capsule == "capsule"
    assert handle.shape == (4, 8)
    assert handle.dtype == "float16"
    assert handle.device_type == 2
    assert handle.device_id == 7
