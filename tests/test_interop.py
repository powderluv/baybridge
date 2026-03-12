from pathlib import Path

import baybridge as bb


class FakeInteropTensor:
    shape = (4, 8)
    dtype = "torch.float16"

    def __dlpack__(self):
        return "capsule"

    def __dlpack_device__(self):
        return (2, 0)

    def stride(self):
        return (8, 1)

    def data_ptr(self):
        return 16384

    def tolist(self):
        return [[float(col) for col in range(8)] for _ in range(4)]


class MutableInteropTensor:
    def __init__(self, values: list[float]):
        self._values = list(values)
        self.shape = (len(values),)
        self.dtype = "torch.float32"

    def __dlpack__(self):
        return "capsule"

    def __dlpack_device__(self):
        return (2, 0)

    def stride(self):
        return (1,)

    def data_ptr(self):
        return 24576

    def tolist(self):
        return list(self._values)

    def copy_from_runtime_tensor(self, value: bb.Tensor) -> None:
        self._values = value.tolist()


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1)))
def interop_shape_kernel(src: bb.Tensor, out: bb.Tensor):
    out[0] = bb.Float16(bb.dim(src, 0))


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def interop_increment_kernel(src: bb.Tensor):
    tidx, _, _ = bb.arch.thread_idx()
    src[tidx] = src[tidx] + 1.0


def test_compile_accepts_direct_dlpack_like_inputs(tmp_path: Path) -> None:
    src = FakeInteropTensor()
    out = bb.zeros((1,), dtype="f16")

    artifact = bb.compile(interop_shape_kernel, src, out, cache_dir=tmp_path, backend="portable")
    assert artifact.ir is not None
    assert artifact.ir.arguments[0].spec.shape == (4, 8)
    assert artifact.ir.arguments[0].spec.dtype == "f16"


def test_make_ptr_accepts_tensor_handles() -> None:
    handle = bb.from_dlpack(FakeInteropTensor(), assumed_align=32)
    ptr = bb.make_ptr("f16", handle, bb.AddressSpace.GLOBAL, assumed_align=32)
    assert ptr.raw_address == 16384
    assert ptr.assumed_align == 32


def test_kernel_launch_writes_back_to_mutable_tensor_handle() -> None:
    source = MutableInteropTensor([1.0, 2.0, 3.0, 4.0])
    handle = bb.from_dlpack(source)

    interop_increment_kernel(handle).launch(grid=(1, 1, 1), block=(4, 1, 1))

    assert source.tolist() == [2.0, 3.0, 4.0, 5.0]
