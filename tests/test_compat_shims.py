from pathlib import Path

import baybridge as bb


@bb.kernel(launch=bb.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def shim_indexed_add_kernel(
    g_a: bb.Tensor,
    g_b: bb.Tensor,
    g_c: bb.Tensor,
):
    tidx, _, _ = bb.arch.thread_idx()
    g_c[tidx] = g_a[tidx] + g_b[tidx]


def test_compile_supports_generate_line_info_option(tmp_path: Path) -> None:
    a = bb.tensor([1.0, 2.0, 3.0, 4.0], dtype="f32")
    b = bb.tensor([10.0, 20.0, 30.0, 40.0], dtype="f32")
    c = bb.zeros((4,), dtype="f32")

    artifact = bb.compile[bb.GenerateLineInfo(True)](
        shim_indexed_add_kernel,
        a,
        b,
        c,
        cache_dir=tmp_path,
        backend="gpu_text",
    )

    assert artifact.ir is not None
    assert artifact.ir.name == "shim_indexed_add_kernel"


def test_testing_module_provides_benchmark_and_jit_arguments() -> None:
    calls: list[int] = []

    def fn(value: int) -> None:
        calls.append(value)

    arguments = bb.testing.JitArguments(7)
    avg_time_us = bb.testing.benchmark(fn, kernel_arguments=arguments, warmup_iterations=1, iterations=3)

    assert avg_time_us >= 0.0
    assert calls == [7, 7, 7, 7]


def test_runtime_module_reexports_from_dlpack() -> None:
    assert bb.runtime.from_dlpack is bb.from_dlpack
