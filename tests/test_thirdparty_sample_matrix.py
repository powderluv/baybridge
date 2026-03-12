import json
import subprocess
import sys
from pathlib import Path


def _entry(entries: list[dict[str, object]], path: str) -> dict[str, object]:
    for entry in entries:
        if entry["path"] == path:
            return entry
    raise AssertionError(f"missing compatibility entry for {path}")


def _write_ipynb(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [{"cell_type": "code", "metadata": {}, "source": [code]}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_thirdparty_sample_matrix_regenerates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_root = tmp_path / "dsl_samples"
    upstream_root = sample_root / "upstream"
    upstream_root.mkdir(parents=True)
    (sample_root / "FETCHED.txt").write_text(
        "repo_url=https://github.com/NVIDIA/cutlass.git\n"
        "git_ref=deadbeef\n"
        "resolved_head=deadbeef\n"
        "selection_count=11\n",
        encoding="utf-8",
    )

    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/hello_world.ipynb",
        "import baybridge as cute\n@cute.jit\ndef hello_world():\n    cute.printf('hi')\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/elementwise_add.ipynb",
        "import torch\nimport baybridge as cute\n"
        "cute.composition(None, None)\n"
        "cute.full_like(x, 0)\n"
        "cute.make_layout_tv(a, b)\n"
        "cute.make_ordered_layout((1, 4), order=(1, 0))\n"
        "cute.make_identity_tensor((4, 4))\n"
        "cute.elem_less(0, 1)\n"
        "cute.make_fragment((1,), 'f32')\n"
        "cute.print_tensor(x)\n"
        "cute.recast_layout(16, 8, layout)\n"
        "cute.repeat_like(None, shape)\n"
        "cute.where(True, 1, 0)\n"
        "cute.select(shape, mode=[0])\n"
        "cute.zipped_divide(x, (1, 4))\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/elementwise_apply.py",
        "import torch\nimport cuda.bindings.driver\nimport baybridge as cute\n"
        "cute.GenerateLineInfo(True)\n"
        "cute.runtime\n"
        "cute.testing\n"
        "cute.elem_less(0, 1)\n"
        "cute.make_layout_tv(a, b)\n"
        "cute.make_rmem_tensor((1,), 'f32')\n"
        "cute.recast_layout(16, 8, layout)\n"
        "cute.repeat_like(None, shape)\n"
        "cute.where(True, 1, 0)\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/elementwise_add.py",
        "import torch\nimport baybridge as cute\n"
        "cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), 'f32')\n"
        "cute.make_tiled_copy_tv(atom, thr, val)\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/elementwise_add_autotune.py",
        "import torch\nimport baybridge as cute\n"
        "cute.make_rmem_tensor_like(x)\n"
        "cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), 'f32')\n"
        "cute.make_tiled_copy_tv(atom, thr, val)\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/call_bypass_dlpack.py",
        "import torch\nimport baybridge as cute\n"
        "def fn(ptr: cute.Pointer):\n    return cute.assume(ptr, divby=16)\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/call_from_jit.py",
        "import torch\nimport baybridge as cute\n"
        "def fn(ptr: cute.Pointer):\n    return ptr\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/cooperative_launch.py",
        "import baybridge as cute\n@cute.jit\ndef run():\n    cute.arch.sync_grid()\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/inline_ptx.py",
        "import torch\nimport baybridge as cute\n"
        "from baybridge.typing import Boolean, Int32, Int, Constexpr\n"
        "def fn(x: Boolean, y: Constexpr[bool]):\n    return Int32(int(x)) + Int(y is Constexpr)\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/dynamic_smem_size.py",
        "import baybridge as cute\n"
        "@cute.struct\nclass SharedData:\n"
        "    values: cute.struct.MemRange['f32', 64]\n"
        "def run():\n"
        "    alloc = cute.SmemAllocator()\n"
        "    alloc.allocate(SharedData)\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/smem_allocator.py",
        "import torch\nimport baybridge as cute\n"
        "@cute.struct\nclass SharedStorage:\n"
        "    values: cute.struct.MemRange['f32', 32]\n"
        "def run(ptr):\n"
        "    alloc = cute.SmemAllocator()\n"
        "    section = alloc.allocate(64, byte_alignment=128)\n"
        "    cute.recast_ptr(section, dtype='f32')\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/sgemm.py",
        "import torch\nimport baybridge as cute\n"
        "def fn(x, y, layout: cute.ComposedLayout, atom: cute.CopyAtom, tiled: cute.TiledCopy, tiled_mma: cute.TiledMma):\n"
        "    cute.ceil_div((128, 128, 1), (64, 64, 1))\n"
        "    cute.size_in_bytes('f32', cute.make_layout((8, 64), stride=(64, 1)))\n"
        "    cute.tile_to_shape(cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))), (64, 64, 2), (0, 1, 2))\n"
        "    cute.domain_offset((0, 1), x)\n"
        "    cute.local_tile(x, tiler=(2, 4, 2), coord=(1, 0, None), proj=(1, None, 1))\n"
        "    cute.make_tiled_mma(obj)\n"
        "    cute.make_tiled_copy_A(atom, tiled_mma)\n"
        "    cute.make_tiled_copy_B(atom, tiled_mma)\n"
        "    cute.make_tiled_copy_C(atom, tiled_mma)\n"
        "    cute.gemm(x, y, x)\n"
        "    cute.group_modes(x, 0, 1)\n"
        "    cute.basic_copy(x, y)\n"
        "    cute.autovec_copy(x, y)\n"
        "    cute.nvgpu.CopyUniversalOp()\n",
    )
    _write_text(
        upstream_root / "examples/python/CuTeDSL/ampere/tensorop_gemm.py",
        "import torch\nimport baybridge as cute\n"
        "def fn(a, b, c):\n"
        "    cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL), 'f16', num_bits_per_copy=128)\n"
        "    cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), 'f16')\n"
        "    op = cute.nvgpu.warp.MmaF16BF16Op(cute.Float16, cute.Float32, (16, 8, 16))\n"
        "    cute.make_tiled_mma(op, cute.make_layout((2, 2, 1)), permutation_mnk=(32, 16, 16))\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/tour_to_sol_gemm.ipynb",
        "import torch\nimport baybridge as cute\n"
        "from baybridge.nvgpu import cpasync, tcgen05\n"
        "cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)\n"
        "cpasync.prefetch_descriptor(desc)\n"
        "cute.nvgpu.make_tiled_tma_atom_A(x)\n"
        "cute.nvgpu.cpasync.tma_partition(atom, tensor, coord)\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/print.ipynb",
        "import torch\nimport baybridge as cute\n"
        "cute.slice_(x, (1, None))\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/tensor.ipynb",
        "import torch\nimport baybridge as cute\n"
        "def fn(ptr: cute.Pointer):\n    return ptr\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/composed_layout.ipynb",
        "import torch\nimport baybridge as cute\n"
        "cute.make_composed_layout(inner, 0, outer)\n"
        "cute.make_identity_layout((4, 4))\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb",
        "import baybridge as cute\n"
        "layout = cute.make_layout((2, 5), stride=(5, 1))\n"
        "tiler = cute.make_layout((3, 4), stride=(1, 3))\n"
        "cute.logical_divide(layout, tiler=tiler)\n"
        "cute.tiled_divide(layout, tiler=tiler)\n"
        "cute.logical_product(layout, tiler=tiler)\n"
        "cute.zipped_product(layout, tiler=tiler)\n"
        "cute.tiled_product(layout, tiler=tiler)\n"
        "cute.flat_product(layout, tiler=tiler)\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/tensorssa.ipynb",
        "import baybridge as cute\n"
        "x = cute.tensor([[1.0, 2.0], [3.0, 4.0]])\n"
        "v = x.load()\n"
        "isinstance(v, cute.TensorSSA)\n"
        "cute.math.sqrt(v)\n"
        "v.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)\n",
    )
    _write_ipynb(
        upstream_root / "examples/python/CuTeDSL/notebooks/data_types.ipynb",
        "import baybridge as cute\n"
        "@cute.jit\n"
        "def demo():\n"
        "    cute.printf('{}', 1)\n",
    )

    subprocess.run(
        [sys.executable, "tools/analyze_thirdparty_samples.py", "--sample-root", str(sample_root)],
        cwd=repo_root,
        check=True,
    )

    entries = json.loads((sample_root / "compatibility_matrix.json").read_text(encoding="utf-8"))

    hello_world = _entry(entries, "examples/python/CuTeDSL/notebooks/hello_world.ipynb")
    assert hello_world["status"] == "covered_by_baybridge_tests"

    elementwise_add = _entry(entries, "examples/python/CuTeDSL/notebooks/elementwise_add.ipynb")
    assert elementwise_add["status"] == "partially_covered_by_baybridge_tests"
    assert "unknown:composition" not in elementwise_add["blockers"]
    assert "unknown:full_like" not in elementwise_add["blockers"]
    assert "unknown:make_layout_tv" not in elementwise_add["blockers"]
    assert "unknown:make_ordered_layout" not in elementwise_add["blockers"]
    assert "unknown:make_identity_tensor" not in elementwise_add["blockers"]
    assert "unknown:elem_less" not in elementwise_add["blockers"]
    assert "unknown:make_fragment" not in elementwise_add["blockers"]
    assert "unknown:print_tensor" not in elementwise_add["blockers"]
    assert "unknown:recast_layout" not in elementwise_add["blockers"]
    assert "unknown:repeat_like" not in elementwise_add["blockers"]
    assert "unknown:where" not in elementwise_add["blockers"]
    assert "unknown:select" not in elementwise_add["blockers"]
    assert "unknown:zipped_divide" not in elementwise_add["blockers"]

    elementwise_apply = _entry(entries, "examples/python/CuTeDSL/ampere/elementwise_apply.py")
    assert elementwise_apply["status"] == "partially_covered_by_baybridge_tests"
    assert "dep:cuda-python" not in elementwise_apply["blockers"]
    assert "unknown:GenerateLineInfo" not in elementwise_apply["blockers"]
    assert "unknown:runtime" not in elementwise_apply["blockers"]
    assert "unknown:testing" not in elementwise_apply["blockers"]
    assert "unknown:elem_less" not in elementwise_apply["blockers"]
    assert "unknown:make_layout_tv" not in elementwise_apply["blockers"]
    assert "unknown:make_rmem_tensor" not in elementwise_apply["blockers"]
    assert "unknown:recast_layout" not in elementwise_apply["blockers"]
    assert "unknown:repeat_like" not in elementwise_apply["blockers"]
    assert "unknown:where" not in elementwise_apply["blockers"]

    elementwise_add_kernel = _entry(entries, "examples/python/CuTeDSL/ampere/elementwise_add.py")
    assert elementwise_add_kernel["status"] == "partially_covered_by_baybridge_tests"
    assert "unknown:make_copy_atom" not in elementwise_add_kernel["blockers"]
    assert "unknown:make_tiled_copy_tv" not in elementwise_add_kernel["blockers"]

    elementwise_add_autotune = _entry(entries, "examples/python/CuTeDSL/ampere/elementwise_add_autotune.py")
    assert elementwise_add_autotune["status"] == "partially_covered_by_baybridge_tests"
    assert "unknown:make_rmem_tensor_like" not in elementwise_add_autotune["blockers"]
    assert "unknown:make_copy_atom" not in elementwise_add_autotune["blockers"]
    assert "unknown:make_tiled_copy_tv" not in elementwise_add_autotune["blockers"]

    call_bypass = _entry(entries, "examples/python/CuTeDSL/ampere/call_bypass_dlpack.py")
    assert call_bypass["status"] == "blocked_external_dependency"
    assert "unknown:Pointer" not in call_bypass["blockers"]
    assert "unknown:assume" not in call_bypass["blockers"]

    call_from_jit = _entry(entries, "examples/python/CuTeDSL/ampere/call_from_jit.py")
    assert call_from_jit["status"] == "blocked_external_dependency"
    assert "unknown:Pointer" not in call_from_jit["blockers"]

    cooperative_launch = _entry(entries, "examples/python/CuTeDSL/ampere/cooperative_launch.py")
    assert cooperative_launch["status"] == "partially_covered_by_baybridge_tests"

    inline_ptx = _entry(entries, "examples/python/CuTeDSL/ampere/inline_ptx.py")
    assert inline_ptx["status"] == "blocked_external_dependency"
    assert "unknown:typing" not in inline_ptx["blockers"]

    dynamic_smem = _entry(entries, "examples/python/CuTeDSL/ampere/dynamic_smem_size.py")
    assert dynamic_smem["status"] == "partially_covered_by_baybridge_tests"
    assert "unknown:struct" not in dynamic_smem["blockers"]

    smem_allocator = _entry(entries, "examples/python/CuTeDSL/ampere/smem_allocator.py")
    assert smem_allocator["status"] == "partially_covered_by_baybridge_tests"
    assert "unknown:struct" not in smem_allocator["blockers"]
    assert "unknown:recast_ptr" not in smem_allocator["blockers"]

    sgemm = _entry(entries, "examples/python/CuTeDSL/ampere/sgemm.py")
    assert sgemm["status"] == "blocked_external_dependency"
    assert "dep:cuda-python" not in sgemm["blockers"]
    assert "unknown:ComposedLayout" not in sgemm["blockers"]
    assert "unknown:CopyAtom" not in sgemm["blockers"]
    assert "unknown:TiledCopy" not in sgemm["blockers"]
    assert "unknown:ceil_div" not in sgemm["blockers"]
    assert "unknown:domain_offset" not in sgemm["blockers"]
    assert "unknown:local_tile" not in sgemm["blockers"]
    assert "unknown:TiledMma" not in sgemm["blockers"]
    assert "unknown:make_tiled_mma" not in sgemm["blockers"]
    assert "unknown:make_tiled_copy_A" not in sgemm["blockers"]
    assert "unknown:make_tiled_copy_B" not in sgemm["blockers"]
    assert "unknown:make_tiled_copy_C" not in sgemm["blockers"]
    assert "unknown:gemm" not in sgemm["blockers"]
    assert "unknown:group_modes" not in sgemm["blockers"]
    assert "unknown:size_in_bytes" not in sgemm["blockers"]
    assert "unknown:tile_to_shape" not in sgemm["blockers"]
    assert "unknown:make_swizzle" not in sgemm["blockers"]
    assert "unknown:basic_copy" not in sgemm["blockers"]
    assert "unknown:autovec_copy" not in sgemm["blockers"]
    assert "nvidia:nvgpu" not in sgemm["blockers"]

    tensorop_gemm = _entry(entries, "examples/python/CuTeDSL/ampere/tensorop_gemm.py")
    assert tensorop_gemm["status"] == "blocked_external_dependency"
    assert "dep:cuda-python" not in tensorop_gemm["blockers"]
    assert "nvidia:nvgpu" not in tensorop_gemm["blockers"]

    tour_to_sol = _entry(entries, "examples/python/CuTeDSL/notebooks/tour_to_sol_gemm.ipynb")
    assert tour_to_sol["status"] == "blocked_external_dependency"
    assert not any(blocker.startswith("nvidia:") for blocker in tour_to_sol["blockers"])

    print_notebook = _entry(entries, "examples/python/CuTeDSL/notebooks/print.ipynb")
    assert print_notebook["status"] == "blocked_external_dependency"
    assert "unknown:slice_" not in print_notebook["blockers"]

    tensor_notebook = _entry(entries, "examples/python/CuTeDSL/notebooks/tensor.ipynb")
    assert tensor_notebook["status"] == "blocked_external_dependency"
    assert "unknown:Pointer" not in tensor_notebook["blockers"]

    composed_layout = _entry(entries, "examples/python/CuTeDSL/notebooks/composed_layout.ipynb")
    assert composed_layout["status"] == "blocked_external_dependency"
    assert "unknown:make_composed_layout" not in composed_layout["blockers"]
    assert "unknown:make_identity_layout" not in composed_layout["blockers"]

    layout_algebra = _entry(entries, "examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb")
    assert layout_algebra["status"] == "partially_covered_by_baybridge_tests"
    assert "unknown:logical_divide" not in layout_algebra["blockers"]
    assert "unknown:tiled_divide" not in layout_algebra["blockers"]
    assert "unknown:logical_product" not in layout_algebra["blockers"]
    assert "unknown:zipped_product" not in layout_algebra["blockers"]
    assert "unknown:tiled_product" not in layout_algebra["blockers"]
    assert "unknown:flat_product" not in layout_algebra["blockers"]

    tensorssa = _entry(entries, "examples/python/CuTeDSL/notebooks/tensorssa.ipynb")
    assert tensorssa["status"] == "partially_covered_by_baybridge_tests"
    assert "unknown:TensorSSA" not in tensorssa["blockers"]
    assert "unknown:ReductionOp" not in tensorssa["blockers"]
    assert "unknown:math" not in tensorssa["blockers"]

    data_types = _entry(entries, "examples/python/CuTeDSL/notebooks/data_types.ipynb")
    assert data_types["status"] == "covered_by_baybridge_tests"
