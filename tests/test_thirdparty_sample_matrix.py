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
