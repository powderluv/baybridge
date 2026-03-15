#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

SUPPORTED_DSL = {
    "AddressSpace",
    "ComposedLayout",
    "CopyAtom",
    "HierarchicalLayout",
    "IdentityLayout",
    "LaunchConfig",
    "Layout",
    "Boolean",
    "BFloat16",
    "Float16",
    "Float32",
    "Int",
    "Int8",
    "Int32",
    "Int64",
    "Pointer",
    "ReductionOp",
    "Shape",
    "Swizzle",
    "Tensor",
    "TensorSSA",
    "TensorSpec",
    "TiledCopy",
    "TiledMma",
    "ScalarSpec",
    "arch",
    "acos",
    "all_",
    "any_",
    "autovec_copy",
    "assume",
    "asin",
    "atan",
    "atan2",
    "barrier",
    "basic_copy",
    "basic_copy_if",
    "blocked_product",
    "block_dim",
    "block_idx",
    "ceil_div",
    "coalesce",
    "cos",
    "composition",
    "commit_group",
    "compile",
    "cosize",
    "copy",
    "copy_async",
    "depth",
    "dim",
    "domain_offset",
    "range",
    "range_constexpr",
    "empty_like",
    "elem_less",
    "flat_divide",
    "flat_product",
    "full",
    "full_like",
    "from_dlpack",
    "GenerateLineInfo",
    "gemm",
    "grid_dim",
    "group_modes",
    "jit",
    "kernel",
    "lane_id",
    "lane_idx",
    "local_partition",
    "local_tile",
    "load",
    "log",
    "log10",
    "log2",
    "logical_divide",
    "logical_product",
    "make_atom",
    "make_copy_atom",
    "make_composed_layout",
    "make_fragment",
    "make_fragment_like",
    "make_mma_atom",
    "make_fragment_a",
    "make_fragment_b",
    "make_identity_layout",
    "make_identity_tensor",
    "make_layout",
    "make_layout_tv",
    "make_ordered_layout",
    "make_tiled_copy",
    "make_tiled_copy_A",
    "make_tiled_copy_B",
    "make_tiled_copy_C",
    "make_tiled_copy_C_atom",
    "make_tiled_copy_D",
    "make_tiled_copy_S",
    "make_tiled_mma",
    "make_cotiled_copy",
    "make_rmem_tensor_like",
    "make_rmem_tensor",
    "make_swizzle",
    "make_warp_uniform",
    "make_tensor",
    "make_tiled_copy_tv",
    "mma",
    "math",
    "partition",
    "partition_program",
    "partition_thread",
    "partition_wave",
    "ones_like",
    "printf",
    "print_tensor",
    "product_each",
    "program_id",
    "raked_product",
    "recast_layout",
    "repeat_like",
    "repeat",
    "repeat_as_tuple",
    "recast_ptr",
    "runtime",
    "select",
    "size",
    "size_in_bytes",
    "slice_",
    "store",
    "struct",
    "prefetch",
    "tile_to_shape",
    "tiled_divide",
    "tiled_product",
    "tensor",
    "testing",
    "thread_idx",
    "transform_apply",
    "tuple_cat",
    "typing",
    "rsqrt",
    "sin",
    "sqrt",
    "wait_group",
    "warp_idx",
    "wave_id",
    "where",
    "zeros_like",
    "zipped_product",
    "zeros",
    "zipped_divide",
}

MISSING_DSL = set()

NVIDIA_SPECIFIC = {"nvgpu"}

SUPPORTED_NVIDIA_MARKERS = {
    "nvgpu.CopyUniversalOp",
    "nvgpu.MmaUniversalOp",
    "nvgpu.cpasync.CopyBulkTensorTileG2SOp",
    "nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp",
    "nvgpu.cpasync.CopyBulkTensorTileS2GOp",
    "nvgpu.cpasync.CopyDsmemStoreOp",
    "nvgpu.cpasync.CopyG2SOp",
    "nvgpu.cpasync.LoadCacheMode",
    "nvgpu.cpasync.ReductionOp",
    "nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp",
    "nvgpu.cpasync.get_tma_copy_properties",
    "nvgpu.cpasync.is_tma_load",
    "nvgpu.cpasync.is_tma_reduce",
    "nvgpu.cpasync.is_tma_store",
    "nvgpu.cpasync.prefetch_descriptor",
    "nvgpu.cpasync.tma_partition",
    "nvgpu.make_tiled_tma_atom_A",
    "nvgpu.make_tiled_tma_atom_B",
    "nvgpu.warp.MmaF16BF16Op",
    "nvgpu.warp.LdMatrix8x8x16bOp",
    "tcgen05.CtaGroup",
    "tcgen05.Field",
    "tcgen05.Ld32x32bOp",
    "tcgen05.MmaF16BF16Op",
    "tcgen05.OperandMajorMode",
    "tcgen05.OperandSource",
    "tcgen05.Repetition",
    "tcgen05.make_tmem_copy",
}

UNSUPPORTED_NVIDIA_MARKERS = {}

EXTERNAL_IMPORT_MARKERS = {
    "jax": "jax",
    "torch": "torch",
    "tvm": "tvm",
}

STATUS_OVERRIDES = {
    "examples/python/CuTeDSL/notebooks/hello_world.ipynb": {
        "status": "covered_by_baybridge_tests",
        "notes": "Covered by tests/test_examples_runtime.py::test_hello_world_example_runs_via_compile_fallback",
    },
    "examples/python/CuTeDSL/notebooks/elementwise_add.ipynb": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The naive elementwise add portion is covered by tests/test_examples_runtime.py and tests/test_examples_tracing.py",
    },
    "examples/python/CuTeDSL/ampere/elementwise_apply.py": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The core op + input-list + coordinate-predicate + TV-layout kernel shape is covered by tests/test_elementwise_apply_like.py",
    },
    "examples/python/CuTeDSL/ampere/cooperative_launch.py": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The cooperative launch plus grid-wide barrier execution path is covered by tests/test_cooperative_launch.py",
    },
    "examples/python/CuTeDSL/ampere/elementwise_add.py": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The core copy-atom + tiled-copy + predicated TV-layout kernel shape is covered by tests/test_elementwise_add_copyatom_like.py",
    },
    "examples/python/CuTeDSL/ampere/elementwise_add_autotune.py": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The copy-atom/tiled-copy kernel shape plus autotune_jit wrapper surface is covered by tests/test_elementwise_add_copyatom_like.py",
    },
    "examples/python/CuTeDSL/ampere/dynamic_smem_size.py": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The shared-memory struct allocation and inferred smem size path is covered by tests/test_struct_allocator.py",
    },
    "examples/python/CuTeDSL/ampere/smem_allocator.py": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The shared-memory allocator kernel shape with struct fields and recast raw storage is covered by tests/test_struct_allocator.py",
    },
    "examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The notebook's coalesce/divide/product helper surface is covered by tests/test_layout_algebra.py",
    },
    "examples/python/CuTeDSL/notebooks/tensorssa.ipynb": {
        "status": "partially_covered_by_baybridge_tests",
        "notes": "The runtime TensorSSA slice, unary math surface, and reduction API are covered by tests/test_tensor_ssa.py",
    },
    "examples/python/CuTeDSL/notebooks/data_types.ipynb": {
        "status": "covered_by_baybridge_tests",
        "notes": "The typed scalar constructors, casts, operators, and executable kernel path are covered by tests/test_data_types.py",
    },
}


def iter_sources(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.py"))
    yield from sorted(root.rglob("*.ipynb"))


def source_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        snippets = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                snippets.append("".join(cell.get("source", [])))
        return "\n\n".join(snippets)
    return path.read_text(encoding="utf-8")


def dsl_symbols(text: str) -> list[str]:
    return sorted(set(re.findall(r"\bcute\.([A-Za-z_][A-Za-z0-9_]*)", text)))


def external_markers(text: str) -> list[str]:
    found = []
    for marker, label in EXTERNAL_IMPORT_MARKERS.items():
        if marker in text:
            found.append(label)
    return sorted(found)


def dynamic_control_flow(text: str) -> bool:
    return "cutlass.dynamic_expr" in text or "if cutlass.dynamic_expr" in text


def nvidia_markers(text: str, dsl_names: list[str]) -> list[str]:
    if "nvgpu" not in dsl_names and not any(marker in text for marker in UNSUPPORTED_NVIDIA_MARKERS):
        return []
    unsupported = sorted(
        {
            label
            for marker, label in UNSUPPORTED_NVIDIA_MARKERS.items()
            if marker in text
        }
    )
    if unsupported:
        return unsupported
    if any(marker in text for marker in SUPPORTED_NVIDIA_MARKERS):
        return []
    if "nvgpu" in dsl_names:
        return ["nvgpu"]
    return []


def classify(relative_path: str, text: str) -> dict[str, object]:
    override = STATUS_OVERRIDES.get(relative_path)
    dsl_names = dsl_symbols(text)
    missing = sorted(name for name in dsl_names if name in MISSING_DSL)
    nvidia_only = nvidia_markers(text, dsl_names)
    unknown = sorted(name for name in dsl_names if name not in SUPPORTED_DSL | MISSING_DSL | NVIDIA_SPECIFIC)
    deps = external_markers(text)

    if override is not None:
        status = override["status"]
        notes = override["notes"]
    elif nvidia_only:
        status = "blocked_nvidia_specific"
        notes = "Uses NVIDIA-only architecture-specific APIs"
    elif unknown or missing:
        status = "blocked_missing_api"
        notes = "Requires DSL APIs Baybridge does not implement yet"
    elif deps:
        status = "blocked_external_dependency"
        notes = "Requires external framework/runtime integrations not wired into Baybridge"
    elif dynamic_control_flow(text):
        status = "runtime_only_candidate"
        notes = "Could execute on the reference runtime, but traced compilation will still fall back"
    else:
        status = "candidate_for_direct_port"
        notes = "Uses mostly low-level constructs already present in Baybridge"

    blockers = []
    blockers.extend(f"missing:{name}" for name in missing)
    blockers.extend(f"unknown:{name}" for name in unknown)
    blockers.extend(f"nvidia:{name}" for name in nvidia_only)
    blockers.extend(f"dep:{dep}" for dep in deps)
    if dynamic_control_flow(text):
        blockers.append("dynamic_python_control_flow")

    return {
        "path": relative_path,
        "status": status,
        "notes": notes,
        "dsl_symbols": dsl_names,
        "blockers": blockers,
    }


def render_markdown(entries: list[dict[str, object]], metadata_text: str) -> str:
    counts = Counter(entry["status"] for entry in entries)
    lines = [
        "# Third-Party Sample Compatibility Matrix",
        "",
        "This file is generated from the fetched third-party sample corpus.",
        "",
        "## Upstream",
        "",
        "```text",
        metadata_text.strip(),
        "```",
        "",
        "## Status Summary",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Entries",
            "",
            "| Status | Path | Primary blockers |",
            "| --- | --- | --- |",
        ]
    )
    for entry in entries:
        blockers = ", ".join(entry["blockers"][:4]) if entry["blockers"] else "-"
        lines.append(f"| `{entry['status']}` | `{entry['path']}` | `{blockers}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the fetched third-party DSL sample corpus")
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "thirdparty" / "dsl_samples",
    )
    args = parser.parse_args()

    sample_root = args.sample_root.resolve()
    upstream_root = sample_root / "upstream"
    metadata_path = sample_root / "FETCHED.txt"
    if not metadata_path.exists():
        metadata_path = sample_root / "MANIFEST.json"
    metadata_text = metadata_path.read_text(encoding="utf-8")

    entries = []
    for path in iter_sources(upstream_root):
        relative = path.relative_to(upstream_root).as_posix()
        entries.append(classify(relative, source_text(path)))
    entries.sort(key=lambda item: item["path"])

    (sample_root / "compatibility_matrix.json").write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (sample_root / "compatibility_matrix.md").write_text(
        render_markdown(entries, metadata_text) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
