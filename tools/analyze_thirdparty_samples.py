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
    "LaunchConfig",
    "Layout",
    "Pointer",
    "Shape",
    "Tensor",
    "TensorSpec",
    "ScalarSpec",
    "arch",
    "assume",
    "barrier",
    "block_dim",
    "block_idx",
    "composition",
    "commit_group",
    "compile",
    "copy",
    "copy_async",
    "dim",
    "elem_less",
    "full_like",
    "from_dlpack",
    "GenerateLineInfo",
    "grid_dim",
    "jit",
    "kernel",
    "lane_id",
    "load",
    "make_copy_atom",
    "make_composed_layout",
    "make_fragment",
    "make_fragment_like",
    "make_fragment_a",
    "make_fragment_b",
    "make_identity_layout",
    "make_identity_tensor",
    "make_layout",
    "make_layout_tv",
    "make_ordered_layout",
    "make_rmem_tensor_like",
    "make_rmem_tensor",
    "make_tensor",
    "make_tiled_copy_tv",
    "mma",
    "partition",
    "partition_program",
    "partition_thread",
    "partition_wave",
    "printf",
    "print_tensor",
    "product_each",
    "program_id",
    "recast_layout",
    "repeat_like",
    "runtime",
    "select",
    "size",
    "slice_",
    "store",
    "tensor",
    "testing",
    "thread_idx",
    "wait_group",
    "wave_id",
    "where",
    "zeros",
    "zipped_divide",
}

MISSING_DSL = set()

NVIDIA_SPECIFIC = {
    "nvgpu",
}

EXTERNAL_IMPORT_MARKERS = {
    "cuda.bindings.driver": "cuda-python",
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


def classify(relative_path: str, text: str) -> dict[str, object]:
    override = STATUS_OVERRIDES.get(relative_path)
    dsl_names = dsl_symbols(text)
    missing = sorted(name for name in dsl_names if name in MISSING_DSL)
    nvidia_only = sorted(name for name in dsl_names if name in NVIDIA_SPECIFIC)
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
