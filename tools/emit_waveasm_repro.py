#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path


def _load_module(module_ref: str):
    if module_ref.endswith(".py") or "/" in module_ref:
        module_path = Path(module_ref).expanduser().resolve()
        if not module_path.exists():
            raise FileNotFoundError(f"module path does not exist: {module_path}")
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load module from path: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def _resolve_symbol(module, name: str):
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise SystemExit(f"symbol '{name}' was not found in module '{module.__name__}'") from exc


def _sample_args_from_factory(factory) -> tuple:
    produced = factory()
    if isinstance(produced, dict):
        args = produced.get("args", ())
    else:
        args = produced
    if not isinstance(args, tuple):
        args = tuple(args)
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile an annotated Baybridge kernel with a WaveASM backend and print the repro bundle path."
    )
    parser.add_argument("module", help="Python module name or path to a .py file")
    parser.add_argument("symbol", help="Kernel symbol inside the module")
    parser.add_argument(
        "--sample-factory",
        default=None,
        help="Optional symbol in the module that returns sample args for tracing non-annotated kernels",
    )
    parser.add_argument("--backend", default="waveasm_ref", choices=("waveasm_ref", "waveasm_exec"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--target", default=None, help="AMD target arch such as gfx942 or gfx950")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import baybridge as bb

    module = _load_module(args.module)
    kernel = _resolve_symbol(module, args.symbol)
    sample_args = ()
    if args.sample_factory:
        sample_args = _sample_args_from_factory(_resolve_symbol(module, args.sample_factory))

    target = bb.AMDTarget(arch=args.target) if args.target else None
    bundle_dir = bb.emit_waveasm_repro(
        kernel,
        *sample_args,
        target=target,
        backend=args.backend,
        cache_dir=args.cache_dir,
    )
    print(bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
