#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
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


def _call_sample_factory(factory, *, backend_name: str, target) -> object:
    signature = inspect.signature(factory)
    kwargs = {}
    candidates = {
        "backend_name": backend_name,
        "target": target,
        "target_arch": getattr(target, "arch", None),
    }
    accepts_var_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    for name, value in candidates.items():
        if name in signature.parameters or accepts_var_kwargs:
            kwargs[name] = value
    return factory(**kwargs)


def _sample_args_from_factory(factory, *, backend_name: str, target) -> tuple:
    produced = _call_sample_factory(factory, backend_name=backend_name, target=target)
    if isinstance(produced, dict):
        args = produced.get("compile_args", produced.get("args", ()))
    else:
        args = produced
    if not isinstance(args, tuple):
        args = tuple(args)
    return args


def _sample_payload_from_factory(factory, *, backend_name: str, target) -> dict[str, object]:
    produced = _call_sample_factory(factory, backend_name=backend_name, target=target)
    if isinstance(produced, dict):
        compile_args = produced.get("compile_args", produced.get("args", ()))
        run_args = produced.get("run_args", compile_args)
        result_indices = produced.get("result_indices")
    else:
        compile_args = produced
        run_args = produced
        result_indices = None
    if not isinstance(compile_args, tuple):
        compile_args = tuple(compile_args)
    if not isinstance(run_args, tuple):
        run_args = tuple(run_args)
    if result_indices is not None:
        result_indices = tuple(int(index) for index in result_indices)
    return {
        "compile_args": compile_args,
        "run_args": run_args,
        "result_indices": result_indices,
    }


def _is_executable_backend(name: str) -> bool:
    return name in {"hipcc_exec", "hipkittens_exec", "flydsl_exec", "waveasm_exec", "aster_exec", "ptx_exec"}


def _summarize_value(value):
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "tolist"):
        return {
            "type": type(value).__name__,
            "shape": tuple(int(dim) for dim in getattr(value, "shape")),
            "dtype": str(getattr(value, "dtype")),
            "value": value.tolist(),
        }
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _summarize_value(item) for key, item in value.items()}
    return value


def _summarize_timings_ms(timings_ms: list[float]) -> dict[str, float | list[float]]:
    if not timings_ms:
        return {"cold_ms": 0.0, "warm_median_ms": 0.0, "warm_timings_ms": []}
    warm_timings = timings_ms[1:] if len(timings_ms) > 1 else list(timings_ms)
    return {
        "cold_ms": float(timings_ms[0]),
        "warm_timings_ms": warm_timings,
        "warm_median_ms": float(statistics.median(warm_timings)),
    }


def _is_runtime_tensor_like(value) -> bool:
    return hasattr(value, "tolist") and hasattr(value, "shape") and hasattr(value, "dtype") and not (
        hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__")
    )


def _torch_device_available() -> bool:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return False
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return True
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return True


def _read_command_version(command: str) -> str | None:
    resolved = shutil.which(command)
    if resolved is None:
        return None
    for args in ((resolved, "--version"), (resolved, "-version"), (resolved, "-v")):
        try:
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            continue
        output = (result.stdout or result.stderr).strip()
        if output:
            return output.splitlines()[0]
    return resolved


def _module_version(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    try:
        module = importlib.import_module(name)
    except Exception:
        return "import_error"
    version = getattr(module, "__version__", None)
    if version is None and name == "baybridge":
        version = "local"
    return str(version) if version is not None else "imported"


def _make_execution_synchronizer(backend_name: str, run_args: tuple[object, ...]):
    if backend_name in {"hipcc_exec", "hipkittens_exec", "waveasm_exec", "aster_exec"}:
        try:
            from baybridge.hip_runtime import HipRuntime

            hip = HipRuntime()
        except Exception:
            return None

        return hip.synchronize
    if backend_name == "flydsl_exec":
        try:
            torch = importlib.import_module("torch")
        except Exception:
            return None
        cuda = getattr(torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None) if cuda is not None else None
        if callable(synchronize):
            return synchronize
    del run_args
    return None


def _environment_metadata(target: str | None, backends: tuple[str, ...]) -> dict[str, object]:
    tracked_env = {
        key: value
        for key in (
            "BAYBRIDGE_EXEC_ARCH",
            "BAYBRIDGE_FLYDSL_ROOT",
            "BAYBRIDGE_WAVEASM_ROOT",
            "BAYBRIDGE_HIPKITTENS_ROOT",
            "BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC",
            "LD_LIBRARY_PATH",
        )
        if (value := os.environ.get(key))
    }
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "platform": platform.platform(),
        },
        "target": target,
        "requested_backends": list(backends),
        "env": tracked_env,
        "modules": {
            "baybridge": _module_version("baybridge"),
            "torch": _module_version("torch"),
            "torch_device_available": _torch_device_available(),
            "flydsl": _module_version("flydsl"),
        },
        "tools": {
            "hipcc": _read_command_version("hipcc"),
            "waveasm-translate": _read_command_version("waveasm-translate"),
            "clang": _read_command_version("clang"),
            "ld.lld": _read_command_version("ld.lld"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a Baybridge kernel across multiple backends and print a JSON summary."
    )
    parser.add_argument("module", help="Python module name or path to a .py file")
    parser.add_argument("symbol", help="Kernel symbol inside the module")
    parser.add_argument(
        "--sample-factory",
        default=None,
        help="Optional symbol in the module that returns sample args for tracing non-annotated kernels",
    )
    parser.add_argument(
        "--backends",
        default="hipcc_exec,flydsl_exec,waveasm_ref,waveasm_exec",
        help="Comma-separated backend names",
    )
    parser.add_argument("--execute", action="store_true", help="Execute supported backends after compile")
    parser.add_argument("--repeat", type=int, default=1, help="Execution repetitions when --execute is used")
    parser.add_argument(
        "--include-env",
        action="store_true",
        help="Include Python/platform/tool/version metadata alongside the backend results",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--target", default=None, help="GPU target arch such as gfx942, gfx950, or sm_80")
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import baybridge as bb

    module = _load_module(args.module)
    kernel = _resolve_symbol(module, args.symbol)
    sample_factory = None
    if args.sample_factory:
        sample_factory = _resolve_symbol(module, args.sample_factory)
    if args.target:
        target = bb.NvidiaTarget(sm=args.target) if args.target.startswith("sm_") else bb.AMDTarget(arch=args.target)
    else:
        target = None
    cache_dir = args.cache_dir

    backend_names = tuple(item.strip() for item in args.backends.split(",") if item.strip())
    results: list[dict[str, object]] = []
    for backend_name in backend_names:
        sample_args = ()
        if sample_factory is not None:
            sample_args = _sample_args_from_factory(sample_factory, backend_name=backend_name, target=target)
        try:
            artifact = bb.compile(
                kernel,
                *sample_args,
                target=target,
                backend=backend_name,
                cache_dir=cache_dir,
            )
            results.append(
                {
                    "backend": backend_name,
                    "status": "ok",
                    "resolved_backend": artifact.backend_name,
                    "target": artifact.target.arch,
                    "lowered_path": str(artifact.lowered_path) if artifact.lowered_path else None,
                    "debug_bundle_dir": str(artifact.debug_bundle_dir) if artifact.debug_bundle_dir else None,
                    "dialect": artifact.lowered_module.dialect if artifact.lowered_module else None,
                }
            )
            entry = results[-1]
        except Exception as exc:  # pragma: no cover - exercised via subprocess tests
            results.append(
                {
                    "backend": backend_name,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        if not args.execute:
            continue
        if sample_factory is None:
            entry["execute_status"] = "skipped_no_sample_factory"
            continue
        if not _is_executable_backend(artifact.backend_name):
            entry["execute_status"] = "skipped_non_exec_backend"
            continue
        try:
            payload = _sample_payload_from_factory(sample_factory, backend_name=artifact.backend_name, target=target)
            run_args = tuple(payload["run_args"])
            result_indices = payload["result_indices"]
            if artifact.backend_name == "flydsl_exec" and any(_is_runtime_tensor_like(value) for value in run_args):
                if not _torch_device_available():
                    entry["execute_status"] = "skipped_incompatible_runtime_tensors"
                    entry["execute_note"] = (
                        "flydsl_exec RuntimeTensor execution requires a GPU-capable torch build; "
                        "use device-backed DLPack run_args or install ROCm/CUDA-enabled torch"
                    )
                    continue
            synchronize = _make_execution_synchronizer(artifact.backend_name, run_args)
            timings_ms: list[float] = []
            for _ in range(args.repeat):
                if synchronize is not None:
                    synchronize()
                start = time.perf_counter()
                artifact(*run_args)
                if synchronize is not None:
                    synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)
            indices = result_indices
            if indices is None:
                indices = tuple(index for index, value in enumerate(run_args) if hasattr(value, "tolist"))
            entry["execute_status"] = "ok"
            entry["timings_ms"] = timings_ms
            entry.update(_summarize_timings_ms(timings_ms))
            entry["result_summaries"] = {
                str(index): _summarize_value(run_args[index]) for index in indices
            }
        except Exception as exc:  # pragma: no cover - exercised via subprocess tests
            if artifact.backend_name == "flydsl_exec" and type(exc).__name__ == "BackendNotImplementedError":
                message = str(exc)
                if "BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC" in message:
                    entry["execute_status"] = "skipped_unvalidated_real_flydsl_exec"
                    entry["execute_note"] = message
                    continue
            entry["execute_status"] = "error"
            entry["execute_error_type"] = type(exc).__name__
            entry["execute_error"] = str(exc)
    payload: dict[str, object] | list[dict[str, object]]
    if args.include_env:
        payload = {
            "environment": _environment_metadata(args.target, backend_names),
            "results": results,
        }
    else:
        payload = results
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
