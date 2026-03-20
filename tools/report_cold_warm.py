#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _compare_backends_script() -> Path:
    return Path(__file__).resolve().with_name("compare_backends.py")


def _run_compare_backends(args: list[str]) -> object:
    result = subprocess.run(
        [sys.executable, str(_compare_backends_script()), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _annotate_baseline_ratios(summary: dict[str, object], baseline_backend: str) -> bool:
    results = summary.get("results")
    if not isinstance(results, list):
        return False
    baseline_entry = None
    for entry in results:
        if not isinstance(entry, dict):
            continue
        if entry.get("backend") != baseline_backend:
            continue
        if entry.get("execute_status") != "ok":
            return False
        baseline_entry = entry
        break
    if baseline_entry is None:
        return False
    baseline_cold = float(baseline_entry["cold_ms"])
    baseline_warm = float(baseline_entry["warm_median_ms"])
    summary["baseline_backend"] = baseline_backend
    for entry in results:
        if not isinstance(entry, dict) or entry.get("execute_status") != "ok":
            continue
        entry["baseline_backend"] = baseline_backend
        entry["cold_ratio"] = float(entry["cold_ms"]) / baseline_cold
        entry["warm_median_ratio"] = float(entry["warm_median_ms"]) / baseline_warm
    return True


def _summarize_compare_payload(payload: object, baseline_backend: str | None = None) -> dict[str, object]:
    environment = None
    results = payload
    if isinstance(payload, dict):
        environment = payload.get("environment")
        results = payload.get("results", [])
    summaries: list[dict[str, object]] = []
    for entry in results if isinstance(results, list) else []:
        summary = {
            "backend": entry.get("backend"),
            "resolved_backend": entry.get("resolved_backend"),
            "target": entry.get("target"),
            "status": entry.get("status"),
            "execute_status": entry.get("execute_status"),
        }
        if entry.get("execute_status") == "ok" and "cold_ms" in entry and "warm_median_ms" in entry:
            summary.update(
                {
                    "cold_ms": entry["cold_ms"],
                    "warm_median_ms": entry["warm_median_ms"],
                    "warm_timings_ms": entry.get("warm_timings_ms", []),
                    "repeat": len(entry.get("timings_ms", [])),
                }
            )
        else:
            if "execute_error" in entry:
                summary["execute_error"] = entry["execute_error"]
            if "execute_note" in entry:
                summary["execute_note"] = entry["execute_note"]
        summaries.append(summary)
    summary = {"environment": environment, "results": summaries}
    if baseline_backend is not None and not _annotate_baseline_ratios(summary, baseline_backend):
        raise SystemExit(f"baseline backend '{baseline_backend}' was not found in successful results")
    return summary


def _format_summary(summary: dict[str, object]) -> str:
    lines: list[str] = []
    environment = summary.get("environment")
    if isinstance(environment, dict):
        target = environment.get("target")
        if target:
            lines.append(f"target={target}")
    baseline_backend = summary.get("baseline_backend")
    if isinstance(baseline_backend, str):
        lines.append(f"baseline={baseline_backend}")
    results = summary.get("results", [])
    if not isinstance(results, list):
        return "\n".join(lines)
    for entry in results:
        if not isinstance(entry, dict):
            continue
        backend = entry.get("backend", "<unknown>")
        execute_status = entry.get("execute_status")
        if execute_status == "ok":
            line = "{} cold={:.2f} warm_median={:.2f} repeat={}".format(
                backend,
                float(entry["cold_ms"]),
                float(entry["warm_median_ms"]),
                int(entry["repeat"]),
            )
            if "cold_ratio" in entry and "warm_median_ratio" in entry:
                line += " cold_ratio={:.2f} warm_ratio={:.2f}".format(
                    float(entry["cold_ratio"]),
                    float(entry["warm_median_ratio"]),
                )
            lines.append(line)
            continue
        note = entry.get("execute_note") or entry.get("execute_error") or entry.get("status")
        lines.append(f"{backend} {execute_status or 'unknown'} {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run compare_backends.py and print a cold/warm timing summary."
    )
    parser.add_argument("module", help="Python module name or path to a .py file")
    parser.add_argument("symbol", help="Kernel symbol inside the module")
    parser.add_argument("--sample-factory", default=None)
    parser.add_argument("--backends", required=True, help="Comma-separated backend names")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--include-env", action="store_true")
    parser.add_argument(
        "--baseline-backend",
        default=None,
        help="Optional backend name to use as the ratio baseline for cold and warm timings",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    compare_args = [args.module, args.symbol, "--backends", args.backends, "--execute", "--repeat", str(args.repeat)]
    if args.sample_factory:
        compare_args.extend(["--sample-factory", args.sample_factory])
    if args.cache_dir:
        compare_args.extend(["--cache-dir", args.cache_dir])
    if args.target:
        compare_args.extend(["--target", args.target])
    if args.include_env:
        compare_args.append("--include-env")

    payload = _run_compare_backends(compare_args)
    summary = _summarize_compare_payload(payload, baseline_backend=args.baseline_backend)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
