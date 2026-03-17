# Backend Status

This document is the current backend inventory for Baybridge, including execution coverage and the small reproducible benchmark set used to compare the working executable backends.

## Scope

The status below separates three things:
- backend surface area
- validated execution coverage
- what was benchmarkable in the current repo-local remote environments on `mi355` and `mi300`

That split matters because some backends are integrated and tested, but still depend on extra runtime environment setup for ad hoc shell-driven benchmarking.

## Validation Baseline

Current repo state during this documentation pass:
- branch: `main`
- latest committed ASTER benchmark harness: `a11196a` `Add ASTER backend benchmark harness`
- worktree is dirty because the benchmark harness and status doc now include an ASTER-specific microbenchmark path:
  - `src/baybridge/backends/aster_exec.py`
  - `tools/backend_benchmark_kernels.py`
  - `tests/test_backend_benchmark_tools.py`
  - `tests/test_backend_aster_exec.py`
  - `docs/backend-status.md`

Focused local validation after the shared tooling updates in this pass:
- `tests/test_backend_benchmark_tools.py tests/test_backend_waveasm_ref.py -k 'benchmark_tools or compare_backends or emit_waveasm_repro'`
- result: `14 passed, 1 skipped, 4 deselected`

## Backend Inventory

| Backend | Kind | Current role | Validated target(s) | Current practical coverage | Notes |
| --- | --- | --- | --- | --- | --- |
| `portable` | IR only | PortableKernelIR capture | local | all traced kernels | No lowering or execution |
| `mlir_text` | text/ref | textual inspection | local | broad traced subset | Baybridge-specific textual IR |
| `gpu_text` | text/ref | textual GPU-flavored inspection | local | broad traced subset | Useful for debug only |
| `gpu_mlir` | text/ref | structured MLIR emission | local | broad traced subset | Base for external MLIR backends |
| `hipcc_exec` | exec | default general AMD executable path | `gfx950`, `gfx942` | broad traced kernel subset including copy, pointwise, broadcast, reductions, shared memory, many tensor helpers | Primary executable backend today |
| `hipkittens_ref` | ref | reference/source backend for HipKittens families | local, `gfx950`, `gfx942` | GEMM, attention-family, norm-family matching | Not executable |
| `hipkittens_exec` | exec | narrow AMD-native GEMM backend | `gfx950`, `gfx942` | BF16/F16 GEMM on supported tile families, including validated BF16 transpose families | Opt-in or auto-selected only for matching GEMM kernels |
| `flydsl_ref` | ref | reference FlyDSL lowering | local, `gfx950`, `gfx942` | elementwise, reductions, tiled/layout, MFMA-oriented family matching | Not executable |
| `flydsl_exec` | exec | narrow real FlyDSL execution path | validated subset on `gfx950`, `gfx942` | real validated subset: 1D `f32` copy and 1D `f32` pointwise `add/sub/mul/div` | Requires real FlyDSL env and a GPU-capable `torch` in the active venv for easy DLPack benchmarking |
| `waveasm_ref` | ref | WaveASM-oriented MLIR and repro bundle emission | local | supported GPU-MLIR subset | Emits `.waveasm_repro` bundles |
| `waveasm_exec` | exec, experimental | WaveASM HSACO build + HIP module launch | experimental on `gfx950`, `gfx942` | narrow pointwise/shared-memory subset only | Gated by `BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC=1`; upstream correctness issue still blocks real support |
| `aster_ref` | ref | ASTER-oriented MLIR and repro bundle emission | local, `gfx950`, `gfx942` | ASTER reference lowering for supported families | Emits `.aster_repro` bundles |
| `aster_exec` | exec | narrow ASTER executable backend | validated on `gfx950`, `gfx942` | dense contiguous copy: `f32/i32/f16`; dense contiguous binary: `f32/i32 add/sub/mul`; scalar broadcast from dense single-element tensors for supported binary ops; exact MFMA GEMM and fragment-copyout families for `f16/f16 -> f32` and `bf16/bf16 -> f32` on `16x16x16`; 1D and 2D dense tensors | `div` is intentionally not supported; MFMA support is intentionally exact-shape and exact-descriptor only |

## Execution Coverage Matrix

| Family | `hipcc_exec` | `hipkittens_exec` | `flydsl_exec` | `aster_exec` | `waveasm_exec` |
| --- | --- | --- | --- | --- | --- |
| Dense contiguous `f32` copy | Yes | No | Real validated 1D only | Yes | Experimental only |
| Dense contiguous `i32` copy | Yes | No | No | Yes | Experimental only |
| Dense contiguous `f16` copy | Yes | No | No | Yes | Experimental only |
| Dense contiguous `f32` binary `add/sub/mul/div` | Yes | No | Real validated 1D only | Add/sub/mul only | Experimental only |
| Dense contiguous `i32` binary `add/sub/mul/div` | Yes | No | No | Add/sub/mul only | No |
| Scalar broadcasted binary on dense tensors | Yes | No | No | Yes | No |
| Tensor reductions | Yes | No | Executable in Baybridge-side lowering; real upstream validation is still narrower | No | No |
| Shared-memory staging | Yes | No | Integrated, but real upstream shared-memory validation is still incomplete | No | Experimental only |
| GEMM | No dedicated path | Yes, narrow validated subset | No | Yes, exact `16x16x16` MFMA subset only | No |
| Attention / norm families | No dedicated path | ref only | ref only | ref only | ref only |

## Benchmark Method

Checked-in harness:
- kernel/sample module: [`tools/backend_benchmark_kernels.py`](/home/nod/github/baybridge/tools/backend_benchmark_kernels.py)
- runner: [`tools/compare_backends.py`](/home/nod/github/baybridge/tools/compare_backends.py)

Benchmark notes:
- timings are wall-clock `artifact(*args)` times in milliseconds
- compile is done once before measurement
- each run uses `--repeat 7`
- the first execution is usually a clear cold-start outlier
- the tables below report the median of the six warm runs after dropping the first cold-start outlier
- pointwise benchmark size: `65536` elements
- GEMM benchmark shape: `32x16 * 16x32 -> 32x32`
- ASTER microbenchmark size: `4096` elements
- ASTER MFMA benchmark shape: `16x16 * 16x16 -> 16x16`

## Performance Snapshot

### `mi355` (`gfx950`)

| Family | Backend | Median ms | Notes |
| --- | --- | ---: | --- |
| Dense `f32` copy, `65536` elements | `hipcc_exec` | `29.03` | Runtime benchmark completed cleanly |
| Dense `f32` copy, `65536` elements | `flydsl_exec` | `907.73` | Real upstream FlyDSL copy path measured with ROCm torch-backed inputs |
| Indexed `f32` add, `65536` elements | `hipcc_exec` | `33.52` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` add, `65536` elements | `flydsl_exec` | n/a | Correctly skipped as `skipped_unvalidated_real_flydsl_exec` |
| BF16 GEMM `32x16 * 16x32 -> 32x32` | `hipkittens_exec` | `0.84` | Narrow supported microkernel family |
| Dense `f32` copy, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |
| Dense `f32` add, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |

### `mi300` (`gfx942`)

| Family | Backend | Median ms | Notes |
| --- | --- | ---: | --- |
| Dense `f32` copy, `65536` elements | `hipcc_exec` | `46.91` | Runtime benchmark completed cleanly |
| Dense `f32` copy, `65536` elements | `flydsl_exec` | `1370.14` | Real upstream FlyDSL copy path measured with ROCm torch-backed inputs |
| Indexed `f32` add, `65536` elements | `hipcc_exec` | `58.01` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` add, `65536` elements | `flydsl_exec` | n/a | Correctly skipped as `skipped_unvalidated_real_flydsl_exec` |
| BF16 GEMM `32x16 * 16x32 -> 32x32` | `hipkittens_exec` | `1.46` | Narrow supported microkernel family |
| Dense `f32` copy, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |
| Dense `f32` add, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |

## ASTER Microbenchmark Snapshot

These are real checked-in-tool measurements for the current validated ASTER subset, using:
- kernel/sample module: [`tools/backend_benchmark_kernels.py`](/home/nod/github/baybridge/tools/backend_benchmark_kernels.py)
- runner: [`tools/compare_backends.py`](/home/nod/github/baybridge/tools/compare_backends.py)
- ASTER-specific sample factories:
  - `aster_dense_copy_f32_args`
  - `aster_dense_add_f32_args`

They are intentionally broken out from the common `65536`-element table above because ASTER is currently benchmarked through a smaller validated microkernel path.

### `mi355` (`gfx950`)

| Family | Backend | Warm median ms | Notes |
| --- | --- | ---: | --- |
| Dense `f32` copy, `4096` elements | `hipcc_exec` | `1.89` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` copy, `4096` elements | `aster_exec` | `7.33` | Real ASTER executable path |
| Dense `f32` add, `4096` elements | `hipcc_exec` | `2.59` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` add, `4096` elements | `aster_exec` | `7.53` | Real ASTER executable path |
| Dense `f32` sub, `4096` elements | `hipcc_exec` | `2.66` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` sub, `4096` elements | `aster_exec` | `7.53` | Real ASTER executable path |
| Dense `f32` mul, `4096` elements | `hipcc_exec` | `2.58` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` mul, `4096` elements | `aster_exec` | `7.56` | Real ASTER executable path |
| Dense `i32` copy, `4096` elements | `hipcc_exec` | `2.26` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` copy, `4096` elements | `aster_exec` | `7.32` | Real ASTER executable path |
| Dense `i32` add, `4096` elements | `hipcc_exec` | `3.13` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` add, `4096` elements | `aster_exec` | `7.57` | Real ASTER executable path |
| Dense `i32` sub, `4096` elements | `hipcc_exec` | `3.09` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` sub, `4096` elements | `aster_exec` | `7.45` | Real ASTER executable path |
| Dense `i32` mul, `4096` elements | `hipcc_exec` | `3.14` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` mul, `4096` elements | `aster_exec` | `7.64` | Real ASTER executable path |
| Dense `f16` copy, `4096` elements | `hipcc_exec` | `3.92` | Same kernel/sample factory as ASTER baseline |
| Dense `f16` copy, `4096` elements | `aster_exec` | `7.46` | Real ASTER executable path |
| Dense `f32` broadcast add, `4096` elements | `hipcc_exec` | `1.69` | Dense source plus single-element RHS tensor |
| Dense `f32` broadcast add, `4096` elements | `aster_exec` | `7.78` | Broadcast now validated on the aligned scalar-broadcast path |
| Dense `i32` broadcast add, `4096` elements | `hipcc_exec` | `1.98` | Dense source plus single-element RHS tensor |
| Dense `i32` broadcast add, `4096` elements | `aster_exec` | `7.64` | Broadcast now validated on the aligned scalar-broadcast path |

### `mi300` (`gfx942`)

| Family | Backend | Warm median ms | Notes |
| --- | --- | ---: | --- |
| Dense `f32` copy, `4096` elements | `hipcc_exec` | `3.00` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` copy, `4096` elements | `aster_exec` | `10.62` | Real ASTER executable path |
| Dense `f32` add, `4096` elements | `hipcc_exec` | `4.28` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` add, `4096` elements | `aster_exec` | `11.49` | Real ASTER executable path |
| Dense `f32` sub, `4096` elements | `hipcc_exec` | `4.26` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` sub, `4096` elements | `aster_exec` | `11.22` | Real ASTER executable path |
| Dense `f32` mul, `4096` elements | `hipcc_exec` | `4.28` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` mul, `4096` elements | `aster_exec` | `11.21` | Real ASTER executable path |
| Dense `i32` copy, `4096` elements | `hipcc_exec` | `3.89` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` copy, `4096` elements | `aster_exec` | `10.71` | Real ASTER executable path |
| Dense `i32` add, `4096` elements | `hipcc_exec` | `5.38` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` add, `4096` elements | `aster_exec` | `11.20` | Real ASTER executable path |
| Dense `i32` sub, `4096` elements | `hipcc_exec` | `5.29` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` sub, `4096` elements | `aster_exec` | `11.13` | Real ASTER executable path |
| Dense `i32` mul, `4096` elements | `hipcc_exec` | `5.22` | Same kernel/sample factory as ASTER baseline |
| Dense `i32` mul, `4096` elements | `aster_exec` | `11.29` | Real ASTER executable path |
| Dense `f16` copy, `4096` elements | `hipcc_exec` | `6.83` | Same kernel/sample factory as ASTER baseline |
| Dense `f16` copy, `4096` elements | `aster_exec` | `10.78` | Real ASTER executable path |
| Dense `f32` broadcast add, `4096` elements | `hipcc_exec` | `2.68` | Dense source plus single-element RHS tensor |
| Dense `f32` broadcast add, `4096` elements | `aster_exec` | `11.36` | Broadcast now validated on the aligned scalar-broadcast path |
| Dense `i32` broadcast add, `4096` elements | `hipcc_exec` | `3.33` | Dense source plus single-element RHS tensor |
| Dense `i32` broadcast add, `4096` elements | `aster_exec` | `11.37` | Broadcast now validated on the aligned scalar-broadcast path |

## ASTER MFMA Snapshot

These are real checked-in-tool measurements for the validated ASTER MFMA path, using:
- kernel/sample module: [`tools/backend_benchmark_kernels.py`](/home/nod/github/baybridge/tools/backend_benchmark_kernels.py)
- runner: [`tools/compare_backends.py`](/home/nod/github/baybridge/tools/compare_backends.py)
- ASTER-specific sample factories:
  - `aster_mfma_f16_gemm_args`
  - `aster_mfma_bf16_gemm_args`

This is intentionally separated from the HipKittens GEMM row in the main table. The currently validated checked-in executable shapes do not overlap cleanly:
- `hipkittens_exec` is validated on its own tile families
- `aster_exec` is validated on exact `16x16x16` MFMA GEMM and equivalent fragment-copyout forms

The timings below are for the direct `bb.gemm(...)` form. The checked-in fragment-copyout forms route through the same ASTER executable family and were not timed separately.

### `mi355` (`gfx950`)

| Family | Backend | Warm median ms | Notes |
| --- | --- | ---: | --- |
| MFMA GEMM `f16/f16 -> f32`, `16x16 * 16x16 -> 16x16` | `aster_exec` | `3.63` | Exact `16x16x16` ASTER MFMA path |
| MFMA GEMM `bf16/bf16 -> f32`, `16x16 * 16x16 -> 16x16` | `aster_exec` | `3.08` | Exact `16x16x16` ASTER MFMA path |

### `mi300` (`gfx942`)

| Family | Backend | Warm median ms | Notes |
| --- | --- | ---: | --- |
| MFMA GEMM `f16/f16 -> f32`, `16x16 * 16x16 -> 16x16` | `aster_exec` | `2.47` | Exact `16x16x16` ASTER MFMA path |
| MFMA GEMM `bf16/bf16 -> f32`, `16x16 * 16x16 -> 16x16` | `aster_exec` | `2.45` | Exact `16x16x16` ASTER MFMA path |

## ASTER MFMA Cold vs Warm Snapshot

Cold-start timing here is the first recorded execution in the `--repeat 7` run. Warm median is the median of the following six steady-state executions.

| Family | `mi355` cold ms | `mi355` warm median ms | `mi300` cold ms | `mi300` warm median ms |
| --- | ---: | ---: | ---: | ---: |
| MFMA GEMM `f16/f16 -> f32`, `16x16x16` | `1211.51` | `3.63` | `295.70` | `2.47` |
| MFMA GEMM `bf16/bf16 -> f32`, `16x16x16` | `292.94` | `3.08` | `296.97` | `2.45` |

## ASTER Ratio Summary

Warm median ratio of `aster_exec / hipcc_exec` on the matched `4096`-element ASTER microbenchmarks.

| Family | `mi355` ratio | `mi300` ratio |
| --- | ---: | ---: |
| Dense `f32` copy | `3.89x` | `3.55x` |
| Dense `f32` add | `2.91x` | `2.68x` |
| Dense `f32` sub | `2.82x` | `2.63x` |
| Dense `f32` mul | `2.93x` | `2.62x` |
| Dense `i32` copy | `3.22x` | `2.75x` |
| Dense `i32` add | `2.42x` | `2.08x` |
| Dense `i32` sub | `2.41x` | `2.11x` |
| Dense `i32` mul | `2.43x` | `2.16x` |
| Dense `f16` copy | `1.90x` | `1.58x` |
| Dense `f32` broadcast add | `4.60x` | `4.24x` |
| Dense `i32` broadcast add | `3.86x` | `3.41x` |

## ASTER Cold vs Warm Snapshot

Cold-start timing here is the first recorded execution in the `--repeat 7` run. Warm median is the median of the following six steady-state executions.

| Family | `mi355` cold ms | `mi355` warm median ms | `mi300` cold ms | `mi300` warm median ms |
| --- | ---: | ---: | ---: | ---: |
| Dense `f32` copy | `18927.18` | `7.33` | `37555.55` | `10.62` |
| Dense `f32` add | `74842.37` | `7.53` | `149174.42` | `11.49` |
| Dense `f32` sub | `73508.35` | `7.53` | `147932.72` | `11.22` |
| Dense `f32` mul | `73691.14` | `7.56` | `148651.39` | `11.21` |
| Dense `i32` copy | `18901.67` | `7.32` | `37700.85` | `10.71` |
| Dense `i32` add | `74548.61` | `7.57` | `148596.15` | `11.20` |
| Dense `i32` sub | `73920.99` | `7.45` | `148510.31` | `11.13` |
| Dense `i32` mul | `74301.16` | `7.64` | `147868.12` | `11.29` |
| Dense `f16` copy | `179.84` | `7.46` | `161.92` | `10.78` |
| Dense `f32` broadcast add | `76750.55` | `7.78` | `153196.57` | `11.36` |
| Dense `i32` broadcast add | `76450.53` | `7.64` | `153626.47` | `11.37` |

## Environment Notes Behind Missing Numbers

### FlyDSL

Current benchmark shells on both `mi355` and `mi300` now have:
- `torch 2.10.0+rocm7.1`
- `torch.cuda.is_available() == True`

That is enough for the benchmark harness to measure the validated real `flydsl_exec` copy path.

The remaining boundary is semantic, not environmental:
- dense `f32` copy is benchmarkable
- the indexed add benchmark is still correctly skipped because Baybridge keeps that real upstream path behind the unvalidated-exec gate

### ASTER

`aster_exec` is integrated and the current focused pytest path is healthy again on both `gfx950` and `gfx942` for:
- dense contiguous copy: `f32/i32/f16`
- dense contiguous pointwise binary: `f32/i32 add/sub/mul`
- dense scalar-broadcast binary for the same supported ops
- exact MFMA GEMM and fragment-copyout forms for:
  - `f16/f16 -> f32`, `16x16x16`
  - `bf16/bf16 -> f32`, `16x16x16`

Two important boundaries remain:
- `div` is intentionally unsupported in `aster_exec` because ASTER's current pass pipeline rejects the LSIR divide path in Baybridge's kernel form
- ASTER performance is currently published through two dedicated checked-in paths, not the common `65536`-element table:
  - `4096`-element dense copy/binary/broadcast microbenchmarks
  - exact `16x16x16` MFMA GEMM microbenchmarks
- the very large first-run cost is not Baybridge tracing time alone:
  - Baybridge compiles to ASTER MLIR before timing
  - then [aster_exec.py](/home/nod/github/baybridge/src/baybridge/backends/aster_exec.py) still does ASTER-side lazy work on first launch:
    - `_load_aster_modules()`
    - `_compile_to_hsaco(...)` when the `.hsaco` is not present yet
  - so the first timed call includes ASTER runtime import/bootstrap plus HSACO assembly, while warm calls mostly measure the steady-state execution path

So ASTER should currently be treated as:
- validated for its checked-in focused tests
- useful for executable copy, add/sub/mul, scalar-broadcast, and exact MFMA GEMM coverage
- benchmarkable through the checked-in ASTER microbenchmark harness above
- currently closest to `hipcc_exec` on `f16` copy in this microbench set, and furthest behind on scalar broadcast add
- currently shows much stronger warm behavior on the exact MFMA GEMM path than on the ASTER pointwise/broadcast families
- still carries a very large cold-start penalty on most `f32/i32` kernels, which is why the doc now separates cold and warm timing explicitly
- not yet a drop-in replacement for the common large-shape benchmark table used by `hipcc_exec` and `hipkittens_exec`

### WaveASM

`waveasm_exec` remains experimental and is intentionally excluded from the performance tables.

Reasons:
- upstream correctness issues remain open
- Baybridge already gates it behind `BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC=1`
- current usage model is debug/repro bundles, not performance execution

## Recommended Current Default Choices

- General executable backend: `hipcc_exec`
- Narrow GEMM backend on matching shapes: `hipkittens_exec`
- ASTER: use `aster_exec` for the exact validated MFMA or dense pointwise families; otherwise `aster_ref`
- FlyDSL: use `flydsl_ref` or `flydsl_exec` only when the active venv has a real FlyDSL build plus GPU-capable `torch`
- WaveASM: keep `waveasm_exec` disabled by default and use `waveasm_ref` repro bundles instead

## Reproduction Commands

`mi355` example:

```bash
cd ~/tmp/baybridge-codex
PATH=$PWD/.venv/bin:$PATH \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
BAYBRIDGE_HIPKITTENS_ROOT=$HOME/tmp/HipKittens \
.venv/bin/python tools/compare_backends.py \
  tools/backend_benchmark_kernels.py indexed_add_f32_kernel \
  --sample-factory indexed_add_f32_args \
  --backends hipcc_exec \
  --target gfx950 \
  --execute \
  --repeat 7
```

`mi300` example:

```bash
cd ~/tmp/baybridge-codex
PATH=$PWD/.venv/bin:$PATH \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
BAYBRIDGE_HIPKITTENS_ROOT=$HOME/tmp/HipKittens \
.venv/bin/python tools/compare_backends.py \
  tools/backend_benchmark_kernels.py hipkittens_bf16_gemm_kernel \
  --sample-factory hipkittens_bf16_gemm_args \
  --backends hipkittens_exec \
  --target gfx942 \
  --execute \
  --repeat 7
```

ASTER microbenchmark example on `mi355`:

```bash
cd ~/tmp/baybridge-codex
PATH=$PWD/.venv/bin:$PATH \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
BAYBRIDGE_ASTER_ROOT=$HOME/tmp/ASTER \
.venv/bin/python tools/compare_backends.py \
  tools/backend_benchmark_kernels.py dense_add_f32_kernel \
  --sample-factory aster_dense_add_f32_args \
  --backends aster_exec \
  --target gfx950 \
  --execute \
  --repeat 7
```

ASTER MFMA microbenchmark example on `mi355`:

```bash
cd ~/tmp/baybridge-codex
PATH=$PWD/.venv/bin:$PATH \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
BAYBRIDGE_ASTER_ROOT=$HOME/tmp/ASTER \
.venv/bin/python tools/compare_backends.py \
  tools/backend_benchmark_kernels.py aster_mfma_bf16_gemm_kernel \
  --sample-factory aster_mfma_bf16_gemm_args \
  --backends aster_exec \
  --target gfx950 \
  --execute \
  --repeat 7
```
