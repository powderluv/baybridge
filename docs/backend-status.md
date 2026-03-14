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
- latest committed ASTER/runtime fix: `b21303b` `Fix ASTER exec support and runtime bootstrap`
- worktree is dirty because the benchmark harness and status doc now include an ASTER-specific microbenchmark path:
  - `tools/backend_benchmark_kernels.py`
  - `tools/compare_backends.py`
  - `tests/test_backend_benchmark_tools.py`
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
| `aster_exec` | exec | narrow ASTER executable backend | validated on `gfx950`, `gfx942` | dense contiguous copy: `f32/i32/f16`; dense contiguous binary: `f32/i32 add/sub/mul`; scalar broadcast from dense single-element tensors for supported binary ops; 1D and 2D dense tensors | `div` is intentionally not supported; standalone benchmark-shell execution is still less stable than focused pytest execution |

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
| GEMM | No dedicated path | Yes, narrow validated subset | No | No | No |
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

### `mi300` (`gfx942`)

| Family | Backend | Warm median ms | Notes |
| --- | --- | ---: | --- |
| Dense `f32` copy, `4096` elements | `hipcc_exec` | `3.00` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` copy, `4096` elements | `aster_exec` | `10.62` | Real ASTER executable path |
| Dense `f32` add, `4096` elements | `hipcc_exec` | `4.28` | Same kernel/sample factory as ASTER baseline |
| Dense `f32` add, `4096` elements | `aster_exec` | `11.49` | Real ASTER executable path |

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

Two important boundaries remain:
- `div` is intentionally unsupported in `aster_exec` because ASTER's current pass pipeline rejects the LSIR divide path in Baybridge's kernel form
- ASTER performance is currently published through a dedicated `4096`-element checked-in microbenchmark path, not the common `65536`-element table

So ASTER should currently be treated as:
- validated for its checked-in focused tests
- useful for executable copy and add/sub/mul coverage
- benchmarkable through the checked-in ASTER microbenchmark harness above
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
- ASTER: use `aster_ref` for inspection until the standalone runtime environment is more stable
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
