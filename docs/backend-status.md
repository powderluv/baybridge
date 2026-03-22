# Backend Status

This document is the current backend inventory for Baybridge, including execution coverage and the small reproducible benchmark set used to compare the working executable backends.

## Scope

The status below separates three things:
- backend surface area
- validated execution coverage
- what was benchmarkable in the current repo-local remote environments on `mi355` and `mi300`, plus the local NVIDIA host for PTX

That split matters because some backends are integrated and tested, but still depend on extra runtime environment setup for ad hoc shell-driven benchmarking.

## Validation Baseline

Current repo state during this documentation pass:
- branch: `main`
- worktree: dirty
- full local suite:
  - `507 passed, 196 skipped`
- focused local PTX/CUDA validation:
  - `tests/test_backend_ptx_ref.py tests/test_backend_ptx_exec.py tests/test_cuda_driver_runtime.py tests/test_backend_benchmark_tools.py`
  - result: `103 passed`

## Backend Test Inventory

This is the checked-in backend-oriented test inventory, not the full project-wide suite:

| Test file | Backend focus | Test count |
| --- | --- | ---: |
| `tests/test_backend_hipcc_exec.py` | `hipcc_exec` lowering and AMD execution | `14` |
| `tests/test_backend_hipkittens_ref.py` | `hipkittens_ref` family matching and lowering | `13` |
| `tests/test_backend_hipkittens_exec.py` | `hipkittens_exec` lowering, auto-selection, AMD execution | `20` |
| `tests/test_backend_flydsl_ref.py` | `flydsl_ref` lowering | `4` |
| `tests/test_backend_flydsl_exec.py` | `flydsl_exec` lowering, auto-selection, fake/runtime execution, real-FlyDSL opt-in execution | `125` |
| `tests/test_backend_ptx_ref.py` | `ptx_ref` PTX lowering, fallback selection, and driver-JIT-loadable module text | `45` |
| `tests/test_backend_ptx_exec.py` | `ptx_exec` lowering, NVIDIA execution, CUDA tensor-handle execution, and auto-selection | `44` |
| `tests/test_backend_waveasm_ref.py` | `gpu_mlir`, `waveasm_ref`, repro bundle tools, backend compare tooling | `16` |
| `tests/test_backend_waveasm_exec.py` | `waveasm_exec` experimental lowering and fake-toolchain execution | `8` |
| `tests/test_backend_aster_ref.py` | `aster_ref` lowering and tool discovery | `3` |
| `tests/test_backend_aster_exec.py` | `aster_exec` lowering, auto-selection, AMD execution, MFMA, float8, broadcast/tail coverage | `144` |
| `tests/test_backend_benchmark_tools.py` | benchmark sample factories and timing helpers | `10` |
| `tests/test_cuda_driver_runtime.py` | CUDA driver bootstrap used by the PTX backend | `4` |
| `tests/test_hip_runtime.py` | HIP runtime bootstrap used by executable backends | `2` |

## Backend Inventory

| Backend | Kind | Current role | Validated target(s) | Current practical coverage | Notes |
| --- | --- | --- | --- | --- | --- |
| `portable` | IR only | PortableKernelIR capture | local | all traced kernels | No lowering or execution |
| `mlir_text` | text/ref | textual inspection | local | broad traced subset | Baybridge-specific textual IR |
| `gpu_text` | text/ref | textual GPU-flavored inspection | local | broad traced subset | Useful for debug only |
| `gpu_mlir` | text/ref | structured MLIR emission | local | broad traced subset | Base for external MLIR backends |
| `ptx_ref` | ref | Baybridge-owned PTX text lowering | local NVIDIA host | exact rank-1 dense copy, pointwise `add/sub/mul/div`, scalar-broadcast, narrow unary `sqrt/rsqrt`, serial scalar reductions, exact single-block parallel scalar reductions, exact 2D serial reduction bundles, exact 2D parallel reduction bundles, and exact 2D tensor-factory bundles for `f32`/`i32` on canonical indexed or direct `threadIdx.x` forms | Emits PTX text that the CUDA driver JIT can load directly; no toolkit is required in the backend path |
| `ptx_exec` | exec | Baybridge-owned NVIDIA executable path | local NVIDIA host | same exact PTX subset as `ptx_ref`; works with Baybridge `RuntimeTensor` staging, CUDA `TensorHandle` inputs, and scalar kernel params | Uses only `libcuda.so.1`; auto-preferred for `NvidiaTarget` when available and the traced kernel matches the exact PTX subset |
| `hipcc_exec` | exec | default general AMD executable path | `gfx950`, `gfx942` | broad traced kernel subset including copy, pointwise, broadcast, reductions, shared memory, many tensor helpers | Primary executable backend today |
| `hipkittens_ref` | ref | reference/source backend for HipKittens families | local, `gfx950`, `gfx942` | GEMM, attention-family, norm-family matching | Not executable |
| `hipkittens_exec` | exec | narrow AMD-native GEMM backend | `gfx950`, `gfx942` | BF16/F16 GEMM on supported tile families, including validated BF16 transpose families | Opt-in or auto-selected only for matching GEMM kernels; transposed F16 and RMSNorm stay on `hipkittens_ref` on `gfx950` because current upstream HipKittens templates/headers reject those families |
| `flydsl_ref` | ref | reference FlyDSL lowering | local, `gfx950`, `gfx942` | elementwise, reductions, tiled/layout, MFMA-oriented family matching | Not executable |
| `flydsl_exec` | exec | narrow real FlyDSL execution path | validated subset on `gfx950`, `gfx942` | real validated subset: 1D `f32` copy; 1D `f32` pointwise `add/sub/mul/div`; canonical linear indexed 1D `f32` pointwise binary `add/sub/mul/div` of the form `block_idx.x * block_dim.x + thread_idx.x`; 1D `f32` unary math bundles for `exp/log/cos/erf`, `exp2/log2/log10/sqrt`, `sin`, and `rsqrt`; 2D `f32` broadcast binary `add/sub/mul/div` through the specialized row-slice/copy-atom path when `grid == block == (1, 1, 1)`; 2D `f32` reduction bundles for `add` and `mul` through the specialized row-slice/copy-atom path when `grid == block == (1, 1, 1)`; 2D `f32` unary math bundle for `exp/log/cos/erf` through the specialized row-slice/copy-atom path when `grid == block == (1, 1, 1)`; 2D `f32` tensor-factory bundle through the specialized row-slice/copy-atom path when `grid == block == (1, 1, 1)`; 1D `f32` shared-stage copy when the traced kernel is exactly a shared-memory round-trip and `block.x == extent` | Requires real FlyDSL env and a GPU-capable `torch` in the active venv for easy DLPack benchmarking; `acos`, `asin`, `atan`, and `atan2` are still outside the validated real subset |
| `waveasm_ref` | ref | WaveASM-oriented MLIR and repro bundle emission | local | supported GPU-MLIR subset | Emits `.waveasm_repro` bundles |
| `waveasm_exec` | exec, experimental | WaveASM HSACO build + HIP module launch | experimental on `gfx950`, `gfx942` | narrow single-buffer pointwise/shared-memory/math subset only | Gated by `BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC=1`; upstream correctness issue still blocks real support |
| `aster_ref` | ref | ASTER-oriented MLIR and repro bundle emission | local, `gfx950`, `gfx942` | ASTER reference lowering for supported families | Emits `.aster_repro` bundles |
| `aster_exec` | exec | narrow ASTER executable backend | validated on `gfx950`, `gfx942` | dense contiguous copy: `f32/i32/f16`; dense contiguous binary: `f32/i32 add/sub/mul`; scalar broadcast from dense single-element tensors for supported binary ops; exact MFMA GEMM and fragment-copyout families for `f16/f16 -> f32` and `bf16/bf16 -> f32` on `16x16x16`; exact direct `fp8/fp8 -> f32`, `bf8/bf8 -> f32`, `fp8/bf8 -> f32`, and `bf8/fp8 -> f32` MFMA GEMM on `gfx942` for `16x16x32`; 1D and 2D dense tensors | `div` is intentionally not supported; `fp8`/`bf8` are storage-only in Baybridge today; MFMA support is intentionally exact-shape and exact-descriptor only |

## Execution Coverage Matrix

| Family | `hipcc_exec` | `hipkittens_exec` | `flydsl_exec` | `aster_exec` | `waveasm_exec` |
| --- | --- | --- | --- | --- | --- |
| Dense contiguous `f32` copy | Yes | No | Real validated 1D only | Yes | Experimental only |
| Dense contiguous `i32` copy | Yes | No | No | Yes | Experimental only |
| Dense contiguous `f16` copy | Yes | No | No | Yes | Experimental only |
| Dense contiguous `f32` binary `add/sub/mul/div` | Yes | No | Real validated 1D only; canonical linear indexed `add/sub/mul/div` are also validated | Add/sub/mul only | Experimental only |
| Dense contiguous `f32` unary math bundle | Yes | No | Real validated 1D only for `exp/log/cos/erf`, `exp2/log2/log10/sqrt`, `sin`, and `rsqrt`; `acos/asin/atan/atan2` still unvalidated | No | Experimental only |
| Dense contiguous `i32` binary `add/sub/mul/div` | Yes | No | No | Add/sub/mul only | No |
| Scalar broadcasted binary on dense tensors | Yes | No | No | Yes | No |
| 2D `f32` tensor broadcast binary `add/sub/mul/div` | Yes | No | Real validated when `grid == block == (1, 1, 1)` | Yes for scalar-broadcast forms only | No |
| 2D `f32` tensor reduction bundle | Yes | No | Real validated for `add` and `mul` when `grid == block == (1, 1, 1)` | No | No |
| 2D `f32` unary math bundle | Yes | No | Real validated for `exp/log/cos/erf` when `grid == block == (1, 1, 1)` | No | No |
| 2D `f32` tensor-factory bundle | Yes | No | Real validated when `grid == block == (1, 1, 1)` | No | No |
| Tensor reductions | Yes | No | Executable in Baybridge-side lowering; real upstream validation is still narrower outside the validated 2D `f32` bundle | No | No |
| Shared-memory staging | Yes | No | Exact 1D `f32` shared-stage copy is validated when `block.x == extent`; broader real upstream shared-memory validation is still incomplete | No | Experimental only |
| GEMM | No dedicated path | Yes, narrow validated subset | No | Yes, exact `16x16x16` MFMA subset plus direct `fp8`/`bf8` exact and mixed `16x16x32` on `gfx942` only | No |
| Attention / norm families | No dedicated path | ref only | ref only | ref only | ref only |

## NVIDIA PTX Exact Coverage

The PTX path is intentionally separate from the AMD-focused execution matrix above.

- `ptx_ref` and `ptx_exec` are currently validated for:
  - canonical indexed rank-1 dense copy
  - canonical indexed rank-1 dense pointwise `add/sub/mul/div`
  - canonical indexed rank-1 unary `sqrt/rsqrt`
  - canonical indexed rank-1 scalar broadcast from:
    - a scalar kernel parameter
    - a rank-1 extent-1 tensor
  - direct `threadIdx.x` rank-1 dense copy
  - direct `threadIdx.x` rank-1 dense pointwise `add/sub/mul/div`
  - direct `threadIdx.x` rank-1 unary `sqrt/rsqrt`
  - direct `threadIdx.x` scalar broadcast from:
    - a scalar kernel parameter
    - a rank-1 extent-1 tensor
  - serial rank-1 scalar reductions to `dst[0]`:
    - `reduce_add`
    - `reduce_mul`
    - `reduce_max`
    - `reduce_min`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact parallel rank-1 scalar reduction to `dst[0]`:
    - supported ops:
      - `reduce_add`
      - `reduce_mul`
      - `reduce_max`
      - `reduce_min`
    - current launch contract: `grid=(1,1,1)`, `block=(power_of_two,1,1)`
    - current lowering uses a single-block shared-memory reduction
  - exact 2D `f32/i32` reduction bundle:
    - scalar reduction to `dst_scalar[0]`
    - row reduction to `dst_rows`
    - column reduction to `dst_cols`
    - supported ops:
      - `reduce_add`
      - `reduce_mul`
      - `reduce_max`
      - `reduce_min`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact parallel 2D `f32/i32` reduction bundle:
    - scalar reduction to `dst_scalar[0]`
    - row reduction to `dst_rows`
    - column reduction to `dst_cols`
    - supported ops:
      - `reduce_add`
      - `reduce_mul`
      - `reduce_max`
      - `reduce_min`
    - exact current launch contract: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= max(rows, cols)`
    - current lowering uses a single-block shared-memory scalar reduction plus per-thread row/column accumulation
  - exact 2D `f32/i32` tensor-factory bundle:
    - `dst_zero.store(bb.zeros_like(dst_zero))`
    - `dst_one.store(bb.ones_like(dst_one))`
    - `dst_full.store(bb.full_like(dst_full, fill_value))`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
- current dtypes:
  - `f32`
  - `i32`
- runtime paths validated in `ptx_exec`:
  - Baybridge `RuntimeTensor` values through CUDA-driver host staging
  - CUDA `TensorHandle` values with real device pointers
- current runtime dependency:
  - `libcuda.so.1`
- current non-goals:
  - `nvcc`
  - `nvrtc`
  - `ptxas`
  - broader CUDA math/libdevice lowering
- unary PTX note:
  - `sqrt` uses native `sqrt.rn.f32`
  - `rsqrt` uses native `rsqrt.approx.f32`, so its validated runtime path is approximate by construction
- reduction PTX note:
  - current scalar reductions are serial correctness-first lowerings
  - except for the exact single-block reduction family, which now has a shared-memory parallel lowering
  - the current 2D reduction bundle is also serial and correctness-first
  - the exact 2D reduction bundle also has a narrow single-block parallel path now
  - for the 2D bundle, the direct CUDA-handle path benefits much more than the staged path because host copies still dominate staged timings
  - there is not yet a general parallel CUDA reduction path beyond those exact single-block families

## Benchmark Method

Checked-in harness:
- kernel/sample module: [`tools/backend_benchmark_kernels.py`](/home/nod/github/baybridge/tools/backend_benchmark_kernels.py)
- runner: [`tools/compare_backends.py`](/home/nod/github/baybridge/tools/compare_backends.py)

Benchmark notes:
- timings are wall-clock `artifact(*args)` times in milliseconds
- compile is done once before measurement
- each run uses `--repeat 7`
- `tools/compare_backends.py` now emits `cold_ms`, `warm_timings_ms`, and `warm_median_ms` directly
- the first execution is usually a clear cold-start outlier
- the tables below report the median of the six warm runs after dropping the first cold-start outlier
- pointwise benchmark size: `65536` elements
- PTX snapshot target: `sm_80` PTX JIT on the local NVIDIA host
- FlyDSL specialized 1D microbenchmark size: `4096` elements
- FlyDSL specialized 2D microbenchmark shape: `64x64`
- FlyDSL specialized shared-stage microbenchmark size: `256` elements
- GEMM benchmark shape: `32x16 * 16x32 -> 32x32`
- ASTER microbenchmark size: `4096` elements
- ASTER MFMA benchmark shapes:
  - `16x16 * 16x16 -> 16x16`
  - `16x32 * 32x16 -> 16x16` for the `gfx942`-only `fp8`/`bf8` paths

## Performance Snapshot

### `mi355` (`gfx950`)

| Family | Backend | Median ms | Notes |
| --- | --- | ---: | --- |
| Dense `f32` copy, `65536` elements | `hipcc_exec` | `29.03` | Runtime benchmark completed cleanly |
| Dense `f32` copy, `65536` elements | `flydsl_exec` | `907.73` | Real upstream FlyDSL copy path measured with ROCm torch-backed inputs |
| Indexed `f32` add, `65536` elements | `hipcc_exec` | `33.52` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` add, `65536` elements | `flydsl_exec` | `890.75` | Real upstream FlyDSL canonical linear indexed-add path measured with ROCm torch-backed inputs |
| Indexed `f32` sub, `65536` elements | `hipcc_exec` | `33.11` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` sub, `65536` elements | `flydsl_exec` | `919.10` | Real upstream FlyDSL canonical linear indexed-sub path measured with ROCm torch-backed inputs |
| Indexed `f32` mul, `65536` elements | `hipcc_exec` | `32.85` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` mul, `65536` elements | `flydsl_exec` | `962.28` | Real upstream FlyDSL canonical linear indexed-mul path measured with ROCm torch-backed inputs |
| Indexed `f32` div, `65536` elements | `hipcc_exec` | `32.56` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` div, `65536` elements | `flydsl_exec` | `913.03` | Real upstream FlyDSL canonical linear indexed-div path measured with ROCm torch-backed inputs |
| BF16 GEMM `32x16 * 16x32 -> 32x32` | `hipkittens_exec` | `0.84` | Narrow supported microkernel family |
| Dense `f32` copy, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |
| Dense `f32` add, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |

### `mi300` (`gfx942`)

| Family | Backend | Median ms | Notes |
| --- | --- | ---: | --- |
| Dense `f32` copy, `65536` elements | `hipcc_exec` | `46.91` | Runtime benchmark completed cleanly |
| Dense `f32` copy, `65536` elements | `flydsl_exec` | `1370.14` | Real upstream FlyDSL copy path measured with ROCm torch-backed inputs |
| Indexed `f32` add, `65536` elements | `hipcc_exec` | `58.01` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` add, `65536` elements | `flydsl_exec` | `1377.69` | Real upstream FlyDSL canonical linear indexed-add path measured with ROCm torch-backed inputs |
| Indexed `f32` sub, `65536` elements | `hipcc_exec` | `59.77` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` sub, `65536` elements | `flydsl_exec` | `1365.25` | Real upstream FlyDSL canonical linear indexed-sub path measured with ROCm torch-backed inputs |
| Indexed `f32` mul, `65536` elements | `hipcc_exec` | `58.86` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` mul, `65536` elements | `flydsl_exec` | `1369.68` | Real upstream FlyDSL canonical linear indexed-mul path measured with ROCm torch-backed inputs |
| Indexed `f32` div, `65536` elements | `hipcc_exec` | `58.80` | Used the FlyDSL-compatible indexed kernel form |
| Indexed `f32` div, `65536` elements | `flydsl_exec` | `1368.64` | Real upstream FlyDSL canonical linear indexed-div path measured with ROCm torch-backed inputs |
| BF16 GEMM `32x16 * 16x32 -> 32x32` | `hipkittens_exec` | `1.46` | Narrow supported microkernel family |
| Dense `f32` copy, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |
| Dense `f32` add, `65536` elements | `aster_exec` | n/a | ASTER is measured separately on a `4096`-element microbenchmark because that is the validated checked-in harness path today |

## NVIDIA PTX Snapshot

These timings were captured on the local NVIDIA host:
- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- driver path: direct `libcuda.so.1`
- target passed to Baybridge: `sm_80`
- method: checked-in [`tools/compare_backends.py`](/home/nod/github/baybridge/tools/compare_backends.py) with `--execute --repeat 7`
- note:
  - rows without `CUDA handles` use Baybridge `RuntimeTensor` staging
  - rows with `CUDA handles` use `compile_args` as `RuntimeTensor` values but `run_args` as direct CUDA `TensorHandle` values, so they isolate the PTX launch/device path without host copies

| Family | Backend | Cold ms | Warm median ms | Notes |
| --- | --- | ---: | ---: | --- |
| Indexed `f32` copy, `65536` elements | `ptx_exec` | `241.82` | `25.17` | Canonical indexed PTX copy path through driver JIT |
| Indexed `f32` add, `65536` elements | `ptx_exec` | `219.18` | `37.18` | Canonical indexed PTX pointwise add path through driver JIT |
| Indexed `i32` copy, `65536` elements | `ptx_exec` | `259.77` | `32.39` | Canonical indexed PTX copy path through driver JIT |
| Indexed `i32` add, `65536` elements | `ptx_exec` | `167.98` | `48.84` | Canonical indexed PTX pointwise add path through driver JIT |
| Indexed `f32` sqrt, `65536` elements | `ptx_exec` | `343.01` | `25.44` | Canonical indexed PTX unary sqrt path through driver JIT |
| Indexed `f32` rsqrt, `65536` elements | `ptx_exec` | `250.40` | `25.33` | Canonical indexed PTX unary rsqrt path through driver JIT |
| Indexed `f32` scalar-broadcast add, `65536` elements | `ptx_exec` | `273.45` | `25.17` | Canonical indexed PTX scalar-parameter broadcast path through driver JIT |
| Indexed `i32` scalar-broadcast add, `65536` elements | `ptx_exec` | `155.19` | `33.69` | Canonical indexed PTX scalar-parameter broadcast path through driver JIT |
| Scalar `f32` reduction add, `65536` elements | `ptx_exec` | `240.74` | `13.22` | Serial exact PTX scalar-reduction path through driver JIT |
| Parallel scalar `f32` reduction add, `65536` elements | `ptx_exec` | `283.92` | `13.01` | Exact single-block shared-memory PTX reduction-add path through driver JIT |
| Parallel scalar `i32` reduction add, `65536` elements | `ptx_exec` | `278.42` | `16.35` | Exact single-block shared-memory PTX reduction-add path through driver JIT |
| 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `258.23` | `1.06` | Serial exact PTX 2D reduction-add bundle through driver JIT |
| 2D `i32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `310.74` | `1.33` | Serial exact PTX 2D reduction-add bundle through driver JIT |
| Parallel 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `425.35` | `1.16` | Exact single-block shared-memory PTX reduction-add bundle through driver JIT |
| Parallel 2D `f32` reduction mul bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `419.50` | `0.96` | Exact single-block shared-memory PTX reduction-mul bundle through driver JIT |
| Parallel 2D `f32` reduction max bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `368.44` | `1.01` | Exact single-block shared-memory PTX reduction-max bundle through driver JIT |
| Parallel 2D `f32` reduction min bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `419.47` | `1.11` | Exact single-block shared-memory PTX reduction-min bundle through driver JIT |
| 2D `f32` tensor-factory bundle, `64x64` | `ptx_exec` | `436.49` | `7.10` | Exact PTX tensor-factory bundle through driver JIT |
| Parallel 2D `i32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `379.27` | `1.37` | Exact single-block shared-memory PTX reduction-add bundle through driver JIT |
| Parallel 2D `i32` reduction mul bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `377.81` | `1.37` | Exact single-block shared-memory PTX reduction-mul bundle through driver JIT |
| Parallel 2D `i32` reduction max bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `412.13` | `1.24` | Exact single-block shared-memory PTX reduction-max bundle through driver JIT |
| Parallel 2D `i32` reduction min bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `268.32` | `1.26` | Exact single-block shared-memory PTX reduction-min bundle through driver JIT |
| 2D `i32` tensor-factory bundle, `64x64` | `ptx_exec` | `411.07` | `5.07` | Exact PTX tensor-factory bundle through driver JIT |
| Indexed `f32` copy, `65536` elements, CUDA handles | `ptx_exec` | `0.34` | `0.02` | Same canonical PTX copy kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` add, `65536` elements, CUDA handles | `ptx_exec` | `0.30` | `0.02` | Same canonical PTX add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` copy, `65536` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same canonical PTX copy kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` add, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` sqrt, `65536` elements, CUDA handles | `ptx_exec` | `0.32` | `0.02` | Same canonical PTX unary sqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` rsqrt, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX unary rsqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` scalar-broadcast add, `65536` elements, CUDA handles | `ptx_exec` | `0.30` | `0.02` | Same canonical PTX scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` scalar-broadcast add, `65536` elements, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same canonical PTX scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Scalar `f32` reduction add, `65536` elements, CUDA handles | `ptx_exec` | `0.77` | `0.49` | Same serial exact PTX scalar-reduction kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel scalar `f32` reduction add, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block shared-memory PTX reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel scalar `i32` reduction add, `65536` elements, CUDA handles | `ptx_exec` | `0.24` | `0.03` | Same exact single-block shared-memory PTX reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `0.33` | `0.09` | Same serial exact PTX 2D reduction-add bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `26.13` | `0.10` | Same serial exact PTX 2D reduction-add bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `24.52` | `0.03` | Same exact single-block shared-memory PTX reduction-add bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` reduction mul bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `24.09` | `0.02` | Same exact single-block shared-memory PTX reduction-mul bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` reduction max bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `23.76` | `0.03` | Same exact single-block shared-memory PTX reduction-max bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` reduction min bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `23.37` | `0.03` | Same exact single-block shared-memory PTX reduction-min bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` tensor-factory bundle, `64x64`, CUDA handles | `ptx_exec` | `37.28` | `0.05` | Same exact PTX tensor-factory bundle, but executed with direct CUDA `TensorHandle` outputs to avoid host staging |
| Parallel 2D `i32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `25.29` | `0.13` | Same exact single-block shared-memory PTX reduction-add bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` reduction mul bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block shared-memory PTX reduction-mul bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` reduction max bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `0.21` | `0.02` | Same exact single-block shared-memory PTX reduction-max bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` reduction min bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `0.32` | `0.02` | Same exact single-block shared-memory PTX reduction-min bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` tensor-factory bundle, `64x64`, CUDA handles | `ptx_exec` | `27.93` | `0.05` | Same exact PTX tensor-factory bundle, but executed with direct CUDA `TensorHandle` outputs to avoid host staging |
| Direct `f32` sqrt, `128` elements | `ptx_exec` | `129.88` | `0.17` | Direct `threadIdx.x` PTX unary sqrt path |
| Direct `f32` sqrt, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary sqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `f32` rsqrt, `128` elements | `ptx_exec` | `308.19` | `0.19` | Direct `threadIdx.x` PTX unary rsqrt path; uses `rsqrt.approx.f32`, so accuracy is approximate |
| Direct `f32` rsqrt, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary rsqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; accuracy is approximate |

## FlyDSL Specialized Microbenchmark Snapshot

These are real checked-in-tool measurements for the current validated specialized `flydsl_exec` subset, using:
- kernel/sample module: [`tools/backend_benchmark_kernels.py`](/home/nod/github/baybridge/tools/backend_benchmark_kernels.py)
- runner: [`tools/compare_backends.py`](/home/nod/github/baybridge/tools/compare_backends.py)
- FlyDSL-specific sample factories:
  - `flydsl_unary_sin_f32_args`
  - `flydsl_unary_rsqrt_f32_args`
  - `flydsl_broadcast_add_2d_args`
  - `flydsl_reduce_add_2d_args`
  - `flydsl_unary_math_2d_args`

They are intentionally broken out from the common `65536`-element table above because the real validated specialized FlyDSL paths here are exact 1D `4096`-element or 2D `64x64` families, not Baybridge's broader default pointwise benchmark shape.

### `mi355` (`gfx950`)

| Family | Backend | Warm median ms | Notes |
| --- | --- | ---: | --- |
| Unary `f32` `sin`, `4096` elements | `hipcc_exec` | `3.05` | Same kernel/sample factory as FlyDSL baseline |
| Unary `f32` `sin`, `4096` elements | `flydsl_exec` | `910.98` | Real upstream FlyDSL specialized unary path |
| Unary `f32` `rsqrt`, `4096` elements | `hipcc_exec` | `1.78` | Same kernel/sample factory as FlyDSL baseline |
| Unary `f32` `rsqrt`, `4096` elements | `flydsl_exec` | `899.95` | Real upstream FlyDSL specialized unary path |
| 2D `f32` broadcast add, `64x64` | `hipcc_exec` | `0.94` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` broadcast add, `64x64` | `flydsl_exec` | `895.86` | Real upstream FlyDSL specialized row-slice/copy-atom path |
| 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,)` | `hipcc_exec` | `1.01` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,)` | `flydsl_exec` | `952.02` | Real upstream FlyDSL specialized row-slice/copy-atom path |
| 2D `f32` unary math bundle `exp/log/cos/erf`, `64x64` | `hipcc_exec` | `5.99` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` unary math bundle `exp/log/cos/erf`, `64x64` | `flydsl_exec` | `888.14` | Real upstream FlyDSL specialized row-slice/copy-atom path |
| 1D `f32` shared-stage copy, `256` elements | `hipcc_exec` | `0.18` | Same kernel/sample factory as FlyDSL baseline |
| 1D `f32` shared-stage copy, `256` elements | `flydsl_exec` | `890.66` | Real upstream FlyDSL specialized shared-stage copy path |
| 2D `f32` tensor-factory bundle, `64x64` | `hipcc_exec` | `2.58` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` tensor-factory bundle, `64x64` | `flydsl_exec` | `893.77` | Real upstream FlyDSL specialized row-slice/copy-atom path |

### `mi300` (`gfx942`)

| Family | Backend | Warm median ms | Notes |
| --- | --- | ---: | --- |
| Unary `f32` `sin`, `4096` elements | `hipcc_exec` | `4.29` | Same kernel/sample factory as FlyDSL baseline |
| Unary `f32` `sin`, `4096` elements | `flydsl_exec` | `1370.69` | Real upstream FlyDSL specialized unary path |
| Unary `f32` `rsqrt`, `4096` elements | `hipcc_exec` | `2.92` | Same kernel/sample factory as FlyDSL baseline |
| Unary `f32` `rsqrt`, `4096` elements | `flydsl_exec` | `1385.15` | Real upstream FlyDSL specialized unary path |
| 2D `f32` broadcast add, `64x64` | `hipcc_exec` | `1.47` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` broadcast add, `64x64` | `flydsl_exec` | `1374.75` | Real upstream FlyDSL specialized row-slice/copy-atom path |
| 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,)` | `hipcc_exec` | `1.55` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` reduction add bundle, `64x64 -> (1,) + (64,)` | `flydsl_exec` | `1375.22` | Real upstream FlyDSL specialized row-slice/copy-atom path |
| 2D `f32` unary math bundle `exp/log/cos/erf`, `64x64` | `hipcc_exec` | `8.91` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` unary math bundle `exp/log/cos/erf`, `64x64` | `flydsl_exec` | `1374.18` | Real upstream FlyDSL specialized row-slice/copy-atom path |
| 1D `f32` shared-stage copy, `256` elements | `hipcc_exec` | `0.26` | Same kernel/sample factory as FlyDSL baseline |
| 1D `f32` shared-stage copy, `256` elements | `flydsl_exec` | `1374.15` | Real upstream FlyDSL specialized shared-stage copy path |
| 2D `f32` tensor-factory bundle, `64x64` | `hipcc_exec` | `4.22` | Same kernel/sample factory as FlyDSL baseline |
| 2D `f32` tensor-factory bundle, `64x64` | `flydsl_exec` | `1366.74` | Real upstream FlyDSL specialized row-slice/copy-atom path |

## FlyDSL Cold vs Warm Snapshot

Cold-start timing here is the first recorded execution in the `--repeat 7` run. Warm median is the median of the following six steady-state executions.

These representative reruns were captured after Baybridge-side DLPack adaptation caching and default-stream reuse were in place. The important result is that both cold and warm timings stay tightly clustered across very different validated FlyDSL kernels, which points to a dominant fixed cost inside the real FlyDSL runtime/JIT path rather than in Baybridge's wrapper glue.

| Family | `mi355` cold ms | `mi355` warm median ms | `mi300` cold ms | `mi300` warm median ms |
| --- | ---: | ---: | ---: | ---: |
| Indexed `f32` add, `65536` elements | `1150.55` | `910.87` | `1850.27` | `1381.13` |
| Unary `f32` `sin`, `4096` elements | `1152.22` | `907.18` | `1849.00` | `1372.85` |
| 2D `f32` broadcast add, `64x64` | `1147.64` | `910.80` | `1840.72` | `1373.30` |

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
  - `aster_mfma_fp8_gemm_args`
  - `aster_mfma_bf8_gemm_args`
  - `aster_mfma_fp8_bf8_gemm_args`
  - `aster_mfma_bf8_fp8_gemm_args`

This is intentionally separated from the HipKittens GEMM row in the main table. The currently validated checked-in executable shapes do not overlap cleanly:
- `hipkittens_exec` is validated on its own tile families
- `aster_exec` is validated on exact `16x16x16` MFMA GEMM and equivalent fragment-copyout forms, plus direct `gfx942`-only `fp8`/`bf8` exact and mixed `16x16x32` paths

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
| MFMA GEMM `fp8/fp8 -> f32`, `16x32 * 32x16 -> 16x16` | `aster_exec` | `2.52` | Exact `gfx942`-only ASTER MFMA path; benchmark harness uses `bb.pack_fp8(...)` |
| MFMA GEMM `bf8/bf8 -> f32`, `16x32 * 32x16 -> 16x16` | `aster_exec` | `2.57` | Exact `gfx942`-only ASTER MFMA path; benchmark harness uses `bb.pack_bf8(...)` |
| MFMA GEMM `fp8/bf8 -> f32`, `16x32 * 32x16 -> 16x16` | `aster_exec` | `2.51` | Exact `gfx942`-only ASTER MFMA path; benchmark harness uses `bb.pack_fp8(...)` and `bb.pack_bf8(...)` |
| MFMA GEMM `bf8/fp8 -> f32`, `16x32 * 32x16 -> 16x16` | `aster_exec` | `2.53` | Exact `gfx942`-only ASTER MFMA path; benchmark harness uses `bb.pack_bf8(...)` and `bb.pack_fp8(...)` |

## ASTER MFMA Cold vs Warm Snapshot

Cold-start timing here is the first recorded execution in the `--repeat 7` run. Warm median is the median of the following six steady-state executions.

| Family | `mi355` cold ms | `mi355` warm median ms | `mi300` cold ms | `mi300` warm median ms |
| --- | ---: | ---: | ---: | ---: |
| MFMA GEMM `f16/f16 -> f32`, `16x16x16` | `1211.51` | `3.63` | `295.70` | `2.47` |
| MFMA GEMM `bf16/bf16 -> f32`, `16x16x16` | `292.94` | `3.08` | `296.97` | `2.45` |
| MFMA GEMM `fp8/fp8 -> f32`, `16x16x32` | `n/a` | `n/a` | `262.45` | `2.52` |
| MFMA GEMM `bf8/bf8 -> f32`, `16x16x32` | `n/a` | `n/a` | `263.48` | `2.57` |
| MFMA GEMM `fp8/bf8 -> f32`, `16x16x32` | `n/a` | `n/a` | `263.61` | `2.51` |
| MFMA GEMM `bf8/fp8 -> f32`, `16x16x32` | `n/a` | `n/a` | `261.90` | `2.53` |

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

That is enough for the benchmark harness to measure the validated real `flydsl_exec` copy path and the current checked-in specialized FlyDSL microbench families.

The remaining boundary is semantic, not environmental:
- dense `f32` copy is benchmarkable
- specialized 1D unary, shared-stage, and 2D broadcast/reduction/unary/tensor-factory families are benchmarkable through the checked-in exact kernels above
- the canonical linear indexed `add/sub/mul/div` benchmarks are now benchmarkable through the validated real path
- broader indexed families are still narrower than the general `hipcc_exec` path

### ASTER

`aster_exec` is integrated and the current focused pytest path is healthy again on both `gfx950` and `gfx942` for:
- dense contiguous copy: `f32/i32/f16`
- dense contiguous pointwise binary: `f32/i32 add/sub/mul`
- dense scalar-broadcast binary for the same supported ops
- exact MFMA GEMM and fragment-copyout forms for:
  - `f16/f16 -> f32`, `16x16x16`
  - `bf16/bf16 -> f32`, `16x16x16`
- exact direct MFMA GEMM for:
  - `fp8/fp8 -> f32`, `16x16x32`, `gfx942` only
  - `bf8/bf8 -> f32`, `16x16x32`, `gfx942` only
  - `fp8/bf8 -> f32`, `16x16x32`, `gfx942` only
  - `bf8/fp8 -> f32`, `16x16x32`, `gfx942` only

Two important boundaries remain:
- `div` is intentionally unsupported in `aster_exec` because ASTER's current pass pipeline rejects the LSIR divide path in Baybridge's kernel form
- `fp8`/`bf8` in Baybridge are storage-only today:
  - `bb.pack_fp8(...)` and `bb.pack_bf8(...)` now provide a Baybridge-native input path from normal float data
  - raw E4M3FNUZ and E5M2FNUZ byte payload tensors still work when explicit control is needed
  - generic runtime arithmetic on `fp8`/`bf8` is intentionally rejected
  - only the exact ASTER MFMA paths consume them today
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
- useful for executable `gfx942` exact and mixed FP8/BF8 MFMA through the new pack helpers or raw byte-payload tensors
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

Cold/warm FlyDSL diagnostic example on `mi355`:

```bash
cd ~/tmp/baybridge-codex
PATH=$PWD/.venv/bin:$PATH \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
LD_LIBRARY_PATH=$HOME/tmp/FlyDSL/build-fly-fullmlir/python_packages/flydsl/_mlir/_mlir_libs:$HOME/tmp/llvm-project/mlir_install/lib:${LD_LIBRARY_PATH:-} \
BAYBRIDGE_FLYDSL_ROOT=$HOME/tmp/FlyDSL \
.venv/bin/python tools/report_cold_warm.py \
  tools/backend_benchmark_kernels.py indexed_add_f32_kernel \
  --sample-factory indexed_add_f32_args \
  --backends hipcc_exec,flydsl_exec \
  --baseline-backend hipcc_exec \
  --target gfx950 \
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
