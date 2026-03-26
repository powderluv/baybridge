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
  - `1292 passed, 196 skipped`
- focused local PTX/CUDA/target validation:
  - `tests/test_backend_ptx_ref.py tests/test_backend_ptx_exec.py tests/test_cuda_driver_runtime.py tests/test_backend_benchmark_tools.py tests/test_target.py`
  - result: `885 passed`

## Backend Test Inventory

This is the checked-in backend-oriented test inventory, not the full project-wide suite:

| Test file | Backend focus | Test count |
| --- | --- | ---: |
| `tests/test_backend_hipcc_exec.py` | `hipcc_exec` lowering and AMD execution | `14` |
| `tests/test_backend_hipkittens_ref.py` | `hipkittens_ref` family matching and lowering | `13` |
| `tests/test_backend_hipkittens_exec.py` | `hipkittens_exec` lowering, auto-selection, AMD execution | `20` |
| `tests/test_backend_flydsl_ref.py` | `flydsl_ref` lowering | `4` |
| `tests/test_backend_flydsl_exec.py` | `flydsl_exec` lowering, auto-selection, fake/runtime execution, real-FlyDSL opt-in execution | `125` |
| `tests/test_backend_ptx_ref.py` | `ptx_ref` PTX lowering, fallback selection, launch-contract validation, and driver-JIT-loadable module text | `417` |
| `tests/test_backend_ptx_exec.py` | `ptx_exec` lowering, NVIDIA execution, CUDA tensor-handle execution, raw CUDA DLPack execution, stream forwarding, and auto-selection | `426` |
| `tests/test_backend_waveasm_ref.py` | `gpu_mlir`, `waveasm_ref`, repro bundle tools, backend compare tooling | `16` |
| `tests/test_backend_waveasm_exec.py` | `waveasm_exec` experimental lowering and fake-toolchain execution | `8` |
| `tests/test_backend_aster_ref.py` | `aster_ref` lowering and tool discovery | `3` |
| `tests/test_backend_aster_exec.py` | `aster_exec` lowering, auto-selection, AMD execution, MFMA, float8, broadcast/tail coverage | `144` |
| `tests/test_backend_benchmark_tools.py` | benchmark sample factories, target parsing, and timing helpers | `33` |
| `tests/test_cuda_driver_runtime.py` | CUDA driver bootstrap used by the PTX backend | `5` |
| `tests/test_target.py` | target normalization, including `NvidiaTarget` SM canonicalization | `4` |
| `tests/test_hip_runtime.py` | HIP runtime bootstrap used by executable backends | `2` |

## Backend Inventory

| Backend | Kind | Current role | Validated target(s) | Current practical coverage | Notes |
| --- | --- | --- | --- | --- | --- |
| `portable` | IR only | PortableKernelIR capture | local | all traced kernels | No lowering or execution |
| `mlir_text` | text/ref | textual inspection | local | broad traced subset | Baybridge-specific textual IR |
| `gpu_text` | text/ref | textual GPU-flavored inspection | local | broad traced subset | Useful for debug only |
| `gpu_mlir` | text/ref | structured MLIR emission | local | broad traced subset | Base for external MLIR backends |
| `ptx_ref` | ref | Baybridge-owned PTX text lowering | local NVIDIA host | exact rank-1 dense copy for `f32/i32/i1`, compare-to-`i1`, scalar-broadcast compare-to-`i1`, rank-1 `select` from `i1` predicate tensors including scalar-branch forms from scalar kernel params and rank-1 extent-1 tensors, pointwise `f32/i32 add/sub/mul/div/max/min` plus rank-1 `i32 bitand/bitor/bitxor` and rank-1 `i1 and/or/xor`, exact rank-1 unary `neg/abs` for `f32/i32`, exact rank-1 PTX unary `round/floor/ceil/trunc/sqrt/rsqrt/sin/cos/acos/asin/atan/exp/exp2/log/log2/log10/erf` for `f32`, exact rank-1 integer `bitnot` for `i32/i1`, rank-1 scalar-broadcast including `f32/i32 max/min` and `i32 bitand/bitor/bitxor`, exact 2D unary `f32 abs/neg/round/floor/ceil/trunc/sqrt/rsqrt/sin/cos/acos/asin/atan/exp/exp2/log/log2/log10/erf`, exact 2D unary `i32 abs/neg/bitnot`, and exact 2D integer/boolean `i1 bitnot`, exact rank-1 `copy_reduce`, exact 2D dense copy bundles for `f32/i32/i1`, exact 2D `copy_reduce` bundles, exact 2D scalar-broadcast bundles from scalar kernel params, exact 2D scalar-broadcast compare-to-`i1` bundles from scalar kernel params, exact 2D extent-1 tensor-source scalar-broadcast bundles, exact 2D extent-1 tensor-source scalar-broadcast compare-to-`i1` bundles, exact 2D dense compare-to-`i1` bundles, exact 2D broadcast compare-to-`i1` bundles, exact 2D dense and broadcast `select` bundles from `i1` predicate tensors, exact 2D scalar-branch `select` bundles from scalar kernel params and rank-1 extent-1 tensors, exact 2D dense tensor-binary bundles including `f32/i32 max/min` and `i1 bitand/bitor/bitxor`, exact 2D broadcast tensor-binary bundles including `f32/i32 max/min` and `i1 bitand/bitor/bitxor`, exact 2D rowwise/columnwise reduction families including `i32 reduce_and/reduce_or/reduce_xor`, exact 2D row+column reduction bundles without the scalar output, exact parallel 2D row-tiled variants of those copy/copy-reduce/select/scalar-select/scalar-broadcast/scalar-broadcast compare-to-`i1`/compare-to-`i1`/tensor-binary/unary families plus the extent-1 tensor-source scalar-broadcast and scalar-broadcast compare-to-`i1` paths, exact multiblock output-tiled 2D rowwise/columnwise reductions, exact multiblock output-tiled 2D row+column reduction bundles without the scalar output, serial scalar reductions including `i32 reduce_or`, exact single-block parallel scalar reductions including `i32 reduce_and/reduce_or/reduce_xor`, exact 2D serial reduction bundles including `i32 reduce_and/reduce_or/reduce_xor`, exact 2D parallel reduction bundles including `i32 reduce_and/reduce_or/reduce_xor`, and exact 2D tensor-factory bundles plus their exact parallel row-tiled variants for `f32`/`i32` on canonical indexed or direct `threadIdx.x` forms, plus exact `f32` `atan2` across the validated rank-1, scalar-broadcast, dense/broadcast 2D, and row-tiled 2D PTX tensor-binary families | Emits PTX text that the CUDA driver JIT can load directly; no toolkit is required in the backend path |
| `ptx_exec` | exec | Baybridge-owned NVIDIA executable path | local NVIDIA host | same exact PTX subset as `ptx_ref`; works with Baybridge `RuntimeTensor` staging, CUDA `TensorHandle` inputs, raw CUDA DLPack-capable tensor inputs, scalar kernel params, validated `i1` tensor data paths for exact copy, rank-1 and 2D `f32` unary `abs/neg/round/floor/ceil/trunc/sqrt/rsqrt/sin/cos/acos/asin/atan/exp/exp2/log/log2/log10/erf`, rank-1 and 2D `i32` unary `abs/neg`, rank-1 and 2D integer/boolean `bitnot`, and bitwise tensor-binary/broadcast-binary PTX families, staged boolean `i1` tensor outputs for the exact rank-1, serial 2D, and exact parallel 2D row-tiled compare families including scalar-broadcast compare-to-`i1` from both scalar kernel params and rank-1 extent-1 tensors, and `i1` predicate-tensor inputs for the exact PTX `select` family across rank-1 direct/indexed forms plus serial and exact parallel row-tiled 2D dense, broadcast, and scalar-branch select forms, plus exact `f32` `atan2` across the validated rank-1, scalar-broadcast, dense/broadcast 2D, and row-tiled 2D PTX tensor-binary families | Uses only `libcuda.so.1`; auto-preferred for `NvidiaTarget` when available and the traced kernel matches the exact PTX subset with device-resident sample args; staged tensor sample args now keep reduction-style PTX kernels on `ptx_ref` by default, and all-staged tensor sample args keep every PTX family on `ptx_ref` unless `backend=\"ptx_exec\"` is requested explicitly; once selected, `ptx_exec` warns once per built launcher when tensor args are staged through host memory, reuses per-argument device allocations across repeated staged launches, and still measures best with direct CUDA handles or raw CUDA DLPack inputs rather than staged `RuntimeTensor` values |
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
  - canonical indexed rank-1 dense copy:
    - `f32`
    - `i32`
    - `i1`
  - canonical indexed rank-1 compare-to-`i1`:
    - `f32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - `i32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
  - canonical indexed rank-1 `select` from `i1` predicate tensors:
    - tensor-vs-tensor branch dtypes:
      - `f32`
      - `i32`
      - `i1`
    - scalar-branch forms from:
      - a scalar kernel parameter
      - a rank-1 extent-1 tensor
    - scalar-branch dtypes:
      - `f32`
      - `i32`
      - `i1`
  - canonical indexed rank-1 scalar-broadcast compare-to-`i1` from:
    - a scalar kernel parameter
    - a rank-1 extent-1 tensor
    - `f32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - `i32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
  - canonical indexed rank-1 dense pointwise:
    - `f32`: `add/sub/mul/div/max/min/atan2`
    - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
    - `i1`: `and`, `or`, `xor`
  - canonical indexed rank-1 unary `neg/abs`:
    - `f32`
    - `i32`
  - canonical indexed rank-1 integer `bitnot`:
    - `i32`
    - `i1`
  - canonical indexed rank-1 unary:
    - `f32`: `round`, `floor`, `ceil`, `trunc`, `sqrt`, `rsqrt`, `sin`, `cos`, `acos`, `asin`, `atan`, `exp`, `exp2`, `log`, `log2`, `log10`, `erf`
  - canonical indexed rank-1 scalar broadcast from:
    - a scalar kernel parameter
    - a rank-1 extent-1 tensor
    - supported ops:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
  - exact current launch contract for indexed rank-1 families:
    - `grid=(grid.x >= 1,1,1)`, `block=(block.x >= 1,1,1)`
    - `grid.x * block.x >= extent`
  - direct `threadIdx.x` rank-1 dense copy:
    - `f32`
    - `i32`
    - `i1`
  - direct `threadIdx.x` rank-1 integer `bitnot`:
    - `i32`
    - `i1`
  - direct `threadIdx.x` rank-1 compare-to-`i1`:
    - `f32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - `i32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
  - direct `threadIdx.x` rank-1 `select` from `i1` predicate tensors:
    - tensor-vs-tensor branch dtypes:
      - `f32`
      - `i32`
      - `i1`
    - scalar-branch forms from:
      - a scalar kernel parameter
      - a rank-1 extent-1 tensor
    - scalar-branch dtypes:
      - `f32`
      - `i32`
      - `i1`
  - direct `threadIdx.x` rank-1 scalar-broadcast compare-to-`i1` from:
    - a scalar kernel parameter
    - a rank-1 extent-1 tensor
    - `f32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - `i32`: `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
  - direct `threadIdx.x` rank-1 dense pointwise:
    - `f32`: `add/sub/mul/div/max/min/atan2`
    - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
    - `i1`: `and`, `or`, `xor`
  - direct `threadIdx.x` rank-1 unary:
    - `f32`: `abs`, `neg`, `round`, `floor`, `ceil`, `trunc`, `sqrt`, `rsqrt`, `sin`, `cos`, `acos`, `asin`, `atan`, `exp`, `exp2`, `log`, `log2`, `log10`, `erf`
    - `i32`: `abs`, `neg`
  - direct `threadIdx.x` scalar broadcast from:
    - a scalar kernel parameter
    - a rank-1 extent-1 tensor
    - supported ops:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
  - exact current launch contract for direct rank-1 families:
    - `grid=(1,1,1)`, `block=(block.x >= extent,1,1)`
  - exact rank-1 `copy_reduce`:
    - direct `bb.copy(copy_reduce_atom, src, dst)` form
    - supported reductions:
      - `add`
      - `max`
      - `min`
      - `and`
      - `or`
      - `xor`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - direct single-block: `grid=(1,1,1)`, `block=(block.x >= extent,1,1)`
      - indexed multi-block: `grid=(grid.x >= 1,1,1)`, `block=(block.x >= 1,1,1)`, `grid.x * block.x >= extent`
  - exact 2D `f32/i32/i1` dense copy bundles:
    - `bb.copy(src, dst)`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact 2D `copy_reduce` bundles:
    - `bb.copy(copy_reduce_atom, src, dst)`
    - supported reductions:
      - `f32`: `add`, `max`, `min`
      - `i32`: `add`, `max`, `min`, `and`, `or`, `xor`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
  - exact 2D `f32/i32` scalar-broadcast bundles from scalar kernel parameters:
    - `dst.store(src.load() <op> alpha)`
    - supported ops:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact 2D `f32/i32` scalar-broadcast compare-to-`i1` bundles from scalar kernel parameters:
    - `dst.store(src.load() <cmp> alpha)`
    - supported ops:
      - `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
  - exact 2D `f32/i32` extent-1 tensor-source scalar-broadcast bundles:
    - `dst.store(src.load() <op> alpha[0])`
    - supported ops:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
  - exact 2D `f32/i32` extent-1 tensor-source scalar-broadcast compare-to-`i1` bundles:
    - `dst.store(src.load() <cmp> alpha[0])`
    - supported ops:
      - `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
  - exact 2D `f32/i32` dense compare-to-`i1` bundles:
    - `dst.store(lhs.load() <cmp> rhs.load())`
    - supported ops:
      - `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
  - exact 2D `f32/i32` dense tensor-binary bundles:
    - `dst.store(lhs.load() <op> rhs.load())`
    - supported ops:
      - `f32`: `add`, `sub`, `mul`, `div`, `max`, `min`
      - `i32`: `add`, `sub`, `mul`, `div`, `max`, `min`, `bitand`, `bitor`, `bitxor`
      - `i1`: `bitand`, `bitor`, `bitxor`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact 2D `f32/i32` broadcast compare-to-`i1` bundles:
    - `dst.store(lhs.load() <cmp> rhs.load())`
    - broadcast-compatible `lhs` and `rhs` shapes expanded to the destination shape
    - supported ops:
      - `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
  - exact 2D `f32/i32` broadcast tensor-binary bundles:
    - `dst.store(lhs.load() <op> rhs.load())`
    - broadcast-compatible `lhs` and `rhs` shapes expanded to the destination shape
    - supported ops:
      - `f32`: `add`, `sub`, `mul`, `div`, `max`, `min`
      - `i32`: `add`, `sub`, `mul`, `div`, `max`, `min`, `bitand`, `bitor`, `bitxor`
      - `i1`: `bitand`, `bitor`, `bitxor`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact 2D unary bundles:
    - `f32`:
      - `dst.store(abs(src.load()))`
      - `dst.store(-src.load())`
      - `dst.store(bb.sqrt(src.load()))`
      - `dst.store(bb.rsqrt(src.load()))`
      - `dst.store(bb.sin(src.load()))`
      - `dst.store(bb.cos(src.load()))`
      - `dst.store(bb.acos(src.load()))`
      - `dst.store(bb.asin(src.load()))`
      - `dst.store(bb.atan(src.load()))`
      - `dst.store(bb.exp(src.load()))`
      - `dst.store(bb.exp2(src.load()))`
      - `dst.store(bb.log(src.load()))`
      - `dst.store(bb.log2(src.load()))`
      - `dst.store(bb.log10(src.load()))`
      - `dst.store(bb.erf(src.load()))`
    - `i32`:
      - `dst.store(abs(src.load()))`
      - `dst.store(-src.load())`
      - `dst.store(~src.load())`
    - `i1`:
      - `dst.store(~src.load())`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact 2D `f32/i32` rowwise or columnwise reduction families:
    - `dst_rows.store(src.load().reduce(bb.ReductionOp.<op>, init, reduction_profile=(None, 1)))`
    - `dst_cols.store(src.load().reduce(bb.ReductionOp.<op>, init, reduction_profile=(1, None)))`
    - supported ops:
      - `f32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`
      - `i32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`, `reduce_and`, `reduce_or`, `reduce_xor`
    - exact current launch contracts:
      - serial: `grid=(1,1,1)`, `block=(1,1,1)`
      - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`
    - current parallel lowering maps one thread to one output row or column and loops over the reduced axis inside the thread
  - exact parallel 2D row-tiled variants of the same tensor families above:
    - dense copy for `f32/i32/i1`
    - scalar-broadcast from scalar kernel parameters:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
    - scalar-broadcast compare-to-`i1` from scalar kernel parameters:
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
    - extent-1 tensor-source scalar-broadcast:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
    - extent-1 tensor-source scalar-broadcast compare-to-`i1`:
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
    - dense tensor-binary:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
      - `i1`: `bitand/bitor/bitxor`
    - dense compare-to-`i1`:
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
    - broadcast tensor-binary:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
      - `i1`: `bitand/bitor/bitxor`
    - broadcast compare-to-`i1`:
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
    - dense unary:
      - `f32`: `abs`, `neg`, `round/floor/ceil/trunc`, `sqrt/rsqrt`, `sin/cos/acos/asin/atan/exp/exp2/log/log2/log10/erf`
      - `i32`: `abs`, `neg`, `bitnot`
      - `i1`: `bitnot`
    - exact current launch contract:
      - `grid=(grid.x, grid.y, 1)`, `block=(power_of_two,1,1)`
      - `grid.x * block.x >= cols`
      - `grid.y >= 1`
    - current lowering maps each thread to a column tile and advances rows by `grid.y`
  - serial rank-1 scalar reductions to `dst[0]`:
    - `f32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`
    - `i32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`, `reduce_and`, `reduce_or`, `reduce_xor`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact parallel rank-1 scalar reduction to `dst[0]`:
    - supported ops:
      - `f32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`
      - `i32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`, `reduce_and`, `reduce_or`, `reduce_xor`
    - current launch contract: `grid=(1,1,1)`, `block=(power_of_two,1,1)`
    - current lowering uses a single-block shared-memory reduction
  - exact 2D `f32/i32` reduction bundle:
    - scalar reduction to `dst_scalar[0]`
    - row reduction to `dst_rows`
    - column reduction to `dst_cols`
    - supported ops:
      - `f32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`
      - `i32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`, `reduce_and`, `reduce_or`, `reduce_xor`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact parallel 2D `f32/i32` reduction bundle:
    - scalar reduction to `dst_scalar[0]`
    - row reduction to `dst_rows`
    - column reduction to `dst_cols`
    - supported ops:
      - `f32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`
      - `i32`: `reduce_add`, `reduce_mul`, `reduce_max`, `reduce_min`, `reduce_and`, `reduce_or`, `reduce_xor`
    - exact current launch contract: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= max(rows, cols)`
    - current lowering uses a single-block shared-memory scalar reduction plus per-thread row/column accumulation
  - exact 2D `f32/i32` tensor-factory bundle:
    - `dst_zero.store(bb.zeros_like(dst_zero))`
    - `dst_one.store(bb.ones_like(dst_one))`
    - `dst_full.store(bb.full_like(dst_full, fill_value))`
    - exact current launch contract: `grid=(1,1,1)`, `block=(1,1,1)`
  - exact parallel 2D row-tiled `f32/i32` tensor-factory bundle:
    - same `zeros_like` / `ones_like` / `full_like` bundle as above
    - exact current launch contract:
      - `grid=(grid.x, grid.y, 1)`, `block=(power_of_two,1,1)`
      - `grid.x * block.x >= cols`
      - `grid.y >= 1`
    - current lowering maps each thread to a column tile and advances rows by `grid.y`
- current dtypes:
  - `f32`
  - `i32`
  - `i1` for exact copy and bitwise tensor families only
  - `i1` output tensors for the exact compare family
- runtime paths validated in `ptx_exec`:
  - Baybridge `RuntimeTensor` values through CUDA-driver host staging
  - CUDA `TensorHandle` values with real device pointers
  - raw CUDA DLPack-capable tensor inputs, which normalize to CUDA tensor handles at launch time
  - `stream=` now propagates through DLPack normalization before backend launch, so CUDA producers that implement `__dlpack__(stream=...)` see the launch stream
  - when all tensor sample args are staged `RuntimeTensor` values, `compile(...)` now auto-selects `ptx_ref` and warns how to request staged `ptx_exec` explicitly
  - when sample args mix staged tensors with CUDA-resident tensors, `compile(...)` still auto-selects `ptx_exec` and warns about the staged tensor arguments for non-reduction PTX families
  - for PTX reduction-style kernels, mixed staged/device tensor sample args now stay on `ptx_ref` unless `backend=\"ptx_exec\"` is requested explicitly
  - `ptx_exec` warns once per built launcher when tensor arguments are staged through host memory
  - `ptx_exec` now reuses per-argument device allocations across repeated staged launches
  - the current performance-oriented PTX path is direct CUDA `TensorHandle` execution or raw CUDA DLPack inputs; staged `RuntimeTensor` execution remains correctness-first and host-copy dominated
  - representative raw CUDA DLPack timings currently track the direct `TensorHandle` path closely:
    - indexed `f32` add `65536`: cold `0.43 ms`, warm `0.03 ms`
    - indexed `i32` add `65536`: cold `0.48 ms`, warm `0.03 ms`
    - parallel 2D `f32` dense add `64x64`: cold `0.29 ms`, warm `0.04 ms`
    - parallel 2D `i32` dense add `64x64`: cold `0.33 ms`, warm `0.04 ms`
    - 2D `i32` scalar-parameter bitand broadcast `64x64`: cold `0.41 ms`, warm `0.15 ms`
    - parallel 2D `i32` scalar-parameter bitand broadcast `64x64`: cold `0.30 ms`, warm `0.04 ms`
    - 2D `f32` extent-1 tensor-source scalar-broadcast add `64x64`: cold `0.39 ms`, warm `0.16 ms`
    - parallel 2D `f32` extent-1 tensor-source scalar-broadcast add `64x64`: cold `0.28 ms`, warm `0.04 ms`
    - 2D `i32` extent-1 tensor-source scalar-broadcast add `64x64`: cold `0.42 ms`, warm `0.16 ms`
    - 2D `i32` extent-1 tensor-source scalar-broadcast bitor `64x64`: cold `0.41 ms`, warm `0.16 ms`
    - parallel 2D `i32` extent-1 tensor-source scalar-broadcast add `64x64`: cold `0.29 ms`, warm `0.04 ms`
    - parallel 2D `i32` extent-1 tensor-source scalar-broadcast bitor `64x64`: cold `0.29 ms`, warm `0.04 ms`
- current runtime dependency:
  - `libcuda.so.1`
- current non-goals:
  - `nvcc`
  - `nvrtc`
  - `ptxas`
  - broader CUDA math/libdevice lowering
  - unary PTX note:
  - driver-only unary math is the current PTX boundary:
    - `round` uses `cvt.rni.f32.f32`
    - `floor` uses `cvt.rmi.f32.f32`
    - `ceil` uses `cvt.rpi.f32.f32`
    - `trunc` uses `cvt.rzi.f32.f32`
    - `sqrt` uses `sqrt.rn.f32`
    - `rsqrt` uses `rsqrt.approx.f32`
    - `sin` uses `sin.approx.f32`
    - `cos` uses `cos.approx.f32`
    - `acos` uses `sqrt(max(0, 1 - x*x))` plus the same explicit `atan2` core with the square-root term as `y`
    - `asin` uses `sqrt(max(0, 1 - x*x))` plus the same explicit `atan2` core
    - `atan` uses an explicit reciprocal-plus-piecewise polynomial PTX approximation
    - `atan2` uses the same reciprocal-plus-piecewise polynomial core plus explicit quadrant correction
    - `exp2` uses `ex2.approx.f32`
    - `log2` uses `lg2.approx.f32`
    - `exp` uses `ex2.approx.f32` after scaling by `log2(e)`
    - `log` uses `lg2.approx.f32` times `ln(2)`
    - `log10` uses `lg2.approx.f32` times `log10(2)`
    - `erf` uses an explicit polynomial-plus-`ex2.approx.f32` approximation
  - the approximate instructions above are validated exactly as emitted, so their runtime path is intentionally approximate by construction
- reduction PTX note:
  - serial correctness-first reduction lowerings still exist for the exact scalar, 2D row/column, and 2D bundle traced forms
  - exact single-block shared-memory parallel lowerings now also exist for the scalar reduction family and the 2D reduction-bundle family
  - exact per-output-thread lowerings now also exist for the 2D rowwise/columnwise reduction family
  - that 2D row/column family now also has an exact multiblock output-tiling path when `grid.x * block.x >= output_extent`
  - exact row+column reduction bundles without the scalar output now also exist in serial, single-block parallel, and exact multiblock output-tiled forms
  - for the 2D bundle, the direct CUDA-handle path benefits much more than the staged path because host copies still dominate staged timings
  - there is not yet a general parallel CUDA scalar-or-bundle reduction path beyond those exact single-block families
- parallel 2D PTX note:
  - exact parallel row-tiled copy, scalar-broadcast, tensor-binary, broadcast-tensor-binary, unary, `copy_reduce`, and tensor-factory paths now exist for 2D kernels
  - the direct CUDA-handle path is where that parallelization matters; staged timings are still largely host-copy bound
  - representative staged remeasurements after per-argument allocation caching stayed close to the earlier warm medians:
    - indexed `f32` add `65536`: `38.10 ms`
    - parallel 2D `f32` dense add `64x64`: `2.43 ms`
  - representative multiblock row-tiled PTX timings are now included below for the exact `grid=(2,4,1)`, `block=(32,1,1)` path on `64x64` tensors
  - representative multiblock row/column reduction timings are also included below for the exact `grid=(2,1,1)`, `block=(32,1,1)` output-tiled path on `64x64` tensors

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
  - rows with `raw DLPack` use `compile_args` as `RuntimeTensor` values but `run_args` as raw CUDA DLPack-capable tensors, so they exercise Baybridge's stream-aware DLPack normalization path before the same device launch
  - `ptx_exec` emits a one-shot runtime warning per built launcher when tensor arguments are staged through host memory

### Representative Row-Tiled Multiblock PTX

These representative rows exercise the exact multiblock row-tiled PTX launch contract:
- `grid=(2,4,1)`
- `block=(32,1,1)`

| Family | Staged cold/warm ms | CUDA handles cold/warm ms | raw DLPack cold/warm ms |
| --- | ---: | ---: | ---: |
| Multiblock 2D `f32` dense copy, `64x64` | `272.86 / 1.61` | `0.25 / 0.02` | `0.26 / 0.03` |
| Multiblock 2D `f32` dense add, `64x64` | `134.16 / 2.39` | `0.24 / 0.02` | `0.29 / 0.04` |
| Multiblock 2D `f32` dense tensor-source scalar-broadcast add, `64x64` | `132.04 / 1.65` | `0.25 / 0.02` | `0.26 / 0.03` |
| Multiblock 2D `f32` `copy_reduce` add, `64x64` | `132.42 / 1.60` | `0.26 / 0.02` | `0.29 / 0.03` |
| Multiblock 2D `f32` tensor-factory bundle, `64x64` | `135.08 / 2.41` | `0.25 / 0.02` | `0.29 / 0.03` |
| Multiblock 2D `i32` dense tensor-source scalar-branch select, `64x64` | `140.73 / 3.20` | `0.31 / 0.03` | `0.33 / 0.04` |
| Multiblock 2D `f32` dense abs, `64x64` | `133.10 / 1.59` | `0.26 / 0.02` | `0.27 / 0.03` |
| Multiblock 2D `f32` dense round, `64x64` | `129.84 / 1.59` | `0.26 / 0.02` | `0.27 / 0.03` |
| Multiblock 2D `f32` dense ceil, `64x64` | `127.82 / 1.62` | `0.24 / 0.02` | `0.26 / 0.03` |
| Multiblock 2D `i32` dense abs, `64x64` | `132.95 / 2.16` | `0.24 / 0.02` | `0.26 / 0.03` |
| Multiblock 2D `i32` dense bitnot, `64x64` | `133.13 / 2.18` | `0.24 / 0.02` | `0.27 / 0.03` |
| Multiblock 2D `f32` dense exp2, `64x64` | `136.83 / 1.73` | `0.29 / 0.02` | `0.30 / 0.03` |
| Multiblock 2D `f32` dense asin, `64x64` | `124.94 / 1.19` | `0.23 / 0.02` | `0.24 / 0.03` |
| Multiblock 2D `f32` dense acos, `64x64` | `731.84 / 2.28` | `13.26 / 0.94` | `14.27 / 0.73` |
| Multiblock 2D `f32` dense atan, `64x64` | `135.48 / 1.74` | `0.30 / 0.02` | `0.32 / 0.03` |
| Multiblock 2D `f32` dense atan2, `64x64` | `140.22 / 2.53` | `0.29 / 0.03` | `0.32 / 0.04` |
| Multiblock 2D `f32` dense log10, `64x64` | `131.04 / 1.61` | `0.25 / 0.02` | `0.27 / 0.03` |
| Multiblock 2D `f32` dense erf, `64x64` | `135.10 / 1.60` | `0.25 / 0.02` | `0.28 / 0.03` |
| Multiblock 2D `f32` row+column reduction bundle add, `64x64 -> (64,) + (64,)` | `352.46 / 0.89` | `19.17 / 0.02` | `29.36 / 0.03` |

### Representative Multiblock PTX Row/Column Reductions

These representative rows exercise the exact multiblock PTX row/column reduction launch contract:
- `grid=(2,1,1)`
- `block=(32,1,1)`
- `grid.x * block.x >= output_extent`

| Family | Staged cold/warm ms | CUDA handles cold/warm ms | raw DLPack cold/warm ms |
| --- | ---: | ---: | ---: |
| Multiblock 2D `f32` row reduction add, `64x64 -> (64,)` | `536.49 / 0.86` | `23.31 / 0.02` | `13.22 / 0.03` |
| Multiblock 2D `f32` column reduction add, `64x64 -> (64,)` | `508.08 / 0.84` | `12.36 / 0.02` | `13.22 / 0.03` |

| Family | Backend | Cold ms | Warm median ms | Notes |
| --- | --- | ---: | ---: | --- |
| Indexed `f32` copy, `65536` elements | `ptx_exec` | `241.82` | `25.17` | Canonical indexed PTX copy path through driver JIT |
| Indexed `f32` add, `65536` elements | `ptx_exec` | `219.18` | `37.18` | Canonical indexed PTX pointwise add path through driver JIT |
| Indexed `i32` copy, `65536` elements | `ptx_exec` | `259.77` | `32.39` | Canonical indexed PTX copy path through driver JIT |
| Indexed `i32` add, `65536` elements | `ptx_exec` | `167.98` | `48.84` | Canonical indexed PTX pointwise add path through driver JIT |
| Indexed `i32` bitand, `65536` elements | `ptx_exec` | `388.51` | `49.31` | Canonical indexed PTX pointwise bitwise-and path through driver JIT |
| Indexed `i32` bitor, `65536` elements | `ptx_exec` | `352.67` | `49.01` | Canonical indexed PTX pointwise bitwise-or path through driver JIT |
| Indexed `i32` bitxor, `65536` elements | `ptx_exec` | `277.59` | `49.48` | Canonical indexed PTX pointwise bitwise-xor path through driver JIT |
| Indexed `i32` bitnot, `65536` elements | `ptx_exec` | `271.39` | `34.86` | Canonical indexed PTX unary bitwise-not path through driver JIT |
| Indexed `f32` abs, `65536` elements | `ptx_exec` | `261.63` | `24.81` | Canonical indexed PTX unary absolute-value path through driver JIT |
| Indexed `f32` round, `65536` elements | `ptx_exec` | `253.14` | `25.30` | Canonical indexed PTX unary round-to-nearest path through driver JIT; lowered via native PTX `cvt.rni.f32.f32` |
| Indexed `f32` floor, `65536` elements | `ptx_exec` | `253.74` | `25.04` | Canonical indexed PTX unary floor path through driver JIT; lowered via native PTX `cvt.rmi.f32.f32` |
| Indexed `i32` abs, `65536` elements | `ptx_exec` | `159.97` | `34.50` | Canonical indexed PTX unary absolute-value path through driver JIT |
| Indexed `f32` neg, `65536` elements | `ptx_exec` | `272.36` | `26.22` | Canonical indexed PTX unary negation path through driver JIT |
| Indexed `i32` neg, `65536` elements | `ptx_exec` | `165.91` | `34.98` | Canonical indexed PTX unary negation path through driver JIT |
| Indexed `f32` sqrt, `65536` elements | `ptx_exec` | `343.01` | `25.44` | Canonical indexed PTX unary sqrt path through driver JIT |
| Indexed `f32` rsqrt, `65536` elements | `ptx_exec` | `250.40` | `25.33` | Canonical indexed PTX unary rsqrt path through driver JIT |
| Indexed `f32` sin, `65536` elements | `ptx_exec` | `256.54` | `26.82` | Canonical indexed PTX unary sine path through driver JIT; uses `sin.approx.f32`, so accuracy is approximate |
| Indexed `f32` asin, `65536` elements | `ptx_exec` | `240.37` | `19.03` | Canonical indexed PTX unary asin path through driver JIT; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Indexed `f32` acos, `65536` elements | `ptx_exec` | `533.84` | `26.68` | Canonical indexed PTX unary acos path through driver JIT; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Indexed `f32` atan, `65536` elements | `ptx_exec` | `255.82` | `26.74` | Canonical indexed PTX unary atan path through driver JIT; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Indexed `f32` atan2, `65536` elements | `ptx_exec` | `285.95` | `40.36` | Canonical indexed PTX binary atan2 path through driver JIT; lowered via the same reciprocal-plus-piecewise polynomial PTX approximation plus explicit quadrant correction, so accuracy is approximate |
| Indexed `f32` exp, `65536` elements | `ptx_exec` | `248.33` | `25.66` | Canonical indexed PTX unary exp path through driver JIT; lowered via `ex2.approx.f32` after scaling by `log2(e)`, so accuracy is approximate |
| Indexed `f32` erf, `65536` elements | `ptx_exec` | `255.08` | `25.13` | Canonical indexed PTX unary erf path through driver JIT; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Indexed `f32` scalar-broadcast add, `65536` elements | `ptx_exec` | `273.45` | `25.17` | Canonical indexed PTX scalar-parameter broadcast path through driver JIT |
| Indexed `i32` scalar-broadcast add, `65536` elements | `ptx_exec` | `155.19` | `33.69` | Canonical indexed PTX scalar-parameter broadcast path through driver JIT |
| Indexed `i32` scalar-broadcast bitor, `65536` elements | `ptx_exec` | `366.32` | `32.62` | Canonical indexed PTX scalar-parameter bitwise-or broadcast path through driver JIT |
| Indexed `i32` tensor-source scalar-broadcast bitand, `65536` elements | `ptx_exec` | `160.25` | `32.04` | Canonical indexed PTX extent-1 tensor-source bitwise-and broadcast path through driver JIT |
| Indexed `i32` tensor-source scalar-broadcast bitor, `65536` elements | `ptx_exec` | `160.09` | `32.44` | Canonical indexed PTX extent-1 tensor-source bitwise-or broadcast path through driver JIT |
| Indexed `f32` scalar-branch select, `65536` elements | `ptx_exec` | `268.20` | `38.36` | Canonical indexed PTX `select` from an `i1` predicate tensor with a scalar-parameter false branch through driver JIT |
| Indexed `i32` tensor-source scalar-branch select, `65536` elements | `ptx_exec` | `176.78` | `46.87` | Canonical indexed PTX `select` from an `i1` predicate tensor with an extent-1 tensor true branch through driver JIT |
| Rank-1 `f32` copy-reduce add, `65536` elements | `ptx_exec` | `268.25` | `27.46` | Exact serial PTX rank-1 `copy_reduce` path through driver JIT |
| Rank-1 `f32` copy-reduce max, `65536` elements | `ptx_exec` | `256.13` | `27.16` | Same exact serial PTX rank-1 `copy_reduce` family, validated on the `max` reduction path |
| Rank-1 `i32` copy-reduce xor, `65536` elements | `ptx_exec` | `168.77` | `34.41` | Exact serial PTX rank-1 `copy_reduce` xor path through driver JIT |
| Rank-1 `i32` copy-reduce or, `65536` elements | `ptx_exec` | `171.94` | `34.80` | Same exact serial PTX rank-1 `copy_reduce` family, validated on the `or` reduction path |
| Indexed `f32` copy-reduce add, `65536` elements | `ptx_exec` | `269.18` | `24.79` | Exact indexed PTX rank-1 `copy_reduce` add path through driver JIT |
| Indexed `f32` copy-reduce max, `65536` elements | `ptx_exec` | `164.49` | `25.10` | Same indexed PTX rank-1 `copy_reduce` family, validated on the `max` reduction path |
| Indexed `i32` copy-reduce xor, `65536` elements | `ptx_exec` | `384.77` | `32.32` | Exact indexed PTX rank-1 `copy_reduce` xor path through driver JIT |
| Indexed `i32` copy-reduce or, `65536` elements | `ptx_exec` | `176.31` | `33.22` | Same indexed PTX rank-1 `copy_reduce` family, validated on the `or` reduction path |
| 2D `f32` dense copy, `64x64` | `ptx_exec` | `285.70` | `1.88` | Exact serial PTX 2D dense copy path through driver JIT |
| Parallel 2D `f32` dense copy, `64x64` | `ptx_exec` | `226.88` | `1.71` | Exact single-block PTX 2D dense copy path with one thread per output column |
| 2D `i32` dense copy, `64x64` | `ptx_exec` | `286.23` | `2.35` | Exact serial PTX 2D dense copy path through driver JIT |
| Parallel 2D `i32` dense copy, `64x64` | `ptx_exec` | `242.78` | `2.21` | Exact single-block PTX 2D dense copy path with one thread per output column |
| 2D `f32` dense scalar-broadcast add, `64x64` | `ptx_exec` | `289.29` | `1.88` | Exact serial PTX 2D scalar-parameter broadcast path through driver JIT |
| Parallel 2D `f32` dense scalar-broadcast add, `64x64` | `ptx_exec` | `136.05` | `1.69` | Exact single-block PTX 2D scalar-parameter broadcast path with one thread per output column |
| 2D `i32` dense scalar-broadcast add, `64x64` | `ptx_exec` | `256.73` | `2.57` | Exact serial PTX 2D scalar-parameter broadcast path through driver JIT |
| 2D `i32` dense scalar-broadcast bitand, `64x64` | `ptx_exec` | `229.80` | `2.14` | Exact serial PTX 2D scalar-parameter bitwise-and broadcast path through driver JIT |
| Parallel 2D `i32` dense scalar-broadcast add, `64x64` | `ptx_exec` | `139.95` | `2.18` | Exact single-block PTX 2D scalar-parameter broadcast path with one thread per output column |
| Parallel 2D `i32` dense scalar-broadcast bitand, `64x64` | `ptx_exec` | `137.25` | `2.07` | Exact single-block PTX 2D scalar-parameter bitwise-and broadcast path with one thread per output column |
| 2D `f32` dense tensor-source scalar-broadcast add, `64x64` | `ptx_exec` | `242.02` | `1.77` | Exact serial PTX 2D extent-1 tensor-source scalar-broadcast path through driver JIT |
| Parallel 2D `f32` dense tensor-source scalar-broadcast add, `64x64` | `ptx_exec` | `139.22` | `1.65` | Exact single-block PTX 2D extent-1 tensor-source scalar-broadcast path with one thread per output column |
| 2D `i32` dense tensor-source scalar-broadcast add, `64x64` | `ptx_exec` | `145.60` | `2.29` | Exact serial PTX 2D extent-1 tensor-source scalar-broadcast path through driver JIT |
| 2D `i32` dense tensor-source scalar-broadcast bitor, `64x64` | `ptx_exec` | `138.39` | `2.23` | Exact serial PTX 2D extent-1 tensor-source bitwise-or broadcast path through driver JIT |
| Parallel 2D `i32` dense tensor-source scalar-broadcast add, `64x64` | `ptx_exec` | `142.59` | `2.10` | Exact single-block PTX 2D extent-1 tensor-source scalar-broadcast path with one thread per output column |
| Parallel 2D `i32` dense tensor-source scalar-broadcast bitor, `64x64` | `ptx_exec` | `136.81` | `2.08` | Exact single-block PTX 2D extent-1 tensor-source bitwise-or broadcast path with one thread per output column |
| 2D `f32` dense select, `64x64` | `ptx_exec` | `299.84` | `3.68` | Exact serial PTX 2D dense tensor-select path through driver JIT |
| 2D `i32` broadcast select, `64x64` | `ptx_exec` | `149.24` | `2.23` | Exact serial PTX 2D broadcast tensor-select path through driver JIT |
| Parallel 2D `f32` dense scalar-branch select, `64x64` | `ptx_exec` | `136.72` | `2.60` | Exact single-block PTX 2D scalar-branch select path with one thread per output column |
| 2D `f32` copy-reduce add, `64x64` | `ptx_exec` | `146.73` | `1.73` | Exact serial PTX 2D `copy_reduce` path through driver JIT |
| 2D `f32` copy-reduce max, `64x64` | `ptx_exec` | `146.99` | `1.74` | Same exact serial PTX 2D `copy_reduce` family, validated on the `max` reduction path |
| Parallel 2D `f32` copy-reduce add, `64x64` | `ptx_exec` | `145.55` | `1.63` | Exact single-block PTX 2D `copy_reduce` path with one thread per output column |
| Parallel 2D `i32` copy-reduce or, `64x64` | `ptx_exec` | `141.83` | `2.09` | Same exact single-block PTX 2D `copy_reduce` family, validated on the `or` reduction path |
| 2D `f32` row reduction add, `64x64 -> (64,)` | `ptx_exec` | `236.66` | `0.89` | Exact serial PTX 2D rowwise reduction-add path through driver JIT |
| 2D `f32` row reduction mul, `64x64 -> (64,)` | `ptx_exec` | `239.09` | `0.89` | Exact serial PTX 2D rowwise reduction-mul path through driver JIT |
| 2D `f32` row reduction max, `64x64 -> (64,)` | `ptx_exec` | `123.82` | `0.87` | Exact serial PTX 2D rowwise reduction-max path through driver JIT |
| 2D `f32` row reduction min, `64x64 -> (64,)` | `ptx_exec` | `123.18` | `0.88` | Exact serial PTX 2D rowwise reduction-min path through driver JIT |
| Parallel 2D `f32` row reduction add, `64x64 -> (64,)` | `ptx_exec` | `138.11` | `0.83` | Exact single-block PTX 2D rowwise reduction-add path with one thread per output row |
| 2D `f32` column reduction add, `64x64 -> (64,)` | `ptx_exec` | `239.78` | `0.85` | Exact serial PTX 2D columnwise reduction-add path through driver JIT |
| 2D `f32` column reduction mul, `64x64 -> (64,)` | `ptx_exec` | `120.83` | `0.86` | Exact serial PTX 2D columnwise reduction-mul path through driver JIT |
| 2D `f32` column reduction max, `64x64 -> (64,)` | `ptx_exec` | `120.67` | `0.86` | Exact serial PTX 2D columnwise reduction-max path through driver JIT |
| 2D `f32` column reduction min, `64x64 -> (64,)` | `ptx_exec` | `121.78` | `0.86` | Exact serial PTX 2D columnwise reduction-min path through driver JIT |
| Parallel 2D `f32` column reduction add, `64x64 -> (64,)` | `ptx_exec` | `135.08` | `0.83` | Exact single-block PTX 2D columnwise reduction-add path with one thread per output column |
| Parallel 2D `f32` column reduction mul, `64x64 -> (64,)` | `ptx_exec` | `243.42` | `1.06` | Exact single-block PTX 2D columnwise reduction-mul path with one thread per output column |
| Parallel 2D `f32` column reduction max, `64x64 -> (64,)` | `ptx_exec` | `132.98` | `0.84` | Exact single-block PTX 2D columnwise reduction-max path with one thread per output column |
| Parallel 2D `f32` column reduction min, `64x64 -> (64,)` | `ptx_exec` | `131.77` | `0.84` | Exact single-block PTX 2D columnwise reduction-min path with one thread per output column |
| Parallel 2D `f32` row reduction mul, `64x64 -> (64,)` | `ptx_exec` | `249.58` | `0.83` | Exact single-block PTX 2D rowwise reduction-mul path with one thread per output row |
| Parallel 2D `f32` row reduction max, `64x64 -> (64,)` | `ptx_exec` | `139.88` | `0.84` | Exact single-block PTX 2D rowwise reduction-max path with one thread per output row |
| Parallel 2D `f32` row reduction min, `64x64 -> (64,)` | `ptx_exec` | `135.70` | `0.84` | Exact single-block PTX 2D rowwise reduction-min path with one thread per output row |
| 2D `i32` row reduction add, `64x64 -> (64,)` | `ptx_exec` | `137.51` | `1.13` | Exact serial PTX 2D rowwise reduction-add path through driver JIT |
| 2D `i32` row reduction mul, `64x64 -> (64,)` | `ptx_exec` | `120.92` | `1.12` | Exact serial PTX 2D rowwise reduction-mul path through driver JIT |
| 2D `i32` row reduction max, `64x64 -> (64,)` | `ptx_exec` | `120.34` | `1.11` | Exact serial PTX 2D rowwise reduction-max path through driver JIT |
| 2D `i32` row reduction min, `64x64 -> (64,)` | `ptx_exec` | `122.42` | `1.11` | Exact serial PTX 2D rowwise reduction-min path through driver JIT |
| Parallel 2D `i32` row reduction add, `64x64 -> (64,)` | `ptx_exec` | `134.11` | `1.07` | Exact single-block PTX 2D rowwise reduction-add path with one thread per output row |
| 2D `i32` column reduction add, `64x64 -> (64,)` | `ptx_exec` | `134.26` | `1.12` | Exact serial PTX 2D columnwise reduction-add path through driver JIT |
| 2D `i32` column reduction mul, `64x64 -> (64,)` | `ptx_exec` | `121.10` | `1.11` | Exact serial PTX 2D columnwise reduction-mul path through driver JIT |
| 2D `i32` column reduction max, `64x64 -> (64,)` | `ptx_exec` | `122.95` | `1.10` | Exact serial PTX 2D columnwise reduction-max path through driver JIT |
| 2D `i32` column reduction min, `64x64 -> (64,)` | `ptx_exec` | `121.93` | `1.10` | Exact serial PTX 2D columnwise reduction-min path through driver JIT |
| Parallel 2D `i32` column reduction add, `64x64 -> (64,)` | `ptx_exec` | `133.54` | `1.08` | Exact single-block PTX 2D columnwise reduction-add path with one thread per output column |
| Parallel 2D `i32` column reduction mul, `64x64 -> (64,)` | `ptx_exec` | `135.57` | `1.06` | Exact single-block PTX 2D columnwise reduction-mul path with one thread per output column |
| Parallel 2D `i32` column reduction max, `64x64 -> (64,)` | `ptx_exec` | `137.77` | `1.07` | Exact single-block PTX 2D columnwise reduction-max path with one thread per output column |
| Parallel 2D `i32` column reduction min, `64x64 -> (64,)` | `ptx_exec` | `134.35` | `1.07` | Exact single-block PTX 2D columnwise reduction-min path with one thread per output column |
| Parallel 2D `i32` column reduction or, `64x64 -> (64,)` | `ptx_exec` | `393.44` | `1.10` | Exact single-block PTX 2D columnwise reduction-or path with one thread per output column |
| Parallel 2D `i32` row reduction mul, `64x64 -> (64,)` | `ptx_exec` | `136.94` | `1.07` | Exact single-block PTX 2D rowwise reduction-mul path with one thread per output row |
| Parallel 2D `i32` row reduction max, `64x64 -> (64,)` | `ptx_exec` | `136.43` | `1.08` | Exact single-block PTX 2D rowwise reduction-max path with one thread per output row |
| Parallel 2D `i32` row reduction min, `64x64 -> (64,)` | `ptx_exec` | `133.51` | `1.07` | Exact single-block PTX 2D rowwise reduction-min path with one thread per output row |
| Parallel 2D `i32` row reduction and, `64x64 -> (64,)` | `ptx_exec` | `223.42` | `1.08` | Exact single-block PTX 2D rowwise reduction-and path with one thread per output row |
| 2D `f32` dense add, `64x64` | `ptx_exec` | `243.80` | `2.64` | Exact serial PTX 2D dense tensor-binary path through driver JIT |
| Parallel 2D `f32` dense add, `64x64` | `ptx_exec` | `130.06` | `2.59` | Exact single-block PTX 2D dense tensor-binary path with one thread per output column |
| 2D `i32` dense add, `64x64` | `ptx_exec` | `244.26` | `3.37` | Exact serial PTX 2D dense tensor-binary path through driver JIT |
| Parallel 2D `i32` dense add, `64x64` | `ptx_exec` | `99.98` | `3.23` | Exact single-block PTX 2D dense tensor-binary path with one thread per output column |
| 2D `f32` broadcast add, `64x64` | `ptx_exec` | `152.06` | `1.04` | Exact serial PTX 2D broadcast tensor-binary path through driver JIT |
| Parallel 2D `f32` broadcast add, `64x64` | `ptx_exec` | `122.12` | `0.97` | Exact single-block PTX 2D broadcast tensor-binary path with one thread per output column |
| 2D `i32` broadcast add, `64x64` | `ptx_exec` | `229.76` | `1.29` | Exact serial PTX 2D broadcast tensor-binary path through driver JIT |
| Parallel 2D `i32` broadcast add, `64x64` | `ptx_exec` | `138.91` | `1.19` | Exact single-block PTX 2D broadcast tensor-binary path with one thread per output column |
| 2D `f32` dense abs, `64x64` | `ptx_exec` | `136.79` | `1.74` | Exact serial PTX 2D unary absolute-value path through driver JIT |
| 2D `f32` dense round, `64x64` | `ptx_exec` | `136.68` | `1.77` | Exact serial PTX 2D unary round-to-nearest path through driver JIT; lowered via native PTX `cvt.rni.f32.f32` |
| 2D `f32` dense trunc, `64x64` | `ptx_exec` | `137.89` | `1.78` | Exact serial PTX 2D unary truncation path through driver JIT; lowered via native PTX `cvt.rzi.f32.f32` |
| Parallel 2D `f32` dense abs, `64x64` | `ptx_exec` | `133.10` | `1.63` | Exact single-block PTX 2D unary absolute-value path with one thread per output column |
| Parallel 2D `f32` dense round, `64x64` | `ptx_exec` | `131.20` | `1.59` | Exact single-block PTX 2D unary round-to-nearest path with one thread per output column; lowered via native PTX `cvt.rni.f32.f32` |
| Parallel 2D `f32` dense floor, `64x64` | `ptx_exec` | `129.86` | `1.62` | Exact single-block PTX 2D unary floor path with one thread per output column; lowered via native PTX `cvt.rmi.f32.f32` |
| 2D `f32` dense sqrt, `64x64` | `ptx_exec` | `145.57` | `1.95` | Exact serial PTX 2D unary sqrt path through driver JIT |
| 2D `f32` dense cos, `64x64` | `ptx_exec` | `141.49` | `1.85` | Exact serial PTX 2D unary cosine path through driver JIT; uses `cos.approx.f32`, so accuracy is approximate |
| 2D `f32` dense asin, `64x64` | `ptx_exec` | `127.41` | `1.64` | Exact serial PTX 2D unary asin path through driver JIT; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| 2D `f32` dense acos, `64x64` | `ptx_exec` | `779.52` | `2.59` | Exact serial PTX 2D unary acos path through driver JIT; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| 2D `f32` dense atan, `64x64` | `ptx_exec` | `139.09` | `1.95` | Exact serial PTX 2D unary atan path through driver JIT; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| 2D `f32` dense log, `64x64` | `ptx_exec` | `142.24` | `1.78` | Exact serial PTX 2D unary log path through driver JIT; lowered via `lg2.approx.f32` times `ln(2)`, so accuracy is approximate |
| 2D `f32` dense erf, `64x64` | `ptx_exec` | `132.82` | `1.90` | Exact serial PTX 2D unary erf path through driver JIT; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Parallel 2D `f32` dense sqrt, `64x64` | `ptx_exec` | `122.31` | `1.76` | Exact single-block PTX 2D unary sqrt path with one thread per output column |
| Parallel 2D `f32` dense log2, `64x64` | `ptx_exec` | `138.22` | `1.70` | Exact single-block PTX 2D unary log2 path with one thread per output column; uses `lg2.approx.f32`, so accuracy is approximate |
| Parallel 2D `f32` dense asin, `64x64` | `ptx_exec` | `121.46` | `1.20` | Exact single-block PTX 2D unary asin path with one thread per output column; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Parallel 2D `f32` dense acos, `64x64` | `ptx_exec` | `733.80` | `2.06` | Exact single-block PTX 2D unary acos path with one thread per output column; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Parallel 2D `f32` dense atan, `64x64` | `ptx_exec` | `138.68` | `1.71` | Exact single-block PTX 2D unary atan path with one thread per output column; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Parallel 2D `f32` dense exp, `64x64` | `ptx_exec` | `130.89` | `1.60` | Exact single-block PTX 2D unary exp path with one thread per output column; lowered via `ex2.approx.f32` after scaling by `log2(e)`, so accuracy is approximate |
| Parallel 2D `f32` dense erf, `64x64` | `ptx_exec` | `139.10` | `1.61` | Exact single-block PTX 2D unary erf path with one thread per output column; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| 2D `i32` dense abs, `64x64` | `ptx_exec` | `137.16` | `2.27` | Exact serial PTX 2D unary absolute-value path through driver JIT |
| Parallel 2D `i32` dense abs, `64x64` | `ptx_exec` | `132.85` | `2.17` | Exact single-block PTX 2D unary absolute-value path with one thread per output column |
| 2D `i32` dense bitnot, `64x64` | `ptx_exec` | `136.60` | `2.30` | Exact serial PTX 2D unary bitwise-not path through driver JIT |
| Parallel 2D `i32` dense bitnot, `64x64` | `ptx_exec` | `130.66` | `2.17` | Exact single-block PTX 2D unary bitwise-not path with one thread per output column |
| 2D `f32` dense rsqrt, `64x64` | `ptx_exec` | `156.08` | `1.90` | Exact serial PTX 2D unary rsqrt path through driver JIT; uses `rsqrt.approx.f32`, so accuracy is approximate |
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
| Parallel 2D `f32` tensor-factory bundle, `64x64` | `ptx_exec` | `253.36` | `2.51` | Exact single-block PTX tensor-factory bundle with one thread per output column |
| Parallel 2D `i32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `379.27` | `1.37` | Exact single-block shared-memory PTX reduction-add bundle through driver JIT |
| Parallel 2D `i32` reduction mul bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `377.81` | `1.37` | Exact single-block shared-memory PTX reduction-mul bundle through driver JIT |
| Parallel 2D `i32` reduction max bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `412.13` | `1.24` | Exact single-block shared-memory PTX reduction-max bundle through driver JIT |
| Parallel 2D `i32` reduction min bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `268.32` | `1.26` | Exact single-block shared-memory PTX reduction-min bundle through driver JIT |
| Parallel 2D `i32` reduction xor bundle, `64x64 -> (1,) + (64,) + (64,)` | `ptx_exec` | `420.80` | `1.16` | Exact single-block shared-memory PTX reduction-xor bundle through driver JIT |
| 2D `i32` tensor-factory bundle, `64x64` | `ptx_exec` | `411.07` | `5.07` | Exact PTX tensor-factory bundle through driver JIT |
| Parallel 2D `i32` tensor-factory bundle, `64x64` | `ptx_exec` | `138.85` | `3.10` | Exact single-block PTX tensor-factory bundle with one thread per output column |
| Indexed `f32` copy, `65536` elements, CUDA handles | `ptx_exec` | `0.34` | `0.02` | Same canonical PTX copy kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` add, `65536` elements, CUDA handles | `ptx_exec` | `0.30` | `0.02` | Same canonical PTX add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` copy, `65536` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same canonical PTX copy kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` add, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` bitand, `65536` elements, CUDA handles | `ptx_exec` | `0.41` | `0.03` | Same canonical PTX bitwise-and kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` bitor, `65536` elements, CUDA handles | `ptx_exec` | `0.34` | `0.02` | Same canonical PTX bitwise-or kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` bitxor, `65536` elements, CUDA handles | `ptx_exec` | `0.26` | `0.02` | Same canonical PTX bitwise-xor kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` bitnot, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX unary bitwise-not kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` abs, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` round, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX unary round-to-nearest kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via native PTX `cvt.rni.f32.f32` |
| Indexed `f32` floor, `65536` elements, CUDA handles | `ptx_exec` | `0.29` | `0.02` | Same canonical PTX unary floor kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via native PTX `cvt.rmi.f32.f32` |
| Indexed `i32` abs, `65536` elements, CUDA handles | `ptx_exec` | `0.29` | `0.02` | Same canonical PTX unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` neg, `65536` elements, CUDA handles | `ptx_exec` | `0.26` | `0.02` | Same canonical PTX unary negation kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` neg, `65536` elements, CUDA handles | `ptx_exec` | `0.26` | `0.02` | Same canonical PTX unary negation kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` sqrt, `65536` elements, CUDA handles | `ptx_exec` | `0.32` | `0.02` | Same canonical PTX unary sqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` rsqrt, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX unary rsqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` sin, `65536` elements, CUDA handles | `ptx_exec` | `0.30` | `0.02` | Same canonical PTX unary sine kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; accuracy is approximate |
| Indexed `f32` asin, `65536` elements, CUDA handles | `ptx_exec` | `0.26` | `0.02` | Same canonical PTX unary asin kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Indexed `f32` acos, `65536` elements, CUDA handles | `ptx_exec` | `0.58` | `0.12` | Same canonical PTX unary acos kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Indexed `f32` atan, `65536` elements, CUDA handles | `ptx_exec` | `0.34` | `0.02` | Same canonical PTX unary atan kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Indexed `f32` atan2, `65536` elements, CUDA handles | `ptx_exec` | `0.32` | `0.02` | Same canonical PTX binary atan2 kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via the same reciprocal-plus-piecewise polynomial PTX approximation plus explicit quadrant correction, so accuracy is approximate |
| Indexed `f32` exp, `65536` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same canonical PTX unary exp kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `ex2.approx.f32` after scaling by `log2(e)`, so accuracy is approximate |
| Indexed `f32` erf, `65536` elements, CUDA handles | `ptx_exec` | `0.26` | `0.02` | Same canonical PTX unary erf kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Indexed `f32` scalar-broadcast add, `65536` elements, CUDA handles | `ptx_exec` | `0.30` | `0.02` | Same canonical PTX scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` scalar-broadcast add, `65536` elements, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same canonical PTX scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` scalar-broadcast bitor, `65536` elements, CUDA handles | `ptx_exec` | `0.36` | `0.13` | Same canonical PTX scalar-broadcast bitor kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` tensor-source scalar-broadcast bitand, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX extent-1 tensor-source scalar-broadcast bitand kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `i32` tensor-source scalar-broadcast bitor, `65536` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same canonical PTX extent-1 tensor-source scalar-broadcast bitor kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` scalar-branch select, `65536` elements, CUDA handles | `ptx_exec` | `0.27` | `0.02` | Same canonical indexed PTX `select` kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Indexed `i32` tensor-source scalar-branch select, `65536` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same canonical indexed PTX `select` kernel with an extent-1 tensor branch, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Rank-1 `f32` copy-reduce add, `65536` elements, CUDA handles | `ptx_exec` | `18.27` | `2.20` | Same exact PTX rank-1 `copy_reduce` add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Rank-1 `f32` copy-reduce max, `65536` elements, CUDA handles | `ptx_exec` | `2.48` | `2.20` | Same exact PTX rank-1 `copy_reduce` max kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Rank-1 `i32` copy-reduce xor, `65536` elements, CUDA handles | `ptx_exec` | `18.20` | `2.20` | Same exact PTX rank-1 `copy_reduce` xor kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Rank-1 `i32` copy-reduce or, `65536` elements, CUDA handles | `ptx_exec` | `2.47` | `2.19` | Same exact PTX rank-1 `copy_reduce` or kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Indexed `f32` copy-reduce add, `65536` elements, CUDA handles | `ptx_exec` | `7.38` | `0.02` | Same indexed PTX rank-1 `copy_reduce` add kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Indexed `f32` copy-reduce max, `65536` elements, CUDA handles | `ptx_exec` | `0.27` | `0.02` | Same indexed PTX rank-1 `copy_reduce` max kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Indexed `i32` copy-reduce xor, `65536` elements, CUDA handles | `ptx_exec` | `0.33` | `0.02` | Same indexed PTX rank-1 `copy_reduce` xor kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Indexed `i32` copy-reduce or, `65536` elements, CUDA handles | `ptx_exec` | `0.26` | `0.02` | Same indexed PTX rank-1 `copy_reduce` or kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` dense copy, `64x64`, CUDA handles | `ptx_exec` | `14.77` | `0.15` | Same exact PTX 2D dense copy kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` dense copy, `64x64`, CUDA handles | `ptx_exec` | `0.24` | `0.03` | Same exact single-block PTX 2D dense copy kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` dense copy, `64x64`, CUDA handles | `ptx_exec` | `14.99` | `0.14` | Same exact PTX 2D dense copy kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` dense copy, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D dense copy kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` dense scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `15.89` | `0.16` | Same exact PTX 2D scalar-parameter broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` dense scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.28` | `0.03` | Same exact single-block PTX 2D scalar-parameter broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` dense scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `16.59` | `0.15` | Same exact PTX 2D scalar-parameter broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` dense scalar-broadcast bitand, `64x64`, CUDA handles | `ptx_exec` | `0.38` | `0.15` | Same exact PTX 2D scalar-parameter bitwise-and broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` dense scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.27` | `0.03` | Same exact single-block PTX 2D scalar-parameter broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` dense scalar-broadcast bitand, `64x64`, CUDA handles | `ptx_exec` | `0.28` | `0.03` | Same exact single-block PTX 2D scalar-parameter bitwise-and broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` dense tensor-source scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.37` | `0.15` | Same exact PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` dense tensor-source scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.29` | `0.03` | Same exact single-block PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` dense tensor-source scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.38` | `0.15` | Same exact PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` dense tensor-source scalar-broadcast bitor, `64x64`, CUDA handles | `ptx_exec` | `0.37` | `0.15` | Same exact PTX 2D extent-1 tensor-source bitwise-or broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` dense tensor-source scalar-broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.21` | `0.02` | Same exact single-block PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` dense tensor-source scalar-broadcast bitor, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D extent-1 tensor-source bitwise-or broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` dense select, `64x64`, CUDA handles | `ptx_exec` | `0.49` | `0.26` | Same exact PTX 2D dense tensor-select kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` broadcast select, `64x64`, CUDA handles | `ptx_exec` | `0.46` | `0.17` | Same exact PTX 2D broadcast tensor-select kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` dense scalar-branch select, `64x64`, CUDA handles | `ptx_exec` | `0.31` | `0.04` | Same exact single-block PTX 2D scalar-branch select kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` copy-reduce add, `64x64`, CUDA handles | `ptx_exec` | `22.09` | `0.15` | Same exact PTX 2D `copy_reduce` add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` copy-reduce max, `64x64`, CUDA handles | `ptx_exec` | `0.41` | `0.15` | Same exact PTX 2D `copy_reduce` max kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` copy-reduce add, `64x64`, CUDA handles | `ptx_exec` | `22.12` | `0.03` | Same exact single-block PTX 2D `copy_reduce` add kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` copy-reduce or, `64x64`, CUDA handles | `ptx_exec` | `0.27` | `0.03` | Same exact single-block PTX 2D `copy_reduce` or kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` row reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.27` | `0.05` | Same exact PTX 2D rowwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` row reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.05` | Same exact PTX 2D rowwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` row reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.25` | `0.06` | Same exact PTX 2D rowwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` row reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.27` | `0.05` | Same exact PTX 2D rowwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` row reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same exact single-block PTX 2D rowwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` column reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.04` | Same exact PTX 2D columnwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` column reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.04` | Same exact PTX 2D columnwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` column reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.04` | Same exact PTX 2D columnwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` column reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.04` | Same exact PTX 2D columnwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` column reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same exact single-block PTX 2D columnwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` column reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same exact single-block PTX 2D columnwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` column reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D columnwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` column reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D columnwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` row reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same exact single-block PTX 2D rowwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` row reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D rowwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` row reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D rowwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` row reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.05` | Same exact PTX 2D rowwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` row reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.27` | `0.06` | Same exact PTX 2D rowwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` row reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.06` | Same exact PTX 2D rowwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` row reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.06` | Same exact PTX 2D rowwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` row reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same exact single-block PTX 2D rowwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` column reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.04` | Same exact PTX 2D columnwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` column reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.04` | Same exact PTX 2D columnwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` column reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.26` | `0.04` | Same exact PTX 2D columnwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` column reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.25` | `0.04` | Same exact PTX 2D columnwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` column reduction add, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D columnwise reduction-add kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` column reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D columnwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` column reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D columnwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` column reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D columnwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` column reduction or, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `14.05` | `0.02` | Same exact single-block PTX 2D columnwise reduction-or kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` row reduction mul, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D rowwise reduction-mul kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` row reduction max, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D rowwise reduction-max kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` row reduction min, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block PTX 2D rowwise reduction-min kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `i32` row reduction and, `64x64 -> (64,)`, CUDA handles | `ptx_exec` | `0.21` | `0.02` | Same exact single-block PTX 2D rowwise reduction-and kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` row reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.29` | `0.06` | Same exact PTX 2D rowwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` row reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.29` | `0.06` | Same exact PTX 2D rowwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` row reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.31` | `0.06` | Same exact PTX 2D rowwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` row reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.30` | `0.06` | Same exact PTX 2D rowwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` row reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D rowwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` column reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.28` | `0.05` | Same exact PTX 2D columnwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` column reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.29` | `0.05` | Same exact PTX 2D columnwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` column reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.27` | `0.05` | Same exact PTX 2D columnwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` column reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.28` | `0.05` | Same exact PTX 2D columnwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` column reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D columnwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` column reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D columnwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` column reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.02` | Same exact single-block PTX 2D columnwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` column reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D columnwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` row reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D rowwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` row reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D rowwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` row reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D rowwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` row reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.27` | `0.06` | Same exact PTX 2D rowwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` row reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.29` | `0.06` | Same exact PTX 2D rowwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` row reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.30` | `0.06` | Same exact PTX 2D rowwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` row reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.29` | `0.06` | Same exact PTX 2D rowwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` row reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.27` | `0.03` | Same exact single-block PTX 2D rowwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` column reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.27` | `0.04` | Same exact PTX 2D columnwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` column reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.28` | `0.05` | Same exact PTX 2D columnwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` column reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.27` | `0.05` | Same exact PTX 2D columnwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` column reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.29` | `0.05` | Same exact PTX 2D columnwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` column reduction add, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D columnwise reduction-add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` column reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D columnwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` column reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D columnwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` column reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D columnwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` column reduction or, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.27` | `0.13` | Same exact single-block PTX 2D columnwise reduction-or kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` row reduction mul, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D rowwise reduction-mul kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` row reduction max, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D rowwise reduction-max kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` row reduction min, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D rowwise reduction-min kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` row reduction and, `64x64 -> (64,)`, raw DLPack | `ptx_exec` | `0.30` | `0.03` | Same exact single-block PTX 2D rowwise reduction-and kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` reduction xor bundle, `64x64 -> (1,) + (64,) + (64,)`, raw DLPack | `ptx_exec` | `24.20` | `0.04` | Same exact single-block shared-memory PTX reduction-xor bundle, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` dense scalar-broadcast bitand, `64x64`, raw DLPack | `ptx_exec` | `0.41` | `0.15` | Same exact PTX 2D scalar-parameter bitwise-and broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` dense scalar-broadcast bitand, `64x64`, raw DLPack | `ptx_exec` | `0.30` | `0.04` | Same exact single-block PTX 2D scalar-parameter bitwise-and broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` dense tensor-source scalar-broadcast add, `64x64`, raw DLPack | `ptx_exec` | `0.39` | `0.16` | Same exact PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` dense tensor-source scalar-broadcast add, `64x64`, raw DLPack | `ptx_exec` | `0.28` | `0.04` | Same exact single-block PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` dense tensor-source scalar-broadcast add, `64x64`, raw DLPack | `ptx_exec` | `0.42` | `0.16` | Same exact PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` dense tensor-source scalar-broadcast bitor, `64x64`, raw DLPack | `ptx_exec` | `0.41` | `0.16` | Same exact PTX 2D extent-1 tensor-source bitwise-or broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` dense tensor-source scalar-broadcast add, `64x64`, raw DLPack | `ptx_exec` | `0.29` | `0.04` | Same exact single-block PTX 2D extent-1 tensor-source scalar-broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` dense tensor-source scalar-broadcast bitor, `64x64`, raw DLPack | `ptx_exec` | `0.29` | `0.04` | Same exact single-block PTX 2D extent-1 tensor-source bitwise-or broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` dense select, `64x64`, raw DLPack | `ptx_exec` | `0.58` | `0.28` | Same exact PTX 2D dense tensor-select kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` broadcast select, `64x64`, raw DLPack | `ptx_exec` | `0.47` | `0.18` | Same exact PTX 2D broadcast tensor-select kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` dense scalar-branch select, `64x64`, raw DLPack | `ptx_exec` | `0.33` | `0.05` | Same exact single-block PTX 2D scalar-branch select kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` dense abs, `64x64`, raw DLPack | `ptx_exec` | `0.41` | `0.15` | Same exact PTX 2D unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` dense round, `64x64`, raw DLPack | `ptx_exec` | `0.42` | `0.17` | Same exact PTX 2D unary round-to-nearest kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rni.f32.f32` |
| 2D `f32` dense trunc, `64x64`, raw DLPack | `ptx_exec` | `0.42` | `0.17` | Same exact PTX 2D unary truncation kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rzi.f32.f32` |
| Parallel 2D `f32` dense abs, `64x64`, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` dense round, `64x64`, raw DLPack | `ptx_exec` | `0.28` | `0.03` | Same exact single-block PTX 2D unary round-to-nearest kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rni.f32.f32` |
| Parallel 2D `f32` dense floor, `64x64`, raw DLPack | `ptx_exec` | `0.28` | `0.03` | Same exact single-block PTX 2D unary floor kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rmi.f32.f32` |
| 2D `f32` dense cos, `64x64`, raw DLPack | `ptx_exec` | `0.47` | `0.19` | Same exact PTX 2D unary cosine kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; accuracy is approximate |
| 2D `f32` dense asin, `64x64`, raw DLPack | `ptx_exec` | `0.70` | `0.47` | Same exact PTX 2D unary asin kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| 2D `f32` dense acos, `64x64`, raw DLPack | `ptx_exec` | `12.08` | `0.93` | Same exact PTX 2D unary acos kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| 2D `f32` dense atan, `64x64`, raw DLPack | `ptx_exec` | `0.55` | `0.27` | Same exact PTX 2D unary atan kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| 2D `f32` dense log, `64x64`, raw DLPack | `ptx_exec` | `0.47` | `0.21` | Same exact PTX 2D unary log kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `lg2.approx.f32` times `ln(2)`, so accuracy is approximate |
| 2D `f32` dense erf, `64x64`, raw DLPack | `ptx_exec` | `0.56` | `0.31` | Same exact PTX 2D unary erf kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Parallel 2D `f32` dense log2, `64x64`, raw DLPack | `ptx_exec` | `0.31` | `0.04` | Same exact single-block PTX 2D unary log2 kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; accuracy is approximate |
| Parallel 2D `f32` dense asin, `64x64`, raw DLPack | `ptx_exec` | `0.29` | `0.04` | Same exact single-block PTX 2D unary asin kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Parallel 2D `f32` dense acos, `64x64`, raw DLPack | `ptx_exec` | `0.43` | `0.04` | Same exact single-block PTX 2D unary acos kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Parallel 2D `f32` dense atan, `64x64`, raw DLPack | `ptx_exec` | `0.32` | `0.04` | Same exact single-block PTX 2D unary atan kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Parallel 2D `f32` dense exp, `64x64`, raw DLPack | `ptx_exec` | `0.25` | `0.04` | Same exact single-block PTX 2D unary exp kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `ex2.approx.f32` after scaling by `log2(e)`, so accuracy is approximate |
| Parallel 2D `f32` dense erf, `64x64`, raw DLPack | `ptx_exec` | `0.27` | `0.04` | Same exact single-block PTX 2D unary erf kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| 2D `i32` dense abs, `64x64`, raw DLPack | `ptx_exec` | `0.40` | `0.16` | Same exact PTX 2D unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` dense abs, `64x64`, raw DLPack | `ptx_exec` | `0.27` | `0.04` | Same exact single-block PTX 2D unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `i32` dense bitnot, `64x64`, raw DLPack | `ptx_exec` | `0.41` | `0.15` | Same exact PTX 2D unary bitwise-not kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `i32` dense bitnot, `64x64`, raw DLPack | `ptx_exec` | `0.28` | `0.03` | Same exact single-block PTX 2D unary bitwise-not kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` bitand, `65536` elements, raw DLPack | `ptx_exec` | `0.40` | `0.03` | Same canonical PTX bitwise-and kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` bitor, `65536` elements, raw DLPack | `ptx_exec` | `0.32` | `0.03` | Same canonical PTX bitwise-or kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` bitxor, `65536` elements, raw DLPack | `ptx_exec` | `0.32` | `0.03` | Same canonical PTX bitwise-xor kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` bitnot, `65536` elements, raw DLPack | `ptx_exec` | `0.27` | `0.02` | Same canonical PTX unary bitwise-not kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `f32` abs, `65536` elements, raw DLPack | `ptx_exec` | `0.27` | `0.03` | Same canonical PTX unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `f32` round, `65536` elements, raw DLPack | `ptx_exec` | `0.27` | `0.02` | Same canonical PTX unary round-to-nearest kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rni.f32.f32` |
| Indexed `f32` floor, `65536` elements, raw DLPack | `ptx_exec` | `0.26` | `0.02` | Same canonical PTX unary floor kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rmi.f32.f32` |
| Indexed `i32` abs, `65536` elements, raw DLPack | `ptx_exec` | `0.27` | `0.02` | Same canonical PTX unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `f32` neg, `65536` elements, raw DLPack | `ptx_exec` | `0.27` | `0.03` | Same canonical PTX unary negation kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` neg, `65536` elements, raw DLPack | `ptx_exec` | `0.35` | `0.03` | Same canonical PTX unary negation kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `f32` sin, `65536` elements, raw DLPack | `ptx_exec` | `0.32` | `0.03` | Same canonical PTX unary sine kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; accuracy is approximate |
| Indexed `f32` asin, `65536` elements, raw DLPack | `ptx_exec` | `0.26` | `0.02` | Same canonical PTX unary asin kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Indexed `f32` acos, `65536` elements, raw DLPack | `ptx_exec` | `0.46` | `0.12` | Same canonical PTX unary acos kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Indexed `f32` atan, `65536` elements, raw DLPack | `ptx_exec` | `0.31` | `0.03` | Same canonical PTX unary atan kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Indexed `f32` atan2, `65536` elements, raw DLPack | `ptx_exec` | `0.35` | `0.04` | Same canonical PTX binary atan2 kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via the same reciprocal-plus-piecewise polynomial PTX approximation plus explicit quadrant correction, so accuracy is approximate |
| Indexed `f32` exp, `65536` elements, raw DLPack | `ptx_exec` | `0.31` | `0.02` | Same canonical PTX unary exp kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `ex2.approx.f32` after scaling by `log2(e)`, so accuracy is approximate |
| Indexed `f32` erf, `65536` elements, raw DLPack | `ptx_exec` | `0.31` | `0.02` | Same canonical PTX unary erf kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Indexed `i32` scalar-broadcast bitor, `65536` elements, raw DLPack | `ptx_exec` | `0.45` | `0.13` | Same canonical PTX scalar-broadcast bitor kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` tensor-source scalar-broadcast bitand, `65536` elements, raw DLPack | `ptx_exec` | `0.28` | `0.03` | Same canonical PTX extent-1 tensor-source scalar-broadcast bitand kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` tensor-source scalar-broadcast bitor, `65536` elements, raw DLPack | `ptx_exec` | `0.28` | `0.03` | Same canonical PTX extent-1 tensor-source scalar-broadcast bitor kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `f32` scalar-branch select, `65536` elements, raw DLPack | `ptx_exec` | `0.32` | `0.04` | Same canonical indexed PTX `select` kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` tensor-source scalar-branch select, `65536` elements, raw DLPack | `ptx_exec` | `0.28` | `0.04` | Same canonical indexed PTX `select` kernel with an extent-1 tensor branch, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Rank-1 `f32` copy-reduce add, `65536` elements, raw DLPack | `ptx_exec` | `18.87` | `2.22` | Same exact PTX rank-1 `copy_reduce` add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Rank-1 `i32` copy-reduce xor, `65536` elements, raw DLPack | `ptx_exec` | `18.21` | `2.21` | Same exact PTX rank-1 `copy_reduce` xor kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `f32` copy-reduce add, `65536` elements, raw DLPack | `ptx_exec` | `7.11` | `0.03` | Same indexed PTX rank-1 `copy_reduce` add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Indexed `i32` copy-reduce xor, `65536` elements, raw DLPack | `ptx_exec` | `0.34` | `0.03` | Same indexed PTX rank-1 `copy_reduce` xor kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` copy-reduce add, `64x64`, raw DLPack | `ptx_exec` | `22.47` | `0.16` | Same exact PTX 2D `copy_reduce` add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Parallel 2D `f32` copy-reduce add, `64x64`, raw DLPack | `ptx_exec` | `21.91` | `0.03` | Same exact single-block PTX 2D `copy_reduce` add kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| 2D `f32` dense add, `64x64`, CUDA handles | `ptx_exec` | `0.38` | `0.15` | Same exact PTX 2D dense tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` dense add, `64x64`, CUDA handles | `ptx_exec` | `0.26` | `0.03` | Same exact single-block PTX 2D dense tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` dense add, `64x64`, CUDA handles | `ptx_exec` | `0.39` | `0.16` | Same exact PTX 2D dense tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` dense add, `64x64`, CUDA handles | `ptx_exec` | `0.28` | `0.03` | Same exact single-block PTX 2D dense tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.31` | `0.10` | Same exact PTX 2D broadcast tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `f32` broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same exact single-block PTX 2D broadcast tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.31` | `0.10` | Same exact PTX 2D broadcast tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` broadcast add, `64x64`, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same exact single-block PTX 2D broadcast tensor-binary kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` dense abs, `64x64`, CUDA handles | `ptx_exec` | `0.38` | `0.15` | Same exact PTX 2D unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` dense round, `64x64`, CUDA handles | `ptx_exec` | `0.39` | `0.17` | Same exact PTX 2D unary round-to-nearest kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via native PTX `cvt.rni.f32.f32` |
| 2D `f32` dense trunc, `64x64`, CUDA handles | `ptx_exec` | `0.40` | `0.16` | Same exact PTX 2D unary truncation kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via native PTX `cvt.rzi.f32.f32` |
| Parallel 2D `f32` dense abs, `64x64`, CUDA handles | `ptx_exec` | `0.24` | `0.03` | Same exact single-block PTX 2D unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` dense round, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D unary round-to-nearest kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; lowered via native PTX `cvt.rni.f32.f32` |
| Parallel 2D `f32` dense floor, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D unary floor kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; lowered via native PTX `cvt.rmi.f32.f32` |
| 2D `f32` dense sqrt, `64x64`, CUDA handles | `ptx_exec` | `0.42` | `0.23` | Same exact PTX 2D unary sqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `f32` dense cos, `64x64`, CUDA handles | `ptx_exec` | `0.46` | `0.18` | Same exact PTX 2D unary cosine kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; accuracy is approximate |
| 2D `f32` dense asin, `64x64`, CUDA handles | `ptx_exec` | `0.67` | `0.46` | Same exact PTX 2D unary asin kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| 2D `f32` dense acos, `64x64`, CUDA handles | `ptx_exec` | `13.45` | `0.93` | Same exact PTX 2D unary acos kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| 2D `f32` dense atan, `64x64`, CUDA handles | `ptx_exec` | `0.53` | `0.26` | Same exact PTX 2D unary atan kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| 2D `f32` dense log, `64x64`, CUDA handles | `ptx_exec` | `0.45` | `0.20` | Same exact PTX 2D unary log kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `lg2.approx.f32` times `ln(2)`, so accuracy is approximate |
| 2D `f32` dense erf, `64x64`, CUDA handles | `ptx_exec` | `0.53` | `0.31` | Same exact PTX 2D unary erf kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Parallel 2D `f32` dense sqrt, `64x64`, CUDA handles | `ptx_exec` | `0.24` | `0.03` | Same exact single-block PTX 2D unary sqrt kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| Parallel 2D `f32` dense log2, `64x64`, CUDA handles | `ptx_exec` | `0.29` | `0.03` | Same exact single-block PTX 2D unary log2 kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; accuracy is approximate |
| Parallel 2D `f32` dense asin, `64x64`, CUDA handles | `ptx_exec` | `0.23` | `0.03` | Same exact single-block PTX 2D unary asin kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Parallel 2D `f32` dense acos, `64x64`, CUDA handles | `ptx_exec` | `0.37` | `0.05` | Same exact single-block PTX 2D unary acos kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Parallel 2D `f32` dense atan, `64x64`, CUDA handles | `ptx_exec` | `0.30` | `0.03` | Same exact single-block PTX 2D unary atan kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Parallel 2D `f32` dense exp, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D unary exp kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; lowered via `ex2.approx.f32` after scaling by `log2(e)`, so accuracy is approximate |
| Parallel 2D `f32` dense erf, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D unary erf kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| 2D `i32` dense abs, `64x64`, CUDA handles | `ptx_exec` | `0.36` | `0.15` | Same exact PTX 2D unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` dense abs, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `i32` dense bitnot, `64x64`, CUDA handles | `ptx_exec` | `0.39` | `0.15` | Same exact PTX 2D unary bitwise-not kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` dense bitnot, `64x64`, CUDA handles | `ptx_exec` | `0.25` | `0.03` | Same exact single-block PTX 2D unary bitwise-not kernel, but executed with direct CUDA `TensorHandle` inputs to isolate the device path |
| 2D `f32` dense rsqrt, `64x64`, CUDA handles | `ptx_exec` | `0.44` | `0.20` | Same exact PTX 2D unary rsqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; accuracy is approximate |
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
| Parallel 2D `f32` tensor-factory bundle, `64x64`, CUDA handles | `ptx_exec` | `0.26` | `0.02` | Same exact single-block PTX tensor-factory bundle, but executed with direct CUDA `TensorHandle` outputs to isolate the device path |
| Parallel 2D `i32` reduction add bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `25.29` | `0.13` | Same exact single-block shared-memory PTX reduction-add bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` reduction mul bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same exact single-block shared-memory PTX reduction-mul bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` reduction max bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `0.21` | `0.02` | Same exact single-block shared-memory PTX reduction-max bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` reduction min bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `0.32` | `0.02` | Same exact single-block shared-memory PTX reduction-min bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Parallel 2D `i32` reduction xor bundle, `64x64 -> (1,) + (64,) + (64,)`, CUDA handles | `ptx_exec` | `23.51` | `0.12` | Same exact single-block shared-memory PTX reduction-xor bundle, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| 2D `i32` tensor-factory bundle, `64x64`, CUDA handles | `ptx_exec` | `27.93` | `0.05` | Same exact PTX tensor-factory bundle, but executed with direct CUDA `TensorHandle` outputs to avoid host staging |
| Parallel 2D `i32` tensor-factory bundle, `64x64`, CUDA handles | `ptx_exec` | `0.28` | `0.02` | Same exact single-block PTX tensor-factory bundle, but executed with direct CUDA `TensorHandle` outputs to isolate the device path |
| Direct `f32` sqrt, `128` elements | `ptx_exec` | `129.88` | `0.17` | Direct `threadIdx.x` PTX unary sqrt path |
| Direct `f32` sqrt, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary sqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `f32` rsqrt, `128` elements | `ptx_exec` | `308.19` | `0.19` | Direct `threadIdx.x` PTX unary rsqrt path; uses `rsqrt.approx.f32`, so accuracy is approximate |
| Direct `f32` exp2, `128` elements | `ptx_exec` | `132.63` | `0.10` | Direct `threadIdx.x` PTX unary exp2 path; uses `ex2.approx.f32`, so accuracy is approximate |
| Direct `f32` asin, `128` elements | `ptx_exec` | `133.16` | `0.08` | Direct `threadIdx.x` PTX unary asin path; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Direct `f32` acos, `128` elements | `ptx_exec` | `532.36` | `0.17` | Direct `threadIdx.x` PTX unary acos path; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Direct `f32` atan, `128` elements | `ptx_exec` | `133.07` | `0.10` | Direct `threadIdx.x` PTX unary atan path; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Direct `f32` log10, `128` elements | `ptx_exec` | `126.18` | `0.09` | Direct `threadIdx.x` PTX unary log10 path; lowered via `lg2.approx.f32` times `log10(2)`, so accuracy is approximate |
| Direct `f32` erf, `128` elements | `ptx_exec` | `126.12` | `0.09` | Direct `threadIdx.x` PTX unary erf path; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Direct `f32` rsqrt, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary rsqrt kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; accuracy is approximate |
| Direct `f32` exp2, `128` elements, CUDA handles | `ptx_exec` | `0.28` | `0.03` | Same direct PTX unary exp2 kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; accuracy is approximate |
| Direct `f32` asin, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary asin kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Direct `f32` acos, `128` elements, CUDA handles | `ptx_exec` | `11.87` | `0.02` | Same direct PTX unary acos kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Direct `f32` atan, `128` elements, CUDA handles | `ptx_exec` | `0.27` | `0.02` | Same direct PTX unary atan kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Direct `f32` log10, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary log10 kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via `lg2.approx.f32` times `log10(2)`, so accuracy is approximate |
| Direct `f32` erf, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary erf kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Direct `f32` abs, `128` elements | `ptx_exec` | `128.42` | `0.09` | Direct `threadIdx.x` PTX unary absolute-value path |
| Direct `f32` round, `128` elements | `ptx_exec` | `122.85` | `0.09` | Direct `threadIdx.x` PTX unary round-to-nearest path; lowered via native PTX `cvt.rni.f32.f32` |
| Direct `f32` ceil, `128` elements | `ptx_exec` | `124.65` | `0.09` | Direct `threadIdx.x` PTX unary ceil path; lowered via native PTX `cvt.rpi.f32.f32` |
| Direct `f32` abs, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `f32` round, `128` elements, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same direct PTX unary round-to-nearest kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via native PTX `cvt.rni.f32.f32` |
| Direct `f32` ceil, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary ceil kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging; lowered via native PTX `cvt.rpi.f32.f32` |
| Direct `f32` abs, `128` elements, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same direct PTX unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `f32` round, `128` elements, raw DLPack | `ptx_exec` | `0.25` | `0.02` | Same direct PTX unary round-to-nearest kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rni.f32.f32` |
| Direct `f32` ceil, `128` elements, raw DLPack | `ptx_exec` | `0.24` | `0.02` | Same direct PTX unary ceil kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via native PTX `cvt.rpi.f32.f32` |
| Direct `f32` neg, `128` elements | `ptx_exec` | `126.33` | `0.09` | Direct `threadIdx.x` PTX unary negation path |
| Direct `f32` neg, `128` elements, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same direct PTX unary negation kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `f32` neg, `128` elements, raw DLPack | `ptx_exec` | `0.26` | `0.02` | Same direct PTX unary negation kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `f32` exp2, `128` elements, raw DLPack | `ptx_exec` | `0.30` | `0.03` | Same direct PTX unary exp2 kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; accuracy is approximate |
| Direct `f32` asin, `128` elements, raw DLPack | `ptx_exec` | `0.23` | `0.02` | Same direct PTX unary asin kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core, so accuracy is approximate |
| Direct `f32` acos, `128` elements, raw DLPack | `ptx_exec` | `12.23` | `0.06` | Same direct PTX unary acos kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `sqrt(max(0, 1 - x*x))` plus the explicit PTX `atan2` core with the square-root term as `y`, so accuracy is approximate |
| Direct `f32` atan, `128` elements, raw DLPack | `ptx_exec` | `0.31` | `0.03` | Same direct PTX unary atan kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit reciprocal-plus-piecewise polynomial PTX approximation, so accuracy is approximate |
| Direct `f32` log10, `128` elements, raw DLPack | `ptx_exec` | `0.25` | `0.02` | Same direct PTX unary log10 kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via `lg2.approx.f32` times `log10(2)`, so accuracy is approximate |
| Direct `f32` erf, `128` elements, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same direct PTX unary erf kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles; lowered via an explicit polynomial-plus-`ex2.approx.f32` approximation, so accuracy is approximate |
| Direct `i32` abs, `128` elements | `ptx_exec` | `127.12` | `0.11` | Direct `threadIdx.x` PTX unary absolute-value path |
| Direct `i32` abs, `128` elements, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same direct PTX unary absolute-value kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `i32` abs, `128` elements, raw DLPack | `ptx_exec` | `0.27` | `0.02` | Same direct PTX unary absolute-value kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `i32` neg, `128` elements | `ptx_exec` | `125.79` | `0.13` | Direct `threadIdx.x` PTX unary negation path |
| Direct `i32` neg, `128` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same direct PTX unary negation kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `i32` neg, `128` elements, raw DLPack | `ptx_exec` | `0.27` | `0.02` | Same direct PTX unary negation kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `f32` scalar-branch select, `128` elements | `ptx_exec` | `126.44` | `0.14` | Direct `threadIdx.x` PTX `select` from an `i1` predicate tensor with a scalar-parameter true branch |
| Direct `f32` scalar-branch select, `128` elements, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same direct PTX `select` kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `f32` scalar-branch select, `128` elements, raw DLPack | `ptx_exec` | `0.25` | `0.03` | Same direct PTX `select` kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `i32` scalar-broadcast bitand, `128` elements | `ptx_exec` | `227.12` | `0.11` | Direct `threadIdx.x` PTX scalar-parameter bitwise-and broadcast path |
| Direct `i32` scalar-broadcast bitand, `128` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same direct PTX scalar-parameter bitwise-and broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `i32` scalar-broadcast bitand, `128` elements, raw DLPack | `ptx_exec` | `0.27` | `0.03` | Same direct PTX scalar-parameter bitwise-and broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `i32` scalar-broadcast bitor, `128` elements | `ptx_exec` | `232.24` | `0.11` | Direct `threadIdx.x` PTX scalar-parameter bitwise-or broadcast path |
| Direct `i32` scalar-broadcast bitor, `128` elements, CUDA handles | `ptx_exec` | `0.25` | `0.02` | Same direct PTX scalar-parameter bitwise-or broadcast kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `i32` scalar-broadcast bitor, `128` elements, raw DLPack | `ptx_exec` | `0.26` | `0.03` | Same direct PTX scalar-parameter bitwise-or broadcast kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `i32` bitand, `128` elements | `ptx_exec` | `222.52` | `0.15` | Direct `threadIdx.x` PTX bitwise-and path |
| Direct `i32` bitand, `128` elements, CUDA handles | `ptx_exec` | `0.23` | `0.02` | Same direct PTX bitwise-and kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `i32` bitand, `128` elements, raw DLPack | `ptx_exec` | `0.28` | `0.03` | Same direct PTX bitwise-and kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |
| Direct `i32` bitor, `128` elements | `ptx_exec` | `128.97` | `0.16` | Direct `threadIdx.x` PTX bitwise-or path |
| Direct `i32` bitor, `128` elements, CUDA handles | `ptx_exec` | `0.24` | `0.02` | Same direct PTX bitwise-or kernel, but executed with direct CUDA `TensorHandle` inputs to avoid host staging |
| Direct `i32` bitor, `128` elements, raw DLPack | `ptx_exec` | `0.27` | `0.03` | Same direct PTX bitwise-or kernel, but executed through stream-aware raw CUDA DLPack normalization instead of explicit tensor handles |

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
