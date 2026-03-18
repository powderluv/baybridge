# baybridge

`baybridge` is an in-progress AMD-focused kernel DSL and portability layer.

Today it provides:
- a Python frontend with an example-friendly `import baybridge as cute` surface
- a portable kernel IR
- textual lowering backends (`mlir_text`, `gpu_text`, `gpu_mlir`, `waveasm_ref`, `aster_ref`, and `hipkittens_ref`)
- executable HIP lowering for a narrow validated subset
- an optional executable HipKittens backend for a narrow BF16 GEMM subset
- a reference runtime for basic examples and `@jit` wrapper execution

It does not yet provide:
- HSACO production or HIP launch
- PTX-to-AMDGCN translation
- broad executable lowering for the full DSL surface

## What Works Today

These paths are usable now:
- basic `@kernel` and `@jit` authoring
- topology builtins such as `arch.thread_idx()`, `arch.block_idx()`, `block_dim()`, `grid_dim()`, `lane_id()`
- runtime tensors
- direct scalar indexing in traced kernels: `g_c[row, col] = g_a[row, col] + g_b[row, col]`
- `compile(...)` from explicit `TensorSpec` annotations
- `compile(...)` from sample runtime arguments
- `@jit` wrappers that launch a single traceable kernel
- runtime fallback for simple `@jit` wrappers that cannot be traced yet

Current boundary:
- Python control flow on traced dynamic values is not supported.
- If a launched kernel uses `if thread_idx(...) == ...` style control flow, `compile(...)` falls back to the reference runtime for `@jit` wrappers instead of producing IR.
- Direct `@kernel` compilation fails in that case rather than silently degrading.

## Setup

`baybridge` is currently tested with a repo-local `.venv`.

```bash
git clone <repo>
cd baybridge
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Run the test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
```

## Quick Start

### 1. Reference Runtime: Hello World

This runs immediately through the reference runtime.

```python
import baybridge as cute


@cute.kernel
def hello_world_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello world")


@cute.jit
def hello_world():
    cute.printf("hello world")
    hello_world_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


hello_world()

compiled = cute.compile(hello_world)
compiled()
```

Notes:
- `hello_world()` executes directly on the reference runtime.
- `compile(hello_world)` succeeds, but because the launched kernel uses dynamic Python control flow, the compiled artifact executes through the runtime path rather than producing portable IR.

### 2. Traced Kernel: Indexed Add

This kernel is traceable because it avoids dynamic Python `if`.

```python
import baybridge as cute


@cute.kernel(launch=cute.LaunchConfig(grid=(1, 1, 1), block=(4, 1, 1)))
def indexed_add_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    _, n = g_a.shape
    col = tidx % n
    row = tidx // n
    g_c[row, col] = g_a[row, col] + g_b[row, col]


a = cute.tensor([[1, 2], [3, 4]], dtype="f32")
b = cute.tensor([[10, 20], [30, 40]], dtype="f32")
c = cute.zeros((2, 2), dtype="f32")

artifact = cute.compile(indexed_add_kernel, a, b, c, backend="gpu_text")
print(artifact.ir.to_dict())
print(artifact.lowered_module.text)
```

This produces:
- portable IR containing `thread_idx`, `constant`, `mod`, `floordiv`, `load`, `add`, and `store`
- `gpu_text` output containing `memref.load` and `memref.store`

### 3. Traced Launch Wrapper

If a `@jit` wrapper only specializes and launches a traceable kernel, `compile(...)` returns the launched kernel IR.

```python
import baybridge as cute


@cute.jit
def indexed_add(m_a: cute.Tensor, m_b: cute.Tensor, m_c: cute.Tensor):
    indexed_add_kernel(m_a, m_b, m_c).launch(grid=(1, 1, 1), block=(4, 1, 1))


compiled = cute.compile(indexed_add, a, b, c, backend="portable")
print(compiled.ir.name)                 # indexed_add_kernel
print(compiled.ir.metadata["wrapped_by"])  # indexed_add
```

## Core API

### Decorators

- the `kernel` decorator
  - device-style kernel definition
  - direct `compile(...)` produces kernel IR when the body is traceable
- the `jit` decorator
  - host-style specialization/wrapper function
  - can execute directly on the reference runtime
  - can compile either to launched kernel IR or to a runtime fallback artifact

### Tensor Inputs

You can supply tensor information in two ways:

1. Explicit annotations

```python
@cute.kernel
def copy_kernel(
    src: cute.TensorSpec(shape=(128,), dtype="f16"),
    dst: cute.TensorSpec(shape=(128,), dtype="f16"),
):
    cute.copy(src, dst)
```

2. Sample runtime arguments

```python
a = cute.tensor([[1, 2], [3, 4]], dtype="f32")
b = cute.tensor([[10, 20], [30, 40]], dtype="f32")
c = cute.zeros((2, 2), dtype="f32")

artifact = cute.compile(indexed_add_kernel, a, b, c)
```

### Runtime Tensors

The reference runtime uses lightweight CPU-side tensors:

```python
import baybridge as cute

a = cute.tensor([[1, 2, 3], [4, 5, 6]], dtype="f32")
b = cute.zeros((2, 3), dtype="f32")

print(a.shape)     # (2, 3)
print(a.dtype)     # f32
print(a.tolist())  # [[1, 2, 3], [4, 5, 6]]
```

Useful helpers:
- `baybridge.tensor(data, dtype=...)`
- `baybridge.zeros(shape, dtype=...)`
- `baybridge.clone(tensor)`
- `baybridge.size(value, mode=...)`

### Backends

For a current backend inventory, validation matrix, and benchmark notes, see [docs/backend-status.md](/home/nod/github/baybridge/docs/backend-status.md).

`compile(...)` supports:
- `backend="portable"`
  - stores portable IR only
- `backend="mlir_text"`
  - lowers portable IR to a baybridge-specific textual form
- `backend="gpu_text"`
  - lowers portable IR to GPU/ROCDL-flavored textual IR
- `backend="gpu_mlir"`
  - lowers portable IR to a stricter MLIR GPU/module form suitable for external backend consumption
- `backend="waveasm_ref"`
  - lowers supported kernels to WaveASM-oriented MLIR plus tool invocation hints
  - this is a reference backend, not an executable backend
  - emits a `*.waveasm_repro/` bundle next to the lowered artifact with `kernel.mlir`, `repro.sh`, and `manifest.json`
  - the bundle path is exposed on the compiled artifact as `artifact.debug_bundle_dir`
  - you can also generate it directly with `baybridge.emit_waveasm_repro(...)` or `tools/emit_waveasm_repro.py`
  - `tools/compare_backends.py` can compile the same kernel through multiple backends, optionally execute supported ones, and report bundle/debug paths in JSON
  - pass `--include-env` to `tools/compare_backends.py` to include Python/platform/tool/version metadata for cross-machine diffs
  - sample factories used with `tools/compare_backends.py` may optionally accept `backend_name` and `target_arch` so they can return backend-specific compile/run args
  - `BAYBRIDGE_WAVEASM_ROOT` can point at a Wave checkout to improve tool discovery hints
- `backend="waveasm_exec"`
  - experimental backend behind `BAYBRIDGE_EXPERIMENTAL_WAVEASM_EXEC=1`
  - lowers a narrow standard-MLIR subset to WaveASM-consumable MLIR and launches the resulting HSACO through HIP's module API
- `backend="aster_ref"`
  - lowers supported kernels to ASTER-oriented reference MLIR plus tool/runtime hints
  - this is a reference backend, not an executable backend
  - emits a `*.aster_repro/` bundle next to the lowered artifact with `kernel.mlir`, `repro.sh`, and `manifest.json`
  - `BAYBRIDGE_ASTER_ROOT` can point at an ASTER checkout or install root to improve tool and package discovery
  - current experimental family:
    - single-global-tensor pointwise kernels
    - single-global-tensor pointwise math kernels
    - single-global-tensor shared-memory staging
  - not enabled by default because current upstream WaveASM execution still has correctness issues for Baybridge kernels, including the scalar global `memref.load/store` SRD aliasing bug tracked in `iree-org/wave#1117`
  - does not yet cover reductions, multi-buffer copy, GEMM, or Baybridge custom tensor-SSA ops
  - also emits a `*.waveasm_repro/` bundle so the exact MLIR/toolchain path can be reproduced outside Baybridge
  - the bundle path is exposed on the compiled artifact as `artifact.debug_bundle_dir`
  - requires `waveasm-translate`, `clang++` or `clang`, and prefers `ld.lld` for final HSACO linking
- `backend="flydsl_ref"`
  - lowers a supported Baybridge subset to a FlyDSL-oriented reference module
  - this is a reference backend, not an executable backend
  - if `BAYBRIDGE_FLYDSL_ROOT` points at a FlyDSL checkout, the artifact includes a direct root/build hint
- `backend="flydsl_exec"`
  - lowers a validated Baybridge tensor subset to a FlyDSL-flavored Python module and launcher scaffold
  - current supported family:
    - pointwise tensor kernels
    - broadcasted tensor binary ops
    - tensor factories via `fill`
    - simple reductions
    - shared-memory staging and whole-tensor copy
  - requires a built and importable FlyDSL environment, not just a source checkout
  - `BAYBRIDGE_FLYDSL_ROOT` should point at a checkout with `build-fly/python_packages` or `build/python_packages`
  - fake/local harness execution can launch with:
    - `baybridge.from_dlpack(...)` tensor handles
    - raw DLPack-capable tensors such as `torch.Tensor`
    - plain `baybridge.RuntimeTensor` values when `torch` is importable
  - real upstream FlyDSL execution is currently validated for a narrow subset:
    - 1D `f32` pointwise binary ops through the copy-atom/register path
      - hardware-validated today: add, sub, mul, div
    - 1D `f32` copy
    - exact generic-lowering kernels that are now hardware-validated:
      - 2D `f32` broadcast add for `(2,1) + (1,3) -> (2,3)`
      - 2D `f32` reduce bundle for `(2,3) -> (1,)` and `(2,3) -> (2,)`
      - 2D `f32` tensor-factory bundle on `(2,2)`
      - 1D `f32` math bundle on `(3,)` covering `exp`, `log`, `cos`, `erf`, and `atan2`
  - broader real upstream FlyDSL execution is still gated behind:
    - `BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC=1`
  - reason:
    - Baybridge's current generic lowering still does not match FlyDSL's real buffer/tile access model
    - the validated add/copy path uses a specialized upstream-compatible lowering
  - `tools/compare_backends.py --execute` reports that gate as `skipped_unvalidated_real_flydsl_exec`
  - intended as the first executable FlyDSL bridge, not full FlyDSL coverage yet
- `backend="hipkittens_ref"`
  - lowers matched GEMM and attention-family kernels to a HipKittens-oriented C++ reference artifact
  - this is a reference backend, not an executable backend
  - if `BAYBRIDGE_HIPKITTENS_ROOT` points at a HipKittens checkout, the artifact includes a direct include-path build hint
- `backend="hipcc_exec"`
  - builds and runs a narrow executable HIP subset through `hipcc`
- `backend="hipkittens_exec"`
  - builds and runs a narrow executable HipKittens subset
  - current supported kernel family: pure BF16/F16 GEMM composed from HipKittens micro tiles
  - currently validated for `gfx950`
  - `gfx942` executable support is currently limited to the BF16 slice; F16 exec is not wired yet
  - supported micro tiles:
    - `f16/f16 -> f32`
    - `bf16/bf16 -> f32`
    - `(32, 16) x (16, 32) -> (32, 32)`
    - `(16, 32) x (32, 16) -> (16, 16)`
  - full tensor shapes may be larger as long as they are exact multiples of one supported micro tile and the reduction dimension is tiled accordingly
  - requires `BAYBRIDGE_HIPKITTENS_ROOT` to point at a HipKittens checkout

Default backend behavior:
- if `compile(...)` is called without an explicit backend, the traced kernel matches `hipkittens_exec` on `gfx950`, and `BAYBRIDGE_HIPKITTENS_ROOT` points at a usable checkout, Baybridge auto-prefers `hipkittens_exec`
- the same auto-preference path can apply to `gfx942`, but only when the local ROCm toolchain satisfies the HipKittens `cdna3` header requirements
- otherwise, if the traced kernel matches a HipKittens tensorop GEMM or attention family, Baybridge auto-prefers `hipkittens_ref`
- otherwise, if the traced kernel matches the validated `flydsl_exec` subset and the active FlyDSL environment is ready, Baybridge auto-prefers `flydsl_exec` only when:
  - the inputs match the validated path
  - and the kernel falls into either:
    - the currently validated real-upstream subset
    - or the broader experimental subset with `BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC=1`
- otherwise it falls back to the normal default textual backend
- if `BAYBRIDGE_EXEC_ARCH` is set, Baybridge uses that architecture as the default compile target

Example:

```python
artifact = cute.compile(indexed_add_kernel, a, b, c, backend="gpu_text")
print(artifact.lowered_module.text)
```

HipKittens reference example:

```python
artifact = cute.compile(gemm_kernel, a, b, c, backend="hipkittens_ref")
print(artifact.lowered_module.dialect)  # hipkittens_cpp
print(artifact.lowered_module.text)
```

HipKittens executable example:

```python
artifact = cute.compile(
    bf16_gemm_kernel,
    a,
    b,
    c,
    backend="hipkittens_exec",
    target=cute.AMDTarget(arch="gfx950"),
)
artifact(a, b, c)
```

## Working Examples In This Repo

These tests are the best executable documentation today:
- `tests/test_examples_runtime.py`
  - hello world
  - naive elementwise add
- `tests/test_examples_tracing.py`
  - direct indexed kernel tracing
  - traced launch wrapper behavior
- `tests/test_backend_gpu.py`
  - `gpu_text` lowering surface
- `tests/test_backend_hipkittens_ref.py`
  - HipKittens reference backend family matching
- `tests/test_backend_hipkittens_exec.py`
  - HipKittens executable BF16 GEMM path
- `tests/test_validation.py`
  - explicit unsupported cases and diagnostics

## Limitations

Not implemented yet:
- `zipped_divide`
- composition-based tiled tensor views
- slice/view objects with `.load()` / `.store()` semantics
- executable ROCm lowering
- PTX translation

Important limitations in the current tracer:
- dynamic Python `if`, `while`, and other control flow on traced scalars are rejected
- runtime fallback is only used for `@jit` wrappers
- direct kernel compilation expects the kernel body itself to be traceable

## Project Layout

- `src/baybridge/frontend.py`
  - user-facing DSL surface
- `src/baybridge/runtime.py`
  - reference runtime tensors and launch execution
- `src/baybridge/tracing.py`
  - portable IR builder and traced values
- `src/baybridge/compiler.py`
  - compile/cache flow and runtime fallback policy
- `src/baybridge/backends/mlir_text.py`
  - portable textual lowering
- `src/baybridge/backends/gpu_text.py`
  - GPU/ROCDL-flavored textual lowering
- `src/baybridge/backends/hipkittens_ref.py`
  - HipKittens-oriented reference lowering for matched kernel families
- `src/baybridge/backends/hipkittens_exec.py`
  - narrow executable HipKittens lowering for supported BF16 GEMM tiles

## Status

`baybridge` is past the scaffolding stage and is now useful for:
- trying the frontend
- validating portability decisions
- exercising a reference runtime
- inspecting the portable IR and textual AMD-oriented lowerings

It is not yet a drop-in ROCm execution backend for full upstream DSL programs.
