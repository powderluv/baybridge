# baybridge

`baybridge` is an in-progress AMD-focused kernel DSL and portability layer.

Today it provides:
- a Python frontend with an example-friendly `import baybridge as cute` surface
- a portable kernel IR
- textual lowering backends (`mlir_text` and `gpu_text`)
- executable HIP lowering for a narrow validated subset
- a reference runtime for basic examples and `@jit` wrapper execution

It does not yet provide:
- executable ROCm code generation
- HSACO production or HIP launch
- PTX-to-AMDGCN translation

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

`compile(...)` supports:
- `backend="portable"`
  - stores portable IR only
- `backend="mlir_text"`
  - lowers portable IR to a baybridge-specific textual form
- `backend="gpu_text"`
  - lowers portable IR to GPU/ROCDL-flavored textual IR

Example:

```python
artifact = cute.compile(indexed_add_kernel, a, b, c, backend="gpu_text")
print(artifact.lowered_module.text)
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

## Status

`baybridge` is past the scaffolding stage and is now useful for:
- trying the frontend
- validating portability decisions
- exercising a reference runtime
- inspecting the portable IR and textual AMD-oriented lowerings

It is not yet a drop-in ROCm execution backend for full upstream DSL programs.
