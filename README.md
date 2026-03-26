# baybridge

`baybridge` is an in-progress AMD-first kernel DSL and portability layer.

Today it provides:
- a Python frontend with an example-friendly `import baybridge as cute` surface
- a portable kernel IR
- textual/reference lowering backends (`mlir_text`, `gpu_text`, `gpu_mlir`, `waveasm_ref`, `aster_ref`, `hipkittens_ref`, and `ptx_ref`)
- executable HIP lowering for a narrow validated subset
- an executable PTX backend for a narrow validated subset via the CUDA driver JIT
- an optional executable HipKittens backend for a narrow BF16 GEMM subset
- a reference runtime for basic examples and `@jit` wrapper execution

It does not yet provide:
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
- `backend="ptx_ref"`
  - lowers a narrow exact Baybridge subset to driver-JIT-loadable PTX text
  - current validated subset:
    - canonical indexed rank-1 dense copy:
      - `f32`
      - `i32`
      - `i1`
    - canonical indexed rank-1 dense pointwise:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
      - `i1`: `and/or/xor`
    - canonical indexed rank-1 unary `neg/abs`:
      - `f32`
      - `i32`
    - canonical indexed rank-1 integer `bitnot`:
      - `i32`
      - `i1`
    - canonical indexed rank-1 compare-to-`i1`:
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
    - canonical indexed rank-1 scalar-broadcast compare-to-`i1` from:
      - a scalar kernel parameter
      - a rank-1 extent-1 tensor
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
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
    - canonical indexed rank-1 unary:
      - `f32`: `round/floor/ceil/trunc/sqrt/rsqrt/sin/cos/acos/asin/atan/exp/exp2/log/log2/log10/erf`
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
    - direct `threadIdx.x` rank-1 dense pointwise:
      - `f32`: `add/sub/mul/div/max/min/atan2`
      - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
      - `i1`: `and/or/xor`
    - direct `threadIdx.x` rank-1 unary `neg/abs`:
      - `f32`
      - `i32`
    - direct `threadIdx.x` rank-1 integer `bitnot`:
      - `i32`
      - `i1`
    - direct `threadIdx.x` rank-1 compare-to-`i1`:
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
    - direct `threadIdx.x` rank-1 scalar-broadcast compare-to-`i1` from:
      - a scalar kernel parameter
      - a rank-1 extent-1 tensor
      - `f32`: `cmp_lt/le/gt/ge/eq/ne`
      - `i32`: `cmp_lt/le/gt/ge/eq/ne`
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
    - direct `threadIdx.x` rank-1 unary:
      - `f32`: `abs`, `neg`, `round`, `floor`, `ceil`, `trunc`, `sqrt/rsqrt`, `sin`, `cos`, `acos`, `asin`, `atan`, `exp`, `exp2`, `log`, `log2`, `log10`, `erf`
      - `i32`: `abs`, `neg`
    - direct `threadIdx.x` scalar broadcast from either:
      - a scalar kernel parameter
      - a rank-1 extent-1 tensor
      - supported ops:
        - `f32`: `add/sub/mul/div/max/min/atan2`
        - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
    - exact current launch contract for direct rank-1 families:
      - `grid=(1,1,1)`, `block=(block.x >= extent,1,1)`
    - exact rank-1 `copy_reduce`:
      - `bb.copy(copy_reduce_atom, src, dst)`
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
        - `cmp_lt/le/gt/ge/eq/ne`
      - exact current launch contracts:
        - serial: `grid=(1,1,1)`, `block=(1,1,1)`
        - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
    - exact 2D `f32/i32` tensor-source scalar-broadcast bundles from rank-1 extent-1 tensors:
      - `dst.store(src.load() <op> alpha[0])`
      - supported ops:
        - `f32`: `add/sub/mul/div/max/min/atan2`
        - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
      - exact current launch contracts:
        - serial: `grid=(1,1,1)`, `block=(1,1,1)`
        - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
    - exact 2D `f32/i32` tensor-source scalar-broadcast compare-to-`i1` bundles from rank-1 extent-1 tensors:
      - `dst.store(src.load() <cmp> alpha[0])`
      - supported ops:
        - `cmp_lt/le/gt/ge/eq/ne`
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
    - exact 2D `f32/i32` dense compare-to-`i1` bundles:
      - `dst.store(lhs.load() <cmp> rhs.load())`
      - supported ops:
        - `cmp_lt/le/gt/ge/eq/ne`
      - exact current launch contracts:
        - serial: `grid=(1,1,1)`, `block=(1,1,1)`
        - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
    - exact 2D dense `select` bundles from `i1` predicate tensors:
      - `dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))`
      - currently validated on:
        - dense `f32`
        - broadcast-compatible `i32`
      - exact current launch contracts:
        - serial: `grid=(1,1,1)`, `block=(1,1,1)`
        - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
    - exact 2D scalar-branch `select` bundles from `i1` predicate tensors:
      - `dst.store(bb.where(pred.load(), src.load(), alpha))`
      - `dst.store(bb.where(pred.load(), alpha[0], src.load()))`
      - scalar-branch forms from:
        - a scalar kernel parameter
        - a rank-1 extent-1 tensor
      - currently validated on:
        - `f32` scalar-kernel-parameter branches
        - `i32` extent-1-tensor scalar branches
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
    - exact 2D `f32/i32` broadcast compare-to-`i1` bundles:
      - `dst.store(lhs.load() <cmp> rhs.load())`
      - broadcast-compatible `lhs` and `rhs` shapes expanded to the destination shape
      - supported ops:
        - `cmp_lt/le/gt/ge/eq/ne`
      - exact current launch contracts:
        - serial: `grid=(1,1,1)`, `block=(1,1,1)`
        - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
    - exact 2D broadcast `select` bundles from `i1` predicate tensors:
      - `dst.store(bb.where(pred.load(), lhs.load(), rhs.load()))`
      - broadcast-compatible `lhs` and `rhs` shapes expanded to the predicate/destination shape
      - currently validated on:
        - `i32`
      - exact current launch contracts:
        - serial: `grid=(1,1,1)`, `block=(1,1,1)`
        - parallel: `grid=(1,1,1)`, `block=(power_of_two,1,1)`, `block.x >= cols`
    - exact 2D unary bundles:
      - `f32`:
        - `dst.store(abs(src.load()))`
        - `dst.store(-src.load())`
        - `dst.store(bb.sqrt(src.load()))`
        - `dst.store(bb.rsqrt(src.load()))`
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
      - dense `select` from `i1` predicate tensors:
        - tensor-vs-tensor branches:
          - validated dense `f32`
          - validated broadcast `i32`
        - scalar-branch forms:
          - validated `f32` from scalar kernel params
          - validated `i32` from rank-1 extent-1 tensors
      - broadcast tensor-binary:
        - `f32`: `add/sub/mul/div/max/min/atan2`
        - `i32`: `add/sub/mul/div/max/min`, `bitand`, `bitor`, `bitxor`
        - `i1`: `bitand/bitor/bitxor`
      - broadcast compare-to-`i1`:
        - `f32`: `cmp_lt/le/gt/ge/eq/ne`
        - `i32`: `cmp_lt/le/gt/ge/eq/ne`
      - broadcast `select` from `i1` predicate tensors:
        - validated on `i32`
      - dense unary:
        - `f32`: `abs`, `neg`, `round`, `floor`, `ceil`, `trunc`, `sqrt/rsqrt`, `sin`, `cos`, `acos`, `asin`, `atan`, `exp`, `exp2`, `log`, `log2`, `log10`, `erf`
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
- `backend="ptx_exec"`
  - launches the same exact PTX subset through `libcuda.so.1`
  - no `nvcc`, `nvrtc`, or `ptxas` is required in the backend path
  - works with:
    - Baybridge `RuntimeTensor` values through host staging
    - CUDA `TensorHandle` values with real device pointers
    - raw CUDA DLPack-capable tensor objects
  - `compile(...)` auto-selects `ptx_exec` on `NvidiaTarget` when the sample args show a device-resident tensor path
  - when all tensor sample args are staged `RuntimeTensor` values, `compile(...)` now prefers `ptx_ref` and warns how to request `backend=\"ptx_exec\"` explicitly if you still want correctness-first staged execution
  - for PTX reduction-style kernels, any staged `RuntimeTensor` tensor sample args now keep auto-selection on `ptx_ref`; Baybridge only auto-selects `ptx_exec` there when tensor sample args are fully device-resident
  - warns once per built launcher when tensor arguments are staged through host memory
  - reuses per-argument device allocations across repeated staged `RuntimeTensor` launches, but the staged path is still primarily host-copy bound in current measurements
  - when you pass `stream=...`, DLPack-capable inputs are normalized with that launch stream before the PTX backend runs
  - current measured PTX performance is materially better with direct CUDA `TensorHandle` inputs or raw CUDA DLPack objects; staged `RuntimeTensor` execution remains correctness-first and host-copy dominated
  - current representative raw CUDA DLPack timings are effectively on top of the direct `TensorHandle` path for the same PTX kernels
  - scalar kernel parameters are supported for the exact PTX scalar-broadcast family
  - boolean `i1` tensors are now validated on the exact PTX data path for:
    - rank-1 copy
    - rank-1 boolean pointwise `and/or/xor`
    - rank-1 boolean `bitnot`
    - serial, exact single-block parallel, and exact row-tiled multiblock 2D copy
    - serial, exact single-block parallel, and exact row-tiled multiblock 2D tensor-binary/broadcast-binary `bitand/bitor/bitxor`
  - boolean `i1` output tensors are now validated for the exact rank-1, serial 2D, and exact parallel 2D row-tiled compare families in both `ptx_ref` and `ptx_exec`, including scalar-broadcast compare-to-`i1`
  - boolean `i1` predicate tensors are now also validated as inputs to the exact PTX `select` family in both rank-1 and 2D forms:
    - rank-1 direct and canonical indexed forms
    - serial and exact parallel 2D row-tiled tensor-select forms:
      - dense `f32`
      - broadcast `i32`
    - serial and exact parallel 2D row-tiled scalar-branch select forms:
      - `f32` from scalar kernel params
      - `i32` from rank-1 extent-1 tensors
    - rank-1 direct and canonical indexed select still remains validated for:
      - `f32`
      - `i32`
      - `i1`
  - current PTX native math stays driver-only and libdevice-free:
    - exact rank-1 unary now exists for:
      - `f32`: `abs`, `neg`, `round`, `floor`, `ceil`, `trunc`, `sqrt`, `rsqrt`, `sin`, `cos`, `acos`, `asin`, `atan`, `exp`, `exp2`, `log`, `log2`, `log10`, `erf`
      - `i32`: `abs`, `neg`
    - exact 2D serial/single-block-parallel/row-tiled unary now exists for:
      - `f32`: `abs`, `neg`, `round`, `floor`, `ceil`, `trunc`, `sqrt`, `rsqrt`, `sin`, `cos`, `acos`, `asin`, `atan`, `exp`, `exp2`, `log`, `log2`, `log10`, `erf`
      - `i32`: `abs`, `neg`, `bitnot`
      - `i1`: `bitnot`
    - `round` lowers to native PTX `cvt.rni.f32.f32`
    - `floor` lowers to native PTX `cvt.rmi.f32.f32`
    - `ceil` lowers to native PTX `cvt.rpi.f32.f32`
    - `trunc` lowers to native PTX `cvt.rzi.f32.f32`
    - `sqrt` lowers to native PTX `sqrt.rn.f32`
    - `rsqrt` lowers to native PTX `rsqrt.approx.f32`
    - `sin` lowers to native PTX `sin.approx.f32`
    - `cos` lowers to native PTX `cos.approx.f32`
    - `acos` lowers via `sqrt(max(0, 1 - x*x))` plus the same explicit `atan2` core with the square-root term as `y`; no `libdevice` is used
    - `asin` lowers via `sqrt(max(0, 1 - x*x))` plus the same explicit `atan2` core; no `libdevice` is used
    - `atan` lowers via an explicit reciprocal-plus-piecewise polynomial PTX approximation; no `libdevice` is used
    - `atan2` lowers via the same reciprocal-plus-piecewise polynomial core plus explicit quadrant correction; no `libdevice` is used
    - `exp` lowers via native PTX `ex2.approx.f32` after scaling by `log2(e)`
    - `exp2` lowers to native PTX `ex2.approx.f32`
    - `log` lowers via native PTX `lg2.approx.f32` times `ln(2)`
    - `log2` lowers to native PTX `lg2.approx.f32`
    - `log10` lowers via native PTX `lg2.approx.f32` times `log10(2)`
    - `erf` lowers via an explicit PTX polynomial-plus-`ex2.approx.f32` approximation; no `libdevice` is used
  - current PTX reductions stay intentionally exact:
    - serial lowerings still exist for the traced 1D scalar-reduction form, the exact traced 2D row/column reduction form, and the exact traced 2D reduction-bundle form
    - exact single-block shared-memory lowerings now also exist for the parallel scalar-reduction family and the parallel 2D reduction-bundle family
    - exact per-output-thread lowerings now also exist for the parallel 2D row/column reduction family
    - that 2D row/column reduction family now also has an exact multiblock output-tiling path when `grid.x * block.x >= output_extent`
    - exact row+column reduction bundles without the scalar output now also exist in serial, single-block parallel, and exact multiblock output-tiled forms
  - current PTX `copy_reduce` stays intentionally exact:
    - rank-1 serial/direct/indexed
    - 2D serial
    - 2D single-block parallel
    - broader fused NVIDIA-only TMA-reduce semantics are still not claimed here
  - broader CUDA math that would require `libdevice` is still intentionally deferred
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
      - canonical linear indexed pointwise binary ops of the form `block_idx.x * block_dim.x + thread_idx.x` are also hardware-validated today:
        - add, sub, mul, div
    - 1D `f32` copy
    - 1D `f32` unary math bundle through the copy-atom/register path
      - hardware-validated today:
        - exp, log, cos, erf
        - exp2, log2, log10, sqrt
        - sin
        - rsqrt
    - 2D `f32` broadcast binary through the row-slice/copy-atom path when `grid == block == (1, 1, 1)`
      - hardware-validated today: add, sub, mul, div
    - 2D `f32` reduction bundle through the row-slice/copy-atom path when `grid == block == (1, 1, 1)`
      - hardware-validated today:
        - add: full reduction to `(1,)` plus row reduction to `(M,)`
        - mul: full reduction to `(1,)` plus row reduction to `(M,)`
    - 2D `f32` unary math bundle through the row-slice/copy-atom path when `grid == block == (1, 1, 1)`
      - hardware-validated today:
        - exp, log, cos, erf
    - 2D `f32` tensor-factory bundle through the row-slice/copy-atom path when `grid == block == (1, 1, 1)`
    - 1D `f32` shared-memory staging copy when the traced kernel is exactly a shared-memory round-trip and `block.x == extent`
    - `acos`, `asin`, `atan`, and `atan2` remain outside the validated real subset because the current upstream FlyDSL pipeline does not lower the corresponding scalar libcalls cleanly on the active AMD environments
  - broader real upstream FlyDSL execution is still gated behind:
    - `BAYBRIDGE_EXPERIMENTAL_REAL_FLYDSL_EXEC=1`
  - reason:
    - Baybridge's current generic lowering still does not match FlyDSL's real buffer/tile access model
    - the validated subset uses specialized upstream-compatible lowerings
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
  - transposed F16 GEMM stays on `hipkittens_ref` on `gfx950`; current upstream HipKittens gfx950 MMA templates reject that family
  - RMSNorm stays on `hipkittens_ref` on `gfx950`; current upstream HipKittens gfx950 headers do not compile the generated RMSNorm kernel
  - supported micro tiles:
    - `f16/f16 -> f32`
    - `bf16/bf16 -> f32`
    - `(32, 16) x (16, 32) -> (32, 32)`
    - `(16, 32) x (32, 16) -> (16, 16)`
  - full tensor shapes may be larger as long as they are exact multiples of one supported micro tile and the reduction dimension is tiled accordingly
  - requires `BAYBRIDGE_HIPKITTENS_ROOT` to point at a HipKittens checkout

Default backend behavior:
- if the compile target is `cute.NvidiaTarget(...)` and no explicit backend is given, Baybridge auto-selects:
  - `ptx_exec` when the CUDA driver is available and the traced kernel matches the current exact PTX subset
  - otherwise `ptx_ref`
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

PTX executable example:

```python
artifact = cute.compile(
    indexed_add_kernel,
    a,
    b,
    c,
    backend="ptx_exec",
    target=cute.NvidiaTarget(sm="sm_80"),
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
- broad NVIDIA executable lowering beyond the current exact PTX subset

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
