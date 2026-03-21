# NVIDIA PTX Backend Plan

This document defines the implementation plan for a real NVIDIA backend in Baybridge.

## Recommendation

Do not route the NVIDIA path through `flydsl_*` or `aster_*`.

Use a Baybridge-owned PTX pipeline instead:

`frontend -> PortableKernelIR -> NVVM/PTX lowering -> PTX text -> CUDA Driver API`

Keep the only NVIDIA runtime dependency as the kernel driver:
- `libcuda.so.1` at runtime

Do not depend on:
- `nvcc`
- `nvrtc`
- `ptxas`
- CUDA headers
- CUDA source templates
- NVIDIA libraries beyond the driver

Use open toolchain pieces for code generation:
- MLIR NVVM dialect
- LLVM NVPTX backend

## Why Not FlyDSL Or ASTER

`flydsl` and `aster` are useful Baybridge backends today, but they are the wrong seam for PTX generation.

`flydsl`
- is already wired as a real AMD-focused backend
- has runtime and lowering assumptions built around ROCm/FlyDSL execution
- does not provide a credible PTX codegen surface

`aster`
- is an AMDGCN-oriented backend
- already sits near AMD assembly and MFMA-specific lowering
- would require a second backend compiler hidden inside ASTER just to reach PTX

If Baybridge wants a real NVIDIA backend, it should own:
- the target model
- the PTX lowering contract
- the CUDA driver launch path

That is the same architectural pattern Baybridge already uses for:
- `waveasm_ref` / `waveasm_exec`
- `aster_ref` / `aster_exec`
- `flydsl_ref` / `flydsl_exec`

The difference is that the NVIDIA backend should be Baybridge-native, not tunneled through an AMD-specific backend.

## Core Design

Introduce a new backend family:
- `ptx_ref`
- `ptx_exec`

Shared implementation:
- `src/baybridge/backends/ptx_bridge.py`

Optional structured intermediate backend later:
- `nvvm_ref`

Runtime support:
- `src/baybridge/cuda_driver.py` or equivalent real CUDA driver loader

Target model:
- `NvidiaTarget` or a generalized cross-vendor target abstraction extended with:
  - `sm` / compute capability
  - PTX version
  - warp size `32`

## Minimal Host Install Requirements

For Baybridge's planned NVIDIA backend, the minimal host requirement is:
- the NVIDIA kernel driver
- the user-mode CUDA driver library:
  - `libcuda.so.1`

That is enough for:
- PTX JIT loading through the CUDA Driver API
- kernel launch through the CUDA Driver API

Do not require by default:
- CUDA Toolkit
- `nvcc`
- `nvrtc`
- `ptxas`
- CUDA headers
- `cuda-runtime`
- `cuda-compat`

The backend should target PTX versions that the installed driver can JIT directly.

`cuda-compat` should stay optional and out of the default plan. If someone wants to run PTX that is newer than the installed driver understands, that is a deployment-specific compatibility problem, not a Baybridge baseline requirement.

## Minimal Installation Instructions

### General Linux Guidance

Use the official NVIDIA driver packages for the target Linux distribution.

Preferred installation method:
- distribution-specific packages from the official NVIDIA repositories

Avoid by default:
- `.run` installer unless the target environment really requires it

Reason:
- NVIDIA recommends using the distribution-specific packages where possible
- this keeps driver updates and kernel-module lifecycle inside the system package manager

Pre-installation requirements:
- supported Linux distribution
- matching kernel headers and development packages for the running kernel

### Ubuntu / Debian-Style Minimal Example

Install the kernel headers for the running kernel:

```bash
sudo apt update
sudo apt install linux-headers-$(uname -r)
```

Install an NVIDIA driver package:

```bash
sudo apt install nvidia-driver-<branch>
```

Reboot:

```bash
sudo reboot
```

Do not install for the Baybridge baseline:

```bash
# not required for the Baybridge PTX backend baseline
sudo apt install cuda
sudo apt install cuda-toolkit
sudo apt install nvidia-cuda-toolkit
```

### Driver Verification Checklist

After reboot, verify the host is usable for the planned `ptx_exec` backend.

GPU visible:

```bash
nvidia-smi
```

Expected:
- NVIDIA GPU listed
- driver version reported

CUDA driver library visible to the dynamic loader:

```bash
ldconfig -p | grep 'libcuda.so.1'
```

Expected:
- one or more entries for `libcuda.so.1`

Device nodes present:

```bash
ls -l /dev/nvidiactl /dev/nvidia0
```

Expected:
- device nodes exist

Optional sanity check from Python:

```python
import ctypes
ctypes.CDLL("libcuda.so.1")
```

Expected:
- no exception

## Baybridge Deployment Rule

For the first NVIDIA backend slices, Baybridge should treat a machine as PTX-driver-ready only when:
- `libcuda.so.1` is loadable
- at least one NVIDIA device is visible

If either check fails:
- `ptx_exec` should fail with a direct runtime diagnostic
- `ptx_ref` should still work

## Backend Contract

### `backend="ptx_ref"`

Purpose:
- emit inspectable PTX text
- emit the exact launch metadata and driver recipe
- support repro/debug bundles like the existing WaveASM and ASTER paths

Output should include:
- `.ptx`
- `manifest.json`
- `repro.py` or `repro.sh`

### `backend="ptx_exec"`

Purpose:
- load PTX directly through the CUDA driver JIT
- launch kernels through the CUDA Driver API

Required driver entry points:
- `cuInit`
- `cuDeviceGet`
- `cuCtxGetCurrent`
- `cuDevicePrimaryCtxRetain` / `cuCtxCreate` if needed
- `cuModuleLoadDataEx`
- `cuModuleGetFunction`
- `cuLaunchKernel`
- `cuGetErrorName`
- `cuGetErrorString`

Optional for later:
- `cuMemAlloc`
- `cuMemcpyHtoD`
- `cuMemcpyDtoH`
- stream/event APIs

## Phase Plan

### Phase 0: Real CUDA Driver Runtime

Goal:
- stop treating `src/cuda/bindings/*` as enough for a real NVIDIA path
- add a real driver loader that talks to `libcuda.so.1`

Deliverables:
- dynamic `ctypes` loader for CUDA Driver API
- minimal opaque handle types:
  - module
  - function
  - stream
  - context
- explicit error handling and stringification

Rules:
- no CUDA headers required at build time
- no CUDA toolkit required
- runtime should fail clearly when `libcuda.so.1` is absent

Exit criteria:
- a tiny hand-written PTX kernel can load and launch through Baybridge

### Phase 1: Target And Runtime Argument Surface

Goal:
- introduce a real NVIDIA target contract without disturbing AMD behavior

Deliverables:
- `NvidiaTarget(sm="sm_80", ptx_version="8.0")` or equivalent
- compile path that accepts a CUDA target separately from `AMDTarget`
- runtime argument normalization for CUDA device-backed tensors

First supported runtime inputs:
- DLPack-capable CUDA tensors only
  - `torch.Tensor` on CUDA
  - similar tensors if they expose DLPack and device pointers cleanly

Do not start with:
- Baybridge-owned CUDA memory allocation
- general host fallback staging

Reason:
- Baybridge already uses DLPack well elsewhere
- this avoids building a second memory manager before the backend is proven

### Phase 2: `ptx_ref` For A Minimal Portable Subset

Goal:
- directly generate PTX text for the simplest useful subset

First supported kernel families:
- dense contiguous copy
- dense pointwise binary:
  - `add`
  - `sub`
  - `mul`
  - `div`
- canonical indexed pointwise binary:
  - `blockIdx.x * blockDim.x + threadIdx.x`
- scalar broadcast on contiguous tensors
- simple reductions

Do not include yet:
- transcendental math that needs `libdevice`
- tensor cores
- cooperative launch
- dynamic shared-memory-heavy kernels

PTX generation style:
- Baybridge-owned PTX templates/emitters for the exact validated subset
- each family should have a narrow matcher like the real `flydsl_exec` validated path

Reason:
- this gets a real backend running quickly
- it keeps the codegen honest and reviewable

Exit criteria:
- `ptx_ref` emits valid PTX for the first kernel families
- bundle output is reproducible

### Phase 3: `ptx_exec`

Goal:
- execute the Phase 2 PTX subset through the CUDA driver

Deliverables:
- `cuModuleLoadDataEx` launch path
- parameter packing for scalar values and pointers
- real correctness tests against the reference runtime

Initial execution constraints:
- explicit `backend="ptx_exec"` only
- explicit `NvidiaTarget`
- CUDA DLPack inputs only

Exit criteria:
- correctness on at least one NVIDIA machine
- reproducible PTX bundle beside compiled artifacts

### Phase 4: Structured Lowering Through NVVM

Goal:
- move from exact PTX string emission for the basic families to a more scalable structured lowering

Recommended path:
- `PortableKernelIR -> MLIR NVVM/LLVM-compatible IR -> LLVM NVPTX -> PTX`

Why:
- direct PTX emission is fine for the first narrow families
- it becomes brittle for:
  - shared memory
  - advanced control flow
  - tensor core features
  - future optimization work

Deliverables:
- `nvvm_ref` or an internal NVVM lowering layer
- stable PTX generation through LLVM NVPTX

Rules:
- still no NVIDIA toolkit dependency
- LLVM/MLIR is acceptable

### Phase 5: Shared Memory And Broader Tensor Surface

Goal:
- widen the validated PTX subset without pulling in NVIDIA-only advanced features too early

Next families:
- exact shared-memory staging
- exact 2D broadcast
- exact 2D reductions
- selected unary math only after a deliberate strategy exists

Important boundary:
- math ops like `sin`, `exp`, and `log` should not be claimed until Baybridge has:
  - a `libdevice`-free implementation strategy
  - or an explicit optional toolchain path that remains acceptable to the project constraints

### Phase 6: Tensor Core Path

Goal:
- add an explicit NVIDIA tensor-core backend slice only after the scalar/tensor baseline is stable

Recommended implementation order:
- WMMA-style exact families first
- WGMMA/TMA later

Do not start with:
- WGMMA
- TMA
- graph capture integration

Reason:
- those are the NVIDIA analog of Baybridge's most specialized AMD backend paths
- they should come after the generic PTX backend is already credible

## Runtime Architecture Rules

Keep the runtime split clean:

AMD paths:
- HIP runtime
- existing AMD backends

NVIDIA path:
- CUDA driver runtime only

Do not:
- fake CUDA through HIP for the real NVIDIA backend
- reuse the current compatibility-only `src/cuda/bindings/*` layer as the execution engine
- require the CUDA toolkit just to run Baybridge PTX kernels

## Testing Plan

### Local Non-NVIDIA CI

Always test:
- PTX text emission
- matcher behavior
- bundle emission
- targeted driver-loader unit tests with fake `ctypes` handles

These do not need NVIDIA hardware.

### NVIDIA Hardware Gate

Add hardware-gated tests only for:
- `ptx_exec`
- real CUDA DLPack launch
- minimal correctness kernels

Suggested first hardware file:
- `tests/test_backend_ptx_exec.py`

Suggested early test families:
- copy
- indexed add
- scalar broadcast add
- reduction add

## Auto-Selection Policy

Do not auto-select the NVIDIA backend at first.

Phase 1 policy:
- explicit backend only

Phase 2 policy:
- optional auto-selection only when:
  - target is explicitly NVIDIA
  - runtime inputs are CUDA DLPack-capable
  - kernel matches a validated PTX family

Keep it narrower than the AMD default policy until correctness and launch stability are boring.

## Naming Recommendation

Use NVIDIA-specific backend names directly:
- `ptx_ref`
- `ptx_exec`

Do not overload:
- `flydsl_*`
- `aster_*`
- `waveasm_*`

Those names already imply specific backend ecosystems and vendor assumptions.

## First Implementation Slice

If Baybridge starts this work now, the most defensible first slice is:

1. real CUDA driver loader
2. `NvidiaTarget`
3. `ptx_ref` for:
   - copy
   - dense pointwise add/sub/mul/div
   - canonical indexed add
4. `ptx_exec` for the same exact subset using CUDA DLPack tensors

That gives Baybridge:
- a real NVIDIA execution story
- direct PTX generation
- no NVIDIA toolkit dependency
- no ASTER/FlyDSL misuse

## Decision Summary

Build the NVIDIA backend as a Baybridge-native PTX path.

Do not:
- route PTX generation through FlyDSL
- route PTX generation through ASTER

Do:
- use LLVM/MLIR NVVM/NVPTX as open infrastructure where structured lowering helps
- keep the only NVIDIA runtime dependency as the kernel driver
- start with a narrow exact PTX subset and widen from proven hardware results
