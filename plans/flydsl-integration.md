# FlyDSL Integration Plan

This document describes how to integrate FlyDSL into Baybridge without replacing Baybridge's frontend or portability layer.

## Recommendation

Integrate FlyDSL as an optional AMD-native backend and lowering/runtime substrate.

Do not make FlyDSL a second Baybridge frontend.

Reasoning:
- Baybridge already owns the user-facing CuTe-compatible API, tracing model, portable IR, runtime fallback, and backend selection.
- FlyDSL already owns a Python frontend, AST rewrite path, embedded MLIR runtime, and FLIR/fly-dialect lowering stack.
- Trying to route Baybridge users through FlyDSL's Python authoring API would duplicate the frontend and split the programming model.
- The clean seam is Baybridge `PortableKernelIR` -> Fly/FLIR-compatible lowering.

## Current Baybridge Seams

Baybridge already has the backend shape we need:
- `Backend.lower(ir, target) -> LoweredModule`
- optional `build_launcher(...)` for executable backends
- backend auto-selection in `src/baybridge/compiler.py`

Current backend families:
- textual staging: `mlir_text`, `gpu_text`
- direct executable AMD path: `hipcc_exec`
- family-specific AMD path: `hipkittens_ref`, `hipkittens_exec`

FlyDSL fits naturally as:
- `flydsl_ref`: inspectable/reference lowering
- `flydsl_exec`: executable AMD lowering through FlyDSL/FLIR

## What FlyDSL Gives Us

From the current FlyDSL docs and repo:
- Python DSL plus MLIR-native compiler stack through FLIR and the `fly` dialect
- explicit layout algebra and coordinate mapping in the dialect
- embedded MLIR Python runtime, so no separate upstream MLIR wheel is required
- DLPack/Torch tensor adaptation at the host boundary
- explicit `Stream` argument support at the JIT boundary
- AMD-facing low-level primitives in the expression layer, including buffer ops and MFMA-oriented helpers

This makes FlyDSL a good backend substrate for:
- layout-heavy elementwise and tiled-copy kernels
- AMD-native copy/memory pipelines
- MFMA-backed GEMM families
- eventually some attention/norm kernels if the semantic mapping is exact

## What Not To Do

Do not:
- transpile Baybridge Python source into FlyDSL Python source
- mix Baybridge tracing with FlyDSL AST rewriting in the same compilation path
- replace Baybridge `PortableKernelIR` with Fly-specific objects globally
- make FlyDSL mandatory for Baybridge installation

Those choices would collapse the architecture and make Baybridge dependent on a second frontend/compiler stack for all users.

## Integration Architecture

Target architecture:

`Baybridge frontend -> PortableKernelIR -> Fly bridge lowering -> Fly/FLIR -> GPU/ROCDL/AMDGPU runtime`

Concretely:
1. Keep Baybridge tracing exactly as-is.
2. Add a Baybridge-to-Fly bridge layer that lowers a subset of `PortableKernelIR` into either:
   - FlyDSL Python builder calls, or preferably
   - direct MLIR/fly dialect text/modules if FlyDSL exposes that path cleanly.
3. Let FlyDSL own the rest of the lowering pipeline and executable runtime for supported kernels.
4. Keep `hipcc_exec` and `hipkittens_exec` alive for narrow high-value paths and as fallback/reference points.

## Mapping Strategy

### 1. Layout Algebra

Map Baybridge layout ops to Fly layout ops first.

High-confidence mappings:
- `make_layout`
- `make_ordered_layout`
- `size`
- `cosize`
- `composition`
- `coalesce`
- `logical_product`
- `zipped_product`
- `tiled_product`
- `flat_product`
- `logical_divide`
- `zipped_divide`
- `tiled_divide`
- `flat_divide`
- `select`
- `group`
- `recast_layout`
- identity layout/tensor construction

This should be the first integration surface because FlyDSL's strongest differentiator is explicit layout IR.

### 2. Execution Topology And Launch

Map Baybridge topology and launch ops to FlyDSL/MLIR GPU constructs:
- `thread_idx`
- `block_idx`
- `block_dim`
- `grid_dim`
- `barrier`
- `stream` launch parameter

Keep Baybridge cluster and TMEM extras out of the first FlyDSL slice unless FlyDSL has a direct AMD-native equivalent.

### 3. Tensor And Memory Model

Initial supported tensor model for FlyDSL backend:
- contiguous/global tensors
- shared-memory tensors
- register fragments only where the mapping is structurally exact
- DLPack/TensorHandle interop at host boundary

Map Baybridge ops in this order:
- `load`
- `store`
- `copy`
- `copy_async` only if the FlyDSL path gives a real async AMD lowering, not a fake no-op
- `make_tensor` for global/shared/register address spaces
- `local_tile`
- `local_partition`

### 4. MMA / GEMM

Do not start with arbitrary Baybridge `mma` coverage.

Start with a deliberately narrow contract:
- only exact Baybridge `mma` forms that correspond to FlyDSL's AMD MFMA abstractions
- only dtypes and tile shapes we can validate on `mi355` and `mi300`
- only non-transposed forms first unless FlyDSL gives explicit transposed kernels we can prove correct

This mirrors the successful HipKittens integration pattern: exact semantic match first, expand later.

## Proposed Backend Stack

### Phase 0: Reference Backend

Add `flydsl_ref`.

Purpose:
- prove the mapping from Baybridge IR to Fly concepts
- emit inspectable Fly/FLIR-oriented text
- no execution requirement yet

Deliverables:
- `src/baybridge/backends/flydsl_ref.py`
- backend selection entry in `compiler.py`
- tests for layout/tensor/topology family matching
- lowered module dialect name such as `flydsl_mlir` or `flydsl_ref`

### Phase 1: Executable Elementwise Backend

Add `flydsl_exec` for the simplest kernels first:
- scalar elementwise
- bounded copies
- basic shared-memory staging

Requirements:
- installed FlyDSL package or checkout
- version/root detection similar to HipKittens gating
- launcher creation through FlyDSL's compilation/runtime flow

This phase is where we validate the host boundary and executable contract.

### Phase 2: Layout-Heavy Tiled Backend

Extend `flydsl_exec` to:
- `zipped_divide`
- `local_partition`
- thread/value composed tiles
- copy atoms that map cleanly to Fly copy abstractions

This is the phase where FlyDSL can start replacing parts of `hipcc_exec` for structured kernels.

### Phase 3: MFMA/GEMM Families

Add a narrow GEMM family on top of FlyDSL:
- exact tile shapes only
- exact dtype combinations only
- validated independently on `gfx950` and `gfx942`

Do not auto-prefer this backend until:
- correctness is stable
- compile latency is characterized
- performance is at least competitive with the current `hipkittens_exec`/`hipcc_exec` fallback for its covered subset

### Phase 4: Auto-Selection

Only after the executable path is stable:
- auto-prefer `flydsl_exec` for kernels whose semantic family is a better fit for FlyDSL than `hipcc_exec`
- keep `hipkittens_exec` preferred for the GEMM tiles where it remains stronger and already proven

## Baybridge Changes Required

### Backend API

Minimal required changes:
- none for `Backend.lower(...)`
- likely add optional backend capability hooks similar to `HipKittensExecBackend.available(...)`
- add environment detection like:
  - `BAYBRIDGE_FLYDSL_ROOT`
  - or importable `flydsl`

### New Backend Files

Planned files:
- `src/baybridge/backends/flydsl_ref.py`
- `src/baybridge/backends/flydsl_exec.py`
- `tests/test_backend_flydsl_ref.py`
- `tests/test_backend_flydsl_exec.py`

Possible helper layer:
- `src/baybridge/backends/flydsl_bridge.py`

That helper should own the Baybridge-IR-to-Fly mapping logic so the backends stay thin.

### Compiler Selection Logic

Planned selection policy:
1. explicit backend always wins
2. `hipkittens_exec` keeps precedence for the exact GEMM families it already owns
3. `flydsl_exec` becomes eligible for matching elementwise/tiled/layout-heavy kernels when installed and healthy
4. otherwise `hipcc_exec` / textual backends remain fallback

## Validation Strategy

Every phase should be validated on:
- local `.venv` unit tests
- `mi355` TheRock venv
- `mi300` TheRock venv

Validation buckets:
- reference lowering correctness
- executable correctness
- backend selection correctness
- compile cache behavior
- stream handling
- DLPack/TensorHandle interop

Performance validation should start only after correctness stabilizes.

## Risk Register

### Risk 1: Dual Frontend Confusion

If we expose FlyDSL authoring directly through Baybridge, the project will have two frontends.

Mitigation:
- keep FlyDSL behind backend names only
- keep Baybridge API the only user-facing entrypoint

### Risk 2: Semantic Drift

Baybridge `PortableKernelIR` may not capture everything FlyDSL wants structurally for some advanced kernels.

Mitigation:
- begin with elementwise/layout-heavy subsets
- add bridge-only metadata extensions only when necessary
- do not contort Baybridge IR around one backend too early

### Risk 3: Toolchain Weight

FlyDSL includes its own MLIR stack and build expectations.

Mitigation:
- make FlyDSL optional
- gate `flydsl_exec` on explicit availability
- preserve existing `hipcc_exec` and HipKittens backends

### Risk 4: Redundant Backend Investment

FlyDSL and HipKittens may overlap for some AMD-native kernels.

Mitigation:
- use HipKittens for narrow proven GEMM families
- use FlyDSL where layout algebra and more general MLIR lowering are the better fit

## Recommended Kickoff Order

1. Add `flydsl_ref` only.
2. Prove Baybridge layout algebra lowers cleanly into Fly concepts.
3. Add a tiny `flydsl_exec` elementwise kernel family.
4. Add shared-memory/tiled-copy subset.
5. Reassess whether GEMM should go through FlyDSL or remain HipKittens-first.

## Exit Criteria For First Integration Milestone

- `flydsl_ref` exists and lowers a meaningful Baybridge subset
- `flydsl_exec` runs at least one non-trivial kernel family on `mi355` and `mi300`
- backend selection is deterministic and root/version-aware
- Baybridge frontend remains unchanged for users
- FlyDSL is optional, not required

## Sources

- FlyDSL README: https://github.com/ROCm/FlyDSL/blob/main/README.md
- FlyDSL kernel authoring guide: https://github.com/ROCm/FlyDSL/blob/main/docs/kernel_authoring_guide.md
- FlyDSL layout system guide: https://github.com/ROCm/FlyDSL/blob/main/docs/layout_system_guide.md
