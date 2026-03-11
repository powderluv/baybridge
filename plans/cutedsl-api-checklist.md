# CuTe DSL API Checklist

This checklist tracks the remaining CuTe DSL API surface Baybridge should implement or deliberately replace with an AMD-native equivalent.

It separates three things:
- sample-corpus portability
- broader documented CuTe API parity
- NVIDIA-specific features that should be mapped selectively instead of cloned blindly

## Current Snapshot

Current third-party sample matrix after the latest analyzer run:

- `blocked_missing_api`: `0`
- `blocked_external_dependency`: `15`
- `partially_covered_by_baybridge_tests`: `9`
- `covered_by_baybridge_tests`: `2`

Interpretation:

- Baybridge no longer has generic missing-symbol blockers for the selected upstream CuTe sample corpus.
- The main remaining blockers are framework/runtime integration and semantic depth.
- The next work should target API areas that unlock direct sample execution, not just new symbol names.

## Phase 1: Finish The Practical Core

- [x] Add public tensor factories:
  - `full`
  - `empty_like`
  - `ones_like`
  - `zeros_like`
- [x] Add `TensorSSA.broadcast_to`
- [x] Add implicit tensor broadcasting rules for binary tensor ops
- [x] Add `cutlass.range`
- [x] Add `cutlass.range_constexpr`
- [x] Add loop pipelining controls such as `prefetch_stages`
- [x] Add generic tiled-copy constructors:
  - `make_tiled_copy`
  - `make_tiled_copy_S`
  - `make_tiled_copy_D`
  - `make_tiled_copy_C_atom`
  - `make_cotiled_copy`
- [x] Add generic atom constructors:
  - `make_atom`
  - `make_mma_atom`
- [x] Expand the public math surface:
  - `cos`
  - `exp`
  - `log`
  - `log2`
  - `log10`
  - `erf`
  - `atan2`
  - fast-math controls

## Phase 2: Deepen The Object Model

- [ ] Expand `TiledCopy` object parity:
  - richer slice APIs
  - retile semantics
  - more CuTe-like partition helpers
  - better introspection
- [ ] Expand `TiledMma` object parity:
  - fuller fragment constructors
  - richer shape/introspection APIs
  - closer CuTe slice semantics
- [x] Add `local_partition`
- [ ] Add missing layout helper depth where the docs/examples rely on it instead of Baybridge-native substitutes

## Phase 3: Complete `cute.arch`

- [x] Add cluster-level builtins:
  - `arch.cluster_dim`
  - `arch.cluster_idx`
  - `arch.cluster_size`
  - `arch.block_idx_in_cluster`
  - `arch.block_rank_in_cluster`
  - `arch.block_in_cluster_idx`
  - `arch.block_in_cluster_dim`
- [x] Add warp/block collectives:
  - `arch.sync_warp`
  - `arch.elect_one`
- [x] Add shared-memory/runtime queries:
  - `arch.get_dyn_smem_size`
- [x] Add the missing barrier/TMEM objects:
  - `MbarrierArray`
  - `NamedBarrier`
  - `TmaStoreFence`
  - `TmemAllocator`
- [ ] Add fuller `mbarrier_*` and TMA helper semantics where they have a portable or AMD-native mapping

## Phase 4: Framework And Runtime Integration

- [ ] Add direct `torch.Tensor` call paths
- [x] Add stronger `from_dlpack` and pointer-backed interop coverage
- [ ] Add `cuda-python` compatibility shims where the examples depend on them
- [ ] Add CUDA-graphs-equivalent integration or a Baybridge-native substitute
- [ ] Add TVM FFI compile/runtime integration instead of the current gap
- [ ] Replace minimal `testing` shims with more complete benchmark/autotune coverage

## Phase 5: Advanced NVIDIA Surface

- [ ] Decide which NVIDIA-only APIs should stay compatibility shims vs receive true semantics
- [ ] Add missing warpgroup / WGMMA parity only if there is a concrete backend path
- [ ] Add fuller tcgen05 / TMEM depth only if it maps to an AMD-native execution strategy
- [ ] Add TMA-reduce and other advanced copy paths only if they unlock real kernels we care about

## HipKittens / AMD-Native Follow-Ons

- [ ] Add a HipKittens reference family for `rotary` on `main`
- [ ] Consider a narrow executable norm path only if it maps directly to real HipKittens kernels
- [ ] Expand executable GEMM coverage only where Baybridge semantics still match exactly
- [ ] Keep the AMD-native path ahead of CUDA-compat emulation whenever the two conflict

## Suggested Kickoff Order

1. Tensor factories plus `TensorSSA.broadcast_to`
2. `cutlass.range` / `cutlass.range_constexpr` / pipelined loop forms
3. Generic tiled-copy and atom constructors
4. Math surface expansion
5. `cute.arch` cluster and barrier/TMEM completion
6. Framework/runtime integration
7. Remaining NVIDIA-only advanced surface

## Exit Criteria

- [ ] The selected upstream sample corpus has no `blocked_missing_api` entries
- [ ] The remaining blocked entries are either external integrations or deliberate non-goals
- [ ] The Baybridge public API can cover the common CuTe low-level authoring flow without sample-specific shims
- [ ] AMD-native backends remain the default path; compatibility layers do not become the core architecture

## References

- CuTe API: https://docs.nvidia.com/cutlass/4.3.5/media/docs/pythonDSL/cute_dsl_api/cute.html
- CuTe arch API: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html
- CuTe control flow: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html
- CuTe API changelog: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/changelog.html
- Framework integration: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/framework_integration.html
- TVM FFI: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html
