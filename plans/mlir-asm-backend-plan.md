# MLIR-to-ASM Backend Plan

This document chooses and sequences the MLIR-to-assembly backend work for Baybridge.

## Recommendation

Use `waveasm` as Baybridge's first general MLIR-to-ASM backend.

Use `aster` later as an expert-mode backend for selected hot kernel families.

Reasoning:
- Baybridge already owns the frontend, tracing, portability layer, runtime fallback, and backend selection.
- Baybridge needs a backend that can accept MLIR close to the level Baybridge can realistically emit first.
- `waveasm` already positions itself as a backend compiler that accepts `gpu.func` plus common MLIR dialects and emits AMDGCN assembly and HSACO.
- `aster` is stronger for explicit low-level kernel construction, scheduling, and partial register-control work, but it currently sits lower than Baybridge's first practical integration seam.

## Decision Summary

Primary choice:
- `waveasm`

Secondary choice:
- `aster`

Do not:
- build Baybridge around both as competing general backends
- delay a practical `waveasm` path while building a full Baybridge-to-ASTER lowering stack first
- route Baybridge users through Wave DSL or ASTER authoring APIs directly

Keep Baybridge's contract as:
- `frontend -> PortableKernelIR -> backend lowering -> executable artifact`

## Why WaveASM First

WaveASM is the better first backend because it is closer to Baybridge's likely lowering level.

Based on the upstream WaveASM README:
- it is a backend compiler for Wave DSL and a general MLIR-to-AMDGCN path
- it accepts:
  - `gpu.func`
  - `gpu`, `arith`, `vector`, `memref`, `scf`, and `amdgpu` ops
  - or pre-lowered WaveASM IR
- it already includes:
  - AMDGCN assembly emission
  - linear-scan register allocation
  - hazard and wait-state handling
  - HSACO generation through `clang` and `lld`
- it lists `gfx942` and `gfx950` among supported targets

This is the shortest path from Baybridge's current state to a real MLIR-to-assembly backend.

## Why ASTER Later

ASTER is the better expert backend, not the first general one.

Based on the upstream ASTER README:
- it is a very young assembly/codegen system focused on AMD GPUs
- it exposes a Python-first programming model and runtime integration
- it provides:
  - a low-level `amdgcn` MLIR dialect
  - scheduling primitives
  - partial register allocation control
  - assembly emission
  - HSACO generation
- it explicitly calls out future work around connections to higher-level MLIR and end-to-end compilers

That makes ASTER a good fit for:
- hand-tuned nanokernels
- explicit MFMA/LDS/waitcnt scheduling work
- expert GEMM/attention/norm kernels

It is not the best first Baybridge-wide MLIR backend because Baybridge would have to lower too much of its current mid-level semantics directly into ASTER's low-level model.

## Baybridge Integration Architecture

Target architecture for the first MLIR-to-ASM backend:

`Baybridge frontend -> PortableKernelIR -> real MLIR GPU/module lowering -> WaveASM -> ASM/HSACO -> Baybridge runtime`

Target architecture for the expert backend later:

`Baybridge frontend -> PortableKernelIR -> selected expert lowering -> ASTER AMDGCN dialect -> ASM/HSACO -> Baybridge runtime`

This keeps Baybridge's frontend and backend-selection model intact.

## Evaluation Matrix

`waveasm`
- first general backend fit: high
- lowering distance from Baybridge: lower
- current MLIR input compatibility: high
- Python/runtime integration: medium
- expert low-level control: medium
- best use: first executable MLIR-to-ASM backend

`aster`
- first general backend fit: medium
- lowering distance from Baybridge: higher
- current MLIR input compatibility: lower for Baybridge's first path
- Python/runtime integration: high
- expert low-level control: high
- best use: later backend for selected hot kernels

## Phase Plan

### Phase 0: Real MLIR Emitter In Baybridge

Goal:
- stop treating `gpu_text` as only debug text
- produce a real MLIR module at the `gpu.func` / `arith` / `vector` / `memref` / `scf` / `amdgpu` level

Deliverables:
- a new Baybridge MLIR emitter layer
- stable textual MLIR output that upstream tools can parse
- test coverage for:
  - pointwise kernels
  - shared-memory staging
  - reductions
  - broadcasted tensor ops

Exit criteria:
- the emitted MLIR parses in the chosen external backend toolchain
- Baybridge can round-trip representative kernels through textual MLIR artifacts without losing semantics

### Phase 1: `waveasm_ref`

Goal:
- add an inspectable reference backend that lowers Baybridge kernels to WaveASM-consumable MLIR and emits the exact tool invocation recipe

Deliverables:
- `src/baybridge/backends/waveasm_ref.py`
- environment detection for a Wave checkout or installed WaveASM tools
- lowered module dialect like `waveasm_mlir`
- tests that validate:
  - target detection for `gfx942` and `gfx950`
  - pointwise lowering
  - shared-memory lowering
  - reduction lowering

Do not execute kernels yet in this phase.

### Phase 2: `waveasm_exec`

Goal:
- compile Baybridge MLIR through WaveASM to assembly and HSACO, then launch through Baybridge's runtime

Deliverables:
- `src/baybridge/backends/waveasm_exec.py`
- build pipeline:
  - Baybridge MLIR -> `waveasm-translate` -> `.s`
  - `.s` -> HSACO through `clang` / `lld`
  - HSACO load/launch through Baybridge runtime
- validation on:
  - `mi300` / `gfx942`
  - `mi355` / `gfx950`

First supported family:
- pointwise tensor kernels
- bounded load/store kernels
- simple shared-memory staging
- simple reductions

Do not start with GEMM.

### Phase 3: Auto-Selection For WaveASM

Goal:
- allow Baybridge to auto-prefer `waveasm_exec` where it is a better fit than `hipcc_exec`

Rules:
- keep auto-selection opt-in until correctness is stable
- compare compile latency and runtime performance against:
  - `hipcc_exec`
  - `flydsl_exec`
- do not override `hipkittens_exec` for GEMM families it already covers well

Initial auto-preference family:
- pointwise and reduction kernels when WaveASM is installed and healthy

### Phase 4: `aster_ref`

Goal:
- add ASTER as a second, explicit backend for expert kernels and assembly-centric experimentation

Deliverables:
- `src/baybridge/backends/aster_ref.py`
- selected lowering from Baybridge IR into ASTER's lower-level AMDGCN model
- tests for:
  - exact MFMA fragment contracts
  - LDS scheduling-oriented kernels
  - norm or attention microkernels where ASTER's control is useful

Scope this narrowly.

Do not attempt a full Baybridge-wide ASTER lowering in the first slice.

### Phase 5: `aster_exec`

Goal:
- execute a narrow ASTER-backed kernel family through Baybridge

First supported family should be one of:
- MFMA microkernel used by a Baybridge GEMM path
- a norm kernel with explicit LDS and register behavior
- another hot kernel where explicit scheduling materially matters

Exit criteria:
- correctness on `gfx942` and `gfx950`
- stable build/runtime pipeline
- measurable value over the simpler backends

## Integration Rules

### Use WaveASM For
- first general MLIR-to-assembly path
- kernels already expressible as `gpu.func` plus common MLIR dialects
- bringing Baybridge's textual MLIR path closer to a real executable backend

### Use ASTER For
- hot kernels where Baybridge needs direct low-level control
- expert MFMA/LDS scheduling work
- kernels where hand-tuned instruction-level structure matters enough to justify a dedicated lowering

### Do Not Use Either For
- replacing Baybridge's frontend
- replacing the existing `hipkittens_exec` path for already-proven GEMM families without a measured win
- broad auto-selection before correctness and environment detection are solid

## Recommended File Plan

WaveASM:
- `src/baybridge/backends/waveasm_ref.py`
- `src/baybridge/backends/waveasm_exec.py`
- `src/baybridge/backends/waveasm_bridge.py`
- `tests/test_backend_waveasm_ref.py`
- `tests/test_backend_waveasm_exec.py`

ASTER:
- `src/baybridge/backends/aster_ref.py`
- `src/baybridge/backends/aster_exec.py`
- `src/baybridge/backends/aster_bridge.py`
- `tests/test_backend_aster_ref.py`
- `tests/test_backend_aster_exec.py`

## Recommended Kickoff Order

1. Build the real MLIR emitter Baybridge needs for external backend consumption.
2. Add `waveasm_ref`.
3. Add `waveasm_exec` for pointwise/shared-memory/reduction kernels.
4. Benchmark `waveasm_exec` against `hipcc_exec` and `flydsl_exec`.
5. Add `aster_ref` only for one selected expert kernel family.
6. Add `aster_exec` only after the ASTER path proves value on that family.

## Exit Criteria

WaveASM path is successful when:
- Baybridge can compile and run non-trivial kernels through WaveASM on `mi300` and `mi355`
- the path is deterministic and toolchain-aware
- pointwise/shared-memory/reduction kernels have solid coverage

ASTER path is successful when:
- Baybridge can drive at least one hot kernel family through ASTER end-to-end
- the ASTER path produces a measurable benefit in control or performance that simpler backends cannot match
- ASTER stays a narrow expert backend unless Baybridge later grows a dedicated low-level lowering layer

## Sources

- WaveASM README:
  - https://github.com/iree-org/wave/tree/main/waveasm
- Raw WaveASM README:
  - https://raw.githubusercontent.com/iree-org/wave/main/waveasm/README.md
- WaveASM e2e utility:
  - https://raw.githubusercontent.com/iree-org/wave/main/waveasm/waveasm_e2e.py
- ASTER README:
  - https://github.com/iree-org/aster
- Raw ASTER README:
  - https://raw.githubusercontent.com/iree-org/aster/main/README.md
