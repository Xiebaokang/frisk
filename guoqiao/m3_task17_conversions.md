# Task 17: conversion cleanup and test GPU adapter

## Interfaces and implementation

`createOptimizeLayoutConversionsPass()` registers
`frisk-optimize-layout-conversions`. The pass uses ConvertLayoutOp's shared
canonicalization: canonical map checks and exact SSA type guards for identity/
inverse folds, and adjacent composition preserving the final exact tensor type.
An intermediate conversion remains when it has other consumers. It introduces
no representation casts.

`createTestLowerLayoutConversionsPass()` registers
`frisk-test-lower-layout-conversions`, with `scratch-budget=49152` by default.
The FriskLayoutToGPU library also exposes test-only `planRedistribution`,
`planPayload`, and overflow-safe `reserveScratch` in
`Conversion/FriskLayoutToGPU/LayoutConversionPlan.h`. Unit tests and emitter
consume the SAME RedistributionPlan and PayloadTransportPlan.

The planner verifies static encoded types, canonical BitLinear maps and output
names/order, computes `R = rightInverse(Dsrc) compose Ddst`, checks
`Dsrc compose R == Ddst`, and exhaustively validates bounded carrier ownership.
Input names determine bit offsets; omitted names mean replicated carriers.
Physical order is thread-major/register-minor, with
`thread = 32*(warp_group*warp_extent+warp)+lane`. Logical scratch slots are
tensor row-major coordinates, not the map's concatenated output bit index.

For each destination, choose the smallest physical thread/register owner in
the same thread, otherwise same warp, otherwise the proven R owner. The
global scratch writer table is separate: exactly one right-inverse owner per
logical element. Shared writes NEVER use per-destination replica preferences.
The emitter uses finite i32 lookup vectors from the proved plan, avoiding a
second interpretation of named layouts or an invalid claim that local replica
selection is always a linear map.

Register-only plans use local vector extraction. Shuffle plans execute every
possible source-register shuffle at all lanes, THEN select based on the
destination's requested source register. i8/i16/i32/i64/f16/bf16/f32/f64 are
bitcast/zero-extended/split to i32 word payloads, shuffled and reconstructed;
no numeric floating-point conversion occurs. Narrow elements are not packed
together. Shared exchange uses original-dtype workgroup attribution with
`llvm.align = element_bytes`, conditional uniquely elected stores, one uniform
barrier, then all destination loads. Each conversion gets distinct scratch.

## Explicit supported envelope

- Conversion is a direct entry-block op of a single-block `gpu.func` kernel.
  Nested SCF/CFG scopes and helper/host functions are rejected.
- Two execution topologies match: lane=32, CTA=1, at most 1024 physical threads,
  known block size `[32*warp_extent*warp_group_extent,1,1]`. Register counts may
  differ, each at most 256. One CTA is the communication scope per tile;
  different launched blocks may independently execute the kernel.
- Static logical volume ≤65536 and combined source/destination carrier visits
  ≤262144; conservative potential shuffle-word count ≤65536. These are adapter
  size limits, not target capability claims. Other integer widths, float formats,
  index, sub-byte and non-scalar element types are rejected.
- Scratch budget is an adapter limit, NOT SM90's hardware maximum. Account for
  original-dtype bytes, natural/inter-buffer alignment, pre-existing static
  identity-layout workgroup attributions, and all new buffers per kernel.
  Dynamic/non-attributed workgroup results or workgroup function arguments are
  conservatively rejected, including aliases whose footprint is not proved.
- All preflight and aggregate-budget checks occur before ANY mutation. Later
  invalid conversions leave the entire module unchanged. No loop scratch reuse
  is supported; adding loops later requires a post-read reuse barrier/proof.

Task15 Convertible is broader than this envelope. Missing/mismatched block
metadata, differing topology, lane!=32, cross-CTA, unsupported type, or unknown
shared footprint must diagnose rather than implicitly widen the adapter.
Malformed/dynamic distributed tensor IR is already rejected by M1 before a
pass runs; programmatic planner negatives cover dynamic types separately.

## Tensor representation and M6 responsibility

Public SSA remains encoded RankedTensor with the original exact result type.
The test emitter's `builtin.unrealized_conversion_cast` bridges Tensor to the
current thread's `vector<register_extent x element_type>` and back. These are
an explicit non-executable test ABI, not real tensor reshapes or completed
Tensor lowering. The M6 runtime harness must supply/consume register vectors
via actual tile load/store transport and eliminate these bridges before GPU
execution. M3 proves bounded ownership/payload semantics and valid GPU-level
IR; no runtime GPU correctness or performance claim is made.

## Regression evidence and gates

Initial lit RED: 18 old tests passed and 3 new tests failed because both passes
were absent. After the planner API compiled with an empty stub, 4/5 planner
tests failed behavior assertions; implementation made all five pass. Later
behavior REDs caught a missing diagnostic for absent encodings and unaccounted
dynamic shared storage. Their regressions now require explicit errors.

The focused suite contains 11 tests. It enumerates every owner for selected
register/lane/warp/replicated layouts, compares all 720 six-bit permutations
with named-input column reordering, simulates i32-word transport including all
256 i8 patterns and floating NaN/negative-zero payloads, and tests global writer
uniqueness, row-major slots, pure register permutation, byte/alignment budgets,
preflight rollback, and exact canonical-equivalent-but-distinct tensor types.
FileCheck covers all eight types, lane-dependent 64-bit register selection,
same-thread/warp replica locality, shared f16/f64 exchange, aggregate and
pre-existing-buffer budgets, and unsupported launch/control-flow/shared memory.

```bash
cmake --build build --target FriskTransforms FriskLayoutToGPU FriskLayoutUnitTests check-frisk --parallel 32
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter=ConversionPlanTest.*
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
build/bin/frisk-opt test/Transforms/multi-consumer-layout.mlir \
  --split-input-file -frisk-infer-layouts -frisk-optimize-layout-conversions -verify-each
git diff --check
```

The split flag is required because the multi-consumer fixture contains multiple
independent modules. Textual tests use `-verify-each` or expected diagnostics;
no verification-disabling escape hatch is used. Scope not covered: GPU runtime,
full Tensor-to-LLVM lowering, arbitrary/dynamic topology, cluster communication,
loop/barrier reuse, packed sub-byte types, optimized lookup/code generation,
and formal proofs beyond the explicitly bounded enumeration.

Observed final gate before commit: 27/27 lit, 11/11 focused conversion tests,
72/72 full layout unit tests, and 4/4 CTest tests pass; the split-input M3
inference+cleanup command and `git diff --check` exit 0. Existing negative unit
tests intentionally print diagnostics while passing, and CMake retains its
pre-existing CMP0116 deprecation warning.
