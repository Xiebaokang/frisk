# M3 execution notes

Base: `f65c54d`; branch: `feature/m3-distributed-layout`.
The approved scope is Tasks 14–17 in `layout_inference_implementation_plan.md`.

## Implementation clarifications

- Identity `convert_layout` is valid input and is removed by canonicalization.
  The materializer rejects a solver request to insert an identity conversion.
- This MLIR checkout has no `tensor.transpose`; Tensor permutation uses the
  registered `linalg.transpose` operation, with its destination-style operand
  and result kept consistent. No unregistered placeholder operation is used.
- Bootstrap inference remains bounded to 8 variables and 4 candidates per
  connected component/domain respectively; conversion count precedes stable
  assignment/edge order. Full CostVector and runtime GPU validation stay in M5/M6.
- IR remains public RankedTensor SSA. The test-only conversion adapter may use
  per-thread vector carriers internally to expose shuffle/shared exchange.
- StorageAccess legality is compatible logical domains and valid `S(D(h))`,
  not coalescing. Unencoded linear/transpose consumers may share one layout;
  separate incompatible encoded-consumer tests exercise conversion selection.
- SCF while uses the actual MLIR two-tuple contract: init/before/yield versus
  condition/after/results, whose scalar arities may differ. Collection of those
  region constraints belongs to Task 16 alongside coherent type rewriting.
- The test lowering envelope is uniform kernel entry-block execution, full
  32-lane warps, matching known block dimensions, and one CTA per tile. The
  Tensor/vector representation bridges are not executable runtime lowering.
- Shuffle payloads preserve signless i8/i16/i32/i64 and f16/bf16/f32/f64 bits
  through i32 words. Locally preferred replicas and globally unique scratch
  writers are distinct ownership decisions. Scratch uses a conservative
  adapter budget (48 KiB by default), not the SM90 hardware maximum.

## Progress

- [x] Task 14: Tensor carriers, conversion verification and canonicalization.
  Commit `f87034b`; task review spec compliant / quality approved; 11 lit,
  38 unit and 4 CTest cases passed. Symmetric invalid-source encoding regression
  added after review; focused `--verify-diagnostics --canonicalize` passed.
  Plan checklist updates are retained as required project progress documentation.
- [x] Task 15: Distributed constraints, candidates, propagation and edge solving.
  Commits `6198f5f`, `53d573c`; task review approved after named-transpose and
  stable-use-key fixes. 49 unit, 14 lit and 4 CTest cases passed.
- [x] Task 16: Tensor/SCF type conversion and selected-edge materialization.
  Commits `34bbd89`, `11d55ab`; task review approved after canonical-design
  synchronization and public API diagnostic fixes. Full pre-fix gate: 18 lit,
  60 unit, 4 CTest; focused fix gate: 18 lit and 12 materialization units.
- [x] Task 17: Conversion cleanup and static single-CTA test lowering.
  Commit `6138f59`; task review spec compliant / quality approved. Parent fresh
  gate: 27 lit, 72 unit, 4 CTest and multi-consumer inference/cleanup passed.
  Additional review follow-up covers XOR/warp-group ownership, a live consumer
  integration fixture, and clarification that split-input-file is optional.
  Follow-up gate: 27 lit, 74 unit and 4 CTest; the live 2x2 transpose retains
  exactly one conversion and byte-identical inference/cleanup replay.
- [x] Final review and M3 Gate.
  Whole-branch review of `f65c54d..87408c9` approved, no Critical/Important or
  new actionable Minor findings. Parent independently rebuilt the relevant
  targets and verified 27 lit, 74 unit and 4 CTest cases, the original unsplit
  M3 command, and byte-identical infer/cleanup replay; all exited 0.

## Final verification and handoff

```bash
cmake --build build --target FriskTransforms FriskLayoutToGPU check-frisk \
  FriskLayoutUnitTests frisk_attr_test frisk_reduce_layout_test \
  frisk_layout_pass_test frisk_memory_effect_test --parallel 32
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
build/bin/frisk-opt test/Transforms/multi-consumer-layout.mlir \
  -frisk-infer-layouts -frisk-optimize-layout-conversions -verify-each
git diff f65c54d..HEAD --check
```

The implementation remains in the local `feature/m3-distributed-layout` branch
at `/home/baopeihua/frisk/.worktrees/m3-distributed-layout`; the main worktree is
unchanged at `f65c54d`. M4 begins with Task 18; formal CostVector and executable
GPU/runtime gates remain M5/M6. Pre-existing negative-unit diagnostic output
and CMake CMP0116 configuration noise are nonblocking maintenance items, not
test failures or M3 regressions.

No M3 integration into main or remote publication has been performed.
