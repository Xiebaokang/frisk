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

## Progress

- [x] Task 14: Tensor carriers, conversion verification and canonicalization.
  Commit `f87034b`; task review spec compliant / quality approved; 11 lit,
  38 unit and 4 CTest cases passed. Symmetric invalid-source encoding regression
  added after review; focused `--verify-diagnostics --canonicalize` passed.
  Plan checklist updates are retained as required project progress documentation.
- [ ] Task 15: Distributed constraints, candidates, propagation and edge solving.
- [ ] Task 16: Tensor/SCF type conversion and selected-edge materialization.
- [ ] Task 17: Conversion cleanup and static single-CTA test lowering.
- [ ] Final review and M3 Gate.

No M3 integration into main or remote publication has been performed.
