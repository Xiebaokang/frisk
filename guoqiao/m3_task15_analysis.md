# Task 15 distributed analysis contract

Task 15 implements analysis and edge selection; Task 16 owns tensor/SCF
materialization. `--frisk-infer-layouts=analysis-only` collects, propagates,
solves and verifies without changing IR. `dump-analysis` prints deterministic
graph IDs, domain sizes and selected consumer-use keys. Without analysis-only,
distributed materialization currently fails explicitly before changing IR.

## Implemented checklist

- [x] Real SSA distributed variables and exact per-consumer conversion edges.
- [x] StorageAccess for tile_load/store; coalescing remains soft, not legality.
- [x] Registered arith/math elementwise models, dense tensor constants,
  tensor.empty, and linalg.transpose with its DPS operand.
- [x] Hard existing encodings; explicit conversions are not reinserted.
- [x] Defined function argument/result/return bindings; unknown tensor ops,
  tensor calls and external tensor signatures fail explicitly.
- [x] Blocked, lane-striped, warp-striped and fully-replicated candidate
  templates, each verified by the existing distributed type verifier.
- [x] Generate alternatives to closure before monotone fixed-point pruning;
  initialize one unseeded relation domain before moving to another.
- [x] Hard feasibility, then minimum new conversion count, then stable
  assignment and use-key order; no rematerialization or performance objective.
- [x] Existing 8-variable/component and 4-candidate/domain defaults preserved;
  overflow is diagnosed, never silently truncated.
- [x] Solved verifier checks actual authorized use, encodings, nonidentity,
  duplicates, missing conversions, candidate validity and storage capacity.
- [x] Dual linear/transpose storage consumers preserve alternatives and use
  zero conversions. Independently encoded transpose-DPS consumer uses one.
- [x] Forward/reverse non-square transpose, stable reordered solutions,
  conversion-count ordering, hard limits and tampered-solution regressions.
- [ ] SCF if/for/while graph collection and coupled type rebuilding: Task 16.

## Representation and semantic boundaries

`LayoutVar::value` binds a real result/block argument. Synthetic store/input
variables have `use`; function result variables have a FuncOp `anchor` and
`functionResult` index. `graph.lookupVariable(Value)` is authoritative after
finalize; builder maps are pre-finalize only. Convertible endpoints are
directed. Where a result/join already represents the expected layout, the
consumer edge targets it directly to conserve the 8-variable budget.

`LayoutConversionEdge::constraint` refers to the finalized graph constraint;
that constraint provides `stableUseKey` and the original `OpOperand *`.
Task 16 must snapshot owning operation/operand index before replacing IR and
remap those bindings; it must not dereference erased operand pointers.

All relation users share `LayoutRelations.cpp`. SameLayout/Keep preserve exact
encoding equality, including named map metadata, so equal-layout reasoning
does not produce unequal SSA tensor types. Transform permutes logical row
blocks of a canonical bit-linear map; inverse projection uses the inverse
permutation. Transform compatibility uses the actual destination candidate's
output labels, so source `row/column` and destination `width/height` remain
valid independent conventions. Generation prefers declared endpoint labels
as proposals and otherwise retains source positional labels; a synthetic use's
original producer type does not impose a new hard naming constraint. Different
encodings are still not silently treated as the same public tensor type by
SameLayout or Keep.

Storage access is `S(D(h))`: valid D covers the tile and valid S addresses that
same logical shape/element type. Mixed affine-storage/bit-linear-distributed
composition need not be represented as a new map attribute to prove this
domain composition. No contiguous-lane/coalescing requirement is imposed.
Replicated tile_store remains valid because its semantics require lowering to
elect one deterministic owner and write each logical element exactly once.
Legacy M2 storage-copy equality and storage capacity verification remain.

M1's bit-linear verifier is retained: nonzero-rank static power-of-two logical
extents are required; named logical extents of one are unsupported because
named zero-bit dimensions are forbidden. Unit hardware dimensions are omitted
from map inputs, while topology still records their extent one. Bootstrap
distributed candidates/transfer relations are single-CTA only. No new
cross-CTA transfer, arbitrary affine distributed map, or performance model is
introduced.
The analysis conversion relation admits verified single-CTA topologies; the
Task 17 bootstrap lowering subset must explicitly reject execution-topology
combinations it cannot implement, rather than silently mislowering them.

## Task 16 handoff

Extend `collectDistributedLayoutConstraints` before graph finalization. The
builder exposes `convertible(src, expected, use, existing=false)`,
`getOrCreateDistributedUse`, `transform`, and `storageAccess`. SCF tensor ops
currently receive the explicit unknown-model diagnostic.

For for/if slots, connect init/yield uses to existing expected result/argument
variables where possible instead of creating redundant equality-only nodes.
For while, before/init/yield and after/condition/results are distinct tuples:
their arities and types need not match. Model each tuple's coupling separately.
Materialize results, region args, terminators and function signatures together;
dense tensor constant attributes must be retyped with their result type.
Preserve explicit encodings and reject unsupported tensor calls rather than
changing unmodeled ABI uses.

After Task 16 implements materialization, replace the current preflight guard;
retain analysis-only as a read-only diagnostic/testing interface.
