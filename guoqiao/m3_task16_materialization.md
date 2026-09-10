# Task 16 distributed materialization implementation

**Goal:** Materialize exactly the solved SSA layouts and selected consumer
conversions while preserving coherent Tensor/function/SCF types and rollback.

**Architecture:** Preflight the solution, resolve types by original Value, and
snapshot each selected use as owner plus operand index. Rebuild one detached
ModuleOp, verify native MLIR invariants and concrete layout relations, then
transfer its body into the existing root. Original IR is untouched on failure.

## Completed plan

- [x] Add real pipeline RED tests for distributed types/constants, if/for/while,
  and a verified two-consumer transpose contract forcing one conversion.
- [x] Add `LayoutTypeConverter` with explicit Value-aware conversion; reject
  tensors in the inherited Type-only conversion path.
- [x] Implement SCF collection with direct expected-slot conversion edges.
- [x] Rebuild typed operands/results/block arguments/regions/functions and
  DenseElements constants into a detached module using IRMapping.
- [x] Copy native properties, discardable attributes, locations, successors,
  and scalar slots. Include captured values when scheduling non-dominance
  printed block order; preserve each block's original operation order.
- [x] Insert only solution-selected conversions immediately before their
  consumer, without changing the global producer mapping or choosing layouts.
- [x] Verify staged MLIR and every actual materialized Tensor boundary/relation
  before the final body transfer; preserve M2 storage checks and rollback.
- [x] Add focused rollback, Value-vs-Type, replay, missing conversion, and
  programmatically forced valid SCF conversion tests.

## Public interfaces and lifetime

`LayoutTypeConverter(graph, solution)` derives from MLIR TypeConverter;
`convertLayoutBearingTensor(Value)` resolves an original SSA definition only.
Its borrowed graph and solution must outlive converter use. Synthetic consumer
variables are not producer definitions; function-result variables are resolved
separately by FuncOp and signature index.

`materializeDistributedLayouts(root, graph, solution, conversions)` checks that
the conversion argument matches the selected solution. `materializeLayouts`
delegates to it for both storage-only and distributed graphs. The supported
transaction root is ModuleOp; other roots diagnose before mutation. The
existing root object, module attributes, and symbols remain stable.

After successful materialization, graph/solution Value, Operation and OpOperand
bindings point into erased original IR and must not be dereferenced. Destroying
the borrowed containers is safe. Every staged lookup occurs before commit.

## SCF semantics (correction to Task 16's original shorthand)

- if: result is the join slot; each branch yield use may convert to it.
- for: body iter argument and result are hard equal; init/yield uses target
  that slot. Induction variable, bounds, and step remain scalar and unchanged.
- while has **two tuples**, which may differ in arity and type:
  init/before-argument/after-yield and condition-forwarded/after-argument/result.
  The predicate is excluded from forwarded tuple indices. There is no invented
  equality between unrelated before/after positions. Conversion sites are
  before while, before condition, and before after-region yield.

Direct existing expected variables avoid redundant synthetic join nodes. The
bootstrap eight-variable/component and four-candidate/domain limits remain.
The while regression with three selected edges also re-solves the encoded
output within the eight-variable bound.

## Correctness boundaries

Defined function signatures are inferred from entry Values and synthetic result
variables, preserving explicit encodings and argument/result attributes. Tensor
calls and external tensor signatures retain Task 15's explicit unsupported
diagnostics; scalar functions and scalar CFGs remain supported.

Transpose preserves its scalar payload region and permutation; its DPS init
use and result receive coherent types. Dense tensor constants reshape their
ElementsAttr to the result type without changing logical data. Unsupported
non-dense tensor constants retain the collector's diagnostic.

The materialized verifier recollects with `LayoutCollectionMode::RelationsOnly`
(no target enumeration or relation-domain candidate projection), then pins each definition,
synthetic use, and function-result variable to the encoding actually present
in IR. It performs no solving or propagation. In particular, a Convertible
relation cannot imagine a missing future cast; differing actual endpoint types
without an explicit conversion fail solved-relation verification.

## Verification commands and observed RED

Initial `cmake --build build --target check-frisk --parallel 32`: 14 existing
tests passed; three new tests failed with missing distributed materialization
or missing SCF model. Initial unit RED: DenseConstantAndReplay failed at the
temporary guard; MaterializedVerifierRejectsUnusedTensor accepted unresolved IR.

The scalar-CFG capture regression subsequently failed with `unmapped SSA
dependency during layout rebuilding`; scheduling both explicit operands and
region captures fixed that exact failure.

Final gates:

```bash
cmake --build build --target FriskTransforms FriskUnitTests check-frisk --parallel 32
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
git diff --check
```

Lit tests run with `-verify-each` and repeat inference to check replay. Unit
tests additionally exercise chosen if/for/while conversions on verifier-valid
unencoded IR, invalid/missing/duplicate/identity edges, late rebuild rollback,
staged semantic-verifier rollback, and a missing transpose conversion which
native MLIR verification alone cannot detect. A relations-only collection test
also observed a behavior RED (unencoded definitions incorrectly acquired
candidates) before the collection-only early return was implemented.
