#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class CountingTarget final : public LayoutTarget {
public:
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  mutable unsigned enumerations = 0;
  bool skipEnumeration = false;
  void enumerateCandidates(const LayoutVar &var,
                           SmallVectorImpl<LayoutCandidate> &out) const override {
    ++enumerations;
    if (skipEnumeration) return;
    target->enumerateCandidates(var, out);
  }
  LogicalResult verifyCandidate(const LayoutVar &var, Attribute candidate,
                                Location loc) const override {
    return target->verifyCandidate(var, candidate, loc);
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
    return target->evaluate(a);
  }
};
class StorageAliasIntegrationTest : public testing::Test {
protected:
  StorageAliasIntegrationTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  MLIRContext context;
  std::string print(Operation *op) {
    std::string text;
    llvm::raw_string_ostream os(text);
    op->print(os);
    return text;
  }
};

TEST_F(StorageAliasIntegrationTest, InfersRootRelativeStridedSliceAndReverifies) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #root = #frisk.storage<map = #frisk.affine_layout<
      inputs = ["dim0", "dim1"], input_extents = [4, 8],
      outputs = ["byte_offset", "bit_offset"], output_extents = [128, 8],
      map = affine_map<(d0,d1) -> (32*d0+4*d1, 0)>>,
      memory_space = #frisk<memory_space Shared>, alignment = 4, vector_granularity = 4>
    func.func @f(%r: memref<4x8xi32, 3>) {
      %rv = frisk.layout_view %r {layout = #root} : memref<4x8xi32, 3> -> memref<4x8xi32, 3>
      %s = memref.subview %r[1,2] [2,3] [1,2] : memref<4x8xi32, 3> to memref<2x3xi32, strided<[8,2], offset:10>, 3>
      %v = frisk.layout_view %s : memref<2x3xi32, strided<[8,2], offset:10>, 3> -> memref<2x3xi32, strided<[8,2], offset:10>, 3>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveBootstrapLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  // A valid-shaped but wrong slice assignment must not partially bind storage.
  auto bad = *solution;
  for (const auto &var : graph->getVariables()) {
    if (cast<MemRefType>(var.shapedType).getShape() != ArrayRef<int64_t>({2, 3}))
      continue;
    auto encoding = cast<StorageLayoutAttr>(bad.assignments.lookup(var.id));
    auto map = cast<AffineLayoutMapAttr>(encoding.getMap());
    auto affine = map.getAffineMap().getValue();
    auto shifted = AffineLayoutMapAttr::get(&context, map.getInputNames(),
        map.getInputExtents(), map.getOutputNames(), map.getOutputExtents(),
        AffineMapAttr::get(AffineMap::get(2, 0,
            {affine.getResult(0) - 4, affine.getResult(1)}, &context)));
    bad.assignments[var.id] = StorageLayoutAttr::get(&context, shifted,
        encoding.getMemorySpace(), encoding.getAlignment(), encoding.getVectorGranularity());
  }
  auto original = print(*module);
  {
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(materializeLayouts(*module, *graph, bad)));
  }
  EXPECT_EQ(print(*module), original);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  StorageLayoutAttr slice;
  module->walk([&](LayoutViewOp view) { slice = view.getLayoutAttr(); });
  auto map = dyn_cast<AffineLayoutMapAttr>(slice.getMap());
  ASSERT_TRUE(map);
  Builder b(&context);
  for (auto [i, j, expected] : {std::tuple<int, int, int>{0,0,40},
                                {0,2,56}, {1,2,88}}) {
    SmallVector<Attribute> results;
    ASSERT_TRUE(succeeded(map.getAffineMap().getValue().constantFold(
        {b.getIndexAttr(i), b.getIndexAttr(j)}, results)));
    EXPECT_EQ(cast<IntegerAttr>(results[0]).getInt(), expected);
  }
  // Independent verification cannot enumerate candidates, and reconstructs
  // proofs after the IR changes instead of reusing the solved graph's cache.
  CountingTarget counted;
  ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, counted)));
  EXPECT_EQ(counted.enumerations, 0u);
  auto affine = map.getAffineMap().getValue();
  auto wrongMap = AffineLayoutMapAttr::get(&context, map.getInputNames(),
      map.getInputExtents(), map.getOutputNames(), map.getOutputExtents(),
      AffineMapAttr::get(AffineMap::get(2, 0,
          {affine.getResult(0) - 4, affine.getResult(1)}, &context)));
  auto wrong = StorageLayoutAttr::get(&context, wrongMap, slice.getMemorySpace(),
      slice.getAlignment(), slice.getVectorGranularity());
  LayoutViewOp last;
  module->walk([&](LayoutViewOp view) { last = view; });
  last->setAttr("layout", wrong);
  auto tampered = print(*module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, counted)));
  EXPECT_EQ(counted.enumerations, 0u);
  EXPECT_EQ(print(*module), tampered);
}

TEST_F(StorageAliasIntegrationTest, IndependentVerifierRejectsNonAdjacentConflict) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #a = #frisk.storage<map = #frisk.affine_layout<inputs=["d"], input_extents=[2],
      outputs=["byte_offset","bit_offset"], output_extents=[32,8], map=affine_map<(d)->(4*d,0)>>,
      memory_space=#frisk<memory_space Shared>, alignment=1, vector_granularity=1>
    #b = #frisk.storage<map = #frisk.affine_layout<inputs=["d"], input_extents=[2],
      outputs=["byte_offset","bit_offset"], output_extents=[32,8], map=affine_map<(d)->(4*d+16,0)>>,
      memory_space=#frisk<memory_space Shared>, alignment=1, vector_granularity=1>
    #c = #frisk.storage<map = #frisk.affine_layout<inputs=["d"], input_extents=[2],
      outputs=["byte_offset","bit_offset"], output_extents=[32,8], map=affine_map<(d)->(4-4*d,0)>>,
      memory_space=#frisk<memory_space Shared>, alignment=1, vector_granularity=1>
    func.func @f(%r: memref<8xi32,3>) {
      %a = memref.subview %r[0] [2] [1] : memref<8xi32,3> to memref<2xi32, strided<[1]>,3>
      %b = memref.subview %r[4] [2] [1] : memref<8xi32,3> to memref<2xi32, strided<[1], offset:4>,3>
      %c = memref.subview %r[0] [2] [1] : memref<8xi32,3> to memref<2xi32, strided<[1]>,3>
      %av = frisk.layout_view %a {layout=#a} : memref<2xi32,strided<[1]>,3> -> memref<2xi32,strided<[1]>,3>
      %bv = frisk.layout_view %b {layout=#b} : memref<2xi32,strided<[1],offset:4>,3> -> memref<2xi32,strided<[1],offset:4>,3>
      %cv = frisk.layout_view %c {layout=#c} : memref<2xi32,strided<[1]>,3> -> memref<2xi32,strided<[1]>,3>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto before = print(*module);
  auto target = createSM90LayoutTarget();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  auto graph = collectLayoutConstraints(*module, *target,
                                        LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  unsigned pairs = 0;
  for (const auto &relation : graph->getConstraints()) {
    if (relation.kind != ConstraintKind::AliasLayout) continue;
    ++pairs;
    auto lhs = relation.vars[0], rhs = relation.vars[1];
    auto a = graph->getVariable(lhs).candidates.front().value;
    auto b = graph->getVariable(rhs).candidates.front().value;
    (void)proveAliasLayoutRelation(*graph, lhs, a, rhs, b);
    auto evaluations = graph->getCandidatePreparationStatistics().pairProofEvaluations;
    (void)proveAliasLayoutRelation(*graph, rhs, b, lhs, a);
    EXPECT_EQ(graph->getCandidatePreparationStatistics().pairProofEvaluations,
              evaluations);
  }
  EXPECT_EQ(pairs, 3u);
  EXPECT_EQ(graph->getCandidatePreparationStatistics().origins, 0u);
  EXPECT_EQ(graph->getCandidatePreparationStatistics().projectedCandidates, 0u);
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target)));
  EXPECT_EQ(print(*module), before);
}

TEST_F(StorageAliasIntegrationTest, RejectsDynamicAliasDuringCollection) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%r: memref<8xi32,3>, %offset:index) {
      %s = memref.subview %r[%offset] [2] [1] : memref<8xi32,3> to memref<2xi32,strided<[1],offset:?>,3>
      %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:?>,3> -> memref<2xi32,strided<[1],offset:?>,3>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(collectLayoutConstraints(*module, *target)));
  EXPECT_TRUE(failed(collectLayoutConstraints(*module, *target,
                                            LayoutCollectionMode::RelationsOnly)));
}

TEST_F(StorageAliasIntegrationTest, ProjectionWeakensUnsafeVectorGuarantee) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #root = #frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[32],
      outputs=["byte_offset","bit_offset"],output_extents=[32,8],map=affine_map<(d)->(d,0)>>,
      memory_space=#frisk<memory_space Shared>,alignment=16,vector_granularity=16>
    func.func @f(%r:memref<32xi8,3>) {
      %v = frisk.layout_view %r {layout=#root} : memref<32xi8,3> -> memref<32xi8,3>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  LayoutViewOp view;
  module->walk([&](LayoutViewOp v) { view = v; });
  auto root = analyzeStorageAlias(view.getResult());
  ASSERT_TRUE(succeeded(root));
  root->rootAlignment = 16;
  Builder b(&context);
  auto d = b.getAffineDimExpr(0);
  auto child = *root;
  child.viewType = MemRefType::get({16}, b.getI8Type(), {}, 3);
  child.viewToRoot = AffineMap::get(1, 0, {d + 1}, &context);
  auto projected = projectStorageAliasCandidate(*root, view.getLayoutAttr(), child);
  ASSERT_TRUE(succeeded(projected));
  EXPECT_EQ(projected->getAlignment().getInt(), 16);
  EXPECT_EQ(projected->getVectorGranularity().getInt(), 1);
  auto overstated = StorageLayoutAttr::get(&context, projected->getMap(),
      projected->getMemorySpace(), b.getI64IntegerAttr(16), b.getI64IntegerAttr(16));
  EXPECT_EQ(verifyStorageAliasCandidate(child, overstated).status, ProofStatus::Disproven);
  child.viewType = MemRefType::get({8}, b.getI8Type(), {}, 3);
  child.viewToRoot = AffineMap::get(1, 0, {d * 2}, &context);
  projected = projectStorageAliasCandidate(*root, view.getLayoutAttr(), child);
  ASSERT_TRUE(succeeded(projected));
  EXPECT_EQ(projected->getVectorGranularity().getInt(), 1);
  child.viewType = MemRefType::get({12}, b.getI8Type(), {}, 3);
  child.viewToRoot = AffineMap::get(1, 0, {d}, &context);
  projected = projectStorageAliasCandidate(*root, view.getLayoutAttr(), child);
  ASSERT_TRUE(succeeded(projected));
  EXPECT_EQ(projected->getVectorGranularity().getInt(), 4);
}

TEST_F(StorageAliasIntegrationTest, RootProofBudgetIsCheckedBeforeTargetEnumeration) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%r: memref<65537xi8,3>) {
      %s = memref.subview %r[0] [4] [1] : memref<65537xi8,3> to memref<4xi8,strided<[1]>,3>
      %v = frisk.layout_view %s : memref<4xi8,strided<[1]>,3> -> memref<4xi8,strided<[1]>,3>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  CountingTarget target;
  target.skipEnumeration = true;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(collectLayoutConstraints(*module, target)));
  EXPECT_EQ(target.enumerations, 0u);
}
} // namespace
