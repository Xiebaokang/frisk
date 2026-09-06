#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {

class LayoutPropagationTest : public testing::Test {
protected:
  LayoutPropagationTest()
      : loc(UnknownLoc::get(&context)), builder(&context),
        type(MemRefType::get({4}, builder.getF16Type(), {}, 3)),
        a(builder.getStringAttr("a")), b(builder.getStringAttr("b")),
        c(builder.getStringAttr("c")) {}

  void addDomain(LayoutConstraintGraph &graph, LayoutVarID id,
                 std::initializer_list<Attribute> attrs) {
    uint64_t ordinal = 0;
    for (Attribute attr : attrs)
      graph.getVariable(id).candidates.push_back(
          {attr, kInvalidProvenanceID, ordinal++});
    graph.getVariable(id).state = LayoutState::CandidateSet;
  }

  MLIRContext context;
  Location loc;
  Builder builder;
  MemRefType type;
  StringAttr a;
  StringAttr b;
  StringAttr c;
};

TEST_F(LayoutPropagationTest, StrictSeedPropagatesAcrossStorageAccess) {
  LayoutConstraintGraph graph;
  LayoutVarID src = graph.addVariable(LayoutKind::Storage, type, "src");
  LayoutVarID dst = graph.addVariable(LayoutKind::Storage, type, "dst");
  addDomain(graph, src, {a});
  addDomain(graph, dst, {a, b});
  graph.addConstraint(ConstraintKind::RequireEncoding,
                      ConstraintStrength::Hard, {src}, nullptr, "view",
                      "explicit source binding", a);
  graph.addConstraint(ConstraintKind::StorageAccess,
                      ConstraintStrength::Hard, {src, dst}, nullptr, "copy",
                      "whole tile copy");
  ASSERT_TRUE(succeeded(graph.finalize(loc)));

  ASSERT_TRUE(succeeded(propagateStrict(graph)));
  auto dstID = *graph.lookupVariable("dst");
  ASSERT_EQ(graph.getVariable(dstID).candidates.size(), 1u);
  EXPECT_EQ(graph.getVariable(dstID).candidates.front().value, a);
  EXPECT_EQ(graph.getVariable(dstID).state, LayoutState::Resolved);
}

TEST_F(LayoutPropagationTest, StrictPropagationReachesFixedPoint) {
  LayoutConstraintGraph graph;
  LayoutVarID first = graph.addVariable(LayoutKind::Storage, type, "first");
  LayoutVarID middle =
      graph.addVariable(LayoutKind::Storage, type, "middle");
  LayoutVarID last = graph.addVariable(LayoutKind::Storage, type, "last");
  addDomain(graph, first, {a, b});
  addDomain(graph, middle, {a, b});
  addDomain(graph, last, {a, b});
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {first, middle}, nullptr, "first-middle", "chain");
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {middle, last}, nullptr, "middle-last", "chain");
  graph.addConstraint(ConstraintKind::RequireEncoding,
                      ConstraintStrength::Hard, {last}, nullptr, "seed",
                      "late ordered seed", a);
  ASSERT_TRUE(succeeded(graph.finalize(loc)));

  ASSERT_TRUE(succeeded(propagateStrict(graph)));
  for (const LayoutVar &var : graph.getVariables()) {
    ASSERT_EQ(var.candidates.size(), 1u);
    EXPECT_EQ(var.candidates.front().value, a);
    EXPECT_EQ(var.state, LayoutState::Resolved);
  }
}

TEST_F(LayoutPropagationTest, CommonPropagationComputesFixedPointIntersection) {
  LayoutConstraintGraph graph;
  LayoutVarID lhs = graph.addVariable(LayoutKind::Storage, type, "lhs");
  LayoutVarID rhs = graph.addVariable(LayoutKind::Storage, type, "rhs");
  addDomain(graph, lhs, {a, b});
  addDomain(graph, rhs, {b, c});
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {lhs, rhs}, nullptr, "alias", "same source");
  ASSERT_TRUE(succeeded(graph.finalize(loc)));

  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(graph)));
  for (const LayoutVar &var : graph.getVariables()) {
    ASSERT_EQ(var.candidates.size(), 1u);
    EXPECT_EQ(var.candidates.front().value, b);
    EXPECT_EQ(var.state, LayoutState::Resolved);
  }
}

TEST_F(LayoutPropagationTest, IncompatibleHardSeedsConflict) {
  LayoutConstraintGraph graph;
  LayoutVarID lhs = graph.addVariable(LayoutKind::Storage, type, "lhs");
  LayoutVarID rhs = graph.addVariable(LayoutKind::Storage, type, "rhs");
  addDomain(graph, lhs, {a});
  addDomain(graph, rhs, {b});
  graph.addConstraint(ConstraintKind::StorageAccess,
                      ConstraintStrength::Hard, {lhs, rhs}, nullptr, "copy",
                      "incompatible bindings");
  ASSERT_TRUE(succeeded(graph.finalize(loc)));

  EXPECT_TRUE(failed(propagateStrict(graph)));
  EXPECT_EQ(graph.getVariable(*graph.lookupVariable("lhs")).state,
            LayoutState::Conflict);
  EXPECT_EQ(graph.getVariable(*graph.lookupVariable("rhs")).state,
            LayoutState::Conflict);
}

TEST_F(LayoutPropagationTest, SM90EnumeratesVerifiedStorageCandidates) {
  context.getOrLoadDialect<FriskDialect>();
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  for (auto [columns, xorOrdinal] :
       {std::pair<int64_t, uint64_t>{16, 3}, {32, 4}, {64, 5}}) {
    auto matrixType =
        MemRefType::get({64, columns}, builder.getF16Type(), {}, 3);
    LayoutVar var{0, LayoutKind::Storage, matrixType, {},
                  LayoutState::Uninitialized, "matrix", nullptr};
    SmallVector<LayoutCandidate> candidates;
    target->enumerateCandidates(var, candidates);

    ASSERT_EQ(candidates.size(), 4u);
    EXPECT_EQ(candidates.back().stableOrdinal, xorOrdinal);
    for (const LayoutCandidate &candidate : candidates)
      EXPECT_TRUE(
          succeeded(target->verifyCandidate(var, candidate.value, loc)));
  }
}

TEST_F(LayoutPropagationTest, AliasSeedRecordsPropagationProvenance) {
  context.getOrLoadDialect<FriskDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    #linear = #frisk.storage<
      map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
        outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
        map = affine_map<(d0) -> (d0 * 2, 0)>>,
      memory_space = #frisk<memory_space Shared>, alignment = 2,
      vector_granularity = 2>
    module {
      func.func @alias_provenance(%source: memref<4xf16, 3>) {
        %bound = frisk.layout_view %source {layout = #linear}
          : memref<4xf16, 3> -> memref<4xf16, 3>
        %inferred = frisk.layout_view %source
          : memref<4xf16, 3> -> memref<4xf16, 3>
        return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  FailureOr<LayoutConstraintGraph> graph =
      collectLayoutConstraints(module->getOperation(), *target);
  ASSERT_TRUE(succeeded(graph));

  auto inferred = llvm::find_if(graph->getVariables(), [](const LayoutVar &var) {
    return StringRef(var.stableName).contains("/o1/r0/storage");
  });
  ASSERT_NE(inferred, graph->getVariables().end());
  ASSERT_EQ(inferred->candidates.size(), 1u);
  std::string chain;
  llvm::raw_string_ostream stream(chain);
  ASSERT_TRUE(succeeded(graph->printProvenanceChain(
      inferred->candidates.front().provenance, stream)));
  EXPECT_NE(chain.find("same-source-layout-view"), std::string::npos);
  EXPECT_NE(chain.find("layout_view: explicit layout binding"),
            std::string::npos);
}

} // namespace
