#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include <algorithm>
#include <array>
#include <numeric>

using namespace mlir;
using namespace mlir::frisk;

namespace {
class AliasRegionTest : public testing::Test {
protected:
  AliasRegionTest() {
    context.getOrLoadDialect<FriskDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
  }
  MLIRContext context;
};

TEST_F(AliasRegionTest, RelationsOnlyDoesNotCopyStorageSeeds) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @aliases(%arg: memref<4xf32, 3>) {
      %a = frisk.layout_view %arg : memref<4xf32, 3> -> memref<4xf32, 3>
      %b = frisk.layout_view %arg : memref<4xf32, 3> -> memref<4xf32, 3>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  SmallVector<LayoutViewOp> views;
  module->walk([&](LayoutViewOp op) { views.push_back(op); });
  LayoutVar proposal;
  proposal.shapedType = views[0].getType();
  SmallVector<LayoutCandidate> candidates;
  target->enumerateCandidates(proposal, candidates);
  ASSERT_FALSE(candidates.empty());
  views[0]->setAttr("layout", candidates.front().value);
  auto graph = collectLayoutConstraints(*module, *target,
                                       LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  auto id = graph->lookupVariable(views[1].getResult());
  ASSERT_TRUE(id);
  EXPECT_TRUE(graph->getVariable(*id).candidates.empty());
}

TEST_F(AliasRegionTest, WhileGraphRecordsDistinctInputAndOutputTuples) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @tuples(%a: tensor<4xf32>, %b: tensor<8xf32>, %p: i1) {
      %r = scf.while (%x = %a, %y = %b)
          : (tensor<4xf32>, tensor<8xf32>) -> tensor<8xf32> {
        scf.condition(%p) %y : tensor<8xf32>
      } do {
      ^bb0(%z: tensor<8xf32>):
        scf.yield %a, %z : tensor<4xf32>, tensor<8xf32>
      }
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target,
                                       LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  std::string dump;
  llvm::raw_string_ostream stream(dump);
  graph->print(stream);
  EXPECT_NE(dump.find("region-edge while-init slot=0"), std::string::npos);
  EXPECT_NE(dump.find("region-edge while-init slot=1"), std::string::npos);
  EXPECT_NE(dump.find("region-edge while-backedge slot=0"), std::string::npos);
  EXPECT_NE(dump.find("region-edge while-condition slot=0"), std::string::npos);
  EXPECT_NE(dump.find("region-edge while-result slot=0"), std::string::npos);
}
TEST_F(AliasRegionTest, StrictWorklistReportsActualShrinkAndBudget) {
  Builder b(&context);
  auto type = MemRefType::get({4}, b.getF32Type());
  LayoutConstraintGraph graph;
  auto a = b.getStringAttr("a"), other = b.getStringAttr("other");
  SmallVector<LayoutVarID> ids;
  for (StringRef name : {"first", "middle", "last"}) {
    auto id = graph.addVariable(LayoutKind::Storage, type, name);
    graph.getVariable(id).candidates = {{a, kInvalidProvenanceID, 0},
                                       {other, kInvalidProvenanceID, 1}};
    ids.push_back(id);
  }
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {ids[0], ids[1]}, nullptr, "chain", "first-middle");
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {ids[1], ids[2]}, nullptr, "chain", "middle-last");
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
                      {ids[2]}, nullptr, "seed", "late singleton", a);
  ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
  ASSERT_TRUE(succeeded(propagateStrict(graph)));
  std::string dump;
  llvm::raw_string_ostream stream(dump);
  graph.print(stream);
  EXPECT_NE(dump.find("propagation strict initial=6 final=3 deleted=3"),
            std::string::npos);
  const auto &stats = graph.getPropagationStatistics().strict;
  EXPECT_TRUE(stats.ran);
  EXPECT_TRUE(stats.hasValidBounds());
  EXPECT_EQ(stats.initialCandidates, 6u);
  EXPECT_EQ(stats.finalCandidates, 3u);
  EXPECT_EQ(stats.initialConstraints, 3u);
  EXPECT_EQ(stats.deletedCandidates, 3u);
  EXPECT_EQ(stats.domainChanges, 3u);
  EXPECT_EQ(stats.queuePops, 8u);
  EXPECT_EQ(stats.enqueues, 8u);
  EXPECT_EQ(stats.maximumQueue, 3u);
  EXPECT_EQ(stats.popUpperBound, 8u);
  EXPECT_EQ(stats.staticPopUpperBound, 13u);
  EXPECT_EQ(stats.changesByVariable, (SmallVector<uint64_t>{1, 1, 1}));
}

class StringCandidateTarget final : public LayoutTarget {
public:
  void enumerateCandidates(const LayoutVar &,
                           SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &, Attribute candidate,
                                Location) const override {
    return success(isa<StringAttr>(candidate));
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {
    return CostVector{};
  }
};

TEST_F(AliasRegionTest, CommonAccountingAndAssignmentIgnoreInsertionOrder) {
  Builder builder(&context);
  auto type = RankedTensorType::get({4}, builder.getF32Type());
  StringCandidateTarget target;
  std::optional<std::string> baseline;
  // Same logical graph and candidate domains, all variable and edge orders.
  std::array<unsigned, 3> order{0, 1, 2};
  do {
    for (bool reverseEdges : {false, true}) {
      LayoutConstraintGraph graph;
      std::array<LayoutVarID, 3> ids;
      constexpr StringLiteral names[] = {"a", "b", "c"};
      constexpr StringLiteral domains[][3] = {{"a", "b", "c"},
                                              {"b", "c", "d"},
                                              {"c", "d", "e"}};
      for (unsigned i : order) {
        ids[i] = graph.addVariable(LayoutKind::Distributed, type, names[i]);
        for (unsigned candidate = 0; candidate < 3; ++candidate)
          graph.getVariable(ids[i]).candidates.push_back(
              {builder.getStringAttr(domains[i][candidate]),
               kInvalidProvenanceID, candidate});
      }
      for (unsigned i : {unsigned(reverseEdges), unsigned(!reverseEdges)})
        graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                            {ids[i], ids[i+1]}, nullptr, "chain", names[i]);
      // Soft constraints must not affect hard degree or worklist accounting.
      graph.addConstraint(ConstraintKind::Preference, ConstraintStrength::Soft,
                          {ids[0], ids[2]}, nullptr, "soft", "ignored by propagation");
      ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
      ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(graph)));
      const auto &strict = graph.getPropagationStatistics().strict;
      EXPECT_TRUE(strict.ran);
      EXPECT_EQ(strict.initialCandidates, 9u);
      EXPECT_EQ(strict.finalCandidates, 9u);
      EXPECT_EQ(strict.initialConstraints, 2u);
      EXPECT_EQ(strict.deletedCandidates, 0u);
      EXPECT_EQ(strict.domainChanges, 0u);
      EXPECT_EQ(strict.queuePops, 2u);
      EXPECT_EQ(strict.enqueues, 2u);
      EXPECT_EQ(strict.maximumQueue, 2u);
      EXPECT_EQ(strict.popUpperBound, 2u);
      EXPECT_EQ(strict.staticPopUpperBound, 14u);
      EXPECT_EQ(strict.changesByVariable, (SmallVector<uint64_t>{0, 0, 0}));
      EXPECT_TRUE(strict.hasValidBounds());
      const auto &common = graph.getPropagationStatistics().common;
      EXPECT_TRUE(common.ran);
      EXPECT_EQ(common.initialCandidates, 9u);
      EXPECT_EQ(common.finalCandidates, 3u);
      EXPECT_EQ(common.initialConstraints, 2u);
      EXPECT_EQ(common.deletedCandidates, 6u);
      EXPECT_EQ(common.domainChanges, 5u);
      EXPECT_EQ(common.queuePops, 5u);
      EXPECT_EQ(common.enqueues, 5u);
      EXPECT_EQ(common.maximumQueue, 2u);
      EXPECT_EQ(common.popUpperBound, 9u);
      EXPECT_EQ(common.staticPopUpperBound, 14u);
      EXPECT_EQ(common.changesByVariable, (SmallVector<uint64_t>{2, 2, 1}));
      EXPECT_EQ(std::accumulate(common.changesByVariable.begin(),
                                common.changesByVariable.end(), uint64_t{0}),
                common.domainChanges);
      EXPECT_TRUE(common.hasValidBounds());
      auto solution = solveBootstrapLayoutGraph(graph, target);
      ASSERT_TRUE(succeeded(solution));
      std::string summary;
      llvm::raw_string_ostream out(summary);
      strict.print("strict", out);
      common.print("common", out);
      for (const auto &var : graph.getVariables()) {
        EXPECT_EQ(solution->assignments.lookup(var.id), builder.getStringAttr("c"));
        out << var.stableName << '=' << solution->assignments.lookup(var.id) << '\n';
      }
      if (!baseline) baseline = summary;
      EXPECT_EQ(summary, *baseline);
    }
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST_F(AliasRegionTest, ConflictCountsDeletedCandidatesBeforeReturning) {
  Builder builder(&context);
  auto type = RankedTensorType::get({4}, builder.getF32Type());
  for (bool requireConflict : {false, true}) {
    LayoutConstraintGraph graph;
    auto a = graph.addVariable(LayoutKind::Distributed, type, "a");
    graph.getVariable(a).candidates = {{builder.getStringAttr("a"), kInvalidProvenanceID, 0}};
    if (requireConflict) {
      graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
                          {a}, nullptr, "seed", "absent candidate", builder.getStringAttr("b"));
    } else {
      auto b = graph.addVariable(LayoutKind::Distributed, type, "b");
      graph.getVariable(b).candidates = {{builder.getStringAttr("b"), kInvalidProvenanceID, 0}};
      graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                          {a, b}, nullptr, "conflict", "disjoint singleton domains");
    }
    ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
    EXPECT_TRUE(failed(propagateStrict(graph)));
    const auto &stats = graph.getPropagationStatistics().strict;
    uint64_t changed = requireConflict ? 1 : 2;
    EXPECT_EQ(stats.initialCandidates, changed);
    EXPECT_EQ(stats.finalCandidates, 0u);
    EXPECT_EQ(stats.deletedCandidates, changed);
    EXPECT_EQ(stats.domainChanges, changed);
    EXPECT_EQ(stats.initialConstraints, 1u);
    EXPECT_EQ(stats.queuePops, 1u);
    EXPECT_EQ(stats.enqueues, 2u);
    EXPECT_EQ(stats.maximumQueue, 1u);
    EXPECT_EQ(stats.popUpperBound, 1+changed);
    EXPECT_EQ(stats.staticPopUpperBound, 1+changed);
    EXPECT_TRUE(stats.hasValidBounds());
    for (const auto &var : graph.getVariables()) {
      EXPECT_EQ(var.state, LayoutState::Conflict);
      EXPECT_TRUE(var.candidates.empty());
      EXPECT_EQ(stats.changesByVariable[var.id], 1u);
    }
    EXPECT_FALSE(graph.getPropagationStatistics().common.ran);
  }
}

TEST_F(AliasRegionTest, CommonConflictCountsBothDomainsBeforeReturning) {
  Builder builder(&context);
  auto type = RankedTensorType::get({4}, builder.getF32Type());
  LayoutConstraintGraph graph;
  auto a = graph.addVariable(LayoutKind::Distributed, type, "a");
  auto b = graph.addVariable(LayoutKind::Distributed, type, "b");
  graph.getVariable(a).candidates = {
      {builder.getStringAttr("a"), kInvalidProvenanceID, 0},
      {builder.getStringAttr("b"), kInvalidProvenanceID, 1}};
  graph.getVariable(b).candidates = {
      {builder.getStringAttr("c"), kInvalidProvenanceID, 0},
      {builder.getStringAttr("d"), kInvalidProvenanceID, 1}};
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {a, b}, nullptr, "conflict", "disjoint non-singleton domains");
  ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
  EXPECT_TRUE(failed(propagateCommonToFixedPoint(graph)));
  const auto &strict = graph.getPropagationStatistics().strict;
  EXPECT_EQ(strict.initialCandidates, 4u);
  EXPECT_EQ(strict.finalCandidates, 4u);
  EXPECT_EQ(strict.queuePops, 1u);
  EXPECT_EQ(strict.deletedCandidates, 0u);
  EXPECT_TRUE(strict.hasValidBounds());
  const auto &common = graph.getPropagationStatistics().common;
  EXPECT_EQ(common.initialCandidates, 4u);
  EXPECT_EQ(common.finalCandidates, 0u);
  EXPECT_EQ(common.initialConstraints, 1u);
  EXPECT_EQ(common.deletedCandidates, 4u);
  EXPECT_EQ(common.domainChanges, 2u);
  EXPECT_EQ(common.queuePops, 1u);
  EXPECT_EQ(common.enqueues, 2u);
  EXPECT_EQ(common.maximumQueue, 1u);
  EXPECT_EQ(common.popUpperBound, 3u);
  EXPECT_EQ(common.staticPopUpperBound, 5u);
  EXPECT_EQ(common.changesByVariable, (SmallVector<uint64_t>{1, 1}));
  EXPECT_TRUE(common.hasValidBounds());
}

static constexpr StringLiteral forFixture = R"mlir(
  func.func @loop(%x: tensor<4xf32>, %n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %r = scf.for %i = %zero to %n step %one iter_args(%arg = %x) -> tensor<4xf32> {
      scf.yield %arg : tensor<4xf32>
    }
    return
  })mlir";

TEST_F(AliasRegionTest, RealForBackedgeIsRescheduledWithoutForcingEquality) {
  auto module = parseSourceString<ModuleOp>(forFixture, &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  scf::ForOp loop;
  module->walk([&](scf::ForOp op) { loop = op; });
  ASSERT_TRUE(loop);
  auto resultID = graph->lookupVariable(loop.getResult(0));
  auto iterID = graph->lookupVariable(loop.getRegionIterArgs()[0]);
  auto inputID = graph->lookupVariable(loop.getInitArgs()[0]);
  ASSERT_TRUE(resultID && iterID && inputID);
  SmallVector<LayoutCandidate> candidates;
  target->enumerateCandidates(graph->getVariable(*resultID), candidates);
  ASSERT_GE(candidates.size(), 2u);
  for (auto &var : graph->getVariables())
    var.candidates = {candidates[0], candidates[1]};
  graph->addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
                       {*resultID}, loop, "late-result-seed", "pin result after initial queue", candidates[0].value);
  ASSERT_TRUE(succeeded(graph->finalize(module->getLoc())));
  resultID = graph->lookupVariable(loop.getResult(0));
  iterID = graph->lookupVariable(loop.getRegionIterArgs()[0]);
  inputID = graph->lookupVariable(loop.getInitArgs()[0]);
  ASSERT_TRUE(succeeded(propagateStrict(*graph)));
  const auto &stats = graph->getPropagationStatistics().strict;
  EXPECT_EQ(stats.initialCandidates, 6u);
  EXPECT_EQ(stats.finalCandidates, 4u);
  EXPECT_EQ(stats.initialConstraints, 4u);
  EXPECT_EQ(stats.deletedCandidates, 2u);
  EXPECT_EQ(stats.domainChanges, 2u);
  EXPECT_EQ(stats.queuePops, 9u);
  EXPECT_EQ(stats.enqueues, 9u);
  EXPECT_EQ(stats.maximumQueue, 4u);
  EXPECT_EQ(stats.popUpperBound, 10u);
  EXPECT_EQ(stats.staticPopUpperBound, 18u);
  EXPECT_EQ(stats.changesByVariable[*resultID], 1u);
  EXPECT_EQ(stats.changesByVariable[*iterID], 1u);
  EXPECT_EQ(stats.changesByVariable[*inputID], 0u);
  EXPECT_EQ(graph->getVariable(*inputID).candidates.size(), 2u);
  EXPECT_TRUE(stats.hasValidBounds());
  unsigned backedges = 0;
  for (const auto &edge : graph->getRegionEdges()) {
    if (edge.kind != RegionLayoutEdgeKind::ForBackedge) continue;
    ++backedges;
    EXPECT_EQ(edge.source, *iterID);
    EXPECT_EQ(edge.target, *resultID);
    EXPECT_EQ(graph->getConstraint(edge.constraint).kind, ConstraintKind::Convertible);
    EXPECT_EQ(edge.use->get(), loop.getRegionIterArgs()[0]);
  }
  EXPECT_EQ(backedges, 1u);
  // Valid single-CTA backedges are Convertible, not SameLayout: they are woken
  // by these two endpoint changes but cannot reject another valid encoding.
}

TEST_F(AliasRegionTest, FinalizeRemapsRegionVariablesAndConstraintIDsTogether) {
  auto module = parseSourceString<ModuleOp>(forFixture, &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  struct SavedEdge { RegionLayoutEdge edge; std::string source, target, use; };
  for (auto &var : graph->getVariables()) {
    SmallVector<LayoutCandidate> candidates;
    target->enumerateCandidates(var, candidates);
    ASSERT_FALSE(candidates.empty());
    var.candidates = {candidates.front()};
  }
  ASSERT_TRUE(succeeded(propagateStrict(*graph)));
  ASSERT_TRUE(graph->getPropagationStatistics().strict.ran);
  SmallVector<SavedEdge> saved;
  for (const auto &edge : graph->getRegionEdges())
    saved.push_back({edge, graph->getVariable(edge.source).stableName,
                     graph->getVariable(edge.target).stableName,
                     graph->getConstraint(edge.constraint).stableUseKey});
  auto type = graph->getVariables()[0].shapedType;
  auto z = graph->addVariable(LayoutKind::Distributed, type, "!b");
  auto a = graph->addVariable(LayoutKind::Distributed, type, "!a");
  graph->addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                       {z, a}, *module, "sort-first", "force constraint ID remap");
  ASSERT_TRUE(succeeded(graph->finalize(module->getLoc())));
  EXPECT_FALSE(graph->getPropagationStatistics().strict.ran);
  ASSERT_EQ(graph->getRegionEdges().size(), saved.size());
  for (auto [edge, old] : llvm::zip_equal(graph->getRegionEdges(), saved)) {
    EXPECT_EQ(edge.stableKey, old.edge.stableKey);
    EXPECT_EQ(edge.source, old.edge.source+2);
    EXPECT_EQ(edge.target, old.edge.target+2);
    EXPECT_EQ(edge.constraint, old.edge.constraint+1);
    EXPECT_EQ(graph->getVariable(edge.source).stableName, old.source);
    EXPECT_EQ(graph->getVariable(edge.target).stableName, old.target);
    EXPECT_EQ(graph->getConstraint(edge.constraint).stableUseKey, old.use);
    EXPECT_EQ(edge.owner, old.edge.owner);
    EXPECT_EQ(edge.use, old.edge.use);
  }
  EXPECT_TRUE(succeeded(graph->verifyInvariants(module->getLoc())));
}

TEST_F(AliasRegionTest, RegionInvariantsRejectInvalidIdentityAndSlots) {
  auto module = parseSourceString<ModuleOp>(forFixture, &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto original = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(original));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (unsigned invalidCase = 0; invalidCase < 6; ++invalidCase) {
    LayoutConstraintGraph graph = *original;
    auto found = llvm::find_if(graph.getRegionEdges(), [](const auto &edge) {
      return edge.kind == RegionLayoutEdgeKind::ForBackedge;
    });
    ASSERT_NE(found, graph.getRegionEdges().end());
    RegionLayoutEdge edge = *found;
    edge.stableKey += ":invalid";
    if (invalidCase == 0) edge.source = graph.getVariables().size();
    if (invalidCase == 1) edge.constraint = graph.getConstraints().size();
    if (invalidCase == 2) edge.owner = nullptr;
    if (invalidCase == 3) edge.use = nullptr;
    if (invalidCase == 4) edge.slot = 99;
    if (invalidCase == 5) std::swap(edge.source, edge.target);
    graph.addRegionEdge(edge);
    EXPECT_TRUE(failed(graph.verifyInvariants(module->getLoc()))) << "invalid case " << invalidCase;
  }
}

TEST_F(AliasRegionTest, GraphInvariantsRejectIDsThatDifferFromArrayPositions) {
  Builder builder(&context);
  auto type = RankedTensorType::get({4}, builder.getF32Type());
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (bool corruptVariable : {false, true}) {
    LayoutConstraintGraph graph;
    auto a = graph.addVariable(LayoutKind::Distributed, type, "a");
    auto b = graph.addVariable(LayoutKind::Distributed, type, "b");
    auto relation = graph.addConstraint(ConstraintKind::SameLayout,
        ConstraintStrength::Hard, {a, b}, nullptr, "identity", "valid before corruption");
    ASSERT_TRUE(succeeded(graph.verifyInvariants(UnknownLoc::get(&context))));
    if (corruptVariable) graph.getVariable(a).id = 99;
    else graph.getConstraints()[relation].id = 99;
    EXPECT_TRUE(failed(graph.verifyInvariants(UnknownLoc::get(&context))));
    EXPECT_TRUE(failed(graph.finalize(UnknownLoc::get(&context))));
  }
}

TEST_F(AliasRegionTest, ZeroResultRegionsAndDistinctWhileTupleSlots) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @zero(%a: tensor<4xf32>, %p: i1, %n: index) {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.if %p { scf.yield }
      scf.for %i = %zero to %n step %one { scf.yield }
      scf.while (%x = %a) : (tensor<4xf32>) -> () {
        scf.condition(%p)
      } do {
        scf.yield %a : tensor<4xf32>
      }
      return
    }
    func.func @different(%a: tensor<4xf32>, %b: tensor<8xi32>, %p: i1) {
      %r = scf.while (%x = %a, %y = %b) : (tensor<4xf32>, tensor<8xi32>) -> tensor<8xi32> {
        scf.condition(%p) %y : tensor<8xi32>
      } do {
      ^bb0(%z: tensor<8xi32>):
        scf.yield %a, %z : tensor<4xf32>, tensor<8xi32>
      }
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  unsigned zeroEdges = 0, differentEdges = 0;
  for (const auto &edge : graph->getRegionEdges()) {
    auto function = edge.owner->getParentOfType<func::FuncOp>();
    if (function.getName() == "zero") {
      ++zeroEdges;
      EXPECT_TRUE(edge.kind == RegionLayoutEdgeKind::WhileInit ||
                  edge.kind == RegionLayoutEdgeKind::WhileBackedge);
      EXPECT_EQ(edge.slot, 0u);
    } else {
      ++differentEdges;
      if (edge.kind == RegionLayoutEdgeKind::WhileCondition ||
          edge.kind == RegionLayoutEdgeKind::WhileResult)
        EXPECT_EQ(edge.slot, 0u);
    }
    EXPECT_EQ(graph->getVariable(edge.source).shapedType,
              graph->getVariable(edge.target).shapedType);
    if (edge.kind == RegionLayoutEdgeKind::WhileCondition)
      EXPECT_EQ(edge.use->getOperandNumber(), 1u); // predicate is not a forwarded tensor.
  }
  EXPECT_EQ(zeroEdges, 2u);
  EXPECT_EQ(differentEdges, 6u);
  for (const auto &var : graph->getVariables())
    EXPECT_TRUE(isa<RankedTensorType>(var.shapedType));
}
} // namespace
