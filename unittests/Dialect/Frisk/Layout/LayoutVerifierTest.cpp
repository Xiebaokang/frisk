#include "Dialect/Frisk/Analysis/LayoutVerifier.h"

#include "gtest/gtest.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {

class StringLayoutTarget final : public LayoutTarget {
public:
  void enumerateCandidates(
      const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}

  LogicalResult verifyCandidate(const LayoutVar &, Attribute candidate,
                                Location) const override {
    return isa<StringAttr>(candidate) ? success() : failure();
  }

  FailureOr<CostVector>
  evaluate(const CandidateAssignment &) const override {
    return CostVector{};
  }
};

class LayoutVerifierTest : public testing::Test {
protected:
  LayoutVerifierTest()
      : loc(UnknownLoc::get(&context)), builder(&context),
        type(MemRefType::get({4}, builder.getF16Type(), {}, 3)),
        a(builder.getStringAttr("a")), b(builder.getStringAttr("b")) {}

  MLIRContext context;
  Location loc;
  Builder builder;
  MemRefType type;
  StringAttr a;
  StringAttr b;
  StringLayoutTarget target;
};

TEST_F(LayoutVerifierTest, BootstrapSolverUsesStableCandidateOrdinal) {
  LayoutConstraintGraph graph;
  LayoutVarID id = graph.addVariable(LayoutKind::Distributed, type, "only");
  graph.getVariable(id).candidates = {
      {a, kInvalidProvenanceID, 9}, {b, kInvalidProvenanceID, 1}};
  graph.getVariable(id).state = LayoutState::CandidateSet;
  ASSERT_TRUE(succeeded(graph.finalize(loc)));

  FailureOr<LayoutSolution> solution =
      solveBootstrapLayoutGraph(graph, target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_EQ(solution->assignments.lookup(0), b);
  EXPECT_TRUE(
      succeeded(verifySolvedLayoutGraph(graph, *solution, target, loc)));
}

TEST_F(LayoutVerifierTest, RejectsIncompleteSolutionAndDomainLimit) {
  LayoutConstraintGraph graph;
  LayoutVarID id = graph.addVariable(LayoutKind::Storage, type, "only");
  graph.getVariable(id).candidates = {{a, kInvalidProvenanceID, 0}};
  graph.getVariable(id).state = LayoutState::Resolved;
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  LayoutSolution incomplete;
  EXPECT_TRUE(failed(
      verifySolvedLayoutGraph(graph, incomplete, target, loc)));

  LayoutConstraintGraph tooLarge;
  id = tooLarge.addVariable(LayoutKind::Storage, type, "large-domain");
  for (unsigned ordinal = 0; ordinal < 5; ++ordinal)
    tooLarge.getVariable(id).candidates.push_back(
        {builder.getStringAttr("candidate" + Twine(ordinal)),
         kInvalidProvenanceID, ordinal});
  tooLarge.getVariable(id).state = LayoutState::CandidateSet;
  ASSERT_TRUE(succeeded(tooLarge.finalize(loc)));
  EXPECT_TRUE(failed(solveBootstrapLayoutGraph(tooLarge, target)));
}

TEST_F(LayoutVerifierTest, RejectsUnsupportedHardConstraint) {
  LayoutConstraintGraph graph;
  LayoutVarID id = graph.addVariable(LayoutKind::Storage, type, "only");
  graph.getVariable(id).candidates = {{a, kInvalidProvenanceID, 0}};
  graph.getVariable(id).state = LayoutState::Resolved;
  graph.addConstraint(ConstraintKind::ResourceLimit,
                      ConstraintStrength::Hard, {id}, nullptr, "capacity",
                      "unsupported bootstrap resource limit");
  ASSERT_TRUE(succeeded(graph.finalize(loc)));

  EXPECT_TRUE(failed(solveBootstrapLayoutGraph(graph, target)));
}

} // namespace
