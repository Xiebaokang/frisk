#include "Dialect/Frisk/Analysis/LayoutVerifier.h"

#include <algorithm>
#include <array>
#include <random>

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

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

TEST_F(LayoutVerifierTest, ShuffledInsertionProducesCanonicalSolution) {
  std::optional<std::string> baselineGraph;
  std::optional<std::string> baselineSolution;
  constexpr std::array<unsigned, 4> seeds = {1, 7, 42, 20260816};
  for (unsigned baseSeed : seeds) {
    for (unsigned repetition = 0; repetition < 5; ++repetition) {
      std::mt19937 generator(baseSeed + repetition * 65537);
      LayoutConstraintGraph graph;
      std::array<unsigned, 3> variableOrder = {0, 1, 2};
      std::shuffle(variableOrder.begin(), variableOrder.end(), generator);
      constexpr std::array<StringLiteral, 3> names = {"a", "b", "c"};
      std::array<LayoutVarID, 3> ids;
      for (unsigned index : variableOrder) {
        ids[index] =
            graph.addVariable(LayoutKind::Distributed, type, names[index]);
        SmallVector<LayoutCandidate> &candidates =
            graph.getVariable(ids[index]).candidates;
        if (generator() & 1)
          candidates = {{a, kInvalidProvenanceID, 1},
                        {b, kInvalidProvenanceID, 0}};
        else
          candidates = {{b, kInvalidProvenanceID, 0},
                        {a, kInvalidProvenanceID, 1}};
        graph.getVariable(ids[index]).state = LayoutState::CandidateSet;
      }

      std::array<unsigned, 3> constraintOrder = {0, 1, 2};
      std::shuffle(constraintOrder.begin(), constraintOrder.end(), generator);
      for (unsigned constraint : constraintOrder) {
        if (constraint == 0)
          graph.addConstraint(ConstraintKind::SameLayout,
                              ConstraintStrength::Hard, {ids[0], ids[1]},
                              nullptr, "a-b", "chain");
        else if (constraint == 1)
          graph.addConstraint(ConstraintKind::SameLayout,
                              ConstraintStrength::Hard, {ids[1], ids[2]},
                              nullptr, "b-c", "chain");
        else
          graph.addConstraint(ConstraintKind::RequireEncoding,
                              ConstraintStrength::Hard, {ids[2]}, nullptr,
                              "seed-c", "fixed layout", b);
      }

      ASSERT_TRUE(succeeded(graph.finalize(loc)));
      std::string graphDump;
      llvm::raw_string_ostream graphStream(graphDump);
      graphStream << graph;
      graphStream.flush();
      ASSERT_TRUE(succeeded(propagateStrict(graph)));
      ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(graph)));
      FailureOr<LayoutSolution> solution =
          solveBootstrapLayoutGraph(graph, target);
      ASSERT_TRUE(succeeded(solution));
      ASSERT_TRUE(
          succeeded(verifySolvedLayoutGraph(graph, *solution, target, loc)));
      std::string solutionDump;
      llvm::raw_string_ostream solutionStream(solutionDump);
      for (const LayoutVar &var : graph.getVariables())
        solutionStream << var.stableName << '='
                       << solution->assignments.lookup(var.id) << '\n';
      solutionStream.flush();

      if (!baselineGraph) {
        baselineGraph = graphDump;
        baselineSolution = solutionDump;
      } else {
        EXPECT_EQ(graphDump, *baselineGraph);
        EXPECT_EQ(solutionDump, *baselineSolution);
      }
    }
  }
}

} // namespace
