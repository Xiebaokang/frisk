#include "Dialect/Frisk/Analysis/LayoutConstraint.h"

#include "gtest/gtest.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {

TEST(LayoutConstraintTest, FinalizeIsIndependentOfInsertionOrder) {
  MLIRContext context;
  Location loc = UnknownLoc::get(&context);
  auto type = MemRefType::get({4}, Float16Type::get(&context), {}, 3);

  LayoutConstraintGraph first;
  LayoutVarID firstB = first.addVariable(LayoutKind::Storage, type, "b");
  LayoutVarID firstA = first.addVariable(LayoutKind::Storage, type, "a");
  first.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {firstB, firstA}, nullptr, "copy", "whole tile");
  ASSERT_TRUE(succeeded(first.finalize(loc)));

  LayoutConstraintGraph second;
  LayoutVarID secondA = second.addVariable(LayoutKind::Storage, type, "a");
  LayoutVarID secondB = second.addVariable(LayoutKind::Storage, type, "b");
  second.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                       {secondA, secondB}, nullptr, "copy", "whole tile");
  ASSERT_TRUE(succeeded(second.finalize(loc)));

  std::string firstDump;
  std::string secondDump;
  llvm::raw_string_ostream(firstDump) << first;
  llvm::raw_string_ostream(secondDump) << second;
  EXPECT_EQ(firstDump, secondDump);
  ASSERT_EQ(first.getVariables().size(), 2u);
  EXPECT_EQ(first.getVariables()[0].stableName, "a");
  EXPECT_EQ(first.getVariables()[0].id, 0u);
  EXPECT_EQ(first.getVariables()[1].stableName, "b");
  EXPECT_EQ(first.getVariables()[1].id, 1u);
}

TEST(LayoutConstraintTest, RejectsInvalidGraphInvariants) {
  MLIRContext context;
  Location loc = UnknownLoc::get(&context);
  auto type = MemRefType::get({4}, Float16Type::get(&context), {}, 3);

  LayoutConstraintGraph duplicate;
  duplicate.addVariable(LayoutKind::Storage, type, "same");
  duplicate.addVariable(LayoutKind::Storage, type, "same");
  EXPECT_TRUE(failed(duplicate.finalize(loc)));

  LayoutConstraintGraph badReference;
  badReference.addVariable(LayoutKind::Storage, type, "only");
  badReference.addConstraint(ConstraintKind::SameLayout,
                             ConstraintStrength::Hard, {0, 7}, nullptr,
                             "bad", "invalid reference");
  EXPECT_TRUE(failed(badReference.verifyInvariants(loc)));

  LayoutConstraintGraph emptyHard;
  emptyHard.addConstraint(ConstraintKind::RequireEncoding,
                          ConstraintStrength::Hard, {}, nullptr, "bad",
                          "empty hard constraint");
  EXPECT_TRUE(failed(emptyHard.verifyInvariants(loc)));
}

TEST(LayoutConstraintTest, PrintsProvenanceFromLeafToRoot) {
  LayoutConstraintGraph graph;
  ProvenanceID root =
      graph.addProvenance(std::nullopt, nullptr, "seed", "explicit layout");
  ProvenanceID leaf =
      graph.addProvenance(root, nullptr, "copy", "whole tile alias");

  std::string text;
  llvm::raw_string_ostream stream(text);
  ASSERT_TRUE(succeeded(graph.printProvenanceChain(leaf, stream)));
  EXPECT_NE(text.find("copy: whole tile alias"), std::string::npos);
  EXPECT_NE(text.find("seed: explicit layout"), std::string::npos);
}

} // namespace
