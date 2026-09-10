#include "Dialect/Frisk/Transforms/LayoutTypeConverter.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
class LayoutMaterializationTest : public testing::Test {
protected:
  LayoutMaterializationTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect,
                        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  MLIRContext context;
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  std::string print(Operation *op) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    op->print(stream);
    return text;
  }
  FailureOr<LayoutSolution> solve(LayoutConstraintGraph &graph) {
    if (failed(propagateStrict(graph)) || failed(propagateCommonToFixedPoint(graph)))
      return failure();
    return solveBootstrapLayoutGraph(graph, *target);
  }
};

TEST_F(LayoutMaterializationTest, DenseConstantAndReplay) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @constant() -> tensor<4xf16> {
      %c = arith.constant dense<1.0> : tensor<4xf16>
      return %c : tensor<4xf16>
    })mlir", &context);
  ASSERT_TRUE(module);
  for (unsigned iteration = 0; iteration < 2; ++iteration) {
    auto graph = collectLayoutConstraints(*module, *target);
    ASSERT_TRUE(succeeded(graph));
    auto solution = solve(*graph);
    ASSERT_TRUE(succeeded(solution));
    ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
    module->walk([&](arith::ConstantOp constant) {
      auto type = cast<RankedTensorType>(constant.getType());
      EXPECT_TRUE(isa_and_nonnull<DistributedEncodingAttr>(type.getEncoding()));
      EXPECT_EQ(constant.getValue().getType(), type);
    });
  }
}

TEST_F(LayoutMaterializationTest, InvalidAssignmentRollsBackStorageAndTensor) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @load(%p: memref<4xf16, 3>) {
      %v = frisk.layout_view %p : memref<4xf16, 3> -> memref<4xf16, 3>
      %t = frisk.tile_load %v : memref<4xf16, 3> -> tensor<4xf16>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto solution = solve(*graph);
  ASSERT_TRUE(succeeded(solution));
  for (const auto &var : graph->getVariables())
    if (var.kind == LayoutKind::Distributed)
      solution->assignments.erase(var.id);
  auto before = print(*module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(materializeLayouts(*module, *graph, *solution)));
  EXPECT_EQ(before, print(*module));
}

TEST_F(LayoutMaterializationTest, MaterializedVerifierRejectsUnusedTensor) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @unused(%arg: tensor<4xf16>) {
      %t = tensor.empty() : tensor<4xf16>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(LayoutMaterializationTest, ValueAwareTypesNeverCacheByTensorType) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @two(%a: tensor<4xf16>, %b: tensor<4xf16>) { return }
  )mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto function = *module->getOps<func::FuncOp>().begin();
  auto a = function.getArgument(0), b = function.getArgument(1);
  auto aid = graph->lookupVariable(a), bid = graph->lookupVariable(b);
  ASSERT_TRUE(aid && bid);
  auto candidates = graph->getVariable(*aid).candidates;
  ASSERT_GE(candidates.size(), 2u);
  LayoutSolution solution;
  solution.assignments[*aid] = candidates[0].value;
  solution.assignments[*bid] = candidates[1].value;
  LayoutTypeConverter converter(*graph, solution);
  auto at = converter.convertLayoutBearingTensor(a);
  auto bt = converter.convertLayoutBearingTensor(b);
  ASSERT_TRUE(succeeded(at) && succeeded(bt));
  EXPECT_EQ(at->getEncoding(), candidates[0].value);
  EXPECT_EQ(bt->getEncoding(), candidates[1].value);
  EXPECT_NE(*at, *bt);
  EXPECT_FALSE(converter.convertType(a.getType()));
}

TEST_F(LayoutMaterializationTest, PublicConverterDiagnosesInvalidAssignments) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @unencoded(%arg: tensor<4xf16>) { return }
  )mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto function = *module->getOps<func::FuncOp>().begin();
  BlockArgument argument = function.getArgument(0);
  argument.setLoc(FileLineColLoc::get(&context, "converter-test", 7, 3));
  auto id = graph->lookupVariable(argument);
  ASSERT_TRUE(id);
  for (bool missing : {true, false}) {
    LayoutSolution solution;
    if (!missing)
      solution.assignments[*id] = StringAttr::get(&context, "not-a-layout");
    LayoutTypeConverter converter(*graph, solution);
    unsigned count = 0;
    std::string diagnostics;
    ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
      ++count;
      EXPECT_EQ(diagnostic.getSeverity(), DiagnosticSeverity::Error);
      EXPECT_EQ(diagnostic.getLocation(), argument.getLoc());
      llvm::raw_string_ostream stream(diagnostics);
      diagnostic.print(stream);
      return success();
    });
    EXPECT_TRUE(failed(converter.convertLayoutBearingTensor(argument)));
    EXPECT_EQ(count, 1u);
    EXPECT_NE(diagnostics.find(missing ? "missing distributed layout assignment"
                                      : "layout assignment is not a distributed encoding"),
              std::string::npos) << diagnostics;
  }
}

TEST_F(LayoutMaterializationTest, RebuildFailureAfterStorageCloneRollsBack) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @load(%p: memref<4xf16, 3>) {
      %v = frisk.layout_view %p : memref<4xf16, 3> -> memref<4xf16, 3>
      %t = frisk.tile_load %v : memref<4xf16, 3> -> tensor<4xf16>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto solution = solve(*graph);
  ASSERT_TRUE(succeeded(solution));
  auto function = *module->getOps<func::FuncOp>().begin();
  OpBuilder builder(function.getBody().front().getTerminator());
  // A new valid but unmodeled definition tests failure after earlier storage
  // and tensor ops were rebuilt. The original graph's pointers remain live.
  builder.create<tensor::EmptyOp>(function.getLoc(), ArrayRef<int64_t>{4},
                                  builder.getF16Type());
  ASSERT_TRUE(succeeded(verify(*module)));
  auto before = print(*module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(materializeLayouts(*module, *graph, *solution)));
  EXPECT_EQ(before, print(*module));
}

TEST_F(LayoutMaterializationTest, SelectedIfYieldConversionsAndMissingEdgeRollback) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @choose(%p: i1, %x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
      %r = scf.if %p -> tensor<4xf16> {
        scf.yield %x : tensor<4xf16>
      } else {
        scf.yield %y : tensor<4xf16>
      }
      return %r : tensor<4xf16>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_GE(graph->getVariables().front().candidates.size(), 2u);
  auto a = graph->getVariables().front().candidates[0].value;
  auto b = graph->getVariables().front().candidates[1].value;
  for (auto &var : graph->getVariables()) {
    bool join = var.functionResult.has_value() ||
                (var.value && var.value.getDefiningOp<scf::IfOp>());
    var.candidates = {{join ? b : a, kInvalidProvenanceID, 0}};
  }
  auto solution = solve(*graph);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 2u);
  auto before = print(*module);
  {
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    auto missing = *solution;
    missing.conversions.pop_back();
    EXPECT_TRUE(failed(materializeLayouts(*module, *graph, missing)));
    EXPECT_EQ(before, print(*module));
    auto duplicate = *solution;
    duplicate.conversions.push_back(duplicate.conversions.front());
    EXPECT_TRUE(failed(materializeLayouts(*module, *graph, duplicate)));
    EXPECT_EQ(before, print(*module));
    auto identity = *solution;
    identity.conversions[0].targetEncoding = identity.conversions[0].sourceEncoding;
    EXPECT_TRUE(failed(materializeLayouts(*module, *graph, identity)));
    EXPECT_EQ(before, print(*module));
    EXPECT_TRUE(failed(materializeDistributedLayouts(*module, *graph, *solution,
                                                      missing.conversions)));
    EXPECT_EQ(before, print(*module));
    OpOperand *use = solution->conversions.front().use;
    Value originalSource = use->get();
    auto function = *module->getOps<func::FuncOp>().begin();
    Value changedSource = originalSource == function.getArgument(1)
                              ? function.getArgument(2) : function.getArgument(1);
    use->set(changedSource);
    auto changed = print(*module);
    EXPECT_TRUE(failed(materializeLayouts(*module, *graph, *solution)));
    EXPECT_EQ(changed, print(*module));
    use->set(originalSource);
  }
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  ASSERT_TRUE(succeeded(verify(*module)));
  unsigned count = 0;
  module->walk([&](ConvertLayoutOp conversion) {
    ++count;
    EXPECT_TRUE(isa<scf::YieldOp>(conversion->getNextNode()));
    EXPECT_EQ(cast<RankedTensorType>(conversion.getSource().getType()).getEncoding(), a);
    EXPECT_EQ(cast<RankedTensorType>(conversion.getType()).getEncoding(), b);
  });
  EXPECT_EQ(count, 2u);
}

TEST_F(LayoutMaterializationTest, SelectedForInitAndBackedgeConversions) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @loop(%x: tensor<4xf16>, %n: index) -> tensor<4xf16> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %r = scf.for %i = %zero to %n step %one iter_args(%v = %x) -> tensor<4xf16> {
        scf.yield %x : tensor<4xf16>
      }
      return %r : tensor<4xf16>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_GE(graph->getVariables().front().candidates.size(), 2u);
  auto a = graph->getVariables().front().candidates[0].value;
  auto b = graph->getVariables().front().candidates[1].value;
  for (auto &var : graph->getVariables()) {
    bool producer = var.value && isa<BlockArgument>(var.value) &&
                    isa<func::FuncOp>(var.anchor);
    var.candidates = {{producer ? a : b, kInvalidProvenanceID, 0}};
  }
  auto solution = solve(*graph);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 2u);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  ASSERT_TRUE(succeeded(verify(*module)));
  unsigned init = 0, backedge = 0;
  module->walk([&](ConvertLayoutOp conversion) {
    init += isa<scf::ForOp>(conversion->getNextNode());
    backedge += isa<scf::YieldOp>(conversion->getNextNode());
  });
  EXPECT_EQ(init, 1u);
  EXPECT_EQ(backedge, 1u);
}

TEST_F(LayoutMaterializationTest, WhileDistinctTuplesConvertAtAllThreeEdges) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @loop(%x: tensor<4xf16>, %p: i1) -> tensor<4xf16> {
      %r = scf.while (%v = %x, %pred = %p) : (tensor<4xf16>, i1) -> tensor<4xf16> {
        scf.condition(%pred) %v : tensor<4xf16>
      } do {
      ^bb0(%v: tensor<4xf16>):
        scf.yield %v, %p : tensor<4xf16>, i1
      }
      return %r : tensor<4xf16>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_GE(graph->getVariables().front().candidates.size(), 2u);
  auto a = graph->getVariables().front().candidates[0].value;
  auto b = graph->getVariables().front().candidates[1].value;
  for (auto &var : graph->getVariables()) {
    auto loop = dyn_cast_or_null<scf::WhileOp>(var.anchor);
    bool before = loop && var.value == loop.getBeforeArguments()[0];
    var.candidates = {{before ? b : a, kInvalidProvenanceID, 0}};
  }
  auto solution = solve(*graph);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 3u);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  ASSERT_TRUE(succeeded(verify(*module)));
  unsigned init = 0, condition = 0, backedge = 0;
  module->walk([&](ConvertLayoutOp conversion) {
    init += isa<scf::WhileOp>(conversion->getNextNode());
    condition += isa<scf::ConditionOp>(conversion->getNextNode());
    backedge += isa<scf::YieldOp>(conversion->getNextNode());
  });
  EXPECT_EQ(init, 1u);
  EXPECT_EQ(condition, 1u);
  EXPECT_EQ(backedge, 1u);
  auto replay = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(replay));
  auto replaySolution = solve(*replay);
  ASSERT_TRUE(succeeded(replaySolution));
  EXPECT_TRUE(replaySolution->conversions.empty());
}

TEST_F(LayoutMaterializationTest, VerifierCannotImagineMissingTransposeConversion) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @transpose(%x: tensor<4xf16>) {
      %init = tensor.empty() : tensor<4xf16>
      %t = linalg.transpose ins(%x : tensor<4xf16>) outs(%init : tensor<4xf16>) permutation = [0]
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_GE(graph->getVariables().front().candidates.size(), 2u);
  auto a = graph->getVariables().front().candidates[0].value;
  auto b = graph->getVariables().front().candidates[1].value;
  for (auto &var : graph->getVariables()) {
    bool producer = var.value && isa<BlockArgument>(var.value);
    var.candidates = {{producer ? a : b, kInvalidProvenanceID, 0}};
  }
  auto solution = solve(*graph);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 1u);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  ConvertLayoutOp conversion;
  module->walk([&](ConvertLayoutOp op) { conversion = op; });
  ASSERT_TRUE(conversion);
  conversion.getResult().replaceAllUsesWith(conversion.getSource());
  conversion.erase();
  // Transpose accepts different input/init encodings; only the layout relation
  // detects this missing physical conversion, not ordinary MLIR verification.
  ASSERT_TRUE(succeeded(verify(*module)));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(LayoutMaterializationTest, StagedSemanticVerificationFailureRollsBack) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @transpose(%x: tensor<4xf16>, %p: memref<4xf16, 3>) {
      %v = frisk.layout_view %p : memref<4xf16, 3> -> memref<4xf16, 3>
      %init = tensor.empty() : tensor<4xf16>
      %t = linalg.transpose ins(%x : tensor<4xf16>) outs(%init : tensor<4xf16>) permutation = [0]
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  Attribute a, b;
  for (const auto &var : graph->getVariables())
    if (var.kind == LayoutKind::Distributed) {
      ASSERT_GE(var.candidates.size(), 2u);
      a = var.candidates[0].value;
      b = var.candidates[1].value;
      break;
    }
  ASSERT_TRUE(a && b);
  for (auto &var : graph->getVariables()) {
    if (var.kind != LayoutKind::Distributed) continue;
    bool input = var.use || (var.value && isa<BlockArgument>(var.value));
    var.candidates = {{input ? a : b, kInvalidProvenanceID, 0}};
  }
  // Simulate an incomplete upstream relation model: ordinary graph solution
  // validation now passes, but independent staged recollection must reject.
  for (auto &constraint : graph->getConstraints())
    if (constraint.kind == ConstraintKind::TransformLayout)
      constraint.strength = ConstraintStrength::Soft;
  auto solution = solve(*graph);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_TRUE(solution->conversions.empty());
  ASSERT_TRUE(succeeded(verifySolvedLayoutGraph(*graph, *solution, *target, module->getLoc())));
  auto before = print(*module);
  std::string diagnostics;
  ScopedDiagnosticHandler quiet(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(diagnostics);
    diagnostic.print(stream);
    return success();
  });
  EXPECT_TRUE(failed(materializeLayouts(*module, *graph, *solution)));
  EXPECT_NE(diagnostics.find("transpose"), std::string::npos) << diagnostics;
  EXPECT_EQ(before, print(*module));
}

TEST_F(LayoutMaterializationTest, RelationsOnlyCollectionDoesNotGenerateCandidates) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @unencoded(%arg: tensor<4xf16>) { return }
  )mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target,
                                        LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_EQ(graph->getVariables().size(), 1u);
  EXPECT_TRUE(graph->getVariables().front().candidates.empty());
}
} // namespace
