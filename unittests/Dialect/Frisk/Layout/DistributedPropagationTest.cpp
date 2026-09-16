#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
class DistributedPropagationTest : public testing::Test {
protected:
  DistributedPropagationTest() {
    context.getOrLoadDialect<FriskDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<linalg::LinalgDialect>();
    context.getOrLoadDialect<tensor::TensorDialect>();
  }
  MLIRContext context;
  void expectDiagnostic(StringRef expected, llvm::function_ref<bool()> action) {
    std::string text;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      llvm::raw_string_ostream stream(text);
      diagnostic.print(stream);
      stream << '\n';
      return success();
    });
    EXPECT_TRUE(action());
    EXPECT_NE(text.find(expected.str()), std::string::npos) << text;
  }
  OwningOpRef<ModuleOp> dualConsumer() {
    auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @two(%a: memref<8x8xf32, 1>, %b: memref<8x8xf32, 1>,
                   %c: memref<8x8xf32, 1>) {
      %av = frisk.layout_view %a : memref<8x8xf32, 1> -> memref<8x8xf32, 1>
      %bv = frisk.layout_view %b : memref<8x8xf32, 1> -> memref<8x8xf32, 1>
      %cv = frisk.layout_view %c : memref<8x8xf32, 1> -> memref<8x8xf32, 1>
      %t = frisk.tile_load %av : memref<8x8xf32, 1> -> tensor<8x8xf32>
      frisk.tile_store %t, %bv : tensor<8x8xf32>, memref<8x8xf32, 1>
      frisk.tile_store %t, %cv : tensor<8x8xf32>, memref<8x8xf32, 1>
      return
    })mlir", &context);
    if (!module) return {};
    auto target = createSM90LayoutTarget();
    unsigned index = 0;
    module->walk([&](LayoutViewOp view) {
      LayoutVar var;
      var.shapedType = view.getResult().getType();
      SmallVector<LayoutCandidate> candidates;
      target->enumerateCandidates(var, candidates);
      view->setAttr("layout", candidates[index++ == 2 ? 1 : 0].value);
    });
    return module;
  }
};

TEST_F(DistributedPropagationTest, RealDualConsumerKeepsAlternatives) {
  auto module = dualConsumer();
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  unsigned distributed = 0;
  for (const auto &var : graph->getVariables())
    if (var.kind == LayoutKind::Distributed) {
      ++distributed;
      EXPECT_GE(var.candidates.size(), 2u);
    }
  EXPECT_EQ(distributed, 3u);
  ASSERT_TRUE(succeeded(propagateStrict(*graph)));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveBootstrapLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_TRUE(solution->conversions.empty());
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(
      *graph, *solution, *target, module->getLoc())));
}

TEST_F(DistributedPropagationTest, HardConsumerChoicesConvertExactlyOneRealUse) {
  auto module = dualConsumer();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  unsigned consumer = 0;
  for (const auto &var : graph->getVariables()) {
    if (!var.use) continue;
    Attribute selected = var.candidates[consumer++].value;
    graph->addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
                         {var.id}, var.anchor, "consumer-contract",
                         "hard consumer encoding for solver fixture", selected);
  }
  ASSERT_EQ(consumer, 2u);
  ASSERT_TRUE(succeeded(graph->finalize(module->getLoc())));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveBootstrapLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 1u);
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(*graph, *solution, *target, module->getLoc())));
  const auto &edge = solution->conversions.front();
  EXPECT_TRUE(isa<TileStoreOp>(edge.use->getOwner()));
  EXPECT_NE(edge.sourceEncoding, edge.targetEncoding);
  auto sourceID = graph->lookupVariable(edge.use->get());
  ASSERT_TRUE(sourceID);
  EXPECT_EQ(solution->assignments.lookup(*sourceID), edge.sourceEncoding);

  auto missing = *solution;
  missing.conversions.clear();
  expectDiagnostic("missing a required consumer conversion", [&] {
    return failed(verifySolvedLayoutGraph(*graph, missing, *target, module->getLoc()));
  });
  auto duplicate = *solution;
  duplicate.conversions.push_back(edge);
  expectDiagnostic("invalid, duplicate, or identity conversion", [&] {
    return failed(verifySolvedLayoutGraph(*graph, duplicate, *target, module->getLoc()));
  });
  auto identity = *solution;
  identity.conversions.front().targetEncoding = edge.sourceEncoding;
  expectDiagnostic("invalid, duplicate, or identity conversion", [&] {
    return failed(verifySolvedLayoutGraph(*graph, identity, *target, module->getLoc()));
  });
  auto unauthorized = *solution;
  unauthorized.conversions.front().constraint = 9999;
  expectDiagnostic("authorized graph edge", [&] {
    return failed(verifySolvedLayoutGraph(*graph, unauthorized, *target, module->getLoc()));
  });

  auto reordered = *graph;
  std::reverse(reordered.getConstraints().begin(), reordered.getConstraints().end());
  // Model a different valid construction order: IDs identify array entries.
  // This fixture has no region edges requiring a corresponding ID remap.
  ASSERT_TRUE(reordered.getRegionEdges().empty());
  for (auto [id, constraint] : llvm::enumerate(reordered.getConstraints()))
    constraint.id = id;
  for (auto &var : reordered.getVariables())
    std::reverse(var.candidates.begin(), var.candidates.end());
  ASSERT_TRUE(succeeded(reordered.finalize(module->getLoc())));
  auto again = solveBootstrapLayoutGraph(reordered, *target);
  ASSERT_TRUE(succeeded(again));
  EXPECT_EQ(again->assignments, solution->assignments);
  ASSERT_EQ(again->conversions.size(), 1u);
  EXPECT_EQ(again->conversions.front().use, edge.use);
}

TEST_F(DistributedPropagationTest, UnencodedNonSquareTransposeBidirectionalFixedPoint) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @transpose(%arg: tensor<4x8xf32>) -> tensor<8x4xf32> {
      %init = tensor.empty() : tensor<8x4xf32>
      %out = linalg.transpose ins(%arg : tensor<4x8xf32>)
        outs(%init : tensor<8x4xf32>) permutation = [1, 0]
      return %out : tensor<8x4xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  SmallVector<size_t> sizes;
  for (const auto &var : graph->getVariables()) {
    EXPECT_LE(var.candidates.size(), 4u);
    EXPECT_GE(var.candidates.size(), 2u);
    sizes.push_back(var.candidates.size());
  }
  for (const auto &relation : graph->getConstraints()) {
    if (relation.kind != ConstraintKind::TransformLayout) continue;
    auto src = relation.vars[0], dst = relation.vars[1];
    for (const auto &candidate : graph->getVariable(src).candidates) {
      auto forward = projectLayoutCandidate(*graph, relation, src, candidate.value, dst);
      ASSERT_TRUE(succeeded(forward));
      auto backward = projectLayoutCandidate(*graph, relation, dst, *forward, src);
      ASSERT_TRUE(succeeded(backward));
      EXPECT_EQ(*backward, candidate.value);
    }
  }
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  for (auto [index, var] : llvm::enumerate(graph->getVariables()))
    EXPECT_EQ(var.candidates.size(), sizes[index]);
  auto solution = solveBootstrapLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_TRUE(solution->conversions.empty());
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(*graph, *solution, *target, module->getLoc())));
}

TEST_F(DistributedPropagationTest, RejectsUnsupportedTensorShapesAndOperations) {
  for (StringRef source : {
      "func.func @unranked(%arg: tensor<*xf32>) { return }",
      "func.func @dynamic(%arg: tensor<?xf32>) { return }",
      "func.func @odd(%arg: tensor<3xf32>) { return }",
      "func.func @unit(%arg: tensor<1xf32>) { return }",
      "func.func private @external(tensor<8xf32>)"}) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    auto target = createSM90LayoutTarget();
    StringRef expected = source.contains("unranked") ? "requires ranked tensor" :
                         source.contains("external") ? "external tensor signature" :
                         "power-of-two tile extents";
    expectDiagnostic(expected, [&] {
      return failed(collectLayoutConstraints(*module, *target));
    });
  }
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @unknown(%arg: tensor<8xf32>) -> f32 {
      %c0 = arith.constant 0 : index
      %x = tensor.extract %arg[%c0] : tensor<8xf32>
      return %x : f32
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  expectDiagnostic("operation has no layout constraint model", [&] {
    return failed(collectLayoutConstraints(*module, *target));
  });
}

TEST_F(DistributedPropagationTest, ConversionCountPrecedesCandidateOrdinal) {
  auto module = dualConsumer();
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  Attribute desired;
  for (const auto &var : graph->getVariables()) {
    if (!var.use) continue;
    desired = var.candidates[1].value;
    graph->addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
                         {var.id}, var.anchor, "consumer-contract", "prefer common hard target", desired);
  }
  ASSERT_TRUE(succeeded(graph->finalize(module->getLoc())));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveBootstrapLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_TRUE(solution->conversions.empty());
  for (const auto &var : graph->getVariables())
    if (var.kind == LayoutKind::Distributed)
      EXPECT_EQ(solution->assignments.lookup(var.id), desired);
}

TEST_F(DistributedPropagationTest, ExistingExplicitConversionsAreNotReinserted) {
  auto target = createSM90LayoutTarget();
  Builder builder(&context);
  auto type = RankedTensorType::get({8, 8}, builder.getF32Type());
  LayoutVar var;
  var.kind = LayoutKind::Distributed;
  var.shapedType = type;
  SmallVector<LayoutCandidate> candidates;
  target->enumerateCandidates(var, candidates);
  ASSERT_GE(candidates.size(), 2u);
  for (unsigned destination : {0u, 1u}) {
    auto module = parseSourceString<ModuleOp>(
        "func.func @existing(%arg: tensor<8x8xf32>) { return }", &context);
    ASSERT_TRUE(module);
    auto function = module->lookupSymbol<func::FuncOp>("existing");
    auto sourceType = RankedTensorType::get(type.getShape(), type.getElementType(), candidates[0].value);
    auto targetType = RankedTensorType::get(type.getShape(), type.getElementType(), candidates[destination].value);
    function.setType(builder.getFunctionType({sourceType}, {}));
    function.getArgument(0).setType(sourceType);
    OpBuilder ops(&context);
    ops.setInsertionPointToStart(&function.getBody().front());
    ops.create<ConvertLayoutOp>(function.getLoc(), targetType, function.getArgument(0));
    ASSERT_TRUE(succeeded(verify(*module)));
    auto graph = collectLayoutConstraints(*module, *target);
    ASSERT_TRUE(succeeded(graph));
    ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
    auto solution = solveBootstrapLayoutGraph(*graph, *target);
    ASSERT_TRUE(succeeded(solution));
    EXPECT_TRUE(solution->conversions.empty());
    EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(*graph, *solution, *target, module->getLoc())));
  }
}

TEST_F(DistributedPropagationTest, RealUseComponentsRespectEightVariableBound) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @chain(%arg: tensor<8xf32>) {
      %0 = arith.negf %arg : tensor<8xf32>
      %1 = arith.negf %0 : tensor<8xf32>
      %2 = arith.negf %1 : tensor<8xf32>
      %3 = arith.negf %2 : tensor<8xf32>
      %4 = arith.negf %3 : tensor<8xf32>
      %5 = arith.negf %4 : tensor<8xf32>
      %6 = arith.negf %5 : tensor<8xf32>
      %7 = arith.negf %6 : tensor<8xf32>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_EQ(graph->getVariables().size(), 9u);
  expectDiagnostic("too many variables", [&] {
    return failed(solveBootstrapLayoutGraph(*graph, *target));
  });
}

TEST_F(DistributedPropagationTest, DomainOverflowIsNotSilentlyTruncated) {
  auto module = dualConsumer();
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  for (auto &var : graph->getVariables()) {
    if (var.kind != LayoutKind::Distributed || !var.value) continue;
    LayoutConstraint transpose;
    transpose.kind = ConstraintKind::TransformLayout;
    transpose.vars = {var.id, var.id};
    transpose.coordinateTransform = Builder(&context).getDenseI64ArrayAttr({1, 0});
    auto fifth = projectLayoutCandidate(*graph, transpose, var.id, var.candidates[0].value, var.id);
    ASSERT_TRUE(succeeded(fifth));
    ASSERT_TRUE(llvm::none_of(var.candidates, [&](auto candidate) { return candidate.value == *fifth; }));
    var.candidates.push_back({*fifth, kInvalidProvenanceID, 5});
  }
  expectDiagnostic("candidate domain is too large", [&] {
    return failed(solveBootstrapLayoutGraph(*graph, *target));
  });
}

TEST_F(DistributedPropagationTest, FunctionResultBindingsSurviveFinalize) {
  auto module = parseSourceString<ModuleOp>(
      "func.func @id(%arg: tensor<8xf32>) -> tensor<8xf32> { return %arg : tensor<8xf32> }", &context);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto function = module->lookupSymbol<func::FuncOp>("id");
  auto arg = graph->lookupVariable(function.getArgument(0));
  ASSERT_TRUE(arg);
  EXPECT_EQ(graph->getVariable(*arg).value, function.getArgument(0));
  unsigned slots = 0;
  for (const auto &var : graph->getVariables()) {
    if (!var.functionResult) continue;
    EXPECT_EQ(*var.functionResult, 0u);
    EXPECT_EQ(var.anchor, function.getOperation());
    EXPECT_FALSE(var.value);
    ++slots;
  }
  EXPECT_EQ(slots, 1u);
}

TEST_F(DistributedPropagationTest, CustomNamedTransposePreservesEndpointEncodings) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #smap = #frisk.bit_linear<inputs = ["lane"], input_bits = [3],
      outputs = ["row", "column"], output_bits = [1, 2],
      matrix = dense<[[1, 0, 0], [0, 1, 0], [0, 0, 1]]> : tensor<3x3xi1>>
    #dmap = #frisk.bit_linear<inputs = ["lane"], input_bits = [3],
      outputs = ["width", "height"], output_bits = [2, 1],
      matrix = dense<[[0, 1, 0], [0, 0, 1], [1, 0, 0]]> : tensor<3x3xi1>>
    #src = #frisk.distributed<map = #smap, topology = [1, 8, 1, 1, 1], replication = 1>
    #dst = #frisk.distributed<map = #dmap, topology = [1, 8, 1, 1, 1], replication = 1>
    func.func @custom(%arg: tensor<2x4xf32, #src>) {
      %init = tensor.empty() : tensor<4x2xf32, #dst>
      %out = linalg.transpose ins(%arg : tensor<2x4xf32, #src>)
        outs(%init : tensor<4x2xf32, #dst>) permutation = [1, 0]
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  for (const auto &relation : graph->getConstraints()) {
    if (relation.kind != ConstraintKind::TransformLayout) continue;
    auto source = relation.vars[0], destination = relation.vars[1];
    Attribute src = graph->getVariable(source).candidates.front().value;
    Attribute dst = graph->getVariable(destination).candidates.front().value;
    auto forward = projectLayoutCandidate(*graph, relation, source, src, destination);
    auto reverse = projectLayoutCandidate(*graph, relation, destination, dst, source);
    ASSERT_TRUE(succeeded(forward));
    ASSERT_TRUE(succeeded(reverse));
    EXPECT_EQ(*forward, dst);
    EXPECT_EQ(*reverse, src);
    EXPECT_FALSE(layoutEncodingsEqual(src, dst));
    auto encoding = cast<DistributedEncodingAttr>(src);
    auto map = cast<BitLinearLayoutMapAttr>(encoding.getMap());
    Builder attributes(&context);
    auto relabeledMap = BitLinearLayoutMapAttr::get(&context,
        map.getInputNames(), map.getInputBitWidths(),
        attributes.getArrayAttr({attributes.getStringAttr("consumer_axis0"),
                                 attributes.getStringAttr("consumer_axis1")}),
        map.getOutputBitWidths(), map.getMatrix());
    auto relabeled = DistributedEncodingAttr::get(&context, relabeledMap,
        encoding.getTopology(), encoding.getReplication());
    EXPECT_FALSE(layoutEncodingsEqual(src, relabeled));
    // A synthetic use's original type names are not its hard expected names.
    EXPECT_TRUE(layoutRelationCompatible(*graph, relation, source, relabeled,
                                         destination, dst));
    EXPECT_TRUE(layoutRelationCompatible(*graph, relation, destination, dst,
                                         source, relabeled));
  }
  auto solution = solveBootstrapLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_TRUE(solution->conversions.empty());
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(*graph, *solution, *target, module->getLoc())));
}

TEST_F(DistributedPropagationTest, RankZeroIsRejectedBeforeCandidateConstruction) {
  auto module = parseSourceString<ModuleOp>(
      "func.func @scalar(%arg: tensor<f32>) { return }", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  expectDiagnostic("nonzero-rank", [&] {
    return failed(collectLayoutConstraints(*module, *target));
  });
  LayoutVar scalar;
  scalar.kind = LayoutKind::Distributed;
  scalar.shapedType = RankedTensorType::get({}, Builder(&context).getF32Type());
  SmallVector<LayoutCandidate> candidates;
  std::string diagnostics;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(diagnostics);
    diagnostic.print(stream);
    return success();
  });
  target->enumerateCandidates(scalar, candidates);
  EXPECT_TRUE(candidates.empty());
  EXPECT_TRUE(diagnostics.empty()) << diagnostics;
}
} // namespace
