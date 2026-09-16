#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "gtest/gtest.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
FailureOr<SmallVector<int64_t>> evaluateLayout(Attribute map, ArrayRef<int64_t> point) {
  auto affine = dyn_cast<AffineLayoutMapAttr>(map);
  if (!affine)
    return failure();
  Builder builder(map.getContext());
  SmallVector<Attribute> args, outputs;
  for (int64_t coordinate : point)
    args.push_back(builder.getIndexAttr(coordinate));
  if (failed(affine.getAffineMap().getValue().constantFold(args, outputs)))
    return failure();
  SmallVector<int64_t> values;
  for (Attribute output : outputs)
    values.push_back(cast<IntegerAttr>(output).getInt());
  return values;
}
class LayoutAliasAnalysisTest : public testing::Test {
protected:
  LayoutAliasAnalysisTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>();
    quiet = std::make_unique<ScopedDiagnosticHandler>(
        &context, [](Diagnostic &) { return success(); });
  }
  OwningOpRef<ModuleOp> parse(StringRef text) {
    return parseSourceString<ModuleOp>(text, &context);
  }
  Value endpoint(ModuleOp module) {
    Value result;
    module.walk([&](LayoutViewOp view) { result = view.getResult(); });
    return result;
  }
  StorageLayoutAttr layout(ArrayRef<int64_t> shape, AffineExpr bytes,
                           AffineExpr bits = {}, int64_t alignment = 1) {
    Builder b(&context);
    SmallVector<Attribute> names;
    for (unsigned i = 0; i < shape.size(); ++i)
      names.push_back(b.getStringAttr("d" + Twine(i)));
    if (!bits)
      bits = b.getAffineConstantExpr(0);
    auto map = AffineLayoutMapAttr::get(
        &context, b.getArrayAttr(names), b.getDenseI64ArrayAttr(shape),
        b.getStrArrayAttr({"byte_offset", "bit_offset"}),
        b.getDenseI64ArrayAttr({1048576, 8}),
        AffineMapAttr::get(AffineMap::get(shape.size(), 0, {bytes, bits}, &context)));
    return StorageLayoutAttr::get(
        &context, map, MemorySpaceAttr::get(&context, attr::MemorySpace::Shared),
        b.getI64IntegerAttr(alignment), b.getI64IntegerAttr(1));
  }
  MLIRContext context;
  std::unique_ptr<ScopedDiagnosticHandler> quiet;
};

TEST_F(LayoutAliasAnalysisTest, OffsetThreeRootAndStridedSubview) {
  auto module = parse(R"mlir(
    func.func @f(%root: memref<4x8xi32, strided<[8, 1], offset: 3>, 3>) {
      %s = memref.subview %root[1, 2] [2, 3] [1, 2] :
        memref<4x8xi32, strided<[8, 1], offset: 3>, 3> to
        memref<2x3xi32, strided<[8, 2], offset: 13>, 3>
      %v = frisk.layout_view %s : memref<2x3xi32, strided<[8, 2], offset: 13>, 3>
        -> memref<2x3xi32, strided<[8, 2], offset: 13>, 3>
      return
    })mlir");
  ASSERT_TRUE(module);
  auto info = analyzeStorageAlias(endpoint(*module));
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->lowerBit, 96u);
  EXPECT_EQ(info->upperBit, 1120u);
  auto root = analyzeStorageAlias(info->root);
  ASSERT_TRUE(succeeded(root));
  auto linear = buildRootLinearStorageCandidate(*root);
  ASSERT_TRUE(succeeded(linear));
  auto projected = projectStorageAliasCandidate(*root, *linear, *info);
  ASSERT_TRUE(succeeded(projected));
  auto out = evaluateLayout(projected->getMap(), {0, 0});
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ((*out)[0], 52);
  out = evaluateLayout(projected->getMap(), {1, 2});
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ((*out)[0], 100);
  EXPECT_EQ(verifyStorageAliasCandidate(*info, *projected).status, ProofStatus::Proven);
  EXPECT_EQ(proveStorageAliasCompatible(*root, *linear, *info, *projected).status,
            ProofStatus::Proven);
  EXPECT_TRUE(failed(projectStorageAliasCandidate(*info, *projected, *root)));
  auto d0 = getAffineDimExpr(0, &context), d1 = getAffineDimExpr(1, &context);
  EXPECT_EQ(verifyStorageAliasCandidate(*info, layout({2, 3}, 32*d0+8*d1)).status,
            ProofStatus::Disproven);
}

TEST_F(LayoutAliasAnalysisTest, NestedRankReductionRetainsUnitAxis) {
  auto module = parse(R"mlir(
    func.func @f(%r: memref<3x1x8xi32, 3>) {
      %a = memref.subview %r[1, 0, 1] [1, 1, 4] [1, 1, 2] :
        memref<3x1x8xi32, 3> to memref<1x4xi32, strided<[8, 2], offset: 9>, 3>
      %b = memref.subview %a[0, 1] [1, 2] [1, 1] :
        memref<1x4xi32, strided<[8, 2], offset: 9>, 3> to
        memref<1x2xi32, strided<[8, 2], offset: 11>, 3>
      %v = frisk.layout_view %b : memref<1x2xi32, strided<[8, 2], offset: 11>, 3>
        -> memref<1x2xi32, strided<[8, 2], offset: 11>, 3>
      return
    })mlir");
  ASSERT_TRUE(module);
  auto info = analyzeStorageAlias(endpoint(*module));
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->viewToRoot.getNumDims(), 2u);
  EXPECT_EQ(info->viewToRoot.getNumResults(), 3u);
  auto root = analyzeStorageAlias(info->root);
  ASSERT_TRUE(succeeded(root));
  auto linear = buildRootLinearStorageCandidate(*root);
  ASSERT_TRUE(succeeded(linear));
  auto candidate = projectStorageAliasCandidate(*root, *linear, *info);
  ASSERT_TRUE(succeeded(candidate));
  auto out = evaluateLayout(candidate->getMap(), {0, 1});
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ((*out)[0], 52);
}

TEST_F(LayoutAliasAnalysisTest, RejectsUnprovedRoots) {
  for (StringRef type : {"memref<?xi32, 3>", "memref<2x2xi32, strided<[1, 1]>, 3>",
                         "memref<2xi32, strided<[-1], offset: 1>, 3>",
                         "memref<0xi32, 3>",
                         "memref<2xi32, strided<[9223372036854775807]>, 3>"}) {
    auto module = parse(("func.func @f(%r: " + type + ") { return }").str());
    ASSERT_TRUE(module) << type.str();
    EXPECT_TRUE(failed(analyzeStorageAlias(
        module->lookupSymbol<func::FuncOp>("f").getArgument(0)))) << type.str();
  }
}

TEST_F(LayoutAliasAnalysisTest, SemanticEqualityAndDifferentCoordinateCollision) {
  auto module = parse("func.func @f(%r: memref<4xi32, 3>) { return }");
  ASSERT_TRUE(module);
  auto root = analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(root));
  auto d = getAffineDimExpr(0, &context);
  StorageAliasInfo a = *root, b = *root;
  a.viewType = b.viewType = MemRefType::get({2}, IntegerType::get(&context, 32), {}, 3);
  a.viewToRoot = AffineMap::get(1, 0, {d}, &context);
  b.viewToRoot = AffineMap::get(1, 0, {d+2}, &context);
  auto ca = layout({2}, 4*d);
  auto cb = layout({2}, 4*d+8);
  EXPECT_EQ(proveStorageAliasCompatible(a, ca, b, cb).status, ProofStatus::Proven);
  auto collision = proveStorageAliasCompatible(a, ca, b, ca);
  EXPECT_EQ(collision.status, ProofStatus::Disproven);
  EXPECT_FALSE(collision.counterexample.empty());
  auto footprintA = buildStorageAliasFootprint(a, ca);
  auto footprintB = buildStorageAliasFootprint(b, cb);
  EXPECT_EQ(footprintA.proof.status, ProofStatus::Proven);
  EXPECT_EQ(footprintA.entries.size(), 2u);
  EXPECT_EQ(proveStorageAliasFootprints(footprintA, footprintB).status, ProofStatus::Proven);
  auto equivalent = layout({2}, 4*((d+2)-2));
  EXPECT_EQ(proveStorageAliasCompatible(a, ca, a, equivalent).status, ProofStatus::Proven);
  EXPECT_EQ(proveStorageAliasCompatible(a, ca, a, cb).status, ProofStatus::Disproven);
  EXPECT_EQ(verifyStorageAliasCandidate(a, layout({2}, 0*d)).status, ProofStatus::Disproven);
}

TEST_F(LayoutAliasAnalysisTest, PackedBitRangesAndLowerBound) {
  auto module = parse("func.func @f(%r: memref<4xi3, strided<[1], offset: 1>, 3>) { return }");
  ASSERT_TRUE(module);
  auto info = analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->lowerBit, 3u);
  EXPECT_EQ(info->upperBit, 15u);
  auto candidate = buildRootLinearStorageCandidate(*info);
  ASSERT_TRUE(succeeded(candidate));
  EXPECT_EQ(verifyStorageAliasCandidate(*info, *candidate).status, ProofStatus::Proven);
  auto d = getAffineDimExpr(0, &context);
  EXPECT_EQ(verifyStorageAliasCandidate(*info, layout({4}, (d*3).floorDiv(8), (d*3)%8)).status,
            ProofStatus::Disproven);
  EXPECT_EQ(verifyStorageAliasCandidate(*info, layout({4}, (d*2+3).floorDiv(8), (d*2+3)%8)).status,
            ProofStatus::Disproven);
}

TEST_F(LayoutAliasAnalysisTest, XorProjectionUsesIntegerCarryAndPositiveStride) {
  auto module = parse("func.func @f(%r: memref<8xi8, 3>) { return }");
  ASSERT_TRUE(module);
  auto root = analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(root));
  auto map = parseAttribute(R"mlir(#frisk.bit_linear<
    inputs = ["d0"], input_bits = [3], outputs = ["byte_offset", "bit_offset"],
    output_bits = [3, 3], matrix = dense<[[1, 1, 0], [0, 1, 0], [0, 0, 1],
      [0, 0, 0], [0, 0, 0], [0, 0, 0]]> : tensor<6x3xi1>>
  )mlir", &context);
  ASSERT_TRUE(map);
  Builder builder(&context);
  auto candidate = StorageLayoutAttr::get(&context, map,
      MemorySpaceAttr::get(&context, attr::MemorySpace::Shared),
      builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(1));
  auto d = getAffineDimExpr(0, &context);
  for (int64_t stride : {1, 3}) {
    StorageAliasInfo child = *root;
    child.viewType = MemRefType::get({3}, builder.getI8Type(), {}, 3);
    child.viewToRoot = AffineMap::get(1, 0, {stride*d+1}, &context);
    auto projected = projectStorageAliasCandidate(*root, candidate, child);
    ASSERT_TRUE(succeeded(projected));
    for (int64_t x = 0; x < 3; ++x) {
      int64_t q = stride*x+1;
      auto out = evaluateLayout(projected->getMap(), {x});
      ASSERT_TRUE(succeeded(out));
      EXPECT_EQ((*out)[0], q ^ ((q >> 1) & 1));
    }
  }
  auto identity = projectStorageAliasCandidate(*root, candidate, *root);
  ASSERT_TRUE(succeeded(identity));
  EXPECT_EQ(*identity, candidate);
}

TEST_F(LayoutAliasAnalysisTest, ProofBudgetAndAlignmentAreExplicit) {
  auto module = parse("func.func @f(%r: memref<65537xi8, 3>) { return }");
  ASSERT_TRUE(module);
  auto info = analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(info));
  auto d = getAffineDimExpr(0, &context);
  EXPECT_EQ(verifyStorageAliasCandidate(*info, layout({65537}, d)).status, ProofStatus::Unknown);
  info->viewType = MemRefType::get({4}, IntegerType::get(&context, 8), {}, 3);
  EXPECT_EQ(verifyStorageAliasCandidate(*info, layout({4}, d, {}, 16)).status, ProofStatus::Disproven);
  info->rootAlignment = 16;
  EXPECT_EQ(verifyStorageAliasCandidate(*info, layout({4}, d, {}, 16)).status, ProofStatus::Proven);
}

TEST_F(LayoutAliasAnalysisTest, RejectsMismatchedDeclaredDomain) {
  auto module = parse("func.func @f(%r: memref<4xi8, 3>) { return }");
  ASSERT_TRUE(module);
  auto info = analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(verifyStorageAliasCandidate(*info, layout({8}, getAffineDimExpr(0, &context))).status,
            ProofStatus::Disproven);
}

TEST_F(LayoutAliasAnalysisTest, CastCanHideThenRecoverStaticSourceFacts) {
  auto module = parse(R"mlir(
    func.func @f(%r: memref<4xi32, 3>) {
      %a = memref.cast %r : memref<4xi32, 3> to memref<?xi32, 3>
      %b = memref.cast %a : memref<?xi32, 3> to memref<4xi32, 3>
      %v = frisk.layout_view %b : memref<4xi32, 3> -> memref<4xi32, 3>
      return
    })mlir");
  ASSERT_TRUE(module);
  auto info = analyzeStorageAlias(endpoint(*module));
  ASSERT_TRUE(succeeded(info));
  EXPECT_TRUE(info->viewToRoot.isIdentity());
  auto unproved = parse(R"mlir(
    func.func @f(%r: memref<?xi32, 3>) {
      %a = memref.cast %r : memref<?xi32, 3> to memref<4xi32, 3>
      %v = frisk.layout_view %a : memref<4xi32, 3> -> memref<4xi32, 3>
      return
    })mlir");
  ASSERT_TRUE(unproved);
  EXPECT_TRUE(failed(analyzeStorageAlias(endpoint(*unproved))));
}

TEST_F(LayoutAliasAnalysisTest, RejectsDynamicSubviewMetadataAndUnknownViews) {
  auto module = parse(R"mlir(
    func.func @f(%r: memref<8xi32, 3>, %offset: index) {
      %s = memref.subview %r[%offset] [4] [1] : memref<8xi32, 3>
        to memref<4xi32, strided<[1], offset: ?>, 3>
      %v = frisk.layout_view %s : memref<4xi32, strided<[1], offset: ?>, 3>
        -> memref<4xi32, strided<[1], offset: ?>, 3>
      return
    })mlir");
  ASSERT_TRUE(module);
  std::string diagnostic;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &message) {
    diagnostic = message.str();
    return success();
  });
  EXPECT_TRUE(failed(analyzeStorageAlias(endpoint(*module))));
  EXPECT_NE(diagnostic.find("memref.subview"), std::string::npos);
  auto unknown = parse(R"mlir(
    func.func @f(%r: memref<8xi32, 3>) {
      %s = memref.reinterpret_cast %r to offset: [0], sizes: [4], strides: [1]
        : memref<8xi32, 3> to memref<4xi32, 3>
      %v = frisk.layout_view %s : memref<4xi32, 3> -> memref<4xi32, 3>
      return
    })mlir");
  ASSERT_TRUE(unknown);
  EXPECT_TRUE(failed(analyzeStorageAlias(endpoint(*unknown))));
  EXPECT_NE(diagnostic.find("memref.reinterpret_cast"), std::string::npos);
}

TEST_F(LayoutAliasAnalysisTest, RejectsInvalidSubviewBoundsAndArithmetic) {
  for (StringRef metadata : {"[7] [2] [1]", "[0] [2] [0]", "[0] [2] [-1]",
                             "[0] [0] [1]", "[9223372036854775807] [2] [2]"}) {
    std::string text = ("func.func @f(%r: memref<8xi32, 3>) { %s = memref.subview %r" +
        metadata + " : memref<8xi32, 3> to memref<?xi32, strided<[?], offset: ?>, 3> return }").str();
    auto module = parseSourceString<ModuleOp>(text, ParserConfig(&context, false));
    ASSERT_TRUE(module) << metadata.str();
    Value subview;
    module->walk([&](memref::SubViewOp op) { subview = op.getResult(); });
    // A static cast endpoint forces normalization to prove, rather than trust,
    // the invalid source path metadata. Bypass verifier only for adversarial IR.
    OpBuilder builder(&context);
    builder.setInsertionPointAfter(subview.getDefiningOp());
    auto castOp = builder.create<memref::CastOp>(subview.getLoc(),
        MemRefType::get({2}, builder.getI32Type(), StridedLayoutAttr::get(&context,
            ShapedType::kDynamic, {ShapedType::kDynamic}), builder.getI64IntegerAttr(3)), subview);
    EXPECT_TRUE(failed(analyzeStorageAlias(castOp.getResult()))) << metadata.str();
  }
}

TEST_F(LayoutAliasAnalysisTest, GlobalAccessesShareIdentityAndAllocationAlignment) {
  auto module = parse(R"mlir(
    memref.global "private" @g : memref<4xi32, 3> = uninitialized {alignment = 16 : i64}
    func.func @f() {
      %a = memref.get_global @g : memref<4xi32, 3>
      %b = memref.get_global @g : memref<4xi32, 3>
      %c = memref.alloc() {alignment = 32 : i64} : memref<4xi32, 3>
      return
    })mlir");
  ASSERT_TRUE(module);
  SmallVector<StorageAliasInfo> globals;
  module->walk([&](memref::GetGlobalOp op) {
    auto info = analyzeStorageAlias(op.getResult());
    ASSERT_TRUE(succeeded(info));
    globals.push_back(*info);
  });
  ASSERT_EQ(globals.size(), 2u);
  EXPECT_EQ(globals[0].root, globals[1].root);
  EXPECT_EQ(globals[0].rootAlignment, 16u);
  module->walk([&](memref::AllocOp op) {
    auto info = analyzeStorageAlias(op.getResult());
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->rootAlignment, 32u);
  });
}

TEST_F(LayoutAliasAnalysisTest, ChildProjectsOnlyToUniqueContainedLattice) {
  auto module = parse("func.func @f(%r: memref<16xi8, 3>) { return }");
  ASSERT_TRUE(module);
  auto root = analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(root));
  auto d = getAffineDimExpr(0, &context);
  auto child = *root, inside = *root;
  child.viewType = MemRefType::get({6}, IntegerType::get(&context, 8), {}, 3);
  inside.viewType = MemRefType::get({2}, IntegerType::get(&context, 8), {}, 3);
  child.viewToRoot = AffineMap::get(1, 0, {2*d+1}, &context);
  inside.viewToRoot = AffineMap::get(1, 0, {4*d+3}, &context);
  auto candidate = layout({6}, 2*d+1);
  auto projected = projectStorageAliasCandidate(child, candidate, inside);
  ASSERT_TRUE(succeeded(projected));
  auto out = evaluateLayout(projected->getMap(), {1});
  ASSERT_TRUE(succeeded(out));
  EXPECT_EQ((*out)[0], 7);
  inside.viewToRoot = AffineMap::get(1, 0, {4*d+2}, &context);
  EXPECT_TRUE(failed(projectStorageAliasCandidate(child, candidate, inside)));
  EXPECT_TRUE(failed(projectStorageAliasCandidate(child, candidate, *root)));
}

TEST_F(LayoutAliasAnalysisTest, CheckedAffineOverflowIsUnknown) {
  auto module = parse("func.func @f(%r: memref<4xi8, 3>) { return }");
  ASSERT_TRUE(module);
  auto root = analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(root));
  auto d = getAffineDimExpr(0, &context);
  auto expression = (d*int64_t(9223372036854775807LL)).floorDiv(int64_t(9223372036854775806LL));
  EXPECT_EQ(verifyStorageAliasCandidate(*root, layout({4}, expression)).status, ProofStatus::Unknown);
}
} // namespace
