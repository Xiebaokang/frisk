#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

namespace mlir::frisk {
namespace {

class ProductLayoutTest : public testing::Test {
protected:
  ProductLayoutTest() {
    context.getOrLoadDialect<FriskDialect>();
    diagnosticHandler = std::make_unique<ScopedDiagnosticHandler>(
        &context, [](Diagnostic &) { return success(); });
  }

  Attribute parse(StringRef text) { return parseAttribute(text, &context); }

  MLIRContext context;
  std::unique_ptr<ScopedDiagnosticHandler> diagnosticHandler;
};

TEST_F(ProductLayoutTest, AffineMapSupportsNonPowerOfTwoExtent) {
  auto map = dyn_cast_or_null<AffineLayoutMapAttr>(parse(R"mlir(#frisk.affine_layout<inputs = ["i"], input_extents = [6],
      outputs = ["m"], output_extents = [6],
      map = affine_map<(d0) -> (d0)>>
  )mlir"));
  ASSERT_TRUE(map);
  EXPECT_EQ(checkCoverage(map, {6}).status, ProofStatus::Proven);
  EXPECT_EQ(checkInjectivity(map, {6}).status, ProofStatus::Proven);
}

TEST_F(ProductLayoutTest, ProductCoversRaggedShapeWithoutCarry) {
  auto product = dyn_cast_or_null<ProductLayoutMapAttr>(parse(R"mlir(#frisk.product<
      outer = #frisk.affine_layout<inputs = ["outer_m"], input_extents = [2],
        outputs = ["m"], output_extents = [6],
        map = affine_map<(d0) -> (d0 * 4)>>,
      inner = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
        outputs = ["m"], output_bits = [2],
        matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>,
      split_extents = [4]>
  )mlir"));
  ASSERT_TRUE(product);
  EXPECT_EQ(checkCoverage(product, {6}).status, ProofStatus::Proven);
  EXPECT_EQ(checkInjectivity(product, {2, 4}).status,
            ProofStatus::Proven);
  auto canonical = product.canonicalizeMap();
  ASSERT_TRUE(succeeded(canonical));
  EXPECT_EQ(*canonical, Attribute(product));
}

TEST_F(ProductLayoutTest, RejectsUnalignedOuterBase) {
  EXPECT_FALSE(parse(R"mlir(#frisk.product<
      outer = #frisk.affine_layout<inputs = ["outer_m"], input_extents = [2],
        outputs = ["m"], output_extents = [8],
        map = affine_map<(d0) -> (d0 * 3)>>,
      inner = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
        outputs = ["m"], output_bits = [2],
        matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>,
      split_extents = [4]>
  )mlir"));

  EXPECT_FALSE(parse(R"mlir(#frisk.product<
      outer = #frisk.affine_layout<inputs = ["outer_m"], input_extents = [2],
        outputs = ["m"], output_extents = [6],
        map = affine_map<(d0) -> (d0 * 3)>>,
      inner = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
        outputs = ["m"], output_bits = [2],
        matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>,
      split_extents = [3]>
  )mlir"));
}

TEST_F(ProductLayoutTest, DynamicAffineProofRemainsUnknown) {
  auto map = dyn_cast_or_null<AffineLayoutMapAttr>(parse(R"mlir(#frisk.affine_layout<
      inputs = ["i"], input_extents = [-9223372036854775808], outputs = ["m"],
      output_extents = [-9223372036854775808], map = affine_map<(d0)[s0] -> (d0)>>
  )mlir"));
  ASSERT_TRUE(map);
  EXPECT_EQ(checkCoverage(map, {6}).status, ProofStatus::Unknown);
}

TEST_F(ProductLayoutTest, PermutesAndProjectsAffineOutputsByName) {
  auto map = dyn_cast_or_null<AffineLayoutMapAttr>(parse(R"mlir(#frisk.affine_layout<inputs = ["i", "j"], input_extents = [2, 3],
      outputs = ["m", "n"], output_extents = [2, 3],
      map = affine_map<(d0, d1) -> (d0, d1)>>
  )mlir"));
  ASSERT_TRUE(map);

  auto projected = projectLayoutMap(map, {"n"});
  ASSERT_TRUE(succeeded(projected));
  auto projectedAffine = dyn_cast<AffineLayoutMapAttr>(*projected);
  ASSERT_TRUE(projectedAffine);
  ASSERT_EQ(projectedAffine.getOutputNames().size(), 1u);
  EXPECT_EQ(cast<StringAttr>(projectedAffine.getOutputNames()[0]).getValue(),
            "n");
  EXPECT_EQ(projectedAffine.getAffineMap().getValue().getResult(0),
            getAffineDimExpr(1, &context));

  auto permuted = permuteLayoutMap(map, {"n", "m"});
  ASSERT_TRUE(succeeded(permuted));
  auto permutedAffine = dyn_cast<AffineLayoutMapAttr>(*permuted);
  ASSERT_TRUE(permutedAffine);
  EXPECT_EQ(permutedAffine.getAffineMap().getValue().getResult(0),
            getAffineDimExpr(1, &context));
  EXPECT_EQ(permutedAffine.getAffineMap().getValue().getResult(1),
            getAffineDimExpr(0, &context));
}

TEST_F(ProductLayoutTest, ComposesAffineMapsInRhsThenLhsOrder) {
  auto rhs = dyn_cast_or_null<AffineLayoutMapAttr>(parse(R"mlir(#frisk.affine_layout<inputs = ["i"], input_extents = [4],
      outputs = ["x"], output_extents = [5],
      map = affine_map<(d0) -> (d0 + 1)>>
  )mlir"));
  auto lhs = dyn_cast_or_null<AffineLayoutMapAttr>(parse(R"mlir(#frisk.affine_layout<inputs = ["x"], input_extents = [5],
      outputs = ["m"], output_extents = [10],
      map = affine_map<(d0) -> (d0 * 2)>>
  )mlir"));
  ASSERT_TRUE(rhs);
  ASSERT_TRUE(lhs);

  auto composed = composeLayoutMaps(lhs, rhs);
  ASSERT_TRUE(succeeded(composed));
  auto affine = dyn_cast<AffineLayoutMapAttr>(*composed);
  ASSERT_TRUE(affine);
  AffineExpr d0 = getAffineDimExpr(0, &context);
  EXPECT_EQ(affine.getAffineMap().getValue().getResult(0), (d0 + 1) * 2);
}

} // namespace
} // namespace mlir::frisk
