#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

namespace mlir::frisk {
namespace {

class BitLinearLayoutTest : public testing::Test {
protected:
  BitLinearLayoutTest() {
    context.getOrLoadDialect<FriskDialect>();
    diagnosticHandler = std::make_unique<ScopedDiagnosticHandler>(
        &context, [](Diagnostic &) { return success(); });
  }

  BitLinearLayoutMapAttr parse(StringRef text) {
    return dyn_cast_or_null<BitLinearLayoutMapAttr>(
        parseAttribute(text, &context));
  }

  MLIRContext context;
  std::unique_ptr<ScopedDiagnosticHandler> diagnosticHandler;
};

TEST_F(BitLinearLayoutTest, ParsesAndPrintsXorMap) {
  auto map = parse(R"mlir(#frisk.bit_linear<
      inputs = ["lane", "register"], input_bits = [2, 1],
      outputs = ["m", "n"], output_bits = [1, 2],
      matrix = dense<[[1, 0, 0], [0, 1, 1], [0, 0, 1]]> : tensor<3x3xi1>>
  )mlir");
  ASSERT_TRUE(map);
  auto matrix = map.getMatrixValue();
  ASSERT_TRUE(succeeded(matrix));
  EXPECT_EQ(matrix->getNumRows(), 3u);
  EXPECT_EQ(matrix->getNumColumns(), 3u);
  EXPECT_EQ(matrix->rank(), 3u);

  auto canonical = map.canonicalizeMap();
  ASSERT_TRUE(succeeded(canonical));
  EXPECT_EQ(*canonical, Attribute(map));
  EXPECT_EQ(checkInjective(map).status, ProofStatus::Proven);
  EXPECT_EQ(checkSurjective(map).status, ProofStatus::Proven);
}

TEST_F(BitLinearLayoutTest, RejectsInvalidFields) {
  EXPECT_FALSE(parse(R"mlir(#frisk.bit_linear<
      inputs = ["lane"], input_bits = [2], outputs = ["m"],
      output_bits = [2], matrix = dense<0> : tensor<1x1xi1>>
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(#frisk.bit_linear<
      inputs = ["lane", "lane"], input_bits = [1, 1], outputs = ["m"],
      output_bits = [2], matrix = dense<0> : tensor<2x2xi1>>
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(#frisk.bit_linear<
      inputs = ["lane"], input_bits = [0], outputs = ["m"],
      output_bits = [1], matrix = dense<0> : tensor<1x1xi1>>
  )mlir"));
}

TEST_F(BitLinearLayoutTest, DetectsReplicationKernel) {
  auto map = parse(R"mlir(#frisk.bit_linear<
      inputs = ["lane", "register"], input_bits = [1, 1],
      outputs = ["m"], output_bits = [1],
      matrix = dense<[[1, 0]]> : tensor<1x2xi1>>
  )mlir");
  ASSERT_TRUE(map);

  LayoutProof proof = checkInjective(map);
  EXPECT_EQ(proof.status, ProofStatus::Disproven);
  ASSERT_EQ(proof.counterexample.size(), 2u);
  EXPECT_EQ(proof.counterexample[0], 0);
  EXPECT_EQ(proof.counterexample[1], 1);
}

TEST_F(BitLinearLayoutTest, ComposesByNamedIntermediateDimensions) {
  auto rhs = parse(R"mlir(#frisk.bit_linear<
      inputs = ["lane"], input_bits = [2], outputs = ["x"],
      output_bits = [2], matrix = dense<[[1, 0], [1, 1]]> : tensor<2x2xi1>>
  )mlir");
  auto lhs = parse(R"mlir(#frisk.bit_linear<
      inputs = ["x"], input_bits = [2], outputs = ["m"],
      output_bits = [2], matrix = dense<[[0, 1], [1, 1]]> : tensor<2x2xi1>>
  )mlir");
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);

  auto composed = composeBitLinear(lhs, rhs);
  ASSERT_TRUE(succeeded(composed));
  auto composedMatrix = composed->getMatrixValue();
  auto lhsMatrix = lhs.getMatrixValue();
  auto rhsMatrix = rhs.getMatrixValue();
  ASSERT_TRUE(succeeded(composedMatrix));
  ASSERT_TRUE(succeeded(lhsMatrix));
  ASSERT_TRUE(succeeded(rhsMatrix));

  for (uint64_t value = 0; value != 4; ++value) {
    llvm::APInt input(2, value);
    EXPECT_EQ(composedMatrix->apply(input),
              lhsMatrix->apply(rhsMatrix->apply(input)));
  }
}

} // namespace
} // namespace mlir::frisk
