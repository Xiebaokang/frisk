#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"

#include <cstdint>
#include <initializer_list>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

namespace mlir::frisk {
namespace {

static GF2Matrix makeMatrix(std::initializer_list<uint64_t> values,
                            unsigned columns) {
  llvm::SmallVector<llvm::APInt> rows;
  rows.reserve(values.size());
  for (uint64_t value : values)
    rows.emplace_back(columns, value);
  return *GF2Matrix::get(values.size(), columns, rows);
}

TEST(GF2MatrixTest, RejectsMalformedRows) {
  llvm::SmallVector<llvm::APInt> wrongCount{llvm::APInt(2, 1)};
  EXPECT_TRUE(failed(GF2Matrix::get(2, 2, wrongCount)));

  llvm::SmallVector<llvm::APInt> wrongWidth{llvm::APInt(3, 1),
                                            llvm::APInt(3, 2)};
  EXPECT_TRUE(failed(GF2Matrix::get(2, 2, wrongWidth)));
}

TEST(GF2MatrixTest, IdentityAndXorApply) {
  GF2Matrix identity = makeMatrix({0b01, 0b10}, 2);
  GF2Matrix xorMap = makeMatrix({0b01, 0b11}, 2);

  for (uint64_t value = 0; value != 4; ++value)
    EXPECT_EQ(identity.apply(llvm::APInt(2, value)), llvm::APInt(2, value));

  EXPECT_EQ(xorMap.apply(llvm::APInt(2, 0b11)), llvm::APInt(2, 0b01));
  EXPECT_EQ(xorMap.transpose(), makeMatrix({0b11, 0b10}, 2));
}

TEST(GF2MatrixTest, ComposeUsesRhsThenLhs) {
  GF2Matrix distributed = makeMatrix({0b01, 0b11}, 2);
  GF2Matrix storage = makeMatrix({0b10, 0b11}, 2);
  auto composed = storage.compose(distributed);
  ASSERT_TRUE(succeeded(composed));

  for (uint64_t value = 0; value != 4; ++value) {
    llvm::APInt input(2, value);
    EXPECT_EQ(composed->apply(input), storage.apply(distributed.apply(input)));
  }

  EXPECT_TRUE(failed(makeMatrix({0b1}, 1).compose(distributed)));
}

TEST(GF2MatrixTest, RankAndKernelAreDeterministic) {
  GF2Matrix deficient = makeMatrix({0b11, 0b11}, 2);
  EXPECT_EQ(deficient.rank(), 1u);

  auto kernel = deficient.kernelBasis();
  ASSERT_EQ(kernel.size(), 1u);
  EXPECT_EQ(kernel.front(), llvm::APInt(2, 0b11));
  EXPECT_TRUE(deficient.apply(kernel.front()).isZero());
}

TEST(GF2MatrixTest, InverseRoundTripsEveryInput) {
  GF2Matrix matrix = makeMatrix({0b11, 0b01}, 2);
  auto inverse = matrix.inverse();
  ASSERT_TRUE(succeeded(inverse));

  for (uint64_t value = 0; value != 4; ++value) {
    llvm::APInt input(2, value);
    EXPECT_EQ(inverse->apply(matrix.apply(input)), input);
    EXPECT_EQ(matrix.apply(inverse->apply(input)), input);
  }

  EXPECT_TRUE(failed(makeMatrix({0b11, 0b11}, 2).inverse()));
  EXPECT_TRUE(failed(makeMatrix({0b001, 0b010}, 3).inverse()));
}

TEST(GF2MatrixTest, RightInverseExistsExactlyForSurjectiveMap) {
  GF2Matrix surjective = makeMatrix({0b001, 0b110}, 3);
  auto rightInverse = surjective.rightInverse();
  ASSERT_TRUE(succeeded(rightInverse));
  auto identity = surjective.compose(*rightInverse);
  ASSERT_TRUE(succeeded(identity));
  EXPECT_EQ(*identity, makeMatrix({0b01, 0b10}, 2));

  EXPECT_TRUE(failed(makeMatrix({0b011, 0b011}, 3).rightInverse()));
}

TEST(GF2MatrixTest, ComposeMatchesExhaustiveOracle) {
  GF2Matrix lhs = makeMatrix({0b001, 0b101, 0b110}, 3);
  GF2Matrix rhs = makeMatrix({0b0011, 0b0101, 0b1110}, 4);
  auto composed = lhs.compose(rhs);
  ASSERT_TRUE(succeeded(composed));

  for (uint64_t value = 0; value != 16; ++value) {
    llvm::APInt input(4, value);
    EXPECT_EQ(composed->apply(input), lhs.apply(rhs.apply(input)));
  }
}

} // namespace
} // namespace mlir::frisk
