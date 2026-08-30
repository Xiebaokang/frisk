#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

namespace mlir::frisk {
namespace {

class LayoutEncodingTest : public testing::Test {
protected:
  LayoutEncodingTest() {
    context.getOrLoadDialect<FriskDialect>();
    diagnosticHandler = std::make_unique<ScopedDiagnosticHandler>(
        &context, [](Diagnostic &) { return success(); });
  }

  Attribute parse(StringRef text) { return parseAttribute(text, &context); }

  MLIRContext context;
  std::unique_ptr<ScopedDiagnosticHandler> diagnosticHandler;
};

TEST_F(LayoutEncodingTest, DistributedChecksShapeAndReplication) {
  auto encoding = dyn_cast_or_null<DistributedEncodingAttr>(parse(R"mlir(#frisk.distributed<
      map = #frisk.bit_linear<inputs = ["lane"], input_bits = [5],
        outputs = ["m"], output_bits = [5],
        matrix = dense<[[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>,
      topology = [1, 32, 1, 1, 1], replication = 1>
  )mlir"));
  ASSERT_TRUE(encoding);

  auto validType = RankedTensorType::get({32}, Float16Type::get(&context));
  auto invalidType = RankedTensorType::get({16}, Float16Type::get(&context));
  EXPECT_TRUE(succeeded(
      encoding.verifyForType(validType, UnknownLoc::get(&context))));
  EXPECT_TRUE(failed(
      encoding.verifyForType(invalidType, UnknownLoc::get(&context))));
  EXPECT_EQ(encoding.getKind(), LayoutKind::Distributed);
  ASSERT_TRUE(succeeded(encoding.getCanonicalMap(validType)));

  auto replicated = dyn_cast_or_null<DistributedEncodingAttr>(parse(R"mlir(#frisk.distributed<
      map = #frisk.bit_linear<inputs = ["lane"], input_bits = [5],
        outputs = ["m"], output_bits = [4],
        matrix = dense<[[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0]]> : tensor<4x5xi1>>,
      topology = [1, 32, 1, 1, 1], replication = 2>
  )mlir"));
  ASSERT_TRUE(replicated);
  auto replicatedType =
      RankedTensorType::get({16}, Float16Type::get(&context));
  EXPECT_TRUE(succeeded(replicated.verifyForType(
      replicatedType, UnknownLoc::get(&context))));
}

TEST_F(LayoutEncodingTest, StorageChecksMemoryMapAndAlignment) {
  auto storage = dyn_cast_or_null<StorageLayoutAttr>(parse(R"mlir(#frisk.storage<
      map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
        outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
        map = affine_map<(d0) -> (d0 * 2, 0)>>,
      memory_space = #frisk<memory_space Shared>, alignment = 2,
      vector_granularity = 2>
  )mlir"));
  ASSERT_TRUE(storage);
  Builder builder(&context);
  auto sharedType = MemRefType::get(
      {4}, builder.getF16Type(), AffineMapAttr(),
      builder.getI64IntegerAttr(
          static_cast<int64_t>(attr::MemorySpace::Shared)));
  EXPECT_TRUE(succeeded(
      storage.verifyForType(sharedType, UnknownLoc::get(&context))));
  EXPECT_EQ(storage.getKind(), LayoutKind::Storage);

  auto aliasing = dyn_cast_or_null<StorageLayoutAttr>(parse(R"mlir(#frisk.storage<
      map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
        outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
        map = affine_map<(d0) -> (0, 0)>>,
      memory_space = #frisk<memory_space Shared>, alignment = 2,
      vector_granularity = 1>
  )mlir"));
  ASSERT_TRUE(aliasing);
  EXPECT_TRUE(failed(
      aliasing.verifyForType(sharedType, UnknownLoc::get(&context))));

  EXPECT_FALSE(parse(R"mlir(#frisk.storage<
      map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
        outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
        map = affine_map<(d0) -> (d0 * 2, 0)>>,
      memory_space = #frisk<memory_space Shared>, alignment = 0,
      vector_granularity = 1>
  )mlir"));
}

} // namespace
} // namespace mlir::frisk
