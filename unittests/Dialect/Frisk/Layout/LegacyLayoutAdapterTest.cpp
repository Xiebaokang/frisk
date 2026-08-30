#include "Dialect/Frisk/Analysis/LegacyLayoutAdapter.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

namespace mlir::frisk {
namespace {

struct Sm90Layouts {
  LayoutAttr a;
  LayoutAttr b;
  LayoutAttr c;
  MemRefType aType;
  MemRefType bType;
  MemRefType cType;
};

class LegacyLayoutAdapterTest : public testing::Test {
protected:
  LegacyLayoutAdapterTest() {
    context.getOrLoadDialect<FriskDialect>();
    diagnosticHandler = std::make_unique<ScopedDiagnosticHandler>(
        &context, [](Diagnostic &) { return success(); });
  }

  FailureOr<Sm90Layouts>
  buildSm90Layouts(attr::MemorySpace aSpace = attr::MemorySpace::Shared) {
    OpBuilder builder(&context);
    Location loc = builder.getUnknownLoc();
    module = ModuleOp::create(loc);
    module->getOperation()->setAttr("frisk.target",
                                    builder.getStringAttr("sm_90"));
    builder.setInsertionPointToStart(module->getBody());

    auto memorySpace = [&](attr::MemorySpace space) {
      return builder.getI64IntegerAttr(static_cast<int64_t>(space));
    };
    MemRefType aType = MemRefType::get(
        {128, 64}, builder.getF16Type(), AffineMapAttr(),
        memorySpace(aSpace));
    MemRefType bType = MemRefType::get(
        {64, 128}, builder.getF16Type(), AffineMapAttr(),
        memorySpace(attr::MemorySpace::Shared));
    MemRefType cType = MemRefType::get(
        {128, 128}, builder.getF16Type(), AffineMapAttr(),
        memorySpace(attr::MemorySpace::Local));
    auto kernel = builder.create<KernelOp>(
        loc, "sm90_adapter",
        builder.getFunctionType({aType, bType, cType}, {}));
    Block *entry = kernel.addEntryBlock();
    builder.setInsertionPoint(entry->getTerminator());
    auto gemm = builder.create<GemmOp>(
        loc, entry->getArgument(0), entry->getArgument(1),
        entry->getArgument(2), false, false, uint64_t{128}, uint64_t{128},
        uint64_t{64}, attr::GemmWarpPolicy::Square, false);
    gemm->setAttr("frisk.threads", builder.getI64IntegerAttr(128));

    DenseMap<Value, Attribute> layouts;
    if (failed(gemm.inferLayout(builder, layouts)))
      return failure();
    auto get = [&](Value value) { return dyn_cast<LayoutAttr>(layouts[value]); };
    LayoutAttr a = get(gemm.getA());
    LayoutAttr b = get(gemm.getB());
    LayoutAttr c = get(gemm.getC());
    if (!a || !b || !c)
      return failure();
    return Sm90Layouts{a, b, c, aType, bType, cType};
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<ScopedDiagnosticHandler> diagnosticHandler;
};

TEST_F(LegacyLayoutAdapterTest, ConvertsCurrentSm90SsBaseline) {
  FailureOr<Sm90Layouts> baseline = buildSm90Layouts();
  ASSERT_TRUE(succeeded(baseline));
  Location loc = UnknownLoc::get(&context);

  FailureOr<StorageLayoutAttr> a =
      convertLegacyStorage(baseline->a, baseline->aType, loc);
  FailureOr<StorageLayoutAttr> b =
      convertLegacyStorage(baseline->b, baseline->bType, loc);
  FailureOr<DistributedEncodingAttr> c =
      convertLegacyDistributed(baseline->c, baseline->cType, loc);
  ASSERT_TRUE(succeeded(a));
  ASSERT_TRUE(succeeded(b));
  ASSERT_TRUE(succeeded(c));
  EXPECT_TRUE(isa<BitLinearLayoutMapAttr>(a->getMap()));
  EXPECT_TRUE(isa<BitLinearLayoutMapAttr>(b->getMap()));
  EXPECT_TRUE(isa<BitLinearLayoutMapAttr>(c->getMap()));
  SmallVector<int64_t> expectedTopology = {128, 32, 4, 1, 1};
  EXPECT_EQ(c->getTopology().asArrayRef(),
            ArrayRef<int64_t>(expectedTopology));
  EXPECT_TRUE(succeeded(a->verifyForType(baseline->aType, loc)));
  EXPECT_TRUE(succeeded(b->verifyForType(baseline->bType, loc)));
  EXPECT_TRUE(succeeded(c->verifyForType(baseline->cType, loc)));

  EXPECT_TRUE(succeeded(verifyLegacyStorageEquivalent(
      baseline->a, *a, baseline->aType, loc)));
  EXPECT_TRUE(succeeded(verifyLegacyStorageEquivalent(
      baseline->b, *b, baseline->bType, loc)));
  EXPECT_TRUE(succeeded(verifyLegacyDistributedEquivalent(
      baseline->c, *c, baseline->cType, loc)));
}

TEST_F(LegacyLayoutAdapterTest, ConvertsCurrentSm90RsBaseline) {
  FailureOr<Sm90Layouts> baseline =
      buildSm90Layouts(attr::MemorySpace::Local);
  ASSERT_TRUE(succeeded(baseline));
  Location loc = UnknownLoc::get(&context);

  FailureOr<DistributedEncodingAttr> a =
      convertLegacyDistributed(baseline->a, baseline->aType, loc);
  FailureOr<StorageLayoutAttr> b =
      convertLegacyStorage(baseline->b, baseline->bType, loc);
  FailureOr<DistributedEncodingAttr> c =
      convertLegacyDistributed(baseline->c, baseline->cType, loc);
  ASSERT_TRUE(succeeded(a));
  ASSERT_TRUE(succeeded(b));
  ASSERT_TRUE(succeeded(c));
  EXPECT_EQ(a->getReplication().getInt(), 2);
  EXPECT_TRUE(succeeded(a->verifyForType(baseline->aType, loc)));
  EXPECT_TRUE(succeeded(b->verifyForType(baseline->bType, loc)));
  EXPECT_TRUE(succeeded(c->verifyForType(baseline->cType, loc)));
  EXPECT_TRUE(succeeded(verifyLegacyDistributedEquivalent(
      baseline->a, *a, baseline->aType, loc)));
  EXPECT_TRUE(succeeded(verifyLegacyStorageEquivalent(
      baseline->b, *b, baseline->bType, loc)));
  EXPECT_TRUE(succeeded(verifyLegacyDistributedEquivalent(
      baseline->c, *c, baseline->cType, loc)));
}

TEST_F(LegacyLayoutAdapterTest, RejectsUnsupportedLegacyLayouts) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto shape = builder.getDenseI64ArrayAttr({3});
  AffineExpr dim = builder.getAffineDimExpr(0);
  auto index = AffineMapAttr::get(AffineMap::get(1, 0, dim % 2));
  auto thread = AffineMapAttr::get(AffineMap::get(1, 0, dim));
  LayoutAttr legacy = LayoutAttr::get(
      &context, shape, index, thread, builder.getI64IntegerAttr(1));
  auto type = MemRefType::get(
      {3}, builder.getF16Type(), AffineMapAttr(),
      builder.getI64IntegerAttr(
          static_cast<int64_t>(attr::MemorySpace::Local)));
  EXPECT_TRUE(failed(convertLegacyDistributed(legacy, type, loc)));

  auto sharedType = MemRefType::get(
      {3}, builder.getF16Type(), AffineMapAttr(),
      builder.getI64IntegerAttr(
          static_cast<int64_t>(attr::MemorySpace::Shared)));
  auto twoResultMap = AffineMapAttr::get(
      AffineMap::get(1, 0, {dim.floorDiv(2), dim % 2}, &context));
  LayoutAttr unsupportedStorage = LayoutAttr::get(
      &context, shape, twoResultMap, AffineMapAttr(), IntegerAttr());
  EXPECT_TRUE(
      failed(convertLegacyStorage(unsupportedStorage, sharedType, loc)));
}

TEST_F(LegacyLayoutAdapterTest, ConvertsNonPowerOfTwoLinearStorage) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto type = MemRefType::get(
      {6}, builder.getF16Type(), AffineMapAttr(),
      builder.getI64IntegerAttr(
          static_cast<int64_t>(attr::MemorySpace::Shared)));
  AffineExpr dim = builder.getAffineDimExpr(0);
  LayoutAttr legacy = LayoutAttr::get(
      &context, builder.getDenseI64ArrayAttr({6}),
      AffineMapAttr::get(AffineMap::get(1, 0, dim, &context)),
      AffineMapAttr(), IntegerAttr());
  FailureOr<StorageLayoutAttr> converted =
      convertLegacyStorage(legacy, type, loc);
  ASSERT_TRUE(succeeded(converted));
  EXPECT_TRUE(isa<AffineLayoutMapAttr>(converted->getMap()));
  EXPECT_TRUE(succeeded(converted->verifyForType(type, loc)));
  EXPECT_TRUE(
      succeeded(verifyLegacyStorageEquivalent(legacy, *converted, type, loc)));
}

} // namespace
} // namespace mlir::frisk
