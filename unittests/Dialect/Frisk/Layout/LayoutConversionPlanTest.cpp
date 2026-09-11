#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

namespace mlir::frisk::test {
namespace {
class ConversionPlanTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  ConversionPlanTest() { context.getOrLoadDialect<FriskDialect>(); }

  // Each logical bit selects a named source bit, independently of column order.
  RankedTensorType type(ArrayRef<int64_t> topology, ArrayRef<StringRef> names,
                       ArrayRef<unsigned> widths, ArrayRef<unsigned> columns,
                       Type element = {}, ArrayRef<int64_t> shape = {}) {
    SmallVector<Attribute> ns;
    SmallVector<int64_t> ws;
    unsigned bits = 0;
    for (auto [name, width] : llvm::zip_equal(names, widths)) {
      ns.push_back(b.getStringAttr(name)); ws.push_back(width); bits += width;
    }
    SmallVector<APInt> entries;
    for (unsigned col : columns)
      for (unsigned j = 0; j < bits; ++j) entries.emplace_back(1, j == col);
    SmallVector<int64_t> dims(shape);
    if (dims.empty()) dims.push_back(int64_t(1) << columns.size());
    SmallVector<Attribute> outputs;
    SmallVector<int64_t> outWidths;
    for (auto [i, dim] : llvm::enumerate(dims)) {
      outputs.push_back(b.getStringAttr("d" + std::to_string(i)));
      outWidths.push_back(llvm::Log2_64(dim));
    }
    auto map = BitLinearLayoutMapAttr::get(&context, b.getArrayAttr(ns),
        b.getDenseI64ArrayAttr(ws), b.getArrayAttr(outputs),
        b.getDenseI64ArrayAttr(outWidths), DenseIntElementsAttr::get(
            RankedTensorType::get({int64_t(columns.size()), bits}, b.getI1Type()), entries));
    int64_t carriers = 1;
    for (int64_t t : topology) carriers *= t;
    auto enc = DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr(topology), b.getI64IntegerAttr(carriers >> columns.size()));
    return RankedTensorType::get(dims, element ? element : b.getF32Type(), enc);
  }

  // Independent named row-XOR evaluator; does not call GF2Matrix or planner.
  uint64_t logical(RankedTensorType ty, Owner owner) {
    auto map = cast<BitLinearLayoutMapAttr>(cast<DistributedEncodingAttr>(ty.getEncoding()).getMap());
    SmallVector<bool> input;
    for (auto [name, width] : llvm::zip_equal(map.getInputNames(), map.getInputBitWidths().asArrayRef())) {
      StringRef n = cast<StringAttr>(name).getValue();
      uint64_t value = n == "register" ? owner.reg : n == "lane" ? owner.lane :
          n == "warp" ? owner.warp : n == "warp_group" ? owner.warpGroup : 0;
      for (int i = 0; i < width; ++i) input.push_back((value >> i) & 1);
    }
    SmallVector<bool> output;
    auto values = map.getMatrix().getValues<APInt>();
    auto it = values.begin();
    for (int row = 0; row < map.getMatrix().getType().getShape()[0]; ++row) {
      bool bit = false;
      for (bool in : input) bit ^= bool((*it++).getZExtValue()) && in;
      output.push_back(bit);
    }
    uint64_t slot = 0; unsigned offset = 0;
    for (auto [dim, width] : llvm::zip_equal(ty.getShape(), map.getOutputBitWidths().asArrayRef())) {
      unsigned coord = 0;
      for (int i = 0; i < width; ++i) coord |= unsigned(output[offset++]) << i;
      slot = slot * dim + coord;
    }
    return slot;
  }

  void oracle(RankedTensorType src, RankedTensorType dst, const RedistributionPlan &p) {
    std::vector<unsigned> writes(p.volume);
    for (unsigned slot = 0; slot < p.volume; ++slot) {
      EXPECT_EQ(logical(src, p.globalWriters[slot]), slot);
    }
    for (unsigned t = 0; t < p.threads; ++t) {
      for (unsigned r = 0; r < p.sourceRegisters; ++r) {
        Owner own = p.owner(t, r);
        auto slot = logical(src, own);
        if (own == p.globalWriters[slot]) ++writes[slot];
      }
      for (unsigned r = 0; r < p.destinationRegisters; ++r) {
        const auto &entry = p.destinations[t * p.destinationRegisters + r];
        EXPECT_EQ(logical(dst, p.owner(t, r)), entry.slot);
        EXPECT_EQ(logical(src, entry.source), entry.slot);
        if (p.mode != ExchangeMode::Shared)
          EXPECT_EQ(p.thread(entry.source) / 32, t / 32);
        if (p.mode == ExchangeMode::Register)
          EXPECT_EQ(p.thread(entry.source), t);
      }
    }
    for (unsigned count : writes) EXPECT_EQ(count, 1u);
  }
};

TEST_F(ConversionPlanTest, NamedRegisterLaneSwapAndPermutation) {
  auto src = type({2,32,1,1,1}, {"lane","register"}, {5,1}, {0,1,2,3,4,5});
  auto dst = type({2,32,1,1,1}, {"register","lane"}, {1,5}, {0,2,3,4,5,1});
  auto p = planRedistribution(src, dst, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(p)); EXPECT_EQ(p->mode, ExchangeMode::Shuffle);
  oracle(src, dst, *p);
  // Destination lane 1 asks for register 1 at another lane, catching a shuffle
  // of each source lane's own requested register instead of shuffle-then-select.
  bool witnessedWrongSelection = false;
  for (unsigned t = 0; t < p->threads; ++t)
    for (unsigned r = 0; r < p->destinationRegisters; ++r) {
      auto e = p->destinations[t * p->destinationRegisters + r];
      auto sourceThread = p->thread(e.source);
      auto wrongReg = p->destinations[sourceThread * p->destinationRegisters + r].source.reg;
      witnessedWrongSelection |= logical(src, p->owner(sourceThread, wrongReg)) != e.slot;
    }
  EXPECT_TRUE(witnessedWrongSelection);
}

TEST_F(ConversionPlanTest, InvertibleXorRowsUseIndependentOwnershipOracle) {
  auto dst = type({2,32,1,1,1}, {"register","lane"}, {1,5}, {0,1,2,3,4,5});
  auto encoding = cast<DistributedEncodingAttr>(dst.getEncoding());
  auto map = cast<BitLinearLayoutMapAttr>(encoding.getMap());
  SmallVector<APInt> entries(map.getMatrix().getValues<APInt>());
  // y0 = register XOR lane[4], y1 = lane[0] XOR register; y2..5
  // preserve lane[1..4]. Two elementary row additions give an invertible
  // non-permutation matrix, independently checked below without GF2 helpers.
  entries[5] = APInt(1, 1);
  entries[6] = APInt(1, 1);
  auto xorMap = BitLinearLayoutMapAttr::get(&context, map.getInputNames(),
      map.getInputBitWidths(), map.getOutputNames(), map.getOutputBitWidths(),
      DenseIntElementsAttr::get(map.getMatrix().getType(), entries));
  auto xorEncoding = DistributedEncodingAttr::get(&context, xorMap,
      encoding.getTopology(), encoding.getReplication());
  auto src = RankedTensorType::get(dst.getShape(), dst.getElementType(), xorEncoding);
  auto p = planRedistribution(src, dst, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(p));
  EXPECT_EQ(p->mode, ExchangeMode::Shuffle);
  oracle(src, dst, *p);
  for (unsigned lane = 0; lane < 32; ++lane)
    for (unsigned reg = 0; reg < 2; ++reg) {
      Owner selected = p->destinations[lane * 2 + reg].source;
      unsigned expectedRegister = reg ^ (lane >> 4);
      EXPECT_EQ(selected.reg, expectedRegister);
      EXPECT_EQ(selected.lane, (lane & ~1u) | ((lane & 1u) ^ expectedRegister));
      EXPECT_EQ(selected.warp, 0u);
      EXPECT_EQ(selected.warpGroup, 0u);
    }
}

TEST_F(ConversionPlanTest, WarpGroupDecodingAcrossNamedInputOrders) {
  auto src = type({1,32,2,2,1}, {"warp_group","lane","warp"}, {1,5,1},
                  {1,2,3,4,5,6,0});
  auto dst = type({1,32,2,2,1}, {"warp","warp_group","lane"}, {1,1,5},
                  {1,3,4,5,6,0,2});
  auto p = planRedistribution(src, dst, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(p));
  EXPECT_EQ(p->mode, ExchangeMode::Shared);
  EXPECT_EQ(p->threads, 128u);
  EXPECT_EQ(p->scratchBytes, 512u);
  oracle(src, dst, *p);
  for (unsigned thread = 0; thread < 128; ++thread) {
    Owner physical = p->owner(thread, 0);
    EXPECT_EQ(physical.lane, thread % 32);
    EXPECT_EQ(physical.warp, (thread / 32) % 2);
    EXPECT_EQ(physical.warpGroup, thread / 64);
    Owner selected = p->destinations[thread].source;
    // Destination swaps lane[0] and warp_group[0], preserving warp[0].
    EXPECT_EQ(selected.lane, (thread % 32 & ~1u) | (thread / 64));
    EXPECT_EQ(selected.warp, (thread / 32) % 2);
    EXPECT_EQ(selected.warpGroup, thread % 2);
    EXPECT_EQ(p->thread(selected),
              selected.warpGroup * 64 + selected.warp * 32 + selected.lane);
  }
}

TEST_F(ConversionPlanTest, ReplicasPreferThreadThenWarp) {
  auto src = type({32,32,2,1,1}, {"register"}, {5}, {0,1,2,3,4});
  auto dst = type({1,32,2,1,1}, {"lane"}, {5}, {1,0,2,3,4});
  auto p = planRedistribution(src, dst, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(p)); EXPECT_EQ(p->mode, ExchangeMode::Register);
  oracle(src, dst, *p);
  src = type({1,32,2,1,1}, {"lane"}, {5}, {0,1,2,3,4});
  p = planRedistribution(src, dst, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(p)); EXPECT_EQ(p->mode, ExchangeMode::Shuffle);
  oracle(src, dst, *p);
}

TEST_F(ConversionPlanTest, SharedReplicatedGlobalWritersAndRowMajorSlots) {
  auto src = type({2,32,2,1,1}, {"warp","lane"}, {1,5}, {1,2,3,4,5,0}, {}, {8,8});
  auto dst = type({2,32,2,1,1}, {"lane","warp"}, {5,1}, {5,1,2,3,4,0}, {}, {8,8});
  auto p = planRedistribution(src, dst, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(p)); EXPECT_EQ(p->mode, ExchangeMode::Shared);
  EXPECT_EQ(p->scratchBytes, 256u); oracle(src, dst, *p);
}

TEST_F(ConversionPlanTest, PayloadBitsAcrossUniformRegisterShuffles) {
  SmallVector<Type> types{b.getI8Type(), b.getI16Type(), b.getI32Type(), b.getI64Type(),
      b.getF16Type(), b.getBF16Type(), b.getF32Type(), b.getF64Type()};
  for (Type ty : types) {
    auto src = type({2,32,1,1,1}, {"lane","register"}, {5,1}, {0,1,2,3,4,5}, ty);
    auto dst = type({2,32,1,1,1}, {"lane","register"}, {5,1}, {5,1,2,3,4,0}, ty);
    auto p = planRedistribution(src, dst, b.getUnknownLoc());
    ASSERT_TRUE(succeeded(p));
    unsigned bits = ty.getIntOrFloatBitWidth();
    SmallVector<uint64_t> seeds{0ULL, 0x8000000000000000ULL, 0x0123456789abcdefULL,
         0xfedcba9876543210ULL, 0x7ff8123456789abcULL, 0x7fc12345ULL,
         0x8000ULL, 0x7e55ULL, 0x7fc5ULL, 0xffffULL, 0x0001ULL,
         0x7c00ULL, 0xfc00ULL, 0x7f80ULL, 0xff80ULL, 0x80000000ULL,
         0x7f800000ULL, 0x7ff0000000000000ULL};
    if (bits == 8) for (unsigned i = 0; i < 256; ++i) seeds.push_back(i);
    EXPECT_EQ(p->payload.words.size(), bits == 64 ? 2u : 1u);
    for (auto [i, word] : llvm::enumerate(p->payload.words)) {
      EXPECT_EQ(word.bitOffset, 32u * i);
      EXPECT_EQ(word.liveBits, std::min(32u, bits));
    }
    for (uint64_t seed : seeds) {
      for (auto e : p->destinations) {
        APInt expected = APInt(64, seed ^ e.slot).zextOrTrunc(bits);
        APInt actual(bits, 0);
        for (auto word : p->payload.words) {
          APInt shuffled(32, 0);
          for (unsigned r = 0; r < p->sourceRegisters; ++r) {
            APInt value = APInt(64, seed ^ logical(src, p->owner(p->thread(e.source), r))).zextOrTrunc(bits);
            APInt candidate = value.lshr(word.bitOffset).zextOrTrunc(32);
            if (r == e.source.reg) shuffled = candidate;
          }
          actual |= shuffled.zextOrTrunc(bits).shl(word.bitOffset);
        }
        EXPECT_EQ(actual, expected);
      }
    }
  }
}

TEST_F(ConversionPlanTest, ExhaustiveSixBitPermutationsWithRenamedInputOrder) {
  auto src = type({2,32,2,1,1}, {"lane","warp"}, {5,1}, {0,1,2,3,4,5}, {}, {8,8});
  auto reordered = type({2,32,2,1,1}, {"warp","lane"}, {1,5}, {1,2,3,4,5,0}, {}, {8,8});
  SmallVector<unsigned> columns{0,1,2,3,4,5};
  unsigned permutations = 0;
  do {
    auto dst = type({2,32,2,1,1}, {"lane","warp"}, {5,1}, columns, {}, {8,8});
    auto p = planRedistribution(src, dst, b.getUnknownLoc());
    auto q = planRedistribution(reordered, dst, b.getUnknownLoc());
    ASSERT_TRUE(succeeded(p)); ASSERT_TRUE(succeeded(q));
    oracle(src, dst, *p);
    EXPECT_EQ(p->mode, q->mode);
    EXPECT_EQ(p->globalWriters, q->globalWriters);
    for (unsigned i = 0; i < p->destinations.size(); ++i) {
      EXPECT_EQ(p->destinations[i].source, q->destinations[i].source);
      EXPECT_EQ(p->destinations[i].slot, q->destinations[i].slot);
    }
    ++permutations;
  } while (std::next_permutation(columns.begin(), columns.end()));
  EXPECT_EQ(permutations, 720u);
}

TEST_F(ConversionPlanTest, RegisterOnlyPurePermutation) {
  auto src = type({4,32,1,1,1}, {"register","lane"}, {2,5}, {0,1,2,3,4,5,6});
  auto dst = type({4,32,1,1,1}, {"register","lane"}, {2,5}, {1,0,2,3,4,5,6});
  auto p = planRedistribution(src, dst, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(p)); EXPECT_EQ(p->mode, ExchangeMode::Register);
  oracle(src, dst, *p);
  EXPECT_EQ(p->destinations[1].source.reg, 2u);
  EXPECT_EQ(p->destinations[2].source.reg, 1u);
}

TEST_F(ConversionPlanTest, RejectsUnsupportedPlannerInputs) {
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  auto src = type({1,32,1,1,1}, {"lane"}, {5}, {0,1,2,3,4});
  auto partial = type({2,16,1,1,1}, {"lane","register"}, {4,1}, {0,1,2,3,4});
  EXPECT_TRUE(failed(planRedistribution(partial, partial, b.getUnknownLoc())));
  auto cross = type({1,32,1,1,2}, {"lane"}, {5}, {0,1,2,3,4});
  EXPECT_TRUE(failed(planRedistribution(cross, cross, b.getUnknownLoc())));
  EXPECT_TRUE(failed(planRedistribution(src, cross, b.getUnknownLoc())));
  auto dynamic = RankedTensorType::get({ShapedType::kDynamic}, b.getF32Type(), src.getEncoding());
  EXPECT_TRUE(failed(planRedistribution(dynamic, dynamic, b.getUnknownLoc())));
  for (Type ty : {Type(b.getI1Type()), Type(b.getIntegerType(24)), Type(b.getIndexType())}) {
    auto bad = RankedTensorType::get(src.getShape(), ty, src.getEncoding());
    EXPECT_TRUE(failed(planRedistribution(bad, bad, b.getUnknownLoc())));
  }
}

TEST_F(ConversionPlanTest, MissingEncodingHasActionableDiagnostic) {
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream os(diagnostic); d.print(os); return success();
  });
  auto plain = RankedTensorType::get({32}, b.getF32Type());
  EXPECT_TRUE(failed(planRedistribution(plain, plain, b.getUnknownLoc())));
  EXPECT_NE(diagnostic.find("requires distributed source and destination encodings"), std::string::npos);
}

TEST_F(ConversionPlanTest, ScratchAlignmentBudgetAndByteSizes) {
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  for (Type ty : {Type(b.getI8Type()), Type(b.getF16Type()), Type(b.getBF16Type()),
                  Type(b.getI32Type()), Type(b.getF64Type())}) {
    auto src = type({1,32,2,1,1}, {"lane","warp"}, {5,1}, {0,1,2,3,4,5}, ty);
    auto dst = type({1,32,2,1,1}, {"lane","warp"}, {5,1}, {5,1,2,3,4,0}, ty);
    auto p = planRedistribution(src, dst, b.getUnknownLoc());
    ASSERT_TRUE(succeeded(p));
    unsigned bytes = ty.getIntOrFloatBitWidth() / 8;
    EXPECT_EQ(p->scratchBytes, 64 * bytes);
    auto exact = reserveScratch(0, p->scratchBytes, bytes, p->scratchBytes, b.getUnknownLoc());
    ASSERT_TRUE(succeeded(exact)); EXPECT_EQ(*exact, p->scratchBytes);
    EXPECT_TRUE(failed(reserveScratch(0, p->scratchBytes, bytes, p->scratchBytes - 1, b.getUnknownLoc())));
  }
  auto aligned = reserveScratch(1, 8, 8, 16, b.getUnknownLoc());
  ASSERT_TRUE(succeeded(aligned)); EXPECT_EQ(*aligned, 16u);
  EXPECT_TRUE(failed(reserveScratch(1, 8, 8, 15, b.getUnknownLoc())));
  EXPECT_TRUE(failed(reserveScratch(0, 8, 3, 32, b.getUnknownLoc())));
  EXPECT_TRUE(failed(reserveScratch(UINT64_MAX, 8, 8, UINT64_MAX, b.getUnknownLoc())));
}

TEST_F(ConversionPlanTest, PreflightFailurePreservesWholeModule) {
  context.getOrLoadDialect<gpu::GPUDialect>();
  auto src = type({1,32,2,1,1}, {"lane","warp"}, {5,1}, {0,1,2,3,4,5});
  auto dst = type({1,32,2,1,1}, {"lane","warp"}, {5,1}, {5,1,2,3,4,0});
  OwningOpRef<ModuleOp> module = ModuleOp::create(b.getUnknownLoc());
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(module->getBody());
  auto gpuModule = builder.create<gpu::GPUModuleOp>(b.getUnknownLoc(), "kernels");
  for (unsigned i = 0; i < 2; ++i) {
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
    auto function = builder.create<gpu::GPUFuncOp>(b.getUnknownLoc(), "kernel" + std::to_string(i),
        b.getFunctionType({src}, {}));
    function->setAttr("gpu.kernel", b.getUnitAttr());
    // First visited function has a valid shared plan; the second fails late.
    if (i == 1) function->setAttr("known_block_size", b.getDenseI32ArrayAttr({64,1,1}));
    builder.setInsertionPointToStart(&function.getBody().front());
    builder.create<ConvertLayoutOp>(b.getUnknownLoc(), dst, function.getArgument(0));
    builder.create<gpu::ReturnOp>(b.getUnknownLoc());
  }
  ASSERT_TRUE(succeeded(verify(*module)));
  std::string before, after, diagnostic;
  llvm::raw_string_ostream beforeStream(before); module->print(beforeStream);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream os(diagnostic); d.print(os); return success();
  });
  PassManager pm(&context); pm.addPass(createTestLowerLayoutConversionsPass());
  EXPECT_TRUE(failed(pm.run(*module)));
  llvm::raw_string_ostream afterStream(after); module->print(afterStream);
  EXPECT_EQ(before, after);
  EXPECT_NE(diagnostic.find("known_block_size"), std::string::npos);
}

TEST_F(ConversionPlanTest, CleanupPreservesExactCanonicalEquivalentResultType) {
  context.getOrLoadDialect<func::FuncDialect>();
  auto a = type({1,32,1,1,1}, {"lane"}, {5}, {0,1,2,3,4});
  auto middle = type({1,32,1,1,1}, {"lane"}, {5}, {1,0,2,3,4});
  auto enc = cast<DistributedEncodingAttr>(a.getEncoding());
  // Same canonical map, distinct encoding attribute due to replication width.
  auto other = DistributedEncodingAttr::get(&context, enc.getMap(), enc.getTopology(), b.getI32IntegerAttr(1));
  auto out = RankedTensorType::get(a.getShape(), a.getElementType(), other);
  ASSERT_NE(a, out);
  ASSERT_EQ(*enc.getCanonicalMap(a), *other.getCanonicalMap(out));
  OwningOpRef<ModuleOp> module = ModuleOp::create(b.getUnknownLoc());
  OpBuilder builder(&context); builder.setInsertionPointToStart(module->getBody());
  auto f = builder.create<func::FuncOp>(b.getUnknownLoc(), "exact", b.getFunctionType({a}, {out}));
  f.addEntryBlock(); builder.setInsertionPointToStart(&f.front());
  Value x = builder.create<ConvertLayoutOp>(b.getUnknownLoc(), middle, f.getArgument(0));
  Value y = builder.create<ConvertLayoutOp>(b.getUnknownLoc(), out, x);
  builder.create<func::ReturnOp>(b.getUnknownLoc(), y);
  ASSERT_TRUE(succeeded(verify(*module)));
  PassManager pm(&context); pm.addPass(createOptimizeLayoutConversionsPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  unsigned count = 0;
  module->walk([&](ConvertLayoutOp op) {
    ++count; EXPECT_EQ(op.getSource(), f.getArgument(0)); EXPECT_EQ(op.getResult().getType(), out);
  });
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(cast<func::ReturnOp>(f.front().getTerminator()).getOperand(0).getType(), out);
}
} // namespace
} // namespace mlir::frisk::test
