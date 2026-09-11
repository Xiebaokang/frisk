#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_TESTLOWERLAYOUTCONVERSIONS
#include "Dialect/Frisk/Transforms/Passes.h.inc"
namespace {
using namespace mlir::frisk::test;

// The finite lookup tables are the exact planner output, not a second layout
// interpretation. This is deliberately a test adapter, not optimized codegen.
Value lookup(OpBuilder &b, Location loc, Value thread, ArrayRef<uint64_t> values) {
  if (llvm::all_equal(values))
    return b.create<arith::ConstantIntOp>(loc, values.front(), 32);
  auto ty = VectorType::get({int64_t(values.size())}, b.getI32Type());
  SmallVector<APInt> bits;
  for (uint64_t value : values) bits.emplace_back(32, value);
  Value table = b.create<arith::ConstantOp>(loc, ty, DenseIntElementsAttr::get(ty, bits));
  return b.create<vector::ExtractElementOp>(loc, table, thread);
}
Value asIndex(OpBuilder &b, Location loc, Value value) {
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
}

SmallVector<Value> encodeWords(OpBuilder &b, Location loc, Value value,
                               const PayloadTransportPlan &p) {
  auto bitsType = b.getIntegerType(p.bits);
  if (value.getType() != bitsType)
    value = b.create<arith::BitcastOp>(loc, bitsType, value);
  SmallVector<Value> words;
  for (auto word : p.words) {
    Value part = value;
    if (word.bitOffset) {
      Value shift = b.create<arith::ConstantIntOp>(loc, word.bitOffset, p.bits);
      part = b.create<arith::ShRUIOp>(loc, part, shift);
    }
    if (p.bits < 32) part = b.create<arith::ExtUIOp>(loc, b.getI32Type(), part);
    else if (p.bits > 32) part = b.create<arith::TruncIOp>(loc, b.getI32Type(), part);
    words.push_back(part);
  }
  return words;
}
Value decodeWords(OpBuilder &b, Location loc, ArrayRef<Value> words,
                    Type type, const PayloadTransportPlan &p) {
  auto bitsType = b.getIntegerType(p.bits);
  Value result;
  for (auto [word, input] : llvm::zip_equal(p.words, words)) {
    Value value = input;
    if (p.bits < 32) value = b.create<arith::TruncIOp>(loc, bitsType, value);
    else if (p.bits > 32) value = b.create<arith::ExtUIOp>(loc, bitsType, value);
    if (word.bitOffset) {
      Value shift = b.create<arith::ConstantIntOp>(loc, word.bitOffset, p.bits);
      value = b.create<arith::ShLIOp>(loc, value, shift);
    }
    result = result ? b.create<arith::OrIOp>(loc, result, value) : value;
  }
  if (result.getType() != type) result = b.create<arith::BitcastOp>(loc, type, result);
  return result;
}

void emitPlan(ConvertLayoutOp op, gpu::GPUFuncOp function,
              const RedistributionPlan &p) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto sourceType = cast<RankedTensorType>(op.getSource().getType());
  Type element = sourceType.getElementType();
  auto sourceVector = VectorType::get({p.sourceRegisters}, element);
  auto resultVector = VectorType::get({p.destinationRegisters}, element);
  // M6 must replace these Tensor<->thread-vector ABI bridges with real tile
  // load/store transport before executable lowering. They are not tensor casts.
  Value source = b.create<UnrealizedConversionCastOp>(loc, sourceVector, op.getSource()).getResult(0);
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  SmallVector<Value> registers;
  for (unsigned r = 0; r < p.sourceRegisters; ++r) {
    Value index = b.create<arith::ConstantIndexOp>(loc, r);
    registers.push_back(b.create<vector::ExtractElementOp>(loc, source, index));
  }
  Value result = b.create<arith::ConstantOp>(loc, resultVector, b.getZeroAttr(resultVector));
  Value scratch;
  if (p.mode == ExchangeMode::Shared) {
    auto space = gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup);
    auto ty = MemRefType::get({int64_t(p.volume)}, element, MemRefLayoutAttrInterface{}, space);
    unsigned attribution = function.getNumWorkgroupAttributions();
    scratch = function.addWorkgroupAttribution(ty, loc);
    // GPUCommon lowering forwards llvm.align to the workgroup LLVM global.
    function.setworkgroupAttributionAttrs(attribution,
        b.getDictionaryAttr({b.getNamedAttr("llvm.align", b.getI64IntegerAttr(p.payload.bits / 8))}));
    for (unsigned r = 0; r < p.sourceRegisters; ++r) {
      SmallVector<uint64_t> slots, elected;
      for (unsigned t = 0; t < p.threads; ++t) {
        auto slot = p.sourceSlots[t * p.sourceRegisters + r];
        slots.push_back(slot);
        elected.push_back(p.owner(t, r) == p.globalWriters[slot]);
      }
      Value slot = asIndex(b, loc, lookup(b, loc, thread, slots));
      Value writer = lookup(b, loc, thread, elected);
      Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
      Value predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, writer, zero);
      b.create<scf::IfOp>(loc, predicate, [&](OpBuilder &nested, Location l) {
        nested.create<memref::StoreOp>(l, registers[r], scratch, ValueRange{slot});
        nested.create<scf::YieldOp>(l);
      });
    }
    b.create<gpu::BarrierOp>(loc);
  }
  SmallVector<SmallVector<Value>> payloads;
  if (p.mode == ExchangeMode::Shuffle)
    for (Value reg : registers) payloads.push_back(encodeWords(b, loc, reg, p.payload));
  for (unsigned r = 0; r < p.destinationRegisters; ++r) {
    SmallVector<uint64_t> slots, sourceRegs, lanes;
    llvm::SmallSet<unsigned, 8> possibleRegs;
    for (unsigned t = 0; t < p.threads; ++t) {
      auto d = p.destinations[t * p.destinationRegisters + r];
      slots.push_back(d.slot); sourceRegs.push_back(d.source.reg); lanes.push_back(d.source.lane);
      possibleRegs.insert(d.source.reg);
    }
    Value value;
    if (scratch) {
      Value slot = asIndex(b, loc, lookup(b, loc, thread, slots));
      value = b.create<memref::LoadOp>(loc, scratch, ValueRange{slot});
    } else {
      Value sourceReg = lookup(b, loc, thread, sourceRegs);
      if (p.mode == ExchangeMode::Register) {
        value = b.create<vector::ExtractElementOp>(loc, source, asIndex(b, loc, sourceReg));
      } else {
        Value lane = lookup(b, loc, thread, lanes);
        Value width = b.create<arith::ConstantIntOp>(loc, 32, 32);
        SmallVector<Value> selected(p.payload.words.size());
        // All lanes execute every candidate register shuffle. Selection occurs
        // only afterwards and is based on the destination lane's request.
        for (unsigned k : possibleRegs) {
          SmallVector<Value> shuffled;
          for (Value word : payloads[k])
            shuffled.push_back(b.create<gpu::ShuffleOp>(loc, word, lane, width,
                gpu::ShuffleMode::IDX).getShuffleResult());
          Value constant = b.create<arith::ConstantIntOp>(loc, k, 32);
          Value choose = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, sourceReg, constant);
          for (unsigned w = 0; w < selected.size(); ++w)
            selected[w] = selected[w] ? b.create<arith::SelectOp>(loc, choose, shuffled[w], selected[w]) : shuffled[w];
        }
        value = decodeWords(b, loc, selected, element, p.payload);
      }
    }
    Value index = b.create<arith::ConstantIndexOp>(loc, r);
    result = b.create<vector::InsertElementOp>(loc, value, result, index);
  }
  Value tensor = b.create<UnrealizedConversionCastOp>(loc, op.getResult().getType(), result).getResult(0);
  op.getResult().replaceAllUsesWith(tensor);
  op.erase();
}

class TestLowerLayoutConversionsPass final
    : public impl::TestLowerLayoutConversionsBase<TestLowerLayoutConversionsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    struct Pending { ConvertLayoutOp op; gpu::GPUFuncOp function; RedistributionPlan plan; };
    SmallVector<Pending, 0> pending;
    DenseMap<Operation *, uint64_t> usedScratch;
    WalkResult walked = getOperation().walk([&](ConvertLayoutOp op) {
      auto fail = [&](StringRef reason) {
        op.emitError("test lowering requires a static single-CTA redistribution: ") << reason;
        return WalkResult::interrupt();
      };
      auto function = op->getParentOfType<gpu::GPUFuncOp>();
      if (!function || !function.isKernel() || !llvm::hasSingleElement(function.getBody()) ||
          op->getBlock() != &function.getBody().front())
        return fail("requires uniform entry-block execution in a single-block gpu.func kernel");
      auto plan = planRedistribution(cast<RankedTensorType>(op.getSource().getType()),
                                     cast<RankedTensorType>(op.getResult().getType()), op.getLoc());
      if (failed(plan)) return WalkResult::interrupt();
      auto block = function.getKnownBlockSize();
      if (!block || block->size() != 3 || (*block)[0] != int32_t(plan->threads) ||
          (*block)[1] != 1 || (*block)[2] != 1)
        return fail("requires known_block_size matching topology");
      if (plan->mode == ExchangeMode::Shared) {
        auto [it, inserted] = usedScratch.try_emplace(function, 0);
        if (inserted) {
          auto workgroupType = [](Type type) {
            auto memref = dyn_cast<BaseMemRefType>(type);
            if (!memref) return false;
            Attribute space = memref.getMemorySpace();
            if (auto gpuSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(space))
              return gpuSpace.getValue() == gpu::AddressSpace::Workgroup;
            auto integer = dyn_cast_or_null<IntegerAttr>(space);
            return integer && integer.getInt() == 3;
          };
          // Dynamic/shared allocations and incoming aliases are not bounded by
          // the attribution accounting below. Refuse rather than undercount.
          bool unaccounted = llvm::any_of(function.getArgumentTypes(), workgroupType);
          function.walk([&](Operation *nested) {
            unaccounted |= llvm::any_of(nested->getResultTypes(), workgroupType);
          });
          if (unaccounted) return fail("cannot account for non-attributed workgroup storage");
          for (auto [i, arg] : llvm::enumerate(function.getWorkgroupAttributions())) {
            auto ty = dyn_cast<MemRefType>(arg.getType());
            if (!ty || !ty.hasStaticShape() || !ty.getLayout().isIdentity())
              return fail("cannot account for existing workgroup attribution footprint");
            auto payload = planPayload(ty.getElementType(), op.getLoc());
            if (failed(payload)) return WalkResult::interrupt();
            uint64_t bytes = payload->bits / 8;
            for (int64_t extent : ty.getShape()) {
              if (extent < 0 || (extent && bytes > scratchBudget / uint64_t(extent)))
                return fail("existing workgroup attribution exceeds adapter scratch budget");
              bytes *= extent;
            }
            uint64_t alignment = payload->bits / 8;
            if (auto attr = function.getWorkgroupAttributionAttr(i, bString("llvm.align"))) {
              auto integer = dyn_cast<IntegerAttr>(attr);
              if (!integer || integer.getInt() <= 0) return fail("invalid existing workgroup alignment");
              alignment = std::max(alignment, uint64_t(integer.getInt()));
            }
            auto total = reserveScratch(it->second, bytes, alignment, scratchBudget, op.getLoc());
            if (failed(total)) return WalkResult::interrupt();
            it->second = *total;
          }
        }
        auto total = reserveScratch(it->second, plan->scratchBytes, plan->payload.bits / 8,
                                     scratchBudget, op.getLoc());
        if (failed(total)) return WalkResult::interrupt();
        it->second = *total;
      }
      pending.push_back({op, function, std::move(*plan)});
      return WalkResult::advance();
    });
    if (walked.wasInterrupted()) { signalPassFailure(); return; }
    // No mutations until every conversion and aggregate scratch budget passes.
    for (auto &item : pending) emitPlan(item.op, item.function, item.plan);
  }
  StringAttr bString(StringRef name) { return StringAttr::get(&getContext(), name); }
};
} // namespace
std::unique_ptr<Pass> createTestLowerLayoutConversionsPass() {
  return std::make_unique<TestLowerLayoutConversionsPass>();
}
} // namespace mlir::frisk
