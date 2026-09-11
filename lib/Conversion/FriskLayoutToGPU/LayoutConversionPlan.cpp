#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::frisk::test {
namespace {
constexpr uint64_t maxVolume = 65536, maxCarriers = 262144;
InFlightDiagnostic unsupported(Location loc) {
  return emitError(loc) << "test lowering requires a static single-CTA redistribution: ";
}
unsigned field(Owner o, StringRef name) {
  if (name == "register") return o.reg;
  if (name == "lane") return o.lane;
  if (name == "warp") return o.warp;
  if (name == "warp_group") return o.warpGroup;
  return 0; // CTA is proved singleton.
}
APInt encode(BitLinearLayoutMapAttr map, Owner o, unsigned bits) {
  APInt result(bits, 0);
  unsigned offset = 0;
  for (auto [name, width] : llvm::zip_equal(map.getInputNames(), map.getInputBitWidths().asArrayRef())) {
    result |= APInt(bits, field(o, cast<StringAttr>(name).getValue())).shl(offset);
    offset += width;
  }
  return result;
}
Owner decode(BitLinearLayoutMapAttr map, APInt bits) {
  Owner o;
  unsigned offset = 0;
  for (auto [name, width] : llvm::zip_equal(map.getInputNames(), map.getInputBitWidths().asArrayRef())) {
    unsigned v = bits.extractBitsAsZExtValue(width, offset);
    StringRef n = cast<StringAttr>(name).getValue();
    if (n == "register") o.reg = v;
    else if (n == "lane") o.lane = v;
    else if (n == "warp") o.warp = v;
    else if (n == "warp_group") o.warpGroup = v;
    offset += width;
  }
  return o;
}
uint64_t slotOf(BitLinearLayoutMapAttr map, APInt bits, ArrayRef<int64_t> shape) {
  uint64_t slot = 0; unsigned offset = 0;
  for (auto [extent, width] : llvm::zip_equal(shape, map.getOutputBitWidths().asArrayRef())) {
    slot = slot * extent + bits.extractBitsAsZExtValue(width, offset);
    offset += width;
  }
  return slot;
}
APInt bitsOfSlot(uint64_t slot, ArrayRef<int64_t> shape, unsigned bits) {
  SmallVector<uint64_t> coords(shape.size());
  for (int i = int(shape.size()) - 1; i >= 0; --i) {
    coords[i] = slot % shape[i]; slot /= shape[i];
  }
  APInt result(bits, 0); unsigned offset = 0;
  for (auto [coord, extent] : llvm::zip_equal(coords, shape)) {
    result |= APInt(bits, coord).shl(offset); offset += llvm::Log2_64(extent);
  }
  return result;
}
LogicalResult counterexample(Location loc, StringRef reason, uint64_t slot,
                              ArrayRef<int64_t> shape) {
  SmallVector<uint64_t> coords(shape.size());
  for (int i = int(shape.size()) - 1; i >= 0; --i) {
    coords[i] = slot % shape[i]; slot /= shape[i];
  }
  auto diag = unsupported(loc);
  diag << reason << "; first logical coordinate [";
  llvm::interleaveComma(coords, diag); diag << "]";
  return failure();
}
} // namespace

FailureOr<PayloadTransportPlan> planPayload(Type type, Location loc) {
  bool supported = type.isSignlessInteger(8) || type.isSignlessInteger(16) ||
      type.isSignlessInteger(32) || type.isSignlessInteger(64) ||
      type.isF16() || type.isBF16() || type.isF32() || type.isF64();
  if (!supported) {
    unsupported(loc) << "unsupported element type " << type
                     << "; requires i8/i16/i32/i64 or f16/bf16/f32/f64";
    return failure();
  }
  unsigned bits = type.getIntOrFloatBitWidth();
  PayloadTransportPlan p{bits, {}};
  for (unsigned offset = 0; offset < bits; offset += 32)
    p.words.push_back({offset, std::min(32u, bits - offset)});
  return p;
}

FailureOr<uint64_t> reserveScratch(uint64_t used, uint64_t bytes,
                                   uint64_t alignment, uint64_t budget,
                                   Location loc) {
  if (!llvm::isPowerOf2_64(alignment) || used > budget ||
      used > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    unsupported(loc) << "invalid scratch alignment or adapter scratch budget overflow";
    return failure();
  }
  uint64_t aligned = llvm::alignTo(used, alignment);
  if (aligned > budget || bytes > budget - aligned) {
    unsupported(loc) << "adapter scratch budget exceeded (budget " << budget << " bytes)";
    return failure();
  }
  return aligned + bytes;
}

FailureOr<RedistributionPlan> planRedistribution(RankedTensorType source,
                                                RankedTensorType destination,
                                                Location loc) {
  if (!source.hasStaticShape() || !destination.hasStaticShape() ||
      source.getShape() != destination.getShape() ||
      source.getElementType() != destination.getElementType()) {
    unsupported(loc) << "requires identical static shape and element type";
    return failure();
  }
  auto se = dyn_cast_or_null<DistributedEncodingAttr>(source.getEncoding());
  auto de = dyn_cast_or_null<DistributedEncodingAttr>(destination.getEncoding());
  if (!se || !de) {
    unsupported(loc) << "requires distributed source and destination encodings";
    return failure();
  }
  if (failed(se.verifyForType(source, loc)) || failed(de.verifyForType(destination, loc)))
    return failure();
  auto sc = se.getCanonicalMap(source), dc = de.getCanonicalMap(destination);
  if (failed(sc) || failed(dc) || !isa<BitLinearLayoutMapAttr>(*sc) || !isa<BitLinearLayoutMapAttr>(*dc)) {
    unsupported(loc) << "requires BitLinear maps"; return failure();
  }
  auto sm = cast<BitLinearLayoutMapAttr>(*sc), dm = cast<BitLinearLayoutMapAttr>(*dc);
  if (sm.getOutputNames() != dm.getOutputNames() || sm.getOutputBitWidths() != dm.getOutputBitWidths()) {
    unsupported(loc) << "logical output names/order differ"; return failure();
  }
  auto st = se.getTopology().asArrayRef(), dt = de.getTopology().asArrayRef();
  if (st.drop_front() != dt.drop_front()) {
    unsupported(loc) << "source/destination execution topologies differ"; return failure();
  }
  if (st[4] != 1) { unsupported(loc) << "cross-CTA/cluster communication is unsupported"; return failure(); }
  if (st[1] != 32) { unsupported(loc) << "requires lane extent 32"; return failure(); }
  if (st[2] > 32 || st[3] > 32 || st[2] * st[3] > 32 || st[0] > 256 || dt[0] > 256) {
    unsupported(loc) << "adapter thread/register limit exceeded"; return failure();
  }
  RedistributionPlan p;
  p.threads = 32 * st[2] * st[3]; p.warps = st[2];
  p.sourceRegisters = st[0]; p.destinationRegisters = dt[0];
  p.volume = 1;
  for (int64_t extent : source.getShape()) {
    if (extent <= 0 || uint64_t(extent) > maxVolume / p.volume) {
      unsupported(loc) << "adapter logical volume limit exceeded"; return failure();
    }
    p.volume *= extent;
  }
  if (uint64_t(p.threads) * (st[0] + dt[0]) > maxCarriers) {
    unsupported(loc) << "adapter carrier enumeration limit exceeded"; return failure();
  }
  auto payload = planPayload(source.getElementType(), loc);
  if (failed(payload)) return failure();
  p.payload = *payload;
  auto ds = sm.getMatrixValue(), dd = dm.getMatrixValue();
  auto inverse = ds->rightInverse();
  if (failed(inverse)) {
    (void)counterexample(loc, "missing source coverage", 0, source.getShape()); return failure();
  }
  auto redistribution = inverse->compose(*dd);
  auto composed = succeeded(redistribution) ? ds->compose(*redistribution) : FailureOr<GF2Matrix>(failure());
  if (failed(composed) || *composed != *dd) {
    (void)counterexample(loc, "right-inverse composition proof failed", 0, source.getShape()); return failure();
  }
  p.globalWriters.resize(p.volume);
  // First enumerated owner is lexicographically smallest physical-thread/reg.
  DenseMap<std::pair<uint64_t, unsigned>, Owner> threadOwners, warpOwners;
  for (unsigned t = 0; t < p.threads; ++t)
    for (unsigned r = 0; r < p.sourceRegisters; ++r) {
      Owner o = p.owner(t, r);
      uint64_t slot = slotOf(sm, ds->apply(encode(sm, o, ds->getNumColumns())), source.getShape());
      if (slot >= p.volume) {
        (void)counterexample(loc, "source scratch slot out of bounds", slot, source.getShape()); return failure();
      }
      p.sourceSlots.push_back(slot);
      threadOwners.try_emplace({slot, t}, o);
      warpOwners.try_emplace({slot, t / 32}, o);
    }
  for (uint64_t slot = 0; slot < p.volume; ++slot) {
    Owner o = decode(sm, inverse->apply(bitsOfSlot(slot, source.getShape(), ds->getNumRows())));
    if (o.reg >= p.sourceRegisters || o.lane >= 32 || o.warp >= unsigned(st[2]) ||
        o.warpGroup >= unsigned(st[3]) || p.sourceSlots[p.thread(o) * p.sourceRegisters + o.reg] != slot) {
      (void)counterexample(loc, "missing or out-of-bounds global source owner", slot, source.getShape()); return failure();
    }
    p.globalWriters[slot] = o;
  }
  for (unsigned t = 0; t < p.threads; ++t)
    for (unsigned r = 0; r < p.destinationRegisters; ++r) {
      APInt input = encode(dm, p.owner(t, r), dd->getNumColumns());
      uint64_t slot = slotOf(dm, dd->apply(input), destination.getShape());
      Owner o = decode(sm, redistribution->apply(input));
      auto local = threadOwners.find({slot, t});
      auto warp = warpOwners.find({slot, t / 32});
      if (local != threadOwners.end()) o = local->second;
      else if (warp != warpOwners.end()) o = warp->second;
      if (slot >= p.volume || o.reg >= p.sourceRegisters || o.lane >= 32 ||
          o.warp >= unsigned(st[2]) || o.warpGroup >= unsigned(st[3]) ||
          p.sourceSlots[p.thread(o) * p.sourceRegisters + o.reg] != slot) {
        (void)counterexample(loc, "destination lacks a valid source owner", slot, destination.getShape()); return failure();
      }
      if (p.thread(o) / 32 != t / 32) p.mode = ExchangeMode::Shared;
      else if (p.thread(o) != t && p.mode == ExchangeMode::Register) p.mode = ExchangeMode::Shuffle;
      p.destinations.push_back({o, slot});
    }
  if (p.mode == ExchangeMode::Shared) p.scratchBytes = p.volume * (p.payload.bits / 8);
  if (p.mode == ExchangeMode::Shuffle &&
      uint64_t(p.sourceRegisters) * p.destinationRegisters * p.payload.words.size() > 65536) {
    unsupported(loc) << "adapter shuffle code-size limit exceeded"; return failure();
  }
  return p;
}
} // namespace mlir::frisk::test
