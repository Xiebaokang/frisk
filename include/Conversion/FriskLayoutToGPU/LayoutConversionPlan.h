#ifndef FRISK_LAYOUT_CONVERSION_PLAN_H
#define FRISK_LAYOUT_CONVERSION_PLAN_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::frisk::test {
// Test-only per-thread vector ABI. Public Frisk carriers remain tensors.
struct Owner {
  unsigned reg = 0, lane = 0, warp = 0, warpGroup = 0;
  bool operator==(const Owner &o) const {
    return reg == o.reg && lane == o.lane && warp == o.warp && warpGroup == o.warpGroup;
  }
};
enum class ExchangeMode { Register, Shuffle, Shared };
struct PayloadWord { unsigned bitOffset, liveBits; };
struct PayloadTransportPlan {
  unsigned bits;
  SmallVector<PayloadWord, 2> words;
};
struct Destination { Owner source; uint64_t slot; };
struct RedistributionPlan {
  ExchangeMode mode = ExchangeMode::Register;
  unsigned threads = 0, warps = 0, sourceRegisters = 0, destinationRegisters = 0;
  uint64_t volume = 0, scratchBytes = 0;
  PayloadTransportPlan payload;
  // Both carrier tables are physical-thread-major, then register-major.
  SmallVector<Destination> destinations;
  SmallVector<uint64_t> sourceSlots;
  // Independent from locally preferred destination owners: one writer/slot.
  SmallVector<Owner> globalWriters;
  Owner owner(unsigned thread, unsigned reg) const {
    return {reg, thread % 32, (thread / 32) % warps, thread / (32 * warps)};
  }
  unsigned thread(Owner o) const { return (o.warpGroup * warps + o.warp) * 32 + o.lane; }
};
FailureOr<PayloadTransportPlan> planPayload(Type elementType, Location loc);
FailureOr<RedistributionPlan> planRedistribution(RankedTensorType source,
                                                RankedTensorType destination,
                                                Location loc);
// Overflow-safe aligned accounting shared by pass and unit tests.
FailureOr<uint64_t> reserveScratch(uint64_t used, uint64_t bytes,
                                   uint64_t alignment, uint64_t budget,
                                   Location loc);
} // namespace mlir::frisk::test
#endif
