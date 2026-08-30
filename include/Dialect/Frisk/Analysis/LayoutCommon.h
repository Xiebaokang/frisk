#ifndef FRISK_ANALYSIS_LAYOUTCOMMON_H
#define FRISK_ANALYSIS_LAYOUTCOMMON_H

#include <cstdint>
#include <string>

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::frisk {

class LayoutMapAttrInterface;

enum class LayoutKind { Distributed, Storage, Instruction };

enum class ProofStatus { Proven, Disproven, Unknown };

struct LayoutProof {
  ProofStatus status = ProofStatus::Unknown;
  llvm::SmallVector<int64_t> counterexample;
  std::string reason;
};

struct CostVector {
  uint64_t instructionPathAndWork = 0;
  uint64_t memoryTransactions = 0;
  uint64_t bankConflictDegree = 0;
  uint64_t conversionBytesAndSync = 0;
  uint64_t spillRiskAndRegisters = 0;
  uint64_t sharedBytesAndOccupancy = 0;
  uint64_t replication = 0;
  uint64_t codeSize = 0;
  uint64_t deterministicTieBreak = 0;
};

struct LayoutDimension {
  StringAttr name;
  int64_t extent;
};

using LayoutMapAttr = LayoutMapAttrInterface;

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTCOMMON_H
