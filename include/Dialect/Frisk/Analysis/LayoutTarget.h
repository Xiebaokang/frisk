#ifndef FRISK_ANALYSIS_LAYOUTTARGET_H
#define FRISK_ANALYSIS_LAYOUTTARGET_H

#include "Dialect/Frisk/Analysis/LayoutConstraint.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::frisk {

struct CandidateAssignment {
  DenseMap<LayoutVarID, Attribute> values;
};

class LayoutTarget {
public:
  virtual ~LayoutTarget() = default;

  virtual void
  enumerateCandidates(const LayoutVar &var,
                      SmallVectorImpl<LayoutCandidate> &out) const = 0;
  virtual LogicalResult verifyCandidate(const LayoutVar &var,
                                        Attribute candidate,
                                        Location loc) const = 0;
  virtual FailureOr<CostVector>
  evaluate(const CandidateAssignment &assignment) const = 0;
};

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTTARGET_H
