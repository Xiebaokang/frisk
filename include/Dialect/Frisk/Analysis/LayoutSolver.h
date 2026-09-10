#ifndef FRISK_ANALYSIS_LAYOUTSOLVER_H
#define FRISK_ANALYSIS_LAYOUTSOLVER_H

#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/Analysis/LayoutTarget.h"

#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/Value.h"

namespace mlir::frisk {

class LayoutConstraintBuilder {
public:
  explicit LayoutConstraintBuilder(LayoutConstraintGraph &graph)
      : graph(graph) {}

  LayoutVarID getOrCreateStorageVar(Value anchor);
  LayoutVarID getOrCreateDistributedVar(Value value);
  LogicalResult require(LayoutVarID var, Attribute encoding,
                        Operation *source, StringRef rule);
  LogicalResult same(LayoutVarID lhs, LayoutVarID rhs, Operation *source,
                     StringRef rule);
  LayoutVarID getOrCreateDistributedUse(OpOperand &use);
  LogicalResult transform(LayoutVarID src, LayoutVarID dst,
                          Attribute coordinateTransform, Operation *source,
                          StringRef rule);
  LogicalResult storageAccess(LayoutVarID distributed, LayoutVarID storage,
                              AccessKind access, Operation *source,
                              StringRef rule);
  LogicalResult convertible(LayoutVarID src, LayoutVarID dst, OpOperand &use,
                            bool existing = false);

  std::optional<LayoutVarID>
  lookup(Value value, LayoutKind kind = LayoutKind::Storage) const;

private:
  LayoutVarID getOrCreate(Value value, LayoutKind kind);

  LayoutConstraintGraph &graph;
  DenseMap<Value, LayoutVarID> storageVariablesByValue;
  DenseMap<Value, LayoutVarID> distributedVariablesByValue;
  DenseMap<OpOperand *, LayoutVarID> distributedVariablesByUse;
  uint64_t nextStableOrdinal = 0;
};

FailureOr<LayoutConstraintGraph>
collectLayoutConstraints(Operation *root, LayoutTarget &target);

/// Extends the graph before finalization; SCF tensor models are Task 16.
LogicalResult collectDistributedLayoutConstraints(
    Operation *root, LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder);

LogicalResult propagateStrict(LayoutConstraintGraph &graph);
LogicalResult propagateCommonToFixedPoint(LayoutConstraintGraph &graph);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTSOLVER_H
