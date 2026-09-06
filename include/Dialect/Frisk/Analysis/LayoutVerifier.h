#ifndef FRISK_ANALYSIS_LAYOUTVERIFIER_H
#define FRISK_ANALYSIS_LAYOUTVERIFIER_H

#include "Dialect/Frisk/Analysis/LayoutSolver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::frisk {

struct LayoutSolution {
  DenseMap<LayoutVarID, Attribute> assignments;
  SmallVector<LayoutConversionEdge> conversions;
};

struct BootstrapSolverLimits {
  unsigned maxVariables = 8;
  unsigned maxDomainSize = 4;
};

FailureOr<LayoutSolution>
solveBootstrapLayoutGraph(LayoutConstraintGraph &graph, LayoutTarget &target,
                          BootstrapSolverLimits limits = {});

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target, Location loc);

LogicalResult materializeLayouts(Operation *root,
                                 const LayoutConstraintGraph &graph,
                                 const LayoutSolution &solution);

LogicalResult verifyMaterializedLayouts(Operation *root,
                                        LayoutTarget &target);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTVERIFIER_H
