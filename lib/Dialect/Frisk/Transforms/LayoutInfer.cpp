#include "Dialect/Frisk/Transforms/Passes.h"

#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_FRISKINFERLAYOUTS
#include "Dialect/Frisk/Transforms/Passes.h.inc"

namespace {
class FriskInferLayoutsPass final
    : public impl::FriskInferLayoutsBase<FriskInferLayoutsPass> {
public:
  void runOnOperation() override {
    std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
    FailureOr<LayoutConstraintGraph> graph =
        collectLayoutConstraints(getOperation(), *target);
    if (failed(graph) || failed(propagateStrict(*graph)) ||
        failed(propagateCommonToFixedPoint(*graph))) {
      signalPassFailure();
      return;
    }
    FailureOr<LayoutSolution> solution =
        solveBootstrapLayoutGraph(*graph, *target);
    if (failed(solution) ||
        failed(verifySolvedLayoutGraph(*graph, *solution, *target,
                                      getOperation().getLoc())) ||
        failed(materializeLayouts(getOperation(), *graph, *solution)) ||
        failed(verifyMaterializedLayouts(getOperation(), *target)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createFriskInferLayoutsPass() {
  return std::make_unique<FriskInferLayoutsPass>();
}

} // namespace mlir::frisk
