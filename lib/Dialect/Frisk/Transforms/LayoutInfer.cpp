#include "Dialect/Frisk/Transforms/Passes.h"

#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_FRISKINFERLAYOUTS
#include "Dialect/Frisk/Transforms/Passes.h.inc"

namespace {
bool isSupportedSM90Target(StringRef target) {
  return target == "sm90" || target == "sm_90" || target == "sm90a" ||
         target == "sm_90a";
}

LogicalResult verifySM90TargetBoundary(Operation *root) {
  WalkResult result = root->walk([&](Operation *operation) {
    Attribute rawTarget = operation->getAttr("frisk.target");
    if (!rawTarget)
      return WalkResult::advance();
    auto target = dyn_cast<StringAttr>(rawTarget);
    if (target && isSupportedSM90Target(target.getValue()))
      return WalkResult::advance();
    operation->emitError(
        "frisk-infer-layouts supports only NVIDIA SM90/SM90a targets; got ")
        << rawTarget;
    return WalkResult::interrupt();
  });
  return success(!result.wasInterrupted());
}

class FriskInferLayoutsPass final
    : public impl::FriskInferLayoutsBase<FriskInferLayoutsPass> {
public:
  void runOnOperation() override {
    if (failed(verifySM90TargetBoundary(getOperation()))) {
      signalPassFailure();
      return;
    }
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
