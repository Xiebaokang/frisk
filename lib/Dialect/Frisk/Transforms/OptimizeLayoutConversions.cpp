#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_OPTIMIZELAYOUTCONVERSIONS
#include "Dialect/Frisk/Transforms/Passes.h.inc"
namespace {
class OptimizeLayoutConversionsPass final
    : public impl::OptimizeLayoutConversionsBase<OptimizeLayoutConversionsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
}
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass() {
  return std::make_unique<OptimizeLayoutConversionsPass>();
}
}
