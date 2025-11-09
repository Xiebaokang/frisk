#ifndef FRISK_TRANSFORMS_PASSES_H
#define FRISK_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::frisk {

#define GEN_PASS_DECL
#include "Dialect/Frisk/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createLayoutInferPass();



#define GEN_PASS_REGISTRATION
#include "Dialect/Frisk/Transforms/Passes.h.inc"

} // namespace mlir::frisk

#endif // FRISK_TRANSFORMS_PASSES_H