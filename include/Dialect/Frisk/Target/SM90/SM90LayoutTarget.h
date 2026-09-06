#ifndef FRISK_TARGET_SM90_SM90LAYOUTTARGET_H
#define FRISK_TARGET_SM90_SM90LAYOUTTARGET_H

#include <memory>

#include "Dialect/Frisk/Analysis/LayoutTarget.h"

namespace mlir::frisk {

std::unique_ptr<LayoutTarget> createSM90LayoutTarget();

} // namespace mlir::frisk

#endif // FRISK_TARGET_SM90_SM90LAYOUTTARGET_H
