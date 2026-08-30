#ifndef FRISK_IR_FRISKLAYOUTINTERFACES_H
#define FRISK_IR_FRISKLAYOUTINTERFACES_H

#include "Dialect/Frisk/Analysis/LayoutCommon.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::frisk {
class LayoutConstraintBuilder;
} // namespace mlir::frisk

#include "Dialect/Frisk/IR/FriskLayoutAttrInterfaces.h.inc"
#include "Dialect/Frisk/IR/FriskLayoutOpInterfaces.h.inc"

#endif // FRISK_IR_FRISKLAYOUTINTERFACES_H
