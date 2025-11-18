#ifndef FRISK_ATTRIBUTES_H
#define FRISK_ATTRIBUTES_H

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"

#include "Dialect/Frisk/IR/FriskEnums.h"

namespace mlir::frisk {
class FriskDialect;
} // namespace mlir::frisk

#define GET_ATTRDEF_CLASSES
#include "Dialect/Frisk/IR/FriskAttributes.h.inc"

#endif // FRISK_ATTRIBUTES_H
