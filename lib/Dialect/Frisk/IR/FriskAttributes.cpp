#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Frisk/IR/FriskAttributes.cpp.inc"

namespace mlir::frisk {
void FriskDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Frisk/IR/FriskAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"
      >();
}
} // namespace mlir::frisk
