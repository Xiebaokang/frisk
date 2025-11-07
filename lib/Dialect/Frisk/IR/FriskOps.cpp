#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#define GET_OP_CLASSES
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"

// move dialect def in this file to make compiler happy
#include "Dialect/Frisk/IR/FriskDialect.cpp.inc"
namespace mlir {
namespace frisk {

void FriskDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"
      >();
}

} // namespace frisk
} // namespace mlir

namespace mlir {
namespace frisk {



// -- GemmOp --
LogicalResult GemmOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                      Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  // auto lhs_type = cast<MemrefType>(adaptor.getLhs().getType());
  // auto rhs_type = cast<MemrefType>(adaptor.getRhs().getType());

  // auto element_type = lhs_type.getElementType();

  // auto lhs_rank = lhs_type.getRank();
  // auto rhs_rank = rhs_type.getRank();
  // if (lhs_rank < 2 || rhs_rank < 2) {
  //   return emitOptionalError(location, "lhs and rhs rank both need >= 2");
  // }

  // auto lhs_shape = lhs_type.getShape();
  // auto rhs_shape = rhs_type.getShape();
  // if (lhs_shape[lhs_rank - 1] != rhs_shape[rhs_rank - 2]) {
  //   return emitOptionalError(location, "lhs_shape[-2] and rhs_shape[-1] doesn't match");
  // }

  // if (lhs_rank != rhs_rank) {
  //   // batch dim broadcast
  //   if (lhs_rank > 2 && rhs_rank > 2) {
  //     return emitOptionalError(location, "operands broadcast batch dim differently, unaccepted");
  //   }

  //   if (lhs_rank > rhs_rank) {
  //     assert(rhs_rank == 2);
  //     llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
  //     return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
  //     auto return_type = RankedTensorType::get(return_shape, element_type);
  //     inferredReturnTypes.push_back(return_type);
  //   } else {
  //     assert(lhs_rank == 2);
  //     llvm::SmallVector<int64_t, 4> return_shape(rhs_shape);
  //     return_shape[rhs_rank - 2] = lhs_shape[lhs_rank - 1];
  //     auto return_type = RankedTensorType::get(return_shape, element_type);
  //     inferredReturnTypes.push_back(return_type);
  //   }
  // } else {
  //   llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
  //   return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
  //   auto return_type = RankedTensorType::get(return_shape, element_type);
  //   inferredReturnTypes.push_back(return_type);
  // }
  return success();
}

} // namespace frisk
} // namespace mlir