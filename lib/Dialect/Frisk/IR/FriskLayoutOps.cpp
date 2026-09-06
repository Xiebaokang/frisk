#include "Dialect/Frisk/IR/FriskOps.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::frisk {

namespace {

bool isIdentityStorageLayout(StorageLayoutAttr layout, MemRefType type) {
  auto affine = dyn_cast<AffineLayoutMapAttr>(layout.getMap());
  if (!affine || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !type.getElementType().isIntOrFloat())
    return false;

  unsigned elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0 || elementBits % 8 != 0)
    return false;

  MLIRContext *context = type.getContext();
  AffineExpr byteOffset = getAffineConstantExpr(0, context);
  int64_t stride = elementBits / 8;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    byteOffset = byteOffset + getAffineDimExpr(dim, context) * stride;
    stride *= type.getDimSize(dim);
  }
  AffineMap expected = AffineMap::get(
      type.getRank(), 0,
      {byteOffset, getAffineConstantExpr(0, context)}, context);
  return affine.getAffineMap().getValue() == expected;
}

bool hasNonReturnUser(LayoutViewOp op) {
  return llvm::any_of(op.getResult().getUsers(), [&](Operation *user) {
    return user->getName().getStringRef() != "func.return";
  });
}

} // namespace

Value LayoutViewOp::getViewSource() { return getSource(); }

LogicalResult LayoutViewOp::verify() {
  if (getSource().getType() != getResult().getType())
    return emitOpError("source and result must have identical memref types");
  if (StorageLayoutAttr layout = getLayoutAttr())
    return layout.verifyForType(cast<MemRefType>(getResult().getType()),
                                getLoc());
  return success();
}

LogicalResult LayoutViewOp::canonicalize(LayoutViewOp op,
                                         PatternRewriter &rewriter) {
  if (StorageLayoutAttr layout = op.getLayoutAttr()) {
    if (isIdentityStorageLayout(
            layout, cast<MemRefType>(op.getResult().getType())) &&
        !hasNonReturnUser(op)) {
      rewriter.replaceOp(op, op.getSource());
      return success();
    }
  }

  auto inner = op.getSource().getDefiningOp<LayoutViewOp>();
  if (!inner || !op.getLayoutAttr() ||
      op.getLayoutAttr() != inner.getLayoutAttr())
    return failure();

  rewriter.modifyOpInPlace(op, [&] { op->setOperand(0, inner.getSource()); });
  return success();
}

} // namespace mlir::frisk
