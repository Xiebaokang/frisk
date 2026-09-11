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

LogicalResult verifyStaticWholeTile(Operation *op, ShapedType memref,
                                    RankedTensorType tensor,
                                    StringRef mismatchMessage) {
  if (!memref.hasStaticShape() || !tensor.hasStaticShape())
    return op->emitOpError()
           << op->getName().stripDialect()
           << " requires static source and result shapes";
  if (memref.getShape() != tensor.getShape() ||
      memref.getElementType() != tensor.getElementType())
    return op->emitOpError(mismatchMessage);

  Attribute encoding = tensor.getEncoding();
  if (!encoding)
    return success();
  auto distributed = dyn_cast<DistributedEncodingAttr>(encoding);
  if (!distributed)
    return op->emitOpError(
        "tensor encoding must be a DistributedEncodingAttr");
  return distributed.verifyForType(tensor, op->getLoc());
}

bool haveEquivalentDistributedEncodings(RankedTensorType lhs,
                                        RankedTensorType rhs) {
  auto lhsEncoding =
      dyn_cast_or_null<DistributedEncodingAttr>(lhs.getEncoding());
  auto rhsEncoding =
      dyn_cast_or_null<DistributedEncodingAttr>(rhs.getEncoding());
  if (!lhsEncoding || !rhsEncoding ||
      lhsEncoding.getTopology() != rhsEncoding.getTopology() ||
      lhsEncoding.getReplication() != rhsEncoding.getReplication())
    return false;
  FailureOr<Attribute> lhsMap = lhsEncoding.getCanonicalMap(lhs);
  FailureOr<Attribute> rhsMap = rhsEncoding.getCanonicalMap(rhs);
  return succeeded(lhsMap) && succeeded(rhsMap) && *lhsMap == *rhsMap;
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

LogicalResult TileLoadOp::verify() {
  return verifyStaticWholeTile(
      getOperation(), cast<MemRefType>(getSource().getType()),
      cast<RankedTensorType>(getResult().getType()),
      "source memref and result tensor must have identical static shape and "
      "element type");
}

LogicalResult TileStoreOp::verify() {
  auto valueType = cast<RankedTensorType>(getValue().getType());
  auto targetType = cast<MemRefType>(getTarget().getType());
  if (!valueType.hasStaticShape() || !targetType.hasStaticShape())
    return emitOpError("tile_store requires static value and target shapes");
  if (valueType.getShape() != targetType.getShape() ||
      valueType.getElementType() != targetType.getElementType())
    return emitOpError(
        "value tensor and target memref must have identical static shape and "
        "element type");
  Attribute encoding = valueType.getEncoding();
  if (!encoding)
    return success();
  auto distributed = dyn_cast<DistributedEncodingAttr>(encoding);
  if (!distributed)
    return emitOpError("tensor encoding must be a DistributedEncodingAttr");
  return distributed.verifyForType(valueType, getLoc());
}

LogicalResult ConvertLayoutOp::verify() {
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (sourceType.getShape() != resultType.getShape() ||
      sourceType.getElementType() != resultType.getElementType())
    return emitOpError(
        "source and target must have identical shape and element type");

  auto sourceEncoding =
      dyn_cast_or_null<DistributedEncodingAttr>(sourceType.getEncoding());
  auto resultEncoding =
      dyn_cast_or_null<DistributedEncodingAttr>(resultType.getEncoding());
  if (!sourceEncoding || !resultEncoding)
    return emitOpError("source and target must use DistributedEncodingAttr");
  if (failed(sourceEncoding.verifyForType(sourceType, getLoc())) ||
      failed(resultEncoding.verifyForType(resultType, getLoc())))
    return failure();
  return success();
}

LogicalResult ConvertLayoutOp::canonicalize(ConvertLayoutOp op,
                                            PatternRewriter &rewriter) {
  auto sourceType = cast<RankedTensorType>(op.getSource().getType());
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  if (sourceType == resultType &&
      haveEquivalentDistributedEncodings(sourceType, resultType)) {
    rewriter.replaceOp(op, op.getSource());
    return success();
  }

  auto inner = op.getSource().getDefiningOp<ConvertLayoutOp>();
  if (!inner)
    return failure();
  auto innerSourceType =
      cast<RankedTensorType>(inner.getSource().getType());
  if (innerSourceType == resultType &&
      haveEquivalentDistributedEncodings(innerSourceType, resultType)) {
    rewriter.replaceOp(op, inner.getSource());
    return success();
  }
  // Composition preserves the exact requested result type. A semantic map
  // identity with distinct attribute spelling is not an SSA type identity.
  rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, resultType, inner.getSource());
  return success();
}

} // namespace mlir::frisk
