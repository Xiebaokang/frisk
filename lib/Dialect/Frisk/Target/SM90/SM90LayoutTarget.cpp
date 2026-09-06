#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"

#include "Dialect/Frisk/IR/FriskAttributes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::frisk {

namespace {

ArrayAttr getDimensionNames(Builder &builder, unsigned rank) {
  SmallVector<Attribute> names;
  for (unsigned index = 0; index < rank; ++index)
    names.push_back(builder.getStringAttr("dim" + Twine(index)));
  return builder.getArrayAttr(names);
}

std::optional<attr::MemorySpace> getMemorySpace(MemRefType type) {
  std::optional<attr::MemorySpace> space =
      attr::symbolizeMemorySpace(type.getMemorySpaceAsInt());
  if (!space || *space == attr::MemorySpace::Local)
    return std::nullopt;
  return space;
}

StorageLayoutAttr buildAffineStorage(MemRefType type, AffineExpr linear,
                                     int64_t storageElements) {
  MLIRContext *context = type.getContext();
  Builder builder(context);
  int64_t elementBits = type.getElementTypeBitWidth();
  AffineExpr bitAddress = linear * elementBits;
  AffineMap map = AffineMap::get(
      type.getRank(), 0, {bitAddress.floorDiv(8), bitAddress % 8}, context);
  int64_t storageBytes = llvm::divideCeil(
      static_cast<uint64_t>(storageElements) * elementBits, uint64_t{8});
  auto affine = AffineLayoutMapAttr::get(
      context, getDimensionNames(builder, type.getRank()),
      builder.getDenseI64ArrayAttr(type.getShape()),
      builder.getArrayAttr({builder.getStringAttr("byte_offset"),
                            builder.getStringAttr("bit_offset")}),
      builder.getDenseI64ArrayAttr({storageBytes, int64_t{8}}),
      AffineMapAttr::get(map));
  std::optional<attr::MemorySpace> space = getMemorySpace(type);
  if (!space)
    return {};
  int64_t elementBytes = std::max<int64_t>(1, elementBits / 8);
  int64_t alignment = llvm::isPowerOf2_64(elementBytes) ? elementBytes : 1;
  return StorageLayoutAttr::get(
      context, affine, MemorySpaceAttr::get(context, *space),
      builder.getI64IntegerAttr(alignment),
      builder.getI64IntegerAttr(alignment));
}

StorageLayoutAttr buildLinear(MemRefType type) {
  MLIRContext *context = type.getContext();
  AffineExpr linear = getAffineConstantExpr(0, context);
  int64_t stride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    linear = linear + getAffineDimExpr(dim, context) * stride;
    stride *= type.getDimSize(dim);
  }
  return buildAffineStorage(type, linear, stride);
}

StorageLayoutAttr buildTranspose(MemRefType type) {
  if (type.getRank() != 2)
    return {};
  MLIRContext *context = type.getContext();
  AffineExpr linear = getAffineDimExpr(1, context) * type.getDimSize(0) +
                      getAffineDimExpr(0, context);
  return buildAffineStorage(type, linear, type.getNumElements());
}

StorageLayoutAttr buildPadded(MemRefType type) {
  if (type.getRank() != 2)
    return {};
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0 || elementBits % 8 != 0)
    return {};
  int64_t elementsPer32Bytes = std::max<int64_t>(1, 256 / elementBits);
  int64_t columns = type.getDimSize(1);
  int64_t paddedColumns = llvm::alignTo(columns, elementsPer32Bytes);
  if (paddedColumns == columns)
    paddedColumns += elementsPer32Bytes;
  MLIRContext *context = type.getContext();
  AffineExpr linear = getAffineDimExpr(0, context) * paddedColumns +
                      getAffineDimExpr(1, context);
  return buildAffineStorage(type, linear,
                            type.getDimSize(0) * paddedColumns);
}

DenseIntElementsAttr getDenseMatrix(MLIRContext *context, unsigned rows,
                                    unsigned columns,
                                    ArrayRef<uint64_t> basisOutputs) {
  SmallVector<APInt> values;
  values.reserve(rows * columns);
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned column = 0; column < columns; ++column)
      values.emplace_back(1, (basisOutputs[column] >> row) & 1);
  auto matrixType = RankedTensorType::get(
      {static_cast<int64_t>(rows), static_cast<int64_t>(columns)},
      IntegerType::get(context, 1));
  return DenseIntElementsAttr::get(matrixType, values);
}

StorageLayoutAttr buildXor(MemRefType type, unsigned swizzleBytes) {
  if (type.getRank() != 2 || !llvm::isPowerOf2_64(type.getDimSize(0)) ||
      !llvm::isPowerOf2_64(type.getDimSize(1)))
    return {};
  unsigned elementBits = type.getElementTypeBitWidth();
  if (elementBits < 8 || elementBits % 8 != 0 ||
      !llvm::isPowerOf2_64(elementBits / 8))
    return {};

  unsigned rowBits = llvm::Log2_64(type.getDimSize(0));
  unsigned columnBits = llvm::Log2_64(type.getDimSize(1));
  unsigned elementByteBits = llvm::Log2_64(elementBits / 8);
  unsigned byteBits = rowBits + columnBits + elementByteBits;
  unsigned targetByteBit = llvm::Log2_64(swizzleBytes) - 1;
  if (rowBits == 0 || targetByteBit >= byteBits ||
      targetByteBit >= columnBits + elementByteBits)
    return {};

  SmallVector<uint64_t> basisOutputs(rowBits + columnBits);
  for (unsigned bit = 0; bit < rowBits; ++bit)
    basisOutputs[bit] = uint64_t{1}
                        << (columnBits + elementByteBits + bit);
  for (unsigned bit = 0; bit < columnBits; ++bit)
    basisOutputs[rowBits + bit] = uint64_t{1}
                                  << (elementByteBits + bit);
  basisOutputs[0] ^= uint64_t{1} << targetByteBit;

  MLIRContext *context = type.getContext();
  Builder builder(context);
  auto bitLinear = BitLinearLayoutMapAttr::get(
      context, getDimensionNames(builder, type.getRank()),
      builder.getDenseI64ArrayAttr(
          {static_cast<int64_t>(rowBits), static_cast<int64_t>(columnBits)}),
      builder.getArrayAttr({builder.getStringAttr("byte_offset"),
                            builder.getStringAttr("bit_offset")}),
      builder.getDenseI64ArrayAttr(
          {static_cast<int64_t>(byteBits), int64_t{3}}),
      getDenseMatrix(context, byteBits + 3, rowBits + columnBits,
                     basisOutputs));
  std::optional<attr::MemorySpace> space = getMemorySpace(type);
  if (!space)
    return {};
  int64_t alignment = std::max<unsigned>(1, elementBits / 8);
  return StorageLayoutAttr::get(
      context, bitLinear, MemorySpaceAttr::get(context, *space),
      builder.getI64IntegerAttr(alignment),
      builder.getI64IntegerAttr(alignment));
}

class SM90LayoutTarget final : public LayoutTarget {
public:
  void enumerateCandidates(
      const LayoutVar &var,
      SmallVectorImpl<LayoutCandidate> &out) const override {
    auto type = dyn_cast<MemRefType>(var.shapedType);
    if (var.kind != LayoutKind::Storage || !type || !type.hasStaticShape())
      return;
    auto append = [&](Attribute candidate, uint64_t ordinal) {
      if (candidate)
        out.push_back({candidate, 0, ordinal});
    };
    append(buildLinear(type), 0);
    append(buildTranspose(type), 1);
    append(buildPadded(type), 2);
    if (type.getRank() == 2 && type.getElementTypeBitWidth() % 8 == 0) {
      uint64_t rowBytes = type.getDimSize(1) *
                          (type.getElementTypeBitWidth() / 8);
      unsigned swizzleBytes = rowBytes <= 32 ? 32 : rowBytes <= 64 ? 64 : 128;
      uint64_t ordinal = swizzleBytes == 32 ? 3 : swizzleBytes == 64 ? 4 : 5;
      append(buildXor(type, swizzleBytes), ordinal);
    }
  }

  LogicalResult verifyCandidate(const LayoutVar &var, Attribute candidate,
                                Location loc) const override {
    auto type = dyn_cast<MemRefType>(var.shapedType);
    auto storage = dyn_cast<StorageLayoutAttr>(candidate);
    if (var.kind != LayoutKind::Storage || !type || !storage)
      return emitError(loc) << "SM90 storage candidate has incompatible kind";
    return storage.verifyForType(type, loc);
  }

  FailureOr<CostVector>
  evaluate(const CandidateAssignment &assignment) const override {
    CostVector cost;
    for (const auto &entry : assignment.values)
      cost.deterministicTieBreak += entry.first;
    return cost;
  }
};

} // namespace

std::unique_ptr<LayoutTarget> createSM90LayoutTarget() {
  return std::make_unique<SM90LayoutTarget>();
}

} // namespace mlir::frisk
