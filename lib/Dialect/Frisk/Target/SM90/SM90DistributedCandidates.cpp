#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::frisk {
void enumerateSM90DistributedCandidates(
    const LayoutVar &var, SmallVectorImpl<LayoutCandidate> &out) {
  auto type = dyn_cast<RankedTensorType>(var.shapedType);
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return;
  unsigned logicalBits = 0;
  SmallVector<int64_t> widths;
  Builder builder(type.getContext());
  SmallVector<Attribute> outputs;
  for (auto [index, extent] : llvm::enumerate(type.getShape())) {
    if (extent <= 1 || !llvm::isPowerOf2_64(extent)) return;
    unsigned bits = llvm::Log2_64(extent);
    logicalBits += bits;
    widths.push_back(bits);
    outputs.push_back(builder.getStringAttr("dim" + Twine(index)));
  }
  if (logicalBits >= 58) return;
  for (unsigned family = 0; family < 4; ++family) {
    unsigned laneBits = 5, warpBits = family == 2 ? 2 : 0;
    unsigned regBits = family == 3 ? logicalBits :
        (logicalBits > laneBits + warpBits ? logicalBits - laneBits - warpBits : 0);
    unsigned columns = regBits + laneBits + warpBits;
    SmallVector<unsigned> allocation;
    auto append = [&](unsigned start, unsigned count) {
      for (unsigned i = 0; i < count; ++i) allocation.push_back(start + i);
    };
    if (family == 0 || family == 3) {
      append(0, regBits); append(regBits, laneBits); append(regBits + laneBits, warpBits);
    } else if (family == 1) {
      append(regBits, laneBits); append(0, regBits); append(regBits + laneBits, warpBits);
    } else {
      append(regBits + laneBits, warpBits); append(regBits, laneBits); append(0, regBits);
    }
    SmallVector<APInt> matrix(logicalBits * columns, APInt(1, 0));
    // Flat low bits belong to the last logical dimension; matrix row blocks
    // remain in named dim0, dim1, ... order.
    unsigned flatBit = 0, rowEnd = logicalBits;
    for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
      rowEnd -= widths[dim];
      for (unsigned bit = 0; bit < widths[dim]; ++bit)
        matrix[(rowEnd + bit) * columns + allocation[flatBit++]] = APInt(1, 1);
    }
    SmallVector<Attribute> inputs;
    SmallVector<int64_t> inputWidths;
    for (auto [name, bits] : {std::pair<StringRef, unsigned>{"register", regBits},
                             {"lane", laneBits}, {"warp", warpBits}})
      if (bits) {
        inputs.push_back(builder.getStringAttr(name));
        inputWidths.push_back(bits);
      }
    auto map = BitLinearLayoutMapAttr::get(type.getContext(),
        builder.getArrayAttr(inputs), builder.getDenseI64ArrayAttr(inputWidths),
        builder.getArrayAttr(outputs), builder.getDenseI64ArrayAttr(widths),
        DenseIntElementsAttr::get(RankedTensorType::get(
            {logicalBits, columns}, builder.getI1Type()), matrix));
    auto encoding = DistributedEncodingAttr::get(type.getContext(), map,
        builder.getDenseI64ArrayAttr({int64_t{1} << regBits, 32,
                                     int64_t{1} << warpBits, 1, 1}),
        builder.getI64IntegerAttr(int64_t{1} << (columns - logicalBits)));
    if (llvm::none_of(out, [&](const LayoutCandidate &c) { return c.value == encoding; }))
      out.push_back({encoding, kInvalidProvenanceID, family});
  }
}
} // namespace mlir::frisk
