#ifndef FRISK_ANALYSIS_LAYOUTALGEBRA_H
#define FRISK_ANALYSIS_LAYOUTALGEBRA_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/Frisk/Analysis/LayoutCommon.h"

namespace mlir::frisk {

/// A dense matrix over GF(2). Rows and columns are numbered from the least
/// significant bit so all elimination and serialization orders are stable.
class GF2Matrix {
public:
  static FailureOr<GF2Matrix> get(unsigned rows, unsigned columns,
                                  ArrayRef<llvm::APInt> rowBits);

  unsigned getNumRows() const { return numRows; }
  unsigned getNumColumns() const { return numColumns; }

  llvm::APInt apply(const llvm::APInt &input) const;
  GF2Matrix transpose() const;

  /// Returns this(rhs(x)).
  FailureOr<GF2Matrix> compose(const GF2Matrix &rhs) const;

  unsigned rank() const;
  SmallVector<llvm::APInt> kernelBasis() const;
  FailureOr<GF2Matrix> inverse() const;
  FailureOr<GF2Matrix> rightInverse() const;

  bool operator==(const GF2Matrix &rhs) const;
  bool operator!=(const GF2Matrix &rhs) const { return !(*this == rhs); }

private:
  GF2Matrix(unsigned rows, unsigned columns,
            ArrayRef<llvm::APInt> rowBits);

  unsigned numRows;
  unsigned numColumns;
  SmallVector<llvm::APInt> rows;
};

FailureOr<Attribute> composeLayoutMaps(Attribute lhs, Attribute rhs);
FailureOr<Attribute> projectLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
FailureOr<Attribute> permuteLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
LayoutProof checkCoverage(Attribute map, ArrayRef<int64_t> logicalShape);
LayoutProof checkInjectivity(Attribute map, ArrayRef<int64_t> domainShape);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTALGEBRA_H
