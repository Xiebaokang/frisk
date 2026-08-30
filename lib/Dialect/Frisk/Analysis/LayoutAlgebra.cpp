#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"

#include <cassert>
#include <utility>

namespace mlir::frisk {
namespace {

struct RrefResult {
  SmallVector<llvm::APInt> rows;
  SmallVector<llvm::APInt> transform;
  SmallVector<unsigned> pivotColumns;
};

RrefResult computeRref(ArrayRef<llvm::APInt> inputRows, unsigned numRows,
                       unsigned numColumns, bool trackTransform) {
  RrefResult result;
  result.rows.assign(inputRows.begin(), inputRows.end());
  if (trackTransform) {
    result.transform.reserve(numRows);
    for (unsigned row = 0; row < numRows; ++row) {
      result.transform.emplace_back(numRows, 0);
      result.transform.back().setBit(row);
    }
  }

  unsigned pivotRow = 0;
  for (unsigned column = 0; column < numColumns && pivotRow < numRows;
       ++column) {
    unsigned selectedRow = pivotRow;
    while (selectedRow < numRows && !result.rows[selectedRow][column])
      ++selectedRow;
    if (selectedRow == numRows)
      continue;

    if (selectedRow != pivotRow) {
      std::swap(result.rows[selectedRow], result.rows[pivotRow]);
      if (trackTransform)
        std::swap(result.transform[selectedRow],
                  result.transform[pivotRow]);
    }

    for (unsigned row = 0; row < numRows; ++row) {
      if (row == pivotRow || !result.rows[row][column])
        continue;
      result.rows[row] ^= result.rows[pivotRow];
      if (trackTransform)
        result.transform[row] ^= result.transform[pivotRow];
    }

    result.pivotColumns.push_back(column);
    ++pivotRow;
  }
  return result;
}

} // namespace

GF2Matrix::GF2Matrix(unsigned rows, unsigned columns,
                     ArrayRef<llvm::APInt> rowBits)
    : numRows(rows), numColumns(columns), rows(rowBits.begin(), rowBits.end()) {
}

FailureOr<GF2Matrix> GF2Matrix::get(unsigned rows, unsigned columns,
                                     ArrayRef<llvm::APInt> rowBits) {
  if (rows == 0 || columns == 0 || rowBits.size() != rows)
    return failure();
  for (const llvm::APInt &row : rowBits) {
    if (row.getBitWidth() != columns)
      return failure();
  }
  return GF2Matrix(rows, columns, rowBits);
}

llvm::APInt GF2Matrix::apply(const llvm::APInt &input) const {
  assert(input.getBitWidth() == numColumns &&
         "GF2Matrix input width must equal its column count");
  llvm::APInt output(numRows, 0);
  for (unsigned row = 0; row < numRows; ++row) {
    if ((rows[row] & input).popcount() & 1)
      output.setBit(row);
  }
  return output;
}

GF2Matrix GF2Matrix::transpose() const {
  SmallVector<llvm::APInt> transposedRows;
  transposedRows.reserve(numColumns);
  for (unsigned column = 0; column < numColumns; ++column) {
    transposedRows.emplace_back(numRows, 0);
    for (unsigned row = 0; row < numRows; ++row) {
      if (rows[row][column])
        transposedRows.back().setBit(row);
    }
  }
  return GF2Matrix(numColumns, numRows, transposedRows);
}

FailureOr<GF2Matrix> GF2Matrix::compose(const GF2Matrix &rhs) const {
  if (numColumns != rhs.numRows)
    return failure();

  SmallVector<llvm::APInt> composedRows;
  composedRows.reserve(numRows);
  for (const llvm::APInt &lhsRow : rows) {
    llvm::APInt row(rhs.numColumns, 0);
    for (unsigned bit = 0; bit < numColumns; ++bit) {
      if (lhsRow[bit])
        row ^= rhs.rows[bit];
    }
    composedRows.push_back(std::move(row));
  }
  return GF2Matrix(numRows, rhs.numColumns, composedRows);
}

unsigned GF2Matrix::rank() const {
  return computeRref(rows, numRows, numColumns, false).pivotColumns.size();
}

SmallVector<llvm::APInt> GF2Matrix::kernelBasis() const {
  RrefResult rref = computeRref(rows, numRows, numColumns, false);
  SmallVector<bool> isPivot(numColumns, false);
  for (unsigned column : rref.pivotColumns)
    isPivot[column] = true;

  SmallVector<llvm::APInt> basis;
  basis.reserve(numColumns - rref.pivotColumns.size());
  for (unsigned freeColumn = 0; freeColumn < numColumns; ++freeColumn) {
    if (isPivot[freeColumn])
      continue;
    llvm::APInt vector(numColumns, 0);
    vector.setBit(freeColumn);
    for (unsigned pivotRow = 0; pivotRow < rref.pivotColumns.size();
         ++pivotRow) {
      if (rref.rows[pivotRow][freeColumn])
        vector.setBit(rref.pivotColumns[pivotRow]);
    }
    basis.push_back(std::move(vector));
  }
  return basis;
}

FailureOr<GF2Matrix> GF2Matrix::inverse() const {
  if (numRows != numColumns)
    return failure();
  RrefResult rref = computeRref(rows, numRows, numColumns, true);
  if (rref.pivotColumns.size() != numColumns)
    return failure();
  return GF2Matrix(numRows, numColumns, rref.transform);
}

FailureOr<GF2Matrix> GF2Matrix::rightInverse() const {
  RrefResult rref = computeRref(rows, numRows, numColumns, true);
  if (rref.pivotColumns.size() != numRows)
    return failure();

  SmallVector<llvm::APInt> inverseRows;
  inverseRows.reserve(numColumns);
  for (unsigned row = 0; row < numColumns; ++row)
    inverseRows.emplace_back(numRows, 0);
  for (unsigned pivotRow = 0; pivotRow < numRows; ++pivotRow)
    inverseRows[rref.pivotColumns[pivotRow]] = rref.transform[pivotRow];
  return GF2Matrix(numColumns, numRows, inverseRows);
}

bool GF2Matrix::operator==(const GF2Matrix &rhs) const {
  return numRows == rhs.numRows && numColumns == rhs.numColumns &&
         rows == rhs.rows;
}

} // namespace mlir::frisk
