#include "Dialect/Frisk/IR/FriskAttributes.h"

#include <cstdint>
#include <limits>

#include "llvm/ADT/StringSet.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::frisk {
namespace {

ParseResult parseI64List(AsmParser &parser, DenseI64ArrayAttr &result) {
  SmallVector<int64_t> values;
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        int64_t value;
        if (parser.parseInteger(value))
          return failure();
        values.push_back(value);
        return success();
      }))
    return failure();
  result = parser.getBuilder().getDenseI64ArrayAttr(values);
  return success();
}

void printI64List(AsmPrinter &printer, DenseI64ArrayAttr values) {
  printer << '[';
  llvm::interleaveComma(values.asArrayRef(), printer,
                        [&](int64_t value) { printer << value; });
  printer << ']';
}

LogicalResult verifyNames(function_ref<InFlightDiagnostic()> emitError,
                          StringRef kind, ArrayAttr names) {
  llvm::StringSet<> seen;
  for (Attribute value : names) {
    auto name = dyn_cast<StringAttr>(value);
    if (!name)
      return emitError() << kind << " names must contain only StringAttr values";
    if (name.getValue().empty())
      return emitError() << kind << " dimension names must not be empty";
    if (!seen.insert(name.getValue()).second)
      return emitError() << "duplicate " << kind << " dimension name '"
                         << name.getValue() << "'";
  }
  return success();
}

FailureOr<unsigned>
sumBitWidths(function_ref<InFlightDiagnostic()> emitError, StringRef kind,
             DenseI64ArrayAttr widths) {
  uint64_t total = 0;
  for (int64_t width : widths.asArrayRef()) {
    if (width <= 0) {
      emitError() << kind << " bit widths must be positive";
      return failure();
    }
    total += static_cast<uint64_t>(width);
    if (total > std::numeric_limits<unsigned>::max()) {
      emitError() << kind << " total bit width is too large";
      return failure();
    }
  }
  return static_cast<unsigned>(total);
}

DenseIntElementsAttr getDenseMatrix(MLIRContext *context,
                                    const GF2Matrix &matrix) {
  SmallVector<llvm::APInt> values;
  values.reserve(matrix.getNumRows() * matrix.getNumColumns());
  for (unsigned column = 0; column < matrix.getNumColumns(); ++column) {
    llvm::APInt basis(matrix.getNumColumns(), 0);
    basis.setBit(column);
    llvm::APInt output = matrix.apply(basis);
    for (unsigned row = 0; row < matrix.getNumRows(); ++row)
      values.emplace_back(1, output[row]);
  }

  // DenseElements uses row-major order, whereas the loop above visits columns.
  SmallVector<llvm::APInt> rowMajor(values.size(), llvm::APInt(1, 0));
  for (unsigned column = 0; column < matrix.getNumColumns(); ++column) {
    for (unsigned row = 0; row < matrix.getNumRows(); ++row)
      rowMajor[row * matrix.getNumColumns() + column] =
          values[column * matrix.getNumRows() + row];
  }
  auto type = RankedTensorType::get(
      {static_cast<int64_t>(matrix.getNumRows()),
       static_cast<int64_t>(matrix.getNumColumns())},
      IntegerType::get(context, 1));
  return DenseIntElementsAttr::get(type, rowMajor);
}

SmallVector<int64_t> decodeNamedInput(llvm::APInt bits,
                                      DenseI64ArrayAttr widths) {
  SmallVector<int64_t> coordinates;
  coordinates.reserve(widths.size());
  unsigned offset = 0;
  for (int64_t widthValue : widths.asArrayRef()) {
    unsigned width = static_cast<unsigned>(widthValue);
    llvm::APInt coordinate = bits.extractBits(width, offset);
    coordinates.push_back(static_cast<int64_t>(
        coordinate.getLimitedValue(std::numeric_limits<int64_t>::max())));
    offset += width;
  }
  return coordinates;
}

FailureOr<GF2Matrix> getIntermediateAlignment(BitLinearLayoutMapAttr lhs,
                                               BitLinearLayoutMapAttr rhs) {
  ArrayAttr lhsNames = lhs.getInputNames();
  ArrayAttr rhsNames = rhs.getOutputNames();
  auto lhsWidths = lhs.getInputBitWidths().asArrayRef();
  auto rhsWidths = rhs.getOutputBitWidths().asArrayRef();
  if (lhsNames.size() != rhsNames.size())
    return failure();

  unsigned lhsTotal = 0;
  for (int64_t width : lhsWidths)
    lhsTotal += static_cast<unsigned>(width);
  unsigned rhsTotal = 0;
  for (int64_t width : rhsWidths)
    rhsTotal += static_cast<unsigned>(width);
  if (lhsTotal != rhsTotal)
    return failure();

  SmallVector<unsigned> rhsOffsets(rhsNames.size());
  for (unsigned index = 1; index < rhsNames.size(); ++index)
    rhsOffsets[index] =
        rhsOffsets[index - 1] + static_cast<unsigned>(rhsWidths[index - 1]);

  SmallVector<llvm::APInt> rows;
  rows.reserve(lhsTotal);
  for (unsigned lhsIndex = 0; lhsIndex < lhsNames.size(); ++lhsIndex) {
    auto lhsName = cast<StringAttr>(lhsNames[lhsIndex]);
    unsigned rhsIndex = 0;
    while (rhsIndex < rhsNames.size() &&
           cast<StringAttr>(rhsNames[rhsIndex]).getValue() !=
               lhsName.getValue())
      ++rhsIndex;
    if (rhsIndex == rhsNames.size() ||
        rhsWidths[rhsIndex] != lhsWidths[lhsIndex])
      return failure();
    for (unsigned bit = 0; bit < static_cast<unsigned>(lhsWidths[lhsIndex]);
         ++bit) {
      rows.emplace_back(rhsTotal, 0);
      rows.back().setBit(rhsOffsets[rhsIndex] + bit);
    }
  }
  return GF2Matrix::get(lhsTotal, rhsTotal, rows);
}

} // namespace

Attribute BitLinearLayoutMapAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc location = parser.getCurrentLocation();
  ArrayAttr inputNames;
  DenseI64ArrayAttr inputBitWidths;
  ArrayAttr outputNames;
  DenseI64ArrayAttr outputBitWidths;
  DenseIntElementsAttr matrix;

  if (parser.parseLess() || parser.parseKeyword("inputs") ||
      parser.parseEqual() || parser.parseAttribute(inputNames) ||
      parser.parseComma() || parser.parseKeyword("input_bits") ||
      parser.parseEqual() || parseI64List(parser, inputBitWidths) ||
      parser.parseComma() || parser.parseKeyword("outputs") ||
      parser.parseEqual() || parser.parseAttribute(outputNames) ||
      parser.parseComma() || parser.parseKeyword("output_bits") ||
      parser.parseEqual() || parseI64List(parser, outputBitWidths) ||
      parser.parseComma() || parser.parseKeyword("matrix") ||
      parser.parseEqual() || parser.parseAttribute(matrix) ||
      parser.parseGreater())
    return {};

  return parser.getChecked<BitLinearLayoutMapAttr>(
      location, parser.getContext(), inputNames, inputBitWidths, outputNames,
      outputBitWidths, matrix);
}

void BitLinearLayoutMapAttr::print(AsmPrinter &printer) const {
  printer << "<inputs = " << getInputNames() << ", input_bits = ";
  printI64List(printer, getInputBitWidths());
  printer << ", outputs = " << getOutputNames() << ", output_bits = ";
  printI64List(printer, getOutputBitWidths());
  printer << ", matrix = " << getMatrix() << '>';
}

LogicalResult BitLinearLayoutMapAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr inputNames,
    DenseI64ArrayAttr inputBitWidths, ArrayAttr outputNames,
    DenseI64ArrayAttr outputBitWidths, DenseIntElementsAttr matrix) {
  if (inputNames.size() != inputBitWidths.size())
    return emitError() << "input name and bit-width counts must match";
  if (outputNames.size() != outputBitWidths.size())
    return emitError() << "output name and bit-width counts must match";
  if (failed(verifyNames(emitError, "input", inputNames)) ||
      failed(verifyNames(emitError, "output", outputNames)))
    return failure();

  FailureOr<unsigned> inputBits =
      sumBitWidths(emitError, "input", inputBitWidths);
  FailureOr<unsigned> outputBits =
      sumBitWidths(emitError, "output", outputBitWidths);
  if (failed(inputBits) || failed(outputBits))
    return failure();

  auto matrixType = dyn_cast<RankedTensorType>(matrix.getType());
  if (!matrixType || matrixType.getRank() != 2 ||
      !matrixType.getElementType().isInteger(1))
    return emitError() << "matrix must have type tensor<rows x columns x i1>";
  if (matrixType.getDimSize(0) != *outputBits ||
      matrixType.getDimSize(1) != *inputBits)
    return emitError() << "matrix shape must be [" << *outputBits << ", "
                       << *inputBits << "] but got ["
                       << matrixType.getDimSize(0) << ", "
                       << matrixType.getDimSize(1) << "]";
  return success();
}

FailureOr<GF2Matrix> BitLinearLayoutMapAttr::getMatrixValue() const {
  auto matrixType = cast<RankedTensorType>(getMatrix().getType());
  unsigned rowCount = matrixType.getDimSize(0);
  unsigned columnCount = matrixType.getDimSize(1);
  SmallVector<llvm::APInt> rows(rowCount, llvm::APInt(columnCount, 0));
  unsigned index = 0;
  for (const llvm::APInt &value : getMatrix().getValues<llvm::APInt>()) {
    if (!value.isZero())
      rows[index / columnCount].setBit(index % columnCount);
    ++index;
  }
  return GF2Matrix::get(rowCount, columnCount, rows);
}

FailureOr<Attribute> BitLinearLayoutMapAttr::canonicalizeMap() const {
  FailureOr<GF2Matrix> matrix = getMatrixValue();
  if (failed(matrix))
    return failure();
  return Attribute(BitLinearLayoutMapAttr::get(
      getContext(), getInputNames(), getInputBitWidths(), getOutputNames(),
      getOutputBitWidths(), getDenseMatrix(getContext(), *matrix)));
}

LogicalResult BitLinearLayoutMapAttr::verifyMap(Location loc) const {
  return verify([&]() { return emitError(loc); }, getInputNames(),
                getInputBitWidths(), getOutputNames(), getOutputBitWidths(),
                getMatrix());
}

LayoutProof checkInjective(BitLinearLayoutMapAttr map) {
  FailureOr<GF2Matrix> matrix = map.getMatrixValue();
  if (failed(matrix))
    return {ProofStatus::Unknown, {}, "invalid bit-linear matrix"};
  if (matrix->rank() == matrix->getNumColumns())
    return {ProofStatus::Proven, {}, "GF(2) matrix has full column rank"};

  SmallVector<llvm::APInt> kernel = matrix->kernelBasis();
  return {ProofStatus::Disproven,
          decodeNamedInput(kernel.front(), map.getInputBitWidths()),
          "GF(2) matrix has a non-zero kernel (replication)"};
}

LayoutProof checkSurjective(BitLinearLayoutMapAttr map) {
  FailureOr<GF2Matrix> matrix = map.getMatrixValue();
  if (failed(matrix))
    return {ProofStatus::Unknown, {}, "invalid bit-linear matrix"};
  if (matrix->rank() == matrix->getNumRows())
    return {ProofStatus::Proven, {}, "GF(2) matrix has full row rank"};
  return {ProofStatus::Disproven, {},
          "GF(2) matrix image does not cover the output bit space"};
}

FailureOr<BitLinearLayoutMapAttr>
composeBitLinear(BitLinearLayoutMapAttr lhs, BitLinearLayoutMapAttr rhs) {
  if (lhs.getContext() != rhs.getContext())
    return failure();
  FailureOr<GF2Matrix> lhsMatrix = lhs.getMatrixValue();
  FailureOr<GF2Matrix> rhsMatrix = rhs.getMatrixValue();
  FailureOr<GF2Matrix> alignment = getIntermediateAlignment(lhs, rhs);
  if (failed(lhsMatrix) || failed(rhsMatrix) || failed(alignment))
    return failure();

  FailureOr<GF2Matrix> alignedLhs = lhsMatrix->compose(*alignment);
  if (failed(alignedLhs))
    return failure();
  FailureOr<GF2Matrix> composed = alignedLhs->compose(*rhsMatrix);
  if (failed(composed))
    return failure();

  return BitLinearLayoutMapAttr::get(
      lhs.getContext(), rhs.getInputNames(), rhs.getInputBitWidths(),
      lhs.getOutputNames(), lhs.getOutputBitWidths(),
      getDenseMatrix(lhs.getContext(), *composed));
}

} // namespace mlir::frisk
