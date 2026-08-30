#include "Dialect/Frisk/IR/FriskAttributes.h"

#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"

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

LogicalResult verifyExtents(function_ref<InFlightDiagnostic()> emitError,
                            StringRef kind, DenseI64ArrayAttr extents) {
  for (int64_t extent : extents.asArrayRef()) {
    if (extent <= 0 && extent != ShapedType::kDynamic)
      return emitError() << kind
                         << " extents must be positive or ShapedType::kDynamic";
  }
  return success();
}

std::optional<unsigned> findName(ArrayAttr names, StringRef expected) {
  for (auto [index, value] : llvm::enumerate(names)) {
    if (cast<StringAttr>(value).getValue() == expected)
      return index;
  }
  return std::nullopt;
}

FailureOr<SmallVector<unsigned>>
getOutputOrder(ArrayAttr available, ArrayRef<StringRef> requested) {
  llvm::StringSet<> seen;
  SmallVector<unsigned> order;
  order.reserve(requested.size());
  for (StringRef name : requested) {
    if (!seen.insert(name).second)
      return failure();
    std::optional<unsigned> index = findName(available, name);
    if (!index)
      return failure();
    order.push_back(*index);
  }
  return order;
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

Attribute AffineLayoutMapAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc location = parser.getCurrentLocation();
  ArrayAttr inputNames;
  DenseI64ArrayAttr inputExtents;
  ArrayAttr outputNames;
  DenseI64ArrayAttr outputExtents;
  AffineMapAttr affineMap;

  if (parser.parseLess() || parser.parseKeyword("inputs") ||
      parser.parseEqual() || parser.parseAttribute(inputNames) ||
      parser.parseComma() || parser.parseKeyword("input_extents") ||
      parser.parseEqual() || parseI64List(parser, inputExtents) ||
      parser.parseComma() || parser.parseKeyword("outputs") ||
      parser.parseEqual() || parser.parseAttribute(outputNames) ||
      parser.parseComma() || parser.parseKeyword("output_extents") ||
      parser.parseEqual() || parseI64List(parser, outputExtents) ||
      parser.parseComma() || parser.parseKeyword("map") ||
      parser.parseEqual() || parser.parseAttribute(affineMap) ||
      parser.parseGreater())
    return {};

  return parser.getChecked<AffineLayoutMapAttr>(
      location, parser.getContext(), inputNames, inputExtents, outputNames,
      outputExtents, affineMap);
}

void AffineLayoutMapAttr::print(AsmPrinter &printer) const {
  printer << "<inputs = " << getInputNames() << ", input_extents = ";
  printI64List(printer, getInputExtents());
  printer << ", outputs = " << getOutputNames() << ", output_extents = ";
  printI64List(printer, getOutputExtents());
  printer << ", map = " << getAffineMap() << '>';
}

LogicalResult AffineLayoutMapAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr inputNames,
    DenseI64ArrayAttr inputExtents, ArrayAttr outputNames,
    DenseI64ArrayAttr outputExtents, AffineMapAttr affineMapAttr) {
  if (inputNames.size() != inputExtents.size())
    return emitError() << "input name and extent counts must match";
  if (outputNames.size() != outputExtents.size())
    return emitError() << "output name and extent counts must match";
  if (failed(verifyNames(emitError, "input", inputNames)) ||
      failed(verifyNames(emitError, "output", outputNames)) ||
      failed(verifyExtents(emitError, "input", inputExtents)) ||
      failed(verifyExtents(emitError, "output", outputExtents)))
    return failure();

  AffineMap map = affineMapAttr.getValue();
  if (map.getNumDims() != inputNames.size())
    return emitError() << "affine map dimension count must match input names";
  if (map.getNumResults() != outputNames.size())
    return emitError() << "affine map result count must match output names";
  unsigned dynamicInputs = llvm::count(inputExtents.asArrayRef(),
                                       ShapedType::kDynamic);
  if (map.getNumSymbols() != dynamicInputs)
    return emitError() << "affine map symbol count must equal the number of "
                          "dynamic input extents";
  return success();
}

FailureOr<Attribute> AffineLayoutMapAttr::canonicalizeMap() const {
  return Attribute(*this);
}

LogicalResult AffineLayoutMapAttr::verifyMap(Location loc) const {
  return verify([&]() { return emitError(loc); }, getInputNames(),
                getInputExtents(), getOutputNames(), getOutputExtents(),
                getAffineMap());
}

Attribute ProductLayoutMapAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc location = parser.getCurrentLocation();
  Attribute outer;
  Attribute inner;
  DenseI64ArrayAttr splitExtents;
  if (parser.parseLess() || parser.parseKeyword("outer") ||
      parser.parseEqual() || parser.parseAttribute(outer) ||
      parser.parseComma() || parser.parseKeyword("inner") ||
      parser.parseEqual() || parser.parseAttribute(inner) ||
      parser.parseComma() || parser.parseKeyword("split_extents") ||
      parser.parseEqual() || parseI64List(parser, splitExtents) ||
      parser.parseGreater())
    return {};
  return parser.getChecked<ProductLayoutMapAttr>(
      location, parser.getContext(), outer, inner, splitExtents);
}

void ProductLayoutMapAttr::print(AsmPrinter &printer) const {
  printer << "<outer = " << getOuter() << ", inner = " << getInner()
          << ", split_extents = ";
  printI64List(printer, getSplitExtents());
  printer << '>';
}

LogicalResult ProductLayoutMapAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute outerAttr,
    Attribute innerAttr, DenseI64ArrayAttr splitExtents) {
  auto outer = dyn_cast<AffineLayoutMapAttr>(outerAttr);
  auto inner = dyn_cast<BitLinearLayoutMapAttr>(innerAttr);
  if (!outer || !inner)
    return emitError()
           << "product layout requires an affine outer and bit-linear inner";
  if (splitExtents.size() != outer.getOutputNames().size())
    return emitError() << "split extent count must match outer outputs";
  if (inner.getOutputNames().size() != outer.getOutputNames().size())
    return emitError() << "outer and inner output counts must match";

  auto innerWidths = inner.getOutputBitWidths().asArrayRef();
  for (auto [outerIndex, split] : llvm::enumerate(splitExtents.asArrayRef())) {
    if (split <= 0 || !llvm::isPowerOf2_64(static_cast<uint64_t>(split)))
      return emitError() << "split extents must be positive powers of two";
    StringRef name =
        cast<StringAttr>(outer.getOutputNames()[outerIndex]).getValue();
    std::optional<unsigned> innerIndex = findName(inner.getOutputNames(), name);
    if (!innerIndex)
      return emitError() << "inner layout is missing output dimension '" << name
                         << "'";
    if (innerWidths[*innerIndex] != llvm::Log2_64(split))
      return emitError() << "inner bit width for output '" << name
                         << "' must equal log2(split extent)";
    AffineExpr result = outer.getAffineMap().getValue().getResult(outerIndex);
    if (!result.isMultipleOf(split)) {
      std::string expression;
      llvm::raw_string_ostream stream(expression);
      result.print(stream);
      stream.flush();
      return emitError() << "cannot prove outer result '" << expression
                         << "' is aligned to split extent " << split;
    }
  }
  return success();
}

FailureOr<Attribute> ProductLayoutMapAttr::canonicalizeMap() const {
  auto outer = dyn_cast<LayoutMapAttrInterface>(getOuter());
  auto inner = dyn_cast<LayoutMapAttrInterface>(getInner());
  if (!outer || !inner)
    return failure();
  FailureOr<Attribute> canonicalOuter = outer.canonicalizeMap();
  FailureOr<Attribute> canonicalInner = inner.canonicalizeMap();
  if (failed(canonicalOuter) || failed(canonicalInner))
    return failure();
  return Attribute(ProductLayoutMapAttr::get(
      getContext(), *canonicalOuter, *canonicalInner, getSplitExtents()));
}

LogicalResult ProductLayoutMapAttr::verifyMap(Location loc) const {
  return verify([&]() { return emitError(loc); }, getOuter(), getInner(),
                getSplitExtents());
}

namespace {

// M1 keeps this bounded evaluator as a conservative verifier/reference oracle.
// Exact GF(2) proofs bypass it below. Symbolic or larger affine/product maps
// return Unknown and must be handled by a later Presburger proof, never guessed.
constexpr uint64_t kEnumerationLimit = 65536;

FailureOr<SmallVector<int64_t>>
evaluateAffine(AffineLayoutMapAttr map, ArrayRef<int64_t> coordinates) {
  AffineMap affineMap = map.getAffineMap().getValue();
  if (coordinates.size() != affineMap.getNumDims() ||
      affineMap.getNumSymbols() != 0)
    return failure();

  Builder builder(map.getContext());
  SmallVector<Attribute> operands;
  operands.reserve(coordinates.size());
  for (int64_t coordinate : coordinates)
    operands.push_back(builder.getIndexAttr(coordinate));
  SmallVector<Attribute> folded;
  if (failed(affineMap.constantFold(operands, folded)))
    return failure();

  SmallVector<int64_t> result;
  result.reserve(folded.size());
  for (Attribute value : folded)
    result.push_back(cast<IntegerAttr>(value).getInt());
  return result;
}

FailureOr<SmallVector<int64_t>>
evaluateBitLinear(BitLinearLayoutMapAttr map,
                  ArrayRef<int64_t> coordinates) {
  if (coordinates.size() != map.getInputNames().size())
    return failure();
  unsigned totalInputBits = 0;
  for (int64_t width : map.getInputBitWidths().asArrayRef())
    totalInputBits += static_cast<unsigned>(width);
  llvm::APInt input(totalInputBits, 0);
  unsigned offset = 0;
  for (auto [coordinate, widthValue] :
       llvm::zip_equal(coordinates, map.getInputBitWidths().asArrayRef())) {
    if (coordinate < 0)
      return failure();
    unsigned width = static_cast<unsigned>(widthValue);
    llvm::APInt value(width, static_cast<uint64_t>(coordinate));
    if (value.getLimitedValue() != static_cast<uint64_t>(coordinate))
      return failure();
    input.insertBits(value, offset);
    offset += width;
  }

  FailureOr<GF2Matrix> matrix = map.getMatrixValue();
  if (failed(matrix))
    return failure();
  llvm::APInt output = matrix->apply(input);
  SmallVector<int64_t> result;
  result.reserve(map.getOutputNames().size());
  offset = 0;
  for (int64_t widthValue : map.getOutputBitWidths().asArrayRef()) {
    unsigned width = static_cast<unsigned>(widthValue);
    llvm::APInt value = output.extractBits(width, offset);
    if (width > 63 && value.getActiveBits() > 63)
      return failure();
    result.push_back(static_cast<int64_t>(value.getZExtValue()));
    offset += width;
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
evaluateProduct(ProductLayoutMapAttr map, ArrayRef<int64_t> coordinates) {
  auto outer = cast<AffineLayoutMapAttr>(map.getOuter());
  auto inner = cast<BitLinearLayoutMapAttr>(map.getInner());
  unsigned outerInputs = outer.getInputNames().size();
  unsigned innerInputs = inner.getInputNames().size();
  if (coordinates.size() != outerInputs + innerInputs)
    return failure();

  FailureOr<SmallVector<int64_t>> outerValues =
      evaluateAffine(outer, coordinates.take_front(outerInputs));
  FailureOr<SmallVector<int64_t>> innerValues =
      evaluateBitLinear(inner, coordinates.take_back(innerInputs));
  if (failed(outerValues) || failed(innerValues))
    return failure();

  SmallVector<int64_t> result;
  result.reserve(outer.getOutputNames().size());
  for (auto [outerIndex, nameAttr] :
       llvm::enumerate(outer.getOutputNames())) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    std::optional<unsigned> innerIndex = findName(inner.getOutputNames(), name);
    if (!innerIndex)
      return failure();
    int64_t outerValue = (*outerValues)[outerIndex];
    int64_t innerValue = (*innerValues)[*innerIndex];
    if (innerValue > 0 &&
        outerValue > std::numeric_limits<int64_t>::max() - innerValue)
      return failure();
    result.push_back(outerValue + innerValue);
  }
  return result;
}

FailureOr<SmallVector<int64_t>> evaluateLayout(Attribute map,
                                                ArrayRef<int64_t> point) {
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map))
    return evaluateAffine(affine, point);
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map))
    return evaluateBitLinear(bitLinear, point);
  if (auto product = dyn_cast<ProductLayoutMapAttr>(map))
    return evaluateProduct(product, point);
  return failure();
}

FailureOr<SmallVector<int64_t>> getStaticDomain(Attribute map) {
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map)) {
    SmallVector<int64_t> domain(affine.getInputExtents().asArrayRef());
    if (llvm::is_contained(domain, ShapedType::kDynamic))
      return failure();
    return domain;
  }
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map)) {
    SmallVector<int64_t> domain;
    domain.reserve(bitLinear.getInputBitWidths().size());
    for (int64_t width : bitLinear.getInputBitWidths().asArrayRef()) {
      if (width >= 63)
        return failure();
      domain.push_back(int64_t{1} << width);
    }
    return domain;
  }
  if (auto product = dyn_cast<ProductLayoutMapAttr>(map)) {
    auto outer = cast<AffineLayoutMapAttr>(product.getOuter());
    auto inner = cast<BitLinearLayoutMapAttr>(product.getInner());
    FailureOr<SmallVector<int64_t>> outerDomain = getStaticDomain(outer);
    FailureOr<SmallVector<int64_t>> innerDomain = getStaticDomain(inner);
    if (failed(outerDomain) || failed(innerDomain))
      return failure();
    outerDomain->append(innerDomain->begin(), innerDomain->end());
    return *outerDomain;
  }
  return failure();
}

FailureOr<uint64_t> getPointCount(ArrayRef<int64_t> extents) {
  uint64_t count = 1;
  for (int64_t extent : extents) {
    if (extent <= 0 || static_cast<uint64_t>(extent) >
                           kEnumerationLimit / count)
      return failure();
    count *= static_cast<uint64_t>(extent);
  }
  return count;
}

LogicalResult enumeratePoints(
    ArrayRef<int64_t> extents,
    function_ref<LogicalResult(ArrayRef<int64_t>)> callback) {
  FailureOr<uint64_t> pointCount = getPointCount(extents);
  if (failed(pointCount))
    return failure();
  SmallVector<int64_t> point(extents.size(), 0);
  for (uint64_t linear = 0; linear < *pointCount; ++linear) {
    if (failed(callback(point)))
      return failure();
    for (size_t index = extents.size(); index > 0; --index) {
      unsigned dimension = index - 1;
      if (++point[dimension] < extents[dimension])
        break;
      point[dimension] = 0;
    }
  }
  return success();
}

ArrayAttr selectNames(MLIRContext *context, ArrayAttr names,
                      ArrayRef<unsigned> order) {
  SmallVector<Attribute> selected;
  selected.reserve(order.size());
  for (unsigned index : order)
    selected.push_back(names[index]);
  return ArrayAttr::get(context, selected);
}

DenseI64ArrayAttr selectI64(MLIRContext *context, DenseI64ArrayAttr values,
                            ArrayRef<unsigned> order) {
  SmallVector<int64_t> selected;
  selected.reserve(order.size());
  for (unsigned index : order)
    selected.push_back(values[index]);
  return DenseI64ArrayAttr::get(context, selected);
}

FailureOr<Attribute> projectAffine(AffineLayoutMapAttr map,
                                   ArrayRef<StringRef> outputs) {
  FailureOr<SmallVector<unsigned>> order =
      getOutputOrder(map.getOutputNames(), outputs);
  if (failed(order))
    return failure();
  SmallVector<AffineExpr> results;
  results.reserve(order->size());
  for (unsigned index : *order)
    results.push_back(map.getAffineMap().getValue().getResult(index));
  AffineMap projected =
      AffineMap::get(map.getAffineMap().getValue().getNumDims(),
                     map.getAffineMap().getValue().getNumSymbols(), results,
                     map.getContext());
  return Attribute(AffineLayoutMapAttr::get(
      map.getContext(), map.getInputNames(), map.getInputExtents(),
      selectNames(map.getContext(), map.getOutputNames(), *order),
      selectI64(map.getContext(), map.getOutputExtents(), *order),
      AffineMapAttr::get(projected)));
}

FailureOr<Attribute> projectBitLinear(BitLinearLayoutMapAttr map,
                                      ArrayRef<StringRef> outputs) {
  FailureOr<SmallVector<unsigned>> order =
      getOutputOrder(map.getOutputNames(), outputs);
  if (failed(order))
    return failure();

  auto widths = map.getOutputBitWidths().asArrayRef();
  SmallVector<unsigned> offsets(widths.size());
  for (unsigned index = 1; index < widths.size(); ++index)
    offsets[index] = offsets[index - 1] + widths[index - 1];
  unsigned inputBits = 0;
  for (int64_t width : map.getInputBitWidths().asArrayRef())
    inputBits += static_cast<unsigned>(width);

  SmallVector<llvm::APInt> oldValues;
  for (const llvm::APInt &value : map.getMatrix().getValues<llvm::APInt>())
    oldValues.push_back(value);
  SmallVector<llvm::APInt> newValues;
  for (unsigned outputIndex : *order) {
    for (unsigned bit = 0; bit < static_cast<unsigned>(widths[outputIndex]);
         ++bit) {
      unsigned row = offsets[outputIndex] + bit;
      for (unsigned column = 0; column < inputBits; ++column)
        newValues.push_back(oldValues[row * inputBits + column]);
    }
  }

  DenseI64ArrayAttr selectedWidths =
      selectI64(map.getContext(), map.getOutputBitWidths(), *order);
  unsigned outputBits = 0;
  for (int64_t width : selectedWidths.asArrayRef())
    outputBits += static_cast<unsigned>(width);
  auto matrixType = RankedTensorType::get(
      {static_cast<int64_t>(outputBits), static_cast<int64_t>(inputBits)},
      IntegerType::get(map.getContext(), 1));
  return Attribute(BitLinearLayoutMapAttr::get(
      map.getContext(), map.getInputNames(), map.getInputBitWidths(),
      selectNames(map.getContext(), map.getOutputNames(), *order),
      selectedWidths, DenseIntElementsAttr::get(matrixType, newValues)));
}

FailureOr<Attribute> projectProduct(ProductLayoutMapAttr map,
                                    ArrayRef<StringRef> outputs) {
  auto outer = cast<AffineLayoutMapAttr>(map.getOuter());
  FailureOr<SmallVector<unsigned>> order =
      getOutputOrder(outer.getOutputNames(), outputs);
  if (failed(order))
    return failure();
  FailureOr<Attribute> projectedOuter = projectLayoutMap(map.getOuter(), outputs);
  FailureOr<Attribute> projectedInner = projectLayoutMap(map.getInner(), outputs);
  if (failed(projectedOuter) || failed(projectedInner))
    return failure();
  return Attribute(ProductLayoutMapAttr::get(
      map.getContext(), *projectedOuter, *projectedInner,
      selectI64(map.getContext(), map.getSplitExtents(), *order)));
}

ArrayAttr getOutputNames(Attribute map) {
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map))
    return affine.getOutputNames();
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map))
    return bitLinear.getOutputNames();
  if (auto product = dyn_cast<ProductLayoutMapAttr>(map))
    return cast<AffineLayoutMapAttr>(product.getOuter()).getOutputNames();
  return {};
}

FailureOr<Attribute> composeAffine(AffineLayoutMapAttr lhs,
                                   AffineLayoutMapAttr rhs) {
  if (lhs.getContext() != rhs.getContext())
    return failure();
  AffineMap lhsMap = lhs.getAffineMap().getValue();
  AffineMap rhsMap = rhs.getAffineMap().getValue();
  if (lhsMap.getNumSymbols() != 0 || rhsMap.getNumSymbols() != 0)
    return failure();
  FailureOr<SmallVector<unsigned>> order = getOutputOrder(
      rhs.getOutputNames(), llvm::map_to_vector(lhs.getInputNames(), [](Attribute value) {
        return cast<StringAttr>(value).getValue();
      }));
  if (failed(order) || order->size() != rhs.getOutputNames().size())
    return failure();
  for (auto [lhsIndex, rhsIndex] : llvm::enumerate(*order)) {
    int64_t lhsExtent = lhs.getInputExtents()[lhsIndex];
    int64_t rhsExtent = rhs.getOutputExtents()[rhsIndex];
    if (lhsExtent != ShapedType::kDynamic &&
        rhsExtent != ShapedType::kDynamic && lhsExtent != rhsExtent)
      return failure();
  }

  SmallVector<AffineExpr> alignedResults;
  alignedResults.reserve(order->size());
  for (unsigned index : *order)
    alignedResults.push_back(rhsMap.getResult(index));
  AffineMap alignedRhs = AffineMap::get(rhsMap.getNumDims(), 0,
                                        alignedResults, lhs.getContext());
  AffineMap composed = lhsMap.compose(alignedRhs);
  return Attribute(AffineLayoutMapAttr::get(
      lhs.getContext(), rhs.getInputNames(), rhs.getInputExtents(),
      lhs.getOutputNames(), lhs.getOutputExtents(),
      AffineMapAttr::get(composed)));
}

} // namespace

FailureOr<Attribute> composeLayoutMaps(Attribute lhs, Attribute rhs) {
  if (auto lhsBit = dyn_cast<BitLinearLayoutMapAttr>(lhs)) {
    auto rhsBit = dyn_cast<BitLinearLayoutMapAttr>(rhs);
    if (!rhsBit)
      return failure();
    FailureOr<BitLinearLayoutMapAttr> composed =
        composeBitLinear(lhsBit, rhsBit);
    if (failed(composed))
      return failure();
    return Attribute(*composed);
  }
  if (auto lhsAffine = dyn_cast<AffineLayoutMapAttr>(lhs)) {
    auto rhsAffine = dyn_cast<AffineLayoutMapAttr>(rhs);
    if (!rhsAffine)
      return failure();
    return composeAffine(lhsAffine, rhsAffine);
  }
  if (auto lhsProduct = dyn_cast<ProductLayoutMapAttr>(lhs)) {
    auto rhsProduct = dyn_cast<ProductLayoutMapAttr>(rhs);
    if (!rhsProduct ||
        lhsProduct.getSplitExtents() != rhsProduct.getSplitExtents())
      return failure();
    FailureOr<Attribute> outer =
        composeLayoutMaps(lhsProduct.getOuter(), rhsProduct.getOuter());
    FailureOr<Attribute> inner =
        composeLayoutMaps(lhsProduct.getInner(), rhsProduct.getInner());
    if (failed(outer) || failed(inner))
      return failure();
    return Attribute(ProductLayoutMapAttr::get(
        lhsProduct.getContext(), *outer, *inner,
        lhsProduct.getSplitExtents()));
  }
  return failure();
}

FailureOr<Attribute> projectLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs) {
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map))
    return projectAffine(affine, outputs);
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map))
    return projectBitLinear(bitLinear, outputs);
  if (auto product = dyn_cast<ProductLayoutMapAttr>(map))
    return projectProduct(product, outputs);
  return failure();
}

FailureOr<Attribute> permuteLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs) {
  ArrayAttr currentOutputs = getOutputNames(map);
  if (!currentOutputs || currentOutputs.size() != outputs.size())
    return failure();
  return projectLayoutMap(map, outputs);
}

LayoutProof checkCoverage(Attribute map, ArrayRef<int64_t> logicalShape) {
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map)) {
    if (logicalShape.size() == bitLinear.getOutputBitWidths().size()) {
      bool fitsOutputSpace = true;
      for (auto [extent, width] : llvm::zip_equal(
               logicalShape, bitLinear.getOutputBitWidths().asArrayRef())) {
        if (extent <= 0 || width >= 63 || extent > (int64_t{1} << width))
          fitsOutputSpace = false;
      }
      LayoutProof surjective = checkSurjective(bitLinear);
      if (fitsOutputSpace && surjective.status == ProofStatus::Proven)
        return {ProofStatus::Proven, {},
                "surjective GF(2) map covers the bounded logical shape"};
    }
  }

  FailureOr<SmallVector<int64_t>> domain = getStaticDomain(map);
  FailureOr<uint64_t> targetCount = getPointCount(logicalShape);
  if (failed(domain) || failed(targetCount) ||
      getOutputNames(map).size() != logicalShape.size())
    return {ProofStatus::Unknown, {},
            "coverage requires a bounded static domain and result shape"};

  std::set<std::vector<int64_t>> covered;
  if (failed(enumeratePoints(*domain, [&](ArrayRef<int64_t> point) {
        FailureOr<SmallVector<int64_t>> output = evaluateLayout(map, point);
        if (failed(output) || output->size() != logicalShape.size())
          return failure();
        bool inBounds = true;
        for (auto [coordinate, extent] :
             llvm::zip_equal(*output, logicalShape))
          inBounds &= coordinate >= 0 && coordinate < extent;
        if (inBounds)
          covered.insert(std::vector<int64_t>(output->begin(), output->end()));
        return success();
      })))
    return {ProofStatus::Unknown, {}, "layout evaluation was not provable"};

  if (covered.size() == *targetCount)
    return {ProofStatus::Proven, {},
            "every point in the bounded logical shape is covered"};

  SmallVector<int64_t> missing;
  (void)enumeratePoints(logicalShape, [&](ArrayRef<int64_t> point) {
    std::vector<int64_t> key(point.begin(), point.end());
    if (!missing.empty() || covered.count(key))
      return success();
    missing.assign(point.begin(), point.end());
    return success();
  });
  return {ProofStatus::Disproven, std::move(missing),
          "logical shape contains an uncovered point"};
}

LayoutProof checkInjectivity(Attribute map, ArrayRef<int64_t> domainShape) {
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map)) {
    FailureOr<SmallVector<int64_t>> fullDomain = getStaticDomain(bitLinear);
    if (succeeded(fullDomain) && ArrayRef<int64_t>(*fullDomain) == domainShape)
      return checkInjective(bitLinear);
  }

  FailureOr<uint64_t> pointCount = getPointCount(domainShape);
  if (failed(pointCount))
    return {ProofStatus::Unknown, {},
            "injectivity domain is dynamic or exceeds enumeration limit"};

  std::set<std::vector<int64_t>> seen;
  SmallVector<int64_t> duplicate;
  if (failed(enumeratePoints(domainShape, [&](ArrayRef<int64_t> point) {
        FailureOr<SmallVector<int64_t>> output = evaluateLayout(map, point);
        if (failed(output))
          return failure();
        std::vector<int64_t> key(output->begin(), output->end());
        if (!seen.insert(key).second && duplicate.empty())
          duplicate.assign(point.begin(), point.end());
        return success();
      })))
    return {ProofStatus::Unknown, {}, "layout evaluation was not provable"};

  if (!duplicate.empty())
    return {ProofStatus::Disproven, std::move(duplicate),
            "two domain points map to the same output"};
  return {ProofStatus::Proven, {},
          "bounded enumeration found a unique output for every domain point"};
}

namespace {

constexpr std::array<StringLiteral, 5> kCarrierNames = {
    "register", "lane", "warp", "warp_group", "cta"};

std::optional<unsigned> getCarrierIndex(StringRef name) {
  for (auto [index, carrier] : llvm::enumerate(kCarrierNames)) {
    if (name == carrier)
      return index;
  }
  return std::nullopt;
}

void collectMapInputNames(Attribute map, SmallVectorImpl<StringRef> &names) {
  auto append = [&](ArrayAttr values) {
    for (Attribute value : values)
      names.push_back(cast<StringAttr>(value).getValue());
  };
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map)) {
    append(affine.getInputNames());
    return;
  }
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map)) {
    append(bitLinear.getInputNames());
    return;
  }
  if (auto product = dyn_cast<ProductLayoutMapAttr>(map)) {
    collectMapInputNames(product.getOuter(), names);
    collectMapInputNames(product.getInner(), names);
  }
}

LogicalResult verifyStorageOutputNames(
    function_ref<InFlightDiagnostic()> emitError, Attribute map) {
  ArrayAttr names = getOutputNames(map);
  if (!names || names.size() != 2 ||
      cast<StringAttr>(names[0]).getValue() != "byte_offset" ||
      cast<StringAttr>(names[1]).getValue() != "bit_offset")
    return emitError()
           << "storage map outputs must be ['byte_offset', 'bit_offset']";
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map)) {
    if (affine.getOutputExtents()[1] != 8)
      return emitError() << "affine bit_offset extent must be 8";
  } else if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map)) {
    if (bitLinear.getOutputBitWidths()[1] != 3)
      return emitError() << "bit-linear bit_offset width must be 3";
  }
  return success();
}

LogicalResult verifyMapDomainMatchesType(Attribute map, ShapedType type,
                                         Location loc) {
  if (!type.hasRank() || !type.hasStaticShape())
    return emitError(loc) << "layout verification requires a static ranked type";

  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map)) {
    if (affine.getInputExtents().asArrayRef() != type.getShape())
      return emitError(loc)
             << "affine layout input extents must match the shaped type";
    return success();
  }
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map)) {
    auto widths = bitLinear.getInputBitWidths().asArrayRef();
    if (widths.size() != static_cast<size_t>(type.getRank()))
      return emitError(loc)
             << "bit-linear layout input rank must match the shaped type";
    for (auto [extent, width] : llvm::zip_equal(type.getShape(), widths)) {
      if (extent <= 0 || !llvm::isPowerOf2_64(extent) || width >= 63 ||
          extent != (int64_t{1} << width))
        return emitError(loc)
               << "bit-linear input widths must exactly encode type extents";
    }
    return success();
  }
  return emitError(loc)
         << "M1 type verification supports affine or bit-linear encoding maps";
}

} // namespace

Attribute DistributedEncodingAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc location = parser.getCurrentLocation();
  Attribute map;
  DenseI64ArrayAttr topology;
  int64_t replication = 0;
  if (parser.parseLess() || parser.parseKeyword("map") ||
      parser.parseEqual() || parser.parseAttribute(map) || parser.parseComma() ||
      parser.parseKeyword("topology") || parser.parseEqual() ||
      parseI64List(parser, topology) || parser.parseComma() ||
      parser.parseKeyword("replication") || parser.parseEqual() ||
      parser.parseInteger(replication) || parser.parseGreater())
    return {};
  IntegerAttr replicationAttr =
      parser.getBuilder().getI64IntegerAttr(replication);
  return parser.getChecked<DistributedEncodingAttr>(
      location, parser.getContext(), map, topology, replicationAttr);
}

void DistributedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<map = " << getMap() << ", topology = ";
  printI64List(printer, getTopology());
  printer << ", replication = " << getReplication().getInt() << '>';
}

LogicalResult DistributedEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute map,
    DenseI64ArrayAttr topology, IntegerAttr replication) {
  if (!isa<LayoutMapAttrInterface>(map))
    return emitError() << "distributed encoding map must implement "
                          "LayoutMapAttrInterface";
  if (topology.size() != kCarrierNames.size())
    return emitError() << "topology must contain register, lane, warp, "
                          "warp_group, and cta extents";
  for (int64_t extent : topology.asArrayRef()) {
    if (extent <= 0)
      return emitError() << "topology extents must be positive";
  }
  if (topology[1] > 32)
    return emitError() << "lane topology extent must not exceed 32 on SM90";
  if (!replication || replication.getInt() <= 0)
    return emitError() << "replication must be positive";

  SmallVector<StringRef> inputs;
  collectMapInputNames(map, inputs);
  llvm::StringSet<> seen;
  for (StringRef input : inputs) {
    if (!getCarrierIndex(input))
      return emitError() << "distributed layout contains unsupported carrier '"
                         << input << "'";
    if (!seen.insert(input).second)
      return emitError() << "duplicate distributed carrier '" << input << "'";
  }
  return success();
}

FailureOr<Attribute>
DistributedEncodingAttr::getCanonicalMap(ShapedType) const {
  auto map = dyn_cast<LayoutMapAttrInterface>(getMap());
  if (!map)
    return failure();
  return map.canonicalizeMap();
}

LogicalResult DistributedEncodingAttr::verifyForType(ShapedType type,
                                                       Location loc) const {
  if (!type.hasRank() || !type.hasStaticShape())
    return emitError(loc)
           << "distributed encoding requires a static ranked type in M1";
  auto map = dyn_cast<BitLinearLayoutMapAttr>(getMap());
  if (!map)
    return emitError(loc)
           << "M1 distributed type verification requires a bit-linear map";
  if (map.getOutputBitWidths().size() !=
      static_cast<size_t>(type.getRank()))
    return emitError(loc)
           << "distributed logical rank must match the shaped type";
  for (auto [extent, width] : llvm::zip_equal(
           type.getShape(), map.getOutputBitWidths().asArrayRef())) {
    if (extent <= 0 || !llvm::isPowerOf2_64(extent) || width >= 63 ||
        extent != (int64_t{1} << width))
      return emitError(loc)
             << "distributed output widths must exactly encode type extents";
  }

  uint64_t carrierCount = 1;
  for (int64_t extent : getTopology().asArrayRef()) {
    if (!llvm::isPowerOf2_64(extent) ||
        carrierCount > std::numeric_limits<uint64_t>::max() /
                           static_cast<uint64_t>(extent))
      return emitError(loc)
             << "bit-linear topology extents must be powers of two without "
                "overflow";
    carrierCount *= static_cast<uint64_t>(extent);
  }
  for (auto [nameAttr, width] : llvm::zip_equal(
           map.getInputNames(), map.getInputBitWidths().asArrayRef())) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    std::optional<unsigned> index = getCarrierIndex(name);
    if (!index || width >= 63 ||
        getTopology()[*index] != (int64_t{1} << width))
      return emitError(loc)
             << "carrier bit width must exactly match its topology extent";
  }

  FailureOr<GF2Matrix> matrix = map.getMatrixValue();
  if (failed(matrix))
    return emitError(loc) << "distributed GF(2) matrix is malformed";
  unsigned outputBits = 0;
  for (int64_t width : map.getOutputBitWidths().asArrayRef())
    outputBits += static_cast<unsigned>(width);
  unsigned rank = matrix->rank();
  if (rank != outputBits)
    return emitError(loc)
           << "distributed map does not cover the complete logical tile";
  if (rank >= 64 || carrierCount % (uint64_t{1} << rank) != 0)
    return emitError(loc)
           << "distributed replication cannot be derived from topology";
  uint64_t expectedReplication = carrierCount / (uint64_t{1} << rank);
  if (getReplication().getValue().getLimitedValue() != expectedReplication)
    return emitError(loc) << "replication does not match topology and map rank; "
                             "expected "
                          << expectedReplication;
  return success();
}

LayoutKind DistributedEncodingAttr::getKind() const {
  return LayoutKind::Distributed;
}

Attribute StorageLayoutAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc location = parser.getCurrentLocation();
  Attribute map;
  MemorySpaceAttr memorySpace;
  int64_t alignment = 0;
  int64_t vectorGranularity = 0;
  if (parser.parseLess() || parser.parseKeyword("map") ||
      parser.parseEqual() || parser.parseAttribute(map) || parser.parseComma() ||
      parser.parseKeyword("memory_space") || parser.parseEqual() ||
      parser.parseAttribute(memorySpace) || parser.parseComma() ||
      parser.parseKeyword("alignment") || parser.parseEqual() ||
      parser.parseInteger(alignment) || parser.parseComma() ||
      parser.parseKeyword("vector_granularity") || parser.parseEqual() ||
      parser.parseInteger(vectorGranularity) || parser.parseGreater())
    return {};
  Builder &builder = parser.getBuilder();
  return parser.getChecked<StorageLayoutAttr>(
      location, parser.getContext(), map, memorySpace,
      builder.getI64IntegerAttr(alignment),
      builder.getI64IntegerAttr(vectorGranularity));
}

void StorageLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<map = " << getMap() << ", memory_space = " << getMemorySpace()
          << ", alignment = " << getAlignment().getInt()
          << ", vector_granularity = " << getVectorGranularity().getInt()
          << '>';
}

LogicalResult StorageLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute map,
    MemorySpaceAttr memorySpace, IntegerAttr alignment,
    IntegerAttr vectorGranularity) {
  if (!isa<LayoutMapAttrInterface>(map))
    return emitError()
           << "storage encoding map must implement LayoutMapAttrInterface";
  if (!memorySpace || memorySpace.getValue() == attr::MemorySpace::Local)
    return emitError() << "storage encoding requires shared or global memory";
  auto isPositivePowerOfTwo = [](IntegerAttr value) {
    return value && value.getInt() > 0 &&
           llvm::isPowerOf2_64(static_cast<uint64_t>(value.getInt()));
  };
  if (!isPositivePowerOfTwo(alignment))
    return emitError() << "alignment must be a positive power of two";
  if (!isPositivePowerOfTwo(vectorGranularity))
    return emitError()
           << "vector granularity must be a positive power of two";
  if (vectorGranularity.getInt() > alignment.getInt())
    return emitError()
           << "vector granularity must not exceed guaranteed alignment";
  return verifyStorageOutputNames(emitError, map);
}

FailureOr<Attribute> StorageLayoutAttr::getCanonicalMap(ShapedType) const {
  auto map = dyn_cast<LayoutMapAttrInterface>(getMap());
  if (!map)
    return failure();
  return map.canonicalizeMap();
}

LogicalResult StorageLayoutAttr::verifyForType(ShapedType type,
                                                Location loc) const {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return emitError(loc) << "storage encoding requires a MemRefType";
  std::optional<attr::MemorySpace> typeSpace =
      attr::symbolizeMemorySpace(memref.getMemorySpaceAsInt());
  if (!typeSpace || *typeSpace != getMemorySpace().getValue())
    return emitError(loc)
           << "storage encoding memory space must match the MemRefType";
  if (failed(verifyMapDomainMatchesType(getMap(), type, loc)))
    return failure();

  LayoutProof injective = checkInjectivity(getMap(), type.getShape());
  if (injective.status != ProofStatus::Proven)
    return emitError(loc) << "storage map must be provably injective on the "
                             "live logical domain: "
                          << injective.reason;

  bool offsetsValid = true;
  if (failed(enumeratePoints(type.getShape(), [&](ArrayRef<int64_t> point) {
        FailureOr<SmallVector<int64_t>> output =
            evaluateLayout(getMap(), point);
        if (failed(output) || output->size() != 2)
          return failure();
        offsetsValid &= (*output)[0] >= 0 && (*output)[1] >= 0 &&
                        (*output)[1] < 8;
        return success();
      })))
    return emitError(loc)
           << "storage byte/bit offset bounds could not be proven";
  if (!offsetsValid)
    return emitError(loc)
           << "storage map produced a negative byte offset or bit offset "
              "outside [0, 8)";
  return success();
}

LayoutKind StorageLayoutAttr::getKind() const { return LayoutKind::Storage; }

} // namespace mlir::frisk
