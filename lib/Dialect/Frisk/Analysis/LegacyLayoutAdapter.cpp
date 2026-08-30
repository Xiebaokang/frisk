#include "Dialect/Frisk/Analysis/LegacyLayoutAdapter.h"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {
namespace {

constexpr uint64_t kLegacyEnumerationLimit = 65536;
constexpr std::array<StringLiteral, 5> kCarrierNames = {
    "register", "lane", "warp", "warp_group", "cta"};

LogicalResult forEachPoint(
    ArrayRef<int64_t> shape,
    function_ref<LogicalResult(ArrayRef<int64_t>)> callback) {
  uint64_t count = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || static_cast<uint64_t>(extent) >
                           kLegacyEnumerationLimit / count)
      return failure();
    count *= static_cast<uint64_t>(extent);
  }
  SmallVector<int64_t> point(shape.size(), 0);
  for (uint64_t linear = 0; linear < count; ++linear) {
    if (failed(callback(point)))
      return failure();
    for (size_t index = shape.size(); index > 0; --index) {
      unsigned dimension = index - 1;
      if (++point[dimension] < shape[dimension])
        break;
      point[dimension] = 0;
    }
  }
  return success();
}

FailureOr<SmallVector<int64_t>> evaluateAffine(
    AffineMap map, ArrayRef<int64_t> dimensions,
    ArrayRef<int64_t> symbols = {}) {
  if (map.getNumDims() != dimensions.size() ||
      map.getNumSymbols() != symbols.size())
    return failure();
  Builder builder(map.getContext());
  SmallVector<Attribute> operands;
  for (int64_t value : dimensions)
    operands.push_back(builder.getIndexAttr(value));
  for (int64_t value : symbols)
    operands.push_back(builder.getIndexAttr(value));
  SmallVector<Attribute> folded;
  if (failed(map.constantFold(operands, folded)))
    return failure();
  SmallVector<int64_t> result;
  for (Attribute value : folded)
    result.push_back(cast<IntegerAttr>(value).getInt());
  return result;
}

FailureOr<uint64_t> encodePoint(ArrayRef<int64_t> point,
                                ArrayRef<int64_t> widths) {
  if (point.size() != widths.size())
    return failure();
  uint64_t encoded = 0;
  unsigned offset = 0;
  for (auto [coordinate, widthValue] : llvm::zip_equal(point, widths)) {
    if (coordinate < 0 || widthValue <= 0 || widthValue >= 64 ||
        offset + widthValue > 63 ||
        static_cast<uint64_t>(coordinate) >= (uint64_t{1} << widthValue))
      return failure();
    encoded |= static_cast<uint64_t>(coordinate) << offset;
    offset += static_cast<unsigned>(widthValue);
  }
  return encoded;
}

SmallVector<int64_t> getPowerOfTwoWidths(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> widths;
  for (int64_t extent : shape) {
    if (extent <= 1 || !llvm::isPowerOf2_64(extent))
      return {};
    widths.push_back(llvm::Log2_64(static_cast<uint64_t>(extent)));
  }
  return widths;
}

ArrayAttr getDimensionNames(MLIRContext *context, StringRef prefix,
                            size_t count) {
  Builder builder(context);
  SmallVector<Attribute> names;
  for (size_t index = 0; index < count; ++index)
    names.push_back(builder.getStringAttr(
        (prefix + Twine(index)).str()));
  return builder.getArrayAttr(names);
}

DenseIntElementsAttr getDenseMatrix(MLIRContext *context, unsigned rows,
                                    unsigned columns,
                                    ArrayRef<uint64_t> basisOutputs) {
  SmallVector<llvm::APInt> values;
  values.reserve(rows * columns);
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned column = 0; column < columns; ++column)
      values.emplace_back(1, (basisOutputs[column] >> row) & 1);
  }
  auto type = RankedTensorType::get(
      {static_cast<int64_t>(rows), static_cast<int64_t>(columns)},
      IntegerType::get(context, 1));
  return DenseIntElementsAttr::get(type, values);
}

FailureOr<SmallVector<int64_t>> evaluateBitLinear(
    BitLinearLayoutMapAttr map, ArrayRef<int64_t> point) {
  FailureOr<uint64_t> encoded =
      encodePoint(point, map.getInputBitWidths().asArrayRef());
  FailureOr<GF2Matrix> matrix = map.getMatrixValue();
  if (failed(encoded) || failed(matrix))
    return failure();
  unsigned inputBits = matrix->getNumColumns();
  llvm::APInt input(inputBits, *encoded);
  llvm::APInt output = matrix->apply(input);
  SmallVector<int64_t> result;
  unsigned offset = 0;
  for (int64_t widthValue : map.getOutputBitWidths().asArrayRef()) {
    unsigned width = static_cast<unsigned>(widthValue);
    if (width > 63)
      return failure();
    result.push_back(static_cast<int64_t>(
        output.extractBits(width, offset).getZExtValue()));
    offset += width;
  }
  return result;
}

FailureOr<SmallVector<int64_t>> evaluateCanonicalMap(
    Attribute map, ArrayRef<int64_t> point) {
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map))
    return evaluateAffine(affine.getAffineMap().getValue(), point);
  if (auto bitLinear = dyn_cast<BitLinearLayoutMapAttr>(map))
    return evaluateBitLinear(bitLinear, point);
  return failure();
}

struct LegacyStorageInfo {
  SmallVector<int64_t> physicalExtents;
  SmallVector<uint64_t> bitAddresses;
  uint64_t maxByteExclusive = 0;
};

FailureOr<LegacyStorageInfo> analyzeLegacyStorage(LayoutAttr legacy,
                                                   MemRefType type) {
  if (!legacy || legacy.getForwardThread() ||
      legacy.getInputShape().asArrayRef() != type.getShape())
    return failure();
  AffineMap map = legacy.getForwardIndex().getValue();
  if (map.getNumSymbols() != 0 || map.getNumDims() != type.getRank() ||
      map.getNumResults() == 0)
    return failure();
  unsigned elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0)
    return failure();

  SmallVector<SmallVector<int64_t>> physicalPoints;
  LegacyStorageInfo info;
  info.physicalExtents.assign(map.getNumResults(), 0);
  if (failed(forEachPoint(type.getShape(), [&](ArrayRef<int64_t> point) {
        FailureOr<SmallVector<int64_t>> physical = evaluateAffine(map, point);
        if (failed(physical))
          return failure();
        for (auto [index, coordinate] : llvm::enumerate(*physical)) {
          if (coordinate < 0)
            return failure();
          info.physicalExtents[index] =
              std::max(info.physicalExtents[index], coordinate + 1);
        }
        physicalPoints.push_back(std::move(*physical));
        return success();
      })))
    return failure();

  std::set<uint64_t> occupiedBits;
  for (ArrayRef<int64_t> physical : physicalPoints) {
    uint64_t elementOffset = 0;
    for (auto [coordinate, extent] :
         llvm::zip_equal(physical, info.physicalExtents)) {
      uint64_t unsignedExtent = static_cast<uint64_t>(extent);
      uint64_t unsignedCoordinate = static_cast<uint64_t>(coordinate);
      if (elementOffset > std::numeric_limits<uint64_t>::max() /
                              unsignedExtent)
        return failure();
      elementOffset *= unsignedExtent;
      if (unsignedCoordinate >
          std::numeric_limits<uint64_t>::max() - elementOffset)
        return failure();
      elementOffset += unsignedCoordinate;
    }
    if (elementOffset > std::numeric_limits<uint64_t>::max() / elementBits)
      return failure();
    uint64_t bitAddress = elementOffset * elementBits;
    if (!occupiedBits.insert(bitAddress).second)
      return failure();
    info.bitAddresses.push_back(bitAddress);
    info.maxByteExclusive =
        std::max(info.maxByteExclusive, bitAddress / 8 +
                                                (elementBits + 7) / 8);
  }
  return info;
}

uint64_t conservativeAlignment(unsigned elementBits) {
  if (elementBits < 8 || elementBits % 8 != 0)
    return 1;
  uint64_t bytes = elementBits / 8;
  return bytes & (~bytes + 1);
}

FailureOr<BitLinearLayoutMapAttr>
buildBitLinearStorageMap(MemRefType type, const LegacyStorageInfo &info) {
  SmallVector<int64_t> inputWidths =
      getPowerOfTwoWidths(type.getShape());
  if (inputWidths.size() != static_cast<size_t>(type.getRank()))
    return failure();
  unsigned inputBits = 0;
  for (int64_t width : inputWidths)
    inputBits += static_cast<unsigned>(width);
  if (inputBits >= 63 || info.bitAddresses.size() != (uint64_t{1} << inputBits) ||
      info.maxByteExclusive == 0)
    return failure();

  unsigned byteBits = std::max<unsigned>(
      1, llvm::Log2_64_Ceil(info.maxByteExclusive));
  unsigned outputBits = byteBits + 3;
  if (outputBits > 63 || info.bitAddresses.front() != 0)
    return failure();

  SmallVector<uint64_t> basisOutputs(inputBits, 0);
  SmallVector<int64_t> basisPoint(type.getRank(), 0);
  unsigned bit = 0;
  for (auto [dimension, widthValue] : llvm::enumerate(inputWidths)) {
    for (int64_t localBit = 0; localBit < widthValue; ++localBit) {
      basisPoint[dimension] = int64_t{1} << localBit;
      uint64_t linear = 0;
      for (auto [coordinate, extent] :
           llvm::zip_equal(basisPoint, type.getShape()))
        linear = linear * static_cast<uint64_t>(extent) + coordinate;
      uint64_t address = info.bitAddresses[linear];
      basisOutputs[bit++] = (address / 8) | ((address % 8) << byteBits);
      basisPoint[dimension] = 0;
    }
  }

  FailureOr<GF2Matrix> matrix = GF2Matrix::get(
      outputBits, inputBits,
      llvm::map_to_vector(llvm::seq<unsigned>(0, outputBits),
                          [&](unsigned row) {
                            llvm::APInt rowBits(inputBits, 0);
                            for (unsigned column = 0; column < inputBits;
                                 ++column)
                              if ((basisOutputs[column] >> row) & 1)
                                rowBits.setBit(column);
                            return rowBits;
                          }));
  if (failed(matrix))
    return failure();

  uint64_t linear = 0;
  if (failed(forEachPoint(type.getShape(), [&](ArrayRef<int64_t> point) {
        FailureOr<uint64_t> encoded = encodePoint(point, inputWidths);
        if (failed(encoded))
          return failure();
        uint64_t address = info.bitAddresses[linear++];
        uint64_t expected =
            (address / 8) | ((address % 8) << byteBits);
        llvm::APInt actual = matrix->apply(llvm::APInt(inputBits, *encoded));
        return actual.getZExtValue() == expected ? success() : failure();
      })))
    return failure();

  Builder builder(type.getContext());
  return BitLinearLayoutMapAttr::get(
      type.getContext(),
      getDimensionNames(type.getContext(), "dim", type.getRank()),
      builder.getDenseI64ArrayAttr(inputWidths),
      builder.getArrayAttr(
          {builder.getStringAttr("byte_offset"),
           builder.getStringAttr("bit_offset")}),
      builder.getDenseI64ArrayAttr(
          {static_cast<int64_t>(byteBits), int64_t{3}}),
      getDenseMatrix(type.getContext(), outputBits, inputBits, basisOutputs));
}

FailureOr<AffineLayoutMapAttr>
buildAffineStorageMap(LayoutAttr legacy, MemRefType type,
                      const LegacyStorageInfo &info) {
  AffineMap legacyMap = legacy.getForwardIndex().getValue();
  if (legacyMap.getNumResults() != 1 || info.maxByteExclusive == 0)
    return failure();
  int64_t elementBits = type.getElementTypeBitWidth();
  AffineExpr bitAddress = legacyMap.getResult(0) * elementBits;
  SmallVector<AffineExpr> results = {bitAddress.floorDiv(8), bitAddress % 8};
  Builder builder(type.getContext());
  AffineMap map = AffineMap::get(type.getRank(), 0, results,
                                 type.getContext());
  return AffineLayoutMapAttr::get(
      type.getContext(),
      getDimensionNames(type.getContext(), "dim", type.getRank()),
      builder.getDenseI64ArrayAttr(type.getShape()),
      builder.getArrayAttr(
          {builder.getStringAttr("byte_offset"),
           builder.getStringAttr("bit_offset")}),
      builder.getDenseI64ArrayAttr(
          {static_cast<int64_t>(info.maxByteExclusive), int64_t{8}}),
      AffineMapAttr::get(map));
}

FailureOr<SmallVector<int64_t>> getCarrierPoint(BitLinearLayoutMapAttr map,
                                                 int64_t registerIndex,
                                                 int64_t thread) {
  SmallVector<int64_t> point;
  for (Attribute nameAttr : map.getInputNames()) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    if (name == "register")
      point.push_back(registerIndex);
    else if (name == "lane")
      point.push_back(thread % 32);
    else if (name == "warp")
      point.push_back((thread / 32) % 4);
    else if (name == "warp_group")
      point.push_back(thread / 128);
    else if (name == "cta")
      point.push_back(0);
    else
      return failure();
  }
  return point;
}

} // namespace

FailureOr<DistributedEncodingAttr>
convertLegacyDistributed(LayoutAttr legacy, ShapedType type, Location loc) {
  auto reject = [&]() -> FailureOr<DistributedEncodingAttr> {
    emitError(loc) << "legacy layout cannot be represented by the canonical "
                      "layout algebra";
    return failure();
  };
  IntegerAttr legacyReplication = legacy ? legacy.getReplicateSize()
                                         : IntegerAttr();
  if (!legacy || !type.hasRank() || !type.hasStaticShape() ||
      legacy.getInputShape().asArrayRef() != type.getShape() ||
      !legacy.getForwardThread() || !legacyReplication ||
      legacyReplication.getInt() <= 0 ||
      !llvm::isPowerOf2_64(legacyReplication.getInt()))
    return reject();
  int64_t replication = legacyReplication.getInt();
  SmallVector<int64_t> outputWidths = getPowerOfTwoWidths(type.getShape());
  if (outputWidths.size() != static_cast<size_t>(type.getRank()))
    return reject();
  AffineMap registerMap = legacy.getForwardIndex().getValue();
  AffineMap threadMap = legacy.getForwardThread().getValue();
  if (registerMap.getNumResults() != 1 || threadMap.getNumResults() != 1 ||
      registerMap.getNumSymbols() != 0 || threadMap.getNumSymbols() != 0)
    return reject();

  int64_t maxRegister = -1;
  int64_t maxThread = -1;
  std::map<std::pair<int64_t, int64_t>, uint64_t> inverse;
  if (failed(forEachPoint(type.getShape(), [&](ArrayRef<int64_t> logical) {
        FailureOr<SmallVector<int64_t>> reg =
            evaluateAffine(registerMap, logical);
        FailureOr<SmallVector<int64_t>> thread =
            evaluateAffine(threadMap, logical);
        FailureOr<uint64_t> encoded = encodePoint(logical, outputWidths);
        if (failed(reg) || failed(thread) || failed(encoded) || (*reg)[0] < 0 ||
            (*thread)[0] < 0 ||
            !inverse.emplace(std::make_pair((*thread)[0], (*reg)[0]),
                             *encoded)
                 .second)
          return failure();
        maxRegister = std::max(maxRegister, (*reg)[0]);
        maxThread = std::max(maxThread, (*thread)[0]);
        return success();
      })))
    return reject();

  int64_t registerExtent = maxRegister + 1;
  int64_t threadExtent = maxThread + 1;
  if (!llvm::isPowerOf2_64(registerExtent) || threadExtent < 32 ||
      threadExtent % 32 != 0 || !llvm::isPowerOf2_64(threadExtent) ||
      static_cast<uint64_t>(registerExtent) * threadExtent != inverse.size())
    return reject();
  int64_t baseWarpExtent = std::min<int64_t>(4, threadExtent / 32);
  int64_t baseWarpGroupExtent =
      threadExtent / (32 * baseWarpExtent);
  SmallVector<int64_t> baseTopology = {
      registerExtent, 32, baseWarpExtent, baseWarpGroupExtent, 1};
  SmallVector<int64_t> topology = baseTopology;
  int64_t remainingReplication = replication;
  int64_t warpCapacity = 4 / baseWarpExtent;
  int64_t warpReplication =
      std::min(remainingReplication, warpCapacity);
  topology[2] *= warpReplication;
  remainingReplication /= warpReplication;
  topology[3] *= remainingReplication;
  for (int64_t extent : topology)
    if (!llvm::isPowerOf2_64(extent))
      return reject();

  Builder builder(type.getContext());
  SmallVector<Attribute> inputNames;
  SmallVector<int64_t> inputWidths;
  for (auto [name, extent] : llvm::zip_equal(kCarrierNames, topology)) {
    if (extent == 1)
      continue;
    inputNames.push_back(builder.getStringAttr(name));
    inputWidths.push_back(llvm::Log2_64(static_cast<uint64_t>(extent)));
  }
  unsigned inputBits = 0;
  for (int64_t width : inputWidths)
    inputBits += static_cast<unsigned>(width);
  unsigned outputBits = 0;
  for (int64_t width : outputWidths)
    outputBits += static_cast<unsigned>(width);
  if (inputBits >= 63 || outputBits >= 63 || inputBits < outputBits)
    return reject();

  SmallVector<uint64_t> basisOutputs(inputBits, 0);
  unsigned column = 0;
  for (auto [carrierIndex, extent] : llvm::enumerate(topology)) {
    if (extent == 1)
      continue;
    unsigned width = llvm::Log2_64(static_cast<uint64_t>(extent));
    for (unsigned bit = 0; bit < width; ++bit) {
      int64_t coordinate = int64_t{1} << bit;
      if (coordinate >= baseTopology[carrierIndex]) {
        // This topology bit is an explicit replication carrier. A zero
        // matrix column makes its ownership semantics visible and provable.
        basisOutputs[column++] = 0;
        continue;
      }
      int64_t reg = carrierIndex == 0 ? coordinate : 0;
      int64_t lane = carrierIndex == 1 ? coordinate : 0;
      int64_t warp = carrierIndex == 2 ? coordinate : 0;
      int64_t warpGroup = carrierIndex == 3 ? coordinate : 0;
      int64_t thread =
          lane + 32 * (warp + baseWarpExtent * warpGroup);
      auto found = inverse.find({thread, reg});
      if (found == inverse.end())
        return reject();
      basisOutputs[column++] = found->second;
    }
  }
  auto origin = inverse.find({0, 0});
  if (origin == inverse.end() || origin->second != 0)
    return reject();

  FailureOr<GF2Matrix> matrix = GF2Matrix::get(
      outputBits, inputBits,
      llvm::map_to_vector(llvm::seq<unsigned>(0, outputBits),
                          [&](unsigned row) {
                            llvm::APInt rowBits(inputBits, 0);
                            for (unsigned input = 0; input < inputBits; ++input)
                              if ((basisOutputs[input] >> row) & 1)
                                rowBits.setBit(input);
                            return rowBits;
                          }));
  if (failed(matrix))
    return reject();
  for (auto [hardware, logical] : inverse) {
    int64_t thread = hardware.first;
    int64_t reg = hardware.second;
    SmallVector<int64_t> carrierPoint;
    for (Attribute nameAttr : inputNames) {
      StringRef name = cast<StringAttr>(nameAttr).getValue();
      if (name == "register")
        carrierPoint.push_back(reg);
      else if (name == "lane")
        carrierPoint.push_back(thread % 32);
      else if (name == "warp")
        carrierPoint.push_back((thread / 32) % baseWarpExtent);
      else if (name == "warp_group")
        carrierPoint.push_back(thread / (32 * baseWarpExtent));
      else
        carrierPoint.push_back(0);
    }
    FailureOr<uint64_t> encoded = encodePoint(carrierPoint, inputWidths);
    if (failed(encoded) ||
        matrix->apply(llvm::APInt(inputBits, *encoded)).getZExtValue() !=
            logical)
      return reject();
  }

  BitLinearLayoutMapAttr canonical = BitLinearLayoutMapAttr::get(
      type.getContext(), builder.getArrayAttr(inputNames),
      builder.getDenseI64ArrayAttr(inputWidths),
      getDimensionNames(type.getContext(), "dim", type.getRank()),
      builder.getDenseI64ArrayAttr(outputWidths),
      getDenseMatrix(type.getContext(), outputBits, inputBits, basisOutputs));
  DistributedEncodingAttr converted = DistributedEncodingAttr::get(
      type.getContext(), canonical, builder.getDenseI64ArrayAttr(topology),
      builder.getI64IntegerAttr(replication));
  if (failed(converted.verifyForType(type, loc)))
    return failure();
  return converted;
}

FailureOr<StorageLayoutAttr>
convertLegacyStorage(LayoutAttr legacy, MemRefType type, Location loc) {
  auto reject = [&]() -> FailureOr<StorageLayoutAttr> {
    emitError(loc) << "legacy layout cannot be represented by the canonical "
                      "layout algebra";
    return failure();
  };
  std::optional<attr::MemorySpace> memorySpace =
      attr::symbolizeMemorySpace(type.getMemorySpaceAsInt());
  if (!memorySpace || *memorySpace == attr::MemorySpace::Local)
    return reject();
  FailureOr<LegacyStorageInfo> info = analyzeLegacyStorage(legacy, type);
  if (failed(info))
    return reject();

  Attribute canonical;
  if (FailureOr<BitLinearLayoutMapAttr> bitLinear =
          buildBitLinearStorageMap(type, *info);
      succeeded(bitLinear))
    canonical = *bitLinear;
  else if (FailureOr<AffineLayoutMapAttr> affine =
               buildAffineStorageMap(legacy, type, *info);
           succeeded(affine))
    canonical = *affine;
  else
    return reject();

  Builder builder(type.getContext());
  int64_t alignment = conservativeAlignment(type.getElementTypeBitWidth());
  StorageLayoutAttr converted = StorageLayoutAttr::get(
      type.getContext(), canonical,
      MemorySpaceAttr::get(type.getContext(), *memorySpace),
      builder.getI64IntegerAttr(alignment),
      builder.getI64IntegerAttr(alignment));
  if (failed(converted.verifyForType(type, loc)))
    return failure();
  return converted;
}

LogicalResult verifyLegacyDistributedEquivalent(
    LayoutAttr legacy, DistributedEncodingAttr converted, ShapedType type,
    Location loc) {
  auto map = dyn_cast<BitLinearLayoutMapAttr>(converted.getMap());
  if (!map || !legacy.getForwardThread() ||
      legacy.getForwardIndex().getValue().getNumResults() != 1)
    return emitError(loc) << "distributed equivalence requires canonical "
                             "bit-linear and scalar legacy maps";
  SmallVector<int64_t> expected(type.getRank(), 0);
  if (failed(forEachPoint(type.getShape(), [&](ArrayRef<int64_t> logical) {
        FailureOr<SmallVector<int64_t>> reg = evaluateAffine(
            legacy.getForwardIndex().getValue(), logical);
        FailureOr<SmallVector<int64_t>> thread = evaluateAffine(
            legacy.getForwardThread().getValue(), logical);
        if (failed(reg) || failed(thread))
          return failure();
        FailureOr<SmallVector<int64_t>> carrier =
            getCarrierPoint(map, (*reg)[0], (*thread)[0]);
        FailureOr<SmallVector<int64_t>> actual =
            failed(carrier) ? FailureOr<SmallVector<int64_t>>(failure())
                            : evaluateBitLinear(map, *carrier);
        if (failed(actual) || ArrayRef<int64_t>(*actual) != logical) {
          expected.assign(logical.begin(), logical.end());
          return failure();
        }
        return success();
      }))) {
    InFlightDiagnostic diagnostic =
        emitError(loc) << "legacy distributed conversion mismatch at logical [";
    llvm::interleaveComma(expected, diagnostic);
    diagnostic << ']';
    return failure();
  }
  return success();
}

LogicalResult verifyLegacyStorageEquivalent(LayoutAttr legacy,
                                            StorageLayoutAttr converted,
                                            MemRefType type, Location loc) {
  FailureOr<LegacyStorageInfo> info = analyzeLegacyStorage(legacy, type);
  if (failed(info))
    return emitError(loc) << "legacy storage layout is not enumerable";
  uint64_t linear = 0;
  SmallVector<int64_t> mismatch(type.getRank(), 0);
  if (failed(forEachPoint(type.getShape(), [&](ArrayRef<int64_t> logical) {
        FailureOr<SmallVector<int64_t>> actual =
            evaluateCanonicalMap(converted.getMap(), logical);
        uint64_t bitAddress = info->bitAddresses[linear++];
        if (failed(actual) || actual->size() != 2 ||
            (*actual)[0] != static_cast<int64_t>(bitAddress / 8) ||
            (*actual)[1] != static_cast<int64_t>(bitAddress % 8)) {
          mismatch.assign(logical.begin(), logical.end());
          return failure();
        }
        return success();
      }))) {
    InFlightDiagnostic diagnostic =
        emitError(loc) << "legacy storage conversion mismatch at logical [";
    llvm::interleaveComma(mismatch, diagnostic);
    diagnostic << ']';
    return failure();
  }
  return success();
}

} // namespace mlir::frisk
