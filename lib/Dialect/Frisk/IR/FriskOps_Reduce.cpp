#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/Visitors.h"

#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskEnums.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Utils/LayoutUtils.h"

namespace mlir {
namespace frisk {

namespace {

static std::optional<int64_t> inferThreadBlockSize(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto attr = cur->getAttrOfType<IntegerAttr>("frisk.threads"))
      return attr.getInt();
    if (auto parallel = dyn_cast<ParallelOp>(cur))
      return parallel.getThreads();
    cur = cur->getParentOp();
  }
  return std::nullopt;
}

static AffineMapAttr remapReduceDimension(OpBuilder &builder,
                                          AffineMapAttr mapAttr,
                                          unsigned removeDim,
                                          int64_t reduceExtent,
                                          bool useFloorDiv) {
  if (!mapAttr)
    return AffineMapAttr();

  AffineMap map = mapAttr.getValue();
  unsigned srcDims = map.getNumDims();
  if (removeDim >= srcDims)
    return AffineMapAttr();

  unsigned dstDims = srcDims - 1;
  unsigned placeholderDim = dstDims;
  unsigned newDimCount = dstDims + 1;

  MLIRContext *ctx = builder.getContext();
  AffineExpr placeholder = builder.getAffineDimExpr(placeholderDim);

  SmallVector<AffineExpr> dimSubs;
  dimSubs.reserve(srcDims);
  for (unsigned i = 0; i < srcDims; ++i) {
    if (i < removeDim) {
      dimSubs.push_back(builder.getAffineDimExpr(i));
      continue;
    }
    if (i == removeDim) {
      if (ShapedType::isDynamic(reduceExtent)) {
        dimSubs.push_back(placeholder);
      } else if (reduceExtent == 1) {
        dimSubs.push_back(builder.getAffineConstantExpr(0));
      } else {
        AffineExpr constant = builder.getAffineConstantExpr(reduceExtent);
        dimSubs.push_back(useFloorDiv ? placeholder.floorDiv(constant)
                                      : placeholder % constant);
      }
      continue;
    }
    dimSubs.push_back(builder.getAffineDimExpr(i - 1));
  }

  SmallVector<AffineExpr> symSubs;
  symSubs.reserve(map.getNumSymbols());
  for (unsigned i = 0; i < map.getNumSymbols(); ++i)
    symSubs.push_back(builder.getAffineSymbolExpr(i));

  SmallVector<AffineExpr> newResults;
  newResults.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults())
    newResults.push_back(expr.replaceDimsAndSymbols(dimSubs, symSubs));

  AffineMap newMap =
      AffineMap::get(newDimCount, map.getNumSymbols(), newResults, ctx);
  return AffineMapAttr::get(newMap);
}

static DenseI64ArrayAttr buildShapeAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> shape) {
  SmallVector<int64_t> values(shape.begin(), shape.end());
  if (values.empty())
    values.push_back(1);
  return builder.getDenseI64ArrayAttr(values);
}

static bool mapsAgreeOnDomain(MLIRContext *ctx, AffineMap lhs, AffineMap rhs,
                              ArrayRef<int64_t> dimExtents,
                              ArrayRef<int64_t> lhsSymbolExtents,
                              ArrayRef<int64_t> rhsSymbolExtents) {
  if (lhs.getNumDims() != rhs.getNumDims() ||
      lhs.getNumResults() != rhs.getNumResults())
    return false;
  if (lhs.getNumDims() != dimExtents.size() ||
      lhs.getNumSymbols() != lhsSymbolExtents.size() ||
      rhs.getNumSymbols() != rhsSymbolExtents.size())
    return false;

  Builder builder(ctx);
  SmallVector<int64_t> dimValues(dimExtents.size(), 0);
  SmallVector<int64_t> lhsSymValues(lhsSymbolExtents.size(), 0);
  SmallVector<int64_t> rhsSymValues(rhsSymbolExtents.size(), 0);

  auto foldMap = [&](AffineMap map, ArrayRef<int64_t> dims,
                     ArrayRef<int64_t> symbols,
                     SmallVectorImpl<Attribute> &results) -> LogicalResult {
    SmallVector<Attribute> operands;
    operands.reserve(dims.size() + symbols.size());
    for (int64_t value : dims)
      operands.push_back(builder.getI64IntegerAttr(value));
    for (int64_t value : symbols)
      operands.push_back(builder.getI64IntegerAttr(value));
    return map.constantFold(operands, results);
  };

  std::function<bool(unsigned)> enumerateDims = [&](unsigned index) -> bool {
    if (index == dimExtents.size()) {
      std::function<bool(unsigned)> enumerateLhsSymbols =
          [&](unsigned lhsIndex) -> bool {
        if (lhsIndex == lhsSymbolExtents.size()) {
          std::function<bool(unsigned)> enumerateRhsSymbols =
              [&](unsigned rhsIndex) -> bool {
            if (rhsIndex == rhsSymbolExtents.size()) {
              SmallVector<Attribute> lhsResults;
              SmallVector<Attribute> rhsResults;
              if (failed(foldMap(lhs, dimValues, lhsSymValues, lhsResults)) ||
                  failed(foldMap(rhs, dimValues, rhsSymValues, rhsResults)))
                return false;
              return lhsResults == rhsResults;
            }
            for (int64_t value = 0; value < rhsSymbolExtents[rhsIndex];
                 ++value) {
              rhsSymValues[rhsIndex] = value;
              if (!enumerateRhsSymbols(rhsIndex + 1))
                return false;
            }
            return true;
          };
          return enumerateRhsSymbols(0);
        }
        for (int64_t value = 0; value < lhsSymbolExtents[lhsIndex]; ++value) {
          lhsSymValues[lhsIndex] = value;
          if (!enumerateLhsSymbols(lhsIndex + 1))
            return false;
        }
        return true;
      };
      return enumerateLhsSymbols(0);
    }

    for (int64_t value = 0; value < dimExtents[index]; ++value) {
      dimValues[index] = value;
      if (!enumerateDims(index + 1))
        return false;
    }
    return true;
  };

  return enumerateDims(0);
}

static LogicalResult checkLayoutCompatibility(ReduceOp op,
                                              LayoutAttr computed,
                                              LayoutAttr existing) {
  auto emitConflict = [&](StringRef detail) -> LogicalResult {
    op.emitOpError("destination layout conflicts with reduce inference: ")
        << detail << "\nexpected=" << layoutDebugString(computed)
        << "\nactual=" << layoutDebugString(existing);
    return failure();
  };

  if (!existing)
    return success();

  auto dimShape = computed.getInputShape();
  if (dimShape != existing.getInputShape())
    return emitConflict("input shapes disagree");

  auto lhsRep = computed.getReplicateSize();
  auto rhsRep = existing.getReplicateSize();
  int64_t computedRep = lhsRep ? lhsRep.getInt() : 1;
  int64_t existingRep = rhsRep ? rhsRep.getInt() : 1;
  if (computedRep <= 0 || existingRep <= 0)
    return emitConflict("replicate extent must be positive");
  if (computedRep > existingRep)
    return emitConflict("existing layout replicates fewer lanes than inferred");

  SmallVector<int64_t> dimExtents(dimShape.asArrayRef().begin(),
                                  dimShape.asArrayRef().end());
  if (!mapsAgreeOnDomain(op.getContext(), computed.getForwardIndex().getValue(),
                         existing.getForwardIndex().getValue(), dimExtents,
                         /*lhsSymbolExtents=*/{}, /*rhsSymbolExtents=*/{}))
    return emitConflict("forward index maps differ");

  AffineMap computedThread = computed.getForwardThread().getValue();
  AffineMap existingThread = existing.getForwardThread().getValue();
  SmallVector<int64_t> computedSymbols(computedThread.getNumSymbols(),
                                       computedRep);
  SmallVector<int64_t> existingSymbols(existingThread.getNumSymbols(),
                                       computedRep);
  if (!mapsAgreeOnDomain(op.getContext(), computedThread, existingThread,
                         dimExtents, computedSymbols, existingSymbols))
    return emitConflict("thread maps differ");
  return success();
}

} // namespace

LogicalResult ReduceOp::inferLayout(OpBuilder &builder,
                                    DenseMap<Value, Attribute> &layoutMap) {
  auto srcType = dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(getDst().getType());
  if (!srcType || !dstType)
    return emitOpError("layout inference requires memref operands");

  auto parseMemorySpace = [&](MemRefType type,
                              StringRef label) -> std::optional<attr::MemorySpace> {
    if (std::optional<attr::MemorySpace> memorySpace =
            getFriskMemorySpace(type))
      return memorySpace;
    emitOpError() << "operand " << label
                  << " resides in an unsupported memory space";
    return std::nullopt;
  };

  auto srcSpace = parseMemorySpace(srcType, "src");
  auto dstSpace = parseMemorySpace(dstType, "dst");
  if (!srcSpace || !dstSpace)
    return failure();

  if (*srcSpace != attr::MemorySpace::Local ||
      *dstSpace != attr::MemorySpace::Local)
    return success();

  auto srcIt = layoutMap.find(getSrc());
  if (srcIt == layoutMap.end())
    return success();

  auto srcLayout = dyn_cast<LayoutAttr>(srcIt->second);
  if (!srcLayout)
    return emitOpError("source layout entry must be a frisk.layout attribute");

  auto srcThreadMap = srcLayout.getForwardThread();
  if (!srcThreadMap)
    return success();

  DenseI64ArrayAttr layoutShape = srcLayout.getInputShape();
  if (!layoutShape)
    return emitOpError("source layout missing input shape metadata");

  ArrayRef<int64_t> layoutDims = layoutShape.asArrayRef();
  if (layoutDims.size() != static_cast<size_t>(srcType.getRank()))
    return emitOpError("source layout rank does not match source memref rank");

  int64_t dim = getDim();
  if (dim < 0 || dim >= srcType.getRank())
    return emitOpError("invalid reduce dimension ") << dim;

  int64_t reduceExtent = srcType.getShape()[dim];
  if (ShapedType::isDynamic(reduceExtent))
    return emitOpError(
        "layout inference requires static extent along the reduce dimension");
  if (reduceExtent <= 0)
    return emitOpError("reduce extent must be positive for layout inference");

  auto remappedThread = remapReduceDimension(
      builder, srcThreadMap, static_cast<unsigned>(dim), reduceExtent,
      /*useFloorDiv=*/false);

  std::optional<int64_t> replicateValue;
  int64_t baseReplicate = 1;
  if (auto replicateAttr = srcLayout.getReplicateSize()) {
    baseReplicate = replicateAttr.getInt();
    if (baseReplicate <= 0)
      return emitOpError("source layout replicate extent must be positive");
    replicateValue = baseReplicate;
  }

  int64_t uncondensedReplicate = std::max<int64_t>(int64_t(1), baseReplicate);
  if (llvm::MulOverflow(uncondensedReplicate, reduceExtent, uncondensedReplicate))
    return emitOpError("replicate extent overflow while inferring layout");

  auto threads = inferThreadBlockSize(getOperation());
  if (threads && *threads > 0 && uncondensedReplicate > 0 &&
      (*threads % uncondensedReplicate != 0) &&
      (uncondensedReplicate % *threads != 0)) {
    return emitOpError()
           << "reduce layout inference requires thread count divisible by "
              "replicate extent before condense (threads="
           << *threads << ", replicate=" << uncondensedReplicate << ")";
  }

  if (!remappedThread)
    return emitOpError("failed to remap source thread layout for reduce op");
  AffineMap remappedThreadMap = remappedThread.getValue();
  if (remappedThreadMap.getNumDims() == 0)
    return emitOpError("thread layout missing placeholder dimension");

  unsigned placeholder = remappedThreadMap.getNumDims() - 1;
  auto compressedThread =
      compressReplicateDimInMap(builder, remappedThread, placeholder,
                                reduceExtent);
  if (!compressedThread || !compressedThread->mapAttr)
    return emitOpError("unable to condense reduce replicate dimension in "
                       "thread map");

  auto dstThreadMap = compressedThread->mapAttr;
  auto dstIndexMap =
      inferFragmentIndexFromThreadMap(builder, dstThreadMap, dstType.getShape());
  if (!dstIndexMap)
    return emitOpError("failed to infer destination fragment index from "
                       "thread map");

  int64_t condensedReplicate = std::max<int64_t>(
      int64_t(1), compressedThread->replicateExtent);
  int64_t finalReplicate = std::max<int64_t>(int64_t(1), baseReplicate);
  if (llvm::MulOverflow(finalReplicate, condensedReplicate, finalReplicate))
    return emitOpError("replicate extent overflow while inferring layout");
  replicateValue = finalReplicate;

  IntegerAttr replicateAttr = builder.getI64IntegerAttr(finalReplicate);

  auto dstShapeAttr = buildShapeAttr(builder, dstType.getShape());
  LayoutAttr dstLayout = LayoutAttr::get(
      builder.getContext(), dstShapeAttr, *dstIndexMap, dstThreadMap,
      replicateAttr);

  auto dstIt = layoutMap.find(getDst());
  if (dstIt != layoutMap.end()) {
    auto existingLayout = dyn_cast<LayoutAttr>(dstIt->second);
    if (!existingLayout)
      dstIt->second = dstLayout;
    if (auto layout = dyn_cast<LayoutAttr>(dstIt->second)) {
      if (failed(checkLayoutCompatibility(*this, dstLayout, layout)))
        return failure();
      return success();
    }
    dstIt->second = dstLayout;
    return success(true);
  }

  layoutMap.try_emplace(getDst(), dstLayout);
  return success(true);
} 


} // namespace frisk
} // namespace mlir
