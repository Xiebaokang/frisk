#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskEnums.h"
// #include "Dialect/Frisk/IR/FriskInterfaces.h"

// #include "Dialect/Frisk/IR/FriskEnums.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"

// move dialect def in this file to make compiler happy
#include "Dialect/Frisk/IR/FriskDialect.cpp.inc"
namespace mlir {
namespace frisk {

} // namespace frisk
} // namespace mlir

namespace mlir {
namespace frisk {

namespace {

struct TargetInfo {
  bool isCuda = false;
  bool isCDNA = false;
  unsigned smVersion = 0;
  unsigned warpSize = 32;
  StringRef rawName;
};

enum class GemmInst { MMA, WGMMA };

static bool isHopper(const TargetInfo &info) {
  return info.isCuda && info.smVersion >= 90;
}

static bool isAmpere(const TargetInfo &info) {
  return info.isCuda && info.smVersion >= 80 && info.smVersion < 90;
}

static std::optional<TargetInfo> detectTargetInfo(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto attr = cur->getAttrOfType<StringAttr>("frisk.target")) {
      TargetInfo info;
      info.rawName = attr.getValue();
      StringRef value = info.rawName;
      if (value.consume_front("sm_") || value.consume_front("sm")) {
        info.isCuda = true;
        unsigned parsed = 0;
        if (!value.getAsInteger(/*Radix=*/10, parsed)) {
          info.smVersion = parsed;
          info.warpSize = 32;
        }
        return info;
      }
      if (value.consume_front("gfx")) {
        info.isCDNA = value.consume_front("9");
        info.warpSize = info.isCDNA ? 64 : 32;
        return info;
      }
      // Fallthrough—treat unknown string as CUDA-like for now.
      info.isCuda = true;
      info.warpSize = 32;
      return info;
    }
    cur = cur->getParentOp();
  }
  return std::nullopt;
}

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

static AffineExpr getDimExpr(unsigned idx, MLIRContext *ctx) {
  return mlir::getAffineDimExpr(idx, ctx);
}

static AffineExpr getConstExpr(int64_t value, MLIRContext *ctx) {
  return mlir::getAffineConstantExpr(value, ctx);
}

static AffineExpr floorDivConst(AffineExpr expr, int64_t divisor,
                                MLIRContext *ctx) {
  return expr.floorDiv(divisor);
}

static AffineExpr modConst(AffineExpr expr, int64_t divisor, MLIRContext *ctx) {
  return expr - getConstExpr(divisor, ctx) * floorDivConst(expr, divisor, ctx);
}

static AffineExpr xor2x2(AffineExpr a, AffineExpr b, MLIRContext *ctx) {
  AffineExpr sum = a + b;
  AffineExpr two = getConstExpr(2, ctx);
  return sum - two * floorDivConst(sum, 2, ctx);
}

static AffineExpr xor4x4(AffineExpr a, AffineExpr b, MLIRContext *ctx) {
  AffineExpr i0 = modConst(a, 2, ctx);
  AffineExpr j0 = modConst(b, 2, ctx);
  AffineExpr i1 = floorDivConst(a, 2, ctx);
  AffineExpr j1 = floorDivConst(b, 2, ctx);
  return getConstExpr(2, ctx) * xor2x2(i1, j1, ctx) + xor2x2(i0, j0, ctx);
}

static AffineExpr xor8x8(AffineExpr a, AffineExpr b, MLIRContext *ctx) {
  AffineExpr i0 = modConst(a, 2, ctx);
  AffineExpr j0 = modConst(b, 2, ctx);
  AffineExpr i1 = floorDivConst(a, 2, ctx);
  AffineExpr j1 = floorDivConst(b, 2, ctx);
  return getConstExpr(2, ctx) * xor4x4(i1, j1, ctx) + xor2x2(i0, j0, ctx);
}

static LayoutAttr makeLayoutAttr(OpBuilder &builder,
                                 ArrayRef<int64_t> shape,
                                 ArrayRef<AffineExpr> results) {
  MLIRContext *ctx = builder.getContext();
  auto shapeAttr = builder.getDenseI64ArrayAttr(shape);
  auto map = AffineMap::get(shape.size(), 0, results, ctx);
  return LayoutAttr::get(ctx, shapeAttr, AffineMapAttr::get(map),
                         AffineMapAttr(), IntegerAttr());
}

struct FragmentExpr {
  MLIRContext *ctx;
  SmallVector<int64_t, 2> shape;
  AffineExpr indexExpr;
  int64_t indexExtent = 1;
  AffineExpr threadExpr;
  int64_t threadExtent = 1;
  int64_t replicateSize = 1;

  FragmentExpr repeat(ArrayRef<int64_t> repeats, bool repeatOnThread,
                      bool lowerDimFirst) const {
    assert(repeats.size() == shape.size());
    FragmentExpr next = *this;
    SmallVector<int64_t, 2> oldShape = shape;
    for (size_t i = 0; i < repeats.size(); ++i)
      next.shape[i] *= repeats[i];

    auto substituteDims = [&](AffineExpr expr) {
      for (size_t i = 0; i < oldShape.size(); ++i) {
        if (oldShape[i] <= 0)
          continue;
        AffineExpr dim = getDimExpr(i, ctx);
        expr = expr.replace(dim, modConst(dim, oldShape[i], ctx));
      }
      return expr;
    };

    AffineExpr localIndex = substituteDims(indexExpr);
    AffineExpr localThread = substituteDims(threadExpr);

    AffineExpr repeatsIndex = getConstExpr(0, ctx);
    AffineExpr repeatStride = getConstExpr(1, ctx);

    auto addContribution = [&](int64_t dimIdx) {
      AffineExpr dim = getDimExpr(dimIdx, ctx);
      if (oldShape[dimIdx] <= 0)
        return;
      AffineExpr quotient = floorDivConst(dim, oldShape[dimIdx], ctx);
      repeatsIndex = repeatsIndex + quotient * repeatStride;
      repeatStride =
          repeatStride * getConstExpr(repeats[dimIdx], ctx);
    };

    if (lowerDimFirst) {
      for (int64_t i = static_cast<int64_t>(oldShape.size()) - 1; i >= 0; --i)
        addContribution(i);
    } else {
      for (size_t i = 0; i < oldShape.size(); ++i)
        addContribution(i);
    }

    int64_t repeatProduct = 1;
    for (int64_t value : repeats)
      repeatProduct *= value;

    if (repeatOnThread) {
      next.threadExpr = localThread +
                        getConstExpr(threadExtent, ctx) * repeatsIndex;
      next.threadExtent *= repeatProduct;
      next.indexExpr = localIndex;
    } else {
      next.threadExpr = localThread;
      next.indexExpr =
          localIndex + getConstExpr(indexExtent, ctx) * repeatsIndex;
      next.indexExtent *= repeatProduct;
    }
    next.replicateSize = replicateSize;
    return next;
  }
};

static FragmentExpr makeFragment8x8(MLIRContext *ctx) {
  FragmentExpr frag;
  frag.ctx = ctx;
  frag.shape = {8, 8};
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  frag.indexExpr = modConst(j, 2, ctx);
  frag.indexExtent = 2;
  frag.threadExpr = floorDivConst(j, 2, ctx) + getConstExpr(4, ctx) * i;
  frag.threadExtent = 32;
  frag.replicateSize = 1;
  return frag;
}

static LayoutAttr makeFragmentLayout(OpBuilder &builder,
                                     const FragmentExpr &frag) {
  MLIRContext *ctx = builder.getContext();
  auto shapeAttr = builder.getDenseI64ArrayAttr(frag.shape);
  SmallVector<AffineExpr, 1> indexResults = {frag.indexExpr};
  SmallVector<AffineExpr, 1> threadResults = {frag.threadExpr};
  auto indexMap =
      AffineMap::get(frag.shape.size(), 0, indexResults, ctx);
  auto threadMap =
      AffineMap::get(frag.shape.size(), 0, threadResults, ctx);
  auto replicateAttr = builder.getI64IntegerAttr(frag.replicateSize);
  return LayoutAttr::get(ctx, shapeAttr, AffineMapAttr::get(indexMap),
                         AffineMapAttr::get(threadMap), replicateAttr);
}

static std::optional<FragmentExpr>
buildAmpereFragmentC(MLIRContext *ctx, int64_t blockM, int64_t blockN,
                     int64_t warpTileM, int64_t warpTileN) {
  if (warpTileM % 16 != 0 || warpTileN % 8 != 0)
    return std::nullopt;
  if (blockM % warpTileM != 0 || blockN % warpTileN != 0)
    return std::nullopt;

  FragmentExpr base = makeFragment8x8(ctx).repeat({2, 1}, /*repeatOnThread=*/false,
                                                  /*lowerDimFirst=*/true);
  int64_t warpRepeatM = std::max<int64_t>(1, blockM / warpTileM);
  int64_t warpRepeatN = std::max<int64_t>(1, blockN / warpTileN);
  FragmentExpr warpLayout =
      base.repeat({warpRepeatM, warpRepeatN}, /*repeatOnThread=*/true,
                  /*lowerDimFirst=*/false);
  int64_t innerRepeatM = std::max<int64_t>(1, warpTileM / 16);
  int64_t innerRepeatN = std::max<int64_t>(1, warpTileN / 8);
  FragmentExpr blockLayout =
      warpLayout.repeat({innerRepeatM, innerRepeatN},
                        /*repeatOnThread=*/false, /*lowerDimFirst=*/false);
  return blockLayout;
}

static std::optional<FragmentExpr>
buildHopperFragmentC(MLIRContext *ctx, int64_t blockM, int64_t blockN,
                     int64_t warpTileM, int64_t warpTileN) {
  if (warpTileM % 16 != 0 || warpTileN % 8 != 0)
    return std::nullopt;
  if (blockM % warpTileM != 0 || blockN % warpTileN != 0)
    return std::nullopt;

  int64_t warpRepeatN = std::max<int64_t>(1, warpTileN / 8);
  FragmentExpr warpLayout =
      makeFragment8x8(ctx).repeat({2, warpRepeatN}, /*repeatOnThread=*/false,
                                  /*lowerDimFirst=*/false);
  FragmentExpr blockLayout =
      warpLayout.repeat({std::max<int64_t>(1, blockM / warpTileM),
                         std::max<int64_t>(1, blockN / warpTileN)},
                        /*repeatOnThread=*/true, /*lowerDimFirst=*/false);
  FragmentExpr finalLayout =
      blockLayout.repeat({std::max<int64_t>(1, warpTileM / 16), 1},
                         /*repeatOnThread=*/false, /*lowerDimFirst=*/false);
  return finalLayout;
}

static LayoutAttr makeLinearLayout(OpBuilder &builder, int64_t rows,
                                   int64_t cols, int64_t linearStride = -1) {
  MLIRContext *ctx = builder.getContext();
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  int64_t stride = (linearStride < 0) ? cols : linearStride;
  AffineExpr expr = i * getConstExpr(stride, ctx) + j;
  SmallVector<AffineExpr, 1> results = {expr};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makePaddedLayout(OpBuilder &builder, int64_t rows,
                                   int64_t cols, int64_t elementBits) {
  int64_t padded = cols;
  if (elementBits > 0 && (elementBits * cols) % 256 == 0)
    padded += 128 / elementBits;
  return makeLinearLayout(builder, rows, cols, padded);
}

static LayoutAttr makeQuarterBankSwizzle(OpBuilder &builder, int64_t rows,
                                         int64_t cols, int64_t elementBits) {
  MLIRContext *ctx = builder.getContext();
  int64_t vectorSize = 128 / elementBits;
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr ts = floorDivConst(i, 8, ctx);
  AffineExpr s = modConst(i, 8, ctx);
  AffineExpr block = floorDivConst(j, vectorSize, ctx);
  AffineExpr tc = floorDivConst(block, 2, ctx);
  AffineExpr c = modConst(block, 2, ctx);
  AffineExpr vec = modConst(j, vectorSize, ctx);
  AffineExpr cSwizzle = xor2x2(c, floorDivConst(s, 4, ctx), ctx);
  AffineExpr index = vec +
                     (cSwizzle + s * getConstExpr(2, ctx)) *
                         getConstExpr(vectorSize, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makeHalfBankSwizzle(OpBuilder &builder, int64_t rows,
                                      int64_t cols, int64_t elementBits) {
  MLIRContext *ctx = builder.getContext();
  int64_t vectorSize = 128 / elementBits;
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr ts = floorDivConst(i, 8, ctx);
  AffineExpr s = modConst(i, 8, ctx);
  AffineExpr block = floorDivConst(j, vectorSize, ctx);
  AffineExpr tc = floorDivConst(block, 4, ctx);
  AffineExpr c = modConst(block, 4, ctx);
  AffineExpr vec = modConst(j, vectorSize, ctx);
  AffineExpr cSwizzle = xor4x4(c, floorDivConst(s, 2, ctx), ctx);
  AffineExpr index = vec +
                     (cSwizzle + s * getConstExpr(4, ctx)) *
                         getConstExpr(vectorSize, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makeFullBankSwizzle(OpBuilder &builder, int64_t rows,
                                      int64_t cols, int64_t elementBits) {
  MLIRContext *ctx = builder.getContext();
  int64_t vectorSize = 128 / elementBits;
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr ts = floorDivConst(i, 8, ctx);
  AffineExpr s = modConst(i, 8, ctx);
  AffineExpr block = floorDivConst(j, vectorSize, ctx);
  AffineExpr tc = floorDivConst(block, 8, ctx);
  AffineExpr c = modConst(block, 8, ctx);
  AffineExpr vec = modConst(j, vectorSize, ctx);
  AffineExpr cSwizzle = xor8x8(c, s, ctx);
  AffineExpr index = vec +
                     (cSwizzle + s * getConstExpr(8, ctx)) *
                         getConstExpr(vectorSize, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr buildAmpereSharedLayout(OpBuilder &builder, MemRefType type,
                                          bool kInner) {
  if (type.getRank() < 2)
    return LayoutAttr();
  ArrayRef<int64_t> shape = type.getShape();
  int64_t stride = shape[shape.size() - 2];
  int64_t cont = shape.back();
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0)
    return LayoutAttr();

  if (elementBits == 64)
    return makePaddedLayout(builder, stride, cont, elementBits);

  int64_t vectorSize = 128 / elementBits;
  if (!kInner && elementBits == 8)
    return makePaddedLayout(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 8) == 0)
    return makeFullBankSwizzle(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 4) == 0)
    return makeHalfBankSwizzle(builder, stride, cont, elementBits);
  return makePaddedLayout(builder, stride, cont, elementBits);
}

static LayoutAttr buildHopperSharedLayout(OpBuilder &builder, MemRefType type,
                                          bool kInner) {
  if (type.getRank() < 2)
    return LayoutAttr();
  ArrayRef<int64_t> shape = type.getShape();
  int64_t stride = shape[shape.size() - 2];
  int64_t cont = shape.back();
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0)
    return LayoutAttr();

  int64_t vectorSize = 128 / elementBits;
  if (cont % (vectorSize * 8) == 0)
    return makeFullBankSwizzle(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 4) == 0)
    return makeHalfBankSwizzle(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 2) == 0)
    return makeQuarterBankSwizzle(builder, stride, cont, elementBits);
  if (cont % vectorSize == 0)
    return makeLinearLayout(builder, stride, cont);
  return LayoutAttr();
}

static GemmInst inferGemmInst(const TargetInfo &info, int64_t blockSize,
                              int64_t M, int64_t N) {
  if (isHopper(info) && blockSize % 128 == 0 && M >= 64 && N >= 64)
    return GemmInst::WGMMA;
  return GemmInst::MMA;
}

static FailureOr<std::array<int64_t, 2>> extractMatrixShape(MemRefType type) {
  if (type.getRank() < 2)
    return failure();
  int64_t rows = type.getDimSize(type.getRank() - 2);
  int64_t cols = type.getDimSize(type.getRank() - 1);
  if (rows < 0 || cols < 0)
    return failure();
  return std::array<int64_t, 2>{rows, cols};
}

static DenseI64ArrayAttr buildShapeAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> dims) {
  return builder.getDenseI64ArrayAttr(dims);
}

static LayoutAttr buildLinearLayoutAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> dims, int64_t stride,
                                        bool transpose) {
  assert(dims.size() == 2 && "expect 2-D layout");
  int64_t rows = dims[0];
  int64_t cols = dims[1];
  if (transpose)
    std::swap(rows, cols);
  DenseI64ArrayAttr shapeAttr = buildShapeAttr(builder, {rows, cols});

  MLIRContext *ctx = builder.getContext();
  AffineExpr row = builder.getAffineDimExpr(0);
  AffineExpr col = builder.getAffineDimExpr(1);
  AffineExpr expr = row * builder.getAffineConstantExpr(stride) + col;
  auto indexMap = AffineMapAttr::get(AffineMap::get(2, 0, expr, ctx));

  return LayoutAttr::get(ctx, shapeAttr, indexMap, AffineMapAttr(),
                         IntegerAttr());
}

static LayoutAttr buildSharedLayoutForType(OpBuilder &builder, MemRefType type,
                                           bool transpose) {
  auto maybeShape = extractMatrixShape(type);
  if (failed(maybeShape))
    return LayoutAttr();
  auto dims = *maybeShape;
  int64_t elementBits = type.getElementTypeBitWidth();
  int64_t cols = transpose ? dims[0] : dims[1];
  int64_t padded = cols;
  if (elementBits > 0 && (elementBits * cols) % 256 == 0)
    padded += 128 / elementBits;
  return buildLinearLayoutAttr(builder, {dims[0], dims[1]}, padded, transpose);
}

static LayoutAttr buildFragmentLayoutAttr(OpBuilder &builder,
                                          MemRefType type,
                                          int64_t tileRows, int64_t tileCols,
                                          unsigned warpSize) {
  auto maybeShape = extractMatrixShape(type);
  if (failed(maybeShape))
    return LayoutAttr();
  auto dims = *maybeShape;
  DenseI64ArrayAttr shapeAttr = buildShapeAttr(builder, {dims[0], dims[1]});
  MLIRContext *ctx = builder.getContext();

  AffineExpr row = builder.getAffineDimExpr(0);
  AffineExpr col = builder.getAffineDimExpr(1);
  AffineExpr index =
      row * builder.getAffineConstantExpr(dims[1]) + col;
  auto indexMap = AffineMapAttr::get(AffineMap::get(2, 0, index, ctx));

  int64_t laneM = std::max<int64_t>(1, std::min<int64_t>(tileRows, 16));
  int64_t laneN = std::max<int64_t>(1, std::min<int64_t>(tileCols, 8));
  // Clamp to warp size to avoid producing values outside of lane bounds.
  if (laneM * laneN > static_cast<int64_t>(warpSize)) {
    laneN = std::max<int64_t>(1, warpSize / laneM);
  }
  AffineExpr lane =
      (row % laneM) * builder.getAffineConstantExpr(laneN) + (col % laneN);
  auto threadMap = AffineMapAttr::get(AffineMap::get(2, 0, lane, ctx));
  auto replicate = builder.getI64IntegerAttr(1);
  return LayoutAttr::get(ctx, shapeAttr, indexMap, threadMap, replicate);
}

static std::pair<int64_t, int64_t>
computeWarpPartition(attr::GemmWarpPolicy policy, int64_t M, int64_t N,
                     int64_t blockSize, const TargetInfo &target) {
  int64_t warpSize = target.warpSize;
  int64_t numWarps = warpSize ? blockSize / warpSize : 1;
  int64_t kMPerWarp = 16;
  int64_t kNPerWarp = target.isCuda && target.smVersion >= 70 ? 8 : 16;
  if (M == 0 || N == 0 || numWarps == 0)
    return {1, 1};

  int64_t mWarp = 1;
  int64_t nWarp = numWarps;
  switch (policy) {
  case attr::GemmWarpPolicy::FullRow: {
    mWarp = numWarps;
    if (M % (mWarp * kMPerWarp) != 0) {
      int64_t maxMWarp = std::max<int64_t>(1, M / kMPerWarp);
      mWarp = std::max<int64_t>(1, std::min<int64_t>(numWarps, maxMWarp));
      nWarp = std::max<int64_t>(1, numWarps / mWarp);
    } else {
      nWarp = std::max<int64_t>(1, numWarps / mWarp);
    }
    break;
  }
  case attr::GemmWarpPolicy::FullCol: {
    nWarp = numWarps;
    if (N % (nWarp * kNPerWarp) != 0) {
      int64_t maxNWarp = std::max<int64_t>(1, N / kNPerWarp);
      nWarp = std::max<int64_t>(1, std::min<int64_t>(numWarps, maxNWarp));
      mWarp = std::max<int64_t>(1, numWarps / nWarp);
    } else {
      mWarp = std::max<int64_t>(1, numWarps / nWarp);
    }
    break;
  }
  case attr::GemmWarpPolicy::Square:
  default: {
    float idealRatio = N > 0 ? static_cast<float>(M) / static_cast<float>(N)
                             : 1.0f;
    float bestScore = std::numeric_limits<float>::max();
    for (int64_t m = 1; m <= numWarps; ++m) {
      if (numWarps % m != 0)
        continue;
      int64_t n = numWarps / m;
      float mPerWarp = static_cast<float>(M) / (m * kMPerWarp);
      float nPerWarp = static_cast<float>(N) / (n * kNPerWarp);
      if (mPerWarp < 1 || nPerWarp < 1)
        continue;
      float score = std::abs(mPerWarp / nPerWarp - idealRatio);
      if (score < bestScore) {
        bestScore = score;
        mWarp = m;
        nWarp = n;
      }
    }
    break;
  }
  }
  if (mWarp <= 0)
    mWarp = 1;
  if (nWarp <= 0)
    nWarp = 1;
  return {mWarp, nWarp};
}

} // namespace

//===----------------------------------------------------------------------===//
// -- KernelOp --
//===----------------------------------------------------------------------===//
void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name, FunctionType type) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addRegion();
}

LogicalResult KernelOp::verify() {
  auto typeAttr = getFunctionTypeAttr();
  if (!typeAttr)
    return emitOpError("requires a 'function_type' attribute");

  auto functionType = llvm::dyn_cast<FunctionType>(typeAttr.getValue());
  if (!functionType)
    return emitOpError("requires a function type");

  // 验证区域参数
  if (getBody(0)->getNumArguments() != functionType.getNumInputs())
    return emitOpError("region argument count does not match function type");

  for (unsigned i = 0; i < getBody(0)->getNumArguments(); ++i) {
    if (getBody(0)->getArgument(i).getType() != functionType.getInput(i))
      return emitOpError("region argument type mismatch");
  }

  return success();
}

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析符号名称
  StringAttr symName;
  if (parser.parseSymbolName(symName, "sym_name", result.attributes))
    return failure();
  // 解析参数列表
  SmallVector<Type> argTypes;
  if (parser.parseLParen() || parser.parseTypeList(argTypes) || parser.parseRParen())
    return failure();
  // 解析结果类型 - 可选，如果没有结果就是空
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen() || parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  }
  // 创建函数类型属性
  auto functionType = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(functionType));
  // 解析区域
  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> args;
  for (Type argType : argTypes) {
    args.emplace_back();
    args.back().type = argType;
  }
  if (parser.parseRegion(*body, args) || 
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void KernelOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymName());
  // 打印参数类型
  auto funcType = getFunctionType();
  auto funcTy = dyn_cast<FunctionType>(funcType);
  p << "(";
  llvm::interleaveComma(funcTy.getInputs(), p);
  p << ")";
  // 只有当有结果时才打印结果类型
  if (!funcTy.getResults().empty()) {
    p << " -> (";
    llvm::interleaveComma(funcTy.getResults(), p);
    p << ")";
  }
  // 打印区域
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
  // 打印属性
  p.printOptionalAttrDict((*this)->getAttrs(), {"sym_name", "function_type"});
}

//===----------------------------------------------------------------------===//
// -- ParallelOp --
//===----------------------------------------------------------------------===//
void ParallelOp::build(OpBuilder &builder, OperationState &state,
                      llvm::ArrayRef<int64_t> ranges, int64_t thread_num) {
  state.addAttribute("threads", builder.getI64IntegerAttr(thread_num));
  state.addAttribute("ranges", builder.getDenseI64ArrayAttr(ranges));
  state.addRegion();
  // Region *region = state.regions[0].get();
  // Block *entry = new Block();
  // region->push_back(entry);
  // for (unsigned i=0; i<ranges.size(); ++i) {
  //   entry->addArgument(builder.getIndexType(), state.location);
  // }
}

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析迭代变量列表: (%arg1, %arg2)
  SmallVector<OpAsmParser::Argument, 4> inductionVars;
  if (parser.parseArgumentList(inductionVars, AsmParser::Delimiter::Paren))
    return failure();
  // 解析等号和范围: = (8, 8)
  if (parser.parseEqual() || parser.parseLParen())
    return failure();
  SmallVector<int64_t> ranges;
  if (parser.parseCommaSeparatedList([&]() {
        int64_t range;
        if (parser.parseInteger(range))
          return failure();
        ranges.push_back(range);
        return success();
      }) || parser.parseRParen())
    return failure();
  // 解析线程数量: , threads=128
  int64_t threads = 0;
  if (parser.parseComma() || parser.parseKeyword("threads") || 
      parser.parseEqual() || parser.parseInteger(threads))
    return failure();
  // 添加属性
  result.addAttribute("ranges", parser.getBuilder().getDenseI64ArrayAttr(ranges));
  result.addAttribute("threads", parser.getBuilder().getI64IntegerAttr(threads));
  // 解析区域
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVars))
    return failure();
  return success();
}

void ParallelOp::print(OpAsmPrinter &p) {
  // 打印迭代变量: (%arg1, %arg2)
  p << " (";
  llvm::interleaveComma(getBody(0)->getArguments(), p, [&](BlockArgument arg) {
    p << arg;
  });
  p << ")";
  // 打印范围和线程配置: = (8, 8), threads=128
  p << " = (";
  auto grid = getGrid();
  llvm::interleaveComma(grid, p);
  p << "), threads = " << getThreadNum();
  // 打印区域（不打印终止符）
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// -- BlockOp --
//===----------------------------------------------------------------------===//
void BlockOp::build(OpBuilder &builder, OperationState &state, ArrayRef<int64_t> ranges, BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  
  state.addAttribute("ranges", builder.getDenseI64ArrayAttr(ranges));

  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  llvm::SmallVector<Value> inductionVars;
  for (int64_t range : ranges) {
    inductionVars.push_back(bodyBlock->addArgument(builder.getIndexType(), state.location));
  }
  if (!bodyBuilder) {
    ensureTerminator(*bodyRegion, builder, state.location);
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(inductionVars);
    builder.create<EndOp>(state.location);
  }
}

ParseResult BlockOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析参数列表: (%arg3, %arg4)
  SmallVector<OpAsmParser::Argument, 4> blockArgs;
  if (parser.parseArgumentList(blockArgs, OpAsmParser::Delimiter::Paren))  // 解析 "to" 关键字
    return failure();
  if (parser.parseKeyword("to"))  // 解析边界列表: (128, 128)
    return failure();
  SmallVector<int64_t, 4> ranges;
  if (parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
          int64_t range;
          if (parser.parseInteger(range))
            return failure();
          ranges.push_back(range);
          return success();
        }))
    return failure();
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs))  // 解析区域 { ... }
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))  // 解析可选的属性字典
    return failure();
  if (!ranges.empty()) {  // 将边界添加到属性中
    result.addAttribute("ranges", parser.getBuilder().getDenseI64ArrayAttr(ranges));
  }
  return success();
}

void BlockOp::print(OpAsmPrinter &p) {
  // 打印迭代变量: (%arg1, %arg2)
  p << " (";
  llvm::interleaveComma(getBody(0)->getArguments(), p, [&](BlockArgument arg) {
    p << arg;
  });
  p << ")";
  // 打印范围和线程配置: = (8, 8), threads=128
  p << " to (";
  auto ranges = getBlockRanges();
  llvm::interleaveComma(ranges, p);
  p << ")";
  // 打印区域（不打印终止符）
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

LogicalResult BlockOp::verify() {   // 暂时关闭block op的验证方法
  // auto &blockBody = getRegion();
  // if (blockBody.empty()) {
  //   return emitOpError("block must have a body");
  // }
  // auto &block = blockBody.front();
  // bool hasAffineLoadOp = false;
  // bool hasAffineStoreOp = false;
  
  // for (auto &op : block.getOperations()) {
  //   if (isa<affine::AffineLoadOp>(op)) {    // 检查是否是affine.load操作
  //     hasAffineLoadOp = true;
  //   }
  //   else if (isa<affine::AffineStoreOp>(op)) {    // 检查是否是affine.store操作  
  //     hasAffineStoreOp = true;
  //   }
  //   // 如果已经找到两种操作，可以提前退出
  //   if (hasAffineLoadOp && hasAffineStoreOp) {
  //     break;
  //   }
  // }
  // // 验证结果
  // if (!hasAffineLoadOp) {
  //   return emitOpError("block must contain at least one affine.load operation");
  // }
  // if (!hasAffineStoreOp) {
  //   return emitOpError("block must contain at least one affine.store operation");
  // }
  return success();
}

//===----------------------------------------------------------------------===//
// -- ForOp --
//===----------------------------------------------------------------------===//
void ForOp::build(OpBuilder &builder, OperationState &state, 
                  int64_t lowerBound, int64_t upperBound, int64_t step, 
                  BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  state.addAttribute("lower", builder.getI64IntegerAttr(lowerBound));
  state.addAttribute("upper", builder.getI64IntegerAttr(upperBound));
  state.addAttribute("step", builder.getI64IntegerAttr(step));

  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  auto iv = bodyBlock->addArgument(builder.getIndexType(), state.location);
  if (!bodyBuilder) {
    ensureTerminator(*bodyRegion, builder, state.location);
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(iv);
    builder.create<EndOp>(state.location);
  }
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析格式：frisk.for %arg1 = 0 to 1024 step = 32 { ... }
  OpAsmParser::UnresolvedOperand inductionVar;
  IntegerAttr lowerAttr, upperAttr, stepAttr;

  if (parser.parseOperand(inductionVar) || parser.parseEqual())  // 解析循环变量 "%arg1 ="
    return failure();
  auto builder = parser.getBuilder();
  if (parser.parseAttribute(lowerAttr, builder.getIntegerType(64), "lower", result.attributes))  // 解析下界
    return failure();
  if (parser.parseKeyword("to"))  // 解析 "to"
    return failure();
  if (parser.parseAttribute(upperAttr, builder.getIntegerType(64), "upper", result.attributes))  // 解析上界
    return failure();
  if (parser.parseKeyword("step") || parser.parseEqual())  // 解析 "step ="
    return failure();
  if (parser.parseAttribute(stepAttr, builder.getIntegerType(64), "step", result.attributes))  // 解析步长
    return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();  // 解析区域
  llvm::SmallVector<OpAsmParser::Argument, 4> regionArgs;  // 解析区域参数类型（循环变量）
  llvm::SmallVector<Type, 4> regionArgTypes;

  regionArgTypes.push_back(builder.getIndexType());  // 循环变量类型为index
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren, true) || parser.parseRegion(*region, regionArgs))
    return failure();
  result.addRegion(std::move(region));
  return success();
}

void ForOp::print(OpAsmPrinter &p) {
  p << " ";
  p << getInductionVar() << " = " << getLower();  // 打印循环变量 //打印下界
  p << " to " << getUpper(); // 打印上界
  if (getStep() > 1) {
    p << " step = " << getStep();  // 打印步长
  }
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// -- GemmOp --
//===----------------------------------------------------------------------===//
LogicalResult GemmOp::verify() {
  auto AType = dyn_cast<MemRefType>(getA().getType());
  auto BType = dyn_cast<MemRefType>(getB().getType());
  auto CType = dyn_cast<MemRefType>(getC().getType());
  // 基础类型检查
  if (AType.getElementType() != BType.getElementType() ||
      AType.getElementType() != CType.getElementType()) {
      return emitOpError("all operands must have the same element type");
  }
  // 维度兼容性检查
  if (AType.getDimSize(1) != BType.getDimSize(0)) {
      return emitOpError("A columns must equal B rows for matrix multiplication");
  }
  // 输出维度检查
  if (AType.getDimSize(0) != CType.getDimSize(0) ||
      BType.getDimSize(1) != CType.getDimSize(1)) {
      return emitOpError("C dimensions must match matrix multiplication result");
  }
  return success();
}

LogicalResult GemmOp::inferLayout(OpBuilder &builder,
                                  DenseMap<Value, Attribute> &layoutMap) {
  Operation *op = getOperation();
  auto targetInfo = detectTargetInfo(op);
  if (!targetInfo)
    return emitOpError()
           << "layout inference requires a 'frisk.target' string attribute";
  auto blockSize = inferThreadBlockSize(op);
  if (!blockSize)
    return emitOpError()
           << "layout inference requires an enclosing 'frisk.parallel' "
              "operation to provide a thread count";

  auto memA = dyn_cast<MemRefType>(getA().getType());
  auto memB = dyn_cast<MemRefType>(getB().getType());
  auto memC = dyn_cast<MemRefType>(getC().getType());
  if (!memA || !memB || !memC)
    return emitOpError("all operands must be memref values for layout inference");

  auto requireMemorySpace = [&](MemRefType type, attr::MemorySpace expected,
                                StringRef label) -> LogicalResult {
    unsigned memSpace = type.getMemorySpaceAsInt();
    unsigned expectedValue = static_cast<unsigned>(expected);
    if (memSpace != expectedValue) {
      return emitOpError()
             << "operand " << label
             << " must reside in memory space " << expectedValue;
    }
    return success();
  };

  if (failed(requireMemorySpace(memA, attr::MemorySpace::Shared, "A")))
    return failure();
  if (failed(requireMemorySpace(memB, attr::MemorySpace::Shared, "B")))
    return failure();
  if (failed(requireMemorySpace(memC, attr::MemorySpace::Local, "C")))
    return failure();

  bool hopper = isHopper(*targetInfo);
  bool ampere = isAmpere(*targetInfo);
  if (!hopper && !ampere)
    return emitOpError(
        "layout inference currently supports only Ampere or Hopper targets");

  LayoutAttr layoutA = hopper ? buildHopperSharedLayout(builder, memA, !getTransA())
                              : buildAmpereSharedLayout(builder, memA, !getTransA());
  LayoutAttr layoutB = hopper ? buildHopperSharedLayout(builder, memB, getTransB())
                              : buildAmpereSharedLayout(builder, memB, getTransB());
  if (!layoutA || !layoutB)
    return emitOpError("unable to materialize shared-memory swizzled layouts");

  auto warpPartition =
      computeWarpPartition(getPolicyAttr().getValue(), getM(), getN(),
                           *blockSize, *targetInfo);
  int64_t warpCountM = std::max<int64_t>(warpPartition.first, 1);
  int64_t warpCountN = std::max<int64_t>(warpPartition.second, 1);
  int64_t warpTileM = getM() / warpCountM;
  int64_t warpTileN = getN() / warpCountN;
  GemmInst inst = inferGemmInst(*targetInfo, *blockSize, getM(), getN());

  std::optional<FragmentExpr> fragment;
  if (hopper && inst == GemmInst::WGMMA)
    fragment = buildHopperFragmentC(builder.getContext(), getM(), getN(),
                                    warpTileM, warpTileN);
  else
    fragment = buildAmpereFragmentC(builder.getContext(), getM(), getN(),
                                    warpTileM, warpTileN);
  if (!fragment)
    return emitOpError("unable to build accumulator fragment for current warp "
                       "shape; check warp policy and target");
  LayoutAttr layoutC = makeFragmentLayout(builder, *fragment);

  bool updated = false;
  updated |= layoutMap.try_emplace(getA(), layoutA).second;
  updated |= layoutMap.try_emplace(getB(), layoutB).second;
  updated |= layoutMap.try_emplace(getC(), layoutC).second;
  return success(updated);
}
//===----------------------------------------------------------------------===//
// -- AllocBufferOp --
//===----------------------------------------------------------------------===//
// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType) {
//   build(builder, state, shape, elementType, /*alignment=*/0, /*memorySpace=*/0);
// }

// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType, 
//                           int64_t alignment) {
//   build(builder, state, shape, elementType, alignment, /*memorySpace=*/0);
// }

// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType,
//                           int64_t alignment, int64_t memorySpace) {
//   // 创建 memref 类型
//   auto memrefType = MemRefType::get(shape, elementType, /*layout=*/{}, memorySpace);
//   // 添加属性
//   state.addAttribute("shape", builder.getDenseI64ArrayAttr(shape));
//   state.addAttribute("elementType", TypeAttr::get(elementType));
//   state.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
//   state.addAttribute("memorySpace", builder.getI64IntegerAttr(memorySpace));
//   // 添加结果类型
//   state.addTypes(memrefType);
// }

LogicalResult AllocBufferOp::verify() {
  auto resultType = getResult().getType();
  
  // 检查结果类型是否是 memref
  if (!isa<MemRefType>(resultType)) {
    return emitOpError("result must be a memref type");
  }
  
  auto memrefType = cast<MemRefType>(resultType);
  // 检查属性与结果类型是否一致
  auto attrShape = getShape();
  auto attrElementType = getElementType();
  auto attrMemorySpace = getMemorySpace();
  if (memrefType.getShape() != attrShape) {
    return emitOpError("shape attribute must match result memref shape");
  }
  if (memrefType.getElementType() != attrElementType) {
    return emitOpError("elementType attribute must match result memref element type");
  }

  unsigned resultMemorySpace = memrefType.getMemorySpaceAsInt();
  if (resultMemorySpace != static_cast<unsigned>(attrMemorySpace)) {
    return emitOpError("memorySpace attribute must match result memref memory space");
  }
  // 检查对齐值是否有效
  int64_t alignment = getAlignment();
  if (alignment < 0) {
    return emitOpError("alignment must be non-negative");
  }
  // 检查对齐值是否是 2 的幂（可选，但推荐）
  if (alignment > 0 && (alignment & (alignment - 1)) != 0) {
    return emitOpError("alignment must be a power of 2");
  }
  // 检查 memorySpace 是否有效
  switch (attrMemorySpace) {
    case ::mlir::frisk::attr::MemorySpace::Local:
      break;
    case ::mlir::frisk::attr::MemorySpace::Global:
      break;
    case ::mlir::frisk::attr::MemorySpace::Shared:
      break;
    default:
      return emitOpError("memorySpace must be Local, Global, or Shared");
  }
  return success();
}

ParseResult AllocBufferOp::parse(OpAsmParser &parser, OperationState &result) {
  // 直接解析属性字典
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // 解析结果类型: -> memref<32x128xf32>
  if (parser.parseArrow())
    return failure();
  Type resultType;
  if (parser.parseType(resultType))
    return failure();
  // 检查结果类型是否是 memref
  if (!isa<MemRefType>(resultType)) {
    return parser.emitError(parser.getCurrentLocation(), "expected memref type for result");
  }
  auto memrefType = cast<MemRefType>(resultType);
  // 从结果类型提取 shape 和 elementType
  auto shape = memrefType.getShape();
  auto elementType = memrefType.getElementType();
  auto memorySpace = memrefType.getMemorySpace();
  // 添加结果类型
  result.addTypes(resultType);
  // 添加必要的属性（如果属性字典中没有）
  auto &builder = parser.getBuilder();
  if (!result.attributes.get("shape")) {
    result.addAttribute("shape", builder.getDenseI64ArrayAttr(shape));
  }
  if (!result.attributes.get("elementType")) {
    result.addAttribute("elementType", TypeAttr::get(elementType));
  }
  if (!result.attributes.get("memorySpace")) {
    result.addAttribute("memorySpace", memorySpace);
  }
  return success();
}

// 自定义汇编格式打印
void AllocBufferOp::print(OpAsmPrinter &p) {
  // 打印属性字典
  p << " {";
  auto memorySpace = getMemorySpace();
  p << "scope = \"";
  switch (memorySpace) {
    case ::mlir::frisk::attr::MemorySpace::Local:
      p << "local";
      break;
    case ::mlir::frisk::attr::MemorySpace::Global:
      p << "global";
      break;
    case ::mlir::frisk::attr::MemorySpace::Shared:
      p << "shared";
      break;
    default:
      p << "unknown";
      break;
  }
  p << "\"";
  // 打印 alignment（如果不是默认值）
  if (getAlignment() != 0)
    p << ", alignment = " << getAlignment();
  p << "}";
  // 打印结果类型
  p << " -> " << getResult().getType();
}


//===----------------------------------------------------------------------===//
// -- CopyOp --
//===----------------------------------------------------------------------===//
void CopyOp::build(OpBuilder &builder, OperationState &state, 
                  Value srcMemref, Value dstMemref, ValueRange srcIndices, ValueRange dstIndices) {
  auto srcMemrefType = llvm::cast<MemRefType>(srcMemref.getType());
  auto dstMemrefType = llvm::cast<MemRefType>(dstMemref.getType());
  int64_t srcRank = srcMemrefType.getRank();
  int64_t dstRank = dstMemrefType.getRank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto srcMap = srcRank ? builder.getMultiDimIdentityMap(srcRank) : builder.getEmptyAffineMap();
  auto dstMap = dstRank ? builder.getMultiDimIdentityMap(dstRank) : builder.getEmptyAffineMap();
  build(builder, state, srcMemref, dstMemref, srcMap, dstMap, srcIndices, dstIndices);
}

void CopyOp::build(OpBuilder &builder, OperationState &state, 
                    Value srcMemref, Value dstMemref, 
                    AffineMap srcMap, AffineMap dstMap,
                    ValueRange srcIndices, ValueRange dstIndices) {
  assert(srcMap.getNumInputs() == srcIndices.size() && 
    "source map inputs must match source indices count");
  assert(dstMap.getNumInputs() == dstIndices.size() && 
    "destination map inputs must match destination indices count");

  auto srcMemrefType = llvm::cast<MemRefType>(srcMemref.getType());
  auto dstMemrefType = llvm::cast<MemRefType>(dstMemref.getType());
  int64_t srcRank = srcMemrefType.getRank();
  int64_t dstRank = dstMemrefType.getRank();
  std::vector<int64_t> srcExtents;
  auto dstExtents = dstMemrefType.getShape(); 

  assert(srcRank >= dstRank && "src rank msut >= dst rank");
  for (unsigned i=0; i<srcRank; i++) {
    if (i < srcRank - dstRank) {
      srcExtents.push_back(1);
    } else {
      srcExtents.push_back(dstExtents[i-(srcRank-dstRank)]);
    }
  }

  build(builder, state, srcMemref, dstMemref, srcMap, dstMap, srcIndices, dstIndices, 
    builder.getDenseI64ArrayAttr(srcExtents), builder.getDenseI64ArrayAttr(dstExtents));
}

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();
  // 解析源操作数和索引
  OpAsmParser::UnresolvedOperand srcMemrefInfo;
  AffineMapAttr srcMapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcMapOperands;
  if (parser.parseOperand(srcMemrefInfo) || parser.parseLSquare() ||
      parser.parseAffineMapOfSSAIds(srcMapOperands, srcMapAttr, "srcMap", result.attributes) ||
      parser.parseRSquare() || parser.parseComma())
    return failure();
  // 解析目标操作数和索引
  OpAsmParser::UnresolvedOperand dstMemrefInfo;
  AffineMapAttr dstMapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dstMapOperands;
  if (parser.parseOperand(dstMemrefInfo) || parser.parseLSquare() ||
      parser.parseAffineMapOfSSAIds(dstMapOperands, dstMapAttr, "dstMap", result.attributes) ||
      parser.parseRSquare())
    return failure();
  // 解析类型信息
  SmallVector<Type, 2> memrefTypes;
  if (parser.parseColonTypeList(memrefTypes))
    return failure();
  if (memrefTypes.size() != 2)
    return parser.emitError(parser.getNameLoc(), "expected two memref types");
  auto srcType = dyn_cast<MemRefType>(memrefTypes[0]);
  auto dstType = dyn_cast<MemRefType>(memrefTypes[1]);
  if (!srcType || !dstType)
    return parser.emitError(parser.getNameLoc(), "expected memref types");
  // 解析可选的属性字典，但要排除 operandSegmentSizes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // 解析操作数
  if (parser.resolveOperand(srcMemrefInfo, srcType, result.operands) ||
      parser.resolveOperand(dstMemrefInfo, dstType, result.operands) ||
      parser.resolveOperands(srcMapOperands, indexTy, result.operands) ||
      parser.resolveOperands(dstMapOperands, indexTy, result.operands))
    return failure();
  // 设置 operandSegmentSizes 属性（隐藏的）
  SmallVector<int32_t> segmentSizes = {
      1, 1, 
      static_cast<int32_t>(srcMapOperands.size()),  // srcIndices
      static_cast<int32_t>(dstMapOperands.size())   // dstIndices
  };
  result.addAttribute(CopyOp::getOperandSegmentSizesAttrName(result.name), 
    builder.getDenseI32ArrayAttr(segmentSizes));
  return success();
}

void CopyOp::print(OpAsmPrinter &p) {
  p << " " << getSrcMemRef() << "[";
  // 打印源映射和操作数
  if (AffineMapAttr srcMapAttr = (*this)->getAttrOfType<AffineMapAttr>("srcMap")) {
    p.printAffineMapOfSSAIds(srcMapAttr, getSrcIndices());
  }
  p << "], " << getDstMemRef() << "[";
  // 打印目标映射和操作数
  if (AffineMapAttr dstMapAttr = (*this)->getAttrOfType<AffineMapAttr>("dstMap")) {
    p.printAffineMapOfSSAIds(dstMapAttr, getDstIndices());
  }
  auto srcExtentsAttr = getSrcExtentsAttr();
  auto dstExtentsAttr = getDstExtentsAttr();
  p << "] {src_extents = [";
  llvm::interleaveComma(srcExtentsAttr.asArrayRef(), p);
  p << "], dst_extents = [";
  llvm::interleaveComma(dstExtentsAttr.asArrayRef(), p);
  p << "]} ";
  
  // 打印属性字典，但要排除已打印的映射属性和 operandSegmentSizes
  SmallVector<StringRef> elidedAttrs = { "srcMap", "dstMap", "srcExtents", "dstExtents",
      CopyOp::getOperandSegmentSizesAttrName((*this)->getName()).getValue()
  };
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  p << " : " << getSrcMemRef().getType() << ", " << getDstMemRef().getType();
}

LogicalResult CopyOp::verify() {
  AffineMap srcMap = getSrcMap();
  AffineMap dstMap = getDstMap();
  // 正确：srcIndices 长度必须等于 srcMap 输入维度
  if (getSrcIndices().size() != srcMap.getNumInputs()) {
    return emitOpError("expected ") << srcMap.getNumInputs()
           << " source indices, but got " << getSrcIndices().size();
  }
  // 正确：dstIndices 长度必须等于 dstMap 输入维度
  if (getDstIndices().size() != dstMap.getNumInputs()) {
    return emitOpError("expected ") << dstMap.getNumInputs()
           << " destination indices, but got " << getDstIndices().size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// -- FillOp --
//===----------------------------------------------------------------------===//
LogicalResult FillOp::verify() {
  auto memrefType = dyn_cast<MemRefType>(getMemref().getType());
  auto elemType = memrefType.getElementType();
  auto valueAttr = getValueAttr();
  if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    Type valueType = floatAttr.getType();
    if (elemType != valueType) {
      return emitOpError("value type ") << valueType << " does not match memref element type " << elemType;
    }
  } else {
    return emitOpError("fill value must be a float attribute");
  }
  return success();
}

// ParseResult FillOp::parse(OpAsmParser &parser, OperationState &result) {
//   OpAsmParser::UnresolvedOperand dstOperand;
//   Attribute valueAttr;
//   Type dstType;
//   if (parser.parseOperand(dstOperand) || parser.parseComma() ||
//       parser.parseAttribute(valueAttr, "value", result.attributes))// 解析目标操作数
//     return failure();
//   if (parser.parseColonType(dstType))// 解析类型
//     return failure();
//   if (parser.resolveOperand(dstOperand, dstType, result.operands))// 解析操作数
//     return failure();
//   return success();
// }

// void FillOp::print(OpAsmPrinter &p) {
//   p << " " << getMemref() << ", value = " << getValueAttr();
//   p << " : " << getMemref().getType();
// }

//===----------------------------------------------------------------------===//
// -- ReduceOp --
//===----------------------------------------------------------------------===//
void ReduceOp::build(OpBuilder &builder, OperationState &state, 
                    Value src, Value dst, StringRef kind, int64_t dim, bool clear) {
  state.addOperands({src, dst});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("dim", builder.getI64IntegerAttr(dim));
  state.addAttribute("clear", builder.getBoolAttr(clear));
}

LogicalResult ReduceOp::verify() {
  auto srcType = getSrcType();
  auto dstType = getDstType();
  auto kind = getKind();
  auto dim = getDim();
  if (!isa<MemRefType>(srcType) || !isa<MemRefType>(dstType)) {  // 检查源和目标是否是 memref 类型
    return emitOpError("source and destination must be memref types");
  }
  if (dim < 0 || dim >= srcType.getRank()) {  // 检查维度有效性
    return emitOpError("dimension ") << dim << " is out of range for source of rank " << srcType.getRank();
  }
  if (kind != "add" && kind != "mul" && kind != "min" && kind != "max") {  // 检查 reduce 类型有效性
    return emitOpError("unsupported reduce kind: ") << kind << ". Supported: add, mul, min, max";
  }
  // 检查源和目标形状兼容性
  auto srcShape = srcType.getShape();
  auto dstShape = dstType.getShape();
  if (srcType.getRank() != dstType.getRank() + 1) {  // 目标 rank 应该比源 rank 少 1
    return emitOpError("destination rank must be one less than source rank. Got: ")
           << dstType.getRank() << " vs " << srcType.getRank();
  }
  for (int64_t i = 0, j = 0; i < srcType.getRank(); ++i) {  // 检查非归约维度是否匹配
    if (i == dim) continue; // 跳过归约维度
    if (srcShape[i] != dstShape[j]) {
      return emitOpError("non-reduced dimension mismatch: source[")
             << i << "] = " << srcShape[i] << " vs destination["
             << j << "] = " << dstShape[j];
    }
    j++;
  }
  if (srcType.getElementType() != dstType.getElementType()) {  // 检查元素类型兼容性
    return emitOpError("source and destination must have the same element type");
  }
  return success();
}

// // 自定义汇编格式解析
// ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
//   OpAsmParser::UnresolvedOperand srcOperand, dstOperand;
//   Type srcType, dstType;
//   // 解析操作数: %src, %dst
//   if (parser.parseOperand(srcOperand) || parser.parseComma() || parser.parseOperand(dstOperand))
//     return failure();
//   if (parser.parseOptionalAttrDict(result.attributes))// 解析属性字典: {dim=0, clear=1, kind="max"}
//     return failure();
//   // 解析类型: : memref<1xf16>, memref<1024xf16>
//   if (parser.parseColon() || parser.parseType(srcType) || parser.parseComma() || parser.parseType(dstType))
//     return failure();
//   // 解析操作数
//   if (parser.resolveOperand(srcOperand, srcType, result.operands) || 
//       parser.resolveOperand(dstOperand, dstType, result.operands))
//     return failure();
//   return success();
// }

// // 自定义汇编格式打印
// void ReduceOp::print(OpAsmPrinter &p) {
//   p << " " << getSrc() << ", " << getDst();
//   // 打印属性字典
//   p << " {";
//   p << "dim = " << getDim();
//   p << ", clear = " << getClear();
//   p << ", kind = \"" << getKind() << "\"";
//   p << "}";
//   p << " : " << getSrc().getType() << ", " << getDst().getType();
// }



} // namespace frisk
} // namespace mlir
