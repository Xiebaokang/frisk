#include "Dialect/Frisk/Utils/LayoutUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "llvm/ADT/SmallDenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;

namespace mlir::frisk {
namespace {

struct SplitInfo {
  int64_t lowerFactor = 1;
  int64_t extent = 1;
};

static bool exprUsesDim(AffineExpr expr, unsigned dim) {
  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>())
    return dimExpr.getPosition() == dim;
  if (expr.isa<AffineSymbolExpr>() || expr.isa<AffineConstantExpr>())
    return false;

  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>())
    return exprUsesDim(bin.getLHS(), dim) || exprUsesDim(bin.getRHS(), dim);
  return false;
}

static std::optional<int64_t> computeSpan(AffineExpr expr, unsigned dim,
                                          int64_t baseSpan) {
  if (auto cst = expr.dyn_cast<AffineConstantExpr>())
    return int64_t(1);
  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>())
    return dimExpr.getPosition() == dim ? baseSpan : int64_t(1);
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>())
    return int64_t(1);

  auto bin = expr.dyn_cast<AffineBinaryOpExpr>();
  if (!bin)
    return std::nullopt;

  auto maybeL = computeSpan(bin.getLHS(), dim, baseSpan);
  auto maybeR = computeSpan(bin.getRHS(), dim, baseSpan);
  if (!maybeL || !maybeR)
    return std::nullopt;

  int64_t lhs = *maybeL;
  int64_t rhs = *maybeR;
  int64_t span = 1;

  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    if (llvm::AddOverflow(lhs, rhs, span))
      return std::nullopt;
    span = std::max<int64_t>(int64_t(1), span - 1);
    return span;
  }
  case AffineExprKind::Mul: {
    if (auto rhsConst = bin.getRHS().dyn_cast<AffineConstantExpr>()) {
      int64_t constVal = rhsConst.getValue();
      if (constVal == std::numeric_limits<int64_t>::min())
        return std::nullopt;
      int64_t factor = std::abs(constVal);
      if (llvm::MulOverflow(lhs, factor, span))
        return std::nullopt;
      return span;
    }
    if (auto lhsConst = bin.getLHS().dyn_cast<AffineConstantExpr>()) {
      int64_t constVal = lhsConst.getValue();
      if (constVal == std::numeric_limits<int64_t>::min())
        return std::nullopt;
      int64_t factor = std::abs(constVal);
      if (llvm::MulOverflow(rhs, factor, span))
        return std::nullopt;
      return span;
    }
    return std::nullopt;
  }
  case AffineExprKind::FloorDiv: {
    auto rhsConst = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst)
      return std::nullopt;
    int64_t divisor = rhsConst.getValue();
    if (divisor <= 0)
      return std::nullopt;
    return llvm::divideCeil(lhs, divisor);
  }
  case AffineExprKind::Mod: {
    auto rhsConst = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst)
      return std::nullopt;
    int64_t modulus = rhsConst.getValue();
    if (modulus <= 0)
      return std::nullopt;
    return std::min<int64_t>(lhs, modulus);
  }
  default:
    break;
  }

  return std::nullopt;
}

static LogicalResult collectSplits(AffineExpr expr, unsigned dim,
                                   int64_t baseExtent, int64_t lowerFactor,
                                   SmallVectorImpl<SplitInfo> &splits) {
  if (!exprUsesDim(expr, dim))
    return success();

  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
    if (dimExpr.getPosition() != dim)
      return success();
    splits.push_back({lowerFactor, baseExtent});
    return success();
  }

  if (expr.isa<AffineConstantExpr>() || expr.isa<AffineSymbolExpr>())
    return success();

  auto bin = expr.dyn_cast<AffineBinaryOpExpr>();
  if (!bin)
    return failure();

  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    if (failed(collectSplits(bin.getLHS(), dim, baseExtent, lowerFactor,
                             splits)))
      return failure();
    return collectSplits(bin.getRHS(), dim, baseExtent, lowerFactor, splits);
  }
  
  case AffineExprKind::Mul: {
    if (bin.getLHS().isa<AffineConstantExpr>())
      return collectSplits(bin.getRHS(), dim, baseExtent, lowerFactor, splits);
    if (bin.getRHS().isa<AffineConstantExpr>())
      return collectSplits(bin.getLHS(), dim, baseExtent, lowerFactor, splits);
    return failure();
  }

  case AffineExprKind::FloorDiv: {
    auto divisor = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (!divisor || divisor.getValue() <= 0)
      return failure();
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return failure();
    int64_t newExtent = llvm::divideCeil(*span, divisor.getValue());
    int64_t newLower = 0;
    if (llvm::MulOverflow(lowerFactor, divisor.getValue(), newLower))
      return failure();
    return collectSplits(bin.getLHS(), dim, newExtent, newLower, splits);
  }

  case AffineExprKind::Mod: {
    auto modulus = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (!modulus || modulus.getValue() <= 0)
      return failure();
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return failure();
    int64_t newExtent = std::min<int64_t>(*span, modulus.getValue());
    return collectSplits(bin.getLHS(), dim, newExtent, lowerFactor, splits);
  }

  default:
    break;
  }

  return failure();
}

} // namespace

std::optional<int64_t>
computeUsedExtentForDim(AffineMapAttr mapAttr, unsigned placeholderPos,
                        int64_t placeholderExtent) {
  if (!mapAttr)
    return int64_t(1);
  if (placeholderExtent <= 0)
    return std::nullopt;

  AffineMap map = mapAttr.getValue();
  if (placeholderPos >= map.getNumDims())
    return std::nullopt;

  SmallVector<SplitInfo, 4> splits;
  for (AffineExpr result : map.getResults()) {
    if (!exprUsesDim(result, placeholderPos))
      continue;
    if (failed(collectSplits(result, placeholderPos, placeholderExtent,
                             /*lowerFactor=*/1, splits)))
      return std::nullopt;
  }

  if (splits.empty())
    return int64_t(1);

  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 8> uniqueSplits;
  int64_t usedExtent = 1;
  for (const auto &split : splits) {
    auto key = std::make_pair(split.lowerFactor, split.extent);
    if (!uniqueSplits.insert(key).second)
      continue;
    if (split.extent <= 0)
      return std::nullopt;
    if (llvm::MulOverflow(usedExtent, split.extent, usedExtent))
      return std::nullopt;
  }

  return usedExtent;
}

} // namespace mlir::frisk
