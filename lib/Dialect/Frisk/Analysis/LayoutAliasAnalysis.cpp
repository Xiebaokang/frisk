#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::frisk;

namespace {
constexpr uint64_t kPointLimit = 65536;
constexpr unsigned kExprLimit = 4096;
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

FailureOr<int64_t> narrow(__int128 value) {
  if (value < std::numeric_limits<int64_t>::min() || value > kMax)
    return failure();
  return static_cast<int64_t>(value);
}

FailureOr<uint64_t> pointCount(ArrayRef<int64_t> shape) {
  uint64_t n = 1;
  for (int64_t size : shape) {
    if (size <= 0 || uint64_t(size) > kPointLimit / n)
      return failure();
    n *= size;
  }
  return n;
}

LogicalResult visitPoints(ArrayRef<int64_t> shape,
                        function_ref<LogicalResult(ArrayRef<int64_t>)> fn) {
  auto n = pointCount(shape);
  if (failed(n))
    return failure();
  SmallVector<int64_t> point(shape.size(), 0);
  for (uint64_t i = 0; i < *n; ++i) {
    if (failed(fn(point)))
      return failure();
    for (unsigned j = point.size(); j > 0; --j) {
      if (++point[j-1] < shape[j-1])
        break;
      point[j-1] = 0;
    }
  }
  return success();
}

FailureOr<int64_t> evalExpr(AffineExpr expr, ArrayRef<int64_t> point,
                           unsigned &budget) {
  if (!budget--)
    return failure();
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    if (dim.getPosition() >= point.size())
      return failure();
    return point[dim.getPosition()];
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expr))
    return constant.getValue();
  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return failure();
  auto lhs = evalExpr(bin.getLHS(), point, budget);
  auto rhs = evalExpr(bin.getRHS(), point, budget);
  if (failed(lhs) || failed(rhs))
    return failure();
  __int128 a = *lhs, b = *rhs;
  switch (expr.getKind()) {
  case AffineExprKind::Add: return narrow(a+b);
  case AffineExprKind::Mul: return narrow(a*b);
  case AffineExprKind::FloorDiv:
    if (b <= 0) return failure();
    return narrow(a/b - (a%b < 0));
  case AffineExprKind::CeilDiv:
    if (b <= 0) return failure();
    return narrow(a/b + (a%b > 0));
  case AffineExprKind::Mod:
    if (b <= 0) return failure();
    return narrow((a%b+b)%b);
  default: return failure();
  }
}

FailureOr<SmallVector<int64_t>> evalMap(AffineMap map,
                                       ArrayRef<int64_t> point) {
  if (map.getNumSymbols() || map.getNumDims() != point.size())
    return failure();
  unsigned budget = kExprLimit;
  SmallVector<int64_t> result;
  for (AffineExpr expr : map.getResults()) {
    auto value = evalExpr(expr, point, budget);
    if (failed(value)) return failure();
    result.push_back(*value);
  }
  return result;
}

// Exact conversion, including integer carries introduced by later composition.
// Restrict matrix and expression sizes before constructing expression trees.
FailureOr<AffineMap> asAffine(Attribute map) {
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map)) {
    AffineMap value = affine.getAffineMap().getValue();
    unsigned nodes = 0;
    for (AffineExpr expr : value.getResults())
      expr.walk([&](AffineExpr) { ++nodes; });
    if (nodes > kExprLimit || value.getNumSymbols()) return failure();
    return value;
  }
  auto bits = dyn_cast<BitLinearLayoutMapAttr>(map);
  if (!bits) return failure();
  auto widths = bits.getInputBitWidths().asArrayRef();
  unsigned columns = 0, rows = 0;
  for (int64_t width : widths) {
    if (width <= 0 || width > 62) return failure();
    columns += width;
    if (columns > 128) return failure();
  }
  for (int64_t width : bits.getOutputBitWidths().asArrayRef()) {
    if (width <= 0 || width > 62) return failure();
    rows += width;
    if (rows > 128) return failure();
  }
  if (uint64_t(rows)*columns > kExprLimit) return failure();
  auto matrix = bits.getMatrixValue();
  if (failed(matrix)) return failure();
  SmallVector<llvm::APInt> basis;
  for (unsigned column = 0; column < columns; ++column)
    basis.push_back(matrix->apply(llvm::APInt::getOneBitSet(columns, column)));
  auto *ctx = map.getContext();
  SmallVector<AffineExpr> inputBits;
  for (auto [dim, width] : llvm::enumerate(widths))
    for (int64_t bit = 0; bit < width; ++bit)
      inputBits.push_back(getAffineDimExpr(dim, ctx).floorDiv(int64_t{1} << bit) % 2);
  SmallVector<AffineExpr> results;
  unsigned row = 0;
  for (int64_t width : bits.getOutputBitWidths().asArrayRef()) {
    AffineExpr output = getAffineConstantExpr(0, ctx);
    for (int64_t bit = 0; bit < width; ++bit, ++row) {
      AffineExpr parity = getAffineConstantExpr(0, ctx);
      for (unsigned column = 0; column < columns; ++column)
        if (basis[column][row]) parity = parity + inputBits[column];
      output = output + (parity % 2) * (int64_t{1} << bit);
    }
    results.push_back(output);
  }
  return AffineMap::get(widths.size(), 0, results, ctx);
}

FailureOr<SmallVector<int64_t>> domain(Attribute map) {
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(map))
    return SmallVector<int64_t>(affine.getInputExtents().asArrayRef());
  if (auto bits = dyn_cast<BitLinearLayoutMapAttr>(map)) {
    SmallVector<int64_t> result;
    for (int64_t width : bits.getInputBitWidths().asArrayRef()) {
      if (width <= 0 || width > 62) return failure();
      result.push_back(int64_t{1} << width);
    }
    return result;
  }
  return failure();
}

struct Normalized {
  Value root;
  SmallVector<int64_t> shape;
  AffineMap transform;
};

FailureOr<Normalized> normalize(Value value, std::string &reason, unsigned depth = 0) {
  Operation *op = value.getDefiningOp();
  auto reject = [&](const Twine &detail) -> FailureOr<Normalized> {
    reason = (op ? op->getName().getStringRef() : StringRef("block argument")).str() +
             ": " + detail.str();
    return failure();
  };
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || depth > 64) return reject("unranked type or alias path exceeds 64 steps");
  Value source;
  if (auto view = dyn_cast_or_null<LayoutViewOp>(op))
    source = view.getSource();
  else if (auto castOp = dyn_cast_or_null<memref::CastOp>(op))
    source = castOp.getSource();
  else if (auto subview = dyn_cast_or_null<memref::SubViewOp>(op))
    source = subview.getSource();
  if (source) {
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType || sourceType.getElementType() != type.getElementType() ||
        sourceType.getMemorySpace() != type.getMemorySpace())
      return reject("element type or memory space changes across alias path");
    auto base = normalize(source, reason, depth+1);
    if (failed(base)) {
      reason = op->getName().getStringRef().str() + " -> " + reason;
      return failure();
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(op)) {
      auto offsets = subview.getStaticOffsets();
      auto sizes = subview.getStaticSizes();
      auto strides = subview.getStaticStrides();
      if (offsets.size() != base->shape.size()) return reject("subview metadata rank mismatch");
      auto dropped = subview.getDroppedDims();
      SmallVector<int64_t> shape;
      SmallVector<AffineExpr> transform;
      auto *ctx = value.getContext();
      for (unsigned i = 0; i < offsets.size(); ++i) {
        if (offsets[i] < 0 || sizes[i] <= 0 || strides[i] <= 0 ||
            ShapedType::isDynamic(offsets[i]) || ShapedType::isDynamic(sizes[i]) ||
            ShapedType::isDynamic(strides[i]))
          return reject("dimension " + Twine(i) + " requires static nonnegative offset and positive size/stride");
        auto last = narrow(__int128(offsets[i]) + __int128(sizes[i]-1)*strides[i]);
        if (failed(last) || *last >= base->shape[i])
          return reject("dimension " + Twine(i) + " exceeds source extent or overflows checked address arithmetic");
        AffineExpr expr = getAffineConstantExpr(offsets[i], ctx);
        if (dropped.test(i)) {
          if (sizes[i] != 1) return reject("dropped subview dimension is not unit-sized");
        } else {
          expr = expr + getAffineDimExpr(shape.size(), ctx)*strides[i];
          shape.push_back(sizes[i]);
        }
        transform.push_back(expr);
      }
      base->transform = base->transform.compose(AffineMap::get(shape.size(), 0, transform, ctx));
      base->shape = std::move(shape);
    }
    if (type.getRank() != int64_t(base->shape.size())) return reject("alias result rank disagrees with proven source shape");
    for (auto [declared, proven] : llvm::zip_equal(type.getShape(), base->shape))
      if (!ShapedType::isDynamic(declared) && declared != proven)
        return reject("alias result extent disagrees with proven source extent");
    return *base;
  }
  bool root = isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(op);
  if (auto global = dyn_cast_or_null<memref::GetGlobalOp>(op)) {
    auto definition = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        global, global.getNameAttr());
    if (!definition || definition.getType() != type) return reject("unresolved global symbol or ambiguous global type");
    // Canonical first access by IR order; repeated symbol accesses must not be
    // treated as distinct allocations. This value is an identity, not an SSA use.
    auto module = global->getParentOfType<ModuleOp>();
    if (!module) return reject("global access has no enclosing module for stable identity");
    Value canonical;
    module.walk([&](memref::GetGlobalOp other) {
      if (!canonical && SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
              other, other.getNameAttr()) == definition)
        canonical = other.getResult();
    });
    if (!canonical || canonical.getType() != type) return reject("ambiguous repeated global access type");
    value = canonical;
    root = true;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto function = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
    root = function && !function.empty() && argument.getOwner() == &function.front();
  }
  if (!root) return reject("unknown alias semantics or region-carried root (only alloc/alloca/global/function-entry roots supported)");
  if (!type.hasStaticShape()) return reject("root has dynamic shape; static child type cannot establish its live domain");
  for (int64_t extent : type.getShape()) if (extent <= 0) return reject("root live extent must be positive");
  return Normalized{value, SmallVector<int64_t>(type.getShape()),
                    AffineMap::getMultiDimIdentityMap(type.getRank(), value.getContext())};
}

using PointAddress = StorageAliasPointAddress;

LayoutProof proof(ProofStatus status, const Twine &reason,
                  ArrayRef<int64_t> point = {}) {
  return {status, SmallVector<int64_t>(point), reason.str()};
}

LayoutProof indexCandidate(const StorageAliasInfo &info, StorageLayoutAttr candidate,
                            SmallVectorImpl<PointAddress> &index) {
  auto unknown = [&](const Twine &why) { return proof(ProofStatus::Unknown, why); };
  if (!info.root || !info.rootType || !info.viewType || !info.viewToRoot || !candidate)
    return unknown("missing static storage alias descriptor");
  auto shape = info.viewType.getShape();
  if (failed(pointCount(shape))) return unknown("storage alias proof domain exceeds 65536 live points or is dynamic/empty");
  if (info.viewToRoot.getNumDims() != shape.size() ||
      info.viewToRoot.getNumResults() != unsigned(info.rootType.getRank()) ||
      info.rootType.getElementType() != info.viewType.getElementType() ||
      info.rootType.getMemorySpace() != info.viewType.getMemorySpace())
    return unknown("inconsistent root/view descriptor or coordinate transform");
  auto space = getFriskMemorySpace(info.viewType);
  if (!space || *space != candidate.getMemorySpace().getValue())
    return proof(ProofStatus::Disproven, "storage candidate memory space differs from root/view");
  int64_t alignment = candidate.getAlignment().getInt();
  int64_t vector = candidate.getVectorGranularity().getInt();
  if (alignment <= 0 || vector <= 0 || !llvm::isPowerOf2_64(alignment) ||
      !llvm::isPowerOf2_64(vector) || vector > alignment ||
      uint64_t(alignment) > info.rootAlignment || info.rootAlignment % alignment)
    return proof(ProofStatus::Disproven, "storage alignment lacks an explicit root pointer guarantee");
  auto map = asAffine(candidate.getMap());
  auto mapDomain = domain(candidate.getMap());
  if (failed(map) || failed(mapDomain)) return unknown("unsupported storage map representation or expression budget");
  if (mapDomain->size() != shape.size()) return proof(ProofStatus::Disproven, "storage map input rank mismatch");
  for (auto [extent, live] : llvm::zip_equal(*mapDomain, shape))
    if (extent != live || extent <= 0) return proof(ProofStatus::Disproven, "storage map declared domain must exactly match live view shape");
  ArrayAttr outputNames;
  SmallVector<int64_t> outputExtents;
  if (auto affine = dyn_cast<AffineLayoutMapAttr>(candidate.getMap())) {
    outputNames = affine.getOutputNames();
    outputExtents = SmallVector<int64_t>(affine.getOutputExtents().asArrayRef());
  } else if (auto bits = dyn_cast<BitLinearLayoutMapAttr>(candidate.getMap())) {
    outputNames = bits.getOutputNames();
    for (int64_t width : bits.getOutputBitWidths().asArrayRef())
      outputExtents.push_back(int64_t{1} << width);
  }
  if (!outputNames || outputNames.size() != 2 || outputExtents.size() != 2 ||
      cast<StringAttr>(outputNames[0]).getValue() != "byte_offset" ||
      cast<StringAttr>(outputNames[1]).getValue() != "bit_offset")
    return proof(ProofStatus::Disproven, "storage outputs must be byte_offset then bit_offset");
  uint64_t elementBits = info.viewType.getElementTypeBitWidth();
  LayoutProof result = proof(ProofStatus::Proven, "root-relative bit intervals verified");
  std::set<SmallVector<int64_t>> coordinates;
  (void)visitPoints(shape, [&](ArrayRef<int64_t> point) {
    auto root = evalMap(info.viewToRoot, point);
    auto address = evalMap(*map, point);
    if (failed(root) || failed(address)) {
      result = proof(ProofStatus::Unknown, "checked affine evaluation overflow or expression limit", point);
      return failure();
    }
    for (auto [coordinate, extent] : llvm::zip_equal(*root, info.rootType.getShape()))
      if (coordinate < 0 || coordinate >= extent) {
        result = proof(ProofStatus::Disproven, "view coordinate escapes root domain", point);
        return failure();
      }
    if (!coordinates.insert(*root).second) {
      result = proof(ProofStatus::Disproven, "view-to-root coordinate map is noninjective", point);
      return failure();
    }
    if (address->size() != 2 || (*address)[0] < 0 || (*address)[1] < 0 ||
        (*address)[1] >= 8 || (*address)[0] >= outputExtents[0] ||
        (*address)[1] >= outputExtents[1]) {
      result = proof(ProofStatus::Disproven, "storage byte/bit address escapes declared output bounds", point);
      return failure();
    }
    __int128 begin = __int128((*address)[0])*8+(*address)[1];
    __int128 end = begin+elementBits;
    if (begin < info.lowerBit || end > info.upperBit) {
      result = proof(ProofStatus::Disproven, "storage bit interval escapes root accessible span (including lower bound)", point);
      return failure();
    }
    index.push_back({SmallVector<int64_t>(point), *root, uint64_t(begin), uint64_t(end)});
    return success();
  });
  if (result.status != ProofStatus::Proven) return result;
  // vector=1 is the scalar/packed baseline. Larger guarantees apply to
  // complete innermost-row chunks, not merely to the root pointer. The index
  // is still in logical row-major order here (sorted physically below).
  if (vector > 1) {
    if (elementBits % 8)
      return proof(ProofStatus::Disproven, "vector granularity requires byte-sized elements");
    uint64_t elementBytes = elementBits / 8;
    if (elementBytes >= uint64_t(vector)) {
      for (const auto &entry : index)
        if (elementBytes % vector || entry.begin % (uint64_t(vector) * 8))
          return proof(ProofStatus::Disproven, "scalar vector subdivision is not aligned", entry.view);
    } else {
      uint64_t count = vector / elementBytes;
      uint64_t row = shape.empty() ? 1 : shape.back();
      if (vector % elementBytes || row % count)
        return proof(ProofStatus::Disproven, "vector granularity leaves an incomplete logical row chunk");
      for (size_t i = 0; i < index.size(); i += count) {
        if (index[i].begin % (uint64_t(vector) * 8))
          return proof(ProofStatus::Disproven, "vector chunk start is not aligned", index[i].view);
        for (uint64_t j = 1; j < count; ++j)
          if (index[i+j].begin != index[i].begin + j * elementBits)
            return proof(ProofStatus::Disproven, "vector chunk elements are not physically contiguous", index[i+j].view);
      }
    }
  }
  llvm::sort(index, [](const auto &a, const auto &b) { return a.begin < b.begin; });
  for (unsigned i = 1; i < index.size(); ++i)
    if (index[i].begin < index[i-1].end)
      return proof(ProofStatus::Disproven, "storage elements have overlapping physical bit intervals", index[i].view);
  return result;
}

StorageLayoutAttr makeLayout(const StorageAliasInfo &info, AffineMap map,
                              uint64_t alignment, uint64_t vector) {
  Builder b(info.viewType.getContext());
  SmallVector<Attribute> names;
  for (unsigned i = 0; i < unsigned(info.viewType.getRank()); ++i)
    names.push_back(b.getStringAttr("d" + Twine(i)));
  auto attr = AffineLayoutMapAttr::get(b.getContext(), b.getArrayAttr(names),
      b.getDenseI64ArrayAttr(info.viewType.getShape()),
      b.getStrArrayAttr({"byte_offset", "bit_offset"}),
      b.getDenseI64ArrayAttr({int64_t((info.upperBit+7)/8), 8}), AffineMapAttr::get(map));
  return StorageLayoutAttr::get(b.getContext(), attr,
      MemorySpaceAttr::get(b.getContext(), *getFriskMemorySpace(info.viewType)),
      b.getI64IntegerAttr(alignment), b.getI64IntegerAttr(vector));
}
} // namespace

FailureOr<StorageAliasInfo> mlir::frisk::analyzeStorageAlias(Value endpoint) {
  auto reject = [&](StringRef reason) -> FailureOr<StorageAliasInfo> {
    emitError(endpoint.getLoc()) << "unsupported storage alias path: " << reason;
    return failure();
  };
  auto viewType = dyn_cast<MemRefType>(endpoint.getType());
  if (!viewType || !viewType.hasStaticShape()) return reject("endpoint must have static ranked shape");
  std::string reason;
  auto normalized = normalize(endpoint, reason);
  if (failed(normalized)) return reject(reason);
  auto rootType = cast<MemRefType>(normalized->root.getType());
  if (!rootType.getElementType().isIntOrFloat()) return reject("unsupported element type");
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(rootType.getStridesAndOffset(strides, offset)) || offset < 0)
    return reject("root must have a static nonnegative descriptor offset");
  __int128 maximum = offset;
  SmallVector<std::pair<int64_t, int64_t>> axes;
  for (auto [size, stride] : llvm::zip_equal(rootType.getShape(), strides)) {
    if (size <= 0 || stride <= 0 || ShapedType::isDynamic(stride))
      return reject("root must have positive static extents and strides");
    maximum += __int128(size-1)*stride;
    if (maximum >= kMax) return reject("root descriptor address arithmetic overflows");
    if (size > 1) axes.emplace_back(stride, size);
  }
  llvm::sort(axes);
  __int128 occupied = 1;
  bool injective = true;
  for (auto [stride, size] : axes) {
    injective &= stride >= occupied;
    occupied += __int128(size-1)*stride;
  }
  if (!injective) {
    std::set<int64_t> addresses;
    auto check = visitPoints(rootType.getShape(), [&](ArrayRef<int64_t> point) {
      __int128 address = offset;
      for (auto [coordinate, stride] : llvm::zip_equal(point, strides)) address += __int128(coordinate)*stride;
      return success(addresses.insert(int64_t(address)).second);
    });
    if (failed(check)) return reject("root descriptor is noninjective or exceeds injectivity proof budget");
  }
  uint64_t bits = rootType.getElementTypeBitWidth();
  __int128 lower = __int128(offset)*bits, upper = (maximum+1)*bits;
  if (upper > kMax || !bits) return reject("root bit capacity overflows supported signed address arithmetic");
  StorageAliasInfo result{normalized->root, rootType, viewType,
                         normalized->transform, "", uint64_t(lower), uint64_t(upper), 1};
  IntegerAttr alignment;
  if (auto alloc = normalized->root.getDefiningOp<memref::AllocOp>()) alignment = alloc.getAlignmentAttr();
  if (auto alloc = normalized->root.getDefiningOp<memref::AllocaOp>()) alignment = alloc.getAlignmentAttr();
  if (auto global = normalized->root.getDefiningOp<memref::GetGlobalOp>())
    alignment = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(global, global.getNameAttr()).getAlignmentAttr();
  if (alignment && alignment.getInt() > 0 && llvm::isPowerOf2_64(alignment.getInt())) {
    result.rootAlignment = alignment.getInt();
    result.alignmentEvidence = "explicit allocation/global alignment attribute";
  }
  return result;
}

LayoutProof mlir::frisk::verifyStorageAliasCandidate(const StorageAliasInfo &info,
                                                   StorageLayoutAttr candidate) {
  return buildStorageAliasFootprint(info, candidate).proof;
}

StorageAliasFootprint mlir::frisk::buildStorageAliasFootprint(
    const StorageAliasInfo &info, StorageLayoutAttr candidate) {
  StorageAliasFootprint result;
  result.root = info.root;
  result.rootType = info.rootType;
  result.proof = indexCandidate(info, candidate, result.entries);
  if (!info.rootKey.empty()) result.proof.reason = "root " + info.rootKey + ": " + result.proof.reason;
  if (result.proof.status != ProofStatus::Proven) result.entries.clear();
  return result;
}

LayoutProof mlir::frisk::proveStorageAliasCompatible(
    const StorageAliasInfo &a, StorageLayoutAttr ca,
    const StorageAliasInfo &b, StorageLayoutAttr cb) {
  return proveStorageAliasFootprints(buildStorageAliasFootprint(a, ca),
                                     buildStorageAliasFootprint(b, cb));
}

LayoutProof mlir::frisk::proveStorageAliasFootprints(
    const StorageAliasFootprint &a, const StorageAliasFootprint &b) {
  if (a.root != b.root || a.rootType != b.rootType)
    return proof(ProofStatus::Unknown, "no common must-alias root proof");
  if (a.proof.status != ProofStatus::Proven) return a.proof;
  if (b.proof.status != ProofStatus::Proven) return b.proof;
  const auto &ai = a.entries, &bi = b.entries;
  std::map<SmallVector<int64_t>, const PointAddress *> byRoot;
  for (const auto &entry : ai) byRoot.emplace(entry.root, &entry);
  for (const auto &entry : bi) {
    auto found = byRoot.find(entry.root);
    if (found != byRoot.end() && found->second->begin != entry.begin)
      return proof(ProofStatus::Disproven, "same root coordinate has different physical bit addresses", entry.root);
  }
  unsigned i = 0, j = 0;
  while (i < ai.size() && j < bi.size()) {
    const auto &x = ai[i], &y = bi[j];
    if (x.begin < y.end && y.begin < x.end && x.root != y.root)
      return proof(ProofStatus::Disproven, "different root coordinates collide in physical bit intervals", y.root);
    if (x.end <= y.end) ++i; else ++j;
  }
  return proof(ProofStatus::Proven, "common-root coordinates and bit intervals agree");
}

FailureOr<StorageLayoutAttr> mlir::frisk::buildRootLinearStorageCandidate(
    const StorageAliasInfo &info) {
  if (info.viewType != info.rootType || !info.viewToRoot.isIdentity() ||
      !getFriskMemorySpace(info.rootType) ||
      *getFriskMemorySpace(info.rootType) == attr::MemorySpace::Local) return failure();
  SmallVector<int64_t> strides;
  int64_t offset;
  auto rootType = info.rootType;
  if (failed(rootType.getStridesAndOffset(strides, offset))) return failure();
  auto *ctx = info.rootType.getContext();
  AffineExpr address = getAffineConstantExpr(offset, ctx);
  for (auto [dim, stride] : llvm::enumerate(strides)) address = address + getAffineDimExpr(dim, ctx)*stride;
  address = address*info.rootType.getElementTypeBitWidth();
  return makeLayout(info, AffineMap::get(info.rootType.getRank(), 0,
                    {address.floorDiv(8), address%8}, ctx), info.rootAlignment, 1);
}

FailureOr<StorageLayoutAttr> mlir::frisk::projectStorageAliasCandidate(
    const StorageAliasInfo &source, StorageLayoutAttr candidate,
    const StorageAliasInfo &destination) {
  if (source.root != destination.root || source.rootType != destination.rootType)
    return failure();
  if (verifyStorageAliasCandidate(source, candidate).status != ProofStatus::Proven)
    return failure();
  if (source.viewType.getShape() == destination.viewType.getShape() &&
      source.viewToRoot == destination.viewToRoot &&
      verifyStorageAliasCandidate(destination, candidate).status == ProofStatus::Proven)
    return candidate;
  auto map = asAffine(candidate.getMap());
  if (failed(map) || failed(pointCount(destination.viewType.getShape()))) return failure();
  auto *ctx = source.rootType.getContext();
  unsigned sourceRank = source.viewType.getRank();
  unsigned destRank = destination.viewType.getRank();
  SmallVector<int64_t> zero(sourceRank, 0);
  auto origin = evalMap(source.viewToRoot, zero);
  if (failed(origin)) return failure();
  SmallVector<AffineExpr> inverse(sourceRank);
  // Every retained source axis has exactly one positive-stride root axis.
  // Derive its unique inverse, then prove containment pointwise; floorDiv must
  // not round an off-lattice coordinate into a false source preimage.
  for (unsigned s = 0; s < sourceRank; ++s) {
    SmallVector<int64_t> unit(sourceRank, 0);
    unit[s] = 1;
    auto step = evalMap(source.viewToRoot, unit);
    if (failed(step)) return failure();
    int rootAxis = -1;
    int64_t stride = 0;
    for (unsigned r = 0; r < step->size(); ++r) {
      auto delta = narrow(__int128((*step)[r])-(*origin)[r]);
      if (failed(delta) || *delta < 0) return failure();
      if (*delta) {
        if (rootAxis != -1) return failure();
        rootAxis = r;
        stride = *delta;
      }
    }
    if (rootAxis < 0) return failure();
    inverse[s] = (destination.viewToRoot.getResult(rootAxis)-(*origin)[rootAxis]).floorDiv(stride);
  }
  AffineMap embedding = AffineMap::get(destRank, 0, inverse, ctx);
  if (failed(visitPoints(destination.viewType.getShape(), [&](ArrayRef<int64_t> point) {
    auto sourcePoint = evalMap(embedding, point);
    auto destRoot = evalMap(destination.viewToRoot, point);
    if (failed(sourcePoint) || failed(destRoot)) return failure();
    for (auto [coordinate, extent] : llvm::zip_equal(*sourcePoint, source.viewType.getShape()))
      if (coordinate < 0 || coordinate >= extent) return failure();
    auto sourceRoot = evalMap(source.viewToRoot, *sourcePoint);
    return success(succeeded(sourceRoot) && *sourceRoot == *destRoot);
  }))) return failure();
  uint64_t alignment = std::min(uint64_t(candidate.getAlignment().getInt()), destination.rootAlignment);
  uint64_t vector = std::min(uint64_t(candidate.getVectorGranularity().getInt()), alignment);
  auto composed = map->compose(embedding);
  // A projection may weaken inferred vector guarantees. Never weaken a hard
  // source binding: it was verified above before reaching this proposal path.
  for (; vector; vector /= 2) {
    auto projected = makeLayout(destination, composed, alignment, vector);
    if (verifyStorageAliasCandidate(destination, projected).status == ProofStatus::Proven)
      return projected;
  }
  return failure();
}
