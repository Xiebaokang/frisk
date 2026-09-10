#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::frisk {
std::string layoutCandidateKey(Attribute value) {
  std::string key;
  llvm::raw_string_ostream(key) << value;
  return key;
}

bool isSupportedLayoutRelation(ConstraintKind kind) {
  return kind == ConstraintKind::SameLayout ||
         kind == ConstraintKind::AliasLayout ||
         kind == ConstraintKind::StorageAccess ||
         kind == ConstraintKind::TransformLayout ||
         kind == ConstraintKind::Convertible;
}

bool layoutEncodingsEqual(Attribute lhs, Attribute rhs) {
  // SSA tensor type equality is exact, including named canonical-map metadata.
  // Do not equate differently named maps and then produce unequal operand types.
  return lhs == rhs;
}

static bool sameLogicalType(Type lhs, Type rhs) {
  auto a = dyn_cast<ShapedType>(lhs), b = dyn_cast<ShapedType>(rhs);
  return a && b && a.hasStaticShape() && b.hasStaticShape() &&
         a.getShape() == b.getShape() && a.getElementType() == b.getElementType();
}

static FailureOr<Attribute> permuteEncoding(Attribute candidate,
                                           Attribute transform,
                                           ShapedType target, bool inverse) {
  auto encoding = dyn_cast<DistributedEncodingAttr>(candidate);
  auto permutation = dyn_cast_or_null<DenseI64ArrayAttr>(transform);
  if (!encoding || !permutation)
    return failure();
  auto map = dyn_cast<BitLinearLayoutMapAttr>(encoding.getMap());
  if (!map || permutation.size() != map.getOutputBitWidths().size())
    return failure();
  unsigned rank = permutation.size();
  SmallVector<int64_t> order(permutation.asArrayRef());
  SmallVector<bool> seen(rank, false);
  for (int64_t dim : order) {
    if (dim < 0 || dim >= rank || seen[dim])
      return failure();
    seen[dim] = true;
  }
  if (inverse) {
    auto copy = order;
    for (unsigned i = 0; i < rank; ++i)
      order[copy[i]] = i;
  }
  SmallVector<unsigned> starts;
  unsigned rows = 0;
  for (int64_t width : map.getOutputBitWidths().asArrayRef()) {
    starts.push_back(rows);
    rows += width;
  }
  SmallVector<APInt> matrixValues;
  auto matrix = map.getMatrix().getValues<APInt>();
  unsigned columns = map.getMatrix().getType().getShape()[1];
  SmallVector<int64_t> widths;
  SmallVector<Attribute> names;
  Builder builder(candidate.getContext());
  for (auto [index, source] : llvm::enumerate(order)) {
    int64_t width = map.getOutputBitWidths()[source];
    widths.push_back(width);
    names.push_back(builder.getStringAttr("dim" + Twine(index)));
    for (unsigned row = starts[source]; row < starts[source] + width; ++row)
      for (unsigned col = 0; col < columns; ++col)
        matrixValues.push_back(matrix[row * columns + col]);
  }
  auto transformed = BitLinearLayoutMapAttr::get(
      candidate.getContext(), map.getInputNames(), map.getInputBitWidths(),
      builder.getArrayAttr(names), builder.getDenseI64ArrayAttr(widths),
      DenseIntElementsAttr::get(map.getMatrix().getType(), matrixValues));
  auto result = DistributedEncodingAttr::get(candidate.getContext(), transformed,
                                             encoding.getTopology(),
                                             encoding.getReplication());
  // Shape/permutation mismatch is a missing support, not a speculative diagnostic.
  if (target.getRank() != rank)
    return failure();
  for (unsigned i = 0; i < rank; ++i)
    if (widths[i] >= 63 || target.getDimSize(i) != (int64_t{1} << widths[i]))
      return failure();
  return Attribute(result);
}

FailureOr<Attribute> projectLayoutCandidate(
    const LayoutConstraintGraph &graph, const LayoutConstraint &relation,
    LayoutVarID source, Attribute candidate, LayoutVarID target) {
  const LayoutVar &dst = graph.getVariable(target);
  if (relation.kind == ConstraintKind::SameLayout ||
      relation.kind == ConstraintKind::AliasLayout ||
      relation.kind == ConstraintKind::Convertible)
    return candidate;
  if (relation.kind == ConstraintKind::TransformLayout)
    return permuteEncoding(candidate, relation.coordinateTransform,
                           cast<ShapedType>(dst.shapedType),
                           source != relation.vars.front());
  if (relation.kind != ConstraintKind::StorageAccess)
    return failure();
  auto storage = dyn_cast<StorageLayoutAttr>(candidate);
  auto type = dyn_cast<MemRefType>(dst.shapedType);
  if (!storage || !type)
    return failure(); // StorageAccess is not a preferred-order hard equality.
  auto space = getFriskMemorySpace(type);
  if (!space || *space == attr::MemorySpace::Local)
    return failure();
  return Attribute(StorageLayoutAttr::get(
      type.getContext(), storage.getMap(),
      MemorySpaceAttr::get(type.getContext(), *space), storage.getAlignment(),
      storage.getVectorGranularity()));
}

bool layoutRelationCompatible(const LayoutConstraintGraph &graph,
                              const LayoutConstraint &relation,
                              LayoutVarID lhsID, Attribute lhs,
                              LayoutVarID rhsID, Attribute rhs) {
  const LayoutVar &a = graph.getVariable(lhsID), &b = graph.getVariable(rhsID);
  if (relation.kind == ConstraintKind::TransformLayout) {
    auto projected = projectLayoutCandidate(graph, relation, lhsID, lhs, rhsID);
    return succeeded(projected) && layoutEncodingsEqual(*projected, rhs);
  }
  if (relation.kind == ConstraintKind::Convertible) {
    auto d0 = dyn_cast<DistributedEncodingAttr>(lhs);
    auto d1 = dyn_cast<DistributedEncodingAttr>(rhs);
    return d0 && d1 && sameLogicalType(a.shapedType, b.shapedType) &&
           d0.getTopology()[4] == 1 && d1.getTopology()[4] == 1;
  }
  if (relation.kind == ConstraintKind::StorageAccess) {
    auto s0 = dyn_cast<StorageLayoutAttr>(lhs);
    auto s1 = dyn_cast<StorageLayoutAttr>(rhs);
    if (s0 && s1)
      return s0.getMap() == s1.getMap(); // Existing whole-tile M2 copy.
    auto d0 = dyn_cast<DistributedEncodingAttr>(lhs);
    auto d1 = dyn_cast<DistributedEncodingAttr>(rhs);
    if ((d0 && s1) || (s0 && d1)) {
      // Verified D covers the logical tile; verified S addresses that exact
      // domain. Therefore S(D(h)) is valid, independent of coalescing. Store
      // lowering elects a deterministic owner for replicated logical elements.
      return sameLogicalType(a.shapedType, b.shapedType);
    }
  }
  return layoutEncodingsEqual(lhs, rhs);
}
} // namespace mlir::frisk
