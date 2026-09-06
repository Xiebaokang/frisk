#include "Dialect/Frisk/Analysis/LayoutVerifier.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::frisk {

namespace {

bool storageMapsAgree(StorageLayoutAttr lhs, StorageLayoutAttr rhs) {
  return lhs && rhs && lhs.getMap() == rhs.getMap();
}

Value getLayoutViewAliasRoot(Value value) {
  while (auto view = value.getDefiningOp<LayoutViewOp>())
    value = view.getSource();
  return value;
}

} // namespace

LogicalResult materializeLayouts(Operation *,
                                 const LayoutConstraintGraph &graph,
                                 const LayoutSolution &solution) {
  for (const LayoutVar &var : graph.getVariables()) {
    if (!var.anchor)
      continue;
    auto view = dyn_cast<LayoutViewOp>(var.anchor);
    if (!view)
      continue;
    Attribute assignment = solution.assignments.lookup(var.id);
    auto storage = dyn_cast_or_null<StorageLayoutAttr>(assignment);
    if (!storage)
      return view.emitOpError("storage layout solution has non-storage attr");
    if (StorageLayoutAttr existing = view.getLayoutAttr()) {
      if (existing != storage)
        return view.emitOpError(
            "materialization would overwrite an explicit hard binding");
      continue;
    }
    view->setAttr("layout", storage);
  }
  return success();
}

LogicalResult verifyMaterializedLayouts(Operation *root,
                                        LayoutTarget &target) {
  DenseMap<Value, StorageLayoutAttr> layoutByView;
  DenseMap<Value, StorageLayoutAttr> layoutBySource;
  bool valid = true;
  root->walk([&](LayoutViewOp view) {
    StorageLayoutAttr layout = view.getLayoutAttr();
    if (!layout) {
      view.emitOpError("unresolved storage layout for layout variable");
      valid = false;
      return;
    }
    LayoutVar var{0, LayoutKind::Storage, view.getResult().getType(), {},
                  LayoutState::Resolved, "materialized-view", view};
    if (failed(target.verifyCandidate(var, layout, view.getLoc()))) {
      valid = false;
      return;
    }
    auto type = cast<MemRefType>(view.getSource().getType());
    FailureOr<uint64_t> required = getStorageFootprintBytes(layout, type);
    FailureOr<uint64_t> capacity = getMemRefStaticCapacityBytes(type);
    if (failed(required) || failed(capacity)) {
      view.emitOpError(
          "cannot prove storage layout footprint fits the underlying memref "
          "type");
      valid = false;
      return;
    }
    if (*required > *capacity) {
      view.emitOpError("storage layout requires ")
          << *required << " bytes but underlying memref type provides "
          << *capacity << " bytes";
      valid = false;
      return;
    }
    Value aliasRoot = getLayoutViewAliasRoot(view.getSource());
    auto [it, inserted] = layoutBySource.try_emplace(aliasRoot, layout);
    if (!inserted && it->second != layout) {
      view.emitOpError("alias layout views have inconsistent bindings");
      valid = false;
    }
    layoutByView[view.getResult()] = layout;
  });
  root->walk([&](CopyOp copy) {
    StorageLayoutAttr src = layoutByView.lookup(copy.getSrc());
    StorageLayoutAttr dst = layoutByView.lookup(copy.getDst());
    if (!src || !dst) {
      copy.emitOpError(
          "unresolved storage layout for whole-tile copy operand");
      valid = false;
      return;
    }
    if (!storageMapsAgree(src, dst)) {
      copy.emitOpError(
          "materialized whole-tile copy storage maps are incompatible");
      valid = false;
    }
  });
  return success(valid);
}

} // namespace mlir::frisk
