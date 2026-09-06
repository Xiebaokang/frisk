#include "Dialect/Frisk/Analysis/LayoutSolver.h"

#include <algorithm>

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {

namespace {

bool compatible(ConstraintKind kind, Attribute lhs, Attribute rhs) {
  if (kind == ConstraintKind::StorageAccess) {
    auto lhsStorage = dyn_cast<StorageLayoutAttr>(lhs);
    auto rhsStorage = dyn_cast<StorageLayoutAttr>(rhs);
    if (lhsStorage && rhsStorage)
      return lhsStorage.getMap() == rhsStorage.getMap();
  }
  return lhs == rhs;
}

void updateState(LayoutVar &var) {
  if (var.candidates.empty())
    var.state = LayoutState::Conflict;
  else if (var.candidates.size() == 1)
    var.state = LayoutState::Resolved;
  else
    var.state = LayoutState::CandidateSet;
}

bool filterCompatible(LayoutVar &var, ArrayRef<LayoutCandidate> other,
                      ConstraintKind kind) {
  size_t oldSize = var.candidates.size();
  llvm::erase_if(var.candidates, [&](const LayoutCandidate &candidate) {
    return llvm::none_of(other, [&](const LayoutCandidate &otherCandidate) {
      return compatible(kind, candidate.value, otherCandidate.value);
    });
  });
  updateState(var);
  return oldSize != var.candidates.size();
}

LogicalResult emitConflict(const LayoutConstraintGraph &graph,
                           const LayoutConstraint &constraint) {
  const LayoutProvenance &provenance =
      graph.getProvenances()[constraint.provenance];
  if (!provenance.source)
    return failure();
  InFlightDiagnostic diagnostic = provenance.source->emitError(
      "conflicting hard layout constraint '");
  diagnostic << provenance.rule << "': " << provenance.reason;
  for (LayoutVarID id : constraint.vars) {
    const LayoutVar &var = graph.getVariable(id);
    diagnostic.attachNote(provenance.source->getLoc())
        << "layout variable " << var.stableName
        << " has no compatible candidate";
  }
  return failure();
}

LogicalResult applyEqualityConstraint(LayoutConstraintGraph &graph,
                                      const LayoutConstraint &constraint,
                                      bool &changed) {
  if (constraint.vars.size() < 2)
    return success();
  for (size_t lhsIndex = 0; lhsIndex < constraint.vars.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1;
         rhsIndex < constraint.vars.size(); ++rhsIndex) {
      LayoutVar &lhs = graph.getVariable(constraint.vars[lhsIndex]);
      LayoutVar &rhs = graph.getVariable(constraint.vars[rhsIndex]);
      SmallVector<LayoutCandidate> lhsSnapshot(lhs.candidates);
      SmallVector<LayoutCandidate> rhsSnapshot(rhs.candidates);
      changed |= filterCompatible(lhs, rhsSnapshot, constraint.kind);
      changed |= filterCompatible(rhs, lhsSnapshot, constraint.kind);
      if (lhs.state == LayoutState::Conflict ||
          rhs.state == LayoutState::Conflict) {
        lhs.state = LayoutState::Conflict;
        rhs.state = LayoutState::Conflict;
        return emitConflict(graph, constraint);
      }
    }
  }
  return success();
}

bool isEqualityConstraint(ConstraintKind kind) {
  return kind == ConstraintKind::SameLayout ||
         kind == ConstraintKind::AliasLayout ||
         kind == ConstraintKind::StorageAccess;
}

bool isWholeTileStaticCopy(CopyOp copy) {
  MemRefType srcType = copy.getSrcMemRefType();
  MemRefType dstType = copy.getDstMemRefType();
  return srcType.hasStaticShape() && dstType.hasStaticShape() &&
         srcType.getShape() == copy.getSrcExtents() &&
         dstType.getShape() == copy.getDstExtents() &&
         srcType.getShape() == dstType.getShape() &&
         srcType.getElementType() == dstType.getElementType() &&
         copy.getSrcIndices().empty() && copy.getDstIndices().empty() &&
         copy.getSrcMap().getNumInputs() == 0 &&
         copy.getDstMap().getNumInputs() == 0;
}

} // namespace

LayoutVarID LayoutConstraintBuilder::getOrCreate(Value value,
                                                 LayoutKind kind) {
  auto found = variablesByValue.find(value);
  if (found != variablesByValue.end())
    return found->second;

  std::string name;
  llvm::raw_string_ostream(name)
      << (kind == LayoutKind::Storage ? "storage/" : "distributed/")
      << llvm::format_hex_no_prefix(nextStableOrdinal++, 8);
  Operation *anchor = value.getDefiningOp<LayoutViewOp>();
  LayoutVarID id = graph.addVariable(kind, value.getType(), name, anchor);
  variablesByValue.try_emplace(value, id);
  return id;
}

LayoutVarID LayoutConstraintBuilder::getOrCreateStorageVar(Value anchor) {
  return getOrCreate(anchor, LayoutKind::Storage);
}

LayoutVarID
LayoutConstraintBuilder::getOrCreateDistributedVar(Value value) {
  return getOrCreate(value, LayoutKind::Distributed);
}

LogicalResult LayoutConstraintBuilder::require(LayoutVarID var,
                                               Attribute encoding,
                                               Operation *source,
                                               StringRef rule) {
  if (!encoding)
    return source->emitError("required layout encoding is missing");
  LayoutConstraintID constraint = graph.addConstraint(
      ConstraintKind::RequireEncoding, ConstraintStrength::Hard, {var},
      source, rule, "explicit layout binding", encoding);
  LayoutVar &variable = graph.getVariable(var);
  variable.candidates.push_back(
      {encoding, graph.getConstraint(constraint).provenance, 0});
  updateState(variable);
  return success();
}

LogicalResult LayoutConstraintBuilder::same(LayoutVarID lhs,
                                            LayoutVarID rhs,
                                            Operation *source,
                                            StringRef rule) {
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                      {lhs, rhs}, source, rule, "exact layout equality");
  return success();
}

std::optional<LayoutVarID>
LayoutConstraintBuilder::lookup(Value value) const {
  auto found = variablesByValue.find(value);
  if (found == variablesByValue.end())
    return std::nullopt;
  return found->second;
}

FailureOr<LayoutConstraintGraph>
collectLayoutConstraints(Operation *root, LayoutTarget &target) {
  LayoutConstraintGraph graph;
  LayoutConstraintBuilder builder(graph);
  DenseMap<Value, SmallVector<LayoutVarID>> viewsBySource;
  bool failedCollection = false;

  root->walk([&](LayoutViewOp view) {
    LayoutVarID id = builder.getOrCreateStorageVar(view.getResult());
    viewsBySource[view.getSource()].push_back(id);
    if (StorageLayoutAttr layout = view.getLayoutAttr())
      failedCollection |= failed(builder.require(id, layout, view, "layout_view"));
  });

  for (auto &entry : viewsBySource) {
    ArrayRef<LayoutVarID> ids = entry.second;
    for (size_t index = 1; index < ids.size(); ++index)
      graph.addConstraint(ConstraintKind::AliasLayout,
                          ConstraintStrength::Hard,
                          {ids[index - 1], ids[index]},
                          graph.getVariable(ids[index]).anchor,
                          "same-source-layout-view",
                          "views of the same storage value must agree");
  }

  root->walk([&](CopyOp copy) {
    if (!isWholeTileStaticCopy(copy)) {
      copy.emitOpError(
          "unsupported storage layout inference for non-whole-tile or "
          "dynamic copy");
      failedCollection = true;
      return;
    }
    LayoutVarID src = builder.getOrCreateStorageVar(copy.getSrc());
    LayoutVarID dst = builder.getOrCreateStorageVar(copy.getDst());
    graph.addConstraint(ConstraintKind::StorageAccess,
                        ConstraintStrength::Hard, {src, dst}, copy,
                        "whole-tile-copy",
                        "source and destination storage maps must agree");
    graph.addConstraint(ConstraintKind::Preference,
                        ConstraintStrength::Soft, {src, dst}, copy,
                        "coalesced-copy", "prefer coalesced storage access");
  });
  if (failedCollection)
    return failure();
  if (failed(graph.finalize(root->getLoc())))
    return failure();

  for (LayoutVar &var : graph.getVariables()) {
    if (var.candidates.empty()) {
      SmallVector<LayoutCandidate> candidates;
      target.enumerateCandidates(var, candidates);
      for (const LayoutCandidate &candidate : candidates) {
        if (succeeded(target.verifyCandidate(var, candidate.value,
                                             root->getLoc())) &&
            llvm::none_of(var.candidates, [&](const LayoutCandidate &known) {
              return known.value == candidate.value;
            }))
          var.candidates.push_back(candidate);
      }
    }
    llvm::stable_sort(var.candidates,
                      [](const LayoutCandidate &lhs,
                         const LayoutCandidate &rhs) {
      return lhs.stableOrdinal < rhs.stableOrdinal;
    });
    updateState(var);
    if (var.state == LayoutState::Conflict) {
      if (var.anchor)
        var.anchor->emitError("no valid target layout candidates for ")
            << var.stableName;
      return failure();
    }
  }
  return graph;
}

LogicalResult propagateStrict(LayoutConstraintGraph &graph) {
  for (const LayoutConstraint &constraint : graph.getConstraints()) {
    if (constraint.strength != ConstraintStrength::Hard)
      continue;
    if (constraint.kind == ConstraintKind::RequireEncoding) {
      LayoutVar &var = graph.getVariable(constraint.vars.front());
      size_t oldSize = var.candidates.size();
      llvm::erase_if(var.candidates, [&](const LayoutCandidate &candidate) {
        return candidate.value != constraint.requiredEncoding;
      });
      (void)oldSize;
      updateState(var);
      if (var.state == LayoutState::Conflict)
        return emitConflict(graph, constraint);
      continue;
    }
    if (!isEqualityConstraint(constraint.kind))
      continue;
    bool anySingleton = llvm::any_of(constraint.vars, [&](LayoutVarID id) {
      return graph.getVariable(id).candidates.size() == 1;
    });
    if (!anySingleton)
      continue;
    bool changed = false;
    if (failed(applyEqualityConstraint(graph, constraint, changed)))
      return failure();
  }
  return success();
}

LogicalResult propagateCommonToFixedPoint(LayoutConstraintGraph &graph) {
  bool changed;
  do {
    changed = false;
    for (const LayoutConstraint &constraint : graph.getConstraints()) {
      if (constraint.strength != ConstraintStrength::Hard ||
          !isEqualityConstraint(constraint.kind))
        continue;
      if (failed(applyEqualityConstraint(graph, constraint, changed)))
        return failure();
    }
  } while (changed);
  return success();
}

} // namespace mlir::frisk
