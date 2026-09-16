#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"

#include <functional>
#include <deque>

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::frisk {

namespace {

void updateState(LayoutVar &var) {
  if (var.candidates.empty())
    var.state = LayoutState::Conflict;
  else if (var.candidates.size() == 1)
    var.state = LayoutState::Resolved;
  else
    var.state = LayoutState::CandidateSet;
}

bool filterCompatible(LayoutConstraintGraph &graph, LayoutVar &var,
                      LayoutVarID otherID, ArrayRef<LayoutCandidate> other,
                      const LayoutConstraint &relation) {
  size_t oldSize = var.candidates.size();
  llvm::erase_if(var.candidates, [&](const LayoutCandidate &candidate) {
    return llvm::none_of(other, [&](const LayoutCandidate &otherCandidate) {
      return layoutRelationCompatible(graph, relation, var.id, candidate.value,
                                      otherID, otherCandidate.value);
    });
  });
  updateState(var);
  return oldSize != var.candidates.size();
}

LogicalResult emitConflict(const LayoutConstraintGraph &graph,
                           const LayoutConstraint &constraint,
                           ArrayRef<LayoutCandidate> lhsSeeds = {},
                           ArrayRef<LayoutCandidate> rhsSeeds = {}) {
  const LayoutProvenance &provenance =
      graph.getProvenances()[constraint.provenance];
  if (!provenance.source)
    return failure();
  InFlightDiagnostic diagnostic = provenance.source->emitError(
      "conflicting hard layout constraint '");
  diagnostic << provenance.rule << "': " << provenance.reason;
  auto attachSeed = [&](unsigned index, ArrayRef<LayoutCandidate> seeds) {
    if (index >= constraint.vars.size() || seeds.empty())
      return;
    const LayoutCandidate &seed = seeds.front();
    std::string chain;
    llvm::raw_string_ostream stream(chain);
    if (seed.provenance < graph.getProvenances().size() &&
        succeeded(graph.printProvenanceChain(seed.provenance, stream)))
      diagnostic.attachNote(provenance.source->getLoc())
          << "seed for " << graph.getVariable(constraint.vars[index]).stableName
          << ": " << chain;
  };
  attachSeed(0, lhsSeeds);
  attachSeed(1, rhsSeeds);
  if (constraint.kind == ConstraintKind::AliasLayout &&
      !lhsSeeds.empty() && !rhsSeeds.empty()) {
    auto proof = proveAliasLayoutRelation(graph, constraint.vars[0],
        lhsSeeds.front().value, constraint.vars[1], rhsSeeds.front().value);
    std::string coordinate;
    llvm::raw_string_ostream os(coordinate);
    llvm::interleaveComma(proof.counterexample, os);
    diagnostic.attachNote(provenance.source->getLoc())
        << (proof.status == ProofStatus::Unknown ? "unknown alias proof: " : "alias counterexample: ")
        << proof.reason << "; coordinate [" << coordinate << "]";
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
      changed |= filterCompatible(graph, lhs, rhs.id, rhsSnapshot, constraint);
      changed |= filterCompatible(graph, rhs, lhs.id, lhsSnapshot, constraint);
      if (lhs.state == LayoutState::Conflict ||
          rhs.state == LayoutState::Conflict) {
        lhs.state = LayoutState::Conflict;
        rhs.state = LayoutState::Conflict;
        return emitConflict(graph, constraint, lhsSnapshot, rhsSnapshot);
      }
    }
  }
  return success();
}

bool isEqualityConstraint(ConstraintKind kind) {
  return isSupportedLayoutRelation(kind);
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
         copy.getDstMap().getNumInputs() == 0 &&
         copy.getSrcMap().getNumResults() == 0 &&
         copy.getDstMap().getNumResults() == 0;
}


} // namespace

static Operation *findEnclosingSymbol(Operation *operation) {
  Operation *top = operation;
  for (Operation *owner = operation; owner; owner = owner->getParentOp()) {
    top = owner;
    if (owner->hasAttr(SymbolTable::getSymbolAttrName()))
      return owner;
  }
  return top;
}

static unsigned getBlockOrdinal(Operation *scope, Block *target) {
  if (!scope)
    return 0;
  unsigned ordinal = 0;
  bool found = false;
  std::function<void(Operation *)> visit = [&](Operation *operation) {
    for (Region &region : operation->getRegions()) {
      for (Block &block : region) {
        if (found)
          return;
        if (&block == target) {
          found = true;
          return;
        }
        ++ordinal;
        for (Operation &nested : block)
          visit(&nested);
      }
    }
  };
  visit(scope);
  return ordinal;
}

static std::string getQualifiedSymbolName(Operation *operation) {
  SmallVector<StringRef> components;
  for (Operation *owner = operation; owner; owner = owner->getParentOp())
    if (auto name = owner->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      components.push_back(name.getValue());
  if (components.empty())
    return "anonymous";
  std::string qualified;
  llvm::raw_string_ostream stream(qualified);
  bool first = true;
  for (StringRef component : llvm::reverse(components)) {
    if (!first)
      stream << '/';
    first = false;
    stream << component.size() << ':' << component;
  }
  return qualified;
}

static std::string getStableValueName(Value value, LayoutKind kind,
                                      uint64_t fallbackOrdinal) {
  std::string name;
  llvm::raw_string_ostream stream(name);
  StringRef kindName =
      kind == LayoutKind::Storage ? "storage" : "distributed";
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *operation = result.getOwner();
    Block *block = operation->getBlock();
    Operation *symbol = findEnclosingSymbol(operation);
    unsigned blockOrdinal = getBlockOrdinal(symbol, block);
    unsigned operationOrdinal = 0;
    for (Operation &candidate : *block) {
      if (&candidate == operation)
        break;
      ++operationOrdinal;
    }
    stream << getQualifiedSymbolName(operation) << "/b" << blockOrdinal
           << "/o" << operationOrdinal << "/r" << result.getResultNumber()
           << "/" << kindName;
    return name;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    Operation *symbol = findEnclosingSymbol(owner);
    stream << getQualifiedSymbolName(owner) << "/b"
           << getBlockOrdinal(symbol, argument.getOwner()) << "/arg"
           << argument.getArgNumber() << "/" << kindName;
    return name;
  }
  stream << "anonymous/fallback" << fallbackOrdinal << "/" << kindName;
  return name;
}

LayoutVarID LayoutConstraintBuilder::getOrCreate(Value value,
                                                 LayoutKind kind) {
  DenseMap<Value, LayoutVarID> &variablesByValue =
      kind == LayoutKind::Storage ? storageVariablesByValue
                                  : distributedVariablesByValue;
  auto found = variablesByValue.find(value);
  if (found != variablesByValue.end())
    return found->second;

  std::string name = getStableValueName(value, kind, nextStableOrdinal++);
  Operation *anchor = value.getDefiningOp();
  if (!anchor)
    anchor = cast<BlockArgument>(value).getOwner()->getParentOp();
  LayoutVarID id = graph.addVariable(kind, value.getType(), name, anchor);
  graph.getVariable(id).value = value;
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

static std::string getStableUseKey(OpOperand &use) {
  // Stable owner block/operation ordinals and operand slot, never use-list order.
  Operation *owner = use.getOwner();
  unsigned ordinal = 0;
  for (Operation &op : *owner->getBlock()) {
    if (&op == owner) break;
    ++ordinal;
  }
  return getQualifiedSymbolName(owner) + "/b" +
      std::to_string(getBlockOrdinal(findEnclosingSymbol(owner), owner->getBlock())) +
      "/o" + std::to_string(ordinal) + "/use" +
      std::to_string(use.getOperandNumber());
}

LayoutVarID LayoutConstraintBuilder::getOrCreateDistributedUse(OpOperand &use) {
  auto found = distributedVariablesByUse.find(&use);
  if (found != distributedVariablesByUse.end())
    return found->second;
  LayoutVarID src = getOrCreateDistributedVar(use.get());
  std::string key = getStableUseKey(use);
  LayoutVarID dst = graph.addVariable(LayoutKind::Distributed, use.get().getType(),
                                      key, use.getOwner());
  graph.getVariable(dst).use = &use;
  distributedVariablesByUse[&use] = dst;
  (void)convertible(src, dst, use);
  return dst;
}

LogicalResult LayoutConstraintBuilder::convertible(
    LayoutVarID src, LayoutVarID dst, OpOperand &use, bool existing) {
  auto id = graph.addConstraint(ConstraintKind::Convertible,
      ConstraintStrength::Hard, {src, dst}, use.getOwner(), "tensor-use",
      existing ? "existing explicit conversion" : "keep common layout or convert");
  auto &constraint = graph.getConstraints()[id];
  constraint.use = &use;
  constraint.existingConversion = existing;
  constraint.stableUseKey = getStableUseKey(use);
  return success();
}

LogicalResult LayoutConstraintBuilder::transform(
    LayoutVarID src, LayoutVarID dst, Attribute permutation,
    Operation *source, StringRef rule) {
  auto id = graph.addConstraint(ConstraintKind::TransformLayout,
      ConstraintStrength::Hard, {src, dst}, source, rule,
      "coordinate permutation preserves hardware owners");
  graph.getConstraints()[id].coordinateTransform = permutation;
  return success();
}

LogicalResult LayoutConstraintBuilder::storageAccess(
    LayoutVarID distributed, LayoutVarID storage, AccessKind access,
    Operation *source, StringRef rule) {
  auto id = graph.addConstraint(ConstraintKind::StorageAccess,
      ConstraintStrength::Hard, {distributed, storage}, source, rule,
      "storage address is S(D(h)); replicated stores elect one owner");
  graph.getConstraints()[id].access = access;
  graph.addConstraint(ConstraintKind::Preference, ConstraintStrength::Soft,
      {distributed, storage}, source, "coalesced-access", "prefer coalescing");
  return success();
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
LayoutConstraintBuilder::lookup(Value value, LayoutKind kind) const {
  const DenseMap<Value, LayoutVarID> &variablesByValue =
      kind == LayoutKind::Storage ? storageVariablesByValue
                                  : distributedVariablesByValue;
  auto found = variablesByValue.find(value);
  if (found == variablesByValue.end())
    return std::nullopt;
  return found->second;
}

FailureOr<LayoutConstraintGraph>
collectLayoutConstraints(Operation *root, LayoutTarget &target,
                         LayoutCollectionMode mode) {
  LayoutConstraintGraph graph;
  LayoutConstraintBuilder builder(graph);
  DenseMap<Value, SmallVector<LayoutVarID>> viewsBySource;
  SmallVector<Value> sourceOrder;
  bool failedCollection = false;

  root->walk([&](LayoutViewOp view) {
    auto info = analyzeStorageAlias(view.getResult());
    if (failed(info)) { failedCollection = true; return; }
    info->rootKey = getStableValueName(info->root, LayoutKind::Storage, 0);
    LayoutVarID id = builder.getOrCreateStorageVar(view.getResult());
    graph.getVariable(id).storageAlias = *info;
    auto [sourceIt, inserted] = viewsBySource.try_emplace(info->root);
    if (inserted) sourceOrder.push_back(info->root);
    sourceIt->second.push_back(id);
    if (StorageLayoutAttr layout = view.getLayoutAttr())
      failedCollection |= failed(builder.require(id, layout, view, "layout_view"));
  });
  if (failedCollection) return failure();

  for (Value source : sourceOrder) {
    ArrayRef<LayoutVarID> ids = viewsBySource.find(source)->second;
    uint64_t alignment = graph.getVariable(ids.front()).storageAlias->rootAlignment;
    std::string alignmentEvidence =
        graph.getVariable(ids.front()).storageAlias->alignmentEvidence;
    // Only a whole-root binding may declare a root base-alignment contract.
    // Child demands never become root guarantees.
    for (LayoutVarID id : ids) {
      const auto &var = graph.getVariable(id);
      const auto &info = *var.storageAlias;
      if (info.viewType.getShape() != info.rootType.getShape() ||
          !info.viewToRoot.isIdentity()) continue;
      for (const auto &seed : var.candidates)
        if (auto storage = dyn_cast<StorageLayoutAttr>(seed.value)) {
          alignment = std::max<uint64_t>(alignment, storage.getAlignment().getInt());
          alignmentEvidence += "; whole-root binding precondition at " + var.stableName +
              " (not a runtime proof), alignment=" +
              std::to_string(storage.getAlignment().getInt());
        }
    }
    for (auto id : ids) {
      graph.getVariable(id).storageAlias->rootAlignment = alignment;
      graph.getVariable(id).storageAlias->alignmentEvidence = alignmentEvidence;
    }
    // Partial overlap is not transitive: every pair matters, including disjoint
    // logical domains whose proposed physical intervals might collide.
    for (size_t i = 0; i < ids.size(); ++i)
      for (size_t j = i + 1; j < ids.size(); ++j)
        graph.addConstraint(ConstraintKind::AliasLayout, ConstraintStrength::Hard,
            {ids[i], ids[j]}, graph.getVariable(ids[j]).anchor,
            "same-source-layout-view",
            "coordinate and bit-interval agreement on root " +
                graph.getVariable(ids[i]).storageAlias->rootKey);
  }

  root->walk([&](CopyOp copy) {
    if (!isWholeTileStaticCopy(copy)) {
      copy.emitOpError(
          "unsupported storage layout inference for non-whole-tile or "
          "dynamic copy");
      failedCollection = true;
      return;
    }
    if (!copy.getSrc().getDefiningOp<LayoutViewOp>() ||
        !copy.getDst().getDefiningOp<LayoutViewOp>()) {
      copy.emitOpError(
          "M2 storage layout inference requires whole-tile copy operands "
          "to be layout_view results");
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
  failedCollection |= failed(collectDistributedLayoutConstraints(root, graph, builder));
  if (failedCollection)
    return failure();
  if (failed(graph.finalize(root->getLoc())))
    return failure();
  if (mode == LayoutCollectionMode::RelationsOnly)
    return graph;

  if (failed(initializeStorageAliasCandidates(root, graph, target)))
    return failure();

  auto projectCandidatesToFixedPoint = [&]() {
    bool addedCandidate;
    do {
      addedCandidate = false;
      for (const LayoutConstraint &constraint : graph.getConstraints()) {
        if (constraint.strength != ConstraintStrength::Hard ||
            !isEqualityConstraint(constraint.kind))
          continue;
        for (LayoutVarID sourceID : constraint.vars) {
          if (graph.getVariable(sourceID).kind == LayoutKind::Storage) continue;
          SmallVector<LayoutCandidate> sourceCandidates(
              graph.getVariable(sourceID).candidates);
          for (LayoutVarID targetID : constraint.vars) {
            if (sourceID == targetID)
              continue;
            LayoutVar &targetVar = graph.getVariable(targetID);
            if (targetVar.kind == LayoutKind::Storage) continue;
            for (const LayoutCandidate &sourceCandidate : sourceCandidates) {
              FailureOr<Attribute> projected = projectLayoutCandidate(
                  graph, constraint, sourceID, sourceCandidate.value, targetID);
              if (failed(projected) ||
                  llvm::any_of(targetVar.candidates,
                               [&](const LayoutCandidate &known) {
                    return known.value == *projected;
                  }) ||
                  failed(target.verifyCandidate(targetVar, *projected,
                                                root->getLoc())))
                continue;
              std::optional<ProvenanceID> parent;
              if (sourceCandidate.provenance != kInvalidProvenanceID &&
                  sourceCandidate.provenance < graph.getProvenances().size())
                parent = sourceCandidate.provenance;
              Operation *source =
                  graph.getProvenances()[constraint.provenance].source;
              ProvenanceID provenance = graph.addProvenance(
                  parent, source, "constraint-projection",
                  ("candidate projected through " +
                   stringifyConstraintKind(constraint.kind))
                      .str());
              targetVar.candidates.push_back(
                  {*projected, provenance, sourceCandidate.stableOrdinal});
              addedCandidate = true;
            }
          }
        }
      }
    } while (addedCandidate);
  };

  // Let hard seeds initialize connected domains before asking the target for
  // an unconstrained domain. This keeps an arbitrary valid explicit binding
  // from spuriously overflowing the bootstrap domain limit.
  projectCandidatesToFixedPoint();
  for (LayoutVar &var : graph.getVariables()) {
    if (!var.candidates.empty() || var.kind == LayoutKind::Storage)
      continue;
    SmallVector<LayoutCandidate> candidates;
    target.enumerateCandidates(var, candidates);
    for (const LayoutCandidate &candidate : candidates) {
      if (failed(target.verifyCandidate(var, candidate.value,
                                        root->getLoc())) ||
          llvm::any_of(var.candidates, [&](const LayoutCandidate &known) {
            return known.value == candidate.value;
          }))
        continue;
      LayoutCandidate recorded = candidate;
      recorded.provenance = graph.addProvenance(
          std::nullopt, var.anchor ? var.anchor : root, "target-candidate",
          "layout enumerated by the target model");
      var.candidates.push_back(recorded);
    }
    // Initialize one relation domain at a time. A transpose-connected domain
    // receives transformed alternatives, not another independent default set.
    projectCandidatesToFixedPoint();
  }
  projectCandidatesToFixedPoint();

  for (LayoutVar &var : graph.getVariables()) {
    llvm::stable_sort(var.candidates,
                      [](const LayoutCandidate &lhs,
                         const LayoutCandidate &rhs) {
      return std::make_pair(lhs.stableOrdinal, layoutCandidateKey(lhs.value)) <
             std::make_pair(rhs.stableOrdinal, layoutCandidateKey(rhs.value));
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

static LogicalResult runPropagationWorklist(LayoutConstraintGraph &graph,
                                            bool strict) {
  auto &stats = strict ? graph.getPropagationStatistics().strict
                       : graph.getPropagationStatistics().common;
  stats = {};
  stats.ran = true;
  stats.changesByVariable.resize(graph.getVariables().size(), 0);
  SmallVector<SmallVector<LayoutConstraintID>> adjacency(graph.getVariables().size());
  SmallVector<bool> inQueue(graph.getConstraints().size(), false);
  std::deque<LayoutConstraintID> queue;
  auto enqueue = [&](LayoutConstraintID id) {
    if (inQueue[id]) return;
    inQueue[id] = true;
    queue.push_back(id);
    ++stats.enqueues;
    stats.maximumQueue = std::max<uint64_t>(stats.maximumQueue, queue.size());
  };
  for (const auto &relation : graph.getConstraints()) {
    if (relation.strength != ConstraintStrength::Hard ||
        !(isEqualityConstraint(relation.kind) ||
          (strict && relation.kind == ConstraintKind::RequireEncoding)))
      continue;
    enqueue(relation.id);
    for (LayoutVarID id : relation.vars)
      if (!llvm::is_contained(adjacency[id], relation.id))
        adjacency[id].push_back(relation.id);
  }
  stats.initialConstraints = queue.size();
  stats.popUpperBound = stats.staticPopUpperBound = queue.size();
  for (const auto &var : graph.getVariables()) {
    stats.initialCandidates += var.candidates.size();
    stats.staticPopUpperBound += var.candidates.size() * adjacency[var.id].size();
  }
  stats.finalCandidates = stats.initialCandidates;
  while (!queue.empty()) {
    LayoutConstraintID id = queue.front();
    queue.pop_front();
    inQueue[id] = false;
    ++stats.queuePops;
    const auto &relation = graph.getConstraint(id);
    bool require = relation.kind == ConstraintKind::RequireEncoding;
    if (strict && !require &&
        llvm::none_of(relation.vars, [&](LayoutVarID var) {
          return graph.getVariable(var).candidates.size() == 1;
        }))
      continue; // Remains in adjacency: a later seed can enable this relation.
    SmallVector<std::pair<LayoutVarID, size_t>> before;
    for (LayoutVarID var : relation.vars)
      if (llvm::none_of(before, [&](auto entry) { return entry.first == var; }))
        before.emplace_back(var, graph.getVariable(var).candidates.size());
    LogicalResult result = success();
    if (require) {
      auto &var = graph.getVariable(relation.vars.front());
      llvm::erase_if(var.candidates, [&](const auto &candidate) {
        return candidate.value != relation.requiredEncoding;
      });
      updateState(var);
      if (var.state == LayoutState::Conflict)
        result = emitConflict(graph, relation);
    } else {
      bool changed = false;
      result = applyEqualityConstraint(graph, relation, changed);
    }
    for (auto [var, size] : before) {
      size_t after = graph.getVariable(var).candidates.size();
      assert(after <= size && "propagation must only shrink frozen domains");
      if (after == size) continue;
      stats.deletedCandidates += size - after;
      stats.finalCandidates -= size - after;
      ++stats.domainChanges;
      ++stats.changesByVariable[var];
      stats.popUpperBound += adjacency[var].size();
      for (auto affected : adjacency[var]) enqueue(affected);
    }
    assert(stats.hasValidBounds() && "invalid monotone propagation accounting");
    if (failed(result)) return failure();
  }
  assert(stats.hasValidBounds() && "invalid fixed-point accounting");
  return success();
}

LogicalResult propagateStrict(LayoutConstraintGraph &graph) {
  graph.getPropagationStatistics().common = {};
  return runPropagationWorklist(graph, true);
}

LogicalResult propagateCommonToFixedPoint(LayoutConstraintGraph &graph) {
  if (failed(propagateStrict(graph))) return failure();
  return runPropagationWorklist(graph, false);
}

} // namespace mlir::frisk
