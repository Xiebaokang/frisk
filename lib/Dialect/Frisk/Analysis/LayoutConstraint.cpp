#include "Dialect/Frisk/Analysis/LayoutConstraint.h"

#include <algorithm>
#include <numeric>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::frisk {

void PropagationPhaseStatistics::print(StringRef phase, raw_ostream &os) const {
  if (!ran) return;
  os << "propagation " << phase << " initial=" << initialCandidates
     << " final=" << finalCandidates << " deleted=" << deletedCandidates
     << " changes=" << domainChanges << " pops=" << queuePops
     << " initial-constraints=" << initialConstraints
     << " enqueues=" << enqueues << " max-queue=" << maximumQueue
     << " pop-bound=" << popUpperBound
     << " static-pop-bound=" << staticPopUpperBound << "\n";
  os << "domain-changes " << phase << " [";
  llvm::interleaveComma(changesByVariable, os);
  os << "]\n";
}

StringRef stringifyRegionLayoutEdgeKind(RegionLayoutEdgeKind kind) {
  switch (kind) {
  case RegionLayoutEdgeKind::IfYield: return "if-yield";
  case RegionLayoutEdgeKind::ForInit: return "for-init";
  case RegionLayoutEdgeKind::ForBackedge: return "for-backedge";
  case RegionLayoutEdgeKind::ForResult: return "for-result";
  case RegionLayoutEdgeKind::WhileInit: return "while-init";
  case RegionLayoutEdgeKind::WhileBackedge: return "while-backedge";
  case RegionLayoutEdgeKind::WhileCondition: return "while-condition";
  case RegionLayoutEdgeKind::WhileResult: return "while-result";
  }
  llvm_unreachable("unknown region layout edge");
}

StringRef stringifyConstraintKind(ConstraintKind kind) {
  switch (kind) {
  case ConstraintKind::SameLayout:
    return "same-layout";
  case ConstraintKind::Convertible:
    return "convertible";
  case ConstraintKind::TransformLayout:
    return "transform-layout";
  case ConstraintKind::RequireEncoding:
    return "require-encoding";
  case ConstraintKind::InstructionContract:
    return "instruction-contract";
  case ConstraintKind::StorageAccess:
    return "storage-access";
  case ConstraintKind::AliasLayout:
    return "alias-layout";
  case ConstraintKind::Ownership:
    return "ownership";
  case ConstraintKind::ResourceLimit:
    return "resource-limit";
  case ConstraintKind::Preference:
    return "preference";
  }
  llvm_unreachable("unknown constraint kind");
}

StringRef stringifyConstraintStrength(ConstraintStrength strength) {
  return strength == ConstraintStrength::Hard ? "hard" : "soft";
}

LayoutVarID LayoutConstraintGraph::addVariable(LayoutKind kind, Type type,
                                               StringRef stableName,
                                               Operation *anchor) {
  finalized = false;
  LayoutVarID id = variables.size();
  variables.push_back(
      {id, kind, type, {}, LayoutState::Uninitialized, stableName.str(), anchor});
  return id;
}

ProvenanceID LayoutConstraintGraph::addProvenance(
    std::optional<ProvenanceID> parent, Operation *source, StringRef rule,
    StringRef reason) {
  ProvenanceID id = provenances.size();
  provenances.push_back(
      {id, parent, source, rule.str(), reason.str()});
  return id;
}

LayoutConstraintID LayoutConstraintGraph::addConstraint(
    ConstraintKind kind, ConstraintStrength strength,
    ArrayRef<LayoutVarID> vars, Operation *source, StringRef rule,
    StringRef reason, Attribute requiredEncoding) {
  finalized = false;
  ProvenanceID provenance =
      addProvenance(std::nullopt, source, rule, reason);
  LayoutConstraintID id = constraints.size();
  constraints.push_back(
      {id, kind, strength, SmallVector<LayoutVarID>(vars), requiredEncoding,
       provenance});
  return id;
}

static bool isCommutativeConstraint(ConstraintKind kind) {
  return kind == ConstraintKind::SameLayout ||
         kind == ConstraintKind::AliasLayout;
}

static std::string getStableAttributeText(Attribute attribute) {
  if (!attribute)
    return {};
  std::string text;
  llvm::raw_string_ostream(text) << attribute;
  return text;
}

LogicalResult LayoutConstraintGraph::finalize(Location loc) {
  if (failed(verifyInvariants(loc)))
    return failure();

  SmallVector<unsigned> order(variables.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
    return variables[lhs].stableName < variables[rhs].stableName;
  });

  SmallVector<LayoutVarID> remap(variables.size());
  SmallVector<LayoutVar, 0> sortedVariables;
  sortedVariables.reserve(variables.size());
  for (auto [newID, oldID] : llvm::enumerate(order)) {
    remap[oldID] = newID;
    sortedVariables.push_back(std::move(variables[oldID]));
    sortedVariables.back().id = newID;
  }
  variables = std::move(sortedVariables);

  for (LayoutConstraint &constraint : constraints) {
    for (LayoutVarID &id : constraint.vars)
      id = remap[id];
    if (isCommutativeConstraint(constraint.kind))
      llvm::sort(constraint.vars);
  }

  llvm::stable_sort(constraints,
                    [&](const LayoutConstraint &lhs,
                        const LayoutConstraint &rhs) {
    if (lhs.kind != rhs.kind)
      return static_cast<unsigned>(lhs.kind) < static_cast<unsigned>(rhs.kind);
    if (lhs.strength != rhs.strength)
      return static_cast<unsigned>(lhs.strength) <
             static_cast<unsigned>(rhs.strength);
    if (lhs.vars != rhs.vars)
      return std::lexicographical_compare(lhs.vars.begin(), lhs.vars.end(),
                                          rhs.vars.begin(), rhs.vars.end());
    std::string lhsEncoding = getStableAttributeText(lhs.requiredEncoding);
    std::string rhsEncoding = getStableAttributeText(rhs.requiredEncoding);
    if (lhsEncoding != rhsEncoding)
      return lhsEncoding < rhsEncoding;
    if (lhs.stableUseKey != rhs.stableUseKey)
      return lhs.stableUseKey < rhs.stableUseKey;
    auto lhsTransform = getStableAttributeText(lhs.coordinateTransform);
    auto rhsTransform = getStableAttributeText(rhs.coordinateTransform);
    if (lhsTransform != rhsTransform)
      return lhsTransform < rhsTransform;
    const LayoutProvenance &lhsProv = provenances[lhs.provenance];
    const LayoutProvenance &rhsProv = provenances[rhs.provenance];
    return std::tie(lhsProv.rule, lhsProv.reason) <
           std::tie(rhsProv.rule, rhsProv.reason);
  });
  SmallVector<LayoutConstraintID> constraintRemap(constraints.size());
  for (auto [id, constraint] : llvm::enumerate(constraints)) {
    constraintRemap[constraint.id] = id;
    constraint.id = id;
  }
  for (auto &edge : regionEdges) {
    edge.source = remap[edge.source];
    edge.target = remap[edge.target];
    edge.constraint = constraintRemap[edge.constraint];
  }
  // A changed graph invalidates the last run's ID-indexed statistics.
  propagationStatistics = {};
  aliasFootprints.clear();
  aliasPairProofs.clear();
  preparationStatistics = {};
  llvm::sort(regionEdges, [](const auto &lhs, const auto &rhs) {
    return lhs.stableKey < rhs.stableKey;
  });
  finalized = true;
  return verifyInvariants(loc);
}

LogicalResult LayoutConstraintGraph::verifyInvariants(Location loc) const {
  llvm::SmallDenseSet<StringRef, 16> names;
  for (auto [index, var] : llvm::enumerate(variables)) {
    if (var.id != index)
      return emitError(loc) << "layout variable ID differs from its array position";
    if (var.stableName.empty())
      return emitError(loc) << "layout variable has an empty stable name";
    if (!names.insert(var.stableName).second)
      return emitError(loc) << "duplicate layout variable stable name '"
                            << var.stableName << "'";
    if (!var.shapedType || !isa<ShapedType>(var.shapedType))
      return emitError(loc) << "layout variable '" << var.stableName
                            << "' does not have a shaped type";
    if (var.storageAlias) {
      const auto &alias = *var.storageAlias;
      if (var.kind != LayoutKind::Storage || !alias.root || !alias.rootType ||
          !alias.viewType || alias.viewType != var.shapedType ||
          !alias.viewToRoot || alias.rootKey.empty() ||
          alias.viewToRoot.getNumDims() != unsigned(alias.viewType.getRank()) ||
          alias.viewToRoot.getNumResults() != unsigned(alias.rootType.getRank()) ||
          alias.lowerBit >= alias.upperBit)
        return emitError(loc) << "invalid storage alias endpoint metadata";
    }
  }

  for (auto [index, constraint] : llvm::enumerate(constraints)) {
    if (constraint.id != index)
      return emitError(loc) << "layout constraint ID differs from its array position";
    if ((constraint.kind == ConstraintKind::Convertible ||
         constraint.kind == ConstraintKind::TransformLayout ||
         constraint.kind == ConstraintKind::StorageAccess ||
         constraint.kind == ConstraintKind::AliasLayout) &&
        constraint.vars.size() != 2)
      return emitError(loc) << "binary layout relation requires two endpoints";
    if (constraint.kind == ConstraintKind::Convertible &&
        (!constraint.use || constraint.stableUseKey.empty()))
      return emitError(loc) << "convertible relation requires a stable SSA use";
    if (constraint.kind == ConstraintKind::TransformLayout &&
        !constraint.coordinateTransform)
      return emitError(loc) << "transform relation requires a permutation";
    if (constraint.strength == ConstraintStrength::Hard &&
        constraint.vars.empty())
      return emitError(loc) << "hard layout constraint '"
                            << stringifyConstraintKind(constraint.kind)
                            << "' must reference at least one variable";
    for (LayoutVarID id : constraint.vars)
      if (id >= variables.size())
        return emitError(loc) << "layout constraint references invalid variable "
                              << id;
    if (constraint.provenance >= provenances.size())
      return emitError(loc) << "layout constraint has invalid provenance";
    if (constraint.kind == ConstraintKind::AliasLayout) {
      const auto &lhs = variables[constraint.vars[0]];
      const auto &rhs = variables[constraint.vars[1]];
      if (lhs.kind != LayoutKind::Storage || rhs.kind != LayoutKind::Storage ||
          !lhs.storageAlias || !rhs.storageAlias ||
          lhs.storageAlias->root != rhs.storageAlias->root ||
          lhs.storageAlias->rootType != rhs.storageAlias->rootType)
        return emitError(loc) << "alias relation requires two proven common-root storage endpoints";
    }
    if (constraint.kind == ConstraintKind::RequireEncoding) {
      if (constraint.vars.size() != 1)
        return emitError(loc)
               << "require-encoding constraint must reference exactly one "
                  "variable";
      if (!constraint.requiredEncoding)
        return emitError(loc)
               << "require-encoding constraint is missing its encoding";
    }
    if (isCommutativeConstraint(constraint.kind) &&
        constraint.vars.size() < 2)
      return emitError(loc) << "layout equality constraint '"
                            << stringifyConstraintKind(constraint.kind)
                            << "' must reference at least two variables";
  }

  llvm::SmallDenseSet<StringRef, 16> regionKeys;
  for (const auto &edge : regionEdges) {
    if (edge.source >= variables.size() || edge.target >= variables.size() ||
        edge.constraint >= constraints.size())
      return emitError(loc) << "region layout edge references invalid ID";
    if (!edge.owner || edge.stableKey.empty() ||
        !regionKeys.insert(edge.stableKey).second)
      return emitError(loc) << "region layout edge requires unique stable identity";
    const auto &relation = constraints[edge.constraint];
    bool resultEdge = edge.kind == RegionLayoutEdgeKind::ForResult ||
                      edge.kind == RegionLayoutEdgeKind::WhileResult;
    if (relation.strength != ConstraintStrength::Hard ||
        relation.vars.size() != 2 ||
        !llvm::is_contained(relation.vars, edge.source) ||
        !llvm::is_contained(relation.vars, edge.target) ||
        variables[edge.source].kind != LayoutKind::Distributed ||
        variables[edge.target].kind != LayoutKind::Distributed ||
        (resultEdge ? (edge.use || relation.kind != ConstraintKind::SameLayout)
                    : (!edge.use || relation.use != edge.use ||
                       relation.kind != ConstraintKind::Convertible)))
      return emitError(loc) << "region layout edge does not match its hard relation";
    if (edge.use && edge.use->getOwner() != edge.owner &&
        edge.use->getOwner()->getParentOp() != edge.owner)
      return emitError(loc) << "region layout use is outside its owner boundary";
    // A commutative slot relation must not erase the direction of its region
    // edge. Check actual SCF operands/arguments, not just ID membership.
    Value source, target;
    OpOperand *use = nullptr;
    auto operand = [&](Operation *op, unsigned index) -> OpOperand * {
      return op && index < op->getNumOperands() ? &op->getOpOperand(index) : nullptr;
    };
    auto terminator = [](Region &region) -> Operation * {
      return region.empty() || region.front().empty() ? nullptr : region.front().getTerminator();
    };
    if (auto loop = dyn_cast<scf::ForOp>(edge.owner)) {
      if (edge.slot < loop.getNumResults() && !loop.getRegion().empty()) {
        if (edge.kind == RegionLayoutEdgeKind::ForResult) {
          source = loop.getRegionIterArgs()[edge.slot];
          target = loop.getResult(edge.slot);
        } else if (edge.kind == RegionLayoutEdgeKind::ForInit) {
          use = &loop.getInitsMutable()[edge.slot];
          target = loop.getRegionIterArgs()[edge.slot];
        } else if (edge.kind == RegionLayoutEdgeKind::ForBackedge) {
          use = operand(terminator(loop.getRegion()), edge.slot);
          target = loop.getResult(edge.slot);
        }
      }
    } else if (auto loop = dyn_cast<scf::WhileOp>(edge.owner)) {
      if (edge.kind == RegionLayoutEdgeKind::WhileInit ||
          edge.kind == RegionLayoutEdgeKind::WhileBackedge) {
        if (!loop.getBefore().empty() && edge.slot < loop.getBeforeArguments().size()) {
          target = loop.getBeforeArguments()[edge.slot];
          use = edge.kind == RegionLayoutEdgeKind::WhileInit
              ? operand(loop, edge.slot) : operand(terminator(loop.getAfter()), edge.slot);
        }
      } else if (edge.slot < loop.getNumResults() && !loop.getAfter().empty()) {
        target = loop.getResult(edge.slot);
        if (edge.kind == RegionLayoutEdgeKind::WhileResult)
          source = loop.getAfterArguments()[edge.slot];
        else if (edge.kind == RegionLayoutEdgeKind::WhileCondition)
          use = operand(terminator(loop.getBefore()), edge.slot + 1);
      }
    } else if (auto branch = dyn_cast<scf::IfOp>(edge.owner)) {
      if (edge.kind == RegionLayoutEdgeKind::IfYield && edge.use &&
          isa<scf::YieldOp>(edge.use->getOwner()) && edge.slot < branch.getNumResults()) {
        target = branch.getResult(edge.slot);
        use = operand(edge.use->getOwner(), edge.slot);
      }
    }
    if (use) source = use->get();
    if (!source || !target || use != edge.use ||
        variables[edge.source].value != source || variables[edge.target].value != target ||
        (!resultEdge && (relation.vars[0] != edge.source || relation.vars[1] != edge.target)))
      return emitError(loc) << "region layout edge has invalid owner, slot or direction";
  }

  for (const LayoutProvenance &provenance : provenances) {
    if (provenance.parent && *provenance.parent >= provenances.size())
      return emitError(loc) << "layout provenance has invalid parent";
    llvm::SmallDenseSet<ProvenanceID, 8> visited;
    const LayoutProvenance *current = &provenance;
    while (current->parent) {
      if (!visited.insert(current->id).second)
        return emitError(loc) << "layout provenance contains a cycle";
      if (*current->parent >= provenances.size())
        return emitError(loc) << "layout provenance has invalid parent";
      current = &provenances[*current->parent];
    }
  }
  return success();
}

LogicalResult
LayoutConstraintGraph::printProvenanceChain(ProvenanceID id,
                                            raw_ostream &os) const {
  llvm::SmallDenseSet<ProvenanceID, 8> visited;
  while (true) {
    if (id >= provenances.size())
      return failure();
    if (!visited.insert(id).second)
      return failure();
    const LayoutProvenance &provenance = provenances[id];
    os << provenance.rule << ": " << provenance.reason << '\n';
    if (!provenance.parent)
      return success();
    id = *provenance.parent;
  }
}

std::optional<LayoutVarID>
LayoutConstraintGraph::lookupVariable(StringRef stableName) const {
  auto it = llvm::find_if(variables, [&](const LayoutVar &var) {
    return var.stableName == stableName;
  });
  if (it == variables.end())
    return std::nullopt;
  return it->id;
}

std::optional<LayoutVarID>
LayoutConstraintGraph::lookupVariable(Value value) const {
  for (const LayoutVar &var : variables)
    if (var.value && var.value == value)
      return var.id;
  return std::nullopt;
}

void LayoutConstraintGraph::print(raw_ostream &os) const {
  for (const LayoutVar &var : variables) {
    os << "var " << var.id << " " << var.stableName << "\n";
    if (var.storageAlias)
      os << "alias-endpoint " << var.id << " root=" << var.storageAlias->rootKey
         << " transform=" << var.storageAlias->viewToRoot << " bits=["
         << var.storageAlias->lowerBit << "," << var.storageAlias->upperBit
         << ") alignment=" << var.storageAlias->rootAlignment
         << " evidence=" << var.storageAlias->alignmentEvidence << "\n";
  }
  for (const LayoutConstraint &constraint : constraints) {
    os << "constraint " << constraint.id << " "
       << stringifyConstraintStrength(constraint.strength) << " "
       << stringifyConstraintKind(constraint.kind) << " [";
    llvm::interleaveComma(constraint.vars, os);
    const LayoutProvenance &provenance = provenances[constraint.provenance];
    os << "] " << provenance.rule << ": " << provenance.reason << "\n";
  }
  for (const auto &edge : regionEdges)
    os << "region-edge " << stringifyRegionLayoutEdgeKind(edge.kind)
       << " slot=" << edge.slot << " " << edge.source << " -> " << edge.target
       << " constraint=" << edge.constraint << " " << edge.stableKey << "\n";
  propagationStatistics.strict.print("strict", os);
  propagationStatistics.common.print("common", os);
  os << "candidate-preparation origins=" << preparationStatistics.origins
     << " projected=" << preparationStatistics.projectedCandidates
     << " footprints=" << preparationStatistics.footprintEvaluations
     << " pair-proofs=" << preparationStatistics.pairProofEvaluations << "\n";
}

raw_ostream &operator<<(raw_ostream &os,
                        const LayoutConstraintGraph &graph) {
  graph.print(os);
  return os;
}

} // namespace mlir::frisk
