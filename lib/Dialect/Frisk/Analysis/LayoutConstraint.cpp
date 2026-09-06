#include "Dialect/Frisk/Analysis/LayoutConstraint.h"

#include <algorithm>
#include <numeric>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {

StringRef stringifyConstraintKind(ConstraintKind kind) {
  switch (kind) {
  case ConstraintKind::SameLayout:
    return "same-layout";
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
         kind == ConstraintKind::AliasLayout ||
         kind == ConstraintKind::StorageAccess;
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
  SmallVector<LayoutVar> sortedVariables;
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
    const LayoutProvenance &lhsProv = provenances[lhs.provenance];
    const LayoutProvenance &rhsProv = provenances[rhs.provenance];
    return std::tie(lhsProv.rule, lhsProv.reason) <
           std::tie(rhsProv.rule, rhsProv.reason);
  });
  for (auto [id, constraint] : llvm::enumerate(constraints))
    constraint.id = id;
  finalized = true;
  return verifyInvariants(loc);
}

LogicalResult LayoutConstraintGraph::verifyInvariants(Location loc) const {
  llvm::SmallDenseSet<StringRef, 16> names;
  for (const LayoutVar &var : variables) {
    if (var.stableName.empty())
      return emitError(loc) << "layout variable has an empty stable name";
    if (!names.insert(var.stableName).second)
      return emitError(loc) << "duplicate layout variable stable name '"
                            << var.stableName << "'";
    if (!var.shapedType || !isa<ShapedType>(var.shapedType))
      return emitError(loc) << "layout variable '" << var.stableName
                            << "' does not have a shaped type";
  }

  for (const LayoutConstraint &constraint : constraints) {
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
    if (constraint.kind == ConstraintKind::RequireEncoding &&
        !constraint.requiredEncoding)
      return emitError(loc)
             << "require-encoding constraint is missing its encoding";
  }

  for (const LayoutProvenance &provenance : provenances)
    if (provenance.parent && *provenance.parent >= provenances.size())
      return emitError(loc) << "layout provenance has invalid parent";
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

void LayoutConstraintGraph::print(raw_ostream &os) const {
  for (const LayoutVar &var : variables)
    os << "var " << var.id << " " << var.stableName << "\n";
  for (const LayoutConstraint &constraint : constraints) {
    os << "constraint " << constraint.id << " "
       << stringifyConstraintStrength(constraint.strength) << " "
       << stringifyConstraintKind(constraint.kind) << " [";
    llvm::interleaveComma(constraint.vars, os);
    const LayoutProvenance &provenance = provenances[constraint.provenance];
    os << "] " << provenance.rule << ": " << provenance.reason << "\n";
  }
}

raw_ostream &operator<<(raw_ostream &os,
                        const LayoutConstraintGraph &graph) {
  graph.print(os);
  return os;
}

} // namespace mlir::frisk
