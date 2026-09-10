#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"

#include <functional>

#include "Dialect/Frisk/IR/FriskAttributes.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {

namespace {

bool isSupportedBootstrapHardConstraint(ConstraintKind kind) {
  return kind == ConstraintKind::RequireEncoding ||
         isSupportedLayoutRelation(kind);
}

bool satisfiesConstraint(const LayoutConstraintGraph &graph,
                         const LayoutConstraint &constraint,
                         const DenseMap<LayoutVarID, Attribute> &assignment,
                         bool requireComplete) {
  if (constraint.strength != ConstraintStrength::Hard)
    return true;
  for (LayoutVarID id : constraint.vars)
    if (!assignment.count(id))
      return !requireComplete;

  if (constraint.kind == ConstraintKind::RequireEncoding)
    return assignment.lookup(constraint.vars.front()) ==
           constraint.requiredEncoding;
  if (!isSupportedBootstrapHardConstraint(constraint.kind))
    return false;
  for (size_t index = 1; index < constraint.vars.size(); ++index)
    if (!layoutRelationCompatible(graph, constraint,
            constraint.vars.front(), assignment.lookup(constraint.vars.front()),
            constraint.vars[index], assignment.lookup(constraint.vars[index])))
      return false;
  return true;
}

Location getVariableLoc(const LayoutVar &var) {
  return var.anchor ? var.anchor->getLoc()
                    : UnknownLoc::get(var.shapedType.getContext());
}

LogicalResult emitSolverLimit(const LayoutVar &var, StringRef detail) {
  return emitError(getVariableLoc(var))
         << "bootstrap layout solver limit exceeded for " << var.stableName
         << ": " << detail;
}

LogicalResult verifyStorageCapacity(const LayoutVar &var,
                                    Attribute candidate) {
  if (var.kind != LayoutKind::Storage)
    return success();
  Location loc = getVariableLoc(var);
  auto type = dyn_cast<MemRefType>(var.shapedType);
  auto storage = dyn_cast<StorageLayoutAttr>(candidate);
  if (!type || !storage)
    return emitError(loc) << "storage solution for " << var.stableName
                          << " has an incompatible type or encoding";
  FailureOr<uint64_t> required = getStorageFootprintBytes(storage, type);
  FailureOr<uint64_t> capacity = getMemRefStaticCapacityBytes(type);
  if (failed(required) || failed(capacity))
    return emitError(loc)
           << "cannot prove storage layout footprint for " << var.stableName
           << " fits the underlying memref type";
  if (*required > *capacity)
    return emitError(loc) << "storage layout requires " << *required
                          << " bytes but underlying memref type provides "
                          << *capacity << " bytes for " << var.stableName;
  return success();
}

SmallVector<SmallVector<LayoutVarID>>
getHardConstraintComponents(const LayoutConstraintGraph &graph) {
  SmallVector<SmallVector<LayoutVarID>> adjacency(graph.getVariables().size());
  for (const LayoutConstraint &constraint : graph.getConstraints()) {
    if (constraint.strength != ConstraintStrength::Hard ||
        constraint.vars.size() < 2)
      continue;
    for (size_t index = 1; index < constraint.vars.size(); ++index) {
      adjacency[constraint.vars.front()].push_back(constraint.vars[index]);
      adjacency[constraint.vars[index]].push_back(constraint.vars.front());
    }
  }

  SmallVector<SmallVector<LayoutVarID>> components;
  SmallVector<bool> visited(graph.getVariables().size());
  for (LayoutVarID start = 0; start < graph.getVariables().size(); ++start) {
    if (visited[start])
      continue;
    SmallVector<LayoutVarID> component;
    SmallVector<LayoutVarID> worklist = {start};
    visited[start] = true;
    while (!worklist.empty()) {
      LayoutVarID id = worklist.pop_back_val();
      component.push_back(id);
      llvm::sort(adjacency[id]);
      for (LayoutVarID next : adjacency[id]) {
        if (!visited[next]) {
          visited[next] = true;
          worklist.push_back(next);
        }
      }
    }
    llvm::sort(component);
    components.push_back(std::move(component));
  }
  return components;
}

} // namespace

FailureOr<LayoutSolution>
solveBootstrapLayoutGraph(LayoutConstraintGraph &graph, LayoutTarget &,
                          BootstrapSolverLimits limits) {
  for (const LayoutConstraint &constraint : graph.getConstraints()) {
    if (constraint.strength != ConstraintStrength::Hard ||
        isSupportedBootstrapHardConstraint(constraint.kind))
      continue;
    const LayoutProvenance &provenance =
        graph.getProvenances()[constraint.provenance];
    Location loc = provenance.source
                       ? provenance.source->getLoc()
                       : UnknownLoc::get(graph.getVariable(
                                                 constraint.vars.front())
                                             .shapedType.getContext());
    emitError(loc) << "bootstrap layout solver does not support hard "
                      "constraint '"
                   << stringifyConstraintKind(constraint.kind) << "'";
    return failure();
  }
  LayoutSolution solution;
  for (SmallVector<LayoutVarID> &component :
       getHardConstraintComponents(graph)) {
    if (component.size() > limits.maxVariables) {
      const LayoutVar &var = graph.getVariable(component.front());
      if (failed(emitSolverLimit(var, "too many variables")))
        return failure();
    }
    for (LayoutVarID id : component) {
      LayoutVar &var = graph.getVariable(id);
      if (var.candidates.empty()) {
        emitError(getVariableLoc(var))
            << "unresolved storage layout for layout variable "
            << var.stableName;
        return failure();
      }
      if (var.candidates.size() > limits.maxDomainSize) {
        if (failed(emitSolverLimit(var, "candidate domain is too large")))
          return failure();
      }
      llvm::stable_sort(var.candidates,
                        [](const LayoutCandidate &lhs,
                           const LayoutCandidate &rhs) {
        return std::make_pair(lhs.stableOrdinal, layoutCandidateKey(lhs.value)) <
               std::make_pair(rhs.stableOrdinal, layoutCandidateKey(rhs.value));
      });
    }

    DenseMap<LayoutVarID, Attribute> trial = solution.assignments;
    bool found = false;
    size_t bestConversions = std::numeric_limits<size_t>::max();
    SmallVector<LayoutConversionEdge> bestEdges;
    auto countConversions = [&]() {
      SmallVector<LayoutConversionEdge> edges;
      for (const LayoutConstraint &constraint : graph.getConstraints()) {
        if (constraint.kind != ConstraintKind::Convertible ||
            constraint.existingConversion ||
            !llvm::is_contained(component, constraint.vars.front()))
          continue;
        Attribute source = trial.lookup(constraint.vars[0]);
        Attribute target = trial.lookup(constraint.vars[1]);
        if (!source || !target || layoutEncodingsEqual(source, target))
          continue;
        LayoutConversionEdge edge;
        edge.use = constraint.use;
        edge.sourceEncoding = source;
        edge.targetEncoding = target;
        edge.resolution = EdgeResolutionKind::Convert;
        edge.constraint = constraint.id;
        edges.push_back(edge);
      }
      llvm::sort(edges, [&](const auto &lhs, const auto &rhs) {
        return graph.getConstraint(lhs.constraint).stableUseKey <
               graph.getConstraint(rhs.constraint).stableUseKey;
      });
      return edges;
    };
    std::function<void(size_t)> search = [&](size_t index) {
      if (index == component.size()) {
        if (llvm::all_of(graph.getConstraints(),
                         [&](const LayoutConstraint &constraint) {
              return satisfiesConstraint(graph, constraint, trial,
                                         /*requireComplete=*/false);
            })) {
          auto edges = countConversions();
          // Stable DFS assignment order supplies the last tie-break. Do not
          // stop at the first feasible assignment: conversion count comes first.
          if (!found || edges.size() < bestConversions) {
            found = true;
            bestConversions = edges.size();
            bestEdges = std::move(edges);
            for (LayoutVarID id : component)
              solution.assignments[id] = trial.lookup(id);
          }
        }
        return;
      }
      LayoutVarID id = component[index];
      for (const LayoutCandidate &candidate : graph.getVariable(id).candidates) {
        trial[id] = candidate.value;
        bool viable = llvm::all_of(
            graph.getConstraints(), [&](const LayoutConstraint &constraint) {
              return satisfiesConstraint(graph, constraint, trial,
                                         /*requireComplete=*/false);
            });
        if (viable)
          search(index + 1);
        trial.erase(id);
      }
    };
    search(0);
    if (!found) {
      const LayoutVar &var = graph.getVariable(component.front());
      emitError(getVariableLoc(var))
          << "no feasible bootstrap layout assignment for component rooted at "
          << var.stableName;
      return failure();
    }
    llvm::append_range(solution.conversions, bestEdges);
  }
  llvm::sort(solution.conversions, [&](const auto &lhs, const auto &rhs) {
    return graph.getConstraint(lhs.constraint).stableUseKey <
           graph.getConstraint(rhs.constraint).stableUseKey;
  });
  return solution;
}

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target, Location loc) {
  llvm::SmallDenseSet<LayoutConstraintID> converted;
  for (const LayoutConversionEdge &edge : solution.conversions) {
    if (edge.constraint >= graph.getConstraints().size())
      return emitError(loc) << "conversion does not identify an authorized graph edge";
    const auto &constraint = graph.getConstraint(edge.constraint);
    if (constraint.kind != ConstraintKind::Convertible || constraint.existingConversion ||
        !edge.use || constraint.use != edge.use ||
        edge.resolution != EdgeResolutionKind::Convert ||
        !converted.insert(edge.constraint).second ||
        edge.sourceEncoding != solution.assignments.lookup(constraint.vars[0]) ||
        edge.targetEncoding != solution.assignments.lookup(constraint.vars[1]) ||
        layoutEncodingsEqual(edge.sourceEncoding, edge.targetEncoding))
      return emitError(loc) << "invalid, duplicate, or identity conversion in layout solution";
    const auto &source = graph.getVariable(constraint.vars[0]);
    if (!source.value || edge.use->get() != source.value)
      return emitError(loc) << "conversion SSA use no longer matches its producer";
  }
  for (const LayoutVar &var : graph.getVariables()) {
    auto found = solution.assignments.find(var.id);
    if (found == solution.assignments.end())
      return emitError(loc)
             << "unresolved storage layout for layout variable "
             << var.stableName;
    if (llvm::none_of(var.candidates, [&](const LayoutCandidate &candidate) {
          return candidate.value == found->second;
        }))
      return emitError(loc) << "solution for " << var.stableName
                            << " is outside its candidate domain";
    if (failed(target.verifyCandidate(var, found->second,
                                      getVariableLoc(var))))
      return failure();
    if (failed(verifyStorageCapacity(var, found->second)))
      return failure();
  }
  for (const LayoutConstraint &constraint : graph.getConstraints()) {
    if (constraint.kind == ConstraintKind::Convertible && !constraint.existingConversion &&
        !layoutEncodingsEqual(solution.assignments.lookup(constraint.vars[0]),
                              solution.assignments.lookup(constraint.vars[1])) &&
        !converted.count(constraint.id))
      return emitError(loc) << "layout solution is missing a required consumer conversion";
    if (satisfiesConstraint(graph, constraint, solution.assignments,
                            /*requireComplete=*/true))
      continue;
    const LayoutProvenance &provenance =
        graph.getProvenances()[constraint.provenance];
    InFlightDiagnostic diagnostic = emitError(loc)
                                    << "solution violates hard layout constraint '"
                                    << provenance.rule << "': "
                                    << provenance.reason;
    std::string chain;
    llvm::raw_string_ostream stream(chain);
    if (succeeded(graph.printProvenanceChain(constraint.provenance, stream)))
      diagnostic.attachNote(loc) << chain;
    return failure();
  }
  return success();
}

} // namespace mlir::frisk
