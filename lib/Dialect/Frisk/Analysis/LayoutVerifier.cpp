#include "Dialect/Frisk/Analysis/LayoutVerifier.h"

#include <functional>

#include "Dialect/Frisk/IR/FriskAttributes.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {

namespace {

bool candidatesCompatible(ConstraintKind kind, Attribute lhs, Attribute rhs) {
  if (kind == ConstraintKind::StorageAccess) {
    auto lhsStorage = dyn_cast<StorageLayoutAttr>(lhs);
    auto rhsStorage = dyn_cast<StorageLayoutAttr>(rhs);
    if (lhsStorage && rhsStorage)
      return lhsStorage.getMap() == rhsStorage.getMap();
  }
  return lhs == rhs;
}

bool satisfiesConstraint(const LayoutConstraint &constraint,
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
  if (constraint.kind != ConstraintKind::SameLayout &&
      constraint.kind != ConstraintKind::AliasLayout &&
      constraint.kind != ConstraintKind::StorageAccess)
    return true;
  for (size_t index = 1; index < constraint.vars.size(); ++index)
    if (!candidatesCompatible(
            constraint.kind, assignment.lookup(constraint.vars.front()),
            assignment.lookup(constraint.vars[index])))
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
        return lhs.stableOrdinal < rhs.stableOrdinal;
      });
    }

    DenseMap<LayoutVarID, Attribute> trial = solution.assignments;
    bool found = false;
    std::function<void(size_t)> search = [&](size_t index) {
      if (found)
        return;
      if (index == component.size()) {
        if (llvm::all_of(graph.getConstraints(),
                         [&](const LayoutConstraint &constraint) {
              return satisfiesConstraint(constraint, trial,
                                         /*requireComplete=*/false);
            })) {
          found = true;
          for (LayoutVarID id : component)
            solution.assignments[id] = trial.lookup(id);
        }
        return;
      }
      LayoutVarID id = component[index];
      for (const LayoutCandidate &candidate : graph.getVariable(id).candidates) {
        trial[id] = candidate.value;
        bool viable = llvm::all_of(
            graph.getConstraints(), [&](const LayoutConstraint &constraint) {
              return satisfiesConstraint(constraint, trial,
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
  }
  return solution;
}

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target, Location loc) {
  if (!solution.conversions.empty())
    return emitError(loc)
           << "bootstrap storage solver must not materialize conversions";
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
    if (failed(target.verifyCandidate(var, found->second, loc)))
      return failure();
  }
  for (const LayoutConstraint &constraint : graph.getConstraints()) {
    if (satisfiesConstraint(constraint, solution.assignments,
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
