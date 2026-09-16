#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include <deque>

namespace mlir::frisk {
namespace {
/// A proposal may weaken metadata, never strengthen the root's guarantees.
StorageLayoutAttr capGuarantees(StorageLayoutAttr candidate,
                                const StorageAliasInfo &info) {
  Builder b(candidate.getContext());
  uint64_t alignment = std::min<uint64_t>(candidate.getAlignment().getInt(),
                                         info.rootAlignment);
  uint64_t vector = std::min<uint64_t>(candidate.getVectorGranularity().getInt(),
                                      alignment);
  return StorageLayoutAttr::get(candidate.getContext(), candidate.getMap(),
      MemorySpaceAttr::get(candidate.getContext(), *getFriskMemorySpace(info.viewType)),
      b.getI64IntegerAttr(alignment), b.getI64IntegerAttr(vector));
}
} // namespace

LogicalResult initializeStorageAliasCandidates(
    Operation *root, LayoutConstraintGraph &graph, LayoutTarget &target) {
  auto &stats = graph.getCandidatePreparationStatistics();
  SmallVector<SmallVector<LayoutConstraintID>> adjacency(graph.getVariables().size());
  for (const auto &relation : graph.getConstraints()) {
    if (relation.strength != ConstraintStrength::Hard ||
        !isSupportedLayoutRelation(relation.kind) || relation.vars.size() != 2)
      continue;
    if (llvm::any_of(relation.vars, [&](LayoutVarID id) {
          return graph.getVariable(id).kind != LayoutKind::Storage;
        })) continue;
    for (auto id : relation.vars) adjacency[id].push_back(relation.id);
  }
  struct Seed { LayoutVarID endpoint; LayoutCandidate candidate; };
  SmallVector<Seed> explicitSeeds;
  for (const auto &var : graph.getVariables()) {
    if (var.kind != LayoutKind::Storage) continue;
    if (!var.storageAlias)
      return emitError(root->getLoc()) << "storage candidate preparation requires coordinate alias metadata";
    for (const auto &candidate : var.candidates) {
      if (auto encoding = dyn_cast<StorageLayoutAttr>(candidate.value)) {
        auto bytes = getStorageFootprintBytes(encoding, cast<MemRefType>(var.shapedType));
        uint64_t capacity = (var.storageAlias->upperBit + 7) / 8;
        if (succeeded(bytes) && *bytes > capacity)
          return emitError(var.anchor->getLoc()) << "storage layout requires "
              << *bytes << " bytes but underlying memref type provides "
              << capacity << " bytes (root accessible span)";
      }
      auto proof = getStorageAliasFootprint(graph, var.id, candidate.value).proof;
      if (proof.status != ProofStatus::Proven) {
        auto diagnostic = emitError(var.anchor->getLoc())
            << "invalid explicit storage binding for " << var.stableName << ": "
            << proof.reason;
        if (!proof.counterexample.empty()) {
          std::string point;
          llvm::raw_string_ostream os(point);
          llvm::interleaveComma(proof.counterexample, os);
          diagnostic << "; coordinate [" << point << "]";
        }
        return failure();
      }
      explicitSeeds.push_back({var.id, candidate});
    }
  }

  // Each origin visits each actual endpoint at most once, even through a copy
  // cycle. No inverse slice can invent unobserved source coordinates. A finite
  // proposal domain may fail; that is not a claim of mathematical unsatisfiability.
  auto propagateOrigin = [&](ArrayRef<Seed> starts) {
    ++stats.origins;
    SmallVector<bool> visited(graph.getVariables().size(), false);
    std::deque<Seed> queue;
    auto enqueue = [&](LayoutVarID id, LayoutCandidate candidate,
                        Operation *source, StringRef rule) {
      if (visited[id]) return;
      auto &var = graph.getVariable(id);
      if (getStorageAliasFootprint(graph, id, candidate.value).proof.status !=
          ProofStatus::Proven) return;
      // Proposal rejection is silent; hard seeds are diagnosed above.
      ScopedDiagnosticHandler quiet(root->getContext(), [](Diagnostic &) { return success(); });
      if (failed(target.verifyCandidate(var, candidate.value, root->getLoc()))) return;
      visited[id] = true;
      if (llvm::none_of(var.candidates, [&](const auto &known) {
            return known.value == candidate.value;
          })) {
        std::optional<ProvenanceID> parent;
        if (candidate.provenance < graph.getProvenances().size()) parent = candidate.provenance;
        candidate.provenance = graph.addProvenance(parent, source, rule,
            "finite origin projected to " + var.stableName);
        var.candidates.push_back(candidate);
        ++stats.projectedCandidates;
      }
      queue.push_back({id, candidate});
    };
    for (const auto &seed : starts)
      enqueue(seed.endpoint, seed.candidate,
              graph.getVariable(seed.endpoint).anchor, "same-source-layout-view");
    while (!queue.empty()) {
      auto source = queue.front();
      queue.pop_front();
      for (auto constraintID : adjacency[source.endpoint]) {
        const auto &relation = graph.getConstraint(constraintID);
        auto dst = relation.vars[0] == source.endpoint ? relation.vars[1] : relation.vars[0];
        if (visited[dst]) continue;
        auto projected = projectLayoutCandidate(graph, relation, source.endpoint,
                                                 source.candidate.value, dst);
        if (failed(projected)) continue;
        auto encoding = dyn_cast<StorageLayoutAttr>(*projected);
        if (!encoding) continue;
        // Alias projection already applies source/target guarantees. Copy may
        // transfer the map but cannot transfer another allocation's alignment.
        encoding = capGuarantees(encoding, *graph.getVariable(dst).storageAlias);
        LayoutCandidate proposal{encoding, source.candidate.provenance,
                                 source.candidate.stableOrdinal};
        enqueue(dst, proposal, graph.getVariable(dst).anchor,
                relation.kind == ConstraintKind::AliasLayout
                    ? "same-source-layout-view" : "storage-copy-projection");
      }
    }
  };
  for (const auto &seed : explicitSeeds) propagateOrigin(ArrayRef<Seed>(seed));

  SmallVector<Value> initializedRoots;
  for (auto &var : graph.getVariables()) {
    if (var.kind != LayoutKind::Storage || !var.candidates.empty()) continue;
    const auto &info = *var.storageAlias;
    if (llvm::is_contained(initializedRoots, info.root)) continue;
    initializedRoots.push_back(info.root);
    auto rootInfo = analyzeStorageAlias(info.root);
    if (failed(rootInfo)) return failure();
    rootInfo->rootKey = info.rootKey;
    rootInfo->rootAlignment = info.rootAlignment;
    rootInfo->alignmentEvidence = info.alignmentEvidence;
    auto linear = buildRootLinearStorageCandidate(*rootInfo);
    if (failed(linear))
      return emitError(var.anchor->getLoc()) << "unsupported root-relative linear template for "
                                           << info.rootKey;
    auto rootProof = verifyStorageAliasCandidate(*rootInfo, *linear);
    if (rootProof.status != ProofStatus::Proven)
      return emitError(var.anchor->getLoc()) << "cannot initialize supported finite storage candidate domain: "
                                           << rootProof.reason;
    // Never hand an over-budget root to target arithmetic/template generation.
    // A small child does not make the unobserved root domain provable.
    LayoutVar proposalRoot;
    proposalRoot.kind = LayoutKind::Storage;
    proposalRoot.shapedType = rootInfo->rootType;
    proposalRoot.stableName = info.rootKey;
    SmallVector<LayoutCandidate> proposals;
    target.enumerateCandidates(proposalRoot, proposals);
    llvm::erase_if(proposals, [](const auto &candidate) { return candidate.stableOrdinal == 0; });
    proposals.insert(proposals.begin(), {*linear, kInvalidProvenanceID, 0});
    for (auto proposal : proposals) {
      auto encoding = dyn_cast<StorageLayoutAttr>(proposal.value);
      if (!encoding) continue;
      encoding = capGuarantees(encoding, *rootInfo);
      if (verifyStorageAliasCandidate(*rootInfo, encoding).status != ProofStatus::Proven) continue;
      proposal.provenance = graph.addProvenance(std::nullopt, var.anchor,
          "target-root-candidate", "root-relative finite template for " + info.rootKey);
      SmallVector<Seed> starts;
      for (const auto &dst : graph.getVariables()) {
        if (!dst.storageAlias || dst.storageAlias->root != info.root) continue;
        auto projected = projectStorageAliasCandidate(*rootInfo, encoding, *dst.storageAlias);
        if (succeeded(projected))
          starts.push_back({dst.id, {*projected, proposal.provenance, proposal.stableOrdinal}});
      }
      propagateOrigin(starts);
    }
  }
  return success();
}
} // namespace mlir::frisk
