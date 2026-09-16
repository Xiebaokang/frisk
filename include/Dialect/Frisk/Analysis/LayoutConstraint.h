#ifndef FRISK_ANALYSIS_LAYOUTCONSTRAINT_H
#define FRISK_ANALYSIS_LAYOUTCONSTRAINT_H

#include "Dialect/Frisk/Analysis/LayoutCommon.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::frisk {

using LayoutVarID = uint32_t;
using LayoutConstraintID = uint32_t;
using ProvenanceID = uint32_t;
inline constexpr ProvenanceID kInvalidProvenanceID =
    std::numeric_limits<ProvenanceID>::max();

enum class LayoutState { Uninitialized, CandidateSet, Resolved, Conflict };
enum class ConstraintStrength { Hard, Soft };
enum class ConstraintKind {
  SameLayout,
  Convertible,
  TransformLayout,
  RequireEncoding,
  InstructionContract,
  StorageAccess,
  AliasLayout,
  Ownership,
  ResourceLimit,
  Preference
};

struct LayoutCandidate {
  Attribute value;
  ProvenanceID provenance = kInvalidProvenanceID;
  uint64_t stableOrdinal = 0;
};

struct LayoutVar {
  LayoutVarID id = 0;
  LayoutKind kind = LayoutKind::Storage;
  Type shapedType;
  SmallVector<LayoutCandidate> candidates;
  LayoutState state = LayoutState::Uninitialized;
  std::string stableName;
  Operation *anchor = nullptr;
  Value value;
  OpOperand *use = nullptr;
  std::optional<unsigned> functionResult;
  std::optional<StorageAliasInfo> storageAlias;
};

enum class AccessKind { Read, Write };

enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use = nullptr;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution = EdgeResolutionKind::KeepCommonLayout;
  uint64_t bytes = 0;
  uint64_t synchronizationCost = 0;
  LayoutConstraintID constraint = std::numeric_limits<LayoutConstraintID>::max();
};

struct LayoutProvenance {
  ProvenanceID id = 0;
  std::optional<ProvenanceID> parent;
  Operation *source = nullptr;
  std::string rule;
  std::string reason;
};

struct LayoutConstraint {
  LayoutConstraintID id = 0;
  ConstraintKind kind = ConstraintKind::SameLayout;
  ConstraintStrength strength = ConstraintStrength::Hard;
  SmallVector<LayoutVarID> vars;
  Attribute requiredEncoding;
  ProvenanceID provenance = 0;
  Attribute coordinateTransform;
  AccessKind access = AccessKind::Read;
  OpOperand *use = nullptr;
  bool existingConversion = false;
  std::string stableUseKey;
};

enum class RegionLayoutEdgeKind {
  IfYield, ForInit, ForBackedge, ForResult,
  WhileInit, WhileBackedge, WhileCondition, WhileResult
};

/// An auditable view of an existing hard relation, not another constraint.
/// Borrowed IR identities have the same lifetime as the graph's SSA uses.
struct RegionLayoutEdge {
  RegionLayoutEdgeKind kind;
  LayoutVarID source;
  LayoutVarID target;
  LayoutConstraintID constraint;
  Operation *owner;
  OpOperand *use;
  unsigned slot;
  std::string stableKey;
};

StringRef stringifyRegionLayoutEdgeKind(RegionLayoutEdgeKind kind);

struct PropagationPhaseStatistics {
  bool ran = false;
  uint64_t initialCandidates = 0;
  uint64_t finalCandidates = 0;
  uint64_t initialConstraints = 0;
  uint64_t deletedCandidates = 0;
  uint64_t domainChanges = 0;
  uint64_t queuePops = 0;
  uint64_t enqueues = 0;
  uint64_t maximumQueue = 0;
  uint64_t popUpperBound = 0;
  uint64_t staticPopUpperBound = 0;
  SmallVector<uint64_t> changesByVariable;

  bool hasValidBounds() const {
    return deletedCandidates <= initialCandidates &&
           finalCandidates == initialCandidates - deletedCandidates &&
           queuePops <= popUpperBound && popUpperBound <= staticPopUpperBound &&
           queuePops <= enqueues;
  }
  void print(StringRef phase, raw_ostream &os) const;
};

struct LayoutPropagationStatistics {
  PropagationPhaseStatistics strict;
  PropagationPhaseStatistics common;
};

struct LayoutCandidatePreparationStatistics {
  uint64_t origins = 0;
  uint64_t projectedCandidates = 0;
  uint64_t footprintEvaluations = 0;
  uint64_t pairProofEvaluations = 0;
};

class LayoutConstraintGraph {
public:
  using AliasCandidateKey = std::pair<LayoutVarID, Attribute>;
  using AliasPairKey = std::pair<AliasCandidateKey, AliasCandidateKey>;
  auto &getAliasFootprintCache() const { return aliasFootprints; }
  auto &getAliasPairCache() const { return aliasPairProofs; }
  auto &getCandidatePreparationStatistics() const { return preparationStatistics; }
  LayoutVarID addVariable(LayoutKind kind, Type type, StringRef stableName,
                          Operation *anchor = nullptr);
  ProvenanceID addProvenance(std::optional<ProvenanceID> parent,
                             Operation *source, StringRef rule,
                             StringRef reason);
  LayoutConstraintID addConstraint(ConstraintKind kind,
                                   ConstraintStrength strength,
                                   ArrayRef<LayoutVarID> vars,
                                   Operation *source, StringRef rule,
                                   StringRef reason,
                                   Attribute requiredEncoding = {});

  LogicalResult finalize(Location loc);
  LogicalResult verifyInvariants(Location loc) const;
  LogicalResult printProvenanceChain(ProvenanceID id,
                                     raw_ostream &os) const;

  ArrayRef<LayoutVar> getVariables() const { return variables; }
  MutableArrayRef<LayoutVar> getVariables() { return variables; }
  ArrayRef<LayoutConstraint> getConstraints() const { return constraints; }
  MutableArrayRef<LayoutConstraint> getConstraints() { return constraints; }
  ArrayRef<LayoutProvenance> getProvenances() const { return provenances; }
  ArrayRef<RegionLayoutEdge> getRegionEdges() const { return regionEdges; }
  const LayoutPropagationStatistics &getPropagationStatistics() const {
    return propagationStatistics;
  }
  LayoutPropagationStatistics &getPropagationStatistics() {
    return propagationStatistics;
  }
  void addRegionEdge(RegionLayoutEdge edge) {
    finalized = false;
    regionEdges.push_back(std::move(edge));
  }

  LayoutVar &getVariable(LayoutVarID id) { return variables[id]; }
  const LayoutVar &getVariable(LayoutVarID id) const { return variables[id]; }
  const LayoutConstraint &getConstraint(LayoutConstraintID id) const {
    return constraints[id];
  }
  std::optional<LayoutVarID> lookupVariable(StringRef stableName) const;
  std::optional<LayoutVarID> lookupVariable(Value value) const;

  void print(raw_ostream &os) const;

private:
  SmallVector<LayoutVar, 0> variables;
  SmallVector<LayoutConstraint> constraints;
  SmallVector<LayoutProvenance> provenances;
  SmallVector<RegionLayoutEdge> regionEdges;
  LayoutPropagationStatistics propagationStatistics;
  mutable LayoutCandidatePreparationStatistics preparationStatistics;
  mutable DenseMap<AliasCandidateKey, StorageAliasFootprint> aliasFootprints;
  mutable DenseMap<AliasPairKey, LayoutProof> aliasPairProofs;
  bool finalized = false;
};

raw_ostream &operator<<(raw_ostream &os, const LayoutConstraintGraph &graph);

StringRef stringifyConstraintKind(ConstraintKind kind);
StringRef stringifyConstraintStrength(ConstraintStrength strength);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTCONSTRAINT_H
