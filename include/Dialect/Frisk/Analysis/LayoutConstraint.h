#ifndef FRISK_ANALYSIS_LAYOUTCONSTRAINT_H
#define FRISK_ANALYSIS_LAYOUTCONSTRAINT_H

#include "Dialect/Frisk/Analysis/LayoutCommon.h"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::frisk {

using LayoutVarID = uint32_t;
using LayoutConstraintID = uint32_t;
using ProvenanceID = uint32_t;

enum class LayoutState { Uninitialized, CandidateSet, Resolved, Conflict };
enum class ConstraintStrength { Hard, Soft };
enum class ConstraintKind {
  SameLayout,
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
  ProvenanceID provenance = 0;
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
};

enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use = nullptr;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution = EdgeResolutionKind::KeepCommonLayout;
  uint64_t bytes = 0;
  uint64_t synchronizationCost = 0;
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
};

class LayoutConstraintGraph {
public:
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

  LayoutVar &getVariable(LayoutVarID id) { return variables[id]; }
  const LayoutVar &getVariable(LayoutVarID id) const { return variables[id]; }
  const LayoutConstraint &getConstraint(LayoutConstraintID id) const {
    return constraints[id];
  }
  std::optional<LayoutVarID> lookupVariable(StringRef stableName) const;

  void print(raw_ostream &os) const;

private:
  SmallVector<LayoutVar> variables;
  SmallVector<LayoutConstraint> constraints;
  SmallVector<LayoutProvenance> provenances;
  bool finalized = false;
};

raw_ostream &operator<<(raw_ostream &os, const LayoutConstraintGraph &graph);

StringRef stringifyConstraintKind(ConstraintKind kind);
StringRef stringifyConstraintStrength(ConstraintStrength strength);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTCONSTRAINT_H
