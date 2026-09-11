#ifndef FRISK_ANALYSIS_LAYOUTRELATIONS_H
#define FRISK_ANALYSIS_LAYOUTRELATIONS_H
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
namespace mlir::frisk {
bool isSupportedLayoutRelation(ConstraintKind kind);
bool layoutEncodingsEqual(Attribute lhs, Attribute rhs);
bool layoutRelationCompatible(const LayoutConstraintGraph &graph,
                              const LayoutConstraint &relation,
                              LayoutVarID lhsID, Attribute lhs,
                              LayoutVarID rhsID, Attribute rhs);
FailureOr<Attribute> projectLayoutCandidate(
    const LayoutConstraintGraph &graph, const LayoutConstraint &relation,
    LayoutVarID source, Attribute candidate, LayoutVarID target);
std::string layoutCandidateKey(Attribute value);
} // namespace mlir::frisk
#endif
