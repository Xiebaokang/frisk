#include "Dialect/Frisk/Transforms/LayoutTypeConverter.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"

namespace mlir::frisk {
LayoutTypeConverter::LayoutTypeConverter(const LayoutConstraintGraph &graph,
                                         const LayoutSolution &solution)
    : graph(graph), solution(solution) {
  addConversion([](Type type) -> Type {
    return isa<TensorType>(type) ? Type{} : type;
  });
}

FailureOr<RankedTensorType>
LayoutTypeConverter::convertLayoutBearingTensor(Value value) const {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto id = graph.lookupVariable(value);
  if (!type || !id || graph.getVariable(*id).kind != LayoutKind::Distributed) {
    emitError(value.getLoc()) << "tensor has no distributed definition binding";
    return failure();
  }
  Attribute assignment = solution.assignments.lookup(*id);
  if (!assignment) {
    emitError(value.getLoc()) << "missing distributed layout assignment for tensor definition '"
                             << graph.getVariable(*id).stableName << "'";
    return failure();
  }
  auto encoding = dyn_cast<DistributedEncodingAttr>(assignment);
  if (!encoding) {
    emitError(value.getLoc()) << "layout assignment is not a distributed encoding for tensor definition '"
                             << graph.getVariable(*id).stableName << "': " << assignment;
    return failure();
  }
  if (failed(encoding.verifyForType(type, value.getLoc())))
    return failure();
  if (type.getEncoding() && type.getEncoding() != encoding) {
    emitError(value.getLoc()) << "materialization would overwrite an explicit hard binding";
    return failure();
  }
  return RankedTensorType::get(type.getShape(), type.getElementType(), encoding);
}
} // namespace mlir::frisk
