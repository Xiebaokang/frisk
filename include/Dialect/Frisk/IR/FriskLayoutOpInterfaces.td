#ifndef FRISK_LAYOUT_OP_INTERFACES_TD
#define FRISK_LAYOUT_OP_INTERFACES_TD

include "mlir/IR/OpBase.td"

def LayoutConstraintOpInterface : OpInterface<"LayoutConstraintOpInterface"> {
  let cppNamespace = "::mlir::frisk";
  let description = [{
    Collects layout constraints without mutating global inference state.
  }];
  let methods = [
    InterfaceMethod<
      "Collect the layout constraints imposed by this operation.",
      "::mlir::LogicalResult",
      "collectLayoutConstraints",
      (ins "::mlir::frisk::LayoutConstraintBuilder &":$builder)
    >
  ];
}

#endif // FRISK_LAYOUT_OP_INTERFACES_TD
