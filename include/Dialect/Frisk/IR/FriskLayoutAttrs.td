#ifndef FRISK_LAYOUT_ATTRS_TD
#define FRISK_LAYOUT_ATTRS_TD

include "Dialect/Frisk/IR/FriskDialect.td"
include "Dialect/Frisk/IR/FriskLayoutAttrInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def BitLinearLayoutMapAttr
    : AttrDef<Frisk_Dialect, "BitLinearLayoutMap", [LayoutMapAttrInterface]> {
  let cppNamespace = "::mlir::frisk";
  let summary = "A named bit-linear layout map over GF(2)";
  let mnemonic = "bit_linear";
  let parameters = (ins
    AttrParameter<"::mlir::ArrayAttr", "input dimension names">:$inputNames,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "input dimension bit widths">:$inputBitWidths,
    AttrParameter<"::mlir::ArrayAttr", "output dimension names">:$outputNames,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "output dimension bit widths">:$outputBitWidths,
    AttrParameter<"::mlir::DenseIntElementsAttr", "row-major GF(2) matrix">:$matrix
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::frisk::GF2Matrix> getMatrixValue() const;
    ::mlir::FailureOr<::mlir::Attribute> canonicalizeMap() const;
    ::mlir::LogicalResult verifyMap(::mlir::Location loc) const;
  }];
}

#endif // FRISK_LAYOUT_ATTRS_TD
