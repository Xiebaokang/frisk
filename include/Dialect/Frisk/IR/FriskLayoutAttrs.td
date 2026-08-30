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

def AffineLayoutMapAttr
    : AttrDef<Frisk_Dialect, "AffineLayoutMap", [LayoutMapAttrInterface]> {
  let cppNamespace = "::mlir::frisk";
  let summary = "A named affine outer layout map";
  let mnemonic = "affine_layout";
  let parameters = (ins
    AttrParameter<"::mlir::ArrayAttr", "input dimension names">:$inputNames,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "input dimension extents">:$inputExtents,
    AttrParameter<"::mlir::ArrayAttr", "output dimension names">:$outputNames,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "output dimension extents">:$outputExtents,
    AttrParameter<"::mlir::AffineMapAttr", "affine map">:$affineMap
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::Attribute> canonicalizeMap() const;
    ::mlir::LogicalResult verifyMap(::mlir::Location loc) const;
  }];
}

def ProductLayoutMapAttr
    : AttrDef<Frisk_Dialect, "ProductLayoutMap", [LayoutMapAttrInterface]> {
  let cppNamespace = "::mlir::frisk";
  let summary = "A carry-safe product of affine outer and bit-linear inner maps";
  let mnemonic = "product";
  let parameters = (ins
    AttrParameter<"::mlir::Attribute", "affine outer map">:$outer,
    AttrParameter<"::mlir::Attribute", "bit-linear inner map">:$inner,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "per-output inner extents">:$splitExtents
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::Attribute> canonicalizeMap() const;
    ::mlir::LogicalResult verifyMap(::mlir::Location loc) const;
  }];
}

#endif // FRISK_LAYOUT_ATTRS_TD
