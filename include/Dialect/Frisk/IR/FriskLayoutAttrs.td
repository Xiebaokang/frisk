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

def DistributedEncodingAttr
    : AttrDef<Frisk_Dialect, "DistributedEncoding", [LayoutEncodingAttrInterface]> {
  let cppNamespace = "::mlir::frisk";
  let summary = "A hardware-carrier to logical-tile encoding";
  let mnemonic = "distributed";
  let parameters = (ins
    AttrParameter<"::mlir::Attribute", "canonical distributed map">:$map,
    AttrParameter<"::mlir::DenseI64ArrayAttr", "register/lane/warp/warp-group/CTA extents">:$topology,
    AttrParameter<"::mlir::IntegerAttr", "number of hardware owners per logical element">:$replication
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::Attribute>
    getCanonicalMap(::mlir::ShapedType type) const;
    ::mlir::LogicalResult verifyForType(::mlir::ShapedType type,
                                        ::mlir::Location loc) const;
    ::mlir::frisk::LayoutKind getKind() const;
  }];
}

def StorageLayoutAttr
    : AttrDef<Frisk_Dialect, "StorageLayout", [LayoutEncodingAttrInterface]> {
  let cppNamespace = "::mlir::frisk";
  let summary = "A logical-tile to byte/bit storage encoding";
  let description = [{
    Maps live logical coordinates to (byte_offset, bit_offset) relative to the
    proven common storage root's aligned pointer. Descriptor offsets are
    included in that map, never added to its output again. Alias verification
    checks the root accessible interval and coordinate/bit-interval agreement.

    Alignment denotes a root-pointer guarantee, not alignment of every element.
    A whole-root binding may declare an alignment precondition; a child binding
    cannot strengthen its root guarantee. Vector granularity 1 is scalar/packed
    baseline. Larger values require aligned scalar subdivisions or complete,
    contiguous innermost-row vector chunks; projected candidates may weaken it.
    These contextual properties are checked by layout analysis, not by the
    attribute's context-free structural verifier alone.
  }];
  let mnemonic = "storage";
  let parameters = (ins
    AttrParameter<"::mlir::Attribute", "canonical storage map">:$map,
    AttrParameter<"::mlir::frisk::MemorySpaceAttr", "physical memory space">:$memorySpace,
    AttrParameter<"::mlir::IntegerAttr", "base alignment in bytes">:$alignment,
    AttrParameter<"::mlir::IntegerAttr", "safe vector granularity in bytes">:$vectorGranularity
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::Attribute>
    getCanonicalMap(::mlir::ShapedType type) const;
    ::mlir::LogicalResult verifyForType(::mlir::ShapedType type,
                                        ::mlir::Location loc) const;
    ::mlir::frisk::LayoutKind getKind() const;
  }];
}

#endif // FRISK_LAYOUT_ATTRS_TD
