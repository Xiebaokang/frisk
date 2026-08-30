#ifndef FRISK_LAYOUT_ATTR_INTERFACES_TD
#define FRISK_LAYOUT_ATTR_INTERFACES_TD

include "mlir/IR/OpBase.td"

def LayoutMapAttrInterface : AttrInterface<"LayoutMapAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let description = [{
    Common operations supported by canonical Frisk layout maps.
  }];
  let methods = [
    InterfaceMethod<
      "Return the canonical representation of this layout map.",
      "::mlir::FailureOr<::mlir::Attribute>",
      "canonicalizeMap",
      (ins)
    >,
    InterfaceMethod<
      "Verify the map and its named dimensions.",
      "::mlir::LogicalResult",
      "verifyMap",
      (ins "::mlir::Location":$loc)
    >
  ];
}

def LayoutEncodingAttrInterface : AttrInterface<"LayoutEncodingAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let description = [{
    Expands a Frisk layout encoding into its canonical layout map.
  }];
  let methods = [
    InterfaceMethod<
      "Expand this encoding for a shaped carrier type.",
      "::mlir::FailureOr<::mlir::Attribute>",
      "getCanonicalMap",
      (ins "::mlir::ShapedType":$type)
    >,
    InterfaceMethod<
      "Verify this encoding for a shaped carrier type.",
      "::mlir::LogicalResult",
      "verifyForType",
      (ins "::mlir::ShapedType":$type, "::mlir::Location":$loc)
    >,
    InterfaceMethod<
      "Return the semantic domain represented by this encoding.",
      "::mlir::frisk::LayoutKind",
      "getKind",
      (ins)
    >
  ];
}

#endif // FRISK_LAYOUT_ATTR_INTERFACES_TD
