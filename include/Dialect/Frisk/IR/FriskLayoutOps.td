#ifndef FRISK_LAYOUT_OPS
#define FRISK_LAYOUT_OPS

def Frisk_LayoutViewOp : Frisk_Op<"layout_view", [
    Pure,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>
  ]> {
  let summary = "bind a storage layout to a memref view";
  let description = [{
    Creates a fully aliasing view of a memref and optionally binds a Frisk
    storage layout. The operation never allocates or moves data.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    OptionalAttr<StorageLayoutAttr>:$layout
  );
  let results = (outs AnyMemRef:$result);

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // FRISK_LAYOUT_OPS
