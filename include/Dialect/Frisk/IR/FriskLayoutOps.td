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
    A binding uses root-relative byte/bit addresses. For a statically proven
    cast/subview chain, compose logical coordinates with its offsets, positive
    strides and rank reduction; do not add the view descriptor offset twice.
    Same-root bindings agree on shared coordinates and may not map distinct
    root elements to overlapping physical bit intervals. Unsupported alias
    paths are diagnosed by layout inference rather than treated as new roots.
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

def Frisk_TileLoadOp : Frisk_Op<"tile_load"> {
  let summary = "load a whole static memref tile into a tensor carrier";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Frisk_TileStoreOp : Frisk_Op<"tile_store"> {
  let summary = "store a whole static tensor carrier into a memref tile";
  let description = [{
    Stores each logical element exactly once. If the distributed tensor has
    replicated owners, lowering must elect one deterministic source owner;
    replication does not authorize duplicate physical writes.
  }];
  let arguments = (ins
    AnyRankedTensor:$value,
    Arg<AnyMemRef, "", [MemWrite]>:$target
  );
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $target attr-dict `:` type($value) `,` type($target)";
}

def Frisk_ConvertLayoutOp : Frisk_Op<"convert_layout", [Pure]> {
  let summary = "convert a tensor between distributed layouts";
  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // FRISK_LAYOUT_OPS
