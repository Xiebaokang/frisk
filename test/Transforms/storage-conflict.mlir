// RUN: frisk-opt %s -frisk-infer-layouts --split-input-file --verify-diagnostics

#linear = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

#gapped = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [16, 8],
    map = affine_map<(d0) -> (d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @conflicting_alias_views(%source: memref<4xf16, 3>) {
    %first = frisk.layout_view %source {layout = #linear}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    // expected-note@+3 {{seed for conflicting_alias_views/b0/o0/r0/storage: layout_view: explicit layout binding}}
    // expected-note@+2 {{seed for conflicting_alias_views/b0/o1/r0/storage: layout_view: explicit layout binding}}
    // expected-error@+1 {{conflicting hard layout constraint 'same-source-layout-view'}}
    %second = frisk.layout_view %source {layout = #gapped}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return
  }
}


// -----

#linear = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

#gapped = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [16, 8],
    map = affine_map<(d0) -> (d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @conflicting_nested_views(%source: memref<4xf16, 3>) {
    %inner = frisk.layout_view %source {layout = #linear}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    // expected-note@+3 {{seed for conflicting_nested_views/b0/o0/r0/storage: layout_view: explicit layout binding}}
    // expected-note@+2 {{seed for conflicting_nested_views/b0/o1/r0/storage: layout_view: explicit layout binding}}
    // expected-error@+1 {{conflicting hard layout constraint 'same-source-layout-view'}}
    %outer = frisk.layout_view %inner {layout = #gapped}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return
  }
}

// -----

#gapped = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [16, 8],
    map = affine_map<(d0) -> (d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @insufficient_allocation(%source: memref<4xf16, 3>) {
    // expected-error@+1 {{storage layout requires 14 bytes but underlying memref type provides 8 bytes}}
    %view = frisk.layout_view %source {layout = #gapped}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return
  }
}
