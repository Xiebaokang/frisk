// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s

#linear = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @valid(%source: memref<4xf16, 3>) {
    %view = "frisk.layout_view"(%source) {layout = #linear}
      : (memref<4xf16, 3>) -> memref<4xf16, 3>
    %unresolved = frisk.layout_view %source
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return
  }

  func.func @typed_memory_space(
      %source: memref<4xf16, #frisk<memory_space Shared>>) {
    %view = frisk.layout_view %source {layout = #linear}
      : memref<4xf16, #frisk<memory_space Shared>>
        -> memref<4xf16, #frisk<memory_space Shared>>
    return
  }
}

// CHECK: %{{.*}} = frisk.layout_view %{{.*}} {layout = #frisk.storage<{{.*}}>}

// -----

module {
  func.func @mismatched_type(%source: memref<4xf16, 3>) {
    // expected-error@+1 {{source and result must have identical memref types}}
    %view = frisk.layout_view %source
      : memref<4xf16, 3> -> memref<2x2xf16, 3>
    return
  }
}

// -----

#overlapping_elements = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [4, 8],
    map = affine_map<(d0) -> (d0, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 1,
  vector_granularity = 1>

module {
  func.func @overlapping_element_ranges(%source: memref<4xf16, 3>) {
    // expected-error@+1 {{storage element bit ranges must be provably non-overlapping}}
    %view = frisk.layout_view %source {layout = #overlapping_elements}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return
  }
}

// -----

#out_of_bounds = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @result_exceeds_declared_extent(%source: memref<4xf16, 3>) {
    // expected-error@+1 {{storage map results must stay within declared output extents}}
    %view = frisk.layout_view %source {layout = #out_of_bounds}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return
  }
}

// -----

#complex_layout = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @unsupported_element_type(%source: memref<4xcomplex<f16>, 3>) {
    // expected-error@+1 {{storage encoding supports only integer or floating-point element types}}
    %view = frisk.layout_view %source {layout = #complex_layout}
      : memref<4xcomplex<f16>, 3> -> memref<4xcomplex<f16>, 3>
    return
  }
}

// -----

#aliasing = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [1, 8],
    map = affine_map<(d0) -> (0, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 1>

module {
  func.func @non_injective(%source: memref<4xf16, 3>) {
    // expected-error@+1 {{storage map must be provably injective}}
    %view = frisk.layout_view %source {layout = #aliasing}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return
  }
}

// -----

#shared = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @wrong_memory_space(%source: memref<4xf16, 1>) {
    // expected-error@+1 {{storage encoding memory space must match the MemRefType}}
    %view = frisk.layout_view %source {layout = #shared}
      : memref<4xf16, 1> -> memref<4xf16, 1>
    return
  }
}
