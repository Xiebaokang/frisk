// RUN: frisk-opt %s --canonicalize | FileCheck %s

#linear = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [16, 8],
    map = affine_map<(d0) -> (d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

#identity = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @nested_same_layout(%source: memref<4xf16, 3>)
      -> memref<4xf16, 3> {
    %first = frisk.layout_view %source {layout = #linear}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    %second = frisk.layout_view %first {layout = #linear}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return %second : memref<4xf16, 3>
  }
}

// CHECK-LABEL: func.func @nested_same_layout
// CHECK: %[[VIEW:.*]] = frisk.layout_view %arg0
// CHECK-NOT: frisk.layout_view %[[VIEW]]
// CHECK: return %[[VIEW]]

// CHECK-LABEL: func.func @identity_without_layout_anchor
// CHECK-NOT: frisk.layout_view
// CHECK: return %arg0

// CHECK-LABEL: func.func @identity_with_nonreturn_user
// CHECK: frisk.layout_view
// CHECK: memref.cast

// CHECK-LABEL: func.func @non_identity_memref_layout
// CHECK: frisk.layout_view

// CHECK-LABEL: func.func @unsupported_identity_element_type
// CHECK: frisk.layout_view

module {
  func.func @identity_without_layout_anchor(%source: memref<4xf16, 3>)
      -> memref<4xf16, 3> {
    %view = frisk.layout_view %source {layout = #identity}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    return %view : memref<4xf16, 3>
  }
}

module {
  func.func @identity_with_nonreturn_user(%source: memref<4xf16, 3>)
      -> memref<?xf16, 3> {
    %view = frisk.layout_view %source {layout = #identity}
      : memref<4xf16, 3> -> memref<4xf16, 3>
    %cast = memref.cast %view
      : memref<4xf16, 3> to memref<?xf16, 3>
    return %cast : memref<?xf16, 3>
  }
}

module {
  func.func @non_identity_memref_layout(
      %source: memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>)
      -> memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3> {
    %view = frisk.layout_view %source {layout = #identity}
      : memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
        -> memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
    return %view : memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
  }
}

module {
  func.func @unsupported_identity_element_type(
      %source: memref<4xcomplex<f16>, 3>) -> memref<4xcomplex<f16>, 3> {
    %view = frisk.layout_view %source {layout = #linear}
      : memref<4xcomplex<f16>, 3> -> memref<4xcomplex<f16>, 3>
    return %view : memref<4xcomplex<f16>, 3>
  }
}
