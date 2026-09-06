// RUN: frisk-opt %s -frisk-infer-layouts --split-input-file --verify-diagnostics | FileCheck %s

#global_linear = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Global>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @whole_tile_copy(%src: memref<4xf16, 1>,
                             %dst: memref<4xf16, 3>) {
    %src_view = frisk.layout_view %src {layout = #global_linear}
      : memref<4xf16, 1> -> memref<4xf16, 1>
    %dst_view = frisk.layout_view %dst
      : memref<4xf16, 3> -> memref<4xf16, 3>
    "frisk.copy"(%src_view, %dst_view) <{
      srcMap = affine_map<() -> ()>, dstMap = affine_map<() -> ()>,
      srcExtents = array<i64: 4>, dstExtents = array<i64: 4>
    }> {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (memref<4xf16, 1>, memref<4xf16, 3>) -> ()
    return
  }
}

// -----

module {
  func.func @unsupported_constant_result_map(%src: memref<4xf16, 1>,
                                              %dst: memref<4xf16, 3>) {
    %src_view = frisk.layout_view %src
      : memref<4xf16, 1> -> memref<4xf16, 1>
    %dst_view = frisk.layout_view %dst
      : memref<4xf16, 3> -> memref<4xf16, 3>
    // expected-error@+1 {{unsupported storage layout inference for non-whole-tile or dynamic copy}}
    "frisk.copy"(%src_view, %dst_view) <{
      srcMap = affine_map<() -> (0)>, dstMap = affine_map<() -> (0)>,
      srcExtents = array<i64: 4>, dstExtents = array<i64: 4>
    }> {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (memref<4xf16, 1>, memref<4xf16, 3>) -> ()
    return
  }
}

// -----

module {
  func.func @copy_without_layout_anchors(%src: memref<4xf16, 1>,
                                         %dst: memref<4xf16, 3>) {
    // expected-error@+1 {{M2 storage layout inference requires whole-tile copy operands to be layout_view results}}
    "frisk.copy"(%src, %dst) <{
      srcMap = affine_map<() -> ()>, dstMap = affine_map<() -> ()>,
      srcExtents = array<i64: 4>, dstExtents = array<i64: 4>
    }> {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (memref<4xf16, 1>, memref<4xf16, 3>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @whole_tile_copy
// CHECK: frisk.layout_view %arg0 {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Global>
// CHECK: frisk.layout_view %arg1 {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Shared>
// CHECK-NOT: frisk.layout_inference_ran

// -----

module {
  func.func @unsupported_dynamic_copy(%src: memref<?xf16, 1>,
                                      %dst: memref<?xf16, 3>) {
    // expected-error@+1 {{unsupported storage layout inference for non-whole-tile or dynamic copy}}
    "frisk.copy"(%src, %dst) <{
      srcMap = affine_map<() -> ()>, dstMap = affine_map<() -> ()>,
      srcExtents = array<i64: -1>, dstExtents = array<i64: -1>
    }> {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (memref<?xf16, 1>, memref<?xf16, 3>) -> ()
    return
  }
}
