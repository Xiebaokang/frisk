// RUN: frisk-opt %s -frisk-infer-layouts -verify-each | FileCheck %s

#global_linear = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Global>, alignment = 2,
  vector_granularity = 2>

module {
  func.func @materialize_alloc() {
    %storage = memref.alloc() : memref<4xf16, 3>
    %view = frisk.layout_view %storage
      : memref<4xf16, 3> -> memref<4xf16, 3>
    memref.dealloc %storage : memref<4xf16, 3>
    return
  }

  func.func @materialize_copy(%src: memref<4xf16, 1>,
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

// CHECK-LABEL: func.func @materialize_alloc
// CHECK: memref.alloc
// CHECK: frisk.layout_view %{{.*}} {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Shared>
// CHECK-LABEL: func.func @materialize_copy
// CHECK: frisk.layout_view %arg0 {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Global>
// CHECK: frisk.layout_view %arg1 {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Shared>
// CHECK: frisk.copy
