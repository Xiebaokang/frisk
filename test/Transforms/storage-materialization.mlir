// RUN: frisk-opt %s -frisk-infer-layouts -verify-each | FileCheck %s

#global_linear = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Global>, alignment = 2,
  vector_granularity = 2>

#shared_gapped = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [16, 8],
    map = affine_map<(d0) -> (d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 2,
  vector_granularity = 2>

#global_gapped = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [16, 8],
    map = affine_map<(d0) -> (d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Global>, alignment = 2,
  vector_granularity = 2>

#global_arbitrary_2d = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0", "dim1"],
    input_extents = [4, 4], outputs = ["byte_offset", "bit_offset"],
    output_extents = [160, 8],
    map = affine_map<(d0, d1) -> (d0 * 40 + d1 * 4, 0)>>,
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

  func.func @materialize_non_default_alias(
      %source: memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>) {
    %bound = frisk.layout_view %source {layout = #shared_gapped}
      : memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
        -> memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
    %inferred = frisk.layout_view %source
      : memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
        -> memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
    return
  }

  func.func @nested_block_stable_names(%condition: i1,
                                       %lhs: memref<4xf16, 3>,
                                       %rhs: memref<4xf16, 3>) {
    scf.if %condition {
      %lhs_view = frisk.layout_view %lhs
        : memref<4xf16, 3> -> memref<4xf16, 3>
    } else {
      %rhs_view = frisk.layout_view %rhs
        : memref<4xf16, 3> -> memref<4xf16, 3>
    }
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

  func.func @materialize_non_default_copy(
      %src: memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 1>,
      %dst: memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>) {
    %src_view = frisk.layout_view %src {layout = #global_gapped}
      : memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 1>
        -> memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 1>
    %dst_view = frisk.layout_view %dst
      : memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
        -> memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>
    "frisk.copy"(%src_view, %dst_view) <{
      srcMap = affine_map<() -> ()>, dstMap = affine_map<() -> ()>,
      srcExtents = array<i64: 4>, dstExtents = array<i64: 4>
    }> {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 1>,
         memref<4xf16, affine_map<(d0) -> (d0 * 2)>, 3>) -> ()
    return
  }


  func.func @materialize_arbitrary_2d_seed(
      %src: memref<4x4xf16,
        affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 1>,
      %dst: memref<4x4xf16,
        affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 3>) {
    %src_view = frisk.layout_view %src {layout = #global_arbitrary_2d}
      : memref<4x4xf16, affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 1>
        -> memref<4x4xf16,
          affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 1>
    %dst_view = frisk.layout_view %dst
      : memref<4x4xf16, affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 3>
        -> memref<4x4xf16,
          affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 3>
    "frisk.copy"(%src_view, %dst_view) <{
      srcMap = affine_map<() -> ()>, dstMap = affine_map<() -> ()>,
      srcExtents = array<i64: 4, 4>, dstExtents = array<i64: 4, 4>
    }> {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (memref<4x4xf16,
           affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 1>,
         memref<4x4xf16,
           affine_map<(d0, d1) -> (d0 * 20 + d1 * 2)>, 3>) -> ()
    return
  }

  module @left {
    func.func @same(%source: memref<4xf16, 3>) {
      %view = frisk.layout_view %source
        : memref<4xf16, 3> -> memref<4xf16, 3>
      return
    }
  }

  module @right {
    func.func @same(%source: memref<4xf16, 3>) {
      %view = frisk.layout_view %source
        : memref<4xf16, 3> -> memref<4xf16, 3>
      return
    }
  }

  module @"a/b" {
    func.func @same(%source: memref<4xf16, 3>) {
      %view = frisk.layout_view %source
        : memref<4xf16, 3> -> memref<4xf16, 3>
      return
    }
  }

  module @a {
    module @b {
      func.func @same(%source: memref<4xf16, 3>) {
        %view = frisk.layout_view %source
          : memref<4xf16, 3> -> memref<4xf16, 3>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @materialize_alloc
// CHECK: memref.alloc
// CHECK: frisk.layout_view %{{.*}} {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Shared>
// CHECK-LABEL: func.func @materialize_non_default_alias
// CHECK-COUNT-2: map = #frisk.affine_layout<{{.*}}output_extents = [16, 8]
// CHECK-LABEL: func.func @nested_block_stable_names
// CHECK-COUNT-2: frisk.layout_view {{.*}} {layout = #frisk.storage<
// CHECK-LABEL: func.func @materialize_copy
// CHECK: frisk.layout_view %arg0 {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Global>
// CHECK: frisk.layout_view %arg1 {layout = #frisk.storage<{{.*}}memory_space = #frisk<memory_space Shared>
// CHECK: frisk.copy
// CHECK-LABEL: func.func @materialize_non_default_copy
// CHECK: memory_space = #frisk<memory_space Global>
// CHECK: output_extents = [16, 8]{{.*}}memory_space = #frisk<memory_space Shared>
// CHECK: frisk.copy
// CHECK-LABEL: func.func @materialize_arbitrary_2d_seed
// CHECK: memory_space = #frisk<memory_space Global>
// CHECK: output_extents = [160, 8]{{.*}}memory_space = #frisk<memory_space Shared>
// CHECK: frisk.copy
// CHECK: module @left
// CHECK: func.func @same
// CHECK: frisk.layout_view {{.*}} {layout = #frisk.storage<
// CHECK: module @right
// CHECK: func.func @same
// CHECK: frisk.layout_view {{.*}} {layout = #frisk.storage<
// CHECK: module @"a/b"
// CHECK: func.func @same
// CHECK: frisk.layout_view {{.*}} {layout = #frisk.storage<
// CHECK: module @a
// CHECK: module @b
// CHECK: func.func @same
// CHECK: frisk.layout_view {{.*}} {layout = #frisk.storage<
