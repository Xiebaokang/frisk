// RUN: frisk-opt %s -frisk-infer-layouts | FileCheck %s
// RUN: frisk-opt %s -frisk-infer-layouts -frisk-infer-layouts | FileCheck %s
// RUN: frisk-opt %s -frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s --check-prefix=GRAPH

#root = #frisk.storage<map = #frisk.affine_layout<
  inputs = ["dim0", "dim1"], input_extents = [4, 8],
  outputs = ["byte_offset", "bit_offset"], output_extents = [128, 8],
  map = affine_map<(d0,d1) -> (32*d0+4*d1, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 4, vector_granularity = 4>

// Addresses are 40,48,56,72,80,88 bytes from root aligned pointer.
// CHECK-DAG: #[[SLICE:map[0-9]*]] = affine_map<(d0, d1) -> ((d0 + 1) * 32 + (d1 * 2 + 2) * 4, 0)>
// CHECK-DAG: #[[OFFSET:map[0-9]*]] = affine_map<(d0, d1) -> (((d0 + 1) * 8 + d1 * 2 + 2 + 3) * 4, 0)>
// CHECK-LABEL: func.func @strided_slice
// CHECK: memref.subview
// CHECK: frisk.layout_view {{.*}}map = #[[SLICE]]
// GRAPH: alias-endpoint {{.*}} bits=[0,1024) alignment=4 evidence={{.*}}whole-root binding precondition
// GRAPH: alias-endpoint {{.*}}transform=(d0, d1) -> (d0 + 1, d1 * 2 + 2)
// GRAPH: alias-layout
// GRAPH: propagation strict
// GRAPH: propagation common
// GRAPH: candidate-preparation origins=
func.func @strided_slice(%r: memref<4x8xi32, 3>) {
  %rv = frisk.layout_view %r {layout = #root} : memref<4x8xi32, 3> -> memref<4x8xi32, 3>
  %s = memref.subview %r[1,2] [2,3] [1,2] : memref<4x8xi32, 3> to memref<2x3xi32, strided<[8,2], offset:10>, 3>
  %v = frisk.layout_view %s : memref<2x3xi32, strided<[8,2], offset:10>, 3> -> memref<2x3xi32, strided<[8,2], offset:10>, 3>
  return
}

// Default linear root descriptor offset=3 is included once: 52,60,68,84,92,100.
// CHECK-LABEL: func.func @offset_root
// CHECK: frisk.layout_view {{.*}}map = #[[OFFSET]]
func.func @offset_root(%r: memref<4x8xi32, strided<[8,1], offset:3>, 3>) {
  %s = memref.subview %r[1,2] [2,3] [1,2] : memref<4x8xi32,strided<[8,1],offset:3>,3> to memref<2x3xi32,strided<[8,2],offset:13>,3>
  %v = frisk.layout_view %s : memref<2x3xi32,strided<[8,2],offset:13>,3> -> memref<2x3xi32,strided<[8,2],offset:13>,3>
  return
}
