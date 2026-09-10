// RUN: frisk-opt %s --split-input-file --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s

#linear = #frisk.storage<map = #frisk.affine_layout<
  inputs = ["dim0", "dim1"], input_extents = [8, 8],
  outputs = ["byte_offset", "bit_offset"], output_extents = [256, 8],
  map = affine_map<(d0, d1) -> (d0 * 32 + d1 * 4, 0)>>,
  memory_space = #frisk<memory_space Global>, alignment = 4, vector_granularity = 4>
#transposed = #frisk.storage<map = #frisk.affine_layout<
  inputs = ["dim0", "dim1"], input_extents = [8, 8],
  outputs = ["byte_offset", "bit_offset"], output_extents = [256, 8],
  map = affine_map<(d0, d1) -> (d1 * 32 + d0 * 4, 0)>>,
  memory_space = #frisk<memory_space Global>, alignment = 4, vector_granularity = 4>

func.func @two(%a: memref<8x8xf32, 1>, %b: memref<8x8xf32, 1>,
               %c: memref<8x8xf32, 1>) {
  %av = frisk.layout_view %a {layout = #linear} : memref<8x8xf32, 1> -> memref<8x8xf32, 1>
  %bv = frisk.layout_view %b {layout = #linear} : memref<8x8xf32, 1> -> memref<8x8xf32, 1>
  %cv = frisk.layout_view %c {layout = #transposed} : memref<8x8xf32, 1> -> memref<8x8xf32, 1>
  %t = frisk.tile_load %av : memref<8x8xf32, 1> -> tensor<8x8xf32>
  frisk.tile_store %t, %bv : tensor<8x8xf32>, memref<8x8xf32, 1>
  frisk.tile_store %t, %cv : tensor<8x8xf32>, memref<8x8xf32, 1>
  return
}

// CHECK-COUNT-2: hard convertible
// CHECK-COUNT-3: hard storage-access
// CHECK-COUNT-3: distributed domain {{.*}}: 4
// CHECK: conversions: 0

// -----

// A valid DPS transpose imposes an independently encoded consumer contract.
// Only its input use converts; the independent store keeps the producer layout.
#a_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2], outputs = ["dim0"], output_bits = [2], matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#b_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2], outputs = ["dim0"], output_bits = [2], matrix = dense<[[0, 1], [1, 0]]> : tensor<2x2xi1>>
#a = #frisk.distributed<map = #a_map, topology = [1, 4, 1, 1, 1], replication = 1>
#b = #frisk.distributed<map = #b_map, topology = [1, 4, 1, 1, 1], replication = 1>
func.func @contract(%x: tensor<4xf16, #a>, %out: memref<4xf16, 3>) {
  %init = tensor.empty() : tensor<4xf16, #b>
  %0 = linalg.transpose ins(%x : tensor<4xf16, #a>) outs(%init : tensor<4xf16, #b>) permutation = [0]
  %view = frisk.layout_view %out : memref<4xf16, 3> -> memref<4xf16, 3>
  frisk.tile_store %x, %view : tensor<4xf16, #a>, memref<4xf16, 3>
  return
}

// CHECK: conversions: 1
// CHECK-NEXT: convert 8:contract/b0/o1/use0
// CHECK-NOT: frisk.convert_layout
// CHECK: func.func @contract
