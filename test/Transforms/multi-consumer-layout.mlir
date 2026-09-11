// RUN: frisk-opt %s --split-input-file --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s
// RUN: frisk-opt %s -frisk-infer-layouts -frisk-optimize-layout-conversions -verify-each -o %t
// RUN: FileCheck %s --check-prefix=LIVE < %t
// RUN: frisk-opt %t -frisk-infer-layouts -frisk-optimize-layout-conversions -verify-each -o %t.replay
// RUN: cmp %t %t.replay

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
// Use a real 2D permutation: a rank-one identity transpose would fold away even
// when returned. The independently fixed XOR output map forces one conversion.
#a_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2], outputs = ["row", "col"], output_bits = [1, 1], matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#b_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2], outputs = ["row", "col"], output_bits = [1, 1], matrix = dense<[[1, 1], [0, 1]]> : tensor<2x2xi1>>
#a = #frisk.distributed<map = #a_map, topology = [1, 4, 1, 1, 1], replication = 1>
#b = #frisk.distributed<map = #b_map, topology = [1, 4, 1, 1, 1], replication = 1>
func.func @contract(%x: tensor<2x2xf16, #a>, %out: memref<2x2xf16, 3>) -> tensor<2x2xf16, #b> {
  %init = tensor.empty() : tensor<2x2xf16, #b>
  %0 = linalg.transpose ins(%x : tensor<2x2xf16, #a>) outs(%init : tensor<2x2xf16, #b>) permutation = [1, 0]
  %view = frisk.layout_view %out : memref<2x2xf16, 3> -> memref<2x2xf16, 3>
  frisk.tile_store %x, %view : tensor<2x2xf16, #a>, memref<2x2xf16, 3>
  return %0 : tensor<2x2xf16, #b>
}

// CHECK: conversions: 1
// CHECK-NEXT: convert 8:contract/b0/o1/use0
// CHECK-NOT: frisk.convert_layout
// CHECK: func.func @contract

// LIVE-LABEL: func.func @contract
// LIVE-SAME: -> tensor<2x2xf16,
// LIVE-NOT: frisk.convert_layout
// LIVE: %[[CONVERT:.*]] = frisk.convert_layout %arg0
// LIVE-NEXT: %[[TRANSPOSE:.*]] = linalg.transpose ins(%[[CONVERT]]
// LIVE-NOT: frisk.convert_layout
// LIVE: frisk.tile_store %arg0
// LIVE-NOT: frisk.convert_layout
// LIVE: return %[[TRANSPOSE]] : tensor<2x2xf16,
