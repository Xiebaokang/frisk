// RUN: frisk-opt %s -frisk-infer-layouts -verify-each | FileCheck %s
// RUN: frisk-opt %s -frisk-infer-layouts -frisk-infer-layouts -verify-each | FileCheck %s
#a_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2], outputs = ["dim0"], output_bits = [2], matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#b_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2], outputs = ["dim0"], output_bits = [2], matrix = dense<[[0, 1], [1, 0]]> : tensor<2x2xi1>>
#a = #frisk.distributed<map = #a_map, topology = [1, 4, 1, 1, 1], replication = 1>
#b = #frisk.distributed<map = #b_map, topology = [1, 4, 1, 1, 1], replication = 1>
// CHECK-LABEL: func.func @contract
// CHECK-SAME: (%[[X:[a-zA-Z0-9_]+]]: tensor
// CHECK-NOT: frisk.convert_layout
// CHECK: %[[CVT:.*]] = frisk.convert_layout %[[X]]
// CHECK-NEXT: {{.*}}linalg.transpose ins(%[[CVT]]
// CHECK-NOT: frisk.convert_layout
// CHECK: frisk.tile_store %[[X]],
// CHECK-NOT: frisk.convert_layout
func.func @contract(%x: tensor<4xf16, #a>, %out: memref<4xf16, 3>) {
  %init = tensor.empty() : tensor<4xf16, #b>
  %0 = linalg.transpose ins(%x : tensor<4xf16, #a>) outs(%init : tensor<4xf16, #b>) permutation = [0]
  %view = frisk.layout_view %out : memref<4xf16, 3> -> memref<4xf16, 3>
  frisk.tile_store %x, %view : tensor<4xf16, #a>, memref<4xf16, 3>
  return
}
