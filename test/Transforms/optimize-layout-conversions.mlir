// RUN: frisk-opt %s -frisk-optimize-layout-conversions -verify-each | FileCheck %s
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1, 32, 1, 1, 1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1, 32, 1, 1, 1], replication = 1>

func.func @identity(%x: tensor<32xf32, #A>) -> tensor<32xf32, #A> {
  %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #A>
  return %y : tensor<32xf32, #A>
}
// CHECK-LABEL: func.func @identity
// CHECK-NOT: frisk.convert_layout
// CHECK: return %arg0
#c = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#C = #frisk.distributed<map = #c, topology = [1,32,1,1,1], replication = 1>
func.func @compose_multiuse(%x: tensor<32xf32, #A>) -> (tensor<32xf32, #B>, tensor<32xf32, #C>) {
  %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #B>
  %z = frisk.convert_layout %y : tensor<32xf32, #B> -> tensor<32xf32, #C>
  return %y, %z : tensor<32xf32, #B>, tensor<32xf32, #C>
}
// CHECK-LABEL: func.func @compose_multiuse
// CHECK: %[[B:.*]] = frisk.convert_layout %arg0
// CHECK: %[[C:.*]] = frisk.convert_layout %arg0
// CHECK: return %[[B]], %[[C]]
func.func @inverse(%x: tensor<32xf32, #A>) -> tensor<32xf32, #A> {
  %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #B>
  %z = frisk.convert_layout %y : tensor<32xf32, #B> -> tensor<32xf32, #A>
  return %z : tensor<32xf32, #A>
}
// CHECK-LABEL: func.func @inverse
// CHECK-NOT: frisk.convert_layout
// CHECK: return %arg0
