// RUN: frisk-opt %s -frisk-infer-layouts -verify-each | FileCheck %s
// RUN: frisk-opt %s -frisk-infer-layouts -frisk-infer-layouts -verify-each | FileCheck %s

// CHECK-LABEL: func.func @constant
// CHECK-SAME: -> tensor<4xf16, #frisk.distributed
// CHECK: arith.constant dense<1.000000e+00> : tensor<4xf16, #frisk.distributed
// CHECK: return {{.*}} : tensor<4xf16, #frisk.distributed
func.func @constant() -> tensor<4xf16> {
  %c = arith.constant dense<1.0> : tensor<4xf16>
  return %c : tensor<4xf16>
}

// CHECK-LABEL: func.func @elementwise
// CHECK-SAME: tensor<4xf32, #frisk.distributed
// CHECK: arith.addf {{.*}} : tensor<4xf32, #frisk.distributed
func.func @elementwise(%x: tensor<4xf32>) -> tensor<4xf32> {
  %sum = arith.addf %x, %x : tensor<4xf32>
  return %sum : tensor<4xf32>
}
