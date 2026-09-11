// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s
// RUN: frisk-opt %s > %t.original
// RUN: frisk-opt %s --frisk-infer-layouts=analysis-only > %t.analyzed
// RUN: diff %t.original %t.analyzed

func.func @transpose(%arg: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %init = tensor.empty() : tensor<8x4xf32>
  %out = linalg.transpose ins(%arg : tensor<4x8xf32>)
    outs(%init : tensor<8x4xf32>) permutation = [1, 0]
  return %out : tensor<8x4xf32>
}

func.func @elementwise(%arg: tensor<8xf32>) -> tensor<8xf32> {
  %one = arith.constant dense<1.0> : tensor<8xf32>
  %exp = math.exp %arg : tensor<8xf32>
  %sum = arith.addf %exp, %one : tensor<8xf32>
  return %sum : tensor<8xf32>
}

// CHECK: transform-layout
// CHECK: distributed domain
// CHECK: conversions: 0
// CHECK: func.func @transpose
// CHECK: func.func @elementwise
