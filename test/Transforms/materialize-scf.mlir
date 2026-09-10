// RUN: frisk-opt %s -frisk-infer-layouts -verify-each | FileCheck %s
// RUN: frisk-opt %s -frisk-infer-layouts -frisk-infer-layouts -verify-each | FileCheck %s

// CHECK-LABEL: func.func @choose
// CHECK: scf.if {{.*}} -> (tensor<4xf16, #frisk.distributed
// CHECK: scf.yield {{.*}} : tensor<4xf16, #frisk.distributed
// CHECK: scf.yield {{.*}} : tensor<4xf16, #frisk.distributed
func.func @choose(%p: i1, %x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %r = scf.if %p -> tensor<4xf16> {
    scf.yield %x : tensor<4xf16>
  } else {
    scf.yield %y : tensor<4xf16>
  }
  return %r : tensor<4xf16>
}

// CHECK-LABEL: func.func @loop
// CHECK: scf.for {{.*}} iter_args{{.*}} -> (tensor<4xf16, #frisk.distributed
// CHECK: scf.yield {{.*}} : tensor<4xf16, #frisk.distributed
func.func @loop(%x: tensor<4xf16>, %n: index) -> tensor<4xf16> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %r = scf.for %i = %zero to %n step %one iter_args(%v = %x) -> tensor<4xf16> {
    scf.yield %v : tensor<4xf16>
  }
  return %r : tensor<4xf16>
}

// CHECK-LABEL: func.func @while_tuples
// CHECK: scf.while {{.*}} : (tensor<4xf16, #frisk.distributed{{.*}}, i1) -> tensor<4xf16, #frisk.distributed
// CHECK: scf.condition{{.*}} : tensor<4xf16, #frisk.distributed
// CHECK: ^bb0({{.*}}: tensor<4xf16, #frisk.distributed
// CHECK: scf.yield {{.*}} : tensor<4xf16, #frisk.distributed{{.*}}, i1
func.func @while_tuples(%x: tensor<4xf16>, %p: i1) -> tensor<4xf16> {
  %r = scf.while (%v = %x, %pred = %p) : (tensor<4xf16>, i1) -> tensor<4xf16> {
    scf.condition(%pred) %v : tensor<4xf16>
  } do {
  ^bb0(%v: tensor<4xf16>):
    scf.yield %v, %p : tensor<4xf16>, i1
  }
  return %r : tensor<4xf16>
}
