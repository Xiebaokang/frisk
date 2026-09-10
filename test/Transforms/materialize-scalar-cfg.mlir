// RUN: frisk-opt %s -frisk-infer-layouts -verify-each | FileCheck %s
// RUN: frisk-opt %s -frisk-infer-layouts -frisk-infer-layouts -verify-each | FileCheck %s

// The scalar definition dominates its nested capture, but its block is
// printed later. Region captures are dependencies even without op operands.
// CHECK-LABEL: func.func @scalar_cfg
// CHECK-SAME: attributes {test.function = "preserved"}
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: arith.constant {test.constant = "preserved"} 7 : i32
func.func @scalar_cfg(%p: i1) -> i32 attributes {test.function = "preserved"} {
  cf.cond_br %p, ^bb1, ^bb2
^bb3:
  %r = scf.if %p -> i32 {
    scf.yield %x : i32
  } else {
    scf.yield %x : i32
  }
  return %r : i32
^bb1:
  cf.br ^bb2
^bb2:
  %x = arith.constant {test.constant = "preserved"} 7 : i32
  cf.br ^bb3
}
