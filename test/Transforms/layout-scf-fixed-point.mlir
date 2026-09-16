// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: frisk-opt %s --frisk-infer-layouts -verify-each | FileCheck %s --check-prefix=IR
// RUN: frisk-opt %s --frisk-infer-layouts -verify-each > %t.once
// RUN: frisk-opt %t.once --frisk-infer-layouts -verify-each > %t.twice
// RUN: diff %t.once %t.twice

// Convertible backedges preserve M3 conversion semantics. Exact shrinking and
// queue ledgers (including a late seed that reschedules a real for backedge)
// are checked in AliasRegionTest; this is the end-to-end region/replay gate.
// GRAPH-DAG: region-edge for-backedge slot=0
// GRAPH-DAG: region-edge for-init slot=0
// GRAPH-DAG: region-edge for-result slot=0
// GRAPH-DAG: region-edge if-yield slot=0
// GRAPH-DAG: region-edge while-init slot=1
// GRAPH-DAG: region-edge while-backedge slot=1
// GRAPH-DAG: region-edge while-condition slot=0
// GRAPH-DAG: region-edge while-result slot=0
// GRAPH: propagation strict initial={{[0-9]+}} final={{[0-9]+}} deleted={{[0-9]+}} changes={{[0-9]+}} pops={{[0-9]+}} initial-constraints={{[0-9]+}} enqueues={{[0-9]+}} max-queue={{[0-9]+}} pop-bound={{[0-9]+}} static-pop-bound={{[0-9]+}}
// GRAPH: domain-changes strict [
// GRAPH: propagation common initial={{[0-9]+}} final={{[0-9]+}} deleted={{[0-9]+}} changes={{[0-9]+}} pops={{[0-9]+}} initial-constraints={{[0-9]+}} enqueues={{[0-9]+}} max-queue={{[0-9]+}} pop-bound={{[0-9]+}} static-pop-bound={{[0-9]+}}
// GRAPH: domain-changes common [

// IR-LABEL: func.func @nested
// IR: scf.for {{.*}} iter_args{{.*}} -> (tensor<4xf32, #frisk.distributed
// IR: scf.if {{.*}} -> (tensor<4xf32, #frisk.distributed
// IR: scf.yield {{.*}} : tensor<4xf32, #frisk.distributed
func.func @nested(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: i1, %n: index) -> tensor<4xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %r = scf.for %i = %zero to %n step %one iter_args(%v = %a) -> tensor<4xf32> {
    %next = scf.if %p -> tensor<4xf32> {
      scf.yield %v : tensor<4xf32>
    } else {
      scf.yield %b : tensor<4xf32>
    }
    scf.yield %next : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// IR-LABEL: func.func @different_tuples
// IR: scf.while {{.*}} : (tensor<4xf32, #frisk.distributed{{.*}}, tensor<8xi32, #frisk.distributed{{.*}}>) -> tensor<8xi32, #frisk.distributed
// IR: scf.condition({{.*}}) {{.*}} : tensor<8xi32, #frisk.distributed
// IR: scf.yield {{.*}} : tensor<4xf32, #frisk.distributed{{.*}}, tensor<8xi32, #frisk.distributed
func.func @different_tuples(%a: tensor<4xf32>, %b: tensor<8xi32>, %p: i1) -> tensor<8xi32> {
  %r = scf.while (%x = %a, %y = %b) : (tensor<4xf32>, tensor<8xi32>) -> tensor<8xi32> {
    scf.condition(%p) %y : tensor<8xi32>
  } do {
  ^bb0(%z: tensor<8xi32>):
    scf.yield %a, %z : tensor<4xf32>, tensor<8xi32>
  }
  return %r : tensor<8xi32>
}

// IR-LABEL: func.func @zero_results
// IR: scf.if
// IR: scf.for
// IR: scf.while {{.*}} : (tensor<4xf32, #frisk.distributed{{.*}}>) -> ()
// IR: scf.condition({{.*}})
// IR: scf.yield {{.*}} : tensor<4xf32, #frisk.distributed
func.func @zero_results(%a: tensor<4xf32>, %p: i1, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.if %p { scf.yield }
  scf.for %i = %zero to %n step %one { scf.yield }
  scf.while (%x = %a) : (tensor<4xf32>) -> () {
    scf.condition(%p)
  } do {
    scf.yield %a : tensor<4xf32>
  }
  return
}
