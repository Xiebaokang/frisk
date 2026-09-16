// RUN: frisk-opt %s -frisk-infer-layouts -frisk-infer-layouts | FileCheck %s --check-prefix=KEEP
// RUN: frisk-opt %s -frisk-infer-layouts -canonicalize -frisk-infer-layouts --verify-diagnostics

// Known Task18 boundary: Pure/DCE may erase an unused whole-root precondition.
// Actual-only verification must reject missing evidence, not invent a hidden
// root contract. Durable evidence across arbitrary normalization is Task22 work.
// KEEP-LABEL: func.func @contract_lifetime
// KEEP: memref.store
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],
  outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,
  memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
func.func @contract_lifetime(%r:memref<4xi32,3>, %x:i32) {
  %zero = arith.constant 0 : index
  %rv = frisk.layout_view %r {layout=#root} : memref<4xi32,3> -> memref<4xi32,3>
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  // expected-error@+1 {{storage alignment lacks an explicit root pointer guarantee}}
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  memref.store %x, %v[%zero] : memref<2xi32,strided<[1],offset:1>,3>
  return
}
