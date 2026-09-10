// RUN: frisk-opt %s -frisk-test-lower-layout-conversions='scratch-budget=520' -verify-each | FileCheck %s
// RUN: frisk-opt %s -frisk-test-lower-layout-conversions='scratch-budget=519' -verify-diagnostics
#a = #frisk.bit_linear<inputs = ["lane", "warp"], input_bits = [5, 1], outputs = ["m"], output_bits = [6], matrix = dense<[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]> : tensor<6x6xi1>>
#b = #frisk.bit_linear<inputs = ["lane", "warp"], input_bits = [5, 1], outputs = ["m"], output_bits = [6], matrix = dense<[[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]]> : tensor<6x6xi1>>
#A = #frisk.distributed<map = #a, topology = [1, 32, 2, 1, 1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1, 32, 2, 1, 1], replication = 1>

gpu.module @kernels {
  gpu.func @existing(%x: tensor<64xf64, #A>) workgroup(%pre : memref<1xi8, #gpu.address_space<workgroup>>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    // expected-error @+1 {{adapter scratch budget exceeded (budget 519 bytes)}}
    %y = frisk.convert_layout %x : tensor<64xf64, #A> -> tensor<64xf64, #B>
    gpu.return
  }
}
// CHECK-LABEL: gpu.func @existing
// CHECK-SAME: memref<1xi8, #gpu.address_space<workgroup>>
// CHECK-SAME: memref<64xf64, #gpu.address_space<workgroup>> {llvm.align = 8 : i64}
// CHECK: memref.store {{.*}} : memref<64xf64, #gpu.address_space<workgroup>>
// CHECK: gpu.barrier
// CHECK: memref.load {{.*}} : memref<64xf64, #gpu.address_space<workgroup>>
