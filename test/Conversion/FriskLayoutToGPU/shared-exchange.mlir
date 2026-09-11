// RUN: frisk-opt %s -frisk-test-lower-layout-conversions -verify-each -allow-unregistered-dialect | FileCheck %s
#a = #frisk.bit_linear<inputs = ["lane", "warp"], input_bits = [5, 1], outputs = ["m"], output_bits = [6], matrix = dense<[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]> : tensor<6x6xi1>>
#b = #frisk.bit_linear<inputs = ["lane", "warp"], input_bits = [5, 1], outputs = ["m"], output_bits = [6], matrix = dense<[[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]]> : tensor<6x6xi1>>
#A = #frisk.distributed<map = #a, topology = [1, 32, 2, 1, 1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1, 32, 2, 1, 1], replication = 1>

gpu.module @kernels {
  gpu.func @shared(%x: tensor<64xf16, #A>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<64xf16, #A> -> tensor<64xf16, #B>
    "test.consume"(%y) : (tensor<64xf16, #B>) -> ()
    gpu.return
  }
}
// CHECK-LABEL: gpu.func @shared
// CHECK-SAME: workgroup({{.*}}memref<64xf16, #gpu.address_space<workgroup>>
// CHECK: scf.if
// CHECK: memref.store
// CHECK: gpu.barrier
// CHECK: memref.load
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<64xf16,
// CHECK-NOT: gpu.shuffle
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
