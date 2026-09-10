// RUN: frisk-opt %s -frisk-test-lower-layout-conversions -verify-each -allow-unregistered-dialect | FileCheck %s
#r = #frisk.bit_linear<inputs = ["register"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#l = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#p = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#R = #frisk.distributed<map = #r, topology = [32,32,2,1,1], replication = 64>
#L = #frisk.distributed<map = #l, topology = [1,32,2,1,1], replication = 2>
#P = #frisk.distributed<map = #p, topology = [1,32,2,1,1], replication = 2>

gpu.module @kernels {
  gpu.func @same_thread(%x: tensor<32xf32, #R>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xf32, #R> -> tensor<32xf32, #P>
    "test.consume"(%y) : (tensor<32xf32, #P>) -> ()
    gpu.return
  }
  gpu.func @same_warp(%x: tensor<32xf32, #L>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xf32, #L> -> tensor<32xf32, #P>
    "test.consume"(%y) : (tensor<32xf32, #P>) -> ()
    gpu.return
  }
}
// CHECK-LABEL: gpu.func @same_thread
// CHECK-NOT: workgroup
// CHECK-NOT: gpu.shuffle
// CHECK-NOT: gpu.barrier
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @same_warp
// CHECK-NOT: workgroup
// CHECK: gpu.shuffle
// CHECK-NOT: gpu.barrier
// CHECK: gpu.return
