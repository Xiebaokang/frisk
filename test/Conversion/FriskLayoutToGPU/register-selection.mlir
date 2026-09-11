// RUN: frisk-opt %s -frisk-test-lower-layout-conversions -verify-each -allow-unregistered-dialect | FileCheck %s
#a = #frisk.bit_linear<inputs = ["lane", "register"], input_bits = [5,1], outputs = ["m"], output_bits = [6], matrix = dense<[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]> : tensor<6x6xi1>>
#b = #frisk.bit_linear<inputs = ["lane", "register"], input_bits = [5,1], outputs = ["m"], output_bits = [6], matrix = dense<[[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]]> : tensor<6x6xi1>>
#A = #frisk.distributed<map = #a, topology = [2,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [2,32,1,1,1], replication = 1>

gpu.module @kernels {
  gpu.func @register_lane(%x: tensor<64xi64, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<64xi64, #A> -> tensor<64xi64, #B>
    "test.consume"(%y) : (tensor<64xi64, #B>) -> ()
    gpu.return
  }
}
// CHECK-LABEL: gpu.func @register_lane
// CHECK-NOT: scf.if
// CHECK: gpu.shuffle idx [[LO0:%.*]], [[LANE:%.*]], [[WIDTH:%.*]] : i32
// CHECK: gpu.shuffle idx [[HI0:%.*]], [[LANE]], [[WIDTH]] : i32
// CHECK: gpu.shuffle idx [[LO1:%.*]], [[LANE]], [[WIDTH]] : i32
// CHECK: gpu.shuffle idx [[HI1:%.*]], [[LANE]], [[WIDTH]] : i32
// CHECK: arith.select
// CHECK: arith.select
// CHECK: arith.extui
// CHECK: arith.shli
// CHECK: arith.ori
// CHECK-NOT: scf.if
// CHECK: gpu.return
