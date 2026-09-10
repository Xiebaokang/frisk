// RUN: frisk-opt %s -frisk-test-lower-layout-conversions -verify-each -allow-unregistered-dialect | FileCheck %s --implicit-check-not=arith.extf --implicit-check-not=arith.truncf --implicit-check-not=arith.fptosi --implicit-check-not=arith.sitofp
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1, 32, 1, 1, 1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1, 32, 1, 1, 1], replication = 1>

gpu.module @kernels {
  gpu.func @shuffle_i8(%x: tensor<32xi8, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xi8, #A> -> tensor<32xi8, #B>
    "test.consume"(%y) : (tensor<32xi8, #B>) -> ()
    gpu.return
  }
  gpu.func @shuffle_i16(%x: tensor<32xi16, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xi16, #A> -> tensor<32xi16, #B>
    "test.consume"(%y) : (tensor<32xi16, #B>) -> ()
    gpu.return
  }
  gpu.func @shuffle_i32(%x: tensor<32xi32, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xi32, #A> -> tensor<32xi32, #B>
    "test.consume"(%y) : (tensor<32xi32, #B>) -> ()
    gpu.return
  }
  gpu.func @shuffle_i64(%x: tensor<32xi64, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xi64, #A> -> tensor<32xi64, #B>
    "test.consume"(%y) : (tensor<32xi64, #B>) -> ()
    gpu.return
  }
  gpu.func @shuffle_f16(%x: tensor<32xf16, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xf16, #A> -> tensor<32xf16, #B>
    "test.consume"(%y) : (tensor<32xf16, #B>) -> ()
    gpu.return
  }
  gpu.func @shuffle_bf16(%x: tensor<32xbf16, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xbf16, #A> -> tensor<32xbf16, #B>
    "test.consume"(%y) : (tensor<32xbf16, #B>) -> ()
    gpu.return
  }
  gpu.func @shuffle_f32(%x: tensor<32xf32, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #B>
    "test.consume"(%y) : (tensor<32xf32, #B>) -> ()
    gpu.return
  }
  gpu.func @shuffle_f64(%x: tensor<32xf64, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %y = frisk.convert_layout %x : tensor<32xf64, #A> -> tensor<32xf64, #B>
    "test.consume"(%y) : (tensor<32xf64, #B>) -> ()
    gpu.return
  }
}
// CHECK-LABEL: gpu.func @shuffle_i8
// CHECK: builtin.unrealized_conversion_cast
// CHECK: arith.extui
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: arith.trunci
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xi8,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @shuffle_i16
// CHECK: builtin.unrealized_conversion_cast
// CHECK: arith.extui
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: arith.trunci
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xi16,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @shuffle_i32
// CHECK: builtin.unrealized_conversion_cast
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xi32,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @shuffle_i64
// CHECK: builtin.unrealized_conversion_cast
// CHECK: arith.shrui
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: arith.extui
// CHECK: arith.shli
// CHECK: arith.ori
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xi64,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @shuffle_f16
// CHECK: builtin.unrealized_conversion_cast
// CHECK: arith.bitcast
// CHECK: arith.extui
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: arith.trunci
// CHECK: arith.bitcast
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xf16,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @shuffle_bf16
// CHECK: builtin.unrealized_conversion_cast
// CHECK: arith.bitcast
// CHECK: arith.extui
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: arith.trunci
// CHECK: arith.bitcast
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xbf16,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @shuffle_f32
// CHECK: builtin.unrealized_conversion_cast
// CHECK: arith.bitcast
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: arith.bitcast
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xf32,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
// CHECK-LABEL: gpu.func @shuffle_f64
// CHECK: builtin.unrealized_conversion_cast
// CHECK: arith.bitcast
// CHECK: arith.shrui
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: gpu.shuffle idx {{.*}} : i32
// CHECK: arith.extui
// CHECK: arith.shli
// CHECK: arith.ori
// CHECK: arith.bitcast
// CHECK: builtin.unrealized_conversion_cast {{.*}} to tensor<32xf64,
// CHECK-NOT: frisk.convert_layout
// CHECK: gpu.return
