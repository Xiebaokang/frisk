// RUN: frisk-opt %s -split-input-file -frisk-test-lower-layout-conversions -verify-diagnostics
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1,32,1,1,1], replication = 1>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xf32, #A>) kernel  {

    // expected-error @+1 {{requires known_block_size matching topology}}
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #B>

    gpu.return
  }
}
// -----
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1,32,1,1,1], replication = 1>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xf32, #A>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {

    // expected-error @+1 {{requires known_block_size matching topology}}
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #B>

    gpu.return
  }
}
// -----
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1,32,1,1,1], replication = 1>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xf32, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %true = arith.constant true
    scf.if %true {
    // expected-error @+1 {{requires uniform entry-block execution}}
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #B>
    }
    gpu.return
  }
}
// -----
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1,32,1,1,1], replication = 1>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xf32, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
    // expected-error @+1 {{requires uniform entry-block execution}}
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #B>
    }
    gpu.return
  }
}
// -----
#a = #frisk.bit_linear<inputs = ["lane","register"], input_bits = [4,1], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [2,16,1,1,1], replication = 1>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xf32, #A>) kernel attributes {known_block_size = array<i32: 16, 1, 1>} {

    // expected-error @+1 {{requires lane extent 32}}
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #A>

    gpu.return
  }
}
// -----
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1,32,1,1,1], replication = 1>
#C = #frisk.distributed<map = #b, topology = [1,32,2,1,1], replication = 2>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xf32, #A>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {

    // expected-error @+1 {{source/destination execution topologies differ}}
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #C>

    gpu.return
  }
}
// -----
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1,32,1,1,2], replication = 2>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xf32, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {

    // expected-error @+1 {{cross-CTA/cluster communication is unsupported}}
    %y = frisk.convert_layout %x : tensor<32xf32, #A> -> tensor<32xf32, #A>

    gpu.return
  }
}
// -----
#a = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs = ["lane"], input_bits = [5], outputs = ["m"], output_bits = [5], matrix = dense<[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map = #a, topology = [1,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1,32,1,1,1], replication = 1>

gpu.module @kernels {
  gpu.func @bad(%x: tensor<32xi1, #A>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {

    // expected-error @+1 {{unsupported element type}}
    %y = frisk.convert_layout %x : tensor<32xi1, #A> -> tensor<32xi1, #B>

    gpu.return
  }
}
