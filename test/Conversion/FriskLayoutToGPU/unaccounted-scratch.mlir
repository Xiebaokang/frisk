// RUN: frisk-opt %s -frisk-test-lower-layout-conversions -verify-diagnostics
#a = #frisk.bit_linear<inputs = ["lane", "warp"], input_bits = [5, 1], outputs = ["m"], output_bits = [6], matrix = dense<[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]> : tensor<6x6xi1>>
#b = #frisk.bit_linear<inputs = ["lane", "warp"], input_bits = [5, 1], outputs = ["m"], output_bits = [6], matrix = dense<[[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]]> : tensor<6x6xi1>>
#A = #frisk.distributed<map = #a, topology = [1, 32, 2, 1, 1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [1, 32, 2, 1, 1], replication = 1>

gpu.module @kernels {
  gpu.func @dynamic_scratch(%x: tensor<64xf16, #A>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %dynamic = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    // expected-error @+1 {{cannot account for non-attributed workgroup storage}}
    %y = frisk.convert_layout %x : tensor<64xf16, #A> -> tensor<64xf16, #B>
    gpu.return
  }
}
