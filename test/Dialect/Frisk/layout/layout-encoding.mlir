// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s

#map = #frisk.bit_linear<inputs = ["lane"], input_bits = [5],
  outputs = ["m"], output_bits = [5],
  matrix = dense<[[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>
#enc = #frisk.distributed<map = #map, topology = [1, 32, 1, 1, 1], replication = 1>
module attributes {frisk.encoding = #enc} {}

// CHECK: #frisk.distributed<map = #frisk.bit_linear<{{.*}}>, topology = [1, 32, 1, 1, 1], replication = 1>

// -----

// expected-error@+1 {{distributed layout contains unsupported carrier 'thread'}}
#bad = #frisk.distributed<
  map = #frisk.bit_linear<inputs = ["thread"], input_bits = [5],
    outputs = ["m"], output_bits = [5],
    matrix = dense<[[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]]> : tensor<5x5xi1>>,
  topology = [1, 32, 1, 1, 1], replication = 1>
module attributes {frisk.encoding = #bad} {}

// -----

// expected-error@+1 {{alignment must be a positive power of two}}
#bad = #frisk.storage<
  map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [4],
    outputs = ["byte_offset", "bit_offset"], output_extents = [8, 8],
    map = affine_map<(d0) -> (d0 * 2, 0)>>,
  memory_space = #frisk<memory_space Shared>, alignment = 0,
  vector_granularity = 1>
module attributes {frisk.encoding = #bad} {}
