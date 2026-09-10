// RUN: frisk-opt %s --split-input-file --canonicalize --verify-diagnostics | FileCheck %s --check-prefix=CANON

#a_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [2],
  matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#b_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [2],
  matrix = dense<[[0, 1], [1, 0]]> : tensor<2x2xi1>>
#a = #frisk.distributed<map = #a_map, topology = [1, 4, 1, 1, 1], replication = 1>
#b = #frisk.distributed<map = #b_map, topology = [1, 4, 1, 1, 1], replication = 1>

module {
  func.func @valid_and_identity(%x: tensor<4xf16, #a>) -> tensor<4xf16, #a> {
    %converted = frisk.convert_layout %x
      : tensor<4xf16, #a> -> tensor<4xf16, #b>
    %identity = frisk.convert_layout %x
      : tensor<4xf16, #a> -> tensor<4xf16, #a>
    return %identity : tensor<4xf16, #a>
  }
}

// -----

module {
  func.func @shape_mismatch(%x: tensor<4xf16>) {
    // expected-error@+1 {{source and target must have identical shape and element type}}
    %bad = "frisk.convert_layout"(%x) : (tensor<4xf16>) -> tensor<2x2xf16>
    return
  }
}

// -----

module {
  func.func @element_mismatch(%x: tensor<4xf16>) {
    // expected-error@+1 {{source and target must have identical shape and element type}}
    %bad = "frisk.convert_layout"(%x) : (tensor<4xf16>) -> tensor<4xf32>
    return
  }
}

// -----

#b_map_2 = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [2],
  matrix = dense<[[0, 1], [1, 0]]> : tensor<2x2xi1>>
#b_2 = #frisk.distributed<map = #b_map_2, topology = [1, 4, 1, 1, 1], replication = 1>

module {
  func.func @missing_source_encoding(%x: tensor<4xf16>) {
    // expected-error@+1 {{source and target must use DistributedEncodingAttr}}
    %bad = "frisk.convert_layout"(%x) : (tensor<4xf16>) -> tensor<4xf16, #b_2>
    return
  }
}

// -----

#bad_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [1],
  matrix = dense<[[1, 0]]> : tensor<1x2xi1>>
#bad = #frisk.distributed<map = #bad_map, topology = [1, 4, 1, 1, 1], replication = 2>
#a_map_2 = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [2],
  matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#a_2 = #frisk.distributed<map = #a_map_2, topology = [1, 4, 1, 1, 1], replication = 1>

module {
  func.func @invalid_target_encoding(%x: tensor<4xf16, #a_2>) {
    // expected-error@+1 {{distributed output widths must exactly encode type extents}}
    %bad = "frisk.convert_layout"(%x) : (tensor<4xf16, #a_2>) -> tensor<4xf16, #bad>
    return
  }
}

// -----

#a_map_3 = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [2],
  matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#b_map_3 = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [2],
  matrix = dense<[[0, 1], [1, 0]]> : tensor<2x2xi1>>
#a_3 = #frisk.distributed<map = #a_map_3, topology = [1, 4, 1, 1, 1], replication = 1>
#b_3 = #frisk.distributed<map = #b_map_3, topology = [1, 4, 1, 1, 1], replication = 1>

module {
  func.func @canonicalize(%x: tensor<4xf16, #a_3>) -> tensor<4xf16, #a_3> {
    %identity = frisk.convert_layout %x
      : tensor<4xf16, #a_3> -> tensor<4xf16, #a_3>
    %forward = frisk.convert_layout %identity
      : tensor<4xf16, #a_3> -> tensor<4xf16, #b_3>
    %back = frisk.convert_layout %forward
      : tensor<4xf16, #b_3> -> tensor<4xf16, #a_3>
    return %back : tensor<4xf16, #a_3>
  }
}

// CANON-LABEL: func.func @canonicalize
// CANON-NOT: frisk.convert_layout
// CANON: return %arg0
