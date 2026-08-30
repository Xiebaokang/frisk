// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s

#outer = #frisk.affine_layout<inputs = ["outer_m"], input_extents = [2],
  outputs = ["m"], output_extents = [6],
  map = affine_map<(d0) -> (d0 * 4)>>
#inner = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["m"], output_bits = [2],
  matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#product = #frisk.product<outer = #outer, inner = #inner, split_extents = [4]>

module attributes {frisk.map = #product} {}

// CHECK: #frisk.product<outer = #frisk.affine_layout<{{.*}}>, inner = #frisk.bit_linear<{{.*}}>, split_extents = [4]>

#dynamic = #frisk.affine_layout<inputs = ["i"], input_extents = [-9223372036854775808],
  outputs = ["m"], output_extents = [-9223372036854775808],
  map = affine_map<(d0)[s0] -> (d0)>>
module attributes {frisk.map = #dynamic} {}

// -----

// expected-error@+1 {{cannot prove outer result 'd0 * 3' is aligned to split extent 4}}
#bad = #frisk.product<
  outer = #frisk.affine_layout<inputs = ["outer_m"], input_extents = [2],
    outputs = ["m"], output_extents = [8],
    map = affine_map<(d0) -> (d0 * 3)>>,
  inner = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
    outputs = ["m"], output_bits = [2],
    matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>,
  split_extents = [4]>
module attributes {frisk.map = #bad} {}
