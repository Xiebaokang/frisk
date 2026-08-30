// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s

#xor = #frisk.bit_linear<
  inputs = ["lane", "register"], input_bits = [2, 1],
  outputs = ["m", "n"], output_bits = [1, 2],
  matrix = dense<[[1, 0, 0], [0, 1, 1], [0, 0, 1]]> : tensor<3x3xi1>>

module attributes {frisk.map = #xor} {}

// CHECK: #frisk.bit_linear<inputs = ["lane", "register"], input_bits = [2, 1], outputs = ["m", "n"], output_bits = [1, 2], matrix = dense<{{.*}}> : tensor<3x3xi1>>

// -----

// expected-error@+1 {{matrix shape must be [2, 2] but got [1, 1]}}
#bad_shape = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["m"], output_bits = [2], matrix = dense<0> : tensor<1x1xi1>>
module attributes {frisk.map = #bad_shape} {}

// -----

// expected-error@+1 {{duplicate input dimension name 'lane'}}
#duplicate = #frisk.bit_linear<inputs = ["lane", "lane"], input_bits = [1, 1],
  outputs = ["m"], output_bits = [2], matrix = dense<0> : tensor<2x2xi1>>
module attributes {frisk.map = #duplicate} {}

// -----

// expected-error@+1 {{input bit widths must be positive}}
#zero_width = #frisk.bit_linear<inputs = ["lane"], input_bits = [0],
  outputs = ["m"], output_bits = [1], matrix = dense<0> : tensor<1x1xi1>>
module attributes {frisk.map = #zero_width} {}
