// RUN: frisk-opt %s --split-input-file --verify-diagnostics --frisk-infer-layouts=analysis-only

func.func @unknown(%arg: tensor<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{operation has no layout constraint model}}
  %x = tensor.extract %arg[%c0] : tensor<8xf32>
  return %x : f32
}

// -----

// expected-error@+1 {{distributed inference requires ranked tensor types}}
func.func @unranked(%arg: tensor<*xf32>) { return }

// -----

// expected-error@+1 {{static power-of-two tile extents greater than one}}
func.func @non_power_of_two(%arg: tensor<6xf32>) { return }

// -----

// expected-error@+1 {{operation has no layout constraint model for external tensor signature}}
func.func private @external(tensor<8xf32>)

// -----

func.func @chain(%arg: tensor<8xf32>) {
  %0 = arith.negf %arg : tensor<8xf32>
  %1 = arith.negf %0 : tensor<8xf32>
  %2 = arith.negf %1 : tensor<8xf32>
  %3 = arith.negf %2 : tensor<8xf32>
  %4 = arith.negf %3 : tensor<8xf32>
  %5 = arith.negf %4 : tensor<8xf32>
  %6 = arith.negf %5 : tensor<8xf32>
  %7 = arith.negf %6 : tensor<8xf32>
  return
}
// expected-error@-11 {{bootstrap layout solver limit exceeded}}

// -----

// expected-error@+1 {{distributed inference requires nonzero-rank tensor tiles}}
func.func @rank_zero(%arg: tensor<f32>) { return }
