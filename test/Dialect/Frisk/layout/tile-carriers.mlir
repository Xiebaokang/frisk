// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s

#map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [2],
  matrix = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>>
#enc = #frisk.distributed<map = #map, topology = [1, 4, 1, 1, 1], replication = 1>

module {
  func.func @valid(%view: memref<4xf16, #frisk<memory_space Shared>>,
                   %tile: tensor<4xf16, #enc>) {
    %loaded = frisk.tile_load %view
      : memref<4xf16, #frisk<memory_space Shared>> -> tensor<4xf16, #enc>
    frisk.tile_store %tile, %view
      : tensor<4xf16, #enc>, memref<4xf16, #frisk<memory_space Shared>>
    return
  }
}

// CHECK-LABEL: func.func @valid
// CHECK: frisk.tile_load
// CHECK: frisk.tile_store

// -----

module {
  func.func @load_shape_mismatch(%view: memref<4xf16>) {
    // expected-error@+1 {{source memref and result tensor must have identical static shape and element type}}
    %loaded = "frisk.tile_load"(%view) : (memref<4xf16>) -> tensor<2x2xf16>
    return
  }
}

// -----

module {
  func.func @load_element_mismatch(%view: memref<4xf16>) {
    // expected-error@+1 {{source memref and result tensor must have identical static shape and element type}}
    %loaded = "frisk.tile_load"(%view) : (memref<4xf16>) -> tensor<4xf32>
    return
  }
}

// -----

module {
  func.func @store_shape_mismatch(%tile: tensor<4xf16>, %view: memref<2x2xf16>) {
    // expected-error@+1 {{value tensor and target memref must have identical static shape and element type}}
    "frisk.tile_store"(%tile, %view) : (tensor<4xf16>, memref<2x2xf16>) -> ()
    return
  }
}

// -----

module {
  func.func @dynamic_load(%view: memref<?xf16>) {
    // expected-error@+1 {{tile_load requires static source and result shapes}}
    %loaded = "frisk.tile_load"(%view) : (memref<?xf16>) -> tensor<?xf16>
    return
  }
}

// -----

#bad_map = #frisk.bit_linear<inputs = ["lane"], input_bits = [2],
  outputs = ["dim0"], output_bits = [1],
  matrix = dense<[[1, 0]]> : tensor<1x2xi1>>
#bad_enc = #frisk.distributed<map = #bad_map, topology = [1, 4, 1, 1, 1], replication = 2>

module {
  func.func @encoding_checked_for_public_shape(%view: memref<4xf16>) {
    // expected-error@+1 {{distributed output widths must exactly encode type extents}}
    %loaded = "frisk.tile_load"(%view) : (memref<4xf16>) -> tensor<4xf16, #bad_enc>
    return
  }
}
