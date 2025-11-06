
module {
  dp.kernel @matmul(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>) {
    dp.parallel %by, %bx : ((8, 8), threads=128) {
      %cst_0 = arith.constant 0.0 : f32
      %0 = dp.alloc_buffer(memref<128x32xf16>, scope = "shared")
      %1 = dp.alloc_buffer(memref<32x128xf16>, scope = "shared")
      %2 = dp.alloc_buffer(memref<128x128xf32>, scope = "local.fragment")
      %3 = dp.fill(%2, cst_0) : memref<128x128xf32>
      dp.for %k = 0 to 1024 step = 32 {
        dp.copy(%0, %arg0, [%by*128, %k*32], [0, 0]) : memref<128x32xf16>, memref<1024x1024xf16>
        dp.copy(%1, %arg1, [%k*32, %bx*128], [0, 0]) : memref<32x128xf16>, memref<1024x1024xf16>
        dp.gemm(%3, %4, %5) : memref<128x128xf32>, memref<128x32xf16>, memref<32x128xf16>
      }
      dp.copy(%arg2, %3) : memref<1024x1024xf16>, memref<128x128xf32>
    }
  }
}

module {
  dp.kernel @attention(%arg0: memref<1x32x4096x128xf16>, %arg1: memref<1x32x128x4096xf16>, %arg2: memref<1x32x4096x128xf16>, %arg3: memref<1x32x4096x128xf16>) {
    dp.parallel %bz, %by, %bx : ((1, 32, 64), threads=256) {
      %cst = arith.constant 1.25 : f32
      %cst_0 = arith.constant 0.0 : f32
      %cst_1 = arith.constant -inf : f32
      %q = dp.alloc_buffer(memref<64x128xf16>, scope = "shared")
      %k = dp.alloc_buffer(memref<128x64xf16>, scope = "shared")
      %v = dp.alloc_buffer(memref<64x128xf16>, scope = "shared")
      %o = dp.alloc_buffer(memref<64x128xf16>, scope = "shared")
      %acc_s = dp.alloc_buffer(memref<64x64xf32>, scope = "local.fragment")
      %acc_s_cast = dp.alloc_buffer(memref<64x64xf16>, scope = "local.fragment")
      %acc_o = dp.alloc_buffer(memref<64x128xf32>, scope = "local.fragment")
      %scores_max = dp.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %scores_max_prev = dp.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %scores_sacle = dp.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %scores_sum = dp.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %logsum = dp.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      dp.copy(%q, %arg0, [0, 0], [%bz, %by, %bx*64, 0]) : memref<64x128xf16>, memref<1x32x4096x128xf16>
      dp.fill(%acc_o, %cst_0) : memref<64x128xf32>
      dp.fill(%logsum, %cst_0) : memref<64xf32>
      dp.fill(%scores_max, -inf) : memref<64xf32>
      dp.for %iter_k = 0 to 4096 step = 64 {
        dp.copy(%k, %arg1, [0, 0], [%bz, %by, 0, %iter_k*64]) : memref<128x64xf16>, memref<1x32x128x4096xf16>
        dp.fill(%acc_s, %cst_0) : memref<64x64xf32>
        dp.gemm(%acc_s, %q, %k) : memref<64x64xf32>, memref<64x128xf16>, memref<128x64xf16>
        dp.copy(%scores_max_prev, %scores_max, [0], [0]) : memref<64xf32>, memref<64xf32>
        dp.fill(%acc_s, %cst_0) : memref<64x64xf32>
        dp.reduce(%scores_max, %acc_s, "max", dim=1) : memref<64xf32>, memref<64x64xf32>
        dp.block %i = 0 to 64 {
          %0 = affine.load(%scores_max_prev, [%i]) : memref<64xf32>
          %1 = affine.load(%scores_max, [%i]) : memref<64xf32>
          %2 = arith.mulf(%0, %cst) : f32
          %3 = arith.mulf(%1, %cst) : f32
          %4 = arith.subf(%2, %3) : f32
          %5 = math.exp2(%4) : f32
          affine.store(%5, %scores_sacle, [%i]) : memref<64xf32>
        }
        dp.block %i, %j = (0, 0) to (64, 64) {
          %0 = affine.load(%acc_s, [%j, %i]) : memref<64x64xf32>
          %1 = affine.load(%scores_max, [%i]) : memref<64xf32>
          %2 = arith.mulf(%0, %cst) : f32
          %3 = arith.mulf(%1, %cst) : f32
          %4 = arith.subf(%0, %1) : f32
          %5 = math.exp2(%4) : f32
          affine.store(%5, %acc_s, [%j, %i]) : memref<64x64xf32>
        }
        dp.reduce(%scores_sum, %acc_s, "sum", dim=1) : memref<64xf32>, memref<64x64xf32>
        dp.block %i = 0 to 64 {
          %0 = affine.load(%scores_sacle, [%i]) : memref<64xf32>
          %1 = affine.load(%logsum, [%i]) : memref<64xf32>
          %2 = affine.load(%scores_sum, [%i]) : memref<64xf32>
          %3 = arith.mulf(%0, %1) : f32
          %4 = arith.addf(%2, %3) : f32
          affine.store(%4, %logsum, [%i]) : memref<64xf32>
        }
        dp.copy(%acc_s_cast, %acc_s, [0, 0], [0, 0]) : memref<64x64xf16>, memref<64x64xf32>
        dp.block %i, %j = (0, 0) to (64, 128) {
          %0 = affine.load(%acc_o, [%i, %j]) : memref<64x128xf32>
          %1 = affine.load(%scores_scale, [%i]) : memref<64xf16>
          %2 = arith.mulf(%0, %1) : f32
          affine.store(%2, %acc_o, [%i, %j]) : memref<64x128xf32>
        }
        dp.copy(%v, %arg2, [0, 0], [%bz, %by, %iter_k*64, 0]) : memref<64x128xf16>, memref<1x32x4096x128xf16>
        dp.gemm(%acc_o, %acc_s_cast, %v) : memref<64x128xf32>, memref<64x64xf16>, memref<64x128xf16>
      }
      dp.copy(%o, %acc_o, [0, 0], [0, 0]) : memref<64x128xf16>, memref<64x128xf32>
      dp.copy(%arg3, %o, [0, 0], [%bz, %by, %bx*64, 0]) : memref<1x32x4096x128xf16>, memref<64x128xf16>
    }
  }
}