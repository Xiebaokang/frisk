module {
  frisk.kernel @matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
    frisk.parallel %by, %bx : ((8, 8), threads=128) {
      %cst_0 = arith.constant 0.0000000e+00 : f32
      %0 = frisk.alloc_buffer : memref<128x32xf32, 1>
      %1 = frisk.alloc_buffer(memref<32x128xf32>, scope = "shared")
      %2 = frisk.alloc_buffer(memref<128x128xf32>, scope = "local.fragment")
      frisk.fill %2, 0.0 : memref<128x128xf32>
      frisk.for %k = 0 to 1024 step = 32 {
        frisk.copy(%0, %arg0, [0, 0], [%by * 128, %k * 32]) : memref<128x32xf32>, memref<1024x1024xf32>
        frisk.copy(%1, %arg1, [0, 0], [%k * 32, %bx * 128]) : memref<32x128xf32>, memref<1024x1024xf32>
        frisk.gemm(%2, %4, %5) : memref<128x128xf32>, memref<128x32xf32>, memref<32x128xf32>
      }
      frisk.copy(%arg2, %2) : memref<1024x1024xf32>, memref<128x128xf32>
    }
  }
}
frisk.copy %0[%by * 128, %k * 32], %arg0[0, 0] : memref<128x32xf32>, memref<1024x1024xf32>


module {
  frisk.kernel @online_softmax(%arg0: memref<8192x8192xf16>, %arg1: memref<8192x8192xf16>) {
    frisk.parallel %bx : ((8192, ), threads=128) {
      %cst = arith.constant 1.25 : f32
      %cst_1 = arith.constant -inf : f32
      %x = frisk.alloc_buffer(memref<1024xf16>, scope = "local.fragment")
      %y = frisk.alloc_buffer(memref<1024xf16>, scope = "local.fragment")
      %lse = frisk.alloc_buffer(memref<1xf32>, scope = "local.fragment")
      %max_x = frisk.alloc_buffer(memref<1xf16>, scope = "local.fragment")
      %exp_x = frisk.alloc_buffer(memref<1024xf32>, scope = "local.fragment")
      %sum_exp_x = frisk.alloc_buffer(memref<1xf32>, scope = "local.fragment")
      frisk.fill(%lse, %cst_1) : memref<1xf32>
      frisk.for %i = 0 to 8 {
        frisk.copy(%x, %arg0, [0], [%bx, i * 1024]) : memref<1024xf16>, memref<8192x8192xf16>
        frisk.reduce %max_x, %x
         {dim=0, clear=1, kind="max"} : memref<1xf16>, memref<1024xf16>
        frisk.block %j = 0 to 1024 {  exp_x[j] = exp2(x[i] * log2e - max_x[i] * log2e)
          %0 = affine.load %x[j] : memref<1024xf16>
          %1 = affine.load %max_x[0] : memref<1xf16>
          %2 = arith.mulf %0, %cst : f32
          %3 = arith.mulf %1, %cst : f32
          %4 = arith.subf %2, %3 : f32
          %5 = math.exp2f %4 : f32
          affine.store %5, %exp_x[j] : memref<1024xf32>
        }
        frisk.reduce(%sum_exp_x, %exp_x, dim=0, clear=1, "sum") : memref<1xf32>, memref<1024xf32>
        frisk.block %j = 0 to 1 {
          %0 = affine.load %lse[0] : memref<1xf32>
          %1 = affine.load %sum_exp_x[0] : memref<1xf32>
          %2 = affine.load %max_x[0] : memref<1xf16>
          %3 = arith.mulf %2, %cst : f32
          %4 = arith.subf %0, %3 : f32
          %5 = math.exp2f %4 : f32
          %6 = arith.addf %5, %1 : f32
          %7 = math.log2f %6 : f32
          %8 = arith.add %3, %7 : f32
          affine.store %8, %lse[0] : memref<1xf32>
        }
      }
      frisk.for %i = 0 to 8 {
        frisk.copy(%x, %arg0, [0], [%bx, i * 1024]) : memref<1024xf16>, memref<8192x8192xf16>
        frisk.block %j = 0 to 1024 {
          %0 = affine.load %y[%j] : <1024xf16>
          %1 = affine.load %x[%j] : <1024xf16>
          %2 = affine.load %lse[0] : <1xf32>
          %3 = arith.mulf %1, %cst : f32
          %4 = arith.subf %3, %2 : f32
          %5 = math.exp2f %4 : f32
          affine.store %8, %y[%j] : memref<1024xf16>
        }
        frisk.copy(%arg1, %y, [%bx, %i * 1024], [0]) : memref<8192x8192xf16>, memref<1024xf16>
      }
    }
  }
}

frisk.reduce(%max_x, %x, dim=0, clear=1, "max") : memref<1xf16>, memref<1024xf16>

affine.store %cst_2, %max_x[0] : <1xf16>
affine.for %i = 0 to 8 {
  %0 = affine.load %x[%i] : <8xf16>
  %1 = affine.load %max_x[0] : <1xf16>
  %2 = arith.maxnumf %0, %1 : f16
  affine.store %2, %max_x[0] : <1xf16>
}
gpu.shuffle xor  

module {
  frisk.kernel @attention(%arg0: memref<1x32x4096x128xf16>, %arg1: memref<1x32x128x4096xf16>, %arg2: memref<1x32x4096x128xf16>, %arg3: memref<1x32x4096x128xf16>) {
    frisk.parallel %bz, %by, %bx : ((1, 32, 64), threads=256) {
      %cst = arith.constant 1.25 : f32
      %cst_0 = arith.constant 0.0 : f32
      %cst_1 = arith.constant -inf : f32
      %q = frisk.alloc_buffer(memref<64x128xf16>, scope = "shared")
      %k = frisk.alloc_buffer(memref<128x64xf16>, scope = "shared")
      %v = frisk.alloc_buffer(memref<64x128xf16>, scope = "shared")
      %o = frisk.alloc_buffer(memref<64x128xf16>, scope = "shared")
      %acc_s = frisk.alloc_buffer(memref<64x64xf32>, scope = "local.fragment")
      %acc_s_cast = frisk.alloc_buffer(memref<64x64xf16>, scope = "local.fragment")
      %acc_o = frisk.alloc_buffer(memref<64x128xf32>, scope = "local.fragment")
      %scores_max = frisk.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %scores_max_prev = frisk.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %scores_sacle = frisk.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %scores_sum = frisk.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      %logsum = frisk.alloc_buffer(memref<64xf32>, scope = "local.fragment")
      frisk.copy(%q, %arg0, [0, 0], [%bz, %by, %bx*64, 0]) : memref<64x128xf16>, memref<1x32x4096x128xf16>
      frisk.fill(%acc_o, %cst_0) : memref<64x128xf32>
      frisk.fill(%logsum, %cst_0) : memref<64xf32>
      frisk.fill(%scores_max, %cst_1) : memref<64xf32>
      frisk.for %iter_k = 0 to 4096 step = 64 {
        frisk.copy(%k, %arg1, [0, 0], [%bz, %by, 0, %iter_k*64]) : memref<128x64xf16>, memref<1x32x128x4096xf16>
        frisk.fill(%acc_s, %cst_0) : memref<64x64xf32>
        frisk.gemm(%acc_s, %q, %k) : memref<64x64xf32>, memref<64x128xf16>, memref<128x64xf16>
        frisk.copy(%scores_max_prev, %scores_max, [0], [0]) : memref<64xf32>, memref<64xf32>
        frisk.fill(%acc_s, %cst_0) : memref<64x64xf32>
        frisk.reduce(%scores_max, %acc_s, "max", dim=1) : memref<64xf32>, memref<64x64xf32>
        frisk.block %i = 0 to 64 {
          %0 = affine.load(%scores_max_prev, [%i]) : memref<64xf32>
          %1 = affine.load(%scores_max, [%i]) : memref<64xf32>
          %2 = arith.mulf(%0, %cst) : f32
          %3 = arith.mulf(%1, %cst) : f32
          %4 = arith.subf(%2, %3) : f32
          %5 = math.exp2(%4) : f32
          affine.store(%5, %scores_sacle, [%i]) : memref<64xf32>
        }
        frisk.block %i, %j = (0, 0) to (64, 64) {
          %0 = affine.load(%acc_s, [%j, %i]) : memref<64x64xf32>
          %1 = affine.load(%scores_max, [%i]) : memref<64xf32>
          %2 = arith.mulf(%0, %cst) : f32
          %3 = arith.mulf(%1, %cst) : f32
          %4 = arith.subf(%0, %1) : f32
          %5 = math.exp2(%4) : f32
          affine.store(%5, %acc_s, [%j, %i]) : memref<64x64xf32>
        }
        frisk.reduce(%scores_sum, %acc_s, "sum", dim=1) : memref<64xf32>, memref<64x64xf32>
        frisk.block %i = 0 to 64 {
          %0 = affine.load(%scores_sacle, [%i]) : memref<64xf32>
          %1 = affine.load(%logsum, [%i]) : memref<64xf32>
          %2 = affine.load(%scores_sum, [%i]) : memref<64xf32>
          %3 = arith.mulf(%0, %1) : f32
          %4 = arith.addf(%2, %3) : f32
          affine.store(%4, %logsum, [%i]) : memref<64xf32>
        }
        frisk.copy(%acc_s_cast, %acc_s, [0, 0], [0, 0]) : memref<64x64xf16>, memref<64x64xf32>
        frisk.block %i, %j = (0, 0) to (64, 128) {
          %0 = affine.load(%acc_o, [%i, %j]) : memref<64x128xf32>
          %1 = affine.load(%scores_scale, [%i]) : memref<64xf16>
          %2 = arith.mulf(%0, %1) : f32
          affine.store(%2, %acc_o, [%i, %j]) : memref<64x128xf32>
        }
        frisk.copy(%v, %arg2, [0, 0], [%bz, %by, %iter_k*64, 0]) : memref<64x128xf16>, memref<1x32x4096x128xf16>
        frisk.gemm(%acc_o, %acc_s_cast, %v) : memref<64x128xf32>, memref<64x64xf16>, memref<64x128xf16>
      }
      frisk.copy(%o, %acc_o, [0, 0], [0, 0]) : memref<64x128xf16>, memref<64x128xf32>
      frisk.copy(%arg3, %o, [0, 0], [%bz, %by, %bx*64, 0]) : memref<1x32x4096x128xf16>, memref<64x128xf16>
    }
  }
}