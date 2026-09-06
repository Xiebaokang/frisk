# Frisk 常用编译命令

以下命令均在项目根目录执行。

## 首次配置或更新 LLVM 路径

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
```

## 日常编译

```bash
# 完整增量编译
cmake --build build --parallel 32

# 只编译 layout 相关目标
cmake --build build --target FriskIR frisk_attr_test frisk_reduce_layout_test frisk_layout_tool --parallel 32

# 清理并重新编译
./build.sh
```

## Layout 测试

```bash
./build/test_pass/frisk_attr_test
./build/test_pass/frisk_reduce_layout_test
```

## GEMM Layout 快速检查

```bash
# Ampere
./build/exp/layout/frisk_layout_tool \
  --target=sm_80 --block-m=128 --block-n=128 --block-k=64 \
  --threads=128 --a-space=shared --b-space=shared --dtype=fp16 --ldmatrix

# Hopper
./build/exp/layout/frisk_layout_tool \
  --target=sm_90 --block-m=128 --block-n=128 --block-k=64 \
  --threads=128 --a-space=shared --b-space=shared --dtype=fp16 --ldmatrix
```
