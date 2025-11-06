#!/bin/bash

if [ -d "./build" ]; then
    cd build
    rm -rf *
else
    mkdir build && cd build
fi

cmake  .. -GNinja \
  -DCMAKE_BUILD_TYPE=Debug  \
  -DCMAKE_LINKER=lld  \
  -DLLVM_ENABLE_ASSERTIONS=ON  \
  -DMLIR_DIR=/home/xiebaokang/software/install/rocm-llvm-install/lib/cmake/mlir
ninja -j32

so_files=(*.so)
cp ./${so_files[0]} ../python/Frisk
cd ..