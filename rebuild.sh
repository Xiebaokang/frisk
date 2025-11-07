#!/bin/bash

if [ ! -d "./build" ]; then
  ./build.sh
else
  cd ./build
  ninja -j32

  so_files=(*.so)
  if [ ! -d "../python/frisk" ]; then
    mkdir ../python/frisk
  fi
  cp ./${so_files[0]} ../python/frisk
  cd ..
fi