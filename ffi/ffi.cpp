#include "ffi.h"

PYBIND11_MODULE(frisk_ffi, m) {
  m.doc() = "TODO";
  mlir::frisk::init_ffi_ir(m.def_submodule("ir"));
  // mlir::frisk::init_ffi_passes(m.def_submodule("passes"));
}