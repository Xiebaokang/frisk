add_llvm_executable(frisk-opt
  frisk-opt.cpp
)

set_target_properties(frisk-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

llvm_update_compile_flags(frisk-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(frisk-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  FriskIR
  FriskTransforms
  FriskLayoutToGPU
)
