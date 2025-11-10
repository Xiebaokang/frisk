set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set(LLVM_TARGET_DEFINITIONS FriskEnums.td)
# mlir_tablegen(FriskEnums.h.inc -gen-enum-decls)
# mlir_tablegen(FriskEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS FriskOps.td)
mlir_tablegen(FriskOps.h.inc -gen-op-decls)
mlir_tablegen(FriskOps.cpp.inc -gen-op-defs)
add_mlir_doc(FriskOps FriskOps Frisk/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS FriskDialect.td)
mlir_tablegen(FriskDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FriskDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(FriskDialect FriskDialect Frisk/ -gen-dialect-doc)

add_public_tablegen_target(FriskTableGen)