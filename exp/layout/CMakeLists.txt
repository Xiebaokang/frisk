add_executable(frisk_layout_tool
  gemm_layout_tool.cpp
)

llvm_update_compile_flags(frisk_layout_tool)

target_link_libraries(frisk_layout_tool
  PRIVATE
  FriskIR
  LLVMSupport
  MLIRIR
  MLIRFuncDialect
  MLIRParser
)
