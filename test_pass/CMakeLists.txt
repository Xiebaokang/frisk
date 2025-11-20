add_executable(frisk_attr_test
  attr_test.cpp
)

llvm_update_compile_flags(frisk_attr_test)

target_link_libraries(frisk_attr_test
  PRIVATE
  FriskIR
  MLIRFuncDialect
  MLIRParser
)
