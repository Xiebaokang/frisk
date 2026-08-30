add_unittest(FriskUnitTests FriskLayoutUnitTests
  BitLinearLayoutTest.cpp
  GF2MatrixTest.cpp
  LayoutEncodingTest.cpp
  LayoutInterfaceTest.cpp
  LegacyLayoutAdapterTest.cpp
  ProductLayoutTest.cpp
)

set_target_properties(FriskLayoutUnitTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
)

target_link_libraries(FriskLayoutUnitTests
  PRIVATE
  FriskAnalysis
  FriskIR
  FriskLegacyLayoutAdapter
  MLIRFuncDialect
  MLIRIR
  MLIRParser
)
