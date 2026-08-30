add_unittest(FriskUnitTests FriskLayoutUnitTests
  GF2MatrixTest.cpp
  LayoutInterfaceTest.cpp
)

set_target_properties(FriskLayoutUnitTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
)

target_link_libraries(FriskLayoutUnitTests
  PRIVATE
  FriskAnalysis
  FriskIR
  MLIRIR
)
