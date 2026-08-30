add_unittest(FriskUnitTests FriskLayoutUnitTests
  LayoutInterfaceTest.cpp
)

set_target_properties(FriskLayoutUnitTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
)

target_link_libraries(FriskLayoutUnitTests
  PRIVATE
  FriskIR
  MLIRIR
)
