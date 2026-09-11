add_unittest(FriskUnitTests FriskLayoutUnitTests
  LayoutConversionPlanTest.cpp
  BitLinearLayoutTest.cpp
  GF2MatrixTest.cpp
  LayoutEncodingTest.cpp
  LayoutInterfaceTest.cpp
  LayoutConstraintTest.cpp
  LayoutPropagationTest.cpp
  DistributedPropagationTest.cpp
  LayoutVerifierTest.cpp
  LayoutMaterializationTest.cpp
  LegacyLayoutAdapterTest.cpp
  ProductLayoutTest.cpp
)

set_target_properties(FriskLayoutUnitTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
)

target_link_libraries(FriskLayoutUnitTests
  PRIVATE
  FriskLayoutToGPU
  FriskTransforms
  FriskAnalysis
  FriskIR
  FriskLayoutAnalysis
  FriskSM90LayoutTarget
  FriskLegacyLayoutAdapter
  FriskTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRIR
  MLIRParser
)
