add_frisk_library(FriskLayoutToGPU
  LayoutConversionPlan.cpp
  TestLayoutConversionLowering.cpp
  DEPENDS FriskTransformsIncGen
  LINK_LIBS PUBLIC FriskIR FriskLayoutAnalysis MLIRIR MLIRPass
    MLIRArithDialect MLIRGPUDialect MLIRMemRefDialect MLIRSCFDialect MLIRVectorDialect
)
