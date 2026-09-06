add_frisk_library(FriskIR
  FriskAttributes.cpp
  FriskEnums.cpp
  FriskLayoutAttrs.cpp
  FriskLayoutInterfaces.cpp
  FriskLayoutOps.cpp
  FriskOps.cpp
  FriskOps_Reduce.cpp
  ../Utils/LayoutUtils.cpp

  DEPENDS
  FriskTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  FriskAnalysis
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
