add_frisk_library(FriskIR
  FriskAttributes.cpp
  FriskEnums.cpp
  FriskOps.cpp

  DEPENDS
  FriskTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)